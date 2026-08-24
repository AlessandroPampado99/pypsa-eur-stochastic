#!/usr/bin/env python3
"""
Build a stochastic pre-solve network from CSSC representative networks.

Each representative's solved deterministic network supplies all compatible
static and time-dependent component values for its stochastic scenario. The
CSSC reduced probabilities are passed directly to ``Network.set_scenarios``.
The network is prepared before scenarios are created, matching the existing
``stochasticify_network.py`` / ``solve_network.py`` contract.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import (  # noqa: E402
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import prepare_network  # noqa: E402

LOGGER = logging.getLogger(__name__)

SCENARIO_INVARIANT_ATTRIBUTES = {
    "name",
    "bus",
    "type",
    "p_nom_extendable",
    "s_nom_extendable",
    "e_nom_extendable",
    "p_nom_mod",
    "s_nom_mod",
    "e_nom_mod",
    "committable",
    "sign",
    "carrier",
    "weight",
    "p_nom_opt",
    "s_nom_opt",
    "e_nom_opt",
    "build_year",
    "lifetime",
    "active",
}


def read_representatives(path: Path) -> pd.Series:
    """Read representative probabilities and validate their normalization."""
    table = pd.read_csv(path)
    required = {"representative", "probability"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if table["representative"].duplicated().any():
        raise ValueError(f"{path} contains duplicate representatives")
    probabilities = table.set_index("representative")["probability"].astype(float)
    probabilities.index = probabilities.index.map(str)
    if probabilities.empty or not np.isfinite(probabilities).all():
        raise ValueError("Representative probabilities must be non-empty and finite")
    if (probabilities <= 0).any():
        raise ValueError("Representative probabilities must be strictly positive")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"Representative probabilities sum to {probabilities.sum()}, not one"
        )
    return probabilities


def _relabel_snapshots(network: pypsa.Network, snapshots: pd.Index) -> None:
    """Relabel snapshot-dependent tables positionally without changing values."""
    if len(network.snapshots) != len(snapshots):
        raise ValueError("Cannot relabel snapshots with a different length")
    for component_name in network.components.keys():
        for table in network.components[component_name].dynamic.values():
            if table.empty or len(table.index) == len(snapshots):
                table.index = snapshots
            else:
                raise ValueError(
                    f"Unexpected snapshot length in {component_name} time series"
                )
    network._snapshots_data.index = snapshots


def harmonize_snapshots(networks: dict[str, pypsa.Network]) -> None:
    """Drop leap days and give all representative networks one calendar index."""
    canonical: pd.Index | None = None
    canonical_scenario: str | None = None
    for scenario, network in networks.items():
        snapshots = network.snapshots
        if isinstance(snapshots, pd.DatetimeIndex):
            leap_day = (snapshots.month == 2) & (snapshots.day == 29)
            if leap_day.any():
                LOGGER.info(
                    "Removing %d February 29 snapshots from %s",
                    int(leap_day.sum()),
                    scenario,
                )
                network.set_snapshots(snapshots[~leap_day])
            normalized = pd.DatetimeIndex(
                [timestamp.replace(year=2001) for timestamp in network.snapshots],
                name=network.snapshots.name,
            )
        else:
            normalized = network.snapshots.copy()

        if canonical is None:
            canonical = normalized
            canonical_scenario = scenario
        elif not normalized.equals(canonical):
            raise ValueError(
                "Snapshot positions still differ after leap-day removal and calendar "
                f"normalization: {scenario} versus {canonical_scenario}"
            )
        _relabel_snapshots(network, canonical)


def _component_names(index: pd.Index) -> pd.Index:
    """Return component names from deterministic or stochastic indices."""
    if isinstance(index, pd.MultiIndex):
        level = "name" if "name" in index.names else index.names[-1]
        return pd.Index(index.get_level_values(level).unique())
    return pd.Index(index)


def _scenario_static_index(index: pd.Index, scenario: str, names: pd.Index) -> pd.Index:
    """Resolve target static-table rows for one scenario in source-name order."""
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(
            "set_scenarios() did not create stochastic static component tables"
        )
    scenario_level = "scenario" if "scenario" in index.names else index.names[0]
    name_level = "name" if "name" in index.names else index.names[-1]
    lookup = {
        (str(sc), str(name)): key
        for key, sc, name in zip(
            index,
            index.get_level_values(scenario_level),
            index.get_level_values(name_level),
        )
    }
    try:
        return pd.Index([lookup[(scenario, str(name))] for name in names])
    except KeyError as exc:
        raise ValueError(
            f"Stochastic static table lacks scenario/name entry {exc.args[0]}"
        ) from exc


def _scenario_dynamic_columns(
    columns: pd.Index, scenario: str, names: pd.Index
) -> pd.Index:
    """Resolve target time-series columns for one scenario in source-name order."""
    if not isinstance(columns, pd.MultiIndex):
        raise TypeError("set_scenarios() did not create stochastic time-series tables")
    scenario_level = "scenario" if "scenario" in columns.names else columns.names[0]
    name_level = "name" if "name" in columns.names else columns.names[-1]
    lookup = {
        (str(sc), str(name)): key
        for key, sc, name in zip(
            columns,
            columns.get_level_values(scenario_level),
            columns.get_level_values(name_level),
        )
    }
    try:
        return pd.Index([lookup[(scenario, str(name))] for name in names])
    except KeyError as exc:
        raise ValueError(
            f"Stochastic time series lacks scenario/name column {exc.args[0]}"
        ) from exc


def _validate_network_axes(
    base: pypsa.Network, source: pypsa.Network, scenario: str
) -> None:
    """Require snapshots, component types, and component names to match."""
    if not base.snapshots.equals(source.snapshots):
        raise ValueError(f"Snapshot index differs for representative {scenario}")
    base_components = set(base.components.keys())
    source_components = set(source.components.keys())
    if base_components != source_components:
        raise ValueError(f"Component types differ for representative {scenario}")
    for component_name in base_components:
        base_names = set(_component_names(base.components[component_name].static.index))
        source_names = set(
            _component_names(source.components[component_name].static.index)
        )
        extra = sorted(source_names - base_names)
        if extra:
            raise ValueError(
                f"{component_name} in {scenario} has names absent from the stochastic template: {extra[:5]}"
            )


def copy_scenario_values(
    target: pypsa.Network, source: pypsa.Network, scenario: str
) -> None:
    """Copy all compatible static and time-series values into one scenario."""
    _validate_network_axes(target, source, scenario)
    for component_name in target.components.keys():
        target_component = target.components[component_name]
        source_component = source.components[component_name]
        source_static = source_component.static
        target_static = target_component.static
        names = _component_names(source_static.index)
        target_names = _component_names(target_static.index)
        missing_names = target_names.difference(names)
        if len(missing_names):
            missing_rows = _scenario_static_index(
                target_static.index, scenario, missing_names
            )
            for column in (
                "p_nom",
                "e_nom",
                "s_nom",
                "p_min_pu",
                "p_max_pu",
            ):
                if column in target_static.columns:
                    target_static.loc[missing_rows, column] = 0.0
            for target_ts in target_component.dynamic.values():
                if target_ts.empty:
                    continue
                available_missing = missing_names.intersection(
                    _component_names(target_ts.columns)
                )
                if len(available_missing):
                    missing_columns = _scenario_dynamic_columns(
                        target_ts.columns, scenario, available_missing
                    )
                    target_ts.loc[:, missing_columns] = 0.0
        source_static = source_static.reindex(names)
        common_columns = source_static.columns.intersection(
            target_static.columns
        ).difference(SCENARIO_INVARIANT_ATTRIBUTES)
        target_rows = _scenario_static_index(target_static.index, scenario, names)
        target_static.loc[target_rows, common_columns] = source_static.loc[
            names, common_columns
        ].to_numpy()

        for attribute, source_ts in source_component.dynamic.items():
            if source_ts.empty:
                continue
            if attribute not in target_component.dynamic:
                raise ValueError(
                    f"{component_name}.{attribute} from {scenario} is absent in the base network"
                )
            target_ts = target_component.dynamic[attribute]
            if not target_ts.index.equals(source_ts.index):
                raise ValueError(
                    f"Time-series index differs for {component_name}.{attribute} in {scenario}"
                )
            source_names = pd.Index(source_ts.columns)
            target_columns = _scenario_dynamic_columns(
                target_ts.columns, scenario, source_names
            )
            target_ts.loc[:, target_columns] = source_ts.to_numpy()


def build_stochastic_network(
    network_paths: dict[str, Path],
    probabilities: pd.Series,
    solve_options: dict,
    foresight: str,
    planning_horizon: str | None,
    co2_sequestration_potential: float,
    limit_max_growth: object = None,
) -> pypsa.Network:
    """Prepare a base network, create scenarios, and populate their values."""
    missing = set(probabilities.index) - set(network_paths)
    if missing:
        raise ValueError(
            "Missing representative network paths: " + ", ".join(sorted(missing))
        )
    if not hasattr(pypsa.Network, "set_scenarios"):
        raise RuntimeError(
            "This workflow requires a PyPSA version with Network.set_scenarios()"
        )

    sources = {
        scenario: pypsa.Network(network_paths[scenario])
        for scenario in probabilities.index
    }
    harmonize_snapshots(sources)
    for source in sources.values():
        prepare_network(
            source,
            solve_opts=solve_options,
            foresight=foresight,
            planning_horizons=planning_horizon,
            co2_sequestration_potential=co2_sequestration_potential,
            limit_max_growth=limit_max_growth,
        )
    for source in sources.values():
        for component in source.components.values():
            optimized_columns = [
                column for column in component.static.columns if column.endswith("_opt")
            ]
            if optimized_columns:
                component.static.drop(columns=optimized_columns, inplace=True)
    template_scenario = max(
        sources,
        key=lambda scenario: sum(
            len(component.static) for component in sources[scenario].components.values()
        ),
    )
    LOGGER.info("Using %s as the union-rich stochastic template", template_scenario)
    base = sources[template_scenario].copy()
    base.set_scenarios(probabilities.to_dict())
    for scenario, source in sources.items():
        LOGGER.info("Copying deterministic network values for scenario %s", scenario)
        copy_scenario_values(base, source, scenario)
    return base


def main(snakemake) -> None:
    """Snakemake entry point."""
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    probabilities = read_representatives(Path(snakemake.input.representatives))
    network_paths = {
        scenario: Path(path)
        for scenario, path in zip(
            probabilities.index, snakemake.input.scenario_networks
        )
    }
    solve_options = snakemake.params.solving["options"]
    np.random.seed(solve_options.get("seed", 123))
    network = build_stochastic_network(
        network_paths=network_paths,
        probabilities=probabilities,
        solve_options=solve_options,
        foresight=snakemake.params.foresight,
        planning_horizon=snakemake.wildcards.get("planning_horizons"),
        co2_sequestration_potential=snakemake.params.co2_sequestration_potential,
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )
    network.meta = dict(
        snakemake.config,
        wildcards=dict(snakemake.wildcards),
        cssc_representatives=probabilities.to_dict(),
    )
    network.export_to_netcdf(snakemake.output.network)
    Path(snakemake.output.config).parent.mkdir(parents=True, exist_ok=True)
    with Path(snakemake.output.config).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(network.meta, stream, sort_keys=False, allow_unicode=True)
    LOGGER.info(
        "Exported CSSC stochastic pre-solve network to %s", snakemake.output.network
    )


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
