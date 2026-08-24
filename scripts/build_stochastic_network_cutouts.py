#!/usr/bin/env python3
"""
Build a stochastic pre-solve network from CSSC representative networks.

Each representative's solved deterministic network supplies all compatible
static and time-dependent component values for its stochastic scenario. The
CSSC reduced probabilities are passed directly to ``Network.set_scenarios``.
The network is prepared before scenarios are created, matching the existing
``stochasticify_network.py`` / ``solve_network.py`` contract.
"""

from __future__ import annotations

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
        if base_names != source_names:
            missing = sorted(base_names - source_names)
            extra = sorted(source_names - base_names)
            raise ValueError(
                f"{component_name} names differ for {scenario}; missing={missing[:5]}, extra={extra[:5]}"
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
        if source_static.empty:
            continue
        target_static = target_component.static
        names = _component_names(source_static.index)
        source_static = source_static.reindex(names)
        common_columns = source_static.columns.intersection(target_static.columns)
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
    for source in sources.values():
        prepare_network(
            source,
            solve_opts=solve_options,
            foresight=foresight,
            planning_horizons=planning_horizon,
            co2_sequestration_potential=co2_sequestration_potential,
            limit_max_growth=limit_max_growth,
        )
    base = sources[probabilities.index[0]].copy()
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
