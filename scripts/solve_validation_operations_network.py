# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Solve linear operational validation on a sector-coupled network by combining:

1. an operation network defining the operating conditions
2. a solved capacity-expansion network providing the capacities to be fixed

The workflow is:
1. load the operation network
2. load the capacity-source network
3. copy optimal capacities from the source network to the operation network
4. optionally leave selected carriers extendable
5. prepare the operation network
6. solve dispatch only
"""

import logging
from collections.abc import Iterable
from functools import partial
import numpy as np
import pandas as pd
import pypsa

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._benchmark import memory_logger
from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import (
    _has_scenarios,
    collect_kwargs,
    extra_functionality,
    extra_functionality_stochastic_minimal,
    prepare_network,
)
import scripts.solve_network as solve_network_module

logger = logging.getLogger(__name__)


NOMINAL_ATTRS = {
    "Generator": "p_nom",
    "Link": "p_nom",
    "Store": "e_nom",
    "StorageUnit": "p_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
}


def log_remaining_extendable_assets(n: pypsa.Network) -> None:
    """
    Log remaining extendable active assets after capacity fixing.
    """
    rows = []

    for comp_name, attr in NOMINAL_ATTRS.items():
        if comp_name not in n.components.keys():
            continue

        comp = n.components[comp_name]
        df = comp.static
        extendable_col = f"{attr}_extendable"

        if extendable_col not in df.columns:
            continue

        ext_i = comp.extendables.difference(comp.inactive_assets)
        if len(ext_i) == 0:
            continue

        tmp = df.loc[ext_i].copy()
        tmp["component"] = comp_name
        tmp["asset"] = tmp.index
        tmp["nominal_attr"] = attr

        if "carrier" not in tmp.columns:
            tmp["carrier"] = ""

        rows.append(tmp[["component", "asset", "carrier", "nominal_attr"]])

    if rows:
        out = pd.concat(rows, axis=0)
        logger.info(
            "Remaining extendable active assets after capacity fixing:\n%s",
            out.to_string(index=False),
        )
    else:
        logger.info("No extendable active assets remain after capacity fixing.")

def load_capacity_source_network(path: str) -> pypsa.Network:
    """
    Load the capacity-source network.

    If the network is stochastic, select the first scenario only in the static
    component tables, since validation only needs optimized nominal capacities
    from static component data.
    """
    n = pypsa.Network(path)

    scenarios = getattr(n, "scenarios", None)
    if scenarios is None:
        return n

    scenario_list = list(scenarios)
    if len(scenario_list) == 0:
        return n

    first_scenario = scenario_list[0]
    logger.warning(
        "Capacity source network '%s' is stochastic. "
        "Using first scenario '%s' from static component tables as capacity source.",
        path,
        first_scenario,
    )

    for comp_name in NOMINAL_ATTRS:
        try:
            comp = n.components[comp_name]
        except KeyError:
            continue

        df = comp.static
        if df.empty:
            continue

        if isinstance(df.index, pd.MultiIndex) and "scenario" in df.index.names:
            try:
                comp.static = df.xs(first_scenario, level="scenario", axis=0)
            except KeyError:
                logger.warning(
                    "Component %s does not contain scenario '%s' in static table. "
                    "Leaving it unchanged.",
                    comp_name,
                    first_scenario,
                )

    return n

def copy_capacities_from_source_network(
    n_target: pypsa.Network,
    n_source: pypsa.Network,
    exclude_carriers: Iterable[str] | None = None,
    exclude_components: Iterable[str] | None = None,
    exclude_assets: dict[str, Iterable[str]] | None = None,
    missing_capacity_policy: str = "keep_extendable",
    verbose: bool = True,
) -> None:
    """
    Copy optimal capacities from a solved source network into a target network.

    For each extendable asset in the target network:
    - if the asset exists in the source network and a valid source capacity is found,
      copy it into the target nominal capacity and set extendable=False
    - if the asset is missing in the source network (or source capacity is missing),
      warn and apply the selected fallback policy

    Parameters
    ----------
    n_target : pypsa.Network
        Operation network to be solved.
    n_source : pypsa.Network
        Solved network used as source of capacities.
    exclude_carriers : iterable[str] or None, default None
        Carriers to leave extendable in the target network.
    exclude_components : iterable[str] or None, default None
        Components to skip entirely.
    exclude_assets : dict[str, iterable[str]] or None, default None
        Explicit asset names to skip, grouped by component name.
    missing_capacity_policy : str, default "keep_extendable"
        Behaviour when an extendable target asset cannot be matched to a valid
        source capacity. Supported values:
        - "keep_extendable": keep the asset extendable and only warn
        - "fix_current": keep current nominal value and set extendable=False
    verbose : bool, default True
        Whether to log summary tables.

    Returns
    -------
    None
        The target network is modified in place.
    """
    exclude_carriers = set(exclude_carriers or [])
    exclude_components = set(exclude_components or [])
    exclude_assets = {
        comp: set(asset_names) for comp, asset_names in (exclude_assets or {}).items()
    }

    if missing_capacity_policy not in {"keep_extendable", "fix_current"}:
        raise ValueError(
            "missing_capacity_policy must be either "
            "'keep_extendable' or 'fix_current'."
        )

    summary_rows = []

    for comp_name, attr in NOMINAL_ATTRS.items():
        if comp_name in exclude_components:
            if verbose:
                logger.info("Skipping component %s entirely.", comp_name)
            continue

        try:
            comp_target = n_target.components[comp_name]
        except KeyError:
            continue

        try:
            comp_source = n_source.components[comp_name]
        except KeyError:
            logger.warning(
                "Component %s exists in target network but not in source network. "
                "Nothing will be copied for this component.",
                comp_name,
            )
            continue

        df_target = comp_target.static
        df_source = comp_source.static

        extendable_col = f"{attr}_extendable"
        opt_col = f"{attr}_opt"

        if extendable_col not in df_target.columns:
            continue

        ext_i = comp_target.extendables.difference(comp_target.inactive_assets)

        if len(ext_i) == 0:
            continue

        candidate_i = pd.Index(ext_i)

        if exclude_carriers and "carrier" in df_target.columns:
            excluded_by_carrier = df_target.index[
                df_target.index.isin(candidate_i)
                & df_target["carrier"].isin(exclude_carriers)
            ]
            candidate_i = candidate_i.difference(excluded_by_carrier)
        else:
            excluded_by_carrier = pd.Index([])

        explicitly_excluded = pd.Index(exclude_assets.get(comp_name, []))
        explicitly_excluded = explicitly_excluded.intersection(candidate_i)
        candidate_i = candidate_i.difference(explicitly_excluded)

        fixed_assets = []
        missing_in_source = []
        missing_source_value = []

        for asset in candidate_i:
            if asset not in df_source.index:
                missing_in_source.append(asset)
                continue

            source_value = None

            if opt_col in df_source.columns and pd.notna(df_source.at[asset, opt_col]):
                source_value = df_source.at[asset, opt_col]
            elif attr in df_source.columns and pd.notna(df_source.at[asset, attr]):
                source_value = df_source.at[asset, attr]

            if source_value is None or pd.isna(source_value):
                missing_source_value.append(asset)
                continue

            df_target.at[asset, attr] = source_value
            df_target.at[asset, extendable_col] = False
            fixed_assets.append(asset)

        unresolved = pd.Index(missing_in_source).union(pd.Index(missing_source_value))

        if len(missing_in_source) > 0:
            logger.warning(
                "Component %s: %s target assets were not found in the capacity source network. "
                "Examples: %s",
                comp_name,
                len(missing_in_source),
                list(missing_in_source[:10]),
            )

        if len(missing_source_value) > 0:
            logger.warning(
                "Component %s: %s target assets exist in the capacity source network but "
                "have no valid source capacity. Examples: %s",
                comp_name,
                len(missing_source_value),
                list(missing_source_value[:10]),
            )

        if len(unresolved) > 0 and missing_capacity_policy == "fix_current":
            df_target.loc[unresolved, extendable_col] = False
            logger.warning(
                "Component %s: %s unresolved target assets were fixed at their current nominal value.",
                comp_name,
                len(unresolved),
            )

        source_only = pd.Index(df_source.index).difference(df_target.index)
        if len(source_only) > 0:
            logger.warning(
                "Component %s: %s assets exist in the capacity source network but not in the "
                "operation network. They were ignored. Examples: %s",
                comp_name,
                len(source_only),
                list(source_only[:10]),
            )

        summary_rows.append(
            {
                "component": comp_name,
                "attr": attr,
                "extendable_active": len(ext_i),
                "fixed_from_source": len(fixed_assets),
                "excluded_by_carrier": len(excluded_by_carrier),
                "excluded_explicitly": len(explicitly_excluded),
                "missing_in_source": len(missing_in_source),
                "missing_source_value": len(missing_source_value),
                "policy_applied_to_unresolved": (
                    len(unresolved) if missing_capacity_policy == "fix_current" else 0
                ),
            }
        )

    if verbose and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        logger.info(
            "Validation capacity fixing summary:\n%s",
            summary_df.to_string(index=False),
        )

def _validation_extra_functionality(
    n: pypsa.Network,
    planning_horizons: str | None,
) -> None:
    """
    Add the same extra functionality used in solve_network, while ensuring
    that the solve_network module can access the current snakemake object.
    """
    n.config = snakemake.config
    n.params = snakemake.params

    # Inject snakemake into the solve_network module globals.
    # This is needed because extra_functionality() looks up `snakemake`
    # in the module where it is defined, not in this validation script.
    solve_network_module.snakemake = snakemake

    extra_fn = (
        extra_functionality_stochastic_minimal
        if _has_scenarios(n)
        else extra_functionality
    )

    extra_fn(n, n.snapshots, planning_horizons)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_validation_operations_network",
            configfiles=["config/test_stochastic_scenarios/config.yaml"],
            clusters="adm",
            opts="",
            sector_opts="",
            planning_horizons="2050",
            cap_source="stochastic_network",
            op_source="agriculture_full_electric",
            run_prefix="eth_results"
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options
    cf_solving = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    n_operation = pypsa.Network(snakemake.input.operation_network)
    n_capacity = load_capacity_source_network(snakemake.input.capacity_network)

    dispatch_exclude_carriers = cf_solving.get("dispatch_exclude_carriers", [])
    dispatch_exclude_components = cf_solving.get("dispatch_exclude_components", [])
    dispatch_exclude_assets = cf_solving.get("dispatch_exclude_assets", {})
    missing_capacity_policy = cf_solving.get(
        "validation_missing_capacity_policy",
        "keep_extendable",
    )

    logger.info(
        "Validation run: capacities from '%s', operations on '%s'.",
        snakemake.wildcards.cap_source,
        snakemake.wildcards.op_source,
    )
    logger.info("Operation network: %s", snakemake.input.operation_network)
    logger.info("Capacity source network: %s", snakemake.input.capacity_network)
    logger.info("Excluded carriers from fixing: %s", list(dispatch_exclude_carriers))
    logger.info("Missing capacity policy: %s", missing_capacity_policy)

    copy_capacities_from_source_network(
        n_target=n_operation,
        n_source=n_capacity,
        exclude_carriers=dispatch_exclude_carriers,
        exclude_components=dispatch_exclude_components,
        exclude_assets=dispatch_exclude_assets,
        missing_capacity_policy=missing_capacity_policy,
        verbose=True,
    )

    log_remaining_extendable_assets(n_operation)

    prepare_network(
        n_operation,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )

    rolling_horizon = cf_solving.get("rolling_horizon", False)
    mode = "rolling_horizon" if rolling_horizon else "single"

    model_kwargs, solve_kwargs = collect_kwargs(
        snakemake.config,
        snakemake.params.solving,
        planning_horizons,
        log_fn=snakemake.log.solver,
        mode=mode,
    )

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None),
        interval=logging_frequency,
    ) as mem:
        if rolling_horizon:
            logger.info(
                "Solving validation operations network with rolling horizon..."
            )

            n_operation.config = snakemake.config
            n_operation.params = snakemake.params

            extra_fn = (
                extra_functionality_stochastic_minimal
                if _has_scenarios(n_operation)
                else extra_functionality
            )

            model_kwargs["extra_functionality"] = partial(
                extra_fn,
                planning_horizons=planning_horizons,
            )

            n_operation.optimize.optimize_with_rolling_horizon(**model_kwargs)

        else:
            logger.info("Creating validation optimization model...")

            n_operation.config = snakemake.config
            n_operation.params = snakemake.params

            n_operation.optimize.create_model(**model_kwargs)

            logger.info("Adding validation extra functionality...")
            _validation_extra_functionality(
                n_operation,
                planning_horizons=planning_horizons,
            )

            logger.info("Solving validation operations network...")
            n_operation.optimize.solve_model(**solve_kwargs)

    logger.info("Maximum memory usage: %s", mem.mem_usage)

    n_operation.meta = dict(
        snakemake.config,
        **dict(
            wildcards=dict(snakemake.wildcards),
            validation=dict(
                capacities_from=snakemake.wildcards.cap_source,
                operations_on=snakemake.wildcards.op_source,
                capacity_network=snakemake.input.capacity_network,
                operation_network=snakemake.input.operation_network,
            ),
        ),
    )

    n_operation.export_to_netcdf(snakemake.output.network)