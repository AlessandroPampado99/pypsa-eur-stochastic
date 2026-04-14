# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/PyPSA/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch for a sector-coupled network using the capacities
of a previously optimized capacity-expansion solution.

The workflow is:
1. Load an already optimized network
2. Fix all optimal capacities
3. Optionally clean problematic extendable assets with zero/negative capital costs
4. Prepare the network
5. Solve dispatch only
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._benchmark import memory_logger
from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import (
    collect_kwargs,
    prepare_network,
)

logger = logging.getLogger(__name__)


from collections.abc import Iterable

def log_remaining_extendable_assets(n: pypsa.Network) -> None:
    """
    Log remaining extendable active assets after dispatch fixing.
    Useful to verify which assets are still free.
    """
    nominal_attrs = {
        "Generator": "p_nom",
        "Link": "p_nom",
        "Store": "e_nom",
        "StorageUnit": "p_nom",
        "Line": "s_nom",
        "Transformer": "s_nom",
    }

    rows = []

    for comp_name, attr in nominal_attrs.items():
        if comp_name not in n.components:
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
            "Remaining extendable active assets after fixing:\n%s",
            out.to_string(index=False),
        )
    else:
        logger.info("No extendable active assets remain after fixing.")


def fix_optimal_capacities_except_carriers(
    n: pypsa.Network,
    exclude_carriers: Iterable[str] | None = None,
    exclude_components: Iterable[str] | None = None,
    exclude_assets: dict[str, Iterable[str]] | None = None,
    verbose: bool = True,
) -> None:
    """
    Fix optimized capacities for all extendable assets, except those belonging
    to selected carriers/components/assets.

    For each component with a nominal capacity attribute, this function:
    1. copies <attr>_opt into <attr> for extendable assets
    2. sets <attr>_extendable = False

    Assets can be excluded from fixing by:
    - carrier name
    - component name
    - explicit asset names

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network.
    exclude_carriers : iterable[str] or None, default None
        Carriers to leave extendable. Matching is exact on the `carrier` column.
    exclude_components : iterable[str] or None, default None
        Component names to skip entirely, e.g. ["Store", "Link"].
    exclude_assets : dict[str, iterable[str]] or None, default None
        Explicit assets to skip, grouped by component name.
        Example:
            {
                "Store": ["co2 stored", "co2 sequestered"],
                "Generator": ["load shedding"]
            }
    verbose : bool, default True
        Whether to log summary information.

    Returns
    -------
    None
        The network is modified in place.
    """
    exclude_carriers = set(exclude_carriers or [])
    exclude_components = set(exclude_components or [])
    exclude_assets = {
        comp: set(asset_names) for comp, asset_names in (exclude_assets or {}).items()
    }

    # Same mapping conceptually used by PyPSA for nominal capacities
    nominal_attrs = {
        "Generator": "p_nom",
        "Link": "p_nom",
        "Store": "e_nom",
        "StorageUnit": "p_nom",
        "Line": "s_nom",
        "Transformer": "s_nom",
    }

    summary_rows = []

    for comp_name, attr in nominal_attrs.items():
        if comp_name in exclude_components:
            if verbose:
                logger.info("Skipping component %s entirely.", comp_name)
            continue

        if comp_name not in n.components:
            continue

        comp = n.components[comp_name]
        df = comp.static

        extendable_col = f"{attr}_extendable"
        opt_col = f"{attr}_opt"

        if extendable_col not in df.columns:
            continue

        # Active extendable assets only
        ext_i = comp.extendables.difference(comp.inactive_assets)

        if len(ext_i) == 0:
            continue

        # Start from all extendable active assets
        fix_i = pd.Index(ext_i)

        # Exclude by carrier
        if exclude_carriers and "carrier" in df.columns:
            excluded_by_carrier = df.index[
                df.index.isin(fix_i) & df["carrier"].isin(exclude_carriers)
            ]
            fix_i = fix_i.difference(excluded_by_carrier)
        else:
            excluded_by_carrier = pd.Index([])

        # Exclude by explicit asset names
        explicitly_excluded = pd.Index(exclude_assets.get(comp_name, []))
        explicitly_excluded = explicitly_excluded.intersection(fix_i)
        fix_i = fix_i.difference(explicitly_excluded)

        # Nothing left to fix
        if len(fix_i) == 0:
            summary_rows.append(
                {
                    "component": comp_name,
                    "attr": attr,
                    "extendable_active": len(ext_i),
                    "fixed": 0,
                    "excluded_by_carrier": len(excluded_by_carrier),
                    "excluded_explicitly": len(explicitly_excluded),
                }
            )
            continue

        if opt_col not in df.columns:
            raise KeyError(
                f"Component {comp_name} has extendable assets but missing column {opt_col}."
            )

        missing_opt = df.loc[fix_i, opt_col].isna()
        if missing_opt.any():
            bad_assets = list(df.loc[fix_i[missing_opt]].index[:10])
            raise ValueError(
                f"Component {comp_name}: some assets to be fixed have missing {opt_col}. "
                f"Examples: {bad_assets}"
            )

        # Copy optimized capacities into nominal capacities
        df.loc[fix_i, attr] = df.loc[fix_i, opt_col]

        # Disable extendability only for fixed assets
        df.loc[fix_i, extendable_col] = False

        summary_rows.append(
            {
                "component": comp_name,
                "attr": attr,
                "extendable_active": len(ext_i),
                "fixed": len(fix_i),
                "excluded_by_carrier": len(excluded_by_carrier),
                "excluded_explicitly": len(explicitly_excluded),
            }
        )

    if verbose and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        logger.info(
            "Capacity fixing summary:\n%s",
            summary_df.to_string(index=False),
        )


def sanitize_zero_capital_extendables_before_fixing(n: pypsa.Network) -> None:
    """
    Prevent pathological behaviour in dispatch-only runs caused by assets that
    remain extendable with zero or negative capital cost before fixing capacities.

    This is mainly a safeguard. In a clean workflow, the input network should
    already come from a solved expansion problem and contain valid *_opt values.
    """
    nominal_attrs = {
        "Generator": "p_nom",
        "Link": "p_nom",
        "Store": "e_nom",
        "StorageUnit": "p_nom",
        "Line": "s_nom",
        "Transformer": "s_nom",
    }

    for comp_name, attr in nominal_attrs.items():
        if comp_name not in n.components:
            continue

        df = n.components[comp_name].static
        extendable_col = f"{attr}_extendable"
        opt_col = f"{attr}_opt"

        if extendable_col not in df.columns:
            continue

        extendable = df[extendable_col].fillna(False)
        if not extendable.any():
            continue

        cap_cost = df["capital_cost"] if "capital_cost" in df.columns else pd.Series(0.0, index=df.index)
        has_opt = opt_col in df.columns

        # Assets that are still extendable but have no valid optimized capacity
        # and non-positive capital cost are dangerous in a dispatch-only run.
        bad = extendable & (cap_cost <= 0.0)

        if has_opt:
            bad &= df[opt_col].isna()

        if bad.any():
            logger.warning(
                "Found %s %s assets with non-positive capital cost and no valid optimized capacity. "
                "Their extendability will be disabled before fixing capacities: %s",
                int(bad.sum()),
                comp_name,
                list(df.index[bad])[:10],
            )
            df.loc[bad, extendable_col] = False


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_sector_network",
            configfiles="config/config.yaml",
            opts="",
            clusters="10",
            sector_opts="",
            planning_horizons="2050",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options
    cf_solving = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    # Fix capacities from previous optimization
    dispatch_exclude_carriers = snakemake.params.solving["options"].get(
        "dispatch_exclude_carriers", []
    )

    fix_optimal_capacities_except_carriers(
        n,
        exclude_carriers=dispatch_exclude_carriers,
    )

    log_remaining_extendable_assets(n)
    
    # Prepare network for operational solve
    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )

    rolling_horizon = cf_solving.get("rolling_horizon", False)
    mode = "rolling_horizon" if rolling_horizon else "single"

    all_kwargs, _ = collect_kwargs(
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
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        if rolling_horizon:
            logger.info("Solving sector-coupled operations network with rolling horizon...")
            n.optimize.optimize_with_rolling_horizon(**all_kwargs)
        else:
            logger.info("Solving sector-coupled operations network...")
            n.optimize(**all_kwargs)

    logger.info("Maximum memory usage: %s", mem.mem_usage)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)