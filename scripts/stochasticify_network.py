# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prepare a deterministic network and (optionally) convert it into a stochastic one.

This script is intended to run BEFORE solve_network.py:
- Load the pre-solve network
- Call prepare_network(...) to add all components that must exist before set_scenarios
- If enabled, call n.set_scenarios(...) and apply scenario-specific patches
- Export the "pre-solve stochastic" network to NetCDF
"""

import logging
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pypsa
import yaml

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import (
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)

# Import prepare_network from solve_network to avoid duplication
from scripts.solve_network import prepare_network  # noqa: E402


logger = logging.getLogger(__name__)


def _read_yaml_maybe(path: str | None) -> dict:
    """Read a YAML file if path is provided and exists; return {} otherwise."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stochastic config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dict(x: Any, name: str) -> dict:
    """Ensure x is a dict, else raise."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    raise TypeError(f"{name} must be a dict; got {type(x).__name__}")


def _get_level_names(idx: pd.Index) -> pd.Index:
    """Return the 'name' level if MultiIndex, else the index itself."""
    if isinstance(idx, pd.MultiIndex):
        return idx.get_level_values("name")
    return idx


def _select_names_from_component(
    n: pypsa.Network,
    comp: str,
    selector: Mapping[str, Any],
) -> list[str]:
    """
    Select component names (without scenario level) based on a selector.

    Supported selector keys:
    - names: list[str] or str (regex)
    - carrier: str or list[str]
    - bus / bus0 / bus1: str or list[str] (exact match)
    - any other column in the component table: exact match
    """
    selector = _ensure_dict(selector, "selector")
    df = getattr(n, comp)  # e.g. n.generators, n.links, n.loads, ...

    # Work on base index names (no scenario)
    idx_names = _get_level_names(df.index)
    base = df.copy()
    base.index = idx_names

    mask = pd.Series(True, index=base.index)

    # names selector (regex or explicit list)
    names_sel = selector.get("names")
    if isinstance(names_sel, str):
        pattern = re.compile(names_sel)
        mask &= base.index.to_series().apply(lambda s: bool(pattern.search(s)))
    elif isinstance(names_sel, (list, tuple, set)):
        mask &= base.index.isin(list(names_sel))

    # carrier selector
    carrier_sel = selector.get("carrier")
    if carrier_sel is not None and "carrier" in base.columns:
        if isinstance(carrier_sel, str):
            carrier_sel = [carrier_sel]
        mask &= base["carrier"].isin(list(carrier_sel))

    # bus selectors
    for bcol in ("bus", "bus0", "bus1", "bus2", "bus3", "bus4"):
        if bcol in selector and bcol in base.columns:
            val = selector[bcol]
            if isinstance(val, str):
                val = [val]
            mask &= base[bcol].isin(list(val))

    # other columns exact match
    for k, v in selector.items():
        if k in ("names", "carrier", "bus", "bus0", "bus1", "bus2", "bus3", "bus4"):
            continue
        if k in base.columns:
            if isinstance(v, (list, tuple, set)):
                mask &= base[k].isin(list(v))
            else:
                mask &= base[k].eq(v)

    return pd.Index(base.index[mask]).unique().tolist()


def _apply_patch_static(
    df: pd.DataFrame,
    col: str,
    scenario: str,
    names: list[str],
    op: str,
    value: float,
) -> None:
    """Apply a scalar patch to a static component table."""
    if not isinstance(df.index, pd.MultiIndex):
        # Deterministic case
        if op == "set":
            df.loc[names, col] = value
        elif op == "scale":
            df.loc[names, col] = df.loc[names, col] * value
        elif op == "add":
            df.loc[names, col] = df.loc[names, col] + value
        else:
            raise ValueError(f"Unsupported op: {op}")
        return

    # Stochastic case: index is MultiIndex (scenario, name)
    idx = pd.MultiIndex.from_product([[scenario], names], names=["scenario", "name"])
    if op == "set":
        df.loc[idx, col] = value
    elif op == "scale":
        df.loc[idx, col] = df.loc[idx, col] * value
    elif op == "add":
        df.loc[idx, col] = df.loc[idx, col] + value
    else:
        raise ValueError(f"Unsupported op: {op}")


def _apply_patch_timeseries(
    ts: pd.DataFrame,
    scenario: str,
    names: list[str],
    op: str,
    value: Any,
) -> None:
    """
    Apply a patch to a time series DataFrame.

    For stochastic networks, columns are MultiIndex (scenario, name).
    value can be:
    - scalar
    - array-like with length == len(ts.index)
    - DataFrame with columns matching names
    """

    # Keep only names that actually exist in the target time series table
    # (sector-coupled networks often have loads without a time-dependent p_set column)
    if isinstance(ts.columns, pd.MultiIndex):
        # Be robust if column levels are unnamed
        if "scenario" in ts.columns.names:
            scenarios_avail = set(ts.columns.get_level_values("scenario"))
        else:
            scenarios_avail = set(ts.columns.get_level_values(0))

        if scenario not in scenarios_avail:
            logger.warning(
                f"Timeseries patch: scenario '{scenario}' not found in ts.columns; skipping."
            )
            return

        if "name" in ts.columns.names:
            avail_names = set(ts.columns.get_level_values("name"))
        else:
            avail_names = set(ts.columns.get_level_values(1))
    else:
        avail_names = set(ts.columns)

    names = [n for n in pd.Index(names).unique().tolist() if n in avail_names]
    if not names:
        logger.warning(
            f"Timeseries patch matched no existing columns for scenario '{scenario}'; skipping."
        )
        return

    # Now build the column selector
    if not isinstance(ts.columns, pd.MultiIndex):
        cols = names
    else:
        cols = pd.MultiIndex.from_product([[scenario], names], names=["scenario", "name"])

    if np.isscalar(value):
        if op == "set":
            ts.loc[:, cols] = value
        elif op == "scale":
            ts.loc[:, cols] = ts.loc[:, cols] * value
        elif op == "add":
            ts.loc[:, cols] = ts.loc[:, cols] + value
        else:
            raise ValueError(f"Unsupported op: {op}")
        return

    if isinstance(value, pd.DataFrame):
        v = value.reindex(ts.index)
        if v.isnull().values.any():
            raise ValueError("Provided DataFrame value has NaNs after reindexing to snapshots.")
        if not set(names).issubset(set(v.columns)):
            missing = sorted(set(names) - set(v.columns))
            raise ValueError(f"Provided DataFrame value missing columns: {missing}")
        v = v[names]

        if isinstance(ts.columns, pd.MultiIndex):
            v.columns = cols  # lift to (scenario, name)

        if op == "set":
            ts.loc[:, cols] = v.values
        elif op == "scale":
            ts.loc[:, cols] = ts.loc[:, cols].values * v.values
        elif op == "add":
            ts.loc[:, cols] = ts.loc[:, cols].values + v.values
        else:
            raise ValueError(f"Unsupported op: {op}")
        return

    arr = np.asarray(value)
    if arr.ndim != 1 or len(arr) != len(ts.index):
        raise ValueError(
            f"Array-like value must be 1D and match snapshots length ({len(ts.index)}); got shape {arr.shape}"
        )
    if op == "set":
        ts.loc[:, cols] = arr[:, None]
    elif op == "scale":
        ts.loc[:, cols] = ts.loc[:, cols].values * arr[:, None]
    elif op == "add":
        ts.loc[:, cols] = ts.loc[:, cols].values + arr[:, None]
    else:
        raise ValueError(f"Unsupported op: {op}")


def apply_stochastic_config(n: pypsa.Network, config: dict, stochastic_param: dict) -> None:
    """
    Enable scenarios and apply scenario-specific patches.

    Expected configuration structure (minimal):
    stochastic:
      enabled: true
      scenarios:
        low: 0.3
        mid: 0.4
        high: 0.3
      # optional: file: "config/stochastic.yaml"  (merged with this block)
      patches:
        - target: "generators.marginal_cost"
          selector: {carrier: "gas"}
          op: "scale"
          values: {low: 0.8, mid: 1.0, high: 1.3}
        - target: "loads_t.p_set"
          selector: {names: ".*"}   # regex
          op: "scale"
          values: {low: 0.95, mid: 1.0, high: 1.05}
    """
    stoch = _ensure_dict(stochastic_param, "stochastic")
    enabled = bool(stoch.get("enable", stoch.get("enabled", False)))
    if not enabled:
        logger.info("Stochastic disabled: skipping set_scenarios and patches.")
        return

    # Merge optional external YAML
    external = _read_yaml_maybe(stoch.get("file", ""))
    if external:
        merged = dict(external)
        merged.update(stoch)  # inline overrides external
        stoch = merged

    scenarios = stoch.get("scenarios", None)
    if scenarios is None:
        raise ValueError("run.stochastic_scenarios.enable=true but no scenarios provided (in file or inline).")

    logger.info("Enabling stochastic scenarios via n.set_scenarios(...)")
    n.set_scenarios(scenarios)  # New API in PyPSA v1.0+ :contentReference[oaicite:0]{index=0}

    patches = stoch.get("patches", [])
    if not patches:
        logger.info("No stochastic patches provided; network is stochastic but unchanged across scenarios.")
        return

    # Map *_t tables to their base component tables
    def base_comp_from_table(table: str) -> str:
        return table[:-2] if table.endswith("_t") else table

    for i, patch in enumerate(patches, start=1):
        patch = _ensure_dict(patch, f"patch[{i}]")
        target = patch.get("target")
        if not isinstance(target, str) or "." not in target:
            raise ValueError(f"patch[{i}].target must be like 'generators.marginal_cost' or 'loads_t.p_set'")

        table, attr = target.split(".", 1)
        selector = _ensure_dict(patch.get("selector", {}), f"patch[{i}].selector")
        op = patch.get("op", "set")
        values = _ensure_dict(patch.get("values", {}), f"patch[{i}].values")

        # Select base names
        comp_for_selection = base_comp_from_table(table)
        names = _select_names_from_component(n, comp_for_selection, selector)
        if not names:
            logger.warning(f"patch[{i}] matched no components; skipping. target={target}")
            continue

        logger.info(f"Applying patch[{i}] target={target} op={op} matched={len(names)}")

        # Resolve target DF
        if table.endswith("_t"):
            ts_container = getattr(n, table)          # e.g. n.loads_t
            ts = getattr(ts_container, attr)          # e.g. n.loads_t.p_set

            for sc, v in values.items():
                if isinstance(v, str) and v.endswith((".csv", ".parquet", ".pq")):
                    vp = Path(v)
                    if not vp.exists():
                        raise FileNotFoundError(f"patch[{i}] value file not found: {vp}")
                    if vp.suffix == ".csv":
                        dfv = pd.read_csv(vp, index_col=0, parse_dates=True)
                    else:
                        dfv = pd.read_parquet(vp)
                    _apply_patch_timeseries(ts, sc, names, op, dfv)
                else:
                    _apply_patch_timeseries(ts, sc, names, op, v)

        else:
            comp_df = getattr(n, table)               # e.g. n.generators
            if attr not in comp_df.columns:
                raise KeyError(f"patch[{i}] column not found: {table}.{attr}")

            for sc, v in values.items():
                if not np.isscalar(v):
                    raise ValueError(f"patch[{i}] static patch values must be scalar; got {type(v).__name__}")
                _apply_patch_static(comp_df, attr, sc, names, op, float(v))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "stochasticify_sector_network",
            opts="",
            clusters="adm",
            configfiles="config/test_stoch/config.yaml",
            # run="__debug",
            sector_opts="",
            planning_horizons="2050",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # Load network
    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    # Prepare network (must happen BEFORE set_scenarios because it can add components)
    solve_opts = snakemake.params.solving["options"]
    np.random.seed(solve_opts.get("seed", 123))

    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )

    # Enable stochastic + apply patches (if enabled)
    apply_stochastic_config(
        n,
        config=snakemake.config,
        stochastic_param=snakemake.params.get("stochastic_scenarios", {}),
    )

    # Export
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w", encoding="utf-8") as f:
        yaml.dump(
            n.meta,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info(f"Exported stochastic pre-solve network to {snakemake.output.network}")
