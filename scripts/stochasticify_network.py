# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prepare a deterministic network and (optionally) convert it into a stochastic one.

This script is intended to run BEFORE solve_network.py:
- Load the pre-solve network
- Call prepare_network(...) to add all components that must exist before set_scenarios
- If enabled, call n.set_scenarios(...) and apply scenario-specific patches
- If structured scenarios are configured, dispatch scenario builders by scenario name
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import (
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)

from scripts.solve_network import prepare_network  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------
# YAML / generic helpers
# ---------------------------

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


def _merge_stochastic_param(stochastic_param: dict) -> dict:
    """Merge inline stochastic config with optional external YAML."""
    stoch = _ensure_dict(stochastic_param, "stochastic_scenarios")
    external = _read_yaml_maybe(stoch.get("file", ""))
    if external:
        merged = dict(external)
        merged.update(stoch)
        stoch = merged
    return stoch


# ---------------------------
# Structured scenario builders
# ---------------------------

def _scenario_A(n: pypsa.Network, scenario: str | None = None, config: dict | None = None) -> None:
    """No-op scenario A used to validate scenario dispatch."""
    if scenario is None:
        logger.info("Structured scenario A inserted in deterministic mode.")
    else:
        logger.info(f"Structured scenario A inserted for stochastic scenario '{scenario}'.")


def _scenario_B(n: pypsa.Network, scenario: str | None = None, config: dict | None = None) -> None:
    """No-op scenario B used to validate scenario dispatch."""
    if scenario is None:
        logger.info("Structured scenario B inserted in deterministic mode.")
    else:
        logger.info(f"Structured scenario B inserted for stochastic scenario '{scenario}'.")


def _scenario_C(n: pypsa.Network, scenario: str | None = None, config: dict | None = None) -> None:
    """No-op scenario C used to validate scenario dispatch."""
    if scenario is None:
        logger.info("Structured scenario C inserted in deterministic mode.")
    else:
        logger.info(f"Structured scenario C inserted for stochastic scenario '{scenario}'.")


def _scenario_D(n: pypsa.Network, scenario: str | None = None, config: dict | None = None) -> None:
    """No-op scenario D used to validate scenario dispatch."""
    if scenario is None:
        logger.info("Structured scenario D inserted in deterministic mode.")
    else:
        logger.info(f"Structured scenario D inserted for stochastic scenario '{scenario}'.")


STRUCTURED_SCENARIOS = {
    "A": _scenario_A,
    "B": _scenario_B,
    "C": _scenario_C,
    "D": _scenario_D,
}


def _apply_named_structured_scenario(
    n: pypsa.Network,
    scenario_name: str,
    config: dict,
    scenario: str | None = None,
) -> None:
    """Dispatch a structured scenario by name."""
    if scenario_name not in STRUCTURED_SCENARIOS:
        known = ", ".join(sorted(STRUCTURED_SCENARIOS))
        raise ValueError(
            f"Unknown structured scenario '{scenario_name}'. Known structured scenarios: {known}"
        )

    logger.info(
        "Applying structured scenario builder '%s' (target=%s).",
        scenario_name,
        f"stochastic:{scenario}" if scenario is not None else "deterministic",
    )
    STRUCTURED_SCENARIOS[scenario_name](n=n, scenario=scenario, config=config)


def _resolve_deterministic_structured_scenario(
    config: dict,
    wildcards: Mapping[str, Any] | None = None,
) -> str | None:
    """
    Resolve the active structured scenario in deterministic mode.

    Precedence:
    1. config['structured_scenario']
    2. config['scenario']['structured_name']
    3. wildcards['run']
    4. config['run']['name']

    Only exact names present in STRUCTURED_SCENARIOS are accepted.
    """
    wildcards = wildcards or {}

    scenario_block = config.get("scenario", {})
    if not isinstance(scenario_block, dict):
        scenario_block = {}

    run_block = config.get("run", {})
    if not isinstance(run_block, dict):
        run_block = {}

    candidates = [
        config.get("structured_scenario"),
        scenario_block.get("structured_name"),
        wildcards.get("run"),
        run_block.get("name"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate in STRUCTURED_SCENARIOS:
            return candidate

    return None


def _validate_stochastic_structured_scenarios(scenarios: Mapping[str, Any]) -> None:
    """Ensure all stochastic scenario names have a registered structured builder."""
    unknown = sorted(set(scenarios) - set(STRUCTURED_SCENARIOS))
    if unknown:
        known = ", ".join(sorted(STRUCTURED_SCENARIOS))
        raise ValueError(
            f"Stochastic scenario file contains unknown structured scenarios {unknown}. "
            f"Known structured scenarios: {known}"
        )


def _apply_structured_scenarios(
    n: pypsa.Network,
    config: dict,
    stochastic_param: dict,
    wildcards: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Apply structured scenarios in stochastic or deterministic mode.

    Returns
    -------
    is_stochastic : bool
        Whether the stochastic mode was enabled.
    active_names : list[str]
        List of structured scenario names that were applied.
    """
    wildcards = wildcards or {}
    stoch = _merge_stochastic_param(stochastic_param)
    enabled = bool(stoch.get("enable", stoch.get("enabled", False)))

    if enabled:
        scenarios = stoch.get("scenarios", None)
        if scenarios is None:
            raise ValueError(
                "stochastic_scenarios.enable=true but no scenarios were provided "
                "(inline or through the referenced YAML file)."
            )

        _validate_stochastic_structured_scenarios(scenarios)

        logger.info("Enabling stochastic scenarios via n.set_scenarios(...)")
        n.set_scenarios(scenarios)

        active_names = list(scenarios.keys())
        logger.info("Structured stochastic scenarios detected: %s", active_names)

        for sc in active_names:
            _apply_named_structured_scenario(
                n=n,
                scenario_name=sc,
                config=config,
                scenario=sc,
            )

        return True, active_names

    scenario_name = _resolve_deterministic_structured_scenario(config=config, wildcards=wildcards)
    if scenario_name is None:
        logger.info(
            "Stochastic mode disabled and no deterministic structured scenario detected. "
            "No structured scenario builder will be applied."
        )
        return False, []

    logger.info("Deterministic structured scenario detected: %s", scenario_name)
    _apply_named_structured_scenario(
        n=n,
        scenario_name=scenario_name,
        config=config,
        scenario=None,
    )
    return False, [scenario_name]


# ---------------------------
# Patch-based selectors/helpers
# ---------------------------

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
    df = getattr(n, comp)

    idx_names = _get_level_names(df.index)
    base = df.copy()
    base.index = idx_names

    mask = pd.Series(True, index=base.index)

    names_sel = selector.get("names")
    if isinstance(names_sel, str):
        pattern = re.compile(names_sel)
        mask &= base.index.to_series().apply(lambda s: bool(pattern.search(s)))
    elif isinstance(names_sel, (list, tuple, set)):
        mask &= base.index.isin(list(names_sel))

    carrier_sel = selector.get("carrier")
    if carrier_sel is not None and "carrier" in base.columns:
        if isinstance(carrier_sel, str):
            carrier_sel = [carrier_sel]
        mask &= base["carrier"].isin(list(carrier_sel))

    for bcol in ("bus", "bus0", "bus1", "bus2", "bus3", "bus4"):
        if bcol in selector and bcol in base.columns:
            val = selector[bcol]
            if isinstance(val, str):
                val = [val]
            mask &= base[bcol].isin(list(val))

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
        if op == "set":
            df.loc[names, col] = value
        elif op == "scale":
            df.loc[names, col] = df.loc[names, col] * value
        elif op == "add":
            df.loc[names, col] = df.loc[names, col] + value
        else:
            raise ValueError(f"Unsupported op: {op}")
        return

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
    if isinstance(ts.columns, pd.MultiIndex):
        if "scenario" in ts.columns.names:
            scenarios_avail = set(ts.columns.get_level_values("scenario"))
        else:
            scenarios_avail = set(ts.columns.get_level_values(0))

        if scenario not in scenarios_avail:
            logger.warning(
                "Timeseries patch: scenario '%s' not found in ts.columns; skipping.",
                scenario,
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
            "Timeseries patch matched no existing columns for scenario '%s'; skipping.",
            scenario,
        )
        return

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
            v.columns = cols

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


def _apply_patch_config_if_present(
    n: pypsa.Network,
    stochastic_param: dict,
) -> None:
    """
    Apply legacy patch-based stochastic configuration if patches are present.

    This keeps backward compatibility with the previous patch-based interface.
    Patches are only applied in stochastic mode because their values are keyed by scenario.
    """
    stoch = _merge_stochastic_param(stochastic_param)
    enabled = bool(stoch.get("enable", stoch.get("enabled", False)))
    if not enabled:
        return

    patches = stoch.get("patches", [])
    if not patches:
        logger.info("No legacy stochastic patches provided.")
        return

    def base_comp_from_table(table: str) -> str:
        return table[:-2] if table.endswith("_t") else table

    for i, patch in enumerate(patches, start=1):
        patch = _ensure_dict(patch, f"patch[{i}]")
        target = patch.get("target")
        if not isinstance(target, str) or "." not in target:
            raise ValueError(
                f"patch[{i}].target must be like 'generators.marginal_cost' or 'loads_t.p_set'"
            )

        table, attr = target.split(".", 1)
        selector = _ensure_dict(patch.get("selector", {}), f"patch[{i}].selector")
        op = patch.get("op", "set")
        values = _ensure_dict(patch.get("values", {}), f"patch[{i}].values")

        comp_for_selection = base_comp_from_table(table)
        names = _select_names_from_component(n, comp_for_selection, selector)
        if not names:
            logger.warning("patch[%s] matched no components; skipping. target=%s", i, target)
            continue

        logger.info("Applying patch[%s] target=%s op=%s matched=%s", i, target, op, len(names))

        if table.endswith("_t"):
            ts_container = getattr(n, table)
            ts = getattr(ts_container, attr)

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
            comp_df = getattr(n, table)
            if attr not in comp_df.columns:
                raise KeyError(f"patch[{i}] column not found: {table}.{attr}")

            for sc, v in values.items():
                if not np.isscalar(v):
                    raise ValueError(
                        f"patch[{i}] static patch values must be scalar; got {type(v).__name__}"
                    )
                _apply_patch_static(comp_df, attr, sc, names, op, float(v))


def apply_stochastic_config(
    n: pypsa.Network,
    config: dict,
    stochastic_param: dict,
    wildcards: Mapping[str, Any] | None = None,
) -> None:
    """
    Apply structured stochastic/deterministic scenarios and optional legacy patches.

    Behavior
    --------
    - If stochastic mode is enabled:
      * read scenario names from stochastic config
      * call n.set_scenarios(...)
      * dispatch one structured builder per scenario name
      * optionally apply legacy patch-based modifications

    - If stochastic mode is disabled:
      * detect a single structured scenario name from config or run context
      * apply the corresponding builder in deterministic mode
    """
    is_stochastic, active_names = _apply_structured_scenarios(
        n=n,
        config=config,
        stochastic_param=stochastic_param,
        wildcards=wildcards,
    )

    if is_stochastic:
        _apply_patch_config_if_present(n=n, stochastic_param=stochastic_param)

    if active_names:
        logger.info("Applied structured scenario(s): %s", active_names)
    else:
        logger.info("No structured scenario was applied.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "stochasticify_sector_network",
            opts="",
            clusters="adm",
            configfiles="config/test_stoch/config.yaml",
            sector_opts="",
            planning_horizons="2050",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

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

    apply_stochastic_config(
        n,
        config=snakemake.config,
        stochastic_param=snakemake.params.get("stochastic_scenarios", {}),
        wildcards=dict(snakemake.wildcards),
    )

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

    logger.info("Exported stochastic pre-solve network to %s", snakemake.output.network)