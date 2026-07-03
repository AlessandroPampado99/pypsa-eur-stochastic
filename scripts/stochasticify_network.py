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
import sys
from pathlib import Path
from typing import Any, Mapping

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
        if "name" in idx.names:
            return idx.get_level_values("name")
        return idx.get_level_values(-1)
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
# Low-level helpers for loads/links
# ---------------------------

def _base_component_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a component table with scenario removed from the index if present."""
    base = df.copy()
    base.index = _get_level_names(base.index)
    return base


def _base_loads_table(n: pypsa.Network) -> pd.DataFrame:
    """Return loads table indexed only by load name."""
    return _base_component_table(n.loads)


def _base_links_table(n: pypsa.Network) -> pd.DataFrame:
    """Return links table indexed only by link name."""
    return _base_component_table(n.links)


def _ts_column_key(ts: pd.DataFrame, name: str, scenario: str | None = None):
    """
    Return the correct column key for a time series table.

    For deterministic tables the key is the plain component name.
    For stochastic tables the key is the tuple (scenario, name).
    """
    if not isinstance(ts.columns, pd.MultiIndex):
        if name not in ts.columns:
            raise KeyError(f"Column '{name}' not found in deterministic time series table.")
        return name

    if scenario is None:
        raise ValueError(
            f"Time series table is stochastic but no scenario was provided for column '{name}'."
        )

    if "scenario" in ts.columns.names:
        scenarios = ts.columns.get_level_values("scenario")
    else:
        scenarios = ts.columns.get_level_values(0)

    if "name" in ts.columns.names:
        names = ts.columns.get_level_values("name")
    else:
        names = ts.columns.get_level_values(1)

    mask = (scenarios == scenario) & (names == name)
    matches = ts.columns[mask]
    if len(matches) == 0:
        raise KeyError(f"Column for scenario='{scenario}', name='{name}' not found.")
    if len(matches) > 1:
        raise ValueError(f"Multiple columns found for scenario='{scenario}', name='{name}'.")
    return matches[0]


def _read_ts_series(ts: pd.DataFrame, name: str, scenario: str | None = None) -> pd.Series:
    """Return a copy of one time series column."""
    key = _ts_column_key(ts, name=name, scenario=scenario)
    return ts.loc[:, key].copy()


def _write_ts_series(
    ts: pd.DataFrame,
    name: str,
    values: pd.Series | np.ndarray,
    scenario: str | None = None,
) -> None:
    """Overwrite one time series column."""
    key = _ts_column_key(ts, name=name, scenario=scenario)

    if isinstance(values, pd.Series):
        v = values.reindex(ts.index)
        if v.isnull().any():
            raise ValueError(f"NaNs encountered while writing series '{name}'.")
        ts.loc[:, key] = v.values
    else:
        arr = np.asarray(values)
        if arr.ndim != 1 or len(arr) != len(ts.index):
            raise ValueError(
                f"Invalid shape for series '{name}': expected ({len(ts.index)},), got {arr.shape}."
            )
        ts.loc[:, key] = arr

def _scale_loads_by_carrier(
    n: pypsa.Network,
    carrier: str,
    factor: float,
    scenario: str | None = None,
) -> None:
    """Scale all loads belonging to a given carrier."""
    names = _find_load_names_by_carrier(n, carrier)
    if not names:
        raise KeyError(f"No loads found for carrier '{carrier}'.")

    for name in names:
        s = _read_load_series(n, name, scenario=scenario)
        _write_load_series(n, name, s * factor, scenario=scenario)

    logger.info(
        "Scaled %s load(s) with carrier '%s' by factor %.4f.",
        len(names),
        carrier,
        factor,
    )

def _read_link_efficiency_series(
    n: pypsa.Network,
    link_name: str,
    scenario: str | None = None,
) -> pd.Series:
    """Return one link efficiency time series."""
    return _read_ts_series(n.links_t.efficiency, name=link_name, scenario=scenario)


def _carrier_names_from_table(df: pd.DataFrame, carrier: str) -> list[str]:
    """Return component names matching a carrier from a base component table."""
    base = _base_component_table(df)
    if "carrier" not in base.columns:
        return []
    mask = base["carrier"].astype(str).eq(carrier)
    return base.index[mask].tolist()


def _find_load_names_by_carrier(n: pypsa.Network, carrier: str) -> list[str]:
    """Return load names whose carrier matches the requested value."""
    return _carrier_names_from_table(n.loads, carrier)


def _find_link_names_by_carrier(n: pypsa.Network, carrier: str) -> list[str]:
    """Return link names whose carrier matches the requested value."""
    return _carrier_names_from_table(n.links, carrier)


def _extract_prefix(name: str, suffix: str) -> str:
    """Strip a known suffix from a component name and return the prefix."""
    if not name.endswith(suffix):
        raise ValueError(f"Name '{name}' does not end with suffix '{suffix}'.")
    return name[: -len(suffix)]


def _assert_load_exists(n: pypsa.Network, load_name: str) -> None:
    """Raise if a load is missing."""
    loads = _base_loads_table(n)
    if load_name not in loads.index:
        raise KeyError(f"Required load '{load_name}' not found in n.loads.")


def _assert_link_exists(n: pypsa.Network, link_name: str) -> None:
    """Raise if a link is missing."""
    links = _base_links_table(n)
    if link_name not in links.index:
        raise KeyError(f"Required link '{link_name}' not found in n.links.")

def _load_row_key(n: pypsa.Network, load_name: str, scenario: str | None = None):
    """
    Return the correct row key for n.loads.

    For deterministic tables the key is the plain load name.
    For stochastic static tables the key is the tuple (scenario, name).
    """
    if not isinstance(n.loads.index, pd.MultiIndex):
        if load_name not in n.loads.index:
            raise KeyError(f"Load '{load_name}' not found in deterministic n.loads.")
        return load_name

    if scenario is None:
        raise ValueError(
            f"n.loads has a stochastic MultiIndex but no scenario was provided for '{load_name}'."
        )

    if "scenario" in n.loads.index.names:
        scenarios = n.loads.index.get_level_values("scenario")
    else:
        scenarios = n.loads.index.get_level_values(0)

    if "name" in n.loads.index.names:
        names = n.loads.index.get_level_values("name")
    else:
        names = n.loads.index.get_level_values(1)

    mask = (scenarios == scenario) & (names == load_name)
    matches = n.loads.index[mask]
    if len(matches) == 0:
        raise KeyError(f"Static load row for scenario='{scenario}', name='{load_name}' not found.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple static load rows found for scenario='{scenario}', name='{load_name}'."
        )
    return matches[0]


def _has_load_timeseries_column(
    n: pypsa.Network,
    load_name: str,
    scenario: str | None = None,
) -> bool:
    """Return True if the load exists in n.loads_t.p_set."""
    ts = n.loads_t.p_set
    if not isinstance(ts.columns, pd.MultiIndex):
        return load_name in ts.columns

    if scenario is None:
        return False

    if "scenario" in ts.columns.names:
        scenarios = ts.columns.get_level_values("scenario")
    else:
        scenarios = ts.columns.get_level_values(0)

    if "name" in ts.columns.names:
        names = ts.columns.get_level_values("name")
    else:
        names = ts.columns.get_level_values(1)

    return ((scenarios == scenario) & (names == load_name)).any()


def _read_load_series(n: pypsa.Network, load_name: str, scenario: str | None = None) -> pd.Series:
    """
    Return the effective load time series.

    Priority:
    1. n.loads_t.p_set column if present
    2. broadcast static n.loads.p_set over all snapshots
    """
    if _has_load_timeseries_column(n, load_name, scenario=scenario):
        return _read_ts_series(n.loads_t.p_set, name=load_name, scenario=scenario)

    row_key = _load_row_key(n, load_name, scenario=scenario)
    value = n.loads.loc[row_key, "p_set"]
    return pd.Series(float(value), index=n.snapshots)


def _write_static_load_value(
    n: pypsa.Network,
    load_name: str,
    value: float,
    scenario: str | None = None,
) -> None:
    """Write a scalar value into n.loads.p_set."""
    row_key = _load_row_key(n, load_name, scenario=scenario)
    n.loads.loc[row_key, "p_set"] = float(value)


def _ensure_load_timeseries_column(
    n: pypsa.Network,
    load_name: str,
    scenario: str | None = None,
) -> None:
    """
    Ensure that n.loads_t.p_set contains a column for the requested load.

    If missing, initialize it from the current effective series.
    """
    if _has_load_timeseries_column(n, load_name, scenario=scenario):
        return

    base_series = _read_load_series(n, load_name, scenario=scenario)

    ts = n.loads_t.p_set
    if not isinstance(ts.columns, pd.MultiIndex):
        ts.loc[:, load_name] = base_series.values
        ts.columns.name = "name"
        return

    if scenario is None:
        raise ValueError(
            f"Cannot create stochastic time series column for '{load_name}' without scenario."
        )

    new_col = pd.MultiIndex.from_tuples(
        [(scenario, load_name)],
        names=ts.columns.names,
    )
    new_df = pd.DataFrame(base_series.values, index=ts.index, columns=new_col)
    n.loads_t.p_set = pd.concat([ts, new_df], axis=1)


def _write_load_series(
    n: pypsa.Network,
    load_name: str,
    values: pd.Series | np.ndarray,
    scenario: str | None = None,
) -> None:
    """
    Write a load profile.

    - If the profile is constant and the load has no existing timeseries column, write to static n.loads.p_set
    - Otherwise create/use a column in n.loads_t.p_set
    """
    if not isinstance(values, pd.Series):
        arr = np.asarray(values)
        if arr.ndim != 1 or len(arr) != len(n.snapshots):
            raise ValueError(
                f"Invalid shape for series '{load_name}': expected ({len(n.snapshots)},), got {arr.shape}."
            )
        values = pd.Series(arr, index=n.snapshots)
    else:
        values = values.reindex(n.snapshots)
        if values.isnull().any():
            raise ValueError(f"NaNs encountered while writing series '{load_name}'.")

    is_constant = np.allclose(values.values, values.values[0])

    if is_constant and not _has_load_timeseries_column(n, load_name, scenario=scenario):
        _write_static_load_value(n, load_name, float(values.iloc[0]), scenario=scenario)
        return

    _ensure_load_timeseries_column(n, load_name, scenario=scenario)
    _write_ts_series(n.loads_t.p_set, name=load_name, values=values, scenario=scenario)



def _snapshot_weightings(n: pypsa.Network, column: str = "generators") -> pd.Series:
    """Return snapshot weights aligned to n.snapshots."""
    weights = n.snapshot_weightings
    if isinstance(weights, pd.DataFrame):
        if column not in weights.columns:
            raise KeyError(f"snapshot_weightings has no column '{column}'.")
        return weights[column].reindex(n.snapshots).astype(float)
    return pd.Series(weights, index=n.snapshots, dtype=float).reindex(n.snapshots)


def _series_energy_twh(
    n: pypsa.Network,
    series: pd.Series,
    weighting: str = "generators",
) -> float:
    """Compute annual energy in TWh from an MW series and snapshot weights in hours."""
    s = series.reindex(n.snapshots).astype(float)
    if s.isnull().any():
        raise ValueError("NaNs encountered while computing annual energy.")
    return float((s * _snapshot_weightings(n, weighting)).sum() / 1e6)


def _load_annual_energy_twh(
    n: pypsa.Network,
    load_name: str,
    scenario: str | None = None,
    weighting: str = "generators",
) -> float:
    """Compute annual energy in TWh for one load."""
    return _series_energy_twh(
        n,
        _read_load_series(n, load_name, scenario=scenario),
        weighting=weighting,
    )


def _load_has_positive_energy(
    n: pypsa.Network,
    load_name: str,
    scenario: str | None,
    weighting: str,
) -> bool:
    """Return whether an existing load has positive annual energy."""
    return _load_annual_energy_twh(
        n, load_name, scenario=scenario, weighting=weighting
    ) > 0.0


def _select_load_names(n: pypsa.Network, selector: Mapping[str, Any]) -> list[str]:
    """Select load names using the generic component selector."""
    return _select_names_from_component(n, "loads", selector)


def _target_load_for_source(
    source_name: str,
    source_carrier: str | None,
    target_selector: Mapping[str, Any] | None,
    target_carrier: str | None,
) -> str:
    """Infer a target load name from source/target carrier suffixes or explicit selector."""
    target_selector = target_selector or {}
    target_name = target_selector.get("name")
    if target_name:
        return str(target_name)
    names = target_selector.get("names")
    if isinstance(names, str) and not any(ch in names for ch in "^$.*+?[](){}|\\"):
        return names
    if isinstance(names, (list, tuple)) and len(names) == 1:
        return str(names[0])
    if source_carrier and target_carrier and source_name.endswith(source_carrier):
        return f"{_extract_prefix(source_name, source_carrier)}{target_carrier}"
    if target_carrier:
        return target_carrier
    raise ValueError(
        f"Cannot infer target load for source '{source_name}'. Provide target.name, "
        "a single target.names entry, or target_carrier."
    )


def _load_profile_for_energy(
    n: pypsa.Network,
    load_name: str,
    delta_twh: float,
    scenario: str | None,
    weighting: str,
    fallback: pd.Series | None = None,
    allow_flat_fallback: bool = False,
) -> pd.Series:
    """Build an additive/reduction profile with requested annual energy."""
    if delta_twh <= 0:
        return pd.Series(0.0, index=n.snapshots)

    series = _read_load_series(n, load_name, scenario=scenario)
    energy = _series_energy_twh(n, series.clip(lower=0.0), weighting=weighting)
    if energy > 0:
        return series.clip(lower=0.0) * (delta_twh / energy)

    if fallback is not None:
        fallback = fallback.reindex(n.snapshots).astype(float).clip(lower=0.0)
        fallback_energy = _series_energy_twh(n, fallback, weighting=weighting)
        if fallback_energy > 0:
            return fallback * (delta_twh / fallback_energy)

    if allow_flat_fallback:
        weights = _snapshot_weightings(n, weighting)
        hours = float(weights.sum())
        if hours <= 0:
            raise ValueError("Cannot build flat fallback profile with non-positive weights.")
        return pd.Series(delta_twh * 1e6 / hours, index=n.snapshots)

    raise ValueError(
        f"Load '{load_name}' has zero annual energy; configure an explicit fallback profile."
    )


def _reduce_load_energy(
    n: pypsa.Network,
    load_name: str,
    delta_twh: float,
    scenario: str | None,
    weighting: str,
    non_negative_tolerance: float,
) -> pd.Series:
    """Reduce a load proportionally by delta_twh and return the MW reduction profile."""
    current = _read_load_series(n, load_name, scenario=scenario)
    energy = _series_energy_twh(n, current.clip(lower=0.0), weighting=weighting)
    if energy <= 0:
        return pd.Series(0.0, index=n.snapshots)
    actual_delta = min(delta_twh, energy)
    reduction = current.clip(lower=0.0) * (actual_delta / energy)
    updated = current - reduction
    if updated.min() < -non_negative_tolerance:
        raise ValueError(f"Reduction would make load '{load_name}' negative.")
    _write_load_series(n, load_name, updated.clip(lower=0.0), scenario=scenario)
    return reduction


def _increase_load_by_profile(
    n: pypsa.Network,
    load_name: str,
    addition: pd.Series,
    scenario: str | None,
) -> None:
    """Increase an existing load by an additive MW profile."""
    _assert_load_exists(n, load_name)
    current = _read_load_series(n, load_name, scenario=scenario)
    _write_load_series(n, load_name, current + addition.reindex(n.snapshots), scenario=scenario)


def _component_value(base: pd.DataFrame, name: str, column: str) -> Any:
    """Return a scalar component value from a base table that may have duplicate names."""
    value = base.loc[name, column]
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


def _copy_missing_load_like_source(
    n: pypsa.Network,
    source_name: str,
    target_name: str,
    target_carrier: str | None,
) -> None:
    """Create a target load by copying static metadata from a source load."""
    loads = _base_loads_table(n)
    if target_name in loads.index:
        return
    source = loads.loc[source_name].copy()
    source["carrier"] = target_carrier or source.get("carrier", target_name)
    source["p_set"] = 0.0
    n.add("Load", target_name, **source.to_dict())

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


def _normalize_component_table_name(component: str) -> str:
    """
    Normalize a user-facing component name to the corresponding network table name.

    Supported examples:
    - Generator -> generators
    - generators -> generators
    - Link -> links
    - Load -> loads
    """
    mapping = {
        "bus": "buses",
        "buses": "buses",
        "carrier": "carriers",
        "carriers": "carriers",
        "generator": "generators",
        "generators": "generators",
        "load": "loads",
        "loads": "loads",
        "line": "lines",
        "lines": "lines",
        "link": "links",
        "links": "links",
        "store": "stores",
        "stores": "stores",
        "storageunit": "storage_units",
        "storageunits": "storage_units",
        "storage_unit": "storage_units",
        "storage_units": "storage_units",
        "transformer": "transformers",
        "transformers": "transformers",
        "shuntimpedance": "shunt_impedances",
        "shuntimpedances": "shunt_impedances",
        "shunt_impedance": "shunt_impedances",
        "shunt_impedances": "shunt_impedances",
    }

    key = str(component).strip().lower().replace(" ", "").replace("-", "").replace(".", "")
    if key not in mapping:
        raise ValueError(f"Unsupported component '{component}'.")
    return mapping[key]


def _get_component_attr_tables(
    n: pypsa.Network,
    component: str,
    attribute: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame | None]:
    """
    Return static and time-series tables for a component attribute.

    Returns
    -------
    table_name : str
        Base component table name, e.g. 'generators'
    static_df : pd.DataFrame
        Static component table, e.g. n.generators
    ts_df : pd.DataFrame | None
        Time-series attribute table, e.g. n.generators_t.p_max_pu, if it exists
    """
    table_name = _normalize_component_table_name(component)
    static_df = getattr(n, table_name)

    ts_df = None
    ts_container_name = f"{table_name}_t"
    if hasattr(n, ts_container_name):
        ts_container = getattr(n, ts_container_name)
        if hasattr(ts_container, attribute):
            ts_df = getattr(ts_container, attribute)

    return table_name, static_df, ts_df


def _available_timeseries_names(
    ts: pd.DataFrame,
    scenario: str | None = None,
) -> set[str]:
    """
    Return component names available in a time-series table.

    If scenario is provided and ts is stochastic, names are filtered to that scenario.
    """
    if not isinstance(ts.columns, pd.MultiIndex):
        return set(ts.columns.astype(str))

    if scenario is not None:
        if "scenario" in ts.columns.names:
            scenarios = ts.columns.get_level_values("scenario")
        else:
            scenarios = ts.columns.get_level_values(0)

        if "name" in ts.columns.names:
            names = ts.columns.get_level_values("name")
        else:
            names = ts.columns.get_level_values(1)

        return set(pd.Index(names[scenarios == scenario]).astype(str))

    if "name" in ts.columns.names:
        return set(ts.columns.get_level_values("name").astype(str))
    return set(ts.columns.get_level_values(-1).astype(str))


def _split_names_by_target(
    names: list[str],
    ts: pd.DataFrame | None,
    scenario: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split matched component names into time-series-backed and static-only names.
    """
    if ts is None:
        return [], list(names)

    ts_names_avail = _available_timeseries_names(ts, scenario=scenario)
    names_ts = [name for name in names if str(name) in ts_names_avail]
    names_static = [name for name in names if str(name) not in ts_names_avail]
    return names_ts, names_static


def _validate_modify_rule(rule: Mapping[str, Any]) -> None:
    """Validate one generic modification rule."""
    required = {"component", "attribute", "operation", "value"}
    missing = sorted(required - set(rule))
    if missing:
        raise ValueError(f"Missing required keys in rule: {missing}")

    op = str(rule["operation"]).strip().lower()
    if op not in {"set", "scale", "add"}:
        raise ValueError(f"Unsupported operation '{op}'. Allowed: set, scale, add")

    target = str(rule.get("target", "auto")).strip().lower()
    if target not in {"auto", "static", "timeseries"}:
        raise ValueError(
            f"Unsupported target '{target}'. Allowed: auto, static, timeseries"
        )


def _apply_modify_components_rule(
    n: pypsa.Network,
    rule: Mapping[str, Any],
    scenario: str | None = None,
) -> None:
    """
    Apply one generic component modification rule.

    Rule format
    -----------
    {
      "component": "Generator",
      "attribute": "marginal_cost",
      "target": "auto" | "static" | "timeseries",
      "carrier": ["OCGT", "CCGT"],
      "operation": "scale" | "set" | "add",
      "value": 1.15
    }
    """
    rule = _ensure_dict(rule, "rule")
    _validate_modify_rule(rule)

    component = rule["component"]
    attribute = str(rule["attribute"])
    operation = str(rule["operation"]).strip().lower()
    target = str(rule.get("target", "auto")).strip().lower()
    value = rule["value"]

    table_name, static_df, ts_df = _get_component_attr_tables(
        n=n,
        component=component,
        attribute=attribute,
    )

    selector = {
        k: v
        for k, v in rule.items()
        if k not in {"component", "attribute", "target", "operation", "value"}
    }

    names = _select_names_from_component(n, table_name, selector)
    if not names:
        logger.warning(
            "modify_components rule matched no components. component=%s attribute=%s selector=%s",
            component,
            attribute,
            selector,
        )
        return

    if target == "timeseries" and ts_df is None:
        raise ValueError(
            f"Rule requested timeseries target for {component}.{attribute}, "
            f"but no time-series table exists."
        )

    if attribute == "p_max_pu" and target in {"timeseries", "auto"} and ts_df is not None:
        if operation != "scale":
            # Only forbid when the rule actually hits the timeseries branch
            names_ts, _ = _split_names_by_target(names, ts_df, scenario=scenario)
            if names_ts:
                raise ValueError(
                    "For timeseries p_max_pu only 'scale' is allowed."
                )

    if target == "static":
        if attribute not in static_df.columns:
            raise KeyError(f"Column not found in static table: {table_name}.{attribute}")
        _apply_patch_static(static_df, attribute, scenario, names, operation, value)
        logger.info(
            "Applied static modify_components rule on %s.%s to %s component(s)%s.",
            table_name,
            attribute,
            len(names),
            f" for stochastic scenario '{scenario}'" if scenario is not None else " in deterministic mode",
        )
        return

    if target == "timeseries":
        _apply_patch_timeseries(ts_df, scenario, names, operation, value)
        logger.info(
            "Applied timeseries modify_components rule on %s_t.%s to %s component(s)%s.",
            table_name,
            attribute,
            len(names),
            f" for stochastic scenario '{scenario}'" if scenario is not None else " in deterministic mode",
        )
        return

    # target == "auto"
    if ts_df is None:
        if attribute not in static_df.columns:
            raise KeyError(f"Column not found in static table: {table_name}.{attribute}")
        _apply_patch_static(static_df, attribute, scenario, names, operation, value)
        logger.info(
            "Applied auto->static modify_components rule on %s.%s to %s component(s)%s.",
            table_name,
            attribute,
            len(names),
            f" for stochastic scenario '{scenario}'" if scenario is not None else " in deterministic mode",
        )
        return

    names_ts, names_static = _split_names_by_target(names, ts_df, scenario=scenario)

    if names_ts:
        _apply_patch_timeseries(ts_df, scenario, names_ts, operation, value)

    if names_static:
        if attribute not in static_df.columns:
            raise KeyError(f"Column not found in static table: {table_name}.{attribute}")
        _apply_patch_static(static_df, attribute, scenario, names_static, operation, value)

    logger.info(
        "Applied auto modify_components rule on %s.%s: %s timeseries-backed + %s static-only component(s)%s.",
        table_name,
        attribute,
        len(names_ts),
        len(names_static),
        f" for stochastic scenario '{scenario}'" if scenario is not None else " in deterministic mode",
    )


def _scenario_modify_components(
    n: pypsa.Network,
    scenario: str | None = None,
    config: dict | None = None,
) -> None:
    """
    Generic structured scenario applying one or more component modification rules.

    Expected config format
    ----------------------
    {
      "rules": [
        {
          "component": "Generator",
          "attribute": "marginal_cost",
          "target": "auto",
          "carrier": ["OCGT", "CCGT"],
          "operation": "scale",
          "value": 1.15
        }
      ]
    }
    """
    cfg = _ensure_dict(config, "config") if config is not None else {}
    rules = cfg.get("rules", None)
    if not isinstance(rules, list) or not rules:
        raise ValueError(
            "modify_components requires config['rules'] as a non-empty list."
        )

    for i, rule in enumerate(rules, start=1):
        logger.info(
            "Applying modify_components rule %s/%s%s.",
            i,
            len(rules),
            f" for stochastic scenario '{scenario}'" if scenario is not None else "",
        )
        _apply_modify_components_rule(n=n, rule=rule, scenario=scenario)





def _normalize_load_selector(value: Any, name: str) -> dict:
    """Normalize concise load selectors into the generic selector format."""
    if value is None:
        return {}
    if isinstance(value, str):
        return {"carrier": value}
    if isinstance(value, (list, tuple, set)):
        return {"carrier": list(value)}
    selector = dict(_ensure_dict(value, name))
    selector.pop("component", None)
    if "name" in selector and "names" not in selector:
        selector["names"] = selector.pop("name")
    return selector


def _normalize_transition_entry(entry: Mapping[str, Any], entry_name: str) -> dict:
    """
    Normalize concise demand-transition entries to the internal explicit shape.

    Accepted concise form example:
        source: land transport oil
        target: land transport EV
        type: electrify_transport
        cap: 0.2
        source_efficiency: 16.0712
        target_efficiency: 53.19
    """
    raw = dict(_ensure_dict(entry, f"entries.{entry_name}"))
    normalized = dict(raw)

    if "source_carrier" in normalized and "source" not in normalized:
        normalized["source"] = normalized.pop("source_carrier")
    if "target_carrier" in normalized and "target" not in normalized:
        normalized["target"] = normalized["target_carrier"]
    if "cap" in normalized and "cap_fraction" not in normalized:
        normalized["cap_fraction"] = normalized.pop("cap")

    normalized["source"] = _normalize_load_selector(
        normalized.get("source", normalized.get("target")),
        f"entries.{entry_name}.source",
    )
    if "target" in normalized:
        normalized["target"] = _normalize_load_selector(
            normalized.get("target"),
            f"entries.{entry_name}.target",
        )

    transformation = dict(normalized.get("transformation") or {})
    transform_keys = {
        "type",
        "target_carrier",
        "efficiency_factor",
        "source_efficiency",
        "target_efficiency",
        "max_source_reduction_fraction",
        "max_target_reduction_fraction",
        "strict_target_available",
        "create_missing_load",
        "copy_source_bus",
        "cop",
    }
    for key in transform_keys:
        if key in raw and key not in transformation:
            transformation[key] = raw[key]
    if "target_carrier" not in transformation and normalized.get("target"):
        target_carrier = normalized["target"].get("carrier")
        if isinstance(target_carrier, str):
            transformation["target_carrier"] = target_carrier

    if "type" not in transformation:
        raise ValueError(f"entries.{entry_name} must define transformation.type or concise type.")
    normalized["transformation"] = transformation

    for key in transform_keys - {"target_carrier"}:
        if key in normalized and key not in {"cap_fraction", "cap_twh"}:
            normalized.pop(key, None)
    return normalized


def _scenario_definitions(catalogue: Mapping[str, Any]) -> dict:
    """Return scenario definitions, accepting 'definitions' as a concise alias."""
    return _ensure_dict(
        catalogue.get("scenario_definitions", catalogue.get("definitions", {})),
        "scenario_definitions",
    )


def _normalize_action(action: Any, scenario_name: str, index: int) -> dict:
    """Normalize concise scenario actions to {'type': ..., ...}."""
    if isinstance(action, str):
        return {"type": action}
    action = dict(_ensure_dict(action, f"scenario_definitions.{scenario_name}.actions[{index}]"))
    if "type" in action:
        return action
    if len(action) == 1:
        action_type, payload = next(iter(action.items()))
        if payload is None:
            return {"type": action_type}
        if isinstance(payload, list) and action_type == "modify_components":
            return {"type": action_type, "rules": payload}
        if isinstance(payload, dict):
            return {"type": action_type, **payload}
    raise ValueError(
        f"scenario_definitions.{scenario_name}.actions[{index}] must define type "
        "or use a single-key concise action."
    )


def _normalize_scenario_definition(definition: Any, scenario_name: str) -> dict:
    """Normalize concise scenario definitions to {'actions': [...]}."""
    if definition is None:
        raise ValueError(f"scenario_definitions.{scenario_name} cannot be null.")
    if isinstance(definition, str):
        return {"actions": [{"type": definition}]}
    if isinstance(definition, list):
        return {"actions": definition}
    definition = dict(_ensure_dict(definition, f"scenario_definitions.{scenario_name}"))
    if "actions" in definition:
        return definition
    if "demand_transition" in definition or "modify_components" in definition or "base" in definition:
        return {"actions": [definition]}
    raise ValueError(f"scenario_definitions.{scenario_name}.actions must be provided.")

def _entry_cap_twh(entry: Mapping[str, Any], source_energy_twh: float) -> float:
    """Return the methodological cap for one demand-transition entry."""
    cap_fraction = float(entry.get("cap_fraction", 1.0))
    cap = cap_fraction * source_energy_twh
    if entry.get("cap_twh") is not None:
        cap = min(cap, float(entry["cap_twh"]))
    return max(0.0, cap)


def _source_selector(entry: Mapping[str, Any]) -> dict:
    return _normalize_load_selector(entry.get("source", entry.get("target", {})), "entry.source")


def _target_selector(entry: Mapping[str, Any], transformation: Mapping[str, Any]) -> dict:
    selector = _normalize_load_selector(entry.get("target", {}), "entry.target")
    if "carrier" not in selector and transformation.get("target_carrier") is not None:
        selector["carrier"] = transformation["target_carrier"]
    return selector


def _ensure_target_load(
    n: pypsa.Network,
    source_name: str,
    target_name: str,
    target_carrier: str | None,
    transformation: Mapping[str, Any],
) -> None:
    if target_name in _base_loads_table(n).index:
        return
    if not transformation.get("create_missing_load", False):
        raise KeyError(
            f"Target load '{target_name}' is missing. Set create_missing_load: true "
            "and use source-like bus copying or provide an existing target."
        )
    if not transformation.get("copy_source_bus", True):
        raise ValueError(
            f"Cannot create missing load '{target_name}' without source-like bus copying."
        )
    _copy_missing_load_like_source(n, source_name, target_name, target_carrier)


def _apply_shift_like_entry(
    n: pypsa.Network,
    entry: Mapping[str, Any],
    transformation: Mapping[str, Any],
    delta_twh: float,
    scenario: str | None,
    settings: Mapping[str, Any],
    mode: str,
) -> dict:
    weighting = str(settings.get("weighting", "generators"))
    nonneg = float(settings.get("non_negative_tolerance", 1e-8))
    source_names = _select_load_names(n, _source_selector(entry))
    if not source_names:
        raise KeyError(f"Demand-transition entry matched no source loads: {entry}")

    source_energies = {
        name: _load_annual_energy_twh(n, name, scenario=scenario, weighting=weighting)
        for name in source_names
    }
    total_source = sum(source_energies.values())
    if total_source <= 0 or delta_twh <= 0:
        return {"applied_delta_twh": 0.0, "transformed_target_twh": 0.0, "affected_loads": []}

    loads = _base_loads_table(n)
    transformed = 0.0
    affected = []
    for source_name, energy in source_energies.items():
        if energy <= 0:
            continue
        share_delta = delta_twh * energy / total_source
        source_carrier = str(_component_value(loads, source_name, "carrier"))
        target_carrier = transformation.get("target_carrier")
        target_name = _target_load_for_source(
            source_name,
            source_carrier,
            _target_selector(entry, transformation),
            target_carrier,
        )
        _ensure_target_load(n, source_name, target_name, target_carrier, transformation)
        if not _load_has_positive_energy(n, target_name, scenario, weighting):
            logger.warning(
                "Target load '%s' has zero annual energy; leaving it and source '%s' unchanged.",
                target_name,
                source_name,
            )
            continue

        reduction = _reduce_load_energy(
            n, source_name, share_delta, scenario, weighting, nonneg
        )
        actual_source_delta = _series_energy_twh(n, reduction, weighting=weighting)

        if mode in {"shift", "split"}:
            factor = float(transformation.get("efficiency_factor", 1.0))
            addition = reduction * factor
        elif mode == "electrify_transport":
            factor = float(transformation["source_efficiency"]) / float(
                transformation["target_efficiency"]
            )
            addition = reduction * factor
        else:
            raise ValueError(f"Unsupported shift-like mode '{mode}'.")

        _increase_load_by_profile(n, target_name, addition, scenario=scenario)
        target_delta = _series_energy_twh(n, addition, weighting=weighting)
        transformed += target_delta
        affected.append(
            {
                "source": source_name,
                "target": target_name,
                "source_delta_twh": actual_source_delta,
                "target_delta_twh": target_delta,
            }
        )

    return {
        "applied_delta_twh": sum(x["source_delta_twh"] for x in affected),
        "transformed_target_twh": transformed,
        "affected_loads": affected,
    }


def _apply_reverse_shift_entry(
    n: pypsa.Network,
    entry: Mapping[str, Any],
    transformation: Mapping[str, Any],
    delta_twh: float,
    scenario: str | None,
    settings: Mapping[str, Any],
) -> dict:
    weighting = str(settings.get("weighting", "generators"))
    nonneg = float(settings.get("non_negative_tolerance", 1e-8))
    source_eff = float(transformation["source_efficiency"])
    target_eff = float(transformation["target_efficiency"])
    max_target_fraction = float(transformation.get("max_target_reduction_fraction", 1.0))
    source_names = _select_load_names(n, _source_selector(entry))
    if not source_names:
        raise KeyError(f"Demand-transition entry matched no source loads: {entry}")

    loads = _base_loads_table(n)
    target_by_source = {}
    source_energies = {}
    dynamic_caps = {}
    for source_name in source_names:
        source_carrier = str(_component_value(loads, source_name, "carrier"))
        target_carrier = (_target_selector(entry, transformation) or {}).get("carrier")
        target_name = _target_load_for_source(
            source_name,
            source_carrier,
            _target_selector(entry, transformation),
            target_carrier,
        )
        if target_name not in loads.index:
            if transformation.get("strict_target_available", False):
                raise KeyError(f"Target load '{target_name}' for reverse_shift is missing.")
            continue
        source_energy = max(_load_annual_energy_twh(n, source_name, scenario, weighting), 0.0)
        if source_energy <= 0.0:
            logger.warning(
                "Source load '%s' has zero annual energy; leaving it unchanged for reverse_shift.",
                source_name,
            )
            continue
        target_energy = _load_annual_energy_twh(n, target_name, scenario, weighting)
        dynamic_caps[source_name] = max_target_fraction * target_energy * target_eff / source_eff
        target_by_source[source_name] = target_name
        source_energies[source_name] = source_energy

    dynamic_cap = sum(dynamic_caps.values())
    applied = min(delta_twh, dynamic_cap)
    if applied <= 0:
        return {
            "applied_delta_twh": 0.0,
            "transformed_target_twh": 0.0,
            "dynamic_feasibility_cap_twh": dynamic_cap,
            "affected_loads": [],
        }

    weights = source_energies if sum(source_energies.values()) > 0 else dynamic_caps
    total_weight = sum(weights.values())
    affected = []
    transformed = 0.0
    for source_name, target_name in target_by_source.items():
        if total_weight <= 0:
            continue
        share = applied * weights[source_name] / total_weight
        share = min(share, dynamic_caps[source_name])
        target_reduction_twh = share * source_eff / target_eff
        target_reduction = _reduce_load_energy(
            n, target_name, target_reduction_twh, scenario, weighting, nonneg
        )
        actual_target_reduction = _series_energy_twh(
            n, target_reduction, weighting=weighting
        )
        actual_source_increase = actual_target_reduction * target_eff / source_eff
        source_add = _load_profile_for_energy(
            n,
            source_name,
            actual_source_increase,
            scenario,
            weighting,
            fallback=target_reduction * target_eff / source_eff,
            allow_flat_fallback=bool(transformation.get("allow_flat_fallback", False)),
        )
        _increase_load_by_profile(n, source_name, source_add, scenario=scenario)
        transformed += actual_target_reduction
        affected.append(
            {
                "source": source_name,
                "target": target_name,
                "source_delta_twh": actual_source_increase,
                "target_reduction_twh": actual_target_reduction,
            }
        )

    return {
        "applied_delta_twh": sum(x["source_delta_twh"] for x in affected),
        "transformed_target_twh": transformed,
        "dynamic_feasibility_cap_twh": dynamic_cap,
        "affected_loads": affected,
    }


def _apply_add_entry(
    n: pypsa.Network,
    entry: Mapping[str, Any],
    transformation: Mapping[str, Any],
    delta_twh: float,
    scenario: str | None,
    settings: Mapping[str, Any],
) -> dict:
    weighting = str(settings.get("weighting", "generators"))
    selector = _target_selector(entry, transformation) or _source_selector(entry)
    target_names = _select_load_names(n, selector)
    if not target_names:
        raise KeyError(f"add transformation matched no target loads: {entry}")
    energies = {
        name: max(_load_annual_energy_twh(n, name, scenario, weighting), 0.0)
        for name in target_names
    }
    positive_names = [name for name in target_names if energies[name] > 0.0]
    total = sum(energies[name] for name in positive_names)
    if total <= 0:
        logger.warning(
            "add transformation matched target loads but all have zero annual energy; "
            "leaving them unchanged. entry=%s",
            entry,
        )
        return {
            "applied_delta_twh": 0.0,
            "transformed_target_twh": 0.0,
            "affected_loads": [],
            "warnings": ["all matched add targets have zero annual energy"],
        }

    affected = []
    for name in positive_names:
        share = delta_twh * energies[name] / total
        addition = _load_profile_for_energy(
            n,
            name,
            share,
            scenario,
            weighting,
            allow_flat_fallback=False,
        )
        _increase_load_by_profile(n, name, addition, scenario=scenario)
        affected.append({"target": name, "target_delta_twh": share})
    return {"applied_delta_twh": delta_twh, "transformed_target_twh": delta_twh, "affected_loads": affected}


def _apply_electrify_heat_entry(
    n: pypsa.Network,
    entry: Mapping[str, Any],
    transformation: Mapping[str, Any],
    delta_twh: float,
    scenario: str | None,
    settings: Mapping[str, Any],
) -> dict:
    weighting = str(settings.get("weighting", "generators"))
    nonneg = float(settings.get("non_negative_tolerance", 1e-8))
    source_names = _select_load_names(n, _source_selector(entry))
    if not source_names:
        raise KeyError(f"Demand-transition entry matched no heat source loads: {entry}")
    loads = _base_loads_table(n)
    energies = {name: _load_annual_energy_twh(n, name, scenario, weighting) for name in source_names}
    total = sum(energies.values())
    cop_map = _ensure_dict(transformation.get("cop", {}), "transformation.cop")
    affected = []
    transformed = 0.0
    for source_name, energy in energies.items():
        if energy <= 0 or total <= 0:
            continue
        source_carrier = str(_component_value(loads, source_name, "carrier"))
        hp_suffix = cop_map.get(source_carrier)
        if hp_suffix is None:
            raise KeyError(f"No COP heat-pump carrier mapping configured for '{source_carrier}'.")
        prefix = _extract_prefix(source_name, source_carrier)
        hp_name = f"{prefix}{hp_suffix}"
        _assert_link_exists(n, hp_name)
        cop = _read_link_efficiency_series(n, hp_name, scenario=scenario)
        if cop.isnull().any() or (cop <= 0).any():
            raise ValueError(f"COP for link '{hp_name}' must be finite and strictly positive.")
        share = delta_twh * energy / total
        target_name = _target_load_for_source(
            source_name,
            source_carrier,
            _target_selector(entry, transformation),
            transformation.get("target_carrier"),
        )
        _ensure_target_load(n, source_name, target_name, transformation.get("target_carrier"), transformation)
        if not _load_has_positive_energy(n, target_name, scenario, weighting):
            logger.warning(
                "Target load '%s' has zero annual energy; leaving it and source '%s' unchanged.",
                target_name,
                source_name,
            )
            continue
        reduction = _reduce_load_energy(n, source_name, share, scenario, weighting, nonneg)
        addition = reduction / cop.reindex(n.snapshots)
        _increase_load_by_profile(n, target_name, addition, scenario=scenario)
        target_delta = _series_energy_twh(n, addition, weighting=weighting)
        transformed += target_delta
        affected.append(
            {
                "source": source_name,
                "target": target_name,
                "heat_pump": hp_name,
                "source_delta_twh": _series_energy_twh(n, reduction, weighting),
                "target_delta_twh": target_delta,
            }
        )
    return {
        "applied_delta_twh": sum(x["source_delta_twh"] for x in affected),
        "transformed_target_twh": transformed,
        "affected_loads": affected,
    }


def _apply_transition_entry(
    n: pypsa.Network,
    entry_name: str,
    entry: Mapping[str, Any],
    requested_twh: float,
    scenario: str | None,
    settings: Mapping[str, Any],
) -> dict:
    weighting = str(settings.get("weighting", "generators"))
    transformation = _ensure_dict(entry.get("transformation"), f"entries.{entry_name}.transformation")
    mode = str(transformation.get("type", "")).strip()
    source_names = _select_load_names(n, _source_selector(entry))
    source_energy = sum(
        _load_annual_energy_twh(n, name, scenario=scenario, weighting=weighting)
        for name in source_names
    )
    if mode == "add" and "source" not in entry:
        source_energy = requested_twh
    methodological_cap = _entry_cap_twh(entry, source_energy)
    dynamic_cap = methodological_cap
    if mode == "reverse_shift":
        source_eff = float(transformation["source_efficiency"])
        target_eff = float(transformation["target_efficiency"])
        max_frac = float(transformation.get("max_target_reduction_fraction", 1.0))
        loads = _base_loads_table(n)
        target_selector = _target_selector(entry, transformation)
        caps = []
        for source_name in source_names:
            source_carrier = str(_component_value(loads, source_name, "carrier"))
            target_name = _target_load_for_source(
                source_name, source_carrier, target_selector, target_selector.get("carrier")
            )
            if target_name not in loads.index:
                if transformation.get("strict_target_available", False):
                    raise KeyError(f"Target load '{target_name}' for reverse_shift is missing.")
                continue
            if _load_annual_energy_twh(n, source_name, scenario, weighting) <= 0.0:
                continue
            target_energy = _load_annual_energy_twh(n, target_name, scenario, weighting)
            caps.append(max_frac * target_energy * target_eff / source_eff)
        dynamic_cap = sum(caps)

    delta = min(requested_twh, methodological_cap, dynamic_cap)
    if mode in {"shift", "split", "electrify_transport"}:
        result = _apply_shift_like_entry(n, entry, transformation, delta, scenario, settings, mode)
    elif mode == "reverse_shift":
        result = _apply_reverse_shift_entry(n, entry, transformation, delta, scenario, settings)
    elif mode == "electrify_heat":
        result = _apply_electrify_heat_entry(n, entry, transformation, delta, scenario, settings)
    elif mode == "add":
        result = _apply_add_entry(n, entry, transformation, delta, scenario, settings)
    else:
        raise ValueError(f"Unsupported demand_transition transformation type '{mode}'.")

    return {
        "entry": entry_name,
        "source_energy_before_twh": source_energy,
        "methodological_cap_twh": methodological_cap,
        "dynamic_feasibility_cap_twh": result.get("dynamic_feasibility_cap_twh", dynamic_cap),
        "requested_twh": requested_twh,
        **result,
        "remaining_target_twh": max(0.0, requested_twh - result.get("applied_delta_twh", 0.0)),
    }


def _validate_loads_after_transition(
    n: pypsa.Network,
    scenario: str | None,
    settings: Mapping[str, Any],
) -> None:
    tol = float(settings.get("non_negative_tolerance", 1e-8))
    names = _base_loads_table(n).index.unique().tolist()
    for name in names:
        series = _read_load_series(n, name, scenario=scenario)
        values = series.to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"NaNs introduced in load '{name}'.")
        if np.isinf(values).any():
            raise ValueError(f"Infinite values introduced in load '{name}'.")
        if values.min() < -tol:
            raise ValueError(f"Negative load values below tolerance in '{name}'.")


def _action_base(
    n: pypsa.Network,
    action: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    scenario: str | None = None,
) -> dict:
    del n, action, catalogue
    logger.info("Applied base action%s.", f" for stochastic scenario '{scenario}'" if scenario else "")
    return {"type": "base"}


def _action_modify_components(
    n: pypsa.Network,
    action: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    scenario: str | None = None,
) -> dict:
    del catalogue
    _scenario_modify_components(n=n, scenario=scenario, config=dict(action))
    return {"type": "modify_components", "rules": len(action.get("rules", []))}


def _action_demand_transition(
    n: pypsa.Network,
    action: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    scenario: str | None = None,
) -> dict:
    settings = _ensure_dict(catalogue.get("settings", {}), "settings")
    families = _ensure_dict(catalogue.get("families", {}), "families")
    family_name = action.get("family")
    if family_name not in families:
        raise KeyError(f"Unknown demand-transition family '{family_name}'.")
    family = _ensure_dict(families[family_name], f"families.{family_name}")
    target_spec = action.get("target")
    if isinstance(target_spec, str):
        targets = _ensure_dict(family.get("targets", {}), f"families.{family_name}.targets")
        if target_spec not in targets:
            raise KeyError(f"Unknown target '{target_spec}' for family '{family_name}'.")
        requested = float(targets[target_spec])
    else:
        requested = float(target_spec)

    entries = _ensure_dict(family.get("entries", {}), f"families.{family_name}.entries")
    priority = action.get("priority") or list(entries)
    remaining = requested
    entry_reports = []
    for entry_name in priority:
        if entry_name not in entries:
            raise KeyError(f"Unknown demand-transition entry '{entry_name}' in family '{family_name}'.")
        if remaining <= float(settings.get("tolerance_twh", 1e-4)):
            break
        report = _apply_transition_entry(
            n=n,
            entry_name=entry_name,
            entry=_normalize_transition_entry(entries[entry_name], entry_name),
            requested_twh=remaining,
            scenario=scenario,
            settings=settings,
        )
        remaining = max(0.0, remaining - report.get("applied_delta_twh", 0.0))
        report["remaining_target_twh"] = remaining
        entry_reports.append(report)

    tolerance = float(settings.get("tolerance_twh", 1e-4))
    allow_unmet = bool(settings.get("allow_unmet_target", False))
    applied = requested - remaining
    report = {
        "scenario": scenario or "deterministic",
        "family": family_name,
        "requested_target_twh": requested,
        "applied_total_twh": applied,
        "unmet_target_twh": remaining,
        "entries": entry_reports,
    }
    if remaining > tolerance and not allow_unmet:
        raise ValueError(
            f"Demand-transition target for family '{family_name}' is unmet by {remaining:.6g} TWh. "
            f"Allocation report: {report}"
        )
    if remaining > tolerance:
        logger.warning(
            "Demand-transition family '%s' left %.6g TWh unmet%s.",
            family_name,
            remaining,
            f" in scenario '{scenario}'" if scenario else "",
        )
    _validate_loads_after_transition(n, scenario=scenario, settings=settings)
    n.meta.setdefault("demand_transition_reports", {})[scenario or "deterministic"] = report
    logger.info(
        "Demand-transition %s/%s applied %.6g of %.6g TWh%s.",
        family_name,
        target_spec,
        applied,
        requested,
        f" for stochastic scenario '{scenario}'" if scenario else "",
    )
    return {"type": "demand_transition", **report}


ACTION_HANDLERS = {
    "base": _action_base,
    "modify_components": _action_modify_components,
    "demand_transition": _action_demand_transition,
}


def _apply_action_catalogue_scenario(
    n: pypsa.Network,
    catalogue: Mapping[str, Any],
    scenario_name: str,
    scenario: str | None = None,
) -> list[dict]:
    definitions = _scenario_definitions(catalogue)
    if scenario_name not in definitions:
        raise KeyError(
            f"Scenario '{scenario_name}' has a probability/active_scenario but no scenario_definitions entry."
        )
    definition = _normalize_scenario_definition(definitions[scenario_name], scenario_name)
    actions = definition.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"scenario_definitions.{scenario_name}.actions must be a non-empty list.")
    reports = []
    for i, action in enumerate(actions, start=1):
        action = _normalize_action(action, scenario_name, i)
        action_type = action.get("type")
        if action_type not in ACTION_HANDLERS:
            known = ", ".join(sorted(ACTION_HANDLERS))
            raise ValueError(f"Unknown action type '{action_type}'. Known action types: {known}")
        reports.append(ACTION_HANDLERS[action_type](n, action, catalogue, scenario=scenario))
    return reports


def _apply_scenario_catalogue(
    n: pypsa.Network,
    stochastic_param: dict,
) -> tuple[bool, list[str]]:
    """Apply the declarative scenario catalogue in stochastic or deterministic mode."""
    catalogue = _merge_stochastic_param(stochastic_param)
    enabled = bool(catalogue.get("enable", catalogue.get("enabled", False)))
    if enabled:
        scenarios = _ensure_dict(catalogue.get("scenarios"), "scenarios")
        if not scenarios:
            raise ValueError("stochastic_scenarios.enable=true but no scenarios were provided.")
        probabilities = {name: float(prob) for name, prob in scenarios.items()}
        missing = sorted(set(probabilities) - set(_scenario_definitions(catalogue)))
        if missing:
            raise KeyError(
                "Missing scenario_definitions for scenario(s): " + ", ".join(missing)
            )
        logger.info("Enabling stochastic scenarios via n.set_scenarios(...): %s", probabilities)
        n.set_scenarios(probabilities)
        active = []
        for scenario_name in probabilities:
            _apply_action_catalogue_scenario(
                n=n,
                catalogue=catalogue,
                scenario_name=scenario_name,
                scenario=scenario_name,
            )
            active.append(scenario_name)
        return True, active

    active_scenario = catalogue.get("active_scenario")
    if active_scenario is None:
        logger.info("Stochastic mode disabled and no active_scenario configured; no-op.")
        return False, []
    active_scenario = str(active_scenario)
    logger.info("Applying deterministic active_scenario '%s' from scenario catalogue.", active_scenario)
    _apply_action_catalogue_scenario(
        n=n,
        catalogue=catalogue,
        scenario_name=active_scenario,
        scenario=None,
    )
    return False, [active_scenario]


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
    Apply a declarative source-based scenario catalogue.

    - stochastic_scenarios.enable=true reads probabilities from scenarios,
      calls n.set_scenarios(...), and applies scenario_definitions actions.
    - stochastic_scenarios.enable=false with active_scenario applies that single
      scenario definition to the deterministic network without set_scenarios.
    - stochastic_scenarios.enable=false without active_scenario is a no-op.
    """
    del config, wildcards
    is_stochastic, active_names = _apply_scenario_catalogue(
        n=n,
        stochastic_param=stochastic_param,
    )

    if is_stochastic:
        _apply_patch_config_if_present(n=n, stochastic_param=stochastic_param)

    if active_names:
        logger.info("Applied scenario catalogue entry/entries: %s", active_names)
    else:
        logger.info("No scenario catalogue entry was applied.")


# Old named structured scenario builders are intentionally no longer registered.


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "stochasticify_sector_network",
            opts="",
            clusters="adm",
            configfiles="config/test_stochastic_scenarios/config.yaml",
            sector_opts="",
            planning_horizons="2050",
            run="BASE",
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

    action_meta = dict(n.meta)
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.meta.update(action_meta)
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