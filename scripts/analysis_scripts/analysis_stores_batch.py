#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch analysis of Store capacities and usage over many PyPSA networks vs a base network.

This script focuses on PyPSA Stores only, not StorageUnits and not Links.

Outputs one Excel with:
- by_carrier_levels
- by_carrier_country_levels
- by_carrier_vs_base
- by_carrier_country_vs_base

Metrics included:
- store_count
- capacity_energy
- mean_energy
- min_energy
- max_energy
- mean_fill_ratio
- max_fill_ratio

Run:
    python scripts/analysis_stores_batch.py
"""

from pathlib import Path
import sys
import yaml

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from add_electricity import sanitize_carriers


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT_DIR = Path("results/eth_results")
NETWORK_GLOB = "networks/base_s_adm___2050.nc"
BASE_NETWORK_PATH = Path("results/eth_results/base/networks/base_s_adm___2050.nc")
OUTPUT_EXCEL = Path("results/eth_results/csvs/analysis_stores_vs_base.xlsx")

CONFIG_YAML = Path("config/test_stochastic_scenarios/config.yaml")
PLOTTING_YAML = Path("config/plotting.default.yaml")

SCENARIO_NAME_MODE = "folder"

# Scenario folders to exclude from the batch analysis
EXCLUDED_SCENARIOS = {"base", "stochastic_network"}
# Example:
# EXCLUDED_SCENARIOS = {"base", "debug_case"}

# Optional filter on store carriers
STORE_CARRIER_FILTER = None
# Example:
# STORE_CARRIER_FILTER = ["H2 Store", "battery"]

# If True, keep only rows with strictly positive capacity
DROP_ZERO_CAPACITY = True

# If True, compute usage metrics from n.stores_t.e
INCLUDE_USAGE_METRICS = True

# If True, create country-level tables
INCLUDE_COUNTRY_SHEETS = True

ZERO_TOL = 1e-9


# =========================
# INTERNALS
# =========================

def _load_config(cfg_path: Path) -> dict:
    """Load YAML config."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config YAML did not parse to dict: {cfg_path}")
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into dict a."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_and_merge_configs(config_yaml: Path, plotting_yaml: Path) -> dict:
    """Load config + plotting YAML and deep-merge them."""
    cfg = _load_config(config_yaml)
    plot = _load_config(plotting_yaml)
    return _deep_merge(cfg, plot)


def _scenario_label(nc_path: Path, mode: str) -> str:
    """Build a scenario label from a .nc path."""
    scen_folder = nc_path.parent.parent.name
    fname = nc_path.stem
    if mode == "folder":
        return scen_folder
    if mode == "filename":
        return fname
    if mode == "folder__filename":
        return f"{scen_folder}__{fname}"
    raise ValueError(f"Unknown SCENARIO_NAME_MODE: {mode}")


def _filter_excluded_scenarios(candidates: list[Path], excluded_scenarios: set[str]) -> list[Path]:
    """Remove candidate networks whose scenario folder name is excluded."""
    if not excluded_scenarios:
        return candidates

    excluded = set(excluded_scenarios)
    kept = []

    for p in candidates:
        scen_folder = p.parent.parent.name
        if scen_folder not in excluded:
            kept.append(p)

    return kept


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise safe divide returning 0 where denominator is 0."""
    den2 = den.where(den.abs() > ZERO_TOL, np.nan)
    out = num / den2
    return out.fillna(0.0)


def _get_store_capacity_series(n: pypsa.Network) -> pd.Series:
    """
    Return installed store energy capacity.

    Prefer e_nom_opt if present and non-null, otherwise fallback to e_nom.
    """
    stores = n.stores.copy()

    if "e_nom_opt" in stores.columns:
        cap = stores["e_nom_opt"].copy()
        if "e_nom" in stores.columns:
            cap = cap.where(cap.notna(), stores["e_nom"])
    elif "e_nom" in stores.columns:
        cap = stores["e_nom"].copy()
    else:
        raise KeyError("Neither 'e_nom_opt' nor 'e_nom' found in n.stores.")

    return cap.fillna(0.0).astype(float)


def _infer_store_country(n: pypsa.Network, stores: pd.DataFrame) -> pd.Series:
    """
    Infer store country from connected bus.

    Preference:
    1. n.buses.country
    2. n.buses.location
    3. first two characters of bus name
    """
    buses = n.buses.copy()
    bus_index = stores["bus"]

    country = pd.Series(index=stores.index, dtype=object)

    if "country" in buses.columns:
        country = bus_index.map(buses["country"])

    if "location" in buses.columns:
        missing = country.isna() | (country.astype(str).str.strip() == "")
        country.loc[missing] = bus_index.loc[missing].map(buses["location"])

    missing = country.isna() | (country.astype(str).str.strip() == "")
    country.loc[missing] = bus_index.loc[missing].astype(str).str[:2]

    return country.fillna("UNKNOWN").astype(str)


def _get_store_energy_timeseries(n: pypsa.Network, store_index: pd.Index) -> pd.DataFrame | None:
    """
    Return store energy time series aligned to store_index.

    Uses n.stores_t.e if available.
    """
    if not hasattr(n, "stores_t"):
        return None
    if not hasattr(n.stores_t, "e"):
        return None

    e = n.stores_t.e.copy()
    if e.empty:
        return None

    cols = e.columns.intersection(store_index)
    if len(cols) == 0:
        return None

    e = e.loc[:, cols]

    missing_cols = store_index.difference(e.columns)
    for col in missing_cols:
        e[col] = np.nan

    e = e.reindex(columns=store_index)
    return e


def _build_store_level_table(n: pypsa.Network, config: dict) -> pd.DataFrame:
    """
    Build one long table with store metrics at store level.
    """
    sanitize_carriers(n, config)

    if n.stores.empty:
        return pd.DataFrame(
            columns=[
                "store",
                "carrier",
                "country",
                "capacity_energy",
                "mean_energy",
                "min_energy",
                "max_energy",
                "mean_fill_ratio",
                "max_fill_ratio",
            ]
        )

    stores = n.stores.copy()

    if STORE_CARRIER_FILTER is not None:
        stores = stores[stores["carrier"].isin(STORE_CARRIER_FILTER)].copy()

    if stores.empty:
        return pd.DataFrame(
            columns=[
                "store",
                "carrier",
                "country",
                "capacity_energy",
                "mean_energy",
                "min_energy",
                "max_energy",
                "mean_fill_ratio",
                "max_fill_ratio",
            ]
        )

    capacity = _get_store_capacity_series(n).reindex(stores.index).fillna(0.0)
    country = _infer_store_country(n, stores)

    out = pd.DataFrame(
        {
            "store": stores.index,
            "carrier": stores["carrier"].astype(str),
            "country": country,
            "capacity_energy": capacity.astype(float),
        }
    )

    if DROP_ZERO_CAPACITY:
        out = out[out["capacity_energy"].abs() > ZERO_TOL].copy()
        stores = stores.loc[out["store"]].copy()

    if out.empty:
        return pd.DataFrame(
            columns=[
                "store",
                "carrier",
                "country",
                "capacity_energy",
                "mean_energy",
                "min_energy",
                "max_energy",
                "mean_fill_ratio",
                "max_fill_ratio",
            ]
        )

    out = out.set_index("store")

    if INCLUDE_USAGE_METRICS:
        e_ts = _get_store_energy_timeseries(n, out.index)
        if e_ts is not None:
            mean_energy = e_ts.mean(axis=0)
            min_energy = e_ts.min(axis=0)
            max_energy = e_ts.max(axis=0)
        else:
            mean_energy = pd.Series(0.0, index=out.index)
            min_energy = pd.Series(0.0, index=out.index)
            max_energy = pd.Series(0.0, index=out.index)

        out["mean_energy"] = mean_energy.reindex(out.index).fillna(0.0).astype(float)
        out["min_energy"] = min_energy.reindex(out.index).fillna(0.0).astype(float)
        out["max_energy"] = max_energy.reindex(out.index).fillna(0.0).astype(float)
        out["mean_fill_ratio"] = _safe_divide(out["mean_energy"], out["capacity_energy"])
        out["max_fill_ratio"] = _safe_divide(out["max_energy"], out["capacity_energy"])
    else:
        out["mean_energy"] = np.nan
        out["min_energy"] = np.nan
        out["max_energy"] = np.nan
        out["mean_fill_ratio"] = np.nan
        out["max_fill_ratio"] = np.nan

    return out.reset_index()


def _aggregate_store_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate store metrics by group columns.

    capacity_energy is summed.
    mean/min/max energy are summed because they represent system-wide energy content
    across stores at each store-level statistic approximation.

    fill ratios are recomputed at aggregated level as:
    aggregated_mean_energy / aggregated_capacity_energy
    aggregated_max_energy / aggregated_capacity_energy
    """
    if df.empty:
        cols = group_cols + [
            "store_count",
            "capacity_energy",
            "mean_energy",
            "min_energy",
            "max_energy",
            "mean_fill_ratio",
            "max_fill_ratio",
        ]
        return pd.DataFrame(columns=cols)

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            store_count=("store", "count"),
            capacity_energy=("capacity_energy", "sum"),
            mean_energy=("mean_energy", "sum"),
            min_energy=("min_energy", "sum"),
            max_energy=("max_energy", "sum"),
        )
    )

    agg["mean_fill_ratio"] = _safe_divide(agg["mean_energy"], agg["capacity_energy"])
    agg["max_fill_ratio"] = _safe_divide(agg["max_energy"], agg["capacity_energy"])

    return agg


def analyze_one_network(n: pypsa.Network, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze one network and return:
    - by_carrier
    - by_carrier_country
    """
    store_level = _build_store_level_table(n, config)

    by_carrier = _aggregate_store_metrics(store_level, ["carrier"])
    by_carrier_country = _aggregate_store_metrics(store_level, ["carrier", "country"])

    return by_carrier, by_carrier_country


def _wide_metrics_table(levels_long: pd.DataFrame, index_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """
    Pivot long metrics table to wide format with one column per scenario and metric.
    """
    pieces = []

    for metric in metric_cols:
        w = levels_long.pivot_table(
            index=index_cols,
            columns="scenario",
            values=metric,
            aggfunc="sum" if metric != "store_count" else "sum",
            fill_value=0.0,
        )
        w.columns = [f"{metric}__{c}" for c in w.columns]
        pieces.append(w)

    out = pd.concat(pieces, axis=1).reset_index()

    main_metric_cols = [c for c in out.columns if c.startswith("capacity_energy__")]
    out["_max_capacity"] = out[main_metric_cols].max(axis=1) if main_metric_cols else 0.0
    out = out.sort_values(index_cols + ["_max_capacity"], ascending=[True] * len(index_cols) + [False])
    out = out.drop(columns="_max_capacity")

    return out


def _delta_vs_base(wide: pd.DataFrame, base_label: str, index_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """
    Compute absolute and relative changes vs base for each metric.
    """
    out_cols = list(index_cols)
    out = wide[out_cols].copy()

    scenarios = sorted(
        {c.split("__", 1)[1] for c in wide.columns if "__" in c}
    )

    for metric in metric_cols:
        base_col = f"{metric}__{base_label}"
        if base_col not in wide.columns:
            raise ValueError(f"Base column missing: {base_col}")

        out[base_col] = wide[base_col]

        for sc in scenarios:
            sc_col = f"{metric}__{sc}"
            if sc_col not in wide.columns:
                continue

            out[f"delta_{metric}__{sc}"] = wide[sc_col] - wide[base_col]

            denom = np.maximum(np.abs(wide[base_col].to_numpy(dtype=float)), 1e-12)
            out[f"relchg_{metric}__{sc}"] = (wide[sc_col] - wide[base_col]) / denom

    return out


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")
    if not BASE_NETWORK_PATH.exists():
        raise FileNotFoundError(f"BASE_NETWORK_PATH not found: {BASE_NETWORK_PATH}")

    cfg = _load_and_merge_configs(CONFIG_YAML, PLOTTING_YAML)

    candidates = sorted(ROOT_DIR.glob(f"*/{NETWORK_GLOB}"))
    candidates = _filter_excluded_scenarios(candidates, EXCLUDED_SCENARIOS)

    if not candidates:
        raise FileNotFoundError(
            f"No networks found under {ROOT_DIR} with pattern */{NETWORK_GLOB} "
            f"after excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}"
        )

    if EXCLUDED_SCENARIOS:
        print(f"[INFO] Excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}")

    metric_cols = [
        "store_count",
        "capacity_energy",
        "mean_energy",
        "min_energy",
        "max_energy",
        "mean_fill_ratio",
        "max_fill_ratio",
    ]

    # Base
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = pypsa.Network(str(BASE_NETWORK_PATH))
    base_carrier, base_carrier_country = analyze_one_network(n_base, cfg)
    base_carrier["scenario"] = "__BASE__"
    base_carrier_country["scenario"] = "__BASE__"

    all_carrier = [base_carrier]
    all_carrier_country = [base_carrier_country]

    # Scenarios
    for p in candidates:
        label = _scenario_label(p, SCENARIO_NAME_MODE)
        print(f"[SCENARIO={label}] Loading: {p}")
        n = pypsa.Network(str(p))
        df_carrier, df_carrier_country = analyze_one_network(n, cfg)
        df_carrier["scenario"] = label
        df_carrier_country["scenario"] = label
        all_carrier.append(df_carrier)
        all_carrier_country.append(df_carrier_country)

    levels_carrier_long = pd.concat(all_carrier, ignore_index=True)
    levels_carrier_country_long = pd.concat(all_carrier_country, ignore_index=True)

    by_carrier_levels = _wide_metrics_table(
        levels_carrier_long,
        index_cols=["carrier"],
        metric_cols=metric_cols,
    )
    by_carrier_vs_base = _delta_vs_base(
        by_carrier_levels,
        "__BASE__",
        index_cols=["carrier"],
        metric_cols=metric_cols,
    )

    if INCLUDE_COUNTRY_SHEETS:
        by_carrier_country_levels = _wide_metrics_table(
            levels_carrier_country_long,
            index_cols=["carrier", "country"],
            metric_cols=metric_cols,
        )
        by_carrier_country_vs_base = _delta_vs_base(
            by_carrier_country_levels,
            "__BASE__",
            index_cols=["carrier", "country"],
            metric_cols=metric_cols,
        )
    else:
        by_carrier_country_levels = pd.DataFrame()
        by_carrier_country_vs_base = pd.DataFrame()

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        by_carrier_levels.to_excel(writer, sheet_name="by_carrier_levels", index=False)
        by_carrier_vs_base.to_excel(writer, sheet_name="by_carrier_vs_base", index=False)

        if INCLUDE_COUNTRY_SHEETS:
            by_carrier_country_levels.to_excel(writer, sheet_name="by_carrier_country_levels", index=False)
            by_carrier_country_vs_base.to_excel(writer, sheet_name="by_carrier_country_vs_base", index=False)

    print(f"✔ Wrote Excel: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        raise