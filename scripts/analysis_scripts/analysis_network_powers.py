#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch analysis over many PyPSA networks vs a chosen base network,
focusing on optimized component sizes (nom_opt when available).

Output sheets:
- levels_by_component
- levels_by_component_carrier
- vs_base_by_component
- vs_base_by_component_carrier

Run:
    python scripts/analysis_component_sizes_batch.py
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
OUTPUT_EXCEL = Path("results/eth_results/csvs/analysis_component_sizes_vs_base.xlsx")

CONFIG_YAML = Path("config/test_stochastic_scenarios/config.yaml")
PLOTTING_YAML = Path("config/plotting.default.yaml")

EXCLUDED_SCENARIOS = {"base", "stochastic_network"}

SCENARIO_NAME_MODE = "folder"

DROP_ZERO_VALUES = True
ZERO_TOL = 1e-9

FILL_EMPTY_CARRIER = True
EMPTY_CARRIER_LABEL = "<none>"


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


def _filter_excluded_scenarios(candidates: list[Path], excluded_scenarios: set[str]) -> list[Path]:
    """
    Remove candidate networks whose scenario folder name is in excluded_scenarios.
    """
    if not excluded_scenarios:
        return candidates

    excluded = set(excluded_scenarios)
    kept = []

    for p in candidates:
        scen_folder = p.parent.parent.name
        if scen_folder not in excluded:
            kept.append(p)

    return kept


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into dict a (returns a new dict)."""
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


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return df[col] if present, otherwise a constant Series."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _get_final_capacity(df: pd.DataFrame, opt_col: str, base_col: str) -> pd.Series:
    """
    Get final optimized capacity.

    Priority:
    1. use opt_col if present and finite
    2. fallback to base_col
    """
    opt = _safe_series(df, opt_col, default=np.nan)
    base = _safe_series(df, base_col, default=0.0)

    out = opt.copy()
    mask_invalid = ~np.isfinite(out.to_numpy())
    out.loc[mask_invalid] = base.loc[mask_invalid]
    return out.fillna(0.0)


def _clean_carrier(series: pd.Series) -> pd.Series:
    """Normalize carrier labels."""
    out = series.fillna("").astype(str)
    if FILL_EMPTY_CARRIER:
        out = out.replace("", EMPTY_CARRIER_LABEL)
    return out


def _component_metric_records(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract optimized/final nominal capacities for all major components.

    Output columns:
    - component
    - carrier
    - metric
    - value
    """
    records = []

    # -------------------------
    # Generators
    # -------------------------
    if not n.generators.empty:
        df = n.generators.copy()
        records.append(pd.DataFrame({
            "component": "Generator",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "power_final",
            "value": _get_final_capacity(df, "p_nom_opt", "p_nom"),
        }))

    # -------------------------
    # Links
    # -------------------------
    if not n.links.empty:
        df = n.links.copy()
        records.append(pd.DataFrame({
            "component": "Link",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "power_final",
            "value": _get_final_capacity(df, "p_nom_opt", "p_nom"),
        }))

    # -------------------------
    # Stores
    # -------------------------
    if not n.stores.empty:
        df = n.stores.copy()
        records.append(pd.DataFrame({
            "component": "Store",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "energy_final",
            "value": _get_final_capacity(df, "e_nom_opt", "e_nom"),
        }))

    # -------------------------
    # StorageUnits
    # -------------------------
    if not n.storage_units.empty:
        df = n.storage_units.copy()

        p_final = _get_final_capacity(df, "p_nom_opt", "p_nom")
        max_hours = _safe_series(df, "max_hours", default=np.nan)
        e_final = p_final * max_hours
        e_final = e_final.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        records.append(pd.DataFrame({
            "component": "StorageUnit",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "power_final",
            "value": p_final,
        }))

        records.append(pd.DataFrame({
            "component": "StorageUnit",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "energy_final_from_max_hours",
            "value": e_final,
        }))

    # -------------------------
    # Loads
    # -------------------------
    # PyPSA loads typically do not have p_nom_opt; we keep p_set as static nominal demand proxy
    if not n.loads.empty:
        df = n.loads.copy()
        records.append(pd.DataFrame({
            "component": "Load",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "power_set",
            "value": _safe_series(df, "p_set", default=0.0),
        }))

    # -------------------------
    # Lines
    # -------------------------
    if not n.lines.empty:
        df = n.lines.copy()
        records.append(pd.DataFrame({
            "component": "Line",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "apparent_power_final",
            "value": _get_final_capacity(df, "s_nom_opt", "s_nom"),
        }))

    # -------------------------
    # Transformers
    # -------------------------
    if not n.transformers.empty:
        df = n.transformers.copy()
        records.append(pd.DataFrame({
            "component": "Transformer",
            "carrier": _clean_carrier(df.get("carrier", pd.Series("", index=df.index))),
            "metric": "apparent_power_final",
            "value": _get_final_capacity(df, "s_nom_opt", "s_nom"),
        }))

    if not records:
        return pd.DataFrame(columns=["component", "carrier", "metric", "value"])

    out = pd.concat(records, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)

    if DROP_ZERO_VALUES:
        out = out[out["value"].abs() > ZERO_TOL].copy()

    return out


def analyze_one_network(n: pypsa.Network, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute optimized/final capacity tables for one network.

    Returns
    -------
    by_component : pd.DataFrame
        Aggregated by component + metric
    by_component_carrier : pd.DataFrame
        Aggregated by component + carrier + metric
    """
    sanitize_carriers(n, config)

    raw = _component_metric_records(n)

    if raw.empty:
        empty_comp = pd.DataFrame(columns=["component", "metric", "value"])
        empty_comp_car = pd.DataFrame(columns=["component", "carrier", "metric", "value"])
        return empty_comp, empty_comp_car

    by_component = (
        raw.groupby(["component", "metric"], as_index=False)["value"]
        .sum()
    )

    by_component_carrier = (
        raw.groupby(["component", "carrier", "metric"], as_index=False)["value"]
        .sum()
    )

    return by_component, by_component_carrier


def to_wide(df_long: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """
    Pivot long table to wide with values per scenario.
    """
    out = df_long.pivot_table(
        index=index_cols,
        columns="scenario",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    value_cols = [c for c in out.columns if c not in index_cols]
    rename_map = {c: f"value__{c}" for c in value_cols}
    out = out.rename(columns=rename_map)

    vcols = [c for c in out.columns if c.startswith("value__")]
    if vcols:
        out["_max_value"] = out[vcols].max(axis=1)
        out = out.sort_values(index_cols[:-1] + ["_max_value", index_cols[-1]], ascending=True)
        out = out.drop(columns="_max_value")

    return out


def delta_vs_base(wide: pd.DataFrame, base_label: str, index_cols: list[str]) -> pd.DataFrame:
    """Compute delta and relative change vs base label."""
    eps = 1e-12
    base_v = f"value__{base_label}"

    if base_v not in wide.columns:
        raise ValueError(f"Base '{base_label}' not found. Missing column: {base_v}")

    out = wide[index_cols + [base_v]].copy()

    scenarios = sorted({
        c.split("__", 1)[1]
        for c in wide.columns
        if c.startswith("value__")
    })

    for sc in scenarios:
        v = f"value__{sc}"
        out[f"delta_value__{sc}"] = wide[v] - wide[base_v]

        denom_v = np.maximum(np.abs(wide[base_v].to_numpy()), eps)
        out[f"relchg_value__{sc}"] = (wide[v] - wide[base_v]) / denom_v

    return out


def write_debug_raw(
    n: pypsa.Network,
    output_excel: Path,
    scenario_name: str,
):
    """
    Detailed raw dump for one scenario.
    """
    raw = _component_metric_records(n)
    if raw.empty:
        return

    debug_path = output_excel.with_name(
        f"{output_excel.stem}__raw_debug__{scenario_name}.xlsx"
    )

    with pd.ExcelWriter(debug_path, engine="openpyxl") as writer:
        raw.sort_values(
            ["component", "carrier", "metric", "value"],
            ascending=[True, True, True, False],
        ).to_excel(writer, sheet_name="raw_component_sizes", index=False)


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

    # Base
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = pypsa.Network(str(BASE_NETWORK_PATH))
    base_comp, base_comp_car = analyze_one_network(n_base, cfg)

    base_comp["scenario"] = "__BASE__"
    base_comp_car["scenario"] = "__BASE__"

    all_comp = [base_comp]
    all_comp_car = [base_comp_car]

    write_debug_raw(n_base, OUTPUT_EXCEL, "__BASE__")

    # Scenarios
    for p in candidates:
        label = _scenario_label(p, SCENARIO_NAME_MODE)
        print(f"[SCENARIO={label}] Loading: {p}")

        n = pypsa.Network(str(p))
        comp, comp_car = analyze_one_network(n, cfg)

        comp["scenario"] = label
        comp_car["scenario"] = label

        all_comp.append(comp)
        all_comp_car.append(comp_car)

    levels_component_long = pd.concat(all_comp, ignore_index=True)
    levels_component_carrier_long = pd.concat(all_comp_car, ignore_index=True)

    if levels_component_long.empty and levels_component_carrier_long.empty:
        raise ValueError("No component size records were produced.")

    levels_by_component = to_wide(
        levels_component_long,
        index_cols=["component", "metric"],
    )
    levels_by_component_carrier = to_wide(
        levels_component_carrier_long,
        index_cols=["component", "carrier", "metric"],
    )

    vs_base_by_component = delta_vs_base(
        levels_by_component,
        "__BASE__",
        index_cols=["component", "metric"],
    )
    vs_base_by_component_carrier = delta_vs_base(
        levels_by_component_carrier,
        "__BASE__",
        index_cols=["component", "carrier", "metric"],
    )

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        levels_by_component.to_excel(writer, sheet_name="levels_by_component", index=False)
        levels_by_component_carrier.to_excel(writer, sheet_name="levels_by_component_carrier", index=False)
        vs_base_by_component.to_excel(writer, sheet_name="vs_base_by_component", index=False)
        vs_base_by_component_carrier.to_excel(writer, sheet_name="vs_base_by_component_carrier", index=False)

    print(f"✔ Wrote Excel: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        raise