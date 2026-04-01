#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch analysis over many PyPSA networks vs a chosen base network.

This version is meant for BALANCE DEBUGGING, not for plotting-style aggregation.

Main differences vs the old script:
- no filtering by plotting.tech_colors
- no artificial "others" category
- keep all available carriers
- keep component information (Generator, Link, Store, StorageUnit, Load, ...)
- use PyPSA energy_balance directly and transparently

Run:
    python scripts/analysis_networks_batch.py
"""

from pathlib import Path
import sys
import yaml

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from add_electricity import sanitize_carriers


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT_DIR = Path("results/eth_results")
NETWORK_GLOB = "networks/base_s_adm___2050.nc"
BASE_NETWORK_PATH = Path("results/eth_results/base/networks/base_s_adm___2050.nc")
OUTPUT_EXCEL = Path("results/eth_results/csvs/analysis_networks_vs_base.xlsx")

CONFIG_YAML = Path("config/test_stochastic_scenarios/config.yaml")
PLOTTING_YAML = Path("config/plotting.default.yaml")

EXCLUDED_SCENARIOS = {"base", "stochastic_network"}

SCENARIO_NAME_MODE = "folder"

# Optional filters for debugging
BUS_CARRIER_FILTER = None
# Example:
# BUS_CARRIER_FILTER = ["AC"]
# BUS_CARRIER_FILTER = ["AC", "DC"]

DROP_ZERO_VALUES = True
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
    scen_folder = nc_path.parent.parent.name  # <scenario>/networks/<file>.nc
    fname = nc_path.stem
    if mode == "folder":
        return scen_folder
    if mode == "filename":
        return fname
    if mode == "folder__filename":
        return f"{scen_folder}__{fname}"
    raise ValueError(f"Unknown SCENARIO_NAME_MODE: {mode}")


def _to_series(obj) -> pd.Series:
    """
    Convert PyPSA output to a Series if possible.

    energy_balance usually returns a Series, but this keeps the code robust.
    """
    if isinstance(obj, pd.Series):
        return obj

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        raise TypeError(
            "Expected a Series-like energy balance result, got a DataFrame with "
            f"{obj.shape[1]} columns."
        )

    raise TypeError(f"Unsupported energy balance output type: {type(obj)}")


def _energy_balance_to_long(
    n: pypsa.Network,
    bus_carrier: str | None = None,
) -> pd.DataFrame:
    """
    Extract full energy balance in long format.

    Positive values are interpreted as injections to the bus,
    negative values as withdrawals from the bus.
    """
    eb = n.statistics.energy_balance(
        bus_carrier=bus_carrier,
        groupby=["bus", "carrier"],
    )

    s = _to_series(eb)

    if s.empty:
        return pd.DataFrame(columns=["bus", "carrier", "value"])

    df = s.rename("value").reset_index()

    expected = {"bus", "carrier", "value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            "Unexpected columns returned by n.statistics.energy_balance(...). "
            f"Missing columns: {missing}. Got: {list(df.columns)}"
        )

    return df


def _prepare_balance_records(
    n: pypsa.Network,
    bus_carrier_filter=None,
) -> pd.DataFrame:
    """
    Build transparent balance records for the whole network.

    Output columns:
    - group       : bus carrier group (AC, DC, H2, ...)
    - kind        : Supply or Consumption
    - technology  : carrier
    - bus         : bus name
    - value       : positive magnitude
    - signed_value: original signed balance value
    """
    bus_carriers = pd.Index(n.buses.carrier.dropna().unique())

    if bus_carrier_filter is not None:
        bus_carriers = pd.Index([bc for bc in bus_carriers if bc in set(bus_carrier_filter)])

    records = []

    for group in bus_carriers:
        eb_long = _energy_balance_to_long(n, bus_carrier=group)
        if eb_long.empty:
            continue

        eb_long = eb_long.copy()
        eb_long["group"] = group

        pos = eb_long[eb_long["value"] > ZERO_TOL].copy()
        if not pos.empty:
            pos["kind"] = "Supply"
            pos["signed_value"] = pos["value"]
            pos["value"] = pos["value"].abs()
            records.append(pos)

        neg = eb_long[eb_long["value"] < -ZERO_TOL].copy()
        if not neg.empty:
            neg["kind"] = "Consumption"
            neg["signed_value"] = neg["value"]
            neg["value"] = neg["value"].abs()
            records.append(neg)

    if not records:
        return pd.DataFrame(
            columns=["kind", "group", "technology", "bus", "value", "signed_value"]
        )

    out = pd.concat(records, ignore_index=True)
    out = out.rename(columns={"carrier": "technology"})

    return out[
        ["kind", "group", "technology", "bus", "value", "signed_value"]
    ]


def analyze_one_network(n: pypsa.Network, config: dict) -> pd.DataFrame:
    """
    Compute transparent Supply/Consumption tables for one network.

    No hidden residuals, no "others", no plotting filters.
    """
    sanitize_carriers(n, config)
    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = False

    raw = _prepare_balance_records(n, BUS_CARRIER_FILTER)

    if raw.empty:
        return pd.DataFrame(
            columns=["kind", "group", "rank", "technology", "value", "share [%]"]
        )

    agg = (
        raw.groupby(["kind", "group", "technology"], as_index=False)["value"]
        .sum()
    )

    if DROP_ZERO_VALUES:
        agg = agg[agg["value"].abs() > ZERO_TOL].copy()

    if agg.empty:
        return pd.DataFrame(
            columns=["kind", "group", "rank", "technology", "value", "share [%]"]
        )

    out = []

    for (kind, group), g in agg.groupby(["kind", "group"], sort=True):
        g = g.sort_values("value", ascending=False).reset_index(drop=True)
        total = float(g["value"].sum())

        if total <= ZERO_TOL:
            continue

        g["rank"] = g.index + 1
        g["share [%]"] = 100.0 * g["value"] / total
        g["kind"] = kind
        g["group"] = group

        out.append(g[["kind", "group", "rank", "technology", "value", "share [%]"]])

    if not out:
        return pd.DataFrame(
            columns=["kind", "group", "rank", "technology", "value", "share [%]"]
        )

    return pd.concat(out, ignore_index=True)


def to_wide(levels_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long table to wide with rank/value/share per scenario.
    """
    idx = ["kind", "group", "technology"]

    v = levels_long.pivot_table(
        index=idx,
        columns="scenario",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )
    s = levels_long.pivot_table(
        index=idx,
        columns="scenario",
        values="share [%]",
        aggfunc="sum",
        fill_value=0.0,
    )
    r = levels_long.pivot_table(
        index=idx,
        columns="scenario",
        values="rank",
        aggfunc="min",
    )

    v.columns = [f"value__{c}" for c in v.columns]
    s.columns = [f"share__{c}" for c in s.columns]
    r.columns = [f"rank__{c}" for c in r.columns]

    out = pd.concat([r, v, s], axis=1).reset_index()

    vcols = [c for c in out.columns if c.startswith("value__")]
    out["_max_value"] = out[vcols].max(axis=1) if vcols else 0.0

    out = out.sort_values(
        ["kind", "group", "_max_value", "technology"],
        ascending=[True, True, False, True],
    ).drop(columns=["_max_value"])

    return out


def delta_vs_base(wide: pd.DataFrame, base_label: str) -> pd.DataFrame:
    """Compute delta and relative change vs base label."""
    eps = 1e-12
    base_v = f"value__{base_label}"
    base_s = f"share__{base_label}"

    if base_v not in wide.columns or base_s not in wide.columns:
        raise ValueError(f"Base '{base_label}' not found. Missing columns: {base_v} and/or {base_s}")

    out = wide[["kind", "group", "technology", base_v, base_s]].copy()

    scenarios = sorted({c.split("__", 1)[1] for c in wide.columns if c.startswith("value__")})
    for sc in scenarios:
        v = f"value__{sc}"
        s = f"share__{sc}"

        out[f"delta_value__{sc}"] = wide[v] - wide[base_v]
        out[f"delta_share__{sc}"] = wide[s] - wide[base_s]

        denom_v = np.maximum(np.abs(wide[base_v].to_numpy()), eps)
        denom_s = np.maximum(np.abs(wide[base_s].to_numpy()), eps)

        out[f"relchg_value__{sc}"] = (wide[v] - wide[base_v]) / denom_v
        out[f"relchg_share__{sc}"] = (wide[s] - wide[base_s]) / denom_s

    return out


def split_by_kind(df: pd.DataFrame, kind_value: str) -> pd.DataFrame:
    """Return a copy filtered by kind and without the kind column."""
    out = df[df["kind"] == kind_value].copy()
    return out.drop(columns=["kind"])


def write_component_bus_debug(
    n: pypsa.Network,
    output_excel: Path,
    scenario_name: str,
):
    """
    Detailed dump for one scenario at bus level.
    """
    raw = _prepare_balance_records(n, BUS_CARRIER_FILTER)
    if raw.empty:
        return

    debug_path = output_excel.with_name(
        f"{output_excel.stem}__bus_debug__{scenario_name}.xlsx"
    )

    with pd.ExcelWriter(debug_path, engine="openpyxl") as writer:
        raw.sort_values(
            ["group", "kind", "technology", "bus", "value"],
            ascending=[True, True, True, True, False],
        ).to_excel(writer, sheet_name="raw_bus_level", index=False)

        (
            raw.groupby(["group", "kind", "technology", "bus"], as_index=False)["value"]
            .sum()
            .sort_values(["group", "kind", "value"], ascending=[True, True, False])
            .to_excel(writer, sheet_name="agg_bus_level", index=False)
        )


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

    # Analyze base network
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = pypsa.Network(str(BASE_NETWORK_PATH))
    base_long = analyze_one_network(n_base, cfg)
    base_long["scenario"] = "__BASE__"

    all_long = [base_long]

    # Optional detailed debug for base
    write_component_bus_debug(n_base, OUTPUT_EXCEL, "__BASE__")

    # Analyze all candidates
    for p in candidates:
        label = _scenario_label(p, SCENARIO_NAME_MODE)
        print(f"[SCENARIO={label}] Loading: {p}")
        n = pypsa.Network(str(p))
        df = analyze_one_network(n, cfg)
        df["scenario"] = label
        all_long.append(df)

    levels_long = pd.concat(all_long, ignore_index=True)

    if levels_long.empty:
        raise ValueError("No balance records were produced.")

    wide = to_wide(levels_long)
    vs_base = delta_vs_base(wide, "__BASE__")

    levels_consumption = split_by_kind(wide, "Consumption")
    levels_supply = split_by_kind(wide, "Supply")
    vs_base_consumption = split_by_kind(vs_base, "Consumption")
    vs_base_supply = split_by_kind(vs_base, "Supply")

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        levels_consumption.to_excel(writer, sheet_name="levels_consumption", index=False)
        levels_supply.to_excel(writer, sheet_name="levels_supply", index=False)
        vs_base_consumption.to_excel(writer, sheet_name="vs_base_consumption", index=False)
        vs_base_supply.to_excel(writer, sheet_name="vs_base_supply", index=False)

    print(f"✔ Wrote Excel: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        raise