#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch analysis over many PyPSA networks vs a chosen base network.

Run:
    python scripts/analysis_networks_batch.py

Edit the USER SETTINGS below to point to:
- the folder containing scenario subfolders
- the base network path
- the network glob inside each scenario folder
- the output Excel path
- the config yaml to read plotting.tech_colors (recommended)
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

from pypsa.statistics import get_transmission_carriers
from add_electricity import sanitize_carriers


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

# Folder that contains scenario subfolders, e.g.:
# results/sensitivity_2050/
# ├── emissions_2040/
# │   └── networks/*.nc
# ├── ev_use/
# │   └── networks/*.nc
ROOT_DIR = Path("results/eth_results")

# Glob pattern inside each scenario folder
# e.g. "networks/*.nc" (all)
# or "networks/base_s_adm___2050.nc" (one exact file)
NETWORK_GLOB = "networks/base_s_adm___2050.nc"

# Base network path (can be anywhere)
BASE_NETWORK_PATH = Path("results/eth_results/base/networks/base_s_adm___2050.nc")

# Output Excel (single file)
OUTPUT_EXCEL = Path("results/eth_results/csvs/analysis_networks_vs_base.xlsx")

# Config YAML used to obtain plotting.tech_colors (filter technologies)
# Use a FULL config (the one PyPSA-Eur normally uses), not a partial one.
CONFIG_YAML = Path("config/test_stochastic_scenarios/config.yaml")

# Plotting YAML used to obtain plotting.tech_colors
PLOTTING_YAML = Path("config/plotting.default.yaml")
# oppure "config/plotting.default.yaml" o il tuo path reale


# How to label scenarios in Excel columns:
# "folder"            -> scenario folder name (recommended)
# "filename"          -> .nc file name stem
# "folder__filename"  -> both (safe if many files per folder)
SCENARIO_NAME_MODE = "folder"


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
    """Deep-merge dict b into dict a (returns a new dict)."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_and_merge_configs(config_yaml: Path, plotting_yaml: Path) -> dict:
    """Load config + plotting YAML and deep-merge them (plotting overrides on conflict)."""
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


# -----------------------------
# Balance-map logic replica
# -----------------------------
def get_supply_consumption_map(n: pypsa.Network, bus_carrier: str, tech_colors: dict):
    """Replicate plot_balance_map supply/consumption classification."""
    eb = n.statistics.energy_balance(
        bus_carrier=bus_carrier,
        groupby=["bus", "carrier"],
    )

    if eb.empty:
        return set(), set(), pd.Series(dtype=float)

    bus_sizes = eb.groupby(level=["bus", "carrier"]).sum()

    # Remove transmission carriers
    transmission = get_transmission_carriers(n, bus_carrier=bus_carrier).rename({"name": "carrier"})
    transmission_carriers = set(transmission.unique("carrier"))

    bus_sizes = bus_sizes[
        ~bus_sizes.index.get_level_values("carrier").isin(transmission_carriers)
    ]

    # Remove the bus carrier itself
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier") != bus_carrier
    ]

    # Keep only technologies shown in balance maps
    bus_sizes = bus_sizes[
        bus_sizes.index.get_level_values("carrier").isin(tech_colors)
    ]

    if bus_sizes.empty:
        return set(), set(), bus_sizes

    pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
    neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")
    common = pos_carriers.intersection(neg_carriers)

    def total_abs(carrier: str, sign: int) -> float:
        vals = bus_sizes.loc[:, carrier]
        return float(vals[vals * sign > 0].abs().sum())

    supply = set(pos_carriers) - set(common)
    consumption = set(neg_carriers) - set(common)

    for c in common:
        if total_abs(c, +1) >= total_abs(c, -1):
            supply.add(c)
        else:
            consumption.add(c)

    return supply, consumption, bus_sizes


def analyze_one_network(n: pypsa.Network, tech_colors: dict, config: dict) -> pd.DataFrame:
    """Compute Supply/Consumption tables (rank, value, share) for one network."""
    # Same preprocessing as plot_balance_map
    sanitize_carriers(n, config)
    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = True

    records: list[dict] = []
    bus_carriers = n.buses.carrier.unique()

    for group in bus_carriers:
        supply_carriers, cons_carriers, bus_sizes = get_supply_consumption_map(n, group, tech_colors)

        if bus_sizes.empty:
            continue

        # Supply
        for tech in supply_carriers:
            values = bus_sizes.loc[:, tech]
            value = float(values[values > 0].sum())
            if value != 0:
                records.append({"kind": "Supply", "group": group, "technology": tech, "value": value})

        # Consumption
        for tech in cons_carriers:
            values = bus_sizes.loc[:, tech]
            value = float((-values[values < 0]).sum())
            if value != 0:
                records.append({"kind": "Consumption", "group": group, "technology": tech, "value": value})

    df = pd.DataFrame(records)

    def finalize(d: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per group/technology and create placeholder rank/share."""
        if d.empty:
            return pd.DataFrame(columns=["group", "rank", "technology", "value", "share [%]"])

        out = []
        for group, g in d.groupby("group"):
            g = (
                g.groupby("technology", as_index=False)
                .value.sum()
                .sort_values("value", ascending=False)
                .reset_index(drop=True)
            )
            total = float(g.value.sum())
            if total == 0:
                continue
            g["group"] = group
            out.append(g)

        if not out:
            return pd.DataFrame(columns=["group", "rank", "technology", "value", "share [%]"])

        df2 = pd.concat(out, ignore_index=True)
        df2["rank"] = None
        df2["share [%]"] = None
        return df2[["group", "rank", "technology", "value", "share [%]"]]

    supply = finalize(df[df.kind == "Supply"])
    cons = finalize(df[df.kind == "Consumption"])

    # Add "others" to close balance and recompute rank/share
    def add_others_and_recompute_shares(supply: pd.DataFrame, consumption: pd.DataFrame, tol: float = 1e-6):
        supply = supply.copy()
        consumption = consumption.copy()

        groups = sorted(set(supply.group.unique()) | set(consumption.group.unique()))

        new_supply = []
        new_consumption = []

        for group in groups:
            s = float(supply.loc[supply.group == group, "value"].sum())
            c = float(consumption.loc[consumption.group == group, "value"].sum())
            diff = s - c

            if abs(diff) <= tol:
                continue

            if diff > 0:
                new_consumption.append({"group": group, "technology": "others", "value": diff})
            else:
                new_supply.append({"group": group, "technology": "others", "value": -diff})

        if new_supply:
            supply = pd.concat([supply, pd.DataFrame(new_supply)], ignore_index=True)
        if new_consumption:
            consumption = pd.concat([consumption, pd.DataFrame(new_consumption)], ignore_index=True)

        def recompute(d: pd.DataFrame) -> pd.DataFrame:
            out = []
            for group, g in d.groupby("group"):
                g = g.sort_values("value", ascending=False).reset_index(drop=True)
                total = float(g.value.sum())
                if total == 0:
                    continue
                g["rank"] = g.index + 1
                g["share [%]"] = 100.0 * g.value / total
                g["group"] = group
                out.append(g)

            if not out:
                return pd.DataFrame(columns=["group", "rank", "technology", "value", "share [%]"])

            return pd.concat(out, ignore_index=True)[["group", "rank", "technology", "value", "share [%]"]]

        return recompute(supply), recompute(consumption)

    supply, cons = add_others_and_recompute_shares(supply, cons)

    supply["kind"] = "Supply"
    cons["kind"] = "Consumption"
    out = pd.concat([supply, cons], ignore_index=True)

    return out[["kind", "group", "rank", "technology", "value", "share [%]"]]


def to_wide(levels_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long table to wide with rank/value/share per scenario."""
    idx = ["kind", "group", "technology"]

    v = levels_long.pivot_table(index=idx, columns="scenario", values="value", aggfunc="sum", fill_value=0.0)
    s = levels_long.pivot_table(index=idx, columns="scenario", values="share [%]", aggfunc="sum", fill_value=0.0)
    r = levels_long.pivot_table(index=idx, columns="scenario", values="rank", aggfunc="min")

    v.columns = [f"value__{c}" for c in v.columns]
    s.columns = [f"share__{c}" for c in s.columns]
    r.columns = [f"rank__{c}" for c in r.columns]

    out = pd.concat([r, v, s], axis=1).reset_index()

    vcols = [c for c in out.columns if c.startswith("value__")]
    out["_max_value"] = out[vcols].max(axis=1) if vcols else 0.0
    out = out.sort_values(["kind", "group", "_max_value"], ascending=[True, True, False]).drop(columns=["_max_value"])
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


def main():
    # Basic checks
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")
    if not BASE_NETWORK_PATH.exists():
        raise FileNotFoundError(f"BASE_NETWORK_PATH not found: {BASE_NETWORK_PATH}")

    cfg = _load_and_merge_configs(CONFIG_YAML, PLOTTING_YAML)

    # tech_colors is used ONLY as technology filter (same as plot_balance_map)
    try:
        tech_colors = cfg["plotting"]["tech_colors"]
    except Exception as e:
        raise KeyError(
            "Missing plotting.tech_colors. Check CONFIG_YAML + PLOTTING_YAML paths and contents."
        ) from e

    # Find all .nc candidates under ROOT_DIR/<scenario>/<NETWORK_GLOB>
    candidates = sorted(ROOT_DIR.glob(f"*/{NETWORK_GLOB}"))
    if not candidates:
        raise FileNotFoundError(f"No networks found under {ROOT_DIR} with pattern */{NETWORK_GLOB}")

    # Analyze base network
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = pypsa.Network(str(BASE_NETWORK_PATH))
    base_long = analyze_one_network(n_base, tech_colors, cfg)
    base_long["scenario"] = "__BASE__"

    # Analyze all candidates
    all_long = [base_long]

    for p in candidates:
        label = _scenario_label(p, SCENARIO_NAME_MODE)
        print(f"[SCENARIO={label}] Loading: {p}")
        n = pypsa.Network(str(p))
        df = analyze_one_network(n, tech_colors, cfg)
        df["scenario"] = label
        all_long.append(df)

    levels_long = pd.concat(all_long, ignore_index=True)

    # Build wide tables
    wide = to_wide(levels_long)
    vs_base = delta_vs_base(wide, "__BASE__")

    # Split into four sheets
    levels_consumption = split_by_kind(wide, "Consumption")
    levels_supply = split_by_kind(wide, "Supply")
    vs_base_consumption = split_by_kind(vs_base, "Consumption")
    vs_base_supply = split_by_kind(vs_base, "Supply")

    # Write Excel
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