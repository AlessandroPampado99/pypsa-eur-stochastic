#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare total load demand by type across MANY PyPSA networks vs a chosen base network.

What it does:
- Scans a results prefix folder like: results/<PREFIX>/<scenario>/networks/*.nc
- Picks ONE network file per scenario (you can constrain by filename pattern)
- Computes total demand summed over snapshots for each load, using:
    - loads_t.p_set (exogenous demand) if present
    - loads_t.p     (optimized/endogenous demand) if present
  Then sums them load-wise: (p_set + p) if both exist.
- Aggregates by a chosen load field:
    - carrier (default)
    - bus
    - load_name (uses each load name as its type)
- Exports 3 Excel files:
    1) levels: demand_by_type for each scenario (columns=scenarios)
    2) vs_base_delta: scenario - base
    3) vs_base_relchg: (scenario - base) / |base|   (dimensionless, NOT percent)

Requirements:
    pip install pypsa pandas openpyxl
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pypsa


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

# Folder containing scenario subfolders:
# results/<PREFIX>/<scenario>/networks/*.nc
PREFIX_DIR = Path("results/eth_results")

# Choose the base network explicitly (can be anywhere)
BASE_NETWORK_PATH = Path("results/eth_results/base/networks/base_s_adm___2050.nc")
BASE_NAME = "__BASE__"  # column name for base in outputs

# If each scenario has multiple .nc in networks/, constrain which one to pick.
# Options:
#   None -> pick the first .nc found in each scenario folder
#   "base_s_adm___2050.nc" -> pick that exact filename (recommended if consistent)
#   "base_s_" -> substring filter (will pick first match by sorted order)
NETWORK_PICKER: Optional[str] = None

# Group demand by: "carrier" (load type), "bus", or "load_name"
GroupByField = Literal["carrier", "bus", "load_name"]
GROUPBY: GroupByField = "carrier"

# Output folder (will contain three Excel files)
OUT_DIR = Path("results/eth_results/postprocess_demand_compare")

# File name prefix for exported Excel files
OUT_STEM = "demand_compare"


# =========================
# IO & SCANNING
# =========================

def read_network(path: str | Path) -> pypsa.Network:
    """Read a PyPSA network from .nc or .h5/.hdf5."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path}")

    suffix = path.suffix.lower()
    n = pypsa.Network()

    if suffix == ".nc":
        n.import_from_netcdf(path)
    elif suffix in {".h5", ".hdf5"}:
        n.import_from_hdf5(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .nc or .h5/.hdf5")

    return n


def find_scenario_networks(prefix_dir: Path, picker: Optional[str]) -> dict[str, Path]:
    """
    Return {scenario_name: network_path} scanning:
        prefix_dir/<scenario>/networks/*.nc

    If picker is:
      - None: first .nc by sorted order
      - endswith ".nc": exact filename match
      - otherwise: substring match on filename
    """
    if not prefix_dir.exists():
        raise FileNotFoundError(f"PREFIX_DIR not found: {prefix_dir}")

    out: dict[str, Path] = {}
    for scenario_dir in sorted([p for p in prefix_dir.iterdir() if p.is_dir()]):
        networks_dir = scenario_dir / "networks"
        if not networks_dir.exists():
            continue

        ncs = sorted(networks_dir.glob("*.nc"))
        if not ncs:
            continue

        chosen: Optional[Path] = None
        if picker is None:
            chosen = ncs[0]
        else:
            if picker.endswith(".nc"):
                matches = [p for p in ncs if p.name == picker]
            else:
                matches = [p for p in ncs if picker in p.name]
            chosen = sorted(matches)[0] if matches else None

        if chosen is not None:
            out[scenario_dir.name] = chosen

    if not out:
        raise FileNotFoundError(
            f"No scenario networks found in {prefix_dir}/*/networks/*.nc "
            f"(picker={picker!r})."
        )

    return out


# =========================
# DEMAND METRICS
# =========================

def total_demand_by_type(
    n: pypsa.Network,
    groupby: GroupByField = "carrier",
) -> pd.Series:
    """
    Total demand summed over snapshots, aggregated by a chosen load field.

    Uses:
      - loads_t.p_set (if present)
      - loads_t.p     (if present)
    Then sums them per load: total_by_load = p_set + p (where present).
    """
    series_list = []

    # p_set (exogenous)
    if hasattr(n.loads_t, "p_set") and n.loads_t.p_set is not None and not n.loads_t.p_set.empty:
        pset_sum = n.loads_t.p_set.sum(axis=0)
        series_list.append(pset_sum)

    # p (optimized/endogenous)
    if hasattr(n.loads_t, "p") and n.loads_t.p is not None and not n.loads_t.p.empty:
        p_sum = n.loads_t.p.sum(axis=0)
        series_list.append(p_sum)

    if not series_list:
        return pd.Series(dtype=float)

    total_by_load = pd.concat(series_list, axis=1).sum(axis=1)

    if groupby == "load_name":
        out = total_by_load.copy()
        out.index = out.index.astype(str)
        return out.groupby(out.index).sum().sort_index()

    if groupby not in n.loads.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in n.loads columns: {list(n.loads.columns)}"
        )

    labels = (
        n.loads.loc[total_by_load.index, groupby]
        .fillna("unknown")
        .astype(str)
    )

    return total_by_load.groupby(labels).sum().sort_index()


def align_levels(series_by_scenario: dict[str, pd.Series]) -> pd.DataFrame:
    """Align series on a common index and return a wide table (index=type, columns=scenarios)."""
    df = pd.concat(series_by_scenario, axis=1).fillna(0.0)
    df.index.name = "type"
    # nicer stable ordering
    df = df.sort_index()
    return df


def delta_vs_base(levels: pd.DataFrame, base_col: str) -> pd.DataFrame:
    """Compute scenario - base for each column."""
    if base_col not in levels.columns:
        raise ValueError(f"Base column '{base_col}' not in levels table columns.")
    return levels.sub(levels[base_col], axis=0)


def relchg_vs_base(levels: pd.DataFrame, base_col: str, eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute (scenario - base)/|base| (dimensionless).
    Uses eps to avoid division by zero.
    """
    if base_col not in levels.columns:
        raise ValueError(f"Base column '{base_col}' not in levels table columns.")
    base = levels[base_col].to_numpy(dtype=float)
    denom = np.maximum(np.abs(base), eps)
    out = levels.sub(levels[base_col], axis=0).div(denom, axis=0)
    return out


# =========================
# MAIN
# =========================

def main():
    if not BASE_NETWORK_PATH.exists():
        raise FileNotFoundError(f"BASE_NETWORK_PATH not found: {BASE_NETWORK_PATH}")
    if not PREFIX_DIR.exists():
        raise FileNotFoundError(f"PREFIX_DIR not found: {PREFIX_DIR}")

    # --- base ---
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = read_network(BASE_NETWORK_PATH)
    s_base = total_demand_by_type(n_base, groupby=GROUPBY).rename(BASE_NAME)

    # --- scenarios ---
    scenario_paths = find_scenario_networks(PREFIX_DIR, NETWORK_PICKER)
    print(f"Found {len(scenario_paths)} scenario networks under: {PREFIX_DIR}")

    series_by_scenario: dict[str, pd.Series] = {BASE_NAME: s_base}

    for scen, path in scenario_paths.items():
        print(f"[SCENARIO={scen}] Loading: {path}")
        n = read_network(path)
        s = total_demand_by_type(n, groupby=GROUPBY).rename(scen)
        series_by_scenario[scen] = s

    # --- build tables ---
    levels = align_levels(series_by_scenario)
    delta = delta_vs_base(levels, BASE_NAME)
    relchg = relchg_vs_base(levels, BASE_NAME)

    # --- export ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_levels = OUT_DIR / f"{OUT_STEM}_levels.xlsx"
    out_delta = OUT_DIR / f"{OUT_STEM}_vs_base_delta.xlsx"
    out_rel = OUT_DIR / f"{OUT_STEM}_vs_base_relchg.xlsx"

    with pd.ExcelWriter(out_levels, engine="openpyxl") as w:
        levels.to_excel(w, sheet_name="levels")

    with pd.ExcelWriter(out_delta, engine="openpyxl") as w:
        delta.to_excel(w, sheet_name="vs_base_delta")

    with pd.ExcelWriter(out_rel, engine="openpyxl") as w:
        relchg.to_excel(w, sheet_name="vs_base_relchg")

    print("✔ Wrote:")
    print(f"  - {out_levels}")
    print(f"  - {out_delta}")
    print(f"  - {out_rel}")


if __name__ == "__main__":
    main()
