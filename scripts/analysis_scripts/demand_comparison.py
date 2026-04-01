#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare total load demand by type across MANY PyPSA networks vs a chosen base network.

What it does:
- Scans a results prefix folder like: results/<PREFIX>/<scenario>/networks/*.nc
- Picks ONE network file per scenario (you can constrain by filename pattern)
- Computes total load demand using snapshot weights
- Uses per-load priority:
    1) loads_t.p
    2) loads_t.p_set
    3) loads.p_set expanded over snapshots
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

# Scenario folders to exclude from the batch analysis
EXCLUDED_SCENARIOS = {"base", "stochastic_network"}
# Example:
# EXCLUDED_SCENARIOS = {"base", "debug_case"}

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


def _filter_excluded_scenarios(
    scenario_paths: dict[str, Path],
    excluded_scenarios: set[str],
) -> dict[str, Path]:
    """Remove scenarios whose folder name is in excluded_scenarios."""
    if not excluded_scenarios:
        return scenario_paths

    excluded = set(excluded_scenarios)
    return {scen: path for scen, path in scenario_paths.items() if scen not in excluded}


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

def _get_snapshot_weights(n: pypsa.Network) -> pd.Series:
    """Return snapshot weights aligned to n.snapshots."""
    if hasattr(n, "snapshot_weightings") and "generators" in n.snapshot_weightings:
        return n.snapshot_weightings.generators.reindex(n.snapshots).fillna(1.0)
    return pd.Series(1.0, index=n.snapshots)


def total_demand_by_type(
    n: pypsa.Network,
    groupby: GroupByField = "carrier",
) -> pd.Series:
    """
    Total weighted load demand aggregated by a chosen load field.

    Priority per load:
      1) loads_t.p
      2) loads_t.p_set
      3) loads.p_set expanded over snapshots

    Returns weighted energy-like totals (e.g. MWh if power is MW and weights are hours).
    """
    if n.loads.empty:
        return pd.Series(dtype=float)

    weights = _get_snapshot_weights(n)

    # Start from static p_set expanded over snapshots
    static_p_set = n.loads["p_set"].reindex(n.loads.index).fillna(0.0)

    p_load = pd.DataFrame(
        np.tile(static_p_set.to_numpy(), (len(n.snapshots), 1)),
        index=n.snapshots,
        columns=n.loads.index,
    )

    # Overwrite with time-dependent p_set if available
    if hasattr(n.loads_t, "p_set") and n.loads_t.p_set is not None and not n.loads_t.p_set.empty:
        cols = p_load.columns.intersection(n.loads_t.p_set.columns)
        p_load.loc[:, cols] = n.loads_t.p_set.loc[:, cols]

    # Overwrite with actual p if available
    if hasattr(n.loads_t, "p") and n.loads_t.p is not None and not n.loads_t.p.empty:
        cols = p_load.columns.intersection(n.loads_t.p.columns)
        p_load.loc[:, cols] = n.loads_t.p.loc[:, cols]

    # Weighted sum over snapshots
    total_by_load = p_load.mul(weights, axis=0).sum(axis=0)

    if groupby == "load_name":
        out = total_by_load.copy()
        out.index = out.index.astype(str)
        return out.groupby(out.index).sum().sort_index()

    if groupby not in n.loads.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in n.loads columns: {list(n.loads.columns)}"
        )

    labels = n.loads.loc[total_by_load.index, groupby].fillna("unknown").astype(str)

    return total_by_load.groupby(labels).sum().sort_index()


def align_levels(series_by_scenario: dict[str, pd.Series]) -> pd.DataFrame:
    """Align series on a common index and return a wide table (index=type, columns=scenarios)."""
    df = pd.concat(series_by_scenario, axis=1).fillna(0.0)
    df.index.name = "type"
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
    print(f"[BASE] Snapshots: {len(n_base.snapshots)} | Weight sum: {_get_snapshot_weights(n_base).sum()}")
    s_base = total_demand_by_type(n_base, groupby=GROUPBY).rename(BASE_NAME)

    # --- scenarios ---
    scenario_paths = find_scenario_networks(PREFIX_DIR, NETWORK_PICKER)
    scenario_paths = _filter_excluded_scenarios(scenario_paths, EXCLUDED_SCENARIOS)

    if not scenario_paths:
        raise FileNotFoundError(
            f"No scenario networks left after excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}"
        )

    print(f"Found {len(scenario_paths)} scenario networks under: {PREFIX_DIR}")
    if EXCLUDED_SCENARIOS:
        print(f"[INFO] Excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}")

    series_by_scenario: dict[str, pd.Series] = {BASE_NAME: s_base}

    for scen, path in scenario_paths.items():
        print(f"[SCENARIO={scen}] Loading: {path}")
        n = read_network(path)
        print(f"[SCENARIO={scen}] Snapshots: {len(n.snapshots)} | Weight sum: {_get_snapshot_weights(n).sum()}")
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