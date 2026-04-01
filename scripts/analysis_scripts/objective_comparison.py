#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot objective values across many PyPSA networks under a results prefix, INCLUDING a chosen base network.

Scans:
    results/<PREFIX>/<scenario>/networks/*.nc

Also loads:
    BASE_NETWORK_PATH  (can be anywhere)

Extracts:
    - n.objective
    - n.objective_constant
    - total = objective + objective_constant

Outputs:
    - 3 bar plots (PNG):
        1) objective_total
        2) objective
        3) objective_constant

Style:
    - bright consistent colors per scenario
    - black bar edges
    - bold title/axes
    - base highlighted with a dedicated color
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

# Folder containing scenario subfolders:
# results/<PREFIX>/<scenario>/networks/*.nc
PREFIX_DIR = Path("results/eth_results")

# Base network path (explicit, can be anywhere)
BASE_NETWORK_PATH = Path("results/eth_results/base/networks/base_s_adm___2050.nc")
BASE_NAME = "__BASE__"

# Optional: choose exactly which network file inside each scenario networks/ folder.
# - None: pick first .nc found (sorted)
# - "base_s_adm___2050.nc": exact filename match
# - "base_s_"            : substring match in filename
NETWORK_PICKER: Optional[str] = "base_s_adm___2050.nc"

# Optional: exclude some scenario subfolders by exact folder name
# Example:
# EXCLUDE_SCENARIOS = ["scenario_bad", "test_case_1"]
EXCLUDE_SCENARIOS: list[str] = ["base"]  # exclude "base" scenario since it's already loaded as BASE_NETWORK_PATH

# Output folder and filenames
OUT_DIR = Path("results/eth_results/_postprocess_objectives")
OUT_STEM = "objectives"

# If True, plot y-axis on log scale (only if all values > 0)
TRY_LOG = True

# If True, order bars by objective_total ascending (nice for eyeballing)
SORT_BY_TOTAL = False


# =========================
# STYLE
# =========================

BASE_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#8c564b",  # brown
    "#7f7f7f",  # gray
]

BASE_HIGHLIGHT_COLOR = "#111111"  # base = almost black


def safe_label(s: str) -> str:
    s = str(s)
    return s.replace("$", "").replace("{", "").replace("}", "")


def scenario_color_map(scenarios: list[str]) -> dict[str, str]:
    # Deterministic mapping based on list order
    return {sc: BASE_COLORS[i % len(BASE_COLORS)] for i, sc in enumerate(scenarios)}


# =========================
# SCANNING
# =========================

def find_scenario_networks(
    prefix_dir: Path,
    picker: Optional[str],
    exclude_scenarios: Optional[list[str]] = None,
) -> dict[str, Path]:
    """
    Return {scenario_name: network_path} scanning:
        prefix_dir/<scenario>/networks/*.nc

    Parameters
    ----------
    prefix_dir : Path
        Folder containing scenario subfolders.
    picker : Optional[str]
        Rule to select a specific .nc file inside each networks/ folder.
    exclude_scenarios : Optional[list[str]]
        Exact names of scenario folders to exclude from the scan.
    """
    if not prefix_dir.exists():
        raise FileNotFoundError(f"PREFIX_DIR not found: {prefix_dir}")

    exclude_set = set(exclude_scenarios or [])
    out: dict[str, Path] = {}

    for scenario_dir in sorted([p for p in prefix_dir.iterdir() if p.is_dir()]):
        if scenario_dir.name in exclude_set:
            print(f"[SKIP] Excluding scenario folder: {scenario_dir.name}")
            continue

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
            f"(picker={picker!r}, exclude_scenarios={exclude_scenarios})."
        )

    return out


# =========================
# OBJECTIVE EXTRACTION
# =========================

def read_network_nc(path: Path) -> pypsa.Network:
    n = pypsa.Network()
    n.import_from_netcdf(path)
    return n


def _get_float_attr(n: pypsa.Network, name: str) -> float:
    if not hasattr(n, name):
        return float("nan")
    val = getattr(n, name)
    try:
        if val is None:
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")


def extract_objectives(n: pypsa.Network) -> tuple[float, float]:
    """
    Returns (objective, objective_constant) as floats (NaN if missing).
    Best-effort fallbacks included.
    """
    obj = _get_float_attr(n, "objective")
    objc = _get_float_attr(n, "objective_constant")

    if np.isnan(obj):
        obj = _get_float_attr(n, "objective_value")
    if np.isnan(objc):
        objc = _get_float_attr(n, "objective_const")

    return obj, objc


# =========================
# PLOTTING
# =========================

def plot_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    colors: dict[str, str],
):
    scenarios = df["scenario"].tolist()
    values = df[value_col].to_numpy(dtype=float)

    fig_w = max(10, len(scenarios) * 0.40)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    x = np.arange(len(scenarios))
    bar_colors = []
    for s in scenarios:
        if s == BASE_NAME:
            bar_colors.append(BASE_HIGHLIGHT_COLOR)
        else:
            bar_colors.append(colors.get(s, "#cccccc"))

    ax.bar(
        x,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontweight="bold")

    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_xlabel("Scenario", fontweight="bold")
    ax.set_ylabel(value_col, fontweight="bold")

    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # Optional log scale (only if all finite and > 0)
    if TRY_LOG:
        finite = np.isfinite(values)
        if finite.any() and np.all(values[finite] > 0):
            ax.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# MAIN
# =========================

def main():
    if not BASE_NETWORK_PATH.exists():
        raise FileNotFoundError(f"BASE_NETWORK_PATH not found: {BASE_NETWORK_PATH}")

    scenario_paths = find_scenario_networks(
        PREFIX_DIR,
        NETWORK_PICKER,
        exclude_scenarios=EXCLUDE_SCENARIOS,
    )

    # Load base first
    print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
    n_base = read_network_nc(BASE_NETWORK_PATH)
    obj_b, objc_b = extract_objectives(n_base)
    total_b = obj_b + objc_b if (np.isfinite(obj_b) and np.isfinite(objc_b)) else float("nan")

    rows = [{
        "scenario": BASE_NAME,
        "objective": obj_b,
        "objective_constant": objc_b,
        "objective_total": total_b,
        "path": str(BASE_NETWORK_PATH),
    }]

    missing = []
    if not np.isfinite(obj_b) or not np.isfinite(objc_b):
        missing.append(BASE_NAME)

    # Load scenarios
    scenarios = sorted(scenario_paths.keys())
    print(f"Found {len(scenarios)} scenario networks under: {PREFIX_DIR}")

    for scen in scenarios:
        path = scenario_paths[scen]
        print(f"[SCENARIO={scen}] Loading: {path}")
        n = read_network_nc(path)

        obj, objc = extract_objectives(n)
        total = obj + objc if (np.isfinite(obj) and np.isfinite(objc)) else float("nan")

        if not np.isfinite(obj) or not np.isfinite(objc):
            missing.append(scen)

        rows.append({
            "scenario": scen,
            "objective": obj,
            "objective_constant": objc,
            "objective_total": total,
            "path": str(path),
        })

    df = pd.DataFrame(rows)

    # Optional sort
    if SORT_BY_TOTAL:
        df = df.sort_values("objective_total", ascending=True, na_position="last").reset_index(drop=True)
    else:
        # Keep base first, then alpha
        df_base = df[df["scenario"] == BASE_NAME]
        df_rest = df[df["scenario"] != BASE_NAME].sort_values("scenario")
        df = pd.concat([df_base, df_rest], ignore_index=True)

    # Colors: only for non-base scenarios
    non_base_scenarios = [s for s in df["scenario"].tolist() if s != BASE_NAME]
    colors = scenario_color_map(non_base_scenarios)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_bar(
        df=df,
        value_col="objective_total",
        title="Objective total (objective + objective_constant)",
        out_path=OUT_DIR / f"{OUT_STEM}_total.png",
        colors=colors,
    )
    plot_bar(
        df=df,
        value_col="objective",
        title="Objective (variable part)",
        out_path=OUT_DIR / f"{OUT_STEM}_objective.png",
        colors=colors,
    )
    plot_bar(
        df=df,
        value_col="objective_constant",
        title="Objective constant",
        out_path=OUT_DIR / f"{OUT_STEM}_objective_constant.png",
        colors=colors,
    )

    # Save a CSV summary for debugging
    df.to_csv(OUT_DIR / f"{OUT_STEM}_table.csv", index=False)

    if missing:
        print("⚠ Missing objective fields (NaN) for scenarios:")
        for s in missing:
            print(f"  - {s}")

    print(f"✔ Saved plots + table to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()