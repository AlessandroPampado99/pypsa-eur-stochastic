# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:34:26 2026

@author: aless
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot demand levels (real values, not deltas) from an Excel produced by the batch demand compare.

Input:
- Excel file with sheet "levels" (index: type/carrier, columns: scenarios)
  Example file: demand_compare_levels.xlsx

Outputs:
- One bar plot per carrier/type
- Multi-panel figures grouping carriers (default: 6 per figure, 2x3)
- Consistent bright colors per scenario, black bar edges, bold titles/axes

Requirements:
    pip install pandas openpyxl matplotlib
"""

from pathlib import Path
import re
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

EXCEL_PATH = Path("/home/pampado/stochastic/pypsa-eur/results/eth_results/postprocess_demand_compare/demand_compare_levels.xlsx")   # <-- change if needed
SHEET_NAME = "levels"

OUT_DIR = Path("/home/pampado/stochastic/pypsa-eur/results/eth_results/postprocess_demand_compare/demand_level_plots")  # where to save plots

# If the base column exists and you want to exclude it from plots, set True
EXCLUDE_BASE = False
BASE_COLNAME = "__BASE__"

# Plot scaling: "linear" or "log"
YSCALE = "linear"

# Multi-panel settings
N_PER_FIG = 6          # how many carriers per multi-panel page
N_COLS = 3             # columns in multi-panel grid (rows computed automatically)

# Mega figure (all carriers in one figure) - usually too big, keep False unless you really want it
MAKE_MEGA_FIG = False

# Label formatting
ROTATE_XTICKS = 45
FONT_SIZE = 10

# =========================
# STYLE (bright palette)
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

def safe_label(s: str) -> str:
    """Avoid mathtext issues and keep labels plain."""
    s = str(s)
    return s.replace("$", "").replace("{", "").replace("}", "")

def sanitize_filename(s: str) -> str:
    """Make a filesystem-safe filename."""
    s = safe_label(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:180]

def scenario_color_map(scenarios: list[str]) -> dict[str, str]:
    """Assign a fixed bright color to each scenario (deterministic)."""
    return {sc: BASE_COLORS[i % len(BASE_COLORS)] for i, sc in enumerate(scenarios)}

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all values to numeric, keep NaN if not parseable."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def load_levels(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load levels sheet into a DataFrame with index as carrier/type."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Many exports put the index column as first column named "type" or "Unnamed: 0"
    if "type" in df.columns:
        df = df.set_index("type")
    elif df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])

    df.index = df.index.map(lambda x: safe_label(x))
    df.columns = [safe_label(c) for c in df.columns]

    if EXCLUDE_BASE and BASE_COLNAME in df.columns:
        df = df.drop(columns=[BASE_COLNAME])

    df = ensure_numeric(df).fillna(0.0)
    return df

def style_axes(ax: plt.Axes, title: str, ylabel: str):
    """Apply consistent styling to axes."""
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_xlabel("Scenario", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

def plot_one_carrier(
    carrier: str,
    series: pd.Series,
    colors: dict[str, str],
    out_path: Path,
):
    """Plot one carrier as a bar chart across scenarios."""
    scenarios = list(series.index)
    values = series.to_numpy(dtype=float)

    fig_w = max(10, len(scenarios) * 0.40)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    x = np.arange(len(scenarios))
    bar_colors = [colors[s] for s in scenarios]

    ax.bar(
        x, values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=ROTATE_XTICKS, ha="right", fontweight="bold", fontsize=FONT_SIZE)

    if YSCALE == "log":
        # log can't show <=0; if you have zeros, it will be problematic
        ax.set_yscale("log")

    style_axes(
        ax,
        title=f"Demand – {carrier}",
        ylabel="Total demand (sum over snapshots)"
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_carrier_grid(
    carriers: list[str],
    df: pd.DataFrame,
    colors: dict[str, str],
    out_path: Path,
    n_cols: int = 3,
):
    """Plot a grid of carriers (multi-panel) for quick comparison."""
    n = len(carriers)
    n_rows = math.ceil(n / n_cols)

    # Big figure: scale with number of rows
    fig_w = max(16, len(df.columns) * 0.35)  # depends also on number of scenarios
    fig_h = 4.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    scenarios = list(df.columns)
    x = np.arange(len(scenarios))
    bar_colors = [colors[s] for s in scenarios]

    for i, carrier in enumerate(carriers):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        values = df.loc[carrier, :].to_numpy(dtype=float)
        ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.8)

        ax.set_title(f"{carrier}", fontweight="bold", fontsize=12, pad=8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            scenarios,
            rotation=ROTATE_XTICKS,
            ha="right",
            fontweight="bold",
            fontsize=FONT_SIZE,
        )

        if YSCALE == "log":
            ax.set_yscale("log")

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel("Total demand", fontweight="bold")
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    # Turn off unused axes
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    fig.suptitle("Demand comparison (levels)", fontweight="bold", fontsize=16, y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    """Split list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def main():
    df = load_levels(EXCEL_PATH, SHEET_NAME)

    # Determine scenarios and colors
    scenarios = list(df.columns)
    colors = scenario_color_map(scenarios)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- per-carrier plots ---
    carriers = list(df.index)

    single_dir = OUT_DIR / "per_carrier"
    single_dir.mkdir(parents=True, exist_ok=True)

    for carrier in carriers:
        out_png = single_dir / f"{sanitize_filename(carrier)}.png"
        plot_one_carrier(
            carrier=carrier,
            series=df.loc[carrier, :],
            colors=colors,
            out_path=out_png,
        )

    print(f"✔ Saved {len(carriers)} per-carrier plots to: {single_dir}")

    # --- grouped multi-panel figures ---
    grid_dir = OUT_DIR / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_list(carriers, N_PER_FIG)
    for k, chunk in enumerate(chunks, start=1):
        out_png = grid_dir / f"grid_{k:02d}_carriers_{len(chunk)}.png"
        plot_carrier_grid(
            carriers=chunk,
            df=df,
            colors=colors,
            out_path=out_png,
            n_cols=N_COLS,
        )

    print(f"✔ Saved {len(chunks)} grid figures to: {grid_dir}")

    # --- mega figure (optional) ---
    if MAKE_MEGA_FIG:
        out_png = OUT_DIR / "MEGA_all_carriers.png"
        plot_carrier_grid(
            carriers=carriers,
            df=df,
            colors=colors,
            out_path=out_png,
            n_cols=N_COLS,
        )
        print(f"✔ Saved mega figure to: {out_png}")

    print(f"✔ Done. Output folder: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
