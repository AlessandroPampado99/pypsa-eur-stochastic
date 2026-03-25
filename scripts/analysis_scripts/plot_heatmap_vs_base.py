#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:17:10 2026

@author: aless
"""


"""
Generate log-abs heatmaps from vs_base_consumption / vs_base_supply sheets.

Expected columns (at least):
- kind, group, technology
- value____BASE__
- delta_value__<scenario> for multiple scenarios

Outputs (per kind: supply/consumption):
- heatmap_<kind>_levels_all.png
- heatmap_<kind>_levels_top40.png
- heatmap_<kind>_delta_all.png
- heatmap_<kind>_delta_top40.png
- top_<kind>.csv  (ranking by max |delta|)

All heatmaps show:
- technologies on x-axis
- BASE/scenarios on y-axis
- values displayed with transformed scale

Requirements:
    pip install pandas openpyxl matplotlib
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

EXCEL_PATH = Path("results/eth_results/csvs/analysis_networks_vs_base.xlsx")  # <-- your vs_base excel
SHEETS = {
    "consumption": "vs_base_consumption",
    "supply": "vs_base_supply",
}

OUT_DIR = Path("results/eth_results/heatmaps_vs_base_out/eth_trasposto")
ZIP_NAME = Path("results/eth_results/heatmaps_vs_base.zip")

TOP_N = 40

# If scenarios have long names, you can shorten labels here
SHORTEN_SCENARIO_LABELS = False


# =========================
# HELPERS
# =========================

DELTA_PREFIX = "delta_value__"
BASE_VALUE_COL = "value____BASE__"


def safe_label(s: str) -> str:
    s = str(s)
    return s.replace("$", "").replace("{", "").replace("}", "")


def maybe_shorten(s: str) -> str:
    s = safe_label(s)
    if not SHORTEN_SCENARIO_LABELS:
        return s
    return s.split("__")[-1] if "__" in s else s


def scenario_from_delta_col(col: str) -> str:
    # "delta_value__agriculture_elec" -> "agriculture_elec"
    return col.split("__", 1)[1]


def get_delta_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    delta_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(DELTA_PREFIX)]
    scenarios = [scenario_from_delta_col(c) for c in delta_cols]
    return delta_cols, scenarios


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def rank_technologies_by_max_abs_delta(df: pd.DataFrame, delta_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    coerce_numeric(tmp, delta_cols + [BASE_VALUE_COL])
    agg = tmp.groupby("technology", as_index=False)[[BASE_VALUE_COL] + delta_cols].sum(min_count=1)

    M = agg[delta_cols].to_numpy(dtype=float)
    mags = np.abs(M)

    out = pd.DataFrame({
        "technology": agg["technology"].astype(str).map(safe_label),
        "max_abs_delta": np.nanmax(mags, axis=1),
        "sum_abs_delta": np.nansum(mags, axis=1),
        "nonzero_count": np.sum(np.nan_to_num(mags, nan=0.0) > 0.0, axis=1),
    }).sort_values(["max_abs_delta", "sum_abs_delta"], ascending=False)

    return out


def build_matrices(df: pd.DataFrame) -> tuple[pd.Index, list[str], np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      tech_index: Index of technologies
      row_labels: ["BASE", scenario1, ...]
      levels: matrix [n_tech x (1+n_scen)]  where levels[:,0]=base, levels[:,j]=base+delta_j
      deltas: matrix [n_tech x (1+n_scen)]  where deltas[:,0]=0,    deltas[:,j]=delta_j
      scenarios: list of scenario names (no BASE)
    """
    if BASE_VALUE_COL not in df.columns:
        raise KeyError(f"Missing column '{BASE_VALUE_COL}' in sheet.")

    delta_cols, scenarios = get_delta_cols(df)
    if not delta_cols:
        raise KeyError(f"No columns starting with '{DELTA_PREFIX}' found.")

    tmp = df.copy()
    tmp = tmp[[c for c in tmp.columns if not (isinstance(c, str) and c.startswith("Unnamed:"))]].copy()

    coerce_numeric(tmp, [BASE_VALUE_COL] + delta_cols)

    # Aggregate over group/kind etc. -> per technology
    agg = tmp.groupby("technology", as_index=True)[[BASE_VALUE_COL] + delta_cols].sum(min_count=1)

    base = agg[BASE_VALUE_COL].to_numpy(dtype=float)          # shape (n_tech,)
    deltas_only = agg[delta_cols].to_numpy(dtype=float)       # shape (n_tech, n_scen)

    # Build full matrices with BASE as first column
    levels = np.column_stack([base, base[:, None] + deltas_only])
    deltas = np.column_stack([np.zeros_like(base), deltas_only])

    row_labels = ["BASE"] + [maybe_shorten(s) for s in scenarios]

    return agg.index, row_labels, levels, deltas, scenarios


def log1p_pos(M: np.ndarray) -> np.ndarray:
    """
    For non-negative values: log10(1 + x).
    Ensures 0 -> 0 and never goes negative.
    """
    M = np.asarray(M, dtype=float)
    M = np.maximum(M, 0.0)
    return np.log10(1.0 + M)


def signed_log1p(M: np.ndarray) -> np.ndarray:
    """
    Signed log transform: sign(x)*log10(1 + |x|).
    Ensures 0 -> 0; negative values become negative with log-compressed magnitude.
    """
    M = np.asarray(M, dtype=float)
    return np.sign(M) * np.log10(1.0 + np.abs(M))


def heatmap_plot(
    L: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    out_png: Path,
):
    """
    Input matrix L is expected as:
    - rows = technologies
    - columns = scenarios (BASE + deltas)

    The plot is transposed so that:
    - x-axis = technologies
    - y-axis = scenarios
    """
    L_plot = L.T

    # More vertical space per scenario row
    row_height = 1.2
    fig_w = max(10, len(xlabels) * 0.35)
    fig_h = max(6, len(ylabels) * row_height)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(L_plot, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, ha="center")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontweight="bold", fontsize=11)

    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("Technology", fontweight="bold")
    ax.set_ylabel("Scenario", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transformed scale", fontweight="bold")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))


# =========================
# MAIN
# =========================

def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for kind, sheet in SHEETS.items():
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)

        # Ranking / TOP-N selection based on max |delta|
        delta_cols, _ = get_delta_cols(df)
        ranking = rank_technologies_by_max_abs_delta(df, delta_cols)
        ranking.to_csv(OUT_DIR / f"top_{kind}.csv", index=False)

        tech_index, row_labels, levels, deltas, scenarios = build_matrices(df)

        # Order technologies by max |delta| (descending)
        score = np.nanmax(np.abs(deltas[:, 1:]), axis=1)
        order = np.argsort(-np.nan_to_num(score, nan=0.0))

        tech_sorted = [safe_label(t) for t in tech_index.to_numpy(dtype=str)[order]]
        levels_sorted = levels[order, :]
        deltas_sorted = deltas[order, :]

        # Transform for plotting
        levels_L = log1p_pos(levels_sorted)
        deltas_L = signed_log1p(deltas_sorted)

        # ALL
        heatmap_plot(
            levels_L,
            xlabels=tech_sorted,
            ylabels=row_labels,
            title=f"{kind.capitalize()} – Levels",
            out_png=OUT_DIR / f"heatmap_{kind}_levels_all.png",
        )

        heatmap_plot(
            deltas_L,
            xlabels=tech_sorted,
            ylabels=row_labels,
            title=f"{kind.capitalize()} – Variation compared to the base scenario",
            out_png=OUT_DIR / f"heatmap_{kind}_delta_all.png",
        )

        # TOP-N
        top_n = min(TOP_N, len(tech_sorted))

        heatmap_plot(
            levels_L[:top_n, :],
            xlabels=tech_sorted[:top_n],
            ylabels=row_labels,
            title=f"{kind.capitalize()} – Levels",
            out_png=OUT_DIR / f"heatmap_{kind}_levels_top{top_n}.png",
        )

        heatmap_plot(
            deltas_L[:top_n, :],
            xlabels=tech_sorted[:top_n],
            ylabels=row_labels,
            title=f"{kind.capitalize()} – Variation compared to the base scenario",
            out_png=OUT_DIR / f"heatmap_{kind}_delta_top{top_n}.png",
        )

    save_zip(OUT_DIR, ZIP_NAME)
    print(f"Done. Outputs in: {OUT_DIR.resolve()}")
    print(f"Zip saved at: {ZIP_NAME.resolve()}")


if __name__ == "__main__":
    main()