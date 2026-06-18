#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Generate log-abs heatmaps from vs_base_consumption / vs_base_supply sheets.

Expected columns (at least):
- technology
- value____BASE__
- delta_value__<scenario> for multiple scenarios

Outputs (per kind: supply/consumption):
- heatmap_<kind>_levels_all.png
- heatmap_<kind>_levels_topN.png
- heatmap_<kind>_delta_all.png
- heatmap_<kind>_delta_topN.png
- top_<kind>.csv  (ranking by variability metrics)

All heatmaps show:
- technologies on x-axis
- BASE/scenarios on y-axis
- values displayed with transformed scale

Requirements:
    pip install pandas openpyxl matplotlib
"""

from pathlib import Path
import zipfile
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

EXCEL_PATH = Path("results/eth_results/csvs/analysis_networks_vs_base.xlsx")
OUT_DIR = Path("results/eth_results/heatmaps_vs_base_out/eth_taglie")
ZIP_NAME = Path("results/eth_results/heatmaps_vs_base.zip")



SHEETS = {
    "consumption": "vs_base_consumption",
    "supply": "vs_base_supply",
}

TOP_N = 40

# -------------------------
# Scenario filtering
# -------------------------
SHOW_BASE_IN_HEATMAP = False

# If INCLUDED_SCENARIOS is not None, only these scenarios are kept.
# Names must match the suffix after "delta_value__".
INCLUDED_SCENARIOS = None
# Example:
# INCLUDED_SCENARIOS = {"agriculture_elec", "electricity_high"}

# These scenarios are removed if present.
EXCLUDED_SCENARIOS = {"__BASE__"}
# Example:
# EXCLUDED_SCENARIOS = {"stochastic_network", "test_case"}

# -------------------------
# Unit scaling
# -------------------------
# Input values are assumed to be in MWh.
# They are internally converted to TWh for ranking, filtering, CSV, and plotting.
ENABLE_UNIT_SCALING = True
VALUE_SCALE = 1e6
VALUE_UNIT_LABEL = "TWh"

# -------------------------
# Technology filtering
# -------------------------
# Remove technologies whose mean absolute delta across kept scenarios
# is lower than MIN_MEAN_ABS_DELTA_THRESHOLD.
# Threshold is interpreted in TWh when ENABLE_UNIT_SCALING = True.
ENABLE_MIN_MEAN_ABS_DELTA_FILTER = True
MIN_MEAN_ABS_DELTA_THRESHOLD = 1.0

# Remove technologies whose name contains any of these substrings.
# Example: ["discharger", "dummy", "backup"]
ENABLE_TECHNOLOGY_SUBSTRING_EXCLUSION = True
EXCLUDED_TECHNOLOGY_SUBSTRINGS = ["discharger"]

# Case-insensitive matching for technology substring exclusion.
TECHNOLOGY_SUBSTRING_CASE_INSENSITIVE = True

# -------------------------
# Technology ordering
# -------------------------
# Sort technologies by mean absolute delta (descending).
# Most variable technologies will be on the left.
SORT_TECHNOLOGIES_BY_MEAN_ABS_DELTA = True

# Fallback sorting metric if the previous flag is False:
# True  -> sort by max absolute delta descending
# False -> keep original aggregated order
SORT_TECHNOLOGIES_BY_MAX_ABS_DELTA = False

# -------------------------
# Scenario labels
# -------------------------
SHORTEN_SCENARIO_LABELS = False

# Optional explicit renaming for plot labels
SCENARIO_LABEL_MAP = {
    "base": "BASE",
    "agriculture_full_electric": "AFE",
    "agriculture_machinery_full_oil": "AMFO",
    "electricity_optimistic": "EO",
    "industry_h2": "IH2",
    "land_transport_linear_ev": "LTLEV",
    "shipping_full_methanol": "SFM",
    "urban_heat_full_central": "UHFC",
    "stochastic_network": "SP",
}

BASE_LABEL = "BASE"

# -------------------------
# Titles and labels
# -------------------------
GENERAL_TITLES = {
    "levels": "Levels",
    "delta": "Variation compared to the base scenario",
}

KIND_TITLES = {
    "consumption": "Consumption",
    "supply": "Supply",
}

AXIS_LABELS = {
    "x": "Technology",
    "y": "Scenario",
    "cbar": f"Transformed log scale ({VALUE_UNIT_LABEL})",
}

# Optional custom full titles per kind
# If provided, they override automatic title construction
CUSTOM_TITLES = {
    "consumption": {
        "levels": f"Energy consumption levels [{VALUE_UNIT_LABEL}]",
        "delta": f"Energy consumption compared to the base scenario [{VALUE_UNIT_LABEL}]",
    },
    "supply": {
        "levels": f"Energy supply levels [{VALUE_UNIT_LABEL}]",
        "delta": f"Energy supply compared to the base scenario [{VALUE_UNIT_LABEL}]",
    },
}

# File suffixes
LEVELS_TAG = "levels"
DELTA_TAG = "delta"

# =========================
# HELPERS
# =========================

DELTA_PREFIX = "delta_value__"
BASE_VALUE_COL = "value____BASE__"


def safe_label(s: str) -> str:
    """Remove characters that may break matplotlib text rendering."""
    s = str(s)
    return s.replace("$", "").replace("{", "").replace("}", "")


def scenario_from_delta_col(col: str) -> str:
    """Convert 'delta_value__scenario_name' -> 'scenario_name'."""
    return col.split("__", 1)[1]


def maybe_shorten(s: str) -> str:
    """Apply optional scenario label shortening/mapping."""
    s = safe_label(s)

    if s in SCENARIO_LABEL_MAP:
        return SCENARIO_LABEL_MAP[s]

    if not SHORTEN_SCENARIO_LABELS:
        return s

    return s.split("__")[-1] if "__" in s else s


def get_delta_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return delta columns and associated scenario names."""
    delta_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(DELTA_PREFIX)]
    scenarios = [scenario_from_delta_col(c) for c in delta_cols]
    return delta_cols, scenarios


def filter_delta_cols(
    delta_cols: list[str],
    scenarios: list[str],
    included_scenarios: set[str] | None = None,
    excluded_scenarios: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Filter delta columns according to included/excluded scenario sets.
    """
    if included_scenarios is not None:
        included_scenarios = set(included_scenarios)

    excluded_scenarios = set() if excluded_scenarios is None else set(excluded_scenarios)

    kept_cols = []
    kept_scenarios = []

    for col, sc in zip(delta_cols, scenarios):
        if included_scenarios is not None and sc not in included_scenarios:
            continue
        if sc in excluded_scenarios:
            continue

        kept_cols.append(col)
        kept_scenarios.append(sc)

    return kept_cols, kept_scenarios


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """Convert selected columns to numeric in-place."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def aggregate_technologies(
    df: pd.DataFrame,
    delta_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate values by technology after cleaning unnamed columns
    and coercing numeric values.
    """
    if BASE_VALUE_COL not in df.columns:
        raise KeyError(f"Missing column '{BASE_VALUE_COL}' in sheet.")
    if "technology" not in df.columns:
        raise KeyError("Missing column 'technology' in sheet.")

    tmp = df.copy()
    tmp = tmp[[c for c in tmp.columns if not (isinstance(c, str) and c.startswith("Unnamed:"))]].copy()

    cols_to_keep = ["technology", BASE_VALUE_COL] + delta_cols
    cols_to_keep = [c for c in cols_to_keep if c in tmp.columns]
    tmp = tmp[cols_to_keep].copy()

    tmp["technology"] = tmp["technology"].astype(str)
    coerce_numeric(tmp, [BASE_VALUE_COL] + delta_cols)

    agg = tmp.groupby("technology", as_index=True)[[BASE_VALUE_COL] + delta_cols].sum(min_count=1)
    return agg


def exclude_technologies_by_substrings(
    agg: pd.DataFrame,
    excluded_substrings: list[str] | None = None,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    """
    Remove technologies whose index contains any of the provided substrings.
    """
    if not excluded_substrings:
        return agg

    excluded_substrings = [str(x) for x in excluded_substrings if str(x).strip()]
    if not excluded_substrings:
        return agg

    idx = agg.index.astype(str)

    if case_insensitive:
        idx_cmp = idx.str.lower()
        excluded_cmp = [s.lower() for s in excluded_substrings]
    else:
        idx_cmp = idx
        excluded_cmp = excluded_substrings

    mask_excluded = pd.Series(False, index=agg.index)

    for sub in excluded_cmp:
        mask_excluded = mask_excluded | idx_cmp.str.contains(sub, regex=False)

    return agg.loc[~mask_excluded].copy()


def apply_unit_scaling(agg: pd.DataFrame, delta_cols: list[str]) -> pd.DataFrame:
    """
    Scale BASE and delta columns in the aggregated dataframe.
    Used to convert from MWh to TWh when requested.
    """
    if not ENABLE_UNIT_SCALING:
        return agg

    out = agg.copy()
    cols_to_scale = [BASE_VALUE_COL] + list(delta_cols)
    for c in cols_to_scale:
        if c in out.columns:
            out[c] = out[c] / VALUE_SCALE

    return out


def compute_variability_metrics(
    agg: pd.DataFrame,
    delta_cols: list[str],
) -> pd.DataFrame:
    """
    Compute variability metrics from aggregated delta columns.
    """
    if len(agg) == 0:
        return pd.DataFrame(
            columns=[
                "technology",
                "mean_abs_delta",
                "max_abs_delta",
                "sum_abs_delta",
                "nonzero_count",
            ]
        )

    if delta_cols:
        M = agg[delta_cols].to_numpy(dtype=float)
        mags = np.abs(M)
        mean_abs_delta = np.nanmean(mags, axis=1)
        max_abs_delta = np.nanmax(mags, axis=1)
        sum_abs_delta = np.nansum(mags, axis=1)
        nonzero_count = np.sum(np.nan_to_num(mags, nan=0.0) > 0.0, axis=1)
    else:
        mean_abs_delta = np.zeros(len(agg), dtype=float)
        max_abs_delta = np.zeros(len(agg), dtype=float)
        sum_abs_delta = np.zeros(len(agg), dtype=float)
        nonzero_count = np.zeros(len(agg), dtype=int)

    out = pd.DataFrame({
        "technology": agg.index.astype(str).map(safe_label),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "sum_abs_delta": sum_abs_delta,
        "nonzero_count": nonzero_count,
    })

    return out


def apply_min_mean_abs_delta_filter(
    agg: pd.DataFrame,
    delta_cols: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove technologies whose mean absolute delta is below threshold.
    Returns filtered aggregated dataframe and variability metrics.
    """
    metrics = compute_variability_metrics(agg, delta_cols)

    if len(metrics) == 0:
        return agg.copy(), metrics

    keep_mask = metrics["mean_abs_delta"] >= threshold
    kept_techs = metrics.loc[keep_mask, "technology"].astype(str).tolist()

    agg_filtered = agg.loc[agg.index.astype(str).isin(kept_techs)].copy()
    metrics_filtered = metrics.loc[keep_mask].copy()

    return agg_filtered, metrics_filtered


def rank_technologies(
    agg: pd.DataFrame,
    delta_cols: list[str],
) -> pd.DataFrame:
    """
    Rank technologies using variability metrics.
    """
    metrics = compute_variability_metrics(agg, delta_cols)

    if len(metrics) == 0:
        return metrics

    return metrics.sort_values(
        ["mean_abs_delta", "max_abs_delta", "sum_abs_delta"],
        ascending=False,
    ).reset_index(drop=True)


def build_matrices_from_agg(
    agg: pd.DataFrame,
    delta_cols: list[str],
    scenarios: list[str],
    show_base: bool = True,
) -> tuple[pd.Index, list[str], np.ndarray, np.ndarray]:
    """
    Build levels and deltas matrices starting from an already aggregated dataframe.

    Returns:
      tech_index: Index of technologies
      row_labels: [BASE, scenario1, ...] or [scenario1, ...]
      levels: matrix [n_tech x n_rows]
      deltas: matrix [n_tech x n_rows]
    """
    base = agg[BASE_VALUE_COL].to_numpy(dtype=float)
    deltas_only = agg[delta_cols].to_numpy(dtype=float)

    if show_base:
        levels = np.column_stack([base, base[:, None] + deltas_only])
        deltas = np.column_stack([np.zeros_like(base), deltas_only])
        row_labels = [safe_label(BASE_LABEL)] + [maybe_shorten(s) for s in scenarios]
    else:
        levels = base[:, None] + deltas_only
        deltas = deltas_only
        row_labels = [maybe_shorten(s) for s in scenarios]

    return agg.index, row_labels, levels, deltas


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
    Signed log transform: sign(x) * log10(1 + |x|).
    Ensures 0 -> 0.
    """
    M = np.asarray(M, dtype=float)
    return np.sign(M) * np.log10(1.0 + np.abs(M))


def get_plot_title(kind: str, mode: str) -> str:
    """
    Build plot title.

    mode must be one of:
    - "levels"
    - "delta"
    """
    if kind in CUSTOM_TITLES and mode in CUSTOM_TITLES[kind]:
        return CUSTOM_TITLES[kind][mode]

    kind_title = KIND_TITLES.get(kind, kind.capitalize())
    mode_title = GENERAL_TITLES[mode]
    return f"{kind_title} – {mode_title}"


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
    - columns = scenarios

    The plot is transposed so that:
    - x-axis = technologies
    - y-axis = scenarios
    """
    L_plot = L.T

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
    ax.set_xlabel(AXIS_LABELS["x"], fontweight="bold")
    ax.set_ylabel(AXIS_LABELS["y"], fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(AXIS_LABELS["cbar"], fontweight="bold")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_zip(folder: Path, zip_path: Path) -> None:
    """Zip all files inside folder."""
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

        delta_cols_all, scenarios_all = get_delta_cols(df)
        delta_cols_kept, scenarios_kept = filter_delta_cols(
            delta_cols_all,
            scenarios_all,
            included_scenarios=INCLUDED_SCENARIOS,
            excluded_scenarios=EXCLUDED_SCENARIOS,
        )

        if not delta_cols_kept:
            print(f"[WARNING] Sheet '{sheet}' ({kind}): no scenarios left after filtering. Skipping.")
            continue

        agg = aggregate_technologies(df, delta_cols_kept)

        n_before_substring_filter = len(agg)
        if ENABLE_TECHNOLOGY_SUBSTRING_EXCLUSION:
            agg = exclude_technologies_by_substrings(
                agg,
                excluded_substrings=EXCLUDED_TECHNOLOGY_SUBSTRINGS,
                case_insensitive=TECHNOLOGY_SUBSTRING_CASE_INSENSITIVE,
            )
        n_after_substring_filter = len(agg)

        # Convert from MWh to TWh before ranking/filtering/plotting
        agg = apply_unit_scaling(agg, delta_cols_kept)

        n_before_mean_filter = len(agg)
        if ENABLE_MIN_MEAN_ABS_DELTA_FILTER:
            agg, metrics_after_filter = apply_min_mean_abs_delta_filter(
                agg,
                delta_cols_kept,
                threshold=MIN_MEAN_ABS_DELTA_THRESHOLD,
            )
        else:
            metrics_after_filter = compute_variability_metrics(agg, delta_cols_kept)
        n_after_mean_filter = len(agg)

        if len(agg) == 0:
            print(f"[WARNING] Sheet '{sheet}' ({kind}): no technologies left after technology filtering. Skipping.")
            continue

        ranking = rank_technologies(agg, delta_cols_kept)
        ranking.to_csv(OUT_DIR / f"top_{kind}.csv", index=False)

        tech_index, row_labels, levels, deltas = build_matrices_from_agg(
            agg,
            delta_cols=delta_cols_kept,
            scenarios=scenarios_kept,
            show_base=SHOW_BASE_IN_HEATMAP,
        )

        if SORT_TECHNOLOGIES_BY_MEAN_ABS_DELTA:
            score = ranking.set_index("technology").reindex(tech_index.astype(str))["mean_abs_delta"].to_numpy(dtype=float)
            order = np.argsort(-np.nan_to_num(score, nan=0.0))
        elif SORT_TECHNOLOGIES_BY_MAX_ABS_DELTA:
            score = ranking.set_index("technology").reindex(tech_index.astype(str))["max_abs_delta"].to_numpy(dtype=float)
            order = np.argsort(-np.nan_to_num(score, nan=0.0))
        else:
            order = np.arange(len(tech_index))

        tech_sorted = [safe_label(t) for t in tech_index.to_numpy(dtype=str)[order]]
        levels_sorted = levels[order, :]
        deltas_sorted = deltas[order, :]

        levels_L = log1p_pos(levels_sorted)
        deltas_L = signed_log1p(deltas_sorted)

        title_levels = get_plot_title(kind, "levels")
        title_delta = get_plot_title(kind, "delta")

        # ALL
        heatmap_plot(
            levels_L,
            xlabels=tech_sorted,
            ylabels=row_labels,
            title=title_levels,
            out_png=OUT_DIR / f"heatmap_{kind}_{LEVELS_TAG}_all.png",
        )

        heatmap_plot(
            deltas_L,
            xlabels=tech_sorted,
            ylabels=row_labels,
            title=title_delta,
            out_png=OUT_DIR / f"heatmap_{kind}_{DELTA_TAG}_all.png",
        )

        # TOP-N
        top_n = min(TOP_N, len(tech_sorted))

        heatmap_plot(
            levels_L[:top_n, :],
            xlabels=tech_sorted[:top_n],
            ylabels=row_labels,
            title=title_levels,
            out_png=OUT_DIR / f"heatmap_{kind}_{LEVELS_TAG}_top{top_n}.png",
        )

        heatmap_plot(
            deltas_L[:top_n, :],
            xlabels=tech_sorted[:top_n],
            ylabels=row_labels,
            title=title_delta,
            out_png=OUT_DIR / f"heatmap_{kind}_{DELTA_TAG}_top{top_n}.png",
        )

        print(
            f"[OK] {kind}: kept {len(scenarios_kept)} scenarios "
            f"-> {', '.join(scenarios_kept)}"
        )
        print(
            f"     Technologies: {len(df['technology'].dropna().astype(str).unique())} raw unique "
            f"-> {n_before_substring_filter} aggregated "
            f"-> {n_after_substring_filter} after substring filter "
            f"-> {n_after_mean_filter} after mean-abs-delta filter"
        )

    save_zip(OUT_DIR, ZIP_NAME)
    print(f"Done. Outputs in: {OUT_DIR.resolve()}")
    print(f"Zip saved at: {ZIP_NAME.resolve()}")


if __name__ == "__main__":
    main()