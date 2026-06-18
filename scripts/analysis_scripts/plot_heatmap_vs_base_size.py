#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Flexible heatmap generator for PyPSA comparison Excel files.

Supported sheet styles:
1) vs_base-style sheets
   Example columns:
   - technology
   - value____BASE__
   - delta_value__scenarioA
   - delta_value__scenarioB
   ...

2) levels-style sheets
   Example columns:
   - component, carrier, metric
   - value____BASE__
   - value__scenarioA
   - value__scenarioB
   ...

The script automatically:
- detects identifier columns
- supports both "vs_base" and "levels" sheets
- excludes rows with infinite values
- allows scenario filtering
- builds x-labels from available identifier columns
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
# USER SETTINGS
# =========================

EXCEL_PATH = Path("results/eth_results/csvs/analysis_component_sizes_vs_base.xlsx")
OUT_DIR = Path("results/eth_results/heatmaps_flexible_out_power")
ZIP_NAME = Path("results/eth_results/heatmaps_flexible_out.zip")

# You can point to any sheet(s) here
SHEETS = {
    "sizes_by_component_carrier": "levels_by_component_carrier",
    # "sizes_vs_base_by_component_carrier": "vs_base_by_component_carrier",
    # "supply": "vs_base_supply",
    # "consumption": "vs_base_consumption",
}

TOP_N = 40

# -------------------------
# Scenario filtering
# -------------------------
INCLUDED_SCENARIOS = None
# Example:
# INCLUDED_SCENARIOS = {"industry_h2", "urban_heat_full_central"}

EXCLUDED_SCENARIOS = ["base"]
# Example:
# EXCLUDED_SCENARIOS = {"test_case"}

SHOW_BASE_IN_HEATMAP = False
BASE_LABEL = "BASE"

# -------------------------
# Optional row filtering
# -------------------------
INCLUDED_COMPONENTS = None
# Example:
# INCLUDED_COMPONENTS = {"Generator", "Link", "Store"}

EXCLUDED_COMPONENTS = {"Load"}
# Example:
# EXCLUDED_COMPONENTS = {"Load"}

INCLUDED_METRICS = {"power_final"}
# Example:
# INCLUDED_METRICS = {"power_final", "energy_final"}

EXCLUDED_METRICS = set()

INCLUDED_CARRIERS = None
EXCLUDED_CARRIERS = set()

# Optional substring exclusion on metadata columns
ENABLE_METADATA_SUBSTRING_EXCLUSION = False
METADATA_SUBSTRING_CASE_INSENSITIVE = True

EXCLUDED_COMPONENT_SUBSTRINGS = []
EXCLUDED_CARRIER_SUBSTRINGS = ["discharger"]
EXCLUDED_METRIC_SUBSTRINGS = []

# Optional exclusion on the final plotted x-label
ENABLE_ITEM_LABEL_SUBSTRING_EXCLUSION = False
ITEM_LABEL_SUBSTRING_CASE_INSENSITIVE = True
EXCLUDED_ITEM_LABEL_SUBSTRINGS = []
# Example:
# EXCLUDED_ITEM_LABEL_SUBSTRINGS = ["discharger", "dummy"]

# Exclude rows if any used numeric value is infinite
EXCLUDE_INFINITE_ROWS = True

# Drop all-zero rows after filtering
DROP_ALL_ZERO_ROWS = True
ZERO_TOL = 1e-12

# -------------------------
# Unit scaling
# -------------------------
DIVIDE_VALUES_BY_1E3 = True
VALUE_SCALE = 1e3
VALUE_UNIT_LABEL = "GW"

# -------------------------
# Variability filtering
# -------------------------
# Remove plotted items whose mean absolute delta across kept scenarios
# is smaller than MIN_MEAN_ABS_DELTA_THRESHOLD.
ENABLE_MIN_MEAN_ABS_DELTA_FILTER = True
MIN_MEAN_ABS_DELTA_THRESHOLD = 1.0

# -------------------------
# Ordering
# -------------------------
# Sort plotted items by mean absolute delta descending.
SORT_ITEMS_BY_MEAN_ABS_DELTA = True

# Fallback sorting metric if the previous flag is False.
SORT_ITEMS_BY_MAX_ABS_DELTA = False

# -------------------------
# Labels and titles
# -------------------------
SHORTEN_SCENARIO_LABELS = False
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

GENERAL_TITLES = {
    "levels": "Levels",
    "delta": "Variation compared to the base scenario",
}

KIND_TITLES = {
    "sizes_by_component_carrier": "Optimized sizes by component [GW]",
    "sizes_vs_base_by_component_carrier": "Optimized sizes by component [GW]",
    "supply": "Supply",
    "consumption": "Consumption",
}

CUSTOM_TITLES = {
    # "sizes_by_component_carrier": {
    #     "levels": "Installed capacities",
    #     "delta": "Difference from base",
    # }
}

# Columns to use for x-axis labels.
# If None, all identifier columns are used.
LABEL_COLUMNS = ["carrier"]
# LABEL_COLUMNS = ["technology"]
# LABEL_COLUMNS = ["component"]

AXIS_LABELS = {
    "x": "Item",
    "y": "Scenario",
    "cbar": "Transformed log scale",
}

# How to compose x labels from identifier columns
ID_LABEL_SEPARATOR = " | "


# =========================
# HELPERS
# =========================

BASE_VALUE_COL = "value____BASE__"
VALUE_PREFIX = "value__"
DELTA_PREFIX = "delta_value__"


def safe_label(s: str) -> str:
    """Remove characters that may disturb plotting."""
    s = str(s)
    return s.replace("$", "").replace("{", "").replace("}", "")


def maybe_shorten(s: str) -> str:
    """Apply optional scenario renaming/shortening."""
    s = safe_label(s)

    if s in SCENARIO_LABEL_MAP:
        return SCENARIO_LABEL_MAP[s]

    if not SHORTEN_SCENARIO_LABELS:
        return s

    return s.split("__")[-1] if "__" in s else s


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """Convert selected columns to numeric in-place."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def get_identifier_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect identifier columns automatically.

    We treat as identifier columns all columns that are not:
    - unnamed helper columns
    - value__...
    - delta_value__...
    - relchg_value__...
    """
    id_cols = []

    for c in df.columns:
        if not isinstance(c, str):
            continue
        if c.startswith("Unnamed:"):
            continue
        if c.startswith("value__"):
            continue
        if c.startswith("delta_value__"):
            continue
        if c.startswith("relchg_value__"):
            continue
        id_cols.append(c)

    return id_cols


def scenario_from_value_col(col: str) -> str:
    """Convert 'value__scenario_name' -> 'scenario_name'."""
    return col.split("__", 1)[1]


def scenario_from_delta_col(col: str) -> str:
    """Convert 'delta_value__scenario_name' -> 'scenario_name'."""
    return col.split("__", 1)[1]


def get_value_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return scenario value columns excluding BASE.
    """
    value_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith(VALUE_PREFIX) and c != BASE_VALUE_COL
    ]
    scenarios = [scenario_from_value_col(c) for c in value_cols]
    return value_cols, scenarios


def get_delta_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return delta columns excluding the weird delta_value____BASE__ if present.
    """
    delta_cols = [
        c for c in df.columns
        if isinstance(c, str)
        and c.startswith(DELTA_PREFIX)
        and c != "delta_value____BASE__"
    ]
    scenarios = [scenario_from_delta_col(c) for c in delta_cols]
    return delta_cols, scenarios


def filter_scenarios(
    cols: list[str],
    scenarios: list[str],
    included_scenarios: set[str] | None = None,
    excluded_scenarios: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Filter scenario columns."""
    if included_scenarios is not None:
        included_scenarios = set(included_scenarios)

    excluded_scenarios = set() if excluded_scenarios is None else set(excluded_scenarios)

    kept_cols = []
    kept_scenarios = []

    for col, sc in zip(cols, scenarios):
        if included_scenarios is not None and sc not in included_scenarios:
            continue
        if sc in excluded_scenarios:
            continue
        kept_cols.append(col)
        kept_scenarios.append(sc)

    return kept_cols, kept_scenarios


def contains_any_substring(
    s: str,
    substrings: list[str] | set[str] | tuple[str, ...] | None,
    case_insensitive: bool = True,
) -> bool:
    """Check whether a string contains any substring from a collection."""
    if not substrings:
        return False

    text = "" if pd.isna(s) else str(s)

    if case_insensitive:
        text_cmp = text.lower()
        substrings_cmp = [str(x).lower() for x in substrings if str(x).strip()]
    else:
        text_cmp = text
        substrings_cmp = [str(x) for x in substrings if str(x).strip()]

    return any(sub in text_cmp for sub in substrings_cmp)


def filter_rows_by_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Optional filtering by component / carrier / metric."""
    out = df.copy()

    if "component" in out.columns:
        if INCLUDED_COMPONENTS is not None:
            out = out[out["component"].isin(INCLUDED_COMPONENTS)]
        if EXCLUDED_COMPONENTS:
            out = out[~out["component"].isin(EXCLUDED_COMPONENTS)]

        if ENABLE_METADATA_SUBSTRING_EXCLUSION and EXCLUDED_COMPONENT_SUBSTRINGS:
            mask = out["component"].astype(str).map(
                lambda x: contains_any_substring(
                    x,
                    EXCLUDED_COMPONENT_SUBSTRINGS,
                    case_insensitive=METADATA_SUBSTRING_CASE_INSENSITIVE,
                )
            )
            out = out.loc[~mask].copy()

    if "metric" in out.columns:
        if INCLUDED_METRICS is not None:
            out = out[out["metric"].isin(INCLUDED_METRICS)]
        if EXCLUDED_METRICS:
            out = out[~out["metric"].isin(EXCLUDED_METRICS)]

        if ENABLE_METADATA_SUBSTRING_EXCLUSION and EXCLUDED_METRIC_SUBSTRINGS:
            mask = out["metric"].astype(str).map(
                lambda x: contains_any_substring(
                    x,
                    EXCLUDED_METRIC_SUBSTRINGS,
                    case_insensitive=METADATA_SUBSTRING_CASE_INSENSITIVE,
                )
            )
            out = out.loc[~mask].copy()

    if "carrier" in out.columns:
        if INCLUDED_CARRIERS is not None:
            out = out[out["carrier"].isin(INCLUDED_CARRIERS)]
        if EXCLUDED_CARRIERS:
            out = out[~out["carrier"].isin(EXCLUDED_CARRIERS)]

        if ENABLE_METADATA_SUBSTRING_EXCLUSION and EXCLUDED_CARRIER_SUBSTRINGS:
            mask = out["carrier"].astype(str).map(
                lambda x: contains_any_substring(
                    x,
                    EXCLUDED_CARRIER_SUBSTRINGS,
                    case_insensitive=METADATA_SUBSTRING_CASE_INSENSITIVE,
                )
            )
            out = out.loc[~mask].copy()

    return out


def drop_infinite_rows(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Remove rows containing +/-inf in any relevant numeric column.
    """
    out = df.copy()
    if not numeric_cols:
        return out

    A = out[numeric_cols].to_numpy(dtype=float)
    finite_mask = np.isfinite(A).all(axis=1)
    return out.loc[finite_mask].copy()


def drop_all_zero_rows(df: pd.DataFrame, numeric_cols: list[str], tol: float = 1e-12) -> pd.DataFrame:
    """Remove rows that are zero across all relevant numeric columns."""
    out = df.copy()
    if not numeric_cols:
        return out

    A = out[numeric_cols].to_numpy(dtype=float)
    keep_mask = np.any(np.abs(np.nan_to_num(A, nan=0.0)) > tol, axis=1)
    return out.loc[keep_mask].copy()


def build_item_label(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    """
    Create x-axis labels by joining selected identifier columns.

    Priority:
    1. LABEL_COLUMNS, if provided
    2. otherwise all detected identifier columns
    """
    if LABEL_COLUMNS is None:
        cols_for_label = id_cols
    else:
        cols_for_label = [c for c in LABEL_COLUMNS if c in df.columns]

    if not cols_for_label:
        return pd.Series(np.arange(len(df)).astype(str), index=df.index)

    parts = []
    for c in cols_for_label:
        parts.append(df[c].astype(str).fillna("").map(safe_label))

    label = parts[0].copy()
    for p in parts[1:]:
        label = label + ID_LABEL_SEPARATOR + p

    return label


def detect_sheet_mode(df: pd.DataFrame) -> str:
    """
    Detect whether a sheet is:
    - 'vs_base'
    - 'levels'
    """
    has_base = BASE_VALUE_COL in df.columns
    has_delta = any(isinstance(c, str) and c.startswith(DELTA_PREFIX) for c in df.columns)
    has_values = any(
        isinstance(c, str) and c.startswith(VALUE_PREFIX) and c != BASE_VALUE_COL
        for c in df.columns
    )

    if has_base and has_delta:
        return "vs_base"
    if has_base and has_values:
        return "levels"

    raise ValueError(
        "Could not detect sheet mode. Expected either:\n"
        "- value____BASE__ + delta_value__...\n"
        "- value____BASE__ + value__...\n"
    )


def compute_variability_metrics(
    item_index: pd.Index,
    deltas: np.ndarray,
) -> pd.DataFrame:
    """
    Compute variability metrics from delta matrix.
    """
    if deltas.size == 0:
        mags = np.zeros((len(item_index), 0))
    else:
        mags = np.abs(np.asarray(deltas, dtype=float))

    if mags.shape[1] == 0:
        mean_abs_delta = np.zeros(len(item_index), dtype=float)
        max_abs_delta = np.zeros(len(item_index), dtype=float)
        sum_abs_delta = np.zeros(len(item_index), dtype=float)
        nonzero_count = np.zeros(len(item_index), dtype=int)
    else:
        mean_abs_delta = np.nanmean(mags, axis=1)
        max_abs_delta = np.nanmax(mags, axis=1)
        sum_abs_delta = np.nansum(mags, axis=1)
        nonzero_count = np.sum(np.nan_to_num(mags, nan=0.0) > 0.0, axis=1)

    return pd.DataFrame({
        "item": item_index.astype(str),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "sum_abs_delta": sum_abs_delta,
        "nonzero_count": nonzero_count,
    })


def filter_items_by_label_substrings(
    item_index: pd.Index,
    levels: np.ndarray,
    deltas: np.ndarray,
    excluded_substrings: list[str] | None,
    case_insensitive: bool = True,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """
    Exclude plotted items based on substrings in the final item label.
    """
    if not excluded_substrings:
        return item_index, levels, deltas

    keep_mask = np.array([
        not contains_any_substring(lbl, excluded_substrings, case_insensitive=case_insensitive)
        for lbl in item_index.astype(str)
    ], dtype=bool)

    return item_index[keep_mask], levels[keep_mask, :], deltas[keep_mask, :]


def filter_items_by_min_mean_abs_delta(
    item_index: pd.Index,
    levels: np.ndarray,
    deltas: np.ndarray,
    threshold: float,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """
    Remove plotted items whose mean absolute delta is below threshold.
    """
    metrics = compute_variability_metrics(item_index, deltas)
    keep_mask = metrics["mean_abs_delta"].to_numpy(dtype=float) >= threshold

    return item_index[keep_mask], levels[keep_mask, :], deltas[keep_mask, :]


def build_matrices(
    df: pd.DataFrame,
    included_scenarios: set[str] | None = None,
    excluded_scenarios: set[str] | None = None,
    show_base: bool = True,
) -> tuple[pd.Index, list[str], np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Returns:
      item_index: index of labels
      row_labels: BASE + scenarios, or only scenarios
      levels: absolute levels matrix [n_items x n_rows]
      deltas: delta vs base matrix [n_items x n_rows] or [n_items x n_scen]
      scenarios: kept scenarios
      id_cols: identifier columns used to build labels
    """
    if BASE_VALUE_COL not in df.columns:
        raise KeyError(f"Missing required column '{BASE_VALUE_COL}'.")

    mode = detect_sheet_mode(df)
    id_cols = get_identifier_columns(df)

    tmp = df.copy()
    tmp = tmp[[c for c in tmp.columns if not (isinstance(c, str) and c.startswith("Unnamed:"))]].copy()
    tmp = filter_rows_by_metadata(tmp)

    if mode == "vs_base":
        delta_cols, scenarios = get_delta_cols(tmp)
        delta_cols, scenarios = filter_scenarios(
            delta_cols,
            scenarios,
            included_scenarios=included_scenarios,
            excluded_scenarios=excluded_scenarios,
        )
        if not delta_cols:
            raise ValueError("No delta scenario columns left after filtering.")

        numeric_cols = [BASE_VALUE_COL] + delta_cols
        coerce_numeric(tmp, numeric_cols)

        if EXCLUDE_INFINITE_ROWS:
            tmp = drop_infinite_rows(tmp, numeric_cols)

        if DROP_ALL_ZERO_ROWS:
            tmp = drop_all_zero_rows(tmp, numeric_cols, tol=ZERO_TOL)

        if tmp.empty:
            raise ValueError("No rows left after filtering/dropping infinite values.")

        labels = build_item_label(tmp, id_cols)
        tmp = tmp.assign(__item_label__=labels)

        agg = tmp.groupby("__item_label__", as_index=True)[numeric_cols].sum(min_count=1)

        base = agg[BASE_VALUE_COL].to_numpy(dtype=float)
        deltas_only = agg[delta_cols].to_numpy(dtype=float)
        levels_only = base[:, None] + deltas_only

        if DIVIDE_VALUES_BY_1E3:
            base = base / VALUE_SCALE
            deltas_only = deltas_only / VALUE_SCALE
            levels_only = levels_only / VALUE_SCALE

    elif mode == "levels":
        value_cols, scenarios = get_value_cols(tmp)
        value_cols, scenarios = filter_scenarios(
            value_cols,
            scenarios,
            included_scenarios=included_scenarios,
            excluded_scenarios=excluded_scenarios,
        )
        if not value_cols:
            raise ValueError("No scenario value columns left after filtering.")

        numeric_cols = [BASE_VALUE_COL] + value_cols
        coerce_numeric(tmp, numeric_cols)

        if EXCLUDE_INFINITE_ROWS:
            tmp = drop_infinite_rows(tmp, numeric_cols)

        if DROP_ALL_ZERO_ROWS:
            tmp = drop_all_zero_rows(tmp, numeric_cols, tol=ZERO_TOL)

        if tmp.empty:
            raise ValueError("No rows left after filtering/dropping infinite values.")

        labels = build_item_label(tmp, id_cols)
        tmp = tmp.assign(__item_label__=labels)

        agg = tmp.groupby("__item_label__", as_index=True)[numeric_cols].sum(min_count=1)

        base = agg[BASE_VALUE_COL].to_numpy(dtype=float)
        levels_only = agg[value_cols].to_numpy(dtype=float)
        deltas_only = levels_only - base[:, None]

        if DIVIDE_VALUES_BY_1E3:
            base = base / VALUE_SCALE
            levels_only = levels_only / VALUE_SCALE
            deltas_only = deltas_only / VALUE_SCALE

    else:
        raise RuntimeError(f"Unexpected mode: {mode}")

    if show_base:
        levels = np.column_stack([base, levels_only])
        deltas = np.column_stack([np.zeros_like(base), deltas_only])
        row_labels = [safe_label(BASE_LABEL)] + [maybe_shorten(s) for s in scenarios]
    else:
        levels = levels_only
        deltas = deltas_only
        row_labels = [maybe_shorten(s) for s in scenarios]

    return agg.index, row_labels, levels, deltas, scenarios, id_cols


def rank_items(
    item_index: pd.Index,
    deltas: np.ndarray,
) -> pd.DataFrame:
    """
    Rank plotted items by variability.
    """
    metrics = compute_variability_metrics(item_index, deltas)

    return metrics.sort_values(
        ["mean_abs_delta", "max_abs_delta", "sum_abs_delta"],
        ascending=False,
    ).reset_index(drop=True)


def log1p_pos(M: np.ndarray) -> np.ndarray:
    """
    For non-negative values: log10(1 + x).
    """
    M = np.asarray(M, dtype=float)
    M = np.maximum(M, 0.0)
    return np.log10(1.0 + M)


def signed_log1p(M: np.ndarray) -> np.ndarray:
    """
    Signed log transform: sign(x) * log10(1 + |x|).
    """
    M = np.asarray(M, dtype=float)
    return np.sign(M) * np.log10(1.0 + np.abs(M))


def get_plot_title(kind: str, mode: str) -> str:
    """
    Build plot title.
    mode in {'levels', 'delta'}
    """
    if kind in CUSTOM_TITLES and mode in CUSTOM_TITLES[kind]:
        return CUSTOM_TITLES[kind][mode]

    kind_title = KIND_TITLES.get(kind, kind)
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
    Plot heatmap with:
    - x-axis = items
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
        print(f"[INFO] Reading sheet: {sheet}")
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)

        try:
            item_index, row_labels, levels, deltas, scenarios, id_cols = build_matrices(
                df,
                included_scenarios=INCLUDED_SCENARIOS,
                excluded_scenarios=EXCLUDED_SCENARIOS,
                show_base=SHOW_BASE_IN_HEATMAP,
            )
        except Exception as e:
            print(f"[WARNING] Skipping sheet '{sheet}': {e}")
            continue

        n_initial = len(item_index)

        if ENABLE_ITEM_LABEL_SUBSTRING_EXCLUSION:
            item_index, levels, deltas = filter_items_by_label_substrings(
                item_index,
                levels,
                deltas,
                excluded_substrings=EXCLUDED_ITEM_LABEL_SUBSTRINGS,
                case_insensitive=ITEM_LABEL_SUBSTRING_CASE_INSENSITIVE,
            )

        n_after_label_filter = len(item_index)

        if ENABLE_MIN_MEAN_ABS_DELTA_FILTER:
            item_index, levels, deltas = filter_items_by_min_mean_abs_delta(
                item_index,
                levels,
                deltas,
                threshold=MIN_MEAN_ABS_DELTA_THRESHOLD,
            )

        n_after_mean_filter = len(item_index)

        if len(item_index) == 0:
            print(f"[WARNING] Skipping sheet '{sheet}': no items left after item filtering.")
            continue

        ranking = rank_items(item_index, deltas)
        ranking.to_csv(OUT_DIR / f"top_{kind}.csv", index=False)

        if SORT_ITEMS_BY_MEAN_ABS_DELTA:
            score = ranking.set_index("item").reindex(item_index.astype(str))["mean_abs_delta"].to_numpy(dtype=float)
            order = np.argsort(-np.nan_to_num(score, nan=0.0))
        elif SORT_ITEMS_BY_MAX_ABS_DELTA:
            score = ranking.set_index("item").reindex(item_index.astype(str))["max_abs_delta"].to_numpy(dtype=float)
            order = np.argsort(-np.nan_to_num(score, nan=0.0))
        else:
            order = np.arange(len(item_index))

        items_sorted = [safe_label(t) for t in item_index.to_numpy(dtype=str)[order]]
        levels_sorted = levels[order, :]
        deltas_sorted = deltas[order, :]

        levels_L = log1p_pos(levels_sorted)
        deltas_L = signed_log1p(deltas_sorted)

        title_levels = get_plot_title(kind, "levels")
        title_delta = get_plot_title(kind, "delta")

        heatmap_plot(
            levels_L,
            xlabels=items_sorted,
            ylabels=row_labels,
            title=title_levels,
            out_png=OUT_DIR / f"heatmap_{kind}_levels_all.png",
        )

        heatmap_plot(
            deltas_L,
            xlabels=items_sorted,
            ylabels=row_labels,
            title=title_delta,
            out_png=OUT_DIR / f"heatmap_{kind}_delta_all.png",
        )

        top_n = min(TOP_N, len(items_sorted))

        heatmap_plot(
            levels_L[:top_n, :],
            xlabels=items_sorted[:top_n],
            ylabels=row_labels,
            title=title_levels,
            out_png=OUT_DIR / f"heatmap_{kind}_levels_top{top_n}.png",
        )

        heatmap_plot(
            deltas_L[:top_n, :],
            xlabels=items_sorted[:top_n],
            ylabels=row_labels,
            title=title_delta,
            out_png=OUT_DIR / f"heatmap_{kind}_delta_top{top_n}.png",
        )

        print(
            f"[OK] {sheet}: {len(items_sorted)} items plotted, "
            f"{len(scenarios)} scenarios kept, id columns = {id_cols}"
        )
        print(
            f"     Items: {n_initial} before item filters "
            f"-> {n_after_label_filter} after label substring filter "
            f"-> {n_after_mean_filter} after mean-abs-delta filter"
        )

    save_zip(OUT_DIR, ZIP_NAME)
    print(f"Done. Outputs in: {OUT_DIR.resolve()}")
    print(f"Zip saved at: {ZIP_NAME.resolve()}")


if __name__ == "__main__":
    main()