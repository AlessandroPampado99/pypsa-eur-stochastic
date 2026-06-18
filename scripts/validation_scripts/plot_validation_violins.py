#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build violin plots for cross-scenario validation results from PyPSA-Eur networks.

For each scenario used as capacity source (capacities_from), the script collects
the metric obtained when operating that fixed-capacity system on all operation
scenarios (operations_on). The resulting distribution is shown as one violin.

Metrics:
- total cost = capex + opex
- load curtailment = generation from generators with carrier == "load"
- renewable curtailment = available renewable generation minus actual dispatch

Expected files:
- diagonal standard solved network:
    base_s_adm___2050.nc
- diagonal stochastic expected-value solved network:
    base_s_adm___2050__exp.nc
- off-diagonal validation solved network:
    base_s_adm___2050__cap-<cap_source>__op-<op_source>.nc
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT_DIR = Path("results/eth_results")
OUTPUT_DIR = Path("results/eth_results/validation_violins_nolog")

DIAGONAL_FILENAME = "base_s_adm___2050.nc"
DIAGONAL_STOCHASTIC_FILENAME = "base_s_adm___2050__exp.nc"
CROSS_FILENAME_TEMPLATE = "base_s_adm___2050__cap-{cap_source}__op-{op_source}.nc"

EXCLUDED_SCENARIOS = set()
INCLUDE_STOCHASTIC_SCENARIOS = True
STOCHASTIC_SCENARIOS = {"stochastic_network"}

SCENARIO_ORDER = [
    "base",
    "agriculture_full_electric",
    "agriculture_machinery_full_oil",
    "electricity_optimistic",
    "industry_h2",
    "land_transport_linear_ev",
    "shipping_full_methanol",
    "urban_heat_full_central",
    "stochastic_network",
]

SCENARIO_LABELS = {
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

# Cost definition requested by user
COST_SCALE = 1e9
COST_UNIT = "bn. €/a"

LOAD_CURTAILMENT_CARRIERS = {"load"}
LOAD_CURTAILMENT_SCALE = 1e6
LOAD_CURTAILMENT_UNIT = "TWh"

RENEWABLE_CARRIERS = {
    "solar",
    "solar rooftop",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "ror",
}
RES_CURTAILMENT_SCALE = 1e6
RES_CURTAILMENT_UNIT = "TWh"

ALLOW_MISSING_FILES = True

# Plot settings
FIGSIZE = (11, 7)
DPI = 240
SHOW_SCATTER_POINTS = True
SHOW_MEAN_MARKER = True
SHOW_DIAGONAL_MARKER = True

SCATTER_JITTER = 0.07
SCATTER_SIZE = 28
DIAGONAL_MARKER_SIZE = 70
MEAN_MARKER_SIZE = 45

VIOLIN_ALPHA = 0.28
SCATTER_ALPHA = 0.9
USE_LOG_SCALE_FOR_COST = False

INVALID_COST_ATOL = 1e-12

DIAGONAL_LINE_COLOR = "black"
DIAGONAL_LINE_WIDTH = 3.0

MEAN_LINE_COLOR = "darkorange"
MEAN_LINE_WIDTH = 2.5

MEAN_MARKER_COLOR = "darkorange"
DIAGONAL_MARKER_COLOR = "black"

SHOW_REFERENCE_TEXT = True
REFERENCE_TEXT_ONLY_FOR_TOTAL_COST = True

MEAN_TEXT_COLOR = "darkorange"
DIAGONAL_TEXT_COLOR = "black"
REFERENCE_TEXT_FONTSIZE = 7

REFERENCE_TEXT_Y_AXES = 0.001
REFERENCE_TEXT_LINE_SPACING_AXES = 0.020

# =========================
# INTERNAL HELPERS
# =========================

def _format_reference_value(val: float) -> str:
    """
    Format reference values shown below each violin.
    """
    if pd.isna(val):
        return "NaN"
    return f"{val:.1f}"

def _build_scenario_colors(scenarios: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """
    Build one consistent color per scenario using a matplotlib categorical colormap.
    """
    cmap = plt.get_cmap("tab10")
    return {sc: cmap(i % 10) for i, sc in enumerate(scenarios)}


def _with_alpha(color, alpha: float):
    """
    Return the same color with a different alpha.
    """
    rgba = list(mcolors.to_rgba(color))
    rgba[3] = alpha
    return tuple(rgba)

def _scenario_display_name(name: str) -> str:
    """Return plot label for one scenario."""
    return SCENARIO_LABELS.get(name, name)


def _list_scenarios(root_dir: Path) -> list[str]:
    """List scenario folders under ROOT_DIR."""
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root_dir}")

    scenarios = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "networks").exists():
            continue
        scenarios.append(p.name)

    if EXCLUDED_SCENARIOS:
        scenarios = [s for s in scenarios if s not in EXCLUDED_SCENARIOS]

    if not INCLUDE_STOCHASTIC_SCENARIOS:
        scenarios = [s for s in scenarios if s not in STOCHASTIC_SCENARIOS]

    if SCENARIO_ORDER is not None:
        ordered = [s for s in SCENARIO_ORDER if s in scenarios]
        leftovers = [s for s in scenarios if s not in ordered]
        scenarios = ordered + leftovers

    if not scenarios:
        raise ValueError("No scenarios found after applying filters.")

    return scenarios


def _diagonal_network_path(root_dir: Path, scenario: str) -> Path:
    """Return diagonal solved network path for one scenario."""
    fname = (
        DIAGONAL_STOCHASTIC_FILENAME
        if scenario in STOCHASTIC_SCENARIOS
        else DIAGONAL_FILENAME
    )
    return root_dir / scenario / "networks" / fname


def _cross_network_path(root_dir: Path, cap_source: str, op_source: str) -> Path:
    """Return validation network path."""
    fname = CROSS_FILENAME_TEMPLATE.format(
        cap_source=cap_source,
        op_source=op_source,
    )
    return root_dir / cap_source / "networks" / fname


def _pair_network_path(root_dir: Path, cap_source: str, op_source: str) -> Path:
    """Return network path for a cap/op pair."""
    if cap_source == op_source:
        return _diagonal_network_path(root_dir, cap_source)
    return _cross_network_path(root_dir, cap_source, op_source)


def _get_snapshot_weightings(n: pypsa.Network) -> pd.Series:
    """Return a robust weighting series for time aggregation."""
    sw = n.snapshot_weightings

    if isinstance(sw, pd.DataFrame):
        for col in ["objective", "generators", "stores"]:
            if col in sw.columns:
                return sw[col]
        return sw.iloc[:, 0]

    if isinstance(sw, pd.Series):
        return sw

    raise TypeError(f"Unsupported snapshot_weightings type: {type(sw)}")


def _safe_load_network(path: Path) -> pypsa.Network | None:
    """Load a network if present."""
    if not path.exists():
        if ALLOW_MISSING_FILES:
            warnings.warn(f"Missing network file: {path}")
            return None
        raise FileNotFoundError(f"Missing network file: {path}")
    return pypsa.Network(str(path))


def _compute_total_cost(n: pypsa.Network) -> float:
    """
    Compute total cost as capex + opex.

    This intentionally avoids n.objective because fixed-capacity validation
    can make objective_constant comparisons inconsistent.

    If total cost is zero (or numerically close to zero), treat the run as
    invalid and return NaN so that it is excluded from the violin plot.
    """
    capex = float(n.statistics.capex().sum())
    opex = float(n.statistics.opex().sum())
    total = capex + opex

    if not np.isfinite(total):
        return np.nan

    if np.isclose(total, 0.0, atol=INVALID_COST_ATOL, rtol=0.0):
        return np.nan

    return total


def _compute_load_curtailment(n: pypsa.Network) -> float:
    """
    Compute annual load curtailment as generation from generators
    with carrier in LOAD_CURTAILMENT_CARRIERS.
    """
    if n.generators.empty:
        return 0.0

    gens = n.generators.index[n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)]
    if len(gens) == 0:
        return 0.0

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return 0.0

    p = n.generators_t.p.reindex(columns=gens, fill_value=0.0)
    w = _get_snapshot_weightings(n).reindex(p.index)

    total = p.mul(w, axis=0).sum().sum()
    return float(total)


def _get_generator_nominal_power(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    """Return p_nom_opt if available, otherwise p_nom."""
    gdf = n.generators.loc[gens]

    if "p_nom_opt" in gdf.columns:
        p_nom = gdf["p_nom_opt"].fillna(gdf["p_nom"])
    else:
        p_nom = gdf["p_nom"]

    return p_nom.astype(float)


def _compute_renewable_curtailment(n: pypsa.Network) -> float:
    """
    Compute renewable curtailment as available minus actual dispatch.
    """
    if n.generators.empty:
        return 0.0

    gens = n.generators.index[n.generators.carrier.isin(RENEWABLE_CARRIERS)]
    if len(gens) == 0:
        return 0.0

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return 0.0
    if not hasattr(n.generators_t, "p_max_pu"):
        return 0.0

    p = n.generators_t.p.reindex(columns=gens, fill_value=0.0)
    p_max_pu = n.generators_t.p_max_pu.reindex(
        index=p.index, columns=gens, fill_value=0.0
    )
    p_nom = _get_generator_nominal_power(n, gens)

    available = p_max_pu.mul(p_nom, axis=1)
    curtailed = (available - p).clip(lower=0.0)

    w = _get_snapshot_weightings(n).reindex(curtailed.index)
    total = curtailed.mul(w, axis=0).sum().sum()

    return float(total)


def _extract_metrics(n: pypsa.Network) -> dict[str, float]:
    """Extract all required metrics from a network."""
    total_cost = _compute_total_cost(n)

    if pd.isna(total_cost):
        return {
            "total_cost": np.nan,
            "load_curtailment": np.nan,
            "renewable_curtailment": np.nan,
        }

    return {
        "total_cost": total_cost / COST_SCALE,
        "load_curtailment": _compute_load_curtailment(n) / LOAD_CURTAILMENT_SCALE,
        "renewable_curtailment": _compute_renewable_curtailment(n) / RES_CURTAILMENT_SCALE,
    }


def _collect_validation_records(root_dir: Path, scenarios: list[str]) -> pd.DataFrame:
    """
    Build a long dataframe with one row per (cap_source, op_source).
    """
    rows = []

    for cap_source in scenarios:
        for op_source in scenarios:
            path = _pair_network_path(root_dir, cap_source, op_source)
            print(f"[INFO] Loading {cap_source} vs {op_source}: {path}")

            n = _safe_load_network(path)
            if n is None:
                rows.append(
                    {
                        "cap_source": cap_source,
                        "op_source": op_source,
                        "is_diagonal": cap_source == op_source,
                        "total_cost": np.nan,
                        "load_curtailment": np.nan,
                        "renewable_curtailment": np.nan,
                    }
                )
                continue

            vals = _extract_metrics(n)
            rows.append(
                {
                    "cap_source": cap_source,
                    "op_source": op_source,
                    "is_diagonal": cap_source == op_source,
                    **vals,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No validation records could be built.")
    return out


def _metric_plot_config(metric_name: str) -> dict[str, str]:
    """Return title, unit, and output name for one metric."""
    cfg = {
        "total_cost": {
            "title": "Total cost",
            "unit": COST_UNIT,
            "filename": "violin_total_cost.png",
        },
        "load_curtailment": {
            "title": "Load curtailment",
            "unit": LOAD_CURTAILMENT_UNIT,
            "filename": "violin_load_curtailment.png",
        },
        "renewable_curtailment": {
            "title": "Renewable curtailment",
            "unit": RES_CURTAILMENT_UNIT,
            "filename": "violin_renewable_curtailment.png",
        },
    }
    return cfg[metric_name]


def _build_violin_data(
    df: pd.DataFrame,
    scenarios: list[str],
    metric_name: str,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Return:
    - list of arrays for violins (one per cap_source), excluding NaN values
    - list of diagonal values
    - list of mean values over operation scenarios
    """
    violin_data = []
    diagonal_values = []
    mean_values = []

    for cap_source in scenarios:
        sub = df[df["cap_source"] == cap_source].copy()

        vals = sub[metric_name].dropna().to_numpy(dtype=float)
        violin_data.append(vals)

        if len(vals) == 0:
            mean_values.append(np.nan)
        else:
            mean_values.append(float(np.mean(vals)))

        diag = sub.loc[sub["op_source"] == cap_source, metric_name]
        if diag.empty or pd.isna(diag.iloc[0]):
            diagonal_values.append(np.nan)
        else:
            diagonal_values.append(float(diag.iloc[0]))

    return violin_data, diagonal_values, mean_values


def _save_metric_excel(df: pd.DataFrame, output_excel: Path) -> None:
    """Write long table and wide tables to Excel."""
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="long", index=False)

        for metric in ["total_cost", "load_curtailment", "renewable_curtailment"]:
            wide = df.pivot(
                index="cap_source",
                columns="op_source",
                values=metric,
            )
            wide.to_excel(writer, sheet_name=metric[:31])


def _plot_violin_metric(
    df: pd.DataFrame,
    scenarios: list[str],
    metric_name: str,
    output_dir: Path,
    scenario_colors: dict[str, tuple[float, float, float, float]],
) -> None:
    """
    Plot one violin figure for one metric.

    Each violin corresponds to one cap_source and contains all values obtained
    by testing on all operation scenarios.
    """
    cfg = _metric_plot_config(metric_name)
    violin_data, diagonal_values, mean_values = _build_violin_data(
        df,
        scenarios,
        metric_name,
    )

    labels = [_scenario_display_name(s) for s in scenarios]
    positions = np.arange(1, len(scenarios) + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    valid_data = [x for x in violin_data if len(x) > 0]
    if not valid_data:
        raise ValueError(f"No valid data to plot for metric '{metric_name}'.")

    vp = ax.violinplot(
        violin_data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, scenario in zip(vp["bodies"], scenarios, strict=False):
        base_color = scenario_colors[scenario]
        body.set_facecolor(_with_alpha(base_color, VIOLIN_ALPHA))
        body.set_edgecolor(base_color)
        body.set_linewidth(1.0)

    # Scatter all raw values
    if SHOW_SCATTER_POINTS:
        rng = np.random.default_rng(12345)
        for x, vals, scenario in zip(positions, violin_data, scenarios, strict=False):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=scenario_colors[scenario],
                zorder=3,
            )

    mean_values = [
        np.nan if len(vals) == 0 else float(np.mean(vals))
        for vals in violin_data
    ]

    # Mark scenario mean
    if SHOW_MEAN_MARKER:
        for x, mean_val in zip(positions, mean_values, strict=False):
            if pd.isna(mean_val):
                continue

            ax.hlines(
                y=mean_val,
                xmin=x - 0.24,
                xmax=x + 0.24,
                colors=MEAN_LINE_COLOR,
                linewidth=MEAN_LINE_WIDTH,
                zorder=4,
            )

        ax.scatter(
            positions,
            mean_values,
            marker="D",
            s=MEAN_MARKER_SIZE,
            color=MEAN_MARKER_COLOR,
            zorder=5,
            label="Mean over operation scenarios",
        )

    if SHOW_DIAGONAL_MARKER:
        for x, diag_val in zip(positions, diagonal_values, strict=False):
            if pd.isna(diag_val):
                continue

            ax.hlines(
                y=diag_val,
                xmin=x - 0.30,
                xmax=x + 0.30,
                colors=DIAGONAL_LINE_COLOR,
                linewidth=DIAGONAL_LINE_WIDTH,
                zorder=6,
            )

        ax.scatter(
            positions,
            diagonal_values,
            marker="*",
            s=DIAGONAL_MARKER_SIZE,
            color=DIAGONAL_MARKER_COLOR,
            zorder=7,
            label="Own scenario",
        )

    if SHOW_REFERENCE_TEXT and (
        (metric_name == "total_cost") or (not REFERENCE_TEXT_ONLY_FOR_TOTAL_COST)
    ):
        y_diag_axes = REFERENCE_TEXT_Y_AXES
        y_mean_axes = REFERENCE_TEXT_Y_AXES + REFERENCE_TEXT_LINE_SPACING_AXES

        for x, mean_val, diag_val in zip(
            positions, mean_values, diagonal_values, strict=False
        ):
            mean_txt = f"μ={_format_reference_value(mean_val)}"
            diag_txt = f"D={_format_reference_value(diag_val)}"

            ax.text(
                x,
                y_mean_axes,
                mean_txt,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=REFERENCE_TEXT_FONTSIZE,
                color=MEAN_TEXT_COLOR,
                fontweight="bold",
                zorder=8,
            )

            ax.text(
                x,
                y_diag_axes,
                diag_txt,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=REFERENCE_TEXT_FONTSIZE,
                color=DIAGONAL_TEXT_COLOR,
                fontweight="bold",
                zorder=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight="bold")

    ax.set_xlabel("Capacities from", fontsize=12, fontweight="bold")
    ax.set_ylabel(cfg["unit"], fontsize=12, fontweight="bold")
    ax.set_title(
        f"{cfg['title']} across operation scenarios",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    if metric_name == "total_cost" and USE_LOG_SCALE_FOR_COST:
        positive_vals = df[metric_name].dropna()
        positive_vals = positive_vals[positive_vals > 0]

        if not positive_vals.empty:
            ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg["filename"]
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    scenarios = _list_scenarios(ROOT_DIR)

    print("[INFO] Scenarios used:")
    for s in scenarios:
        print(f"  - {s}")

    df = _collect_validation_records(ROOT_DIR, scenarios)
    scenario_colors = _build_scenario_colors(scenarios)

    output_excel = OUTPUT_DIR / "validation_violins.xlsx"
    _save_metric_excel(df, output_excel)

    for metric_name in ["total_cost", "load_curtailment", "renewable_curtailment"]:
        _plot_violin_metric(df, scenarios, metric_name, OUTPUT_DIR, scenario_colors)

    print(f"✔ Excel written to: {output_excel}")
    print(f"✔ Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ ERROR: {exc}", file=sys.stderr)
        raise