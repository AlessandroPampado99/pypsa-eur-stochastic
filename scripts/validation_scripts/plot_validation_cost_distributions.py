#!/usr/bin/env python3
"""Plot validation cost distributions and export their descriptive statistics.

Read the heatmap workbook without modifying it. Rows are capacity solutions;
columns matching d_YYYY are the equally weighted validation population. Diagonal
capacity-expansion costs are reference markers, not extra observations.
Variance and standard deviation use ddof=0 (population statistics).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    "results/cutouts_det_capexp_nols/analysis_output/validation_heatmaps/validation_heatmaps.xlsx"
)
UNIT = "bn. €/a"


def read_costs(path):
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    costs = sheets["total_cost"].apply(pd.to_numeric, errors="raise")
    if not costs.index.is_unique or not costs.columns.is_unique:
        raise ValueError("Workbook must have unique solution and scenario labels.")
    invalid = sheets.get("invalid_total_cost")
    if invalid is not None:
        costs = costs.mask(invalid.reindex_like(costs).fillna(False).astype(bool))
    return costs.replace([np.inf, -np.inf], np.nan)


def summarize(costs, columns):
    values = costs.loc[:, columns]
    summary = pd.DataFrame(index=values.index)
    summary.index.name = "capacity_solution"
    summary["n_valid"] = values.count(axis=1)
    summary["n_missing"] = len(columns) - summary["n_valid"]
    summary["expansion_cost"] = [
        costs.loc[s, s] if s in costs.columns else np.nan for s in costs.index
    ]
    summary["mean"] = values.mean(axis=1)
    summary["variance"] = values.var(axis=1, ddof=0)
    summary["std_dev"] = values.std(axis=1, ddof=0)
    summary["median"] = values.median(axis=1)
    summary["q25"] = values.quantile(0.25, axis=1)
    summary["q75"] = values.quantile(0.75, axis=1)
    summary["min"] = values.min(axis=1)
    summary["max"] = values.max(axis=1)
    summary["worst_validation_scenario"] = [
        row.idxmax() if row.notna().any() else None for _, row in values.iterrows()
    ]
    return values, summary


def plot_distributions(values, summary, output, title, *, by_year=False, shedding=None):
    fig, ax = plt.subplots(
        figsize=(max(9, 0.42 * len(values)), 7), constrained_layout=True
    )
    rng = np.random.default_rng(0)
    for y, (solution, row) in enumerate(values.iterrows()):
        data = row.dropna().to_numpy(dtype=float)
        color = "#287d8e" if str(solution).startswith("stochastic_") else "#7286a3"
        if len(data):
            if (data <= 0).any():
                raise ValueError(
                    f"Logarithmic cost plot requires positive costs: {solution}"
                )
            logs = np.log10(data)
            if len(data) > 1 and np.ptp(logs) > 1e-12:
                violin = ax.violinplot(
                    [logs],
                    positions=[y],
                    vert=True,
                    widths=0.75,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.3)
            ax.scatter(
                y + rng.uniform(-0.13, 0.13, len(data)),
                logs,
                s=9,
                color=color,
                alpha=0.6,
            )
            ax.scatter(
                y,
                np.log10(summary.loc[solution, "mean"]),
                marker="o",
                s=38,
                color="#d55e00",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        reference = summary.loc[solution, "expansion_cost"]
        if pd.notna(reference) and reference > 0:
            ax.scatter(
                y, np.log10(reference), marker="D", s=30, color="#202020", zorder=5
            )
        if by_year:
            stochastic_mean = summary.loc[solution, "stochastic_mean"]
            if pd.notna(stochastic_mean) and stochastic_mean > 0:
                ax.scatter(
                    y,
                    np.log10(stochastic_mean),
                    marker="s",
                    s=40,
                    facecolor="none",
                    edgecolor="#008a78",
                    linewidth=1.5,
                    zorder=6,
                )
    labels = [str(s).removeprefix("d_") for s in values.index]
    ax.set_xticks(range(len(values)), labels, rotation=90)
    ax.set_xlabel(
        "Validation weather year"
        if by_year
        else "Capacity solution (weather year / stochastic case)"
    )
    low, high = ax.get_ylim()
    ticks = [
        m * 10.0**e
        for e in range(int(np.floor(low)), int(np.ceil(high)) + 1)
        for m in (1, 1.2, 1.5, 2, 2.5, 3, 5, 7.5)
        if low <= np.log10(m * 10.0**e) <= high
    ]
    if len(ticks) > 9:
        ticks = ticks[::2]
    ax.set_yticks(np.log10(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**x:,.0f}"))
    ax.set_ylabel(f"Total cost ({UNIT}; logarithmic axis)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if shedding is not None:
        # Labels use an independent annotation row, not the cost-axis units.
        ax.set_ylim(low, high + 0.12 * (high - low))
        for x, solution in enumerate(values.index):
            mean = shedding.loc[solution]
            ax.text(
                x,
                0.96,
                f"{mean:.1f}" if pd.notna(mean) else "N/A",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=90,
                color="black",
                fontsize=8,
            )
    handles = [
        Line2D(
            [],
            [],
            marker="D",
            color="#202020",
            linestyle="",
            label="Same-year capacity-expansion cost"
            if by_year
            else "Capacity-expansion cost (diagonal)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="#d55e00",
            linestyle="",
            label="Mean of deterministic solutions"
            if by_year
            else "Arithmetic mean over validation years",
        ),
    ]
    if by_year:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="none",
                markeredgecolor="#008a78",
                markeredgewidth=1.5,
                linestyle="",
                label="Mean of stochastic solutions",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        fontsize=8,
        ncol=3 if by_year else 2,
    )
    fig.suptitle(
        title
        + (
            "\nViolins: deterministic solutions; equal weight per solution; density estimated in log-cost space"
            if by_year
            else "\nEqual weight per validation year; violin density estimated in log-cost space"
        ),
        fontsize=12,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_load_shedding(path, values, summary, out, *, by_year=False):
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    shedding = sheets["load_curtailment"].apply(pd.to_numeric, errors="raise")
    invalid = sheets.get("invalid_load_curtailment")
    if invalid is not None:
        shedding = shedding.mask(
            invalid.reindex_like(shedding).fillna(False).astype(bool)
        )
    # Match the violin population, including diagonals. For OP plots, average
    # each operating-year column across deterministic capacity solutions.
    if by_year:
        shedding = shedding.T
    shedding = shedding.loc[values.index, values.columns].replace(
        [np.inf, -np.inf], np.nan
    )
    means = shedding.mean(axis=1).rename("mean_total_load_shedding_TWh")
    ordered = means.sort_values(kind="stable", na_position="last")
    export = pd.DataFrame(
        {
            "mean_total_load_shedding_TWh": ordered,
            "n_valid_capacity_solutions"
            if by_year
            else "n_valid_years": shedding.count(axis=1).reindex(ordered.index),
        }
    )
    export.index.name = "operating_year" if by_year else "capacity_solution"
    export.to_csv(
        out
        / (
            "validation_op_mean_load_shedding.csv"
            if by_year
            else "validation_mean_load_shedding.csv"
        )
    )
    plot_distributions(
        values.loc[ordered.index],
        summary.loc[ordered.index],
        out
        / (
            "validation_year_cost_distributions_sorted_by_load_shedding"
            if by_year
            else "validation_cost_distributions_sorted_by_load_shedding"
        ),
        (
            "Validation cost by operating year (OP) — ascending mean total load shedding"
            if by_year
            else "Validation cost by capacity solution — ascending mean total load shedding"
        )
        + (
            "\nBlack numbers: mean load shedding across deterministic capacity solutions (TWh)"
            if by_year
            else "\nBlack numbers: mean total load shedding (TWh)"
        ),
        shedding=ordered,
        by_year=by_year,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.input.parent / "cost_distributions"
    out.mkdir(parents=True, exist_ok=True)
    costs = read_costs(args.input)
    columns = [c for c in costs.columns if re.fullmatch(r"d_\d{4}", str(c))]
    if not columns:
        raise ValueError("No d_YYYY validation columns found.")
    values, summary = summarize(costs, columns)
    plot_by_load_shedding(args.input, values, summary, out)
    deterministic = [s for s in costs.index if re.fullmatch(r"d_\d{4}", str(s))]
    stochastic = costs.index.str.startswith("stochastic_")
    yearly_values, yearly_summary = summarize(costs.T.loc[columns], deterministic)
    yearly_summary.index.name = "validation_year"
    yearly_summary = yearly_summary.rename(
        columns={"worst_validation_scenario": "worst_capacity_solution"}
    )
    yearly_summary["stochastic_mean"] = costs.loc[stochastic, columns].mean(axis=0)
    yearly_summary["n_stochastic_valid"] = costs.loc[stochastic, columns].count(axis=0)
    plot_by_load_shedding(args.input, yearly_values, yearly_summary, out, by_year=True)
    metadata = pd.DataFrame(
        {
            "setting": [
                "source",
                "validation_population",
                "weighting",
                "variance",
                "units",
                "expansion_marker",
                "missing_values",
                "by_year_plot",
            ],
            "value": [
                str(args.input),
                ", ".join(columns),
                "Equal weight per validation year",
                "Population variance, ddof=0",
                "Costs: bn. EUR/a; variance: (bn. EUR/a)^2",
                "Workbook diagonal; expected total cost for stochastic solutions",
                "Excluded; counts reported per solution",
                "One violin per validation year over deterministic solutions (including diagonal); equal-weight arithmetic means within deterministic and stochastic groups; same-year expansion marker from diagonal",
            ],
        }
    )
    with pd.ExcelWriter(out / "validation_cost_statistics.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary")
        values.to_excel(writer, sheet_name="validation_costs")
        metadata.to_excel(writer, sheet_name="method", index=False)
        yearly_summary.to_excel(writer, sheet_name="by_year_summary")
        yearly_values.to_excel(writer, sheet_name="by_year_deterministic_costs")
    plot_distributions(
        yearly_values,
        yearly_summary,
        out / "validation_year_cost_distributions",
        "Validation cost by weather year",
        by_year=True,
    )
    plot_distributions(
        values,
        summary,
        out / "validation_cost_distributions",
        "Validation cost by capacity solution",
    )
    for metric, label in [
        ("mean", "mean validation cost"),
        ("expansion_cost", "capacity-expansion cost"),
    ]:
        ordered = summary.sort_values(metric, kind="stable", na_position="last")
        plot_distributions(
            values.loc[ordered.index],
            ordered,
            out / f"validation_cost_distributions_sorted_by_{metric}",
            f"Validation cost by capacity solution — ascending {label}",
        )
    stochastic = values.index.str.startswith("stochastic_")
    if stochastic.any():
        plot_distributions(
            values.loc[stochastic],
            summary.loc[stochastic],
            out / "stochastic_cost_distributions",
            "Stochastic solutions across validation years",
        )
    print(f"Written cost distributions and statistics to {out}")


if __name__ == "__main__":
    main()
