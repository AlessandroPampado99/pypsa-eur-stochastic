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


def plot_distributions(values, summary, output, title):
    fig, (ax, var_ax) = plt.subplots(
        1,
        2,
        figsize=(16, max(5, 0.36 * len(values) + 2.4)),
        sharey=True,
        gridspec_kw={"width_ratios": [3.5, 1]},
        constrained_layout=True,
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
                    vert=False,
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
                logs,
                y + rng.uniform(-0.13, 0.13, len(data)),
                s=9,
                color=color,
                alpha=0.6,
            )
            ax.scatter(
                np.log10(summary.loc[solution, "mean"]),
                y,
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
                np.log10(reference), y, marker="D", s=30, color="#202020", zorder=5
            )
        variance = summary.loc[solution, "variance"]
        if pd.notna(variance):
            var_ax.barh(y, variance, height=0.6, color=color, alpha=0.8)
    labels = [
        f"{solution}  (n={count})"
        for solution, count in zip(values.index, summary["n_valid"])
    ]
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    low, high = ax.get_xlim()
    ticks = [
        m * 10.0**e
        for e in range(int(np.floor(low)), int(np.ceil(high)) + 1)
        for m in (1, 1.2, 1.5, 2, 2.5, 3, 5, 7.5)
        if low <= np.log10(m * 10.0**e) <= high
    ]
    if len(ticks) > 9:
        ticks = ticks[::2]
    ax.set_xticks(np.log10(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**x:,.0f}"))
    ax.set_xlabel(f"Total cost ({UNIT}; logarithmic axis)")
    ax.set_title("Validation-cost distribution")
    var_ax.set_title("Population variance")
    var_ax.set_xlabel(f"({UNIT})²")
    var_ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for axis in (ax, var_ax):
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            marker="D",
            color="#202020",
            linestyle="",
            label="Capacity-expansion cost (diagonal)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="#d55e00",
            linestyle="",
            label="Arithmetic mean over validation years",
        ),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8
    )
    fig.suptitle(
        title
        + "\nEqual weight per validation year; violin density estimated in log-cost space",
        fontsize=12,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


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
            ],
            "value": [
                str(args.input),
                ", ".join(columns),
                "Equal weight per validation year",
                "Population variance, ddof=0",
                "Costs: bn. EUR/a; variance: (bn. EUR/a)^2",
                "Workbook diagonal; expected total cost for stochastic solutions",
                "Excluded; counts reported per solution",
            ],
        }
    )
    with pd.ExcelWriter(out / "validation_cost_statistics.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary")
        values.to_excel(writer, sheet_name="validation_costs")
        metadata.to_excel(writer, sheet_name="method", index=False)
    plot_distributions(
        values,
        summary,
        out / "validation_cost_distributions",
        "Validation cost by capacity solution",
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
