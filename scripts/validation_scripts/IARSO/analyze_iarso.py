#!/usr/bin/env python3
"""Probability sensitivity of deterministic validation decisions (costs in bn EUR/a)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT
    / "results/demand_uncertainty_2035/analysis_output/validation_heatmaps/validation_heatmaps.xlsx"
)
UNIT = "billion €/year"


def read_costs(path):
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    costs = sheets["total_cost"].apply(pd.to_numeric, errors="raise")
    if not costs.index.is_unique or not costs.columns.is_unique:
        raise ValueError("Candidate and scenario labels must be unique.")
    if "invalid_total_cost" in sheets:
        invalid = sheets["invalid_total_cost"].reindex_like(costs)
        if (
            invalid.isna().any().any()
            or not invalid.isin([True, False, 0, 1]).all().all()
        ):
            raise ValueError("Invalid or incomplete total-cost validity mask.")
        costs = costs.mask(invalid.astype(bool))
    if not np.isfinite(costs.to_numpy()).all():
        raise ValueError(
            "All costs must be finite and valid; probabilities are never renormalized over missing cases."
        )
    return costs


def probabilities(columns, base, start, step):
    end = 1 / len(columns)
    if base not in columns or len(columns) < 2:
        raise ValueError("Need BASE and at least one non-base scenario.")
    if not 0 < start <= end or step <= 0:
        raise ValueError("Require 0 < start <= equal probability and step > 0.")
    grid = np.arange(start, end, step)
    grid = np.append(grid[grid < end - 1e-12], end)
    weights = pd.DataFrame(
        np.repeat(grid[:, None], len(columns), axis=1), columns=columns
    )
    weights[base] = 1 - (len(columns) - 1) * grid
    weights.index = pd.Index(grid, name="nonbase_probability")
    return weights


def ties(values, labels, best=True):
    target = np.min(values) if best else np.max(values)
    return "; ".join(
        str(x) for x in labels[np.isclose(values, target, rtol=1e-10, atol=1e-8)]
    )


def envelope_intervals(costs, base, start, end):
    """Find exact best/worst intervals using all pairwise linear-cost crossings."""
    a = costs[base].to_numpy()
    b = costs.drop(columns=base).sum(axis=1).to_numpy() - (costs.shape[1] - 1) * a
    bounds = [start, end]
    for i in range(len(a)):
        for j in range(i):
            if b[i] != b[j]:
                crossing = (a[j] - a[i]) / (b[i] - b[j])
                if start < crossing < end:
                    bounds.append(crossing)
    bounds = sorted(set(bounds))
    rows = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        values = a + b * ((low + high) / 2)
        best, worst = ties(values, costs.index), ties(values, costs.index, False)
        if rows and (best, worst) == (
            rows[-1]["best_candidate"],
            rows[-1]["worst_candidate"],
        ):
            rows[-1]["p_end"] = high
        else:
            rows.append(
                dict(
                    p_start=low, p_end=high, best_candidate=best, worst_candidate=worst
                )
            )
    return pd.DataFrame(rows)


def analyze(costs, weights):
    c = costs.to_numpy()
    expected = pd.DataFrame(
        weights.to_numpy() @ c.T, index=weights.index, columns=costs.index
    )
    regret = costs.subtract(costs.min(axis=0), axis=1)
    expected_regret = pd.DataFrame(
        weights.to_numpy() @ regret.to_numpy().T,
        index=weights.index,
        columns=costs.index,
    )
    records = []
    distribution = []
    for p, w in weights.iterrows():
        for name, row in costs.iterrows():
            values = row.to_numpy()
            mean = expected.loc[p, name]
            order = np.argsort(values)
            cumulative = np.cumsum(w.to_numpy()[order])
            cumulative[-1] = 1.0
            quantiles = values[order][
                np.searchsorted(cumulative, [0.05, 0.25, 0.5, 0.75, 0.95])
            ]
            records.append(
                dict(
                    nonbase_probability=p,
                    candidate=name,
                    expected_cost=mean,
                    expected_regret=expected_regret.loc[p, name],
                    std_dev=np.sqrt(np.sum(w.to_numpy() * (values - mean) ** 2)),
                    q05=quantiles[0],
                    q25=quantiles[1],
                    median=quantiles[2],
                    q75=quantiles[3],
                    q95=quantiles[4],
                    min_cost=values.min(),
                    max_cost=values.max(),
                    best_operating_scenario=ties(values, costs.columns),
                    worst_operating_scenario=ties(values, costs.columns, False),
                )
            )
            for scenario, value, probability in zip(costs.columns, values, w):
                distribution.append(
                    dict(
                        nonbase_probability=p,
                        candidate=name,
                        operating_scenario=scenario,
                        cost=value,
                        probability=probability,
                        weighted_cost=value * probability,
                    )
                )
    summary = pd.DataFrame(
        {
            "base_probability": weights.max(axis=1),
            "best_candidate": [
                ties(row.to_numpy(), expected.columns) for _, row in expected.iterrows()
            ],
            "best_expected_cost": expected.min(axis=1),
            "worst_candidate": [
                ties(row.to_numpy(), expected.columns, False)
                for _, row in expected.iterrows()
            ],
            "worst_expected_cost": expected.max(axis=1),
        }
    )
    return (
        expected,
        regret,
        expected_regret,
        pd.DataFrame(records),
        pd.DataFrame(distribution),
        summary,
    )


def save_plot(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_results(costs, weights, expected, stats, summary, output):
    colors = dict(zip(costs.index, plt.get_cmap("tab20").colors))
    x = expected.index.to_numpy() * 100
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for name in expected:
        axes[0].plot(x, expected[name], label=name, color=colors[name], linewidth=2)
    axes[0].set_ylabel(f"Expected total cost ({UNIT})")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(
        "I-ARSO probability sensitivity: deterministic candidate decisions"
    )
    for key, marker in (("best_candidate", "o"), ("worst_candidate", "x")):
        y = [list(costs.index).index(name.split("; ")[0]) for name in summary[key]]
        axes[1].scatter(x, y, marker=marker, s=14, label=key.replace("_", " "))
    axes[1].set_yticks(range(len(costs)), costs.index, fontsize=7)
    axes[1].legend(loc="center right")
    axes[1].set_xlabel("Probability of each non-base scenario (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(x[0], x[-1])
    secondary = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda p: 100 - (len(costs.columns) - 1) * p,
            lambda p: (100 - p) / (len(costs.columns) - 1),
        ),
    )
    secondary.set_xlabel("BASE probability (%)")
    save_plot(fig, output, "expected_cost_sensitivity")

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    im = ax.imshow(expected.T, aspect="auto", cmap="viridis")
    ticks = np.unique(np.linspace(0, len(x) - 1, min(9, len(x))).astype(int))
    ax.set_xticks(ticks, [f"{x[t]:.2f}" for t in ticks])
    ax.set_yticks(range(len(costs)), costs.index)
    ax.set_xlabel("Probability of each non-base scenario (%)")
    ax.set_title("Expected cost by candidate and probability assumption")
    fig.colorbar(im, ax=ax, label=f"Expected cost ({UNIT})")
    save_plot(fig, output, "expected_cost_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, layout="constrained")
    for ax, p in zip(axes, [weights.index[0], weights.index[-1]]):
        subset = stats[stats.nonbase_probability == p].set_index("candidate")
        for y, name in enumerate(costs.index):
            s = subset.loc[name]
            ax.plot([s.q05, s.q95], [y, y], color=colors[name], linewidth=2)
            ax.plot([s.q25, s.q75], [y, y], color=colors[name], linewidth=7)
            ax.scatter(s["median"], y, color="black", marker="|", s=100, zorder=3)
            ax.scatter(s.expected_cost, y, color="#d55e00", marker="D", s=25, zorder=4)
        ax.set_title(f"Each non-base: {p:.4%}; BASE: {weights.loc[p].max():.4%}")
        ax.set_xlabel(f"Total cost ({UNIT})")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(costs)), costs.index)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Probability-weighted cost distributions\nThin: 5–95%; thick: 25–75%; black: median; orange diamond: expected cost"
    )
    save_plot(fig, output, "weighted_cost_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, p in zip(axes, [weights.index[0], weights.index[-1]]):
        for name, row in costs.iterrows():
            order = np.argsort(row.to_numpy())
            values = row.to_numpy()[order]
            cumulative = weights.loc[p].to_numpy()[order].cumsum()
            ax.step(
                np.r_[values[0], values],
                np.r_[0, cumulative],
                where="post",
                color=colors[name],
                label=name,
            )
        ax.set_title(f"Each non-base scenario: {p:.4%}")
        ax.set_xlabel(f"Total cost ({UNIT}; logarithmic scale)")
        if (costs.to_numpy() > 0).all():
            ax.set_xscale("log")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative probability")
    axes[1].legend(fontsize=7, ncol=2)
    save_plot(fig, output, "weighted_cost_cdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base", default="BASE")
    parser.add_argument(
        "--start",
        type=float,
        default=0.01,
        help="Initial probability per non-base scenario",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.001,
        help="Probability increment (default 0.1 percentage points)",
    )
    args = parser.parse_args()
    output = args.output_dir or args.input.parent / "IARSO"
    costs = read_costs(args.input)
    weights = probabilities(costs.columns, args.base, args.start, args.step)
    expected, regret, expected_regret, stats, distribution, summary = analyze(
        costs, weights
    )
    intervals = envelope_intervals(
        costs, args.base, weights.index[0], weights.index[-1]
    )
    # Independent identities detect orientation, normalization, and regret errors.
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(expected.iloc[-1], costs.mean(axis=1))
    np.testing.assert_allclose(
        expected.to_numpy() - expected_regret.to_numpy(),
        np.repeat(
            (weights.to_numpy() @ costs.min(axis=0).to_numpy())[:, None],
            len(costs),
            axis=1,
        ),
    )
    if (weights.to_numpy() < 0).any():
        raise AssertionError("Negative probability")
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(
        cost_matrix=costs,
        regret_matrix=regret,
        probabilities=weights,
        expected_cost=expected,
        expected_regret=expected_regret,
        candidate_rank=expected.rank(axis=1, method="min"),
        best_worst=summary,
        distribution_statistics=stats,
        cost_distribution=distribution,
        selection_intervals=intervals,
    )
    with pd.ExcelWriter(output / "iarso_analysis.xlsx") as writer:
        for name, table in tables.items():
            indexed = name not in {
                "distribution_statistics",
                "cost_distribution",
                "selection_intervals",
            }
            table.to_excel(writer, sheet_name=name, index=indexed)
            table.to_csv(output / f"{name}.csv", index=indexed)
    plot_results(costs, weights, expected, stats, summary, output)
    report = [
        "# I-ARSO-style probability sensitivity",
        "",
        f"Input: {args.input.resolve()}",
        "",
        "Rows are deterministic capacity decisions; columns are operating scenarios. All costs are billion EUR/year, as stored in the source workbook.",
        f"For {len(costs.columns) - 1} non-base scenarios, each has probability p and {args.base} has probability 1 - {len(costs.columns) - 1}p.",
        f"Sweep: {weights.index[0]:.6%} to {weights.index[-1]:.6%}, with {args.step:.6%} increments plus the exact equal-probability endpoint.",
        "Expected cost = sum_j p_j C_sj. Regret = C_sj - min_s C_sj; its expected value has exactly the same minimizers as expected cost.",
        "Regret references the best supplied candidate in each column, not a proven global stochastic optimum.",
        "Weighted quantiles use the inverse discrete CDF (no interpolation). Standard deviation is the probability-weighted population standard deviation.",
        "The distribution is over operating scenarios separately for each candidate; candidates themselves are not assigned probabilities.",
        "Best/worst candidate means lowest/highest expected cost. Best/worst operating scenarios are also reported separately for each candidate.",
        "Selection intervals use all exact pairwise crossings of the affine expected-cost functions, so transitions between grid points are not missed. Interval endpoints can have ties.",
        "No missing cases are dropped and no probability mass is renormalized. The original workbook is not modified.",
        "",
        "## Endpoint results",
        "",
        summary.iloc[[0, -1]].to_string(),
        "",
        "## Best/worst selection intervals",
        "",
        intervals.to_string(index=False),
        "",
        "The high off-diagonal cost magnitudes are inherited from the source workbook; this analysis does not revalidate the underlying network solves.",
    ]
    (output / "README.md").write_text("\n\n".join(report) + "\n")
    print(summary.iloc[[0, -1]].to_string())
    print(intervals.to_string(index=False))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
