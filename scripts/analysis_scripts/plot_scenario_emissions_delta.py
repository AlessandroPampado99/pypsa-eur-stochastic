#!/usr/bin/env python3
"""Plot changes in the atmospheric CO2 balance using an exported carrier table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plot_scenario_energy_balance as balance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/demand_uncertainty_2035/analysis_output/graphs/scenario_energy_balance_1%/scenario_energy_balance_co2_by_technology.csv"),
    )
    parser.add_argument("--base", default="__BASE__")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--names", type=Path, default=Path("results/demand_uncertainty_2035/names.txt"))
    args = parser.parse_args()
    source = pd.read_csv(args.input, index_col="scenario")
    if args.base not in source.index:
        raise ValueError(f"Base scenario {args.base!r} is missing from {args.input}")
    if not source.index.is_unique or not np.isfinite(source.to_numpy()).all():
        raise ValueError("Expected unique scenarios and finite carrier values")
    delta = source.subtract(source.loc[args.base], axis="columns").drop(index=args.base)
    balance.SCENARIO_ORDER = delta.index.tolist()
    balance.SCENARIO_LABELS = balance.load_scenario_names(args.names)
    scenarios, families = balance.order_and_group_scenarios(delta.index)
    delta = delta.reindex(scenarios)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "scenario_energy_balance_co2_delta"
    delta.to_csv(f"{stem}_by_technology.csv")
    # Distinguish gross emissions from the net atmospheric balance (which
    # includes the atmospheric store and uptake entries in the original plot).
    gross = source.clip(lower=0).sum(axis=1)
    uptake = source.clip(upper=0).sum(axis=1)
    totals = pd.DataFrame({
        "gross_emissions_delta_MtCO2_per_year": gross - gross.loc[args.base],
        "uptake_and_store_delta_MtCO2_per_year": uptake - uptake.loc[args.base],
        "net_balance_delta_MtCO2_per_year": delta.sum(axis=1),
    }).reindex(scenarios)
    totals.to_csv(f"{stem}_totals.csv")
    # Omit numerical noise below 0.01 Mt/a from the figure only.
    table = delta.loc[:, delta.abs().max() >= 0.01]
    technologies = balance._technology_order(table.columns)
    colors = balance._colors(technologies, balance._load_yaml(balance.PLOTTING_YAML))
    x, ranges = balance._positions(scenarios, families)
    fig, ax = plt.subplots(figsize=(15, 8))
    positive_bottom = np.zeros(len(scenarios))
    negative_bottom = np.zeros(len(scenarios))
    for tech in technologies:
        values = table[tech].to_numpy()
        for positive_side in (True, False):
            heights = np.maximum(values, 0) if positive_side else np.minimum(values, 0)
            bottoms = positive_bottom if positive_side else negative_bottom
            ax.bar(x, heights, balance.BAR_WIDTH, bottom=bottoms,
                   color=colors[tech], edgecolor="black" if not positive_side else "none",
                   linewidth=0.3, hatch=None if positive_side else "//")
            for i, value in enumerate(heights):
                if abs(value) >= 10:
                    ax.text(x[i], bottoms[i] + value / 2, f"{value:+.0f}",
                            ha="center", va="center", fontsize=8)
            bottoms += heights
    span = max(positive_bottom.max() - negative_bottom.min(), 1)
    for i, scenario in enumerate(scenarios):
        value = totals.loc[scenario, "gross_emissions_delta_MtCO2_per_year"]
        ax.text(x[i], positive_bottom[i] + 0.025 * span, f"{value:+.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for j, (family, first, last) in enumerate(ranges):
        ax.text((x[first] + x[last]) / 2, -0.26,
                balance._family_plot_label(family, scenarios[first : last + 1]),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontweight="bold")
        if j < len(ranges) - 1:
            ax.axvline((x[last] + x[ranges[j + 1][1]]) / 2, color="0.5", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [balance._scenario_plot_label(s) for s in scenarios],
                  rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel("Change in atmospheric CO₂ contribution [MtCO₂/a]", fontweight="bold")
    ax.set_xlabel("Scenario", fontweight="bold")
    ax.set_title("CO₂ balance: change relative to BASE", fontweight="bold", pad=32)
    ax.text(0, 1.02, "Bold totals: change in gross emissions; segment labels: carrier changes",
            transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=colors[t], label=t) for t in technologies]
    handles.extend([
        Patch(facecolor="white", edgecolor="black", label="Increase (+)"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Decrease (−)"),
    ])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_ylim(negative_bottom.min() - 0.08 * span, positive_bottom.max() + 0.16 * span)
    fig.subplots_adjust(bottom=0.28)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        print(f"[WRITE] {stem}.{suffix}")
    plt.close(fig)
    print(totals.round(3).to_string())


if __name__ == "__main__":
    main()
