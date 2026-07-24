#!/usr/bin/env python3
"""Create first-pass diagnostic plots from collected cutout metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def year_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{4})", expand=False)


def plot_demands(data_dir: Path, out: Path) -> None:
    df = pd.read_csv(data_dir / "all_demand.csv")
    # Operation-year inputs repeat across capacity designs: retain one copy.
    df = df.sort_values("capacity_scenario").drop_duplicates(
        ["operation_scenario", "demand_group", "country"]
    )
    totals = df.groupby(["operation_scenario", "demand_group"]).energy_mwh.sum().unstack()
    totals.index = year_label(totals.index.to_series())
    ax = (totals / 1e6).plot(figsize=(12, 6), marker="o")
    ax.set(title="Annual demand by weather/operation year", xlabel="Operation year", ylabel="TWh")
    ax.grid(alpha=0.25)
    save(ax.figure, out / "annual_demands.png")

    variability = totals.agg(["mean", "std", "min", "max"]).T
    variability["coefficient_of_variation"] = variability["std"] / variability["mean"]
    variability.sort_values("coefficient_of_variation").to_csv(out / "demand_variability.csv")


def plot_capacity_factors(data_dir: Path, out: Path) -> None:
    df = pd.read_csv(data_dir / "all_renewable_capacity_factors.csv")
    df = df.sort_values("capacity_scenario").drop_duplicates(
        ["operation_scenario", "carrier", "country"]
    )
    df["year"] = year_label(df.operation_scenario)
    selected = df[df.country.isin(["ALL", "IT", "DE", "ES", "FR"])].copy()
    selected["country"] = selected["country"].replace({"ALL": "Europe"})
    for carrier, part in selected.groupby("carrier"):
        pivot = part.pivot(index="year", columns="country", values="mean")
        preferred_order = ["Europe", "IT", "DE", "ES", "FR"]
        pivot = pivot.reindex(
            columns=[country for country in preferred_order if country in pivot.columns]
        )
        ax = pivot.plot(figsize=(11, 5), marker="o")
        ax.set(title=f"Weather capacity factor: {carrier}", xlabel="Year", ylabel="Mean p.u.")
        ax.grid(alpha=0.25)
        save(ax.figure, out / f"capacity_factor_{carrier.replace(' ', '_')}.png")

    summary = df.groupby(["carrier", "country"])["mean"].agg(["mean", "std", "min", "max"])
    summary["coefficient_of_variation"] = summary["std"] / summary["mean"]
    summary.sort_values("coefficient_of_variation", ascending=False).to_csv(
        out / "capacity_factor_variability.csv"
    )


def plot_costs_and_drivers(data_dir: Path, out: Path) -> None:
    diag = pd.read_csv(data_dir / "all_diagnostics.csv")
    wide = diag.pivot_table(index=["capacity_scenario", "operation_scenario"], columns="metric", values="value")
    if "capex_eur" in wide and "opex_eur" in wide:
        wide["total_cost_eur"] = wide.capex_eur + wide.opex_eur
    wide.to_csv(out / "case_summary.csv")

    if "total_cost_eur" in wide:
        costs_by_year = wide.reset_index()
        costs_by_year["year"] = year_label(costs_by_year.capacity_scenario)
        costs_by_year = costs_by_year.set_index("year").sort_index()
        ax = (costs_by_year.total_cost_eur / 1e9).plot(
            figsize=(12, 5), marker="o", color="#7f0000"
        )
        ax.set(
            title="Capacity-expansion total cost by weather year",
            xlabel="Weather year used for capacity expansion",
            ylabel="Billion EUR",
        )
        ax.grid(alpha=0.25)
        save(ax.figure, out / "capacity_expansion_total_cost.png")

    # Cost component composition for the most expensive cases.
    costs = pd.read_csv(data_dir / "all_costs.csv")
    totals = costs.groupby("case_id").cost_eur.sum().nlargest(15)
    expensive = costs[costs.case_id.isin(totals.index)]
    composition = expensive.groupby(["case_id", "carrier"]).cost_eur.sum().unstack(fill_value=0)
    keep = composition.abs().sum().nlargest(12).index
    ax = (composition[keep] / 1e9).plot.bar(stacked=True, figsize=(14, 7))
    ax.set(title="Cost composition of the 15 most expensive cases", ylabel="Billion EUR", xlabel="")
    ax.tick_params(axis="x", rotation=75)
    save(ax.figure, out / "expensive_case_cost_composition.png")

    # Correlations are exploratory, not causal; each expansion cost is merged
    # with the demand and weather inputs used in that same expansion run.
    demand = pd.read_csv(data_dir / "all_demand.csv")
    demand = demand.groupby(["operation_scenario", "demand_group"]).energy_mwh.sum().unstack()
    cf = pd.read_csv(data_dir / "all_renewable_capacity_factors.csv")
    cf = cf[cf.country.eq("ALL")].groupby(["operation_scenario", "carrier"])["mean"].mean().unstack()
    features = demand.join(cf, how="outer", lsuffix="_demand", rsuffix="_cf")
    cases = wide.reset_index().merge(features.reset_index(), on="operation_scenario", how="left")
    numeric = cases.select_dtypes(include="number")
    target = "total_cost_eur" if "total_cost_eur" in numeric else "opex_eur"
    correlations = numeric.corr()[target].sort_values()
    correlations.to_csv(out / "cost_correlations.csv", header=["pearson_correlation"])
    ax = correlations.drop(target).plot.barh(figsize=(9, max(5, len(correlations) * 0.28)))
    ax.set(title=f"Exploratory correlations with {target}", xlabel="Pearson correlation")
    ax.axvline(0, color="black", linewidth=0.8)
    save(ax.figure, out / "cost_correlations.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/cutout_analysis/output/collected"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cutout_analysis/output/plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_demands(args.data_dir, args.output_dir)
    plot_capacity_factors(args.data_dir, args.output_dir)
    plot_costs_and_drivers(args.data_dir, args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
