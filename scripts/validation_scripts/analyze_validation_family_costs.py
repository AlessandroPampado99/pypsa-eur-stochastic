#!/usr/bin/env python3
"""Explain validation costs and load slack by capacity/operation family."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from plot_validation_heatmaps import _get_snapshot_weightings
from plot_validation_diagnostics import load_shedding_decomposition

FAMILIES = ["BASE", "E", "F", "M", "H"]


def family(name):
    if name == "BASE":
        return "BASE"
    return {"ELEC": "E", "OILGAS": "F", "MEOH": "M", "H2": "H"}[name.split("_")[0]]


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def cost_bars(summary, output, name, title):
    columns = ["capex", "opex_without_slack", "slack_penalty"]
    colors = ["#6b8ba4", "#91b57b", "#c95b53"]
    labels = ["CAPEX", "OPEX excluding load slack", "Load-slack penalty"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metrics, legend in ((axes[0], columns, labels), (axes[1], columns[:2], labels[:2])):
        bottom = np.zeros(len(summary))
        for metric, color, label in zip(metrics, colors, legend):
            values = summary[metric].to_numpy()
            ax.bar(summary.index, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_ylabel("Mean cost [billion EUR/year]")
        ax.set_ylim(0, max(bottom) * 1.30)
        ax.set_title("Including slack penalty" if ax is axes[0] else "Excluding slack penalty")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
    fig.suptitle(title)
    save(fig, output, name)


def carrier_bars(table, output, name, title):
    shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    keep = shares.max() >= 0.03
    plotted = table.loc[:, keep].copy()
    plotted["Other"] = table.loc[:, ~keep].sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, max(5, 0.38 * len(table))), layout="constrained")
    left = np.zeros(len(table))
    for i, (carrier, values) in enumerate(plotted.items()):
        ax.barh(table.index, values, left=left, label=carrier,
                color=plt.get_cmap("tab20")(i % 20), hatch="//" if i >= 20 else None)
        left += values.to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("Mean load slack [TWh/year]")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/demand_uncertainty_2035"))
    args = parser.parse_args()
    workbook = args.root / "analysis_output/validation_heatmaps/validation_heatmaps.xlsx"
    output = workbook.parent / "diagnostics/family_cost_explanation"
    output.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(workbook, sheet_name=None, index_col=0)
    rows, carrier_rows = [], []
    for capacity in sheets["total_cost"].index:
        for operation in sheets["total_cost"].columns:
            print(f"Loading {capacity} -> {operation}", flush=True)
            filename = ("base_s_adm___2035.nc" if capacity == operation else
                        f"base_s_adm___2035__cap-{capacity}__op-{operation}.nc")
            path = args.root / capacity / "networks" / filename
            n = pypsa.Network(path)
            ids = n.generators.index[n.generators.carrier.eq("load")]
            dispatch = n.generators_t.p.loc[:, ids]
            weights = _get_snapshot_weightings(n)
            marginal = n.get_switchable_as_dense("Generator", "marginal_cost").loc[dispatch.index, ids]
            penalty = float(dispatch.mul(marginal).mul(weights, axis=0).sum().sum()) / 1e9
            opex = float(n.statistics.opex().sum()) / 1e9
            capex = float(n.statistics.capex().sum()) / 1e9
            decomposition = load_shedding_decomposition(n)
            for metric, value in (("total_cost", capex + opex), ("capex", capex),
                                  ("opex", opex), ("load_curtailment", decomposition.sum())):
                if bool(sheets[f"invalid_{metric}"].loc[capacity, operation]):
                    raise ValueError(f"Invalid workbook cell: {capacity}, {operation}, {metric}")
                np.testing.assert_allclose(value, sheets[metric].loc[capacity, operation], rtol=1e-6, atol=1e-5)
            metadata = dict(capacity=capacity, operation=operation,
                            capacity_family=family(capacity), operation_family=family(operation),
                            diagonal=capacity == operation)
            rows.append(dict(**metadata, capex=capex, opex_without_slack=opex-penalty,
                             slack_penalty=penalty, total_cost=capex+opex,
                             load_slack_TWh=float(decomposition.sum()), network=str(path)))
            carrier_rows.extend(dict(**metadata, carrier=carrier, load_slack_TWh=value)
                                for carrier, value in decomposition.items())
            # Preserve newly computed values if a later network fails; never reuse them.
            pd.DataFrame(rows).to_csv(output / "scenario_cost_components.csv", index=False)
            pd.DataFrame(carrier_rows).to_csv(output / "scenario_carrier_slack.csv", index=False)
    costs = pd.DataFrame(rows)
    carriers = pd.DataFrame(carrier_rows)
    cross = costs.loc[~costs.diagonal]
    metrics = ["capex", "opex_without_slack", "slack_penalty", "total_cost", "load_slack_TWh"]
    tables = {}
    for axis in ("capacity_family", "operation_family"):
        summary = cross.groupby(axis)[metrics].mean().reindex(FAMILIES)
        summary["pair_count"] = cross.groupby(axis).size()
        summary["penalty_share"] = summary.slack_penalty / summary.total_cost
        tables[axis] = summary
        summary.to_csv(output / f"mean_cost_by_{axis}.csv")
        cost_bars(summary, output, f"mean_cost_by_{axis}",
                  "Capacity family tested across operating scenarios" if axis == "capacity_family"
                  else "Operating family served by alternative capacity configurations")
    grouped = cross.groupby(["capacity_family", "operation_family"])[metrics].mean()
    grouped["pair_count"] = cross.groupby(["capacity_family", "operation_family"]).size()
    grouped.to_csv(output / "family_pair_costs.csv")
    diagonal = costs.loc[costs.diagonal].groupby("capacity_family")[metrics].mean().reindex(FAMILIES)
    diagonal.to_csv(output / "own_scenario_costs_by_family.csv")
    tables["own_scenario"] = diagonal
    cc = carriers.loc[~carriers.diagonal]
    for axis in ("capacity_family", "operation_family"):
        table = cc.groupby([axis, "carrier"]).load_slack_TWh.sum().unstack(fill_value=0)
        table = table.div(cross.groupby(axis).size(), axis=0).reindex(FAMILIES)
        table.to_csv(output / f"carrier_slack_by_{axis}.csv")
        carrier_bars(table, output, f"carrier_slack_by_{axis}", f"Load slack by {axis.replace('_', ' ')}")
    pair_counts = cross.groupby(["capacity_family", "operation_family"]).size()
    pair_table = cc.groupby(["capacity_family", "operation_family", "carrier"]).load_slack_TWh.sum().unstack(fill_value=0)
    pair_table = pair_table.div(pair_counts, axis=0)
    pair_table.to_csv(output / "carrier_slack_by_family_pair.csv")
    pair_table.index = [f"{c} → {o}" for c, o in pair_table.index]
    carrier_bars(pair_table, output, "carrier_slack_by_family_pair", "Capacity family → operating family (diagonals excluded)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(axes, ["total_cost", "load_slack_TWh"],
                                 ["Mean cost [billion EUR/year]", "Mean load slack [TWh/year]"]):
        matrix = grouped[metric].unstack().reindex(index=FAMILIES, columns=FAMILIES)
        im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap="Reds")
        ax.set_xticks(range(5), FAMILIES, fontweight="bold")
        ax.set_yticks(range(5), FAMILIES, fontweight="bold")
        ax.set_xlabel("Operating family"); ax.set_ylabel("Capacity family"); ax.set_title(title)
        for i in range(5):
            for j in range(5):
                v = matrix.iloc[i,j]
                if np.isfinite(v):
                    ax.text(j,i,f"{v:,.0f}",ha="center",va="center",color="white" if v > matrix.max().max()/2 else "black")
        fig.colorbar(im, ax=ax)
    save(fig, output, "family_validation_matrices")
    with pd.ExcelWriter(output / "family_cost_explanation.xlsx") as writer:
        for key, table in tables.items():
            table.to_excel(writer, sheet_name=key)
        grouped.to_excel(writer, sheet_name="family_pairs")
        costs.to_excel(writer, sheet_name="scenario_costs", index=False)
    report = ["# Validation costs by family", f"Input: {workbook.resolve()}",
              "Families: BASE; E = ELEC; F = OILGAS; M = MEOH; H = H2.",
              "Means give equal weight to each scenario pair, exclude same-scenario diagonals, and do not use IARSO probabilities. Within-family cells retain tests between different scenarios. BASE → BASE has no off-diagonal test. Pair counts are exported; family sizes differ.",
              "Slack penalty is the weighted sum of carrier=load generator dispatch times its actual static/time-varying marginal cost. CAPEX and OPEX use the same PyPSA statistics as the validation workbook. OPEX excluding slack subtracts this penalty; it is a cost component, not a counterfactual feasible cost without slack.",
              "Every network's CAPEX, OPEX, total cost and carrier slack sum reconcile to the workbook. No previously computed CSV is read.",
              "Slack is grouped by bus carrier. These quantities include intermediate energy carriers and are not a verified final-demand deficit. Carrier plots retain carriers contributing at least 3% in any displayed group and combine the rest as Other.",
              "The results identify cost components and affected carriers, not causal capacity bottlenecks or the role of the CO2 constraint. Those require temporal dispatch/bound checks and controlled re-solves."]
    for key, table in tables.items():
        report += [f"## {key}", table.to_string()]
    (output / "README.md").write_text("\n\n".join(report)+"\n")
    (output / "provenance.json").write_text(json.dumps(dict(workbook=str(workbook.resolve()),
        workbook_mtime=workbook.stat().st_mtime, networks=len(costs)), indent=2))
    print(tables["capacity_family"].to_string(), flush=True)
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
