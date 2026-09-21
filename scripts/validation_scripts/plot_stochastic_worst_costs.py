#!/usr/bin/env python3
"""Track the most expensive internal scenario of each stochastic network.

Scenario totals = common CAPEX + unweighted scenario OPEX. Common CAPEX is
probability-weighted capital expenditure (first-stage expected investment cost).
Scenario probabilities never multiply OPEX when selecting the worst scenario.
Exports every scenario, the selected maxima, and source-file metadata to Excel.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

DEFAULT_ROOT = Path("results/cutouts_det_capexp_nols")
SCALE = 1e9
UNIT = "bn. €/a"


def extract_scenario_costs(path):
    n = pypsa.Network(path)
    probabilities = n.scenario_weightings["weight"].astype(float)
    if (
        probabilities.empty
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
    ):
        raise ValueError(f"Invalid scenario probabilities: {path}")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"Scenario probabilities do not sum to one: {path}")
    values = {}
    for metric in ("capex", "opex"):
        # Keep the scenario index and avoid carrier nice-name replacement on
        # scenario-indexed carrier tables. Statistics applies snapshot weights,
        # but does not apply scenario probabilities.
        detail = getattr(n.statistics, metric)(
            nice_names=False, round=6, drop_zero=False
        )
        if "scenario" not in detail.index.names:
            raise ValueError(f"Missing scenario dimension in {metric}: {path}")
        values[metric] = (
            detail.groupby(level="scenario").sum().reindex(probabilities.index) / SCALE
        )
    common_capex = float(values["capex"].dot(probabilities))
    frame = pd.DataFrame(
        {
            "probability": probabilities,
            "common_capex": common_capex,
            "scenario_opex": values["opex"],
            "raw_scenario_capex": values["capex"],
        }
    )
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Nonfinite costs: {path}")
    frame["total_cost"] = frame.common_capex + frame.scenario_opex
    frame["raw_scenario_total"] = frame.raw_scenario_capex + frame.scenario_opex
    frame["is_worst"] = np.isclose(
        frame.total_cost, frame.total_cost.max(), rtol=1e-12, atol=1e-9
    )
    frame.index.name = "internal_scenario"
    frame = frame.reset_index()
    frame.insert(0, "solution", path.parent.parent.name)
    frame["network_path"] = str(path)
    frame["network_mtime_utc"] = pd.Timestamp(
        path.stat().st_mtime, unit="s", tz="UTC"
    ).isoformat()
    return frame


def select_worst(all_costs):
    records = []
    for solution, group in all_costs.groupby("solution", sort=False):
        row = group.loc[group.total_cost.idxmax()].copy()
        row["n_scenarios"] = len(group)
        row["expected_total_cost"] = group.total_cost.dot(group.probability)
        row["raw_capex_spread"] = (
            group.raw_scenario_capex.max() - group.raw_scenario_capex.min()
        )
        row["tied_worst_scenarios"] = ", ".join(
            group.loc[group.is_worst, "internal_scenario"]
        )
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def plot_worst(worst, output):
    x = np.arange(len(worst))
    capex = worst.common_capex.to_numpy()
    opex = worst.scenario_opex.to_numpy()
    fig, (ax, op_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    ax.bar(x, capex, label="Common CAPEX", color="#426b9b", width=0.65)
    ax.bar(
        x, opex, bottom=capex, label="Worst-scenario OPEX", color="#dc8d3f", width=0.65
    )
    for i, value in enumerate(capex):
        ax.text(
            i,
            value / 2,
            f"{value:,.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    for i, total in enumerate(worst.total_cost):
        ax.annotate(
            f"{total:,.1f}",
            (i, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel(f"CAPEX + OPEX ({UNIT})")
    ax.set_title("Most expensive internal scenario of each stochastic solution")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9)
    ax.margins(y=0.15)
    op_ax.bar(x, opex, color="#dc8d3f", width=0.65)
    for i, value in enumerate(opex):
        op_ax.annotate(
            f"{value:,.2f}",
            (i, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    op_ax.set_ylabel(f"OPEX detail ({UNIT})")
    op_ax.margins(y=0.2)
    labels = [f"{row.solution}\n{row.internal_scenario}" for row in worst.itertuples()]
    op_ax.set_xticks(x, labels, fontsize=9)
    op_ax.set_xlabel(
        "Capacity solution / most expensive internal scenario\nWorst = max(common CAPEX + scenario OPEX); scenario OPEX is not probability-weighted"
    )
    for axis in (ax, op_ax):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (
        args.output_dir
        or args.root_dir / "analysis_output/validation_heatmaps/cost_distributions"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    folders = sorted(
        (
            p
            for p in args.root_dir.glob("stochastic_K*")
            if re.fullmatch(r"stochastic_K\d+", p.name)
        ),
        key=lambda p: int(p.name.split("K")[-1]),
    )
    for folder in folders:
        if not re.fullmatch(r"stochastic_K\d+", folder.name):
            continue
        cluster = folder.name.removeprefix("stochastic_")
        path = folder / "networks" / f"base_s_cssc_{cluster}_adm___2050.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    if not paths:
        raise ValueError("No stochastic_K networks found")
    if args.workers == 1:
        frames = []
        for path in paths:
            frames.append(extract_scenario_costs(path))
            print(f"Evaluated {path.parent.parent.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = []
            for path, frame in zip(paths, pool.map(extract_scenario_costs, paths)):
                frames.append(frame)
                print(f"Evaluated {path.parent.parent.name}", flush=True)
    all_costs = pd.concat(frames, ignore_index=True)
    worst = select_worst(all_costs)
    method = pd.DataFrame(
        {
            "setting": [
                "cost_unit",
                "common_capex",
                "scenario_opex",
                "selection",
                "ties",
                "expected_total_cost",
            ],
            "value": [
                UNIT,
                "Sum(probability * raw scenario CAPEX); common to all scenarios of the solution",
                "PyPSA annual OPEX with snapshot weights, without scenario probabilities",
                "Maximum of common CAPEX + scenario OPEX across internal scenarios",
                "First maximum plotted; all ties retained in workbook",
                "Probability-weighted mean of internal scenario totals",
            ],
        }
    )
    with pd.ExcelWriter(output / "stochastic_worst_internal_costs.xlsx") as writer:
        worst.to_excel(writer, sheet_name="worst_scenarios", index=False)
        all_costs.to_excel(writer, sheet_name="all_internal_scenarios", index=False)
        method.to_excel(writer, sheet_name="method", index=False)
    plot_worst(worst, output / "stochastic_worst_internal_costs")
    print(
        worst[
            [
                "solution",
                "internal_scenario",
                "common_capex",
                "scenario_opex",
                "total_cost",
            ]
        ].to_string(index=False)
    )
    print(f"Written worst-scenario plot and workbook to {output}")


if __name__ == "__main__":
    main()
