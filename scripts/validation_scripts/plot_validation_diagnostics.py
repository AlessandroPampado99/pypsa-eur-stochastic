#!/usr/bin/env python3
"""
Diagnose cross-scenario fragility from the existing validation workbook.

Run with the pypsa-eur environment. Relative CSV statistics are dimensionless;
relative plots use percent. Standard deviations describe the population (ddof=0).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plot_validation_heatmaps import (
    DPI,
    LOAD_CURTAILMENT_CARRIERS,
    LOAD_CURTAILMENT_SCALE,
    _get_snapshot_weightings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/demand_uncertainty_2035/analysis_output/validation_heatmaps/validation_heatmaps.xlsx"
)
CASES = [
    ("ELEC_ROAD", "OILGAS_IND"),
    ("ELEC_IND", "OILGAS_IND"),
    ("ELEC_HEAT", "OILGAS_IND"),
    ("BASE", "OILGAS_IND"),
    ("H2_IND", "ELEC_IND"),
    ("ELEC_IND", "H2_IND"),
]
NORMALISATION_NOTE = (
    "Skipped: demand_compare_levels.xlsx is weighted PyPSA Load accounting, "
    "including emissions and heat-service loads, not a verified final-energy "
    "total. Its generic network picker also does not guarantee scenario provenance. "
    "Supply --final-demand-csv only after verifying scenario, year, energy boundary "
    "and units (columns: scenario,total_final_demand_TWh)."
)


def load_validation_matrices(path):
    """Load costs and TWh curtailment, respecting workbook invalid masks."""
    sheets = pd.read_excel(path, sheet_name=None, index_col=0)
    matrices = {}
    for key in ("total_cost", "load_curtailment"):
        frame = sheets[key].apply(pd.to_numeric, errors="raise")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"Duplicate labels in {key}")
        if set(frame.index) != set(frame.columns):
            raise ValueError("Expected matching capacity and operating scenario sets")
        mask = sheets.get(f"invalid_{key}")
        if mask is not None:
            mask = mask.reindex(index=frame.index, columns=frame.columns)
            if (
                mask.isna().any().any()
                or not mask.isin([True, False, 0, 1]).all().all()
            ):
                raise ValueError(f"Invalid boolean mask for {key}")
            frame = frame.mask(mask.astype(bool))
        matrices[key] = frame.replace([np.inf, -np.inf], np.nan)
    costs, loads = matrices.values()
    if set(costs.index) != set(loads.index) or set(costs.columns) != set(loads.columns):
        raise ValueError("Cost and curtailment scenario labels differ")
    if (loads < -1e-8).any().any():
        raise ValueError("Negative load curtailment in workbook")
    return matrices


def compute_diagnostics(matrix):
    """Exclude diagonals by label; report valid and missing cross-test counts."""
    values = matrix.copy()
    for name in values.index.intersection(values.columns):
        values.loc[name, name] = np.nan

    def summarize(frame, name):
        result = pd.DataFrame(
            {
                "mean": frame.mean(axis=1),
                "median": frame.median(axis=1),
                "minimum": frame.min(axis=1),
                "maximum": frame.max(axis=1),
                "standard_deviation": frame.std(axis=1, ddof=0),
                "n_valid": frame.count(axis=1),
                "n_missing": len(frame.columns) - 1 - frame.count(axis=1),
            }
        )
        result.index.name = name
        return result.sort_values("mean", ascending=False, kind="stable")

    return summarize(values, "capacity_configuration"), summarize(
        values.T, "operating_scenario"
    )


def compute_relative_metrics(matrix, denominator_path):
    """Use only externally verified total final demand; never sum mixed loads."""
    # TODO: reuse a verified final-energy accounting output when the upstream
    # analysis defines one. The current generic Load totals are not that metric.
    if denominator_path is None:
        return None
    demand = pd.read_csv(denominator_path).set_index("scenario")[
        "total_final_demand_TWh"
    ]
    if not demand.index.is_unique:
        raise ValueError("Duplicate demand scenarios")
    demand = demand.reindex(matrix.columns)
    if not np.isfinite(demand).all() or (demand <= 0).any():
        raise ValueError(
            "A positive finite TWh denominator is required for every scenario"
        )
    return matrix.div(demand, axis=1)


def plot_summary_bars(ax, table, kind, relative=False, title=None):
    """Draw sorted means with end labels and unchanged workbook names."""
    values = table["mean"] * (100 if relative else 1)
    bars = ax.barh(table.index, values)
    ax.bar_label(
        bars,
        labels=[f"{v:.2f}" if np.isfinite(v) else "missing" for v in values],
        padding=4,
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_ylabel(
        "Capacity configuration" if kind == "capacity" else "Operating scenario"
    )
    across = (
        "alternative scenarios"
        if kind == "capacity"
        else "alternative capacity configurations"
    )
    ax.set_xlabel(
        f"Mean {'relative ' if relative else ''}load curtailment across\n{across} [{'%' if relative else 'TWh'}]"
    )
    ax.set_title(title, loc="left", pad=12)
    ax.margins(x=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    if relative:
        ax.xaxis.set_major_formatter(PercentFormatter())


def save_figure(fig, output, stem, pdf=False):
    fig.savefig(output / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(tables, output, relative=False):
    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(7, 0.30 * len(tables[0]) + 2)), constrained_layout=True
    )
    for ax, table, kind, title in zip(
        axes,
        tables,
        ("capacity", "scenario"),
        ("(a) Capacity-configuration fragility", "(b) Operating-scenario difficulty"),
    ):
        plot_summary_bars(ax, table, kind, relative, title)
    save_figure(
        fig,
        output,
        "cross_scenario_fragility_and_difficulty" + ("_relative" if relative else ""),
        True,
    )


def export_summary(matrix, output, relative=False):
    tables = compute_diagnostics(matrix)
    for table, kind, title in zip(
        tables,
        ("capacity", "scenario"),
        (
            "Average cross-scenario load curtailment by capacity configuration",
            "Average load curtailment by operating scenario",
        ),
    ):
        stem = f"mean_{'relative_' if relative else ''}load_curtailment_by_{kind}"
        table.to_csv(output / f"{stem}.csv")
        fig, ax = plt.subplots(
            figsize=(11, max(7, 0.30 * len(table) + 2)), constrained_layout=True
        )
        plot_summary_bars(ax, table, kind, relative, title)
        save_figure(fig, output, stem)
    plot_combined_figure(tables, output, relative)
    return tables


def load_shedding_decomposition(n):
    """Reproduce workbook generator-dispatch accounting, grouped by bus carrier."""
    gens = n.generators.index[n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)]
    dispatch = n.generators_t.p.reindex(index=n.snapshots, columns=gens)
    if dispatch.isna().any().any():
        raise ValueError("Missing load-shedding dispatch")
    weights = _get_snapshot_weightings(n).reindex(n.snapshots)
    if not np.isfinite(weights).all():
        raise ValueError("Missing snapshot weights")
    energy = dispatch.mul(weights, axis=0).sum() / LOAD_CURTAILMENT_SCALE
    carriers = n.generators.loc[gens, "bus"].map(n.buses.carrier)
    if carriers.isna().any():
        raise ValueError("Missing bus carriers")
    return energy.groupby(carriers).sum()


def capacity_saturation(n, case):
    """
    Count snapshots at directional dispatch bounds, per asset.

    Generator/Link bounds include time-varying p_max_pu/p_min_pu. Link p0
    and nominal power both refer to the input port. Lines use the active-power
    limit of the linear optimisation, s_nom * s_max_pu. Idle/unavailable assets
    never count as saturated. Stores have no comparable nominal power bound.
    """
    records = []
    for component, attr, dispatch_attr, nominal, limit in (
        ("Generator", "generators", "p", "p_nom", "p"),
        ("Link", "links", "p0", "p_nom", "p"),
        ("Line", "lines", "p0", "s_nom", "s"),
    ):
        static = getattr(n, attr)
        if static.empty:
            continue
        cap = static[nominal].where(
            ~static[nominal + "_extendable"], static[nominal + "_opt"]
        )
        eligible = np.isfinite(cap) & (cap > 1e-6)
        eligible &= ~static.carrier.isin(
            [
                "load",
                "co2",
                "CO2 pipeline",
                "co2 sequestered",
                "process emissions",
                "process emissions CC",
                "HVC to air",
                "DAC",
            ]
        )
        ids = static.index[eligible]
        p = getattr(getattr(n, attr + "_t"), dispatch_attr).reindex(
            index=n.snapshots, columns=ids
        )
        if p.isna().any().any():
            raise ValueError(f"Missing {component} dispatch")
        upper = (
            n.get_switchable_as_dense(component, limit + "_max_pu")
            .loc[:, ids]
            .mul(cap[ids], axis=1)
        )
        lower = (
            -upper
            if component == "Line"
            else n.get_switchable_as_dense(component, "p_min_pu")
            .loc[:, ids]
            .mul(cap[ids], axis=1)
        )
        if not np.isfinite(upper).all().all() or not np.isfinite(lower).all().all():
            raise ValueError(f"Non-finite {component} dispatch bounds")
        hit = ((upper > 1e-6) & (p >= 0.99 * upper)) | (
            (lower < -1e-6) & (p <= 0.99 * lower)
        )
        counts = hit.sum()
        hours = hit.mul(n.snapshot_weightings.generators, axis=0).sum()
        for asset in ids:
            records.append(
                {
                    "case": case,
                    "component": component,
                    "carrier": static.at[asset, "carrier"],
                    "asset": asset,
                    "nominal_capacity": cap[asset],
                    "saturation_snapshots": counts[asset],
                    "saturation_hours": hours[asset],
                    "saturation_fraction": counts[asset] / len(n.snapshots),
                }
            )
    return pd.DataFrame(records)


def plot_decomposition(table, output):
    """Keep any carrier contributing at least 1% in any selected case."""
    table.to_csv(output / "load_curtailment_decomposition_selected_cases.csv")
    shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    keep = shares.max() >= 0.01
    plotted = table.loc[:, keep].copy()
    if (~keep).any():
        plotted["Other"] = table.loc[:, ~keep].sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    bottom = np.zeros(len(table))
    for index, (carrier, values) in enumerate(plotted.items()):
        cycle_length = len(plt.rcParams["axes.prop_cycle"])
        hatch = ("", "//", "xx")[min(index // cycle_length, 2)]
        ax.bar(table.index, values, bottom=bottom, label=carrier, hatch=hatch)
        bottom += values.to_numpy()
    ax.set_ylabel("Total load curtailment [TWh]")
    ax.set_xlabel("Capacity configuration → operating scenario")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output, "load_curtailment_decomposition_selected_cases", True)


def plot_saturation(assets, output, cases):
    """Aggregate within component/carrier using nominal-capacity weights."""
    assets.to_csv(output / "capacity_saturation_by_asset.csv", index=False)
    assets["weighted_fraction"] = assets.saturation_fraction * assets.nominal_capacity
    grouped = assets.groupby(["component", "carrier", "case"])
    summary = grouped[["weighted_fraction", "nominal_capacity"]].sum()
    summary["saturation_fraction"] = (
        summary.weighted_fraction / summary.nominal_capacity
    )
    summary.to_csv(output / "capacity_saturation_by_technology.csv")
    matrix = summary.saturation_fraction.unstack("case").reindex(columns=cases)
    # Retain requested conversion/transfer technologies even below the top 35.
    order = matrix.max(axis=1).sort_values(ascending=False, kind="stable")
    priority = {
        "H2 Electrolysis",
        "SMR",
        "SMR CC",
        "biogas to gas",
        "methanolisation",
        "electricity distribution grid",
        "H2 pipeline",
    }
    selected = set(order[order > 0].head(35).index)
    selected.update(index for index in matrix.index if index[1] in priority)
    matrix = matrix.loc[[index for index in order.index if index in selected]]
    if matrix.empty:
        return False
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(matrix) * 0.29)), constrained_layout=True
    )
    im = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=1, aspect="auto"
    )
    ax.set_yticks(range(len(matrix)), [f"{c}: {t}" for c, t in matrix.index])
    ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
    for i in range(len(matrix)):
        for j in range(len(cases)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.0%}" if pd.notna(value) else "—",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if pd.notna(value) and value < 0.5 else "black",
            )
    ax.set_title("Capacity saturation: top technologies across selected cross-tests")
    fig.colorbar(
        im, ax=ax, label="Capacity-weighted fraction of snapshots at dispatch bound"
    )
    save_figure(fig, output, "capacity_saturation_selected_cases", True)
    return True


def select_costliest_cases(costs, count):
    """Select finite off-diagonal tests, descending by cost with stable ties."""
    if count < 1:
        raise ValueError("Case count must be positive")
    candidates = costs.copy()
    for label in candidates.index.intersection(candidates.columns):
        candidates.loc[label, label] = np.nan
    ranked = candidates.stack().sort_values(ascending=False, kind="stable")
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} valid off-diagonal costs available")
    return list(ranked.head(count).index)


def detailed_diagnostics(root, matrix, costs, output, year, cases=CASES):
    """Read exact cross-test files; report missing outputs without substituting runs."""
    import pypsa

    decomposed, saturation, notes, comparisons = {}, [], [], []
    for capacity, operating in cases:
        case = f"{capacity} → {operating}"
        path = (
            root
            / capacity
            / "networks"
            / f"base_s_adm___{year}__cap-{capacity}__op-{operating}.nc"
        )
        if not path.exists():
            notes.append(f"Missing: {path}")
            continue
        n = pypsa.Network(path)
        try:
            values = load_shedding_decomposition(n)
            expected = matrix.loc[capacity, operating]
            if not np.isfinite(expected) or not np.isclose(
                values.sum(), expected, rtol=1e-6, atol=1e-5
            ):
                raise ValueError(
                    f"Network total {values.sum()} differs from workbook {expected}"
                )
            decomposed[case] = values
            comparisons.append(
                {
                    "case": case,
                    "load_curtailment_TWh": expected,
                    "total_cost_bn_EUR_per_year": costs.loc[capacity, operating],
                    "network": str(path),
                }
            )
        except ValueError as error:
            notes.append(f"Decomposition {case}: {error}")
        try:
            saturation.append(capacity_saturation(n, case))
        except ValueError as error:
            notes.append(f"Saturation {case}: {error}")
    if decomposed:
        plot_decomposition(pd.DataFrame(decomposed).T.fillna(0), output)
        pd.DataFrame(comparisons).to_csv(
            output / "selected_cross_tests.csv", index=False
        )
    saturation_ok = False
    if saturation:
        saturation_ok = plot_saturation(
            pd.concat(saturation, ignore_index=True),
            output,
            [f"{c} → {o}" for c, o in cases],
        )
    notes.append(
        "Upstream requirement: retain solve_validation_operations_network.py's solved NetCDF export at each path above, including generators_t.p, links_t.p0, lines_t.p0, bus/carrier mappings, snapshot weights, nominal/optimised capacities and static/time-dependent per-unit dispatch bounds. No re-optimisation is performed here."
    )
    return bool(decomposed), saturation_ok, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--network-root", type=Path)
    parser.add_argument("--year", default="2035")
    parser.add_argument("--final-demand-csv", type=Path)
    parser.add_argument(
        "--scenario-regex", help="Keep matching workbook labels on both axes"
    )
    parser.add_argument(
        "--top-cost-cases",
        type=int,
        help="Select this many highest-cost off-diagonal tests",
    )
    args = parser.parse_args()
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    output = args.output_dir or args.input.parent / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
        }
    )
    matrices = load_validation_matrices(args.input)
    if args.scenario_regex:
        labels = [
            label
            for label in matrices["load_curtailment"].index
            if re.fullmatch(args.scenario_regex, str(label))
        ]
        if len(labels) < 2:
            raise ValueError("Scenario filter must retain at least two scenarios")
        matrices = {key: frame.loc[labels, labels] for key, frame in matrices.items()}
    cases = (
        select_costliest_cases(matrices["total_cost"], args.top_cost_cases)
        if args.top_cost_cases is not None
        else CASES
    )
    selection_note = (
        f"Selected {len(cases)} highest-cost valid off-diagonal tests."
        if args.top_cost_cases is not None
        else "Using predefined demand-transfer cases."
    )
    selection_note += f" Scenario filter: {args.scenario_regex or 'none'}."
    print(selection_note)
    normalisation_note = NORMALISATION_NOTE
    if args.input.resolve() != DEFAULT_INPUT.resolve():
        normalisation_note = (
            "Skipped: no verified total-final-demand denominator is available in this workbook or nearby analysis outputs. "
            "Generic PyPSA Load totals are not a verified final-energy metric. "
            "Supply --final-demand-csv with scenario,total_final_demand_TWh after verifying the energy boundary and units."
        )
    matrix = matrices["load_curtailment"]
    tables = export_summary(matrix, output)
    relative = compute_relative_metrics(matrix, args.final_demand_csv)
    if relative is not None:
        export_summary(relative, output, True)
    decomposition, saturation, notes = detailed_diagnostics(
        args.network_root or args.input.parents[2],
        matrix,
        matrices["total_cost"],
        output,
        args.year,
        cases,
    )
    methodology = [
        selection_note,
        "Rows: fixed capacity configurations; columns: operating scenarios. Diagonals excluded by label. Invalid/missing cells omitted and counted. Population standard deviation (ddof=0).",
        "Workbook units: load curtailment TWh; total cost billion EUR/year. Selected costs exported for context; no cost-derived metric is invented.",
        "Decomposition uses the same carrier=load generators and snapshot weighting as plot_validation_heatmaps.py and must reconcile to the workbook. Bus carrier names remain unchanged. This is the workbook's load-slack accounting, not necessarily unmet final demand at end-use buses.",
        "Saturation is a dispatch-bound diagnostic, not proof of a causal bottleneck. Time-dependent availability is respected; unavailable zero bounds do not count. Generator/Link/Line technologies remain separate. Storage energy and coupled CHP constraints are not measured. Fractions use unweighted snapshot counts; hours use generator snapshot weights. Technology fractions are nominal-capacity-weighted asset fractions; assets with zero/infinite nominal capacity are excluded. Missing technologies are blank, not zero. All asset/technology results are exported; plot shows top 35 by maximum case fraction plus requested conversion/transfer technologies where present.",
        normalisation_note
        if relative is None
        else f"Relative metrics use verified denominator input: {args.final_demand_csv}; CSVs are fractions, plots percent.",
    ]
    (output / "README.md").write_text("\n\n".join(methodology + notes) + "\n")
    for title, table in zip(
        ("Most fragile capacity configurations", "Most difficult operating scenarios"),
        tables,
    ):
        print(f"\n{title} (mean off-diagonal TWh):")
        print(table["mean"].head(5).to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nNormalised diagnostics generated: {relative is not None}")
    print(f"Carrier decomposition possible: {decomposition}")
    print(f"Capacity-saturation diagnostics possible: {saturation}")
    if relative is None:
        print(normalisation_note)
    for note in notes[:-1]:
        print(note)
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
