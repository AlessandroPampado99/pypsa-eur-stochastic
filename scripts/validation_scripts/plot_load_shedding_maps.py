#!/usr/bin/env python3
"""
Map the load-slack accounting used by the validation diagnostics.

Run with the pypsa-eur environment. Defaults to the six selected weather
cross-tests; accepts another selected_cross_tests.csv or a single --network.
"""

import argparse
import re
import shlex
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Wedge
from plot_validation_heatmaps import (
    LOAD_CURTAILMENT_CARRIERS,
    LOAD_CURTAILMENT_SCALE,
    _get_snapshot_weightings,
)
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = (
    ROOT
    / "results/cutouts_det_capexp_nols/analysis_output/validation_heatmaps/diagnostics"
)
CRS = "EPSG:3035"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() or path.exists() else ROOT / path


def aggregate(n):
    """Keep bus-level data and use the exact workbook weighting convention."""
    gens = n.generators.index[n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)]
    dispatch = n.generators_t.p.reindex(index=n.snapshots, columns=gens)
    weights = _get_snapshot_weightings(n).reindex(n.snapshots)
    if not np.isfinite(dispatch.to_numpy()).all() or not np.isfinite(weights).all():
        raise ValueError("Missing/nonfinite load dispatch or snapshot weights")
    energy = dispatch.mul(weights, axis=0).sum() / LOAD_CURTAILMENT_SCALE
    if (energy < -1e-8).any():
        raise ValueError("Negative load-slack energy cannot be represented as a pie")
    bus_energy = energy.groupby(n.generators.loc[gens, "bus"]).sum()
    table = n.buses.loc[bus_energy.index, ["carrier", "location", "x", "y"]].copy()
    table["load_shedding_TWh"] = bus_energy
    table.index.name = "bus"
    # Empty locations are only resolved when the bus itself is a geographic node.
    table["location"] = table.location.fillna("")
    empty = table.location.eq("")
    table.loc[empty, "location"] = table.index[empty]
    return table.reset_index()


def region_path(network):
    relative = network.resolve().relative_to(ROOT / "results")
    return (
        ROOT
        / "resources"
        / relative.parent.parent
        / "regions_onshore_base_s_adm.geojson"
    )


def prepare(case, network, regions_path, expected=None, europe_only=False):
    n = pypsa.Network(network)
    table = aggregate(n)
    total = table.load_shedding_TWh.sum()
    if expected is not None and not np.isclose(total, expected, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"{case}: network total {total} differs from diagnostics {expected}"
        )
    regions = gpd.read_file(regions_path).set_index("name").to_crs(CRS)
    if not regions.index.is_unique:
        regions = regions.dissolve(by=regions.index)
    table["mapped_to_region"] = table.location.isin(regions.index)
    positive = table.load_shedding_TWh > 1e-8
    unknown = table.loc[positive & ~table.mapped_to_region & table.location.ne("EU")]
    if not unknown.empty and not europe_only:
        raise ValueError(
            f"Unmapped positive load shedding: {unknown[['bus', 'location']].to_dict('records')}"
        )
    mix = (
        table.groupby(["location", "carrier"])
        .load_shedding_TWh.sum()
        .unstack(fill_value=0)
    )
    totals = mix.sum(axis=1)
    regions["load_shedding_TWh"] = totals.reindex(regions.index, fill_value=0)
    # Anchor sector buses at their shared geographic network location.
    coords = n.buses.reindex(mix.index)[["x", "y"]]
    mapped = mix.index.intersection(regions.index)
    if europe_only:
        # Clip overseas polygons as well as excluding non-geographic/global buses.
        window = box(-12, 34, 35, 72)
        regions = regions.to_crs(4326)
        regions.geometry = regions.geometry.intersection(window)
        regions = regions.loc[~regions.geometry.is_empty].to_crs(CRS)
        mapped = mapped.intersection(regions.index)
        inside = coords.loc[mapped, "x"].between(-12, 35) & coords.loc[
            mapped, "y"
        ].between(34, 72)
        mapped = mapped[inside]
        mix = mix.loc[mapped]
    table["included_in_map"] = table.location.isin(mapped)
    if not np.isfinite(coords.loc[mapped].to_numpy()).all():
        raise ValueError("Missing coordinates for geographic bus locations")
    points = gpd.GeoSeries(
        gpd.points_from_xy(coords.loc[mapped, "x"], coords.loc[mapped, "y"]),
        index=mapped,
        crs="EPSG:4326",
    ).to_crs(CRS)
    return dict(
        case=case,
        table=table,
        mix=mix,
        regions=regions,
        points=points,
        total=total,
        network=str(network),
        colors=n.meta.get("plotting", {}).get("tech_colors", {}),
        europe_only=europe_only,
    )


def pie(ax, values, x, y, radius, colors):
    values = values.clip(lower=0)
    total = values.sum()
    if total <= 1e-8:
        return
    angle = 90
    for carrier, value in values.items():
        if value <= 0:
            continue
        end = angle + 360 * value / total
        ax.add_patch(
            Wedge(
                (x, y),
                radius,
                angle,
                end,
                facecolor=colors[carrier],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        )
        angle = end


def plot_case(data, output, colors, vmax, pie_max, pdf=None):
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False
    regions, mix = data["regions"], data["mix"]
    fig, ax = plt.subplots(figsize=(16, 10), layout="none")
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.02, right=0.59, bottom=0.14, top=0.91)
    norm = Normalize(0, vmax)
    regions.plot(
        ax=ax,
        column="load_shedding_TWh",
        cmap="YlOrRd",
        norm=norm,
        edgecolor="#888888",
        linewidth=0.45,
    )
    # Fixed European extent avoids remote overseas geometries dominating the map.
    extent = gpd.GeoSeries(
        gpd.points_from_xy([-12, -12, 35, 35, 10], [34, 72, 34, 72, 34]), crs=4326
    ).to_crs(CRS)
    ax.set_xlim(extent.x.min() - 250000, extent.x.max() + 250000)
    ax.set_ylim(extent.y.min() - 150000, extent.y.max() + 150000)
    max_radius = 145000
    for location, point in data["points"].items():
        values = mix.loc[location]
        total = values.clip(lower=0).sum()
        if total <= 1e-8:
            continue
        radius = max_radius * np.sqrt(total / pie_max)
        pie(ax, values, point.x, point.y, radius, colors)
        ax.annotate(
            location,
            (point.x, point.y),
            xytext=(0, 3 + radius / 18000),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            zorder=4,
        )
    ax.set_axis_off()
    shown = data["table"].loc[data["table"].included_in_map, "load_shedding_TWh"].sum()
    title = (
        f"Mapped load shedding: {shown:.1f} TWh"
        if data["europe_only"]
        else f"Load shedding: {data['total']:.1f} TWh"
    )
    ax.set_title(f"{data['case']}\n{title}", fontsize=15)
    cax = fig.add_axes([0.07, 0.095, 0.47, 0.018])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"),
        cax=cax,
        orientation="horizontal",
        label="Regional load shedding [TWh]",
    )
    handles = [
        Patch(facecolor=color, label=carrier) for carrier, color in colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.59, 0.90),
        title="Bus carrier",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    size_ax = fig.add_axes([0.63, 0.26, 0.30, 0.14])
    size_ax.set_xlim(-1.2, 4.2)
    size_ax.set_ylim(-1.4, 1.6)
    size_ax.set_aspect("equal")
    fig.canvas.draw()
    map_radius_pixels = (
        ax.transData.transform((max_radius, 0))[0] - ax.transData.transform((0, 0))[0]
    )
    legend_unit_pixels = (
        size_ax.transData.transform((1, 0))[0] - size_ax.transData.transform((0, 0))[0]
    )
    for x, fraction in [(0, 0.25), (2.8, 1.0)]:
        size_ax.add_patch(
            plt.Circle(
                (x, 0),
                np.sqrt(fraction) * map_radius_pixels / legend_unit_pixels,
                color="#888888",
            )
        )
        size_ax.text(x, -1.25, f"{pie_max * fraction:.1f} TWh", ha="center", fontsize=9)
    size_ax.set_title("Pie area = total shedding", fontsize=10)
    size_ax.set_axis_off()
    eu = mix.loc["EU"] if "EU" in mix.index else pd.Series(dtype=float)
    eu_total = eu.sum()
    eu_ax = fig.add_axes([0.69, 0.05, 0.19, 0.17])
    eu_ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
    fig.canvas.draw()
    eu_unit_pixels = (
        eu_ax.transData.transform((1, 0))[0] - eu_ax.transData.transform((0, 0))[0]
    )
    pie(
        eu_ax,
        eu,
        0,
        0,
        np.sqrt(max(eu_total, 0) / pie_max) * map_radius_pixels / eu_unit_pixels,
        colors,
    )
    eu_ax.set_title(
        f"EU-wide: {eu_total:.1f} TWh\nNot allocated to regions", fontsize=9
    )
    eu_ax.set_axis_off()
    if data["europe_only"]:
        eu_ax.remove()
        fig.text(
            0.64,
            0.17,
            f"Excluded from map: {data['total'] - shown:.1f} TWh\n"
            "Global / outside European map area; retained in CSVs.",
            fontsize=9,
        )
    fig.text(
        0.10,
        0.025,
        "Weighted load-slack dispatch; bus carriers retained.\n"
        "Sector buses grouped at their network location. Shared scales across cases.",
        fontsize=9,
    )
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", data["case"]).strip("_")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"load_shedding_map_{stem}.{suffix}", dpi=220)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def plot_comparison(data, output, absolute=False, vmax=None):
    """Compare magnitude and normalized regional patterns without conflating them."""
    values = pd.DataFrame(
        {d["case"]: d["regions"].load_shedding_TWh for d in data}
    ).T.fillna(0)
    totals = values.sum(axis=1)
    shares = values.div(totals.where(totals > 1e-6), axis=0) * 100
    if absolute:
        values.to_csv(output / "regional_energy_TWh.csv", index_label="case")
    else:
        shares.to_csv(output / "regional_shares_percent.csv", index_label="case")
    fig, (ax, bars) = plt.subplots(
        1,
        2,
        figsize=(17, max(7, len(data) * 0.28)),
        gridspec_kw={"width_ratios": [5, 1]},
        layout="constrained",
    )
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(
        (values if absolute else shares).to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=90)
    ax.set_yticks(range(len(values)), values.index, fontsize=8)
    ax.set_title(
        "Regional load shedding [TWh]"
        if absolute
        else "Regional share of mapped load shedding [%]\nGrey = negligible total (≤ 0.000001 TWh)"
    )
    fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        shrink=0.65,
        pad=0.04,
        label="Load shedding [TWh]" if absolute else "Regional share [%]",
    )
    bars.barh(range(len(totals)), totals, color="#637b91")
    bars.set_ylim(len(totals) - 0.5, -0.5)
    bars.set_yticks([])
    bars.set_xlabel("Mapped total [TWh]")
    bars.set_title("Magnitude")
    for y, value in enumerate(totals):
        bars.text(value, y, f" {value:.1f}", va="center", fontsize=7)
    bars.set_xlim(0, max(totals.max() * 1.25, 1e-6))
    for suffix in ("png", "pdf"):
        stem = "regional_comparison_absolute" if absolute else "regional_comparison"
        fig.savefig(output / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--cases", type=Path, default=DEFAULT / "selected_cross_tests.csv"
    )
    source.add_argument("--network", type=Path)
    source.add_argument(
        "--operating-scenario",
        help="Discover all cross-tests for this operating scenario, plus its diagonal baseline",
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results/cutouts_det_capexp_nols"
    )
    parser.add_argument("--year", default="2050")
    parser.add_argument(
        "--europe-only",
        action="store_true",
        help="Only map regional locations within 12W–35E, 34–72N; omit global pies",
    )
    parser.add_argument(
        "--regions", type=Path, help="Override inferred case-specific onshore regions"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT / "load_shedding_maps")
    args = parser.parse_args()
    cases = (
        pd.DataFrame([dict(case=args.network.stem, network=str(args.network))])
        if args.network
        else pd.DataFrame()
        if args.operating_scenario
        else pd.read_csv(args.cases)
    )
    if args.operating_scenario:
        paths = sorted(
            args.results_root.glob(
                f"*/networks/base_s_adm___{args.year}__cap-*__op-{args.operating_scenario}.nc"
            )
        )
        rows = [
            dict(
                case=f"{p.name.split('__cap-')[1].split('__op-')[0]} → {args.operating_scenario}",
                network=str(p),
            )
            for p in paths
        ]
        diagonal = (
            args.results_root
            / args.operating_scenario
            / "networks"
            / f"base_s_adm___{args.year}.nc"
        )
        if diagonal.exists():
            rows.append(
                dict(
                    case=f"{args.operating_scenario} → {args.operating_scenario}",
                    network=str(diagonal),
                )
            )
        cases = pd.DataFrame(rows).sort_values("case") if rows else pd.DataFrame()
    if cases.empty:
        raise ValueError("No cases selected")
    data = []
    for row in cases.to_dict("records"):
        network = resolve(row["network"])
        data.append(
            prepare(
                row["case"],
                network,
                args.regions or region_path(network),
                row.get("load_curtailment_TWh"),
                europe_only=args.europe_only,
            )
        )
    carriers = sorted(
        {c for d in data for c in d["mix"].columns if d["mix"][c].sum() > 1e-8}
    )
    palette = plt.get_cmap("tab20")
    colors = {
        c: data[0]["colors"].get(c) or palette(i % 20) for i, c in enumerate(carriers)
    }
    vmax = max(max(d["regions"].load_shedding_TWh.max() for d in data), 1e-6)
    pie_max = max(max(d["mix"].clip(lower=0).sum(axis=1).max() for d in data), 1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output / "selected_cases.csv", index=False)
    tables, regions, summaries = [], [], []
    combined_pdf = PdfPages(args.output / "all_load_shedding_maps.pdf")
    for d in data:
        d["mix"] = d["mix"].reindex(columns=carriers, fill_value=0)
        plot_case(d, args.output, colors, vmax, pie_max, combined_pdf)
        tables.append(d["table"].assign(case=d["case"]))
        regions.append(
            d["regions"][["load_shedding_TWh"]].reset_index().assign(case=d["case"])
        )
        regional = (
            d["table"].loc[d["table"].mapped_to_region, "load_shedding_TWh"].sum()
        )
        summaries.append(
            dict(
                case=d["case"],
                total_TWh=d["total"],
                regional_TWh=regional,
                nonregional_TWh=d["total"] - regional,
                mapped_TWh=d["table"]
                .loc[d["table"].included_in_map, "load_shedding_TWh"]
                .sum(),
                network=d["network"],
            )
        )
    combined_pdf.close()
    plot_comparison(data, args.output)
    plot_comparison(data, args.output, absolute=True)
    pd.concat(tables).to_csv(args.output / "load_shedding_by_bus.csv", index=False)
    pd.concat(regions).to_csv(args.output / "load_shedding_by_region.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "case_totals.csv", index=False)
    (args.output / "README.md").write_text(
        "# Geographic load-shedding diagnostics\n\n"
        "Regional shading: total weighted load-slack dispatch [TWh]. Pie area: total at a "
        "network location; slices: connected bus carriers. "
        + (
            "Only regional locations within 12W–35E and 34–72N are plotted; global buses and overseas polygons are excluded. Excluded values remain in bus CSVs. "
            if args.europe_only
            else "EU-wide slack is shown separately. "
        )
        + "All maps share linear colour and pie-area scales. Zero regions remain visible.\n\n"
        "Accounting matches plot_validation_heatmaps.py (carrier=load, objective snapshot "
        "weights preferred, scale 1e6); it is not necessarily unmet final demand. "
        "Selected-case totals are checked against the input CSV before plotting. "
        "Bus and region CSVs retain the underlying values.\n\n"
        "The regional comparison separates percentage shares from absolute mapped totals. "
        "All maps are also collected in all_load_shedding_maps.pdf.\n\n"
        "Reproduce from the repository root:\n\n"
        f"```bash\npython {shlex.join(sys.argv)}\n```\n"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Saved maps and data to {args.output}")


if __name__ == "__main__":
    main()
