#!/usr/bin/env python3
"""Relate country energy load slack to wind availability and prescribed loads."""

import argparse
import calendar
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.colors import TwoSlopeNorm
from plot_demand_normalized_load_shedding import (
    demand,
    energy_buses,
)
from plot_demand_normalized_load_shedding import (
    plot as plot_normalized,
)
from plot_load_shedding_timeseries import COUNTRY_NAMES
from plot_validation_heatmaps import LOAD_CURTAILMENT_CARRIERS, _get_snapshot_weightings

ROOT = Path(__file__).resolve().parents[2]
COUNTRIES = ["DE", "IT", "SE"]
METRICS = {
    "shedding_GW": "All energy load slack [GW]",
    "onshore_cf": "Onshore wind CF",
    "offshore_cf": "Offshore wind CF",
    "electrical_load_GW": "Electrical loads [GW]",
    "thermal_load_GW": "Thermal loads [GW]",
}


def dense(static, dynamic, field, ids, snapshots):
    result = pd.DataFrame(
        np.tile(static.loc[ids, field].to_numpy(), (len(snapshots), 1)),
        index=snapshots,
        columns=ids,
    )
    columns = ids.intersection(dynamic.columns)
    result.loc[:, columns] = dynamic.reindex(snapshots)[columns]
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"Missing {field} values")
    return result


def extract(directory):
    out = directory / "load_shedding_drivers"
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(directory / "load_shedding/selected_cases.csv")
    expected = pd.read_csv(
        directory / "load_shedding_normalized/annual_country_ratios.csv"
    ).set_index(["case", "location"])
    daily_rows, monthly_rows, annual_rows, capacity_rows = [], [], [], []
    for row in cases.itertuples():
        n = pypsa.Network(row.network)
        weights = _get_snapshot_weightings(n).reindex(n.snapshots)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid snapshot durations")
        locations = n.buses.location.fillna("")
        locations = locations.where(
            locations.ne(""), pd.Series(n.buses.index, index=n.buses.index)
        )
        energy = energy_buses(n)
        gen_loc = n.generators.bus.map(locations)
        load_loc = n.loads.bus.map(locations)
        units = n.loads.bus.map(n.buses.unit)
        annual_load, monthly_load, _, _ = demand(
            n, pd.Index(COUNTRIES, name="location")
        )
        for country in COUNTRIES:
            frame = pd.DataFrame(index=n.snapshots)
            gens = n.generators.index[
                gen_loc.eq(country)
                & n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)
                & n.generators.bus.map(energy)
            ]
            dispatch = n.generators_t.p.reindex(index=n.snapshots, columns=gens)
            if not np.isfinite(dispatch.to_numpy()).all():
                raise ValueError("Missing load-slack dispatch")
            frame["shedding_GW"] = dispatch.sum(axis=1) / 1000
            annual_slack = frame.shedding_GW.dot(weights) / 1000
            if not np.isclose(
                annual_slack,
                expected.loc[(row.case, country), "energy_slack_TWh"],
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("Country slack differs from normalized map data")
            for key, technology in [
                ("onshore_cf", "onwind"),
                ("offshore_cf", "offwind"),
            ]:
                ids = n.generators.index[
                    gen_loc.eq(country)
                    & n.generators.carrier.str.startswith(technology)
                ]
                capacity = (
                    n.generators.loc[ids, "p_nom_opt"]
                    .fillna(n.generators.loc[ids, "p_nom"])
                    .clip(lower=0)
                )
                capacity = capacity[capacity > 1e-6]
                if not np.isfinite(capacity).all():
                    raise ValueError("Invalid wind capacity")
                capacity_rows.append(
                    dict(
                        case=row.case,
                        location=country,
                        technology=technology,
                        capacity_MW=capacity.sum(),
                    )
                )
                if capacity.empty:
                    frame[key] = np.nan
                else:
                    availability = dense(
                        n.generators,
                        n.generators_t.p_max_pu,
                        "p_max_pu",
                        capacity.index,
                        n.snapshots,
                    )
                    if (availability < -1e-8).any().any() or (
                        availability > 1 + 1e-8
                    ).any().any():
                        raise ValueError("Wind availability outside [0,1]")
                    frame[key] = availability.dot(capacity) / capacity.sum()
            for key, mask in [
                ("electrical_load_GW", units.eq("MWh_el")),
                (
                    "thermal_load_GW",
                    n.loads.bus.map(n.buses.carrier).isin(
                        ["rural heat", "urban central heat", "urban decentral heat"]
                    ),
                ),
            ]:
                ids = n.loads.index[load_loc.eq(country) & mask]
                power = (
                    dense(n.loads, n.loads_t.p_set, "p_set", ids, n.snapshots)
                    .mul(-n.loads.loc[ids, "sign"], axis=1)
                    .clip(lower=0)
                )
                frame[key] = power.sum(axis=1) / 1000
            weighted = frame.mul(weights, axis=0)
            monthly_hours = weights.resample("MS").sum()
            monthly = (
                weighted.resample("MS").sum(min_count=1).div(monthly_hours, axis=0)
            )
            monthly["total_load_TWh"] = monthly_load[country].reindex(monthly.index)
            monthly["snapshot_hours"] = monthly_hours
            daily = (
                weighted.resample("D")
                .sum(min_count=1)
                .div(weights.resample("D").sum(), axis=0)
            )
            means = weighted.sum(min_count=1) / weights.sum()
            annual_rows.append(
                dict(
                    case=row.case,
                    location=country,
                    **means.to_dict(),
                    energy_slack_TWh=annual_slack,
                    total_load_TWh=annual_load[country],
                    snapshot_hours=weights.sum(),
                )
            )
            monthly_rows.append(
                monthly.rename_axis("timestamp")
                .reset_index()
                .assign(case=row.case, location=country)
            )
            daily_rows.append(
                daily.rename_axis("timestamp")
                .reset_index()
                .assign(case=row.case, location=country)
            )
        print(f"Extracted wind/load drivers: {row.case}", flush=True)
    pd.concat(daily_rows).to_csv(out / "daily_profiles.csv.gz", index=False)
    pd.concat(monthly_rows).to_csv(out / "monthly_profiles.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(out / "annual_means.csv", index=False)
    pd.DataFrame(capacity_rows).to_csv(out / "wind_capacity_weights.csv", index=False)


def read(directory):
    out = directory / "load_shedding_drivers"
    monthly = pd.read_csv(out / "monthly_profiles.csv", parse_dates=["timestamp"])
    annual = pd.read_csv(out / "annual_means.csv").set_index(["case", "location"])
    monthly["month"] = monthly.timestamp.dt.month
    pu = monthly.copy()
    for metric in METRICS:
        denominator = pd.MultiIndex.from_frame(monthly[["case", "location"]]).map(
            annual[metric]
        )
        threshold = 1e-6 * 1000 / 8760 if metric == "shedding_GW" else 1e-10
        pu[metric] = monthly[metric].to_numpy() / np.where(
            denominator > threshold, denominator, np.nan
        )
    pu.to_csv(out / "monthly_profiles_pu.csv", index=False)
    return dict(
        directory=directory,
        output=out,
        monthly=monthly,
        pu=pu,
        daily=pd.read_csv(out / "daily_profiles.csv.gz", parse_dates=["timestamp"]),
        cases=pd.read_csv(directory / "load_shedding/selected_cases.csv").case.tolist(),
    )


def seasonality(data, scales, pu=False):
    fig, axes = plt.subplots(3, 5, figsize=(25, 23), layout="constrained")
    frame = data["pu"] if pu else data["monthly"]
    for col, (metric, title) in enumerate(METRICS.items()):
        cmap = plt.get_cmap("RdBu_r" if pu else "YlOrRd").copy()
        cmap.set_bad("#dddddd")
        color_scale = (
            {"norm": TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(1.01, scales[metric]))}
            if pu
            else {"vmin": 0, "vmax": scales[metric]}
        )
        for label, country, ax in zip("abc", COUNTRIES, axes[:, col]):
            values = (
                frame.loc[frame.location.eq(country)]
                .pivot(index="case", columns="month", values=metric)
                .reindex(data["cases"])
            )
            im = ax.imshow(values, aspect="auto", cmap=cmap, **color_scale)
            ax.set_xticks(
                range(12), list(calendar.month_abbr)[1:], rotation=90, fontsize=8
            )
            ax.set_yticks(
                range(len(values)), values.index if col == 0 else [], fontsize=7
            )
            display_title = title.replace(" [GW]", "") + " [p.u.]" if pu else title
            ax.set_title(
                f"({label}) {COUNTRY_NAMES[country]}\n{display_title}", fontsize=11
            )
        fig.colorbar(
            im,
            ax=axes[:, col],
            orientation="horizontal",
            shrink=0.85,
            pad=0.02,
            label="Monthly mean / annual mean [p.u.]" if pu else title,
        )
    year = data["monthly"].timestamp.dt.year.iloc[0]
    fig.suptitle(
        f"{year}: country load shedding, wind availability and prescribed demand"
        + (" — per-unit seasonal profiles" if pu else " — monthly weighted means"),
        fontsize=16,
    )
    fig.supxlabel(
        "Wind CF: installed-capacity-weighted potential availability. Grey: unavailable CF or negligible annual mean.\nElectrical load includes EV demand; excludes endogenous conversion consumption. Thermal load is prescribed heat demand. Associations do not establish causality.",
        fontsize=10,
    )
    stem = "seasonality_comparison_pu" if pu else "seasonality_comparison_drivers"
    save(fig, data["output"] / stem)


def save(fig, path):
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=170)
    plt.close(fig)


def relationships(data):
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), layout="constrained")
    predictors = list(METRICS)[1:]
    records = []
    for country, row_axes in zip(COUNTRIES, axes):
        local = data["daily"].loc[data["daily"].location.eq(country)]
        for predictor, ax in zip(predictors, row_axes):
            for is_stoch, color, label in [
                (False, "#47799c", "Deterministic"),
                (True, "#dc7c32", "Stochastic"),
            ]:
                points = local.loc[
                    local.case.str.startswith("stochastic").eq(is_stoch)
                ].dropna(subset=[predictor, "shedding_GW"])
                ax.scatter(
                    points[predictor],
                    points.shedding_GW,
                    s=3,
                    alpha=0.16,
                    color=color,
                    label=label,
                    rasterized=True,
                )
            ax.set_xlabel(METRICS[predictor])
            ax.set_ylabel(f"{COUNTRY_NAMES[country]} load slack [GW]")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
            for case, values in local.groupby("case", sort=False):
                values = values.dropna(subset=[predictor, "shedding_GW"])
                correlation = (
                    values[predictor].corr(values.shedding_GW, method="spearman")
                    if len(values) > 2
                    and values[predictor].std() > 1e-10
                    and values.shedding_GW.std() > 1e-10
                    else np.nan
                )
                records.append(
                    dict(
                        case=case,
                        location=country,
                        predictor=predictor,
                        daily_spearman_r=correlation,
                        n_days=len(values),
                    )
                )
    axes[0, 0].legend(markerscale=3)
    fig.suptitle("Daily load shedding versus wind CF and prescribed loads", fontsize=16)
    fig.supxlabel(
        "One point per case-day. Capacity portfolios differ across cases; serial and seasonal dependence remain. Descriptive associations, not causal effects.",
        fontsize=10,
    )
    save(fig, data["output"] / "daily_driver_relationships")
    pd.DataFrame(records).to_csv(
        data["output"] / "within_case_daily_correlations.csv", index=False
    )


def normalized_seasonal(data):
    directory = data["directory"]
    ts, out = directory / "load_shedding_ts", directory / "load_shedding_normalized"
    selection = pd.read_csv(ts / "selected_countries_and_carriers.csv")
    if selection.location.drop_duplicates().tolist() != COUNTRIES:
        raise ValueError("Regenerate DE IT SE time series before rendering drivers")
    selected = (
        pd.read_csv(ts / "monthly_energy_TWh.csv", parse_dates=["timestamp"])
        .groupby(["case", "location", "timestamp"])
        .value.sum()
        .rename("selected_carrier_slack_TWh")
    )
    denominator = (
        data["monthly"]
        .set_index(["case", "location", "timestamp"])
        .total_load_TWh.rename("modeled_load_TWh")
    )
    monthly = pd.concat([selected, denominator], axis=1)
    if monthly.isna().any().any():
        raise ValueError("Missing selected-carrier or denominator data")
    monthly["slack_to_load_percent"] = (
        100
        * monthly.selected_carrier_slack_TWh
        / monthly.modeled_load_TWh.where(monthly.modeled_load_TWh > 0)
    )
    monthly = monthly.reset_index()
    monthly.to_csv(out / "monthly_country_ratios.csv", index=False)
    selection.to_csv(out / "selected_countries_and_carriers.csv", index=False)
    return dict(
        output=out,
        annual=pd.read_csv(out / "annual_country_ratios.csv"),
        monthly=monthly,
        cases=data["cases"],
        selection=selection,
    )


def selected_regions(data):
    """Make the explicitly requested three-country regional comparisons."""
    for field, stem, title in [
        (
            "energy_slack_TWh",
            "regional_comparison_selected_absolute",
            "Energy load slack [TWh]",
        ),
        (
            "slack_to_load_percent",
            "regional_comparison_selected_normalized",
            "Energy load slack / country loads [%]",
        ),
    ]:
        vmax = max(
            d["annual"].loc[d["annual"].location.isin(COUNTRIES), field].max()
            for d in data
        )
        for d in data:
            values = (
                d["annual"]
                .pivot(index="case", columns="location", values=field)
                .reindex(index=d["cases"], columns=COUNTRIES)
            )
            fig, ax = plt.subplots(figsize=(8, 12), layout="constrained")
            im = ax.imshow(
                values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(vmax, 1e-6)
            )
            ax.set_xticks(range(3), [COUNTRY_NAMES[c] for c in COUNTRIES])
            ax.set_yticks(range(len(values)), values.index, fontsize=8)
            for i in range(len(values)):
                for j in range(3):
                    value = values.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > vmax * 0.60 else "black",
                    )
            ax.set_title(title)
            fig.colorbar(im, ax=ax, orientation="horizontal", label=title, shrink=0.8)
            save(fig, d["output"] / stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    plt.rcParams["figure.autolayout"] = False
    directories = [
        ROOT / "results/cutouts_det_capexp_nols" / f"d_{year}" for year in (2010, 2022)
    ]
    if not args.render_only:
        for directory in directories:
            extract(directory)
    if args.extract_only:
        return
    data = [read(directory) for directory in directories]
    for pu in (False, True):
        scales = {
            metric: max(
                max(d["pu" if pu else "monthly"][metric].max() for d in data), 1e-6
            )
            for metric in METRICS
        }
        for d in data:
            seasonality(d, scales, pu)
    normalized = [normalized_seasonal(d) for d in data]
    selected_regions(normalized)
    for d in normalized:
        plot_normalized(
            d,
            max(v["annual"].slack_to_load_percent.max() for v in normalized),
            max(v["monthly"].slack_to_load_percent.max() for v in normalized),
        )
    for d in data:
        relationships(d)
        (d["output"] / "README.md").write_text(
            "# Country wind/load context for load shedding\n\n"
            "Countries: Germany, Italy, Sweden, in this order. All 39 capacity cases per operating year. "
            "All-country maps and regional matrices retain all countries.\n\n"
            "Shedding is ALL local energy-bus load slack (not just the selected three carriers), reconciled "
            "to annual_country_ratios.csv. Non-energy and EU-wide buses excluded.\n\n"
            "Wind CF = sum(p_nom_opt * p_max_pu) / sum(p_nom_opt) over country generators, onwind separately "
            "from all offwind variants. Missing optimized capacity falls back to p_nom; zero installed capacity "
            "gives undefined CF (grey), not zero. This is potential availability, not dispatched generation. "
            "Case-specific installed-capacity weighting means portfolio changes can affect the country CF.\n\n"
            "Electrical loads = positive prescribed MWh_el Load consumption, including EV loads, industry "
            "and agriculture. Endogenous electricity consumption of heat pumps/electrolysers is not included. "
            "Thermal loads = prescribed rural/urban-central/urban-decentral heat Loads, including industry heat. "
            "Neither load series is inferred from actual production. Static and dynamic setpoints are combined.\n\n"
            "Daily/monthly means use objective-first snapshot-hour weighting. Per-unit is monthly mean divided "
            "by the SAME case/country/metric annual weighted mean: 1 = annual mean. Near-zero shedding annual "
            "energy (<=1e-6 TWh) and zero/missing wind capacity produce grey cells. Scales match across years. "
            "Monthly profiles and installed capacities are exported for audit.\n\n"
            "Daily scatterplots pool case-days, while correlation CSVs report within-case Spearman correlations "
            "without significance claims. Correlations do not establish causes: network transfers, available "
            "generation, storage and conversion limits can mediate shedding. Useful follow-up: inspect hours "
            "with low wind AND high loads together, then check imports, storage state and saturated converters.\n\n"
            "Reproduce: python scripts/validation_scripts/plot_load_shedding_drivers.py. "
            "Use --extract-only to cache data, --render-only to reuse exports. First regenerate original "
            "time series with --countries DE IT SE for both operating years.\n"
        )
        print(f"Saved driver diagnostics to {d['output']}", flush=True)


if __name__ == "__main__":
    main()
