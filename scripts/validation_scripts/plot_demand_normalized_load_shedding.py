#!/usr/bin/env python3
"""Compare country load slack relative to modeled energy Loads, not final demand."""

import argparse
import calendar
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from plot_load_shedding_timeseries import COUNTRY_NAMES
from plot_validation_heatmaps import _get_snapshot_weightings

ROOT = Path(__file__).resolve().parents[2]


def energy_buses(n):
    """Exclude mass flows; handle two legacy energy-bus unit labels explicitly."""
    units = n.buses.unit.fillna("")
    # land transport oil p_set is oil-energy demand (prepare_sector_network.py,
    # add_land_transport_ice: transport service / ICE efficiency).
    # Generic battery buses store electricity despite their empty unit attribute.
    return (
        units.str.startswith("MWh")
        | (n.buses.carrier.eq("land transport oil") & units.eq("land transport"))
        | (n.buses.carrier.eq("battery") & units.eq(""))
    )


def demand(n, countries):
    weights = _get_snapshot_weightings(n).reindex(n.snapshots)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid snapshot weights")
    energy = energy_buses(n)
    buses = n.buses.copy()
    buses["location"] = buses.location.fillna("")
    buses.loc[buses.location.eq(""), "location"] = buses.index[buses.location.eq("")]
    loads = n.loads.copy()
    loads["location"] = loads.bus.map(buses.location)
    loads["bus_unit"] = loads.bus.map(buses.unit)
    loads["included"] = loads.location.isin(countries) & loads.bus.map(energy)
    local = loads.location.isin(countries)
    unknown = local & ~loads.bus.map(energy) & ~loads.bus_unit.eq("t_co2")
    if unknown.any():
        raise ValueError(
            f"Unclassified load units: {loads.loc[unknown, ['bus', 'bus_unit']]}"
        )
    selected = loads.loc[loads.included]
    power = pd.DataFrame(
        np.tile(selected.p_set.to_numpy(), (len(n.snapshots), 1)),
        index=n.snapshots,
        columns=selected.index,
    )
    dynamic = power.columns.intersection(n.loads_t.p_set.columns)
    power.loc[:, dynamic] = n.loads_t.p_set.reindex(n.snapshots)[dynamic]
    power = power.mul(-selected.sign, axis=1)
    if not np.isfinite(power.to_numpy()).all():
        raise ValueError("Missing energy-load setpoints")
    # Negative Loads inject energy and do not represent demand.
    power = power.clip(lower=0)
    weighted = power.mul(weights, axis=0) / 1e6
    by_load = weighted.sum().rename("modeled_load_TWh")
    annual = by_load.groupby(selected.location).sum().reindex(countries, fill_value=0)
    monthly = (
        weighted.resample("MS")
        .sum()
        .T.groupby(selected.location)
        .sum()
        .T.reindex(columns=countries, fill_value=0)
    )
    audit = loads[["bus", "carrier", "location", "bus_unit", "included"]].join(by_load)
    return annual, monthly, audit, energy


def extract(directory):
    maps, ts = directory / "load_shedding", directory / "load_shedding_ts"
    cases = pd.read_csv(maps / "selected_cases.csv")
    bus = pd.read_csv(maps / "load_shedding_by_bus.csv")
    regions = pd.read_csv(maps / "load_shedding_by_region.csv")
    names = regions["name"]
    if "index" in regions:
        names = names.fillna(regions["index"])
    countries = pd.Index(sorted(names.dropna().unique()), name="location")
    selection = pd.read_csv(ts / "selected_countries_and_carriers.csv")
    monthly_slack = pd.read_csv(
        ts / "monthly_energy_TWh.csv", parse_dates=["timestamp"]
    )
    monthly_slack = monthly_slack.groupby(["case", "timestamp", "location"]).value.sum()
    annual_rows, monthly_rows, audits, unit_audits = [], [], [], []
    for row in cases.itertuples():
        n = pypsa.Network(row.network)
        annual, monthly, audit, energy = demand(n, countries)
        local_slack = bus.loc[bus.case.eq(row.case) & bus.included_in_map].copy()
        if not local_slack.empty:
            unknown = ~local_slack.bus.map(energy) & ~local_slack.bus.map(
                n.buses.unit
            ).eq("t_co2")
            if unknown.any():
                raise ValueError("Unclassified local slack units")
        numerator = (
            local_slack.loc[local_slack.bus.map(energy).fillna(False)]
            .groupby("location")
            .load_shedding_TWh.sum()
            .reindex(countries, fill_value=0)
        )
        frame = pd.DataFrame(
            {"energy_slack_TWh": numerator, "modeled_load_TWh": annual}
        )
        frame["slack_to_load_percent"] = 100 * numerator / annual.where(annual > 0)
        annual_rows.append(frame.reset_index().assign(case=row.case))
        for country in selection.location.unique():
            carriers = selection.loc[selection.location.eq(country), "carrier"]
            if not n.buses.loc[n.buses.carrier.isin(carriers)].index.map(energy).all():
                raise ValueError("Selected seasonal carrier has non-energy units")
            num = monthly_slack.xs((row.case, country), level=("case", "location"))
            den = monthly[country].reindex(num.index)
            if den.isna().any():
                raise ValueError("Missing monthly demand")
            values = pd.DataFrame(
                {"selected_carrier_slack_TWh": num, "modeled_load_TWh": den}
            )
            values["slack_to_load_percent"] = 100 * num / den.where(den > 0)
            monthly_rows.append(
                values.rename_axis("timestamp")
                .reset_index()
                .assign(case=row.case, location=country)
            )
        audits.append(audit.rename_axis("load").reset_index().assign(case=row.case))
        unit_audits.append(
            n.buses[["carrier", "unit"]]
            .assign(included_energy_unit=energy)
            .drop_duplicates()
            .assign(case=row.case)
        )
        print(f"Extracted modeled demand: {row.case}", flush=True)
    annual_data, monthly_data = pd.concat(annual_rows), pd.concat(monthly_rows)
    out = directory / "load_shedding_normalized"
    out.mkdir(parents=True, exist_ok=True)
    annual_data.to_csv(out / "annual_country_ratios.csv", index=False)
    monthly_data.to_csv(out / "monthly_country_ratios.csv", index=False)
    pd.concat(audits).to_csv(out / "load_denominator_audit.csv", index=False)
    pd.concat(unit_audits).to_csv(out / "bus_unit_audit.csv", index=False)
    cases.to_csv(out / "selected_cases.csv", index=False)
    selection.to_csv(out / "selected_countries_and_carriers.csv", index=False)
    return dict(
        directory=directory,
        output=out,
        annual=annual_data,
        monthly=monthly_data,
        cases=cases.case.tolist(),
        selection=selection,
    )


def plot(dataset, annual_max, monthly_max):
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#dddddd")
    annual = dataset["annual"]
    values = annual.pivot(
        index="case", columns="location", values="slack_to_load_percent"
    ).reindex(dataset["cases"])
    fig, ax = plt.subplots(figsize=(17, 12), layout="constrained")
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=annual_max)
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=90)
    ax.set_yticks(range(len(values)), values.index, fontsize=8)
    ax.set_title("Annual energy load slack / annual modeled country loads [%]")
    fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        shrink=0.65,
        pad=0.04,
        label="Load-slack-to-load ratio [%]",
    )
    fig.supxlabel(
        "Energy loads only; emissions and global buses excluded. Includes upstream/storage slack; not an unserved-final-demand fraction.",
        fontsize=9,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            dataset["output"] / f"regional_comparison_load_normalized.{suffix}", dpi=200
        )
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 22), layout="constrained")
    monthly = dataset["monthly"].copy()
    monthly["month"] = monthly.timestamp.dt.month
    for label, country, ax in zip("abc", dataset["selection"].location.unique(), axes):
        values = (
            monthly.loc[monthly.location.eq(country)]
            .pivot(index="case", columns="month", values="slack_to_load_percent")
            .reindex(dataset["cases"])
        )
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=monthly_max)
        ax.set_xticks(range(12), list(calendar.month_abbr)[1:])
        ax.set_yticks(range(len(values)), values.index, fontsize=7)
        ax.set_title(
            f"({label}) {COUNTRY_NAMES.get(country, country)} · selected three carriers combined",
            loc="left",
        )
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.4,
        pad=0.02,
        label="Selected-carrier slack / total country loads in the same month [%]",
    )
    fig.suptitle("Monthly load slack normalized by monthly country loads", fontsize=14)
    fig.supxlabel(
        "Modeled energy-load denominator; includes upstream/storage slack. Grey = no positive denominator.",
        fontsize=9,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            dataset["output"] / f"seasonality_comparison_load_normalized.{suffix}",
            dpi=180,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[
            ROOT / "results/cutouts_det_capexp_nols" / f"d_{year}"
            for year in (2010, 2022)
        ],
    )
    args = parser.parse_args()
    plt.rcParams["figure.autolayout"] = False
    datasets = [extract(directory) for directory in args.directories]
    annual_max = max(
        max(d["annual"].slack_to_load_percent.max() for d in datasets), 1e-6
    )
    monthly_max = max(
        max(d["monthly"].slack_to_load_percent.max() for d in datasets), 1e-6
    )
    for d in datasets:
        plot(d, annual_max, monthly_max)
        (d["output"] / "README.md").write_text(
            "# Load-slack-to-load ratios\n\n"
            "Regional numerator: annual carrier=load dispatch at mapped energy buses. "
            "Denominator: sum of positive prescribed Load consumption in the same country and solved case, "
            "using static p_set with time-varying overrides and the same objective-first snapshot weights "
            "as the original diagnostics. Includes electricity, thermal loads and fuel energy demand as modeled. "
            "This is a modeled energy-load total, not a verified final-energy demand total.\n\n"
            "Seasonal numerator: monthly slack from the original selected three carriers per country; "
            "denominator: ALL modeled energy loads in that country in the SAME month. "
            "Original year-specific country/carrier selections are preserved.\n\n"
            "MWh units count as energy. Legacy land transport oil units represent oil energy "
            "(transport service divided by ICE efficiency in prepare_sector_network.py); blank battery "
            "units are treated as electricity. t_co2 buses are excluded from both quantities, as are "
            "global and out-of-Europe locations. Negative Load injections are excluded. Undefined "
            "ratios are grey, while genuine zeros remain zero. The load and unit audit CSVs document the denominator.\n\n"
            "Slack can occur at upstream and storage buses and includes conversion/storage losses. "
            "Ratios are therefore NOT fractions of final demand unserved and are not capped at 100%. "
            "Absolute numerators can differ from the original maps because those retained non-energy slack. "
            "Colour scales are shared across both operating years.\n\n"
            "Reproduce: python scripts/validation_scripts/plot_demand_normalized_load_shedding.py\n"
        )
        print(f"Saved demand-normalized comparisons to {d['output']}", flush=True)


if __name__ == "__main__":
    main()
