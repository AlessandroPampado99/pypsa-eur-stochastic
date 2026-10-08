#!/usr/bin/env python3
"""
Plot seasonal load-slack profiles for the cases used by the geographic maps.

Rank countries and bus carriers using summed weighted energy across cases.
Plot daily and centred seven-day weighted mean dispatch; export native dispatch
and monthly energy. Run with the pypsa-eur environment.
"""

import argparse
import calendar
import re
import shlex
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.backends.backend_pdf import PdfPages
from plot_validation_heatmaps import (
    LOAD_CURTAILMENT_CARRIERS,
    LOAD_CURTAILMENT_SCALE,
    _get_snapshot_weightings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/cutouts_det_capexp_nols/d_2010"
COUNTRY_NAMES = {"DE": "Germany", "IT": "Italy", "FR": "France", "SE": "Sweden"}


def rank_selection(bus, mode, locations=None):
    """Use mapped European buses only, retaining bus carrier labels."""
    mapped = bus.loc[bus.included_in_map]
    totals = mapped.groupby("location").load_shedding_TWh.sum()
    countries = totals.reindex(locations) if locations else totals.nlargest(3)
    if countries.isna().any() or not countries.index.is_unique:
        raise ValueError("Country selection contains unknown or duplicate locations")
    if len(countries) != 3 or (countries <= 0).any():
        raise ValueError("Need three countries with positive shedding")
    global_carriers = (
        mapped.groupby("carrier").load_shedding_TWh.sum().nlargest(3).index
    )
    rows = []
    for country_rank, country in enumerate(countries.index, 1):
        values = (
            mapped.loc[mapped.location.eq(country)]
            .groupby("carrier")
            .load_shedding_TWh.sum()
        )
        carriers = values.nlargest(3).index if mode == "country" else global_carriers
        for carrier_rank, carrier in enumerate(carriers, 1):
            rows.append(
                dict(
                    location=country,
                    country_rank=country_rank,
                    carrier=carrier,
                    carrier_rank=carrier_rank,
                    pooled_TWh=values.get(carrier, 0),
                    country_total_TWh=countries[country],
                )
            )
    return pd.DataFrame(rows)


def extract_profiles(network, selection, expected):
    """Reconcile each selected location/carrier with the geographic CSV."""
    n = pypsa.Network(network)
    if (
        not isinstance(n.snapshots, pd.DatetimeIndex)
        or not n.snapshots.is_unique
        or not n.snapshots.is_monotonic_increasing
    ):
        raise ValueError("Profiles require unique, ordered datetime snapshots")
    weights = _get_snapshot_weightings(n).reindex(n.snapshots)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid snapshot weights")
    gens = n.generators.loc[n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)]
    locations = gens.bus.map(n.buses.location).fillna("")
    locations = locations.where(locations.ne(""), gens.bus)
    carriers = gens.bus.map(n.buses.carrier)
    profiles = {}
    reconciliation = []
    for row in selection.itertuples():
        ids = gens.index[locations.eq(row.location) & carriers.eq(row.carrier)]
        dispatch = n.generators_t.p.reindex(index=n.snapshots, columns=ids)
        if not np.isfinite(dispatch.to_numpy()).all():
            raise ValueError(f"Missing load dispatch for {row.location}, {row.carrier}")
        power = dispatch.sum(axis=1)
        energy = power.dot(weights) / LOAD_CURTAILMENT_SCALE
        target = expected.get((row.location, row.carrier), 0.0)
        if not np.isclose(energy, target, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Profile energy {energy} differs from mapped energy {target}"
            )
        profiles[(row.location, row.carrier)] = power / 1000
        reconciliation.append(
            dict(
                location=row.location,
                carrier=row.carrier,
                profile_TWh=energy,
                mapped_TWh=target,
            )
        )
    power = pd.DataFrame(profiles, index=n.snapshots)
    power.columns.names = ["location", "carrier"]
    power.index.name = "timestamp"
    weighted = power.mul(weights, axis=0)
    daily_hours = weights.resample("D").sum()
    daily = weighted.resample("D").sum().div(daily_hours, axis=0)
    smooth = (
        weighted.resample("D")
        .sum()
        .rolling(7, center=True, min_periods=1)
        .sum()
        .div(daily_hours.rolling(7, center=True, min_periods=1).sum(), axis=0)
    )
    monthly = weighted.resample("MS").sum() / 1000
    return dict(
        power=power,
        weights=weights,
        daily=daily,
        smooth=smooth,
        monthly=monthly,
        reconciliation=reconciliation,
    )


def plot_case(case, data, selection, colors, limits, output, pdf):
    countries = selection.location.drop_duplicates()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    for label, country, ax in zip("abc", countries, axes):
        for carrier in selection.loc[selection.location.eq(country), "carrier"]:
            key = (country, carrier)
            ax.plot(
                data["daily"].index,
                data["daily"][key],
                color=colors[carrier],
                alpha=0.20,
                linewidth=0.65,
            )
            ax.plot(
                data["smooth"].index,
                data["smooth"][key],
                color=colors[carrier],
                linewidth=1.6,
                label=carrier,
            )
        ax.set_title(
            f"({label}) {COUNTRY_NAMES.get(country, country)}",
            loc="left",
            fontweight="bold",
        )
        ax.set_ylabel("Load shedding [GW]")
        ax.set_ylim(0, limits[country])
        ax.grid(axis="y", alpha=0.20)
        ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlim(data["daily"].index.min(), data["daily"].index.max())
    axes[-1].set_xlabel(
        f"Operating year {data['daily'].index[0].year} · faint: daily mean · solid: centred 7-day mean\n"
        "Snapshot-weighted dispatch; fixed carrier selections and country scales across cases"
    )
    fig.suptitle(f"Seasonal load shedding · {case}", fontsize=15)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", case).strip("_")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"load_shedding_ts_{stem}.{suffix}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def seasonal_overview(data, selection, output, absolute=False, vmax=None):
    monthly_values = {
        country: pd.DataFrame(
            {
                case: d["monthly"][country].sum(axis=1).to_numpy()
                for case, d in data.items()
            }
        ).T
        for country in selection.location.drop_duplicates()
    }
    if absolute and vmax is None:
        vmax = max(max(values.max().max() for values in monthly_values.values()), 1e-6)
    if absolute:
        export = pd.concat(monthly_values, names=["location", "case"])
        export.columns = pd.Index(range(1, 13), name="month")
        export.to_csv(output / "seasonality_monthly_energy_TWh.csv")
    fig, axes = plt.subplots(3, 1, figsize=(13, 22), layout="constrained")
    for label, country, ax in zip("abc", selection.location.drop_duplicates(), axes):
        monthly = monthly_values[country]
        total = monthly.sum(axis=1)
        shares = monthly.div(total.where(total > 1e-6), axis=0) * 100
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(
            monthly if absolute else shares,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax if absolute else 100,
        )
        ax.set_xticks(range(12), list(calendar.month_abbr)[1:])
        ax.set_yticks(range(len(data)), data.keys(), fontsize=7)
        ax.set_title(
            f"({label}) {COUNTRY_NAMES.get(country, country)} · selected three carriers combined",
            loc="left",
        )
    fig.colorbar(
        im,
        ax=axes,
        label="Monthly load shedding [TWh]"
        if absolute
        else "Share of annual shedding in each month [%]",
        shrink=0.4,
        pad=0.02,
    )
    fig.suptitle(
        "Monthly load shedding across all cases [TWh]"
        if absolute
        else "Seasonality across all cases\nGrey: selected-carrier annual total ≤ 0.000001 TWh",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        stem = (
            "seasonality_comparison_absolute" if absolute else "seasonality_comparison"
        )
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT / "load_shedding")
    parser.add_argument("--output", type=Path, default=DEFAULT / "load_shedding_ts")
    parser.add_argument(
        "--countries", nargs=3, help="Explicit country order, e.g. DE IT SE"
    )
    parser.add_argument(
        "--carrier-ranking", choices=["country", "europe"], default="country"
    )
    args = parser.parse_args()
    plt.rcParams["figure.autolayout"] = False
    bus = pd.read_csv(args.input / "load_shedding_by_bus.csv")
    cases = pd.read_csv(args.input / "selected_cases.csv")
    selection = rank_selection(bus, args.carrier_ranking, args.countries)
    args.output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output / "selected_countries_and_carriers.csv", index=False)
    print(selection.to_string(index=False), flush=True)
    expected = bus.groupby(["case", "location", "carrier"]).load_shedding_TWh.sum()
    data = {}
    for row in cases.itertuples():
        target = (
            expected.loc[row.case]
            if row.case in expected.index.get_level_values("case")
            else pd.Series(dtype=float)
        )
        data[row.case] = extract_profiles(row.network, selection, target)
        print(f"Extracted and reconciled {row.case}", flush=True)
    reference = next(iter(data.values()))["power"].index
    year = reference[0].year
    expected_snapshots = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="3h", inclusive="left"
    )
    if not reference.equals(expected_snapshots):
        raise ValueError(f"Expected a complete three-hourly operating year {year}")
    if any(not d["power"].index.equals(reference) for d in data.values()):
        raise ValueError("Case timelines differ")
    colors = {
        carrier: plt.get_cmap("tab10")(i)
        for i, carrier in enumerate(selection.carrier.unique())
    }
    limits = {
        country: max(
            max(d["daily"][country].max().max() for d in data.values()) * 1.12, 0.01
        )
        for country in selection.location.unique()
    }
    with PdfPages(args.output / "all_load_shedding_timeseries.pdf") as pdf:
        for case, d in data.items():
            plot_case(case, d, selection, colors, limits, args.output, pdf)
    seasonal_overview(data, selection, args.output)
    seasonal_overview(data, selection, args.output, absolute=True)
    for key, filename in [
        ("power", "native_dispatch_GW.csv.gz"),
        ("daily", "daily_mean_GW.csv.gz"),
        ("smooth", "seven_day_mean_GW.csv.gz"),
        ("monthly", "monthly_energy_TWh.csv"),
    ]:
        long = pd.concat(
            {
                case: d[key].stack(["location", "carrier"], future_stack=True)
                for case, d in data.items()
            },
            names=["case"],
        )
        long.rename("value").to_csv(args.output / filename)
    pd.concat({case: d["weights"] for case, d in data.items()}, names=["case"]).rename(
        "snapshot_weight_hours"
    ).to_csv(args.output / "snapshot_weights.csv.gz")
    pd.concat(
        [
            pd.DataFrame(d["reconciliation"]).assign(case=case)
            for case, d in data.items()
        ]
    ).to_csv(args.output / "energy_reconciliation.csv", index=False)
    (args.output / "README.md").write_text(
        "# Seasonal load-shedding profiles\n\n"
        f"Countries: {', '.join(selection.location.unique())}; "
        f"{'explicit selection' if args.countries else 'top three by pooled shedding'} across {len(data)} mapped cases. "
        f"Carrier ranking: {args.carrier_ranking}; fixed across cases, see selected_countries_and_carriers.csv. "
        "Global and out-of-Europe buses excluded using the map CSV's included_in_map flag.\n\n"
        "Each figure has panels (a), (b), (c) for the ranked countries. Faint lines show daily weighted mean "
        "dispatch [GW]; solid lines show centred seven-day weighted means (shorter windows at year boundaries). "
        "Country y-scales and carrier colours are shared across cases. Profiles retain connected bus carrier labels; "
        "this is load-slack accounting, not necessarily unmet final demand.\n\n"
        "Native three-hourly dispatch, weights, daily means, seven-day means and monthly energy [TWh] are exported. "
        "All nine location/carrier annual energies per case reconcile with the geographic CSV. "
        "The comparison heatmaps show monthly shares of selected-carrier annual energy; negligible totals are grey.\n\n"
        f"```bash\npython {shlex.join(sys.argv)}\n```\n"
    )
    print(
        f"Saved {len(data)} seasonal figures and reconciled profiles to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
