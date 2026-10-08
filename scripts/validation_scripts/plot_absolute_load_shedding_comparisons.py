#!/usr/bin/env python3
"""Regenerate absolute regional and monthly comparisons from exported CSVs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plot_load_shedding_maps import plot_comparison
from plot_load_shedding_timeseries import seasonal_overview

ROOT = Path(__file__).resolve().parents[2]


def read_data(directory):
    """Preserve case, country and carrier selections used by the original plots."""
    maps = directory / "load_shedding"
    ts = directory / "load_shedding_ts"
    cases = pd.read_csv(maps / "selected_cases.csv").case
    regions = pd.read_csv(maps / "load_shedding_by_region.csv")
    # Earlier map exports used an unnamed index for some cases.
    if "index" in regions:
        regions["name"] = regions.get("name", regions["index"]).fillna(regions["index"])
    if regions.name.isna().any() or regions.duplicated(["case", "name"]).any():
        raise ValueError("Missing or duplicate regional labels")
    totals = pd.read_csv(maps / "case_totals.csv").set_index("case").mapped_TWh
    if not np.allclose(
        regions.groupby("case").load_shedding_TWh.sum().reindex(cases),
        totals.reindex(cases),
    ):
        raise ValueError("Regional CSV does not reconcile with mapped totals")
    regional_data = [
        dict(case=case, regions=regions.loc[regions.case.eq(case)].set_index("name"))
        for case in cases
    ]
    selection = pd.read_csv(ts / "selected_countries_and_carriers.csv")
    monthly = pd.read_csv(ts / "monthly_energy_TWh.csv", parse_dates=["timestamp"])
    profiles = {}
    for case in cases:
        values = (
            monthly.loc[monthly.case.eq(case)]
            .pivot(index="timestamp", columns=["location", "carrier"], values="value")
            .sort_index()
        )
        if len(values) != 12 or values.isna().any().any():
            raise ValueError(f"Incomplete monthly data for {case}")
        profiles[case] = dict(monthly=values)
    return dict(
        directory=directory,
        regions=regional_data,
        profiles=profiles,
        selection=selection,
    )


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
        help="Operating-scenario directories containing load_shedding and load_shedding_ts",
    )
    args = parser.parse_args()
    datasets = [read_data(directory) for directory in args.directories]
    regional_max = max(
        max(d["regions"].load_shedding_TWh.max() for d in dataset["regions"])
        for dataset in datasets
    )
    monthly_max = max(
        profile["monthly"][country].sum(axis=1).max()
        for dataset in datasets
        for profile in dataset["profiles"].values()
        for country in dataset["selection"].location.unique()
    )
    for dataset in datasets:
        directory = dataset["directory"]
        plot_comparison(
            dataset["regions"],
            directory / "load_shedding",
            absolute=True,
            vmax=max(regional_max, 1e-6),
        )
        seasonal_overview(
            dataset["profiles"],
            dataset["selection"],
            directory / "load_shedding_ts",
            absolute=True,
            vmax=max(monthly_max, 1e-6),
        )
        for folder in ("load_shedding", "load_shedding_ts"):
            (directory / folder / "absolute_comparisons_README.md").write_text(
                "# Absolute load-shedding comparisons\n\n"
                "The *_comparison_absolute.png and .pdf figures show TWh, with zero values retained. "
                "Regional values sum all mapped carriers. Monthly values sum the selected three "
                "carriers in each country, retaining the original per-year selections. "
                "These selections can differ between operating years.\n\n"
                f"Shared linear scales across {', '.join(str(d['directory'].name) for d in datasets)}: "
                f"regional 0–{regional_max:.6g} TWh; monthly 0–{monthly_max:.6g} TWh. "
                "The regional_energy_TWh.csv and seasonality_monthly_energy_TWh.csv matrices "
                "provide exact cell values in their respective folders.\n\n"
                "Run python scripts/validation_scripts/plot_absolute_load_shedding_comparisons.py "
                "from the repository root to regenerate the default 2010 and 2022 figures.\n"
            )
        print(
            f"Saved absolute regional and seasonal comparisons in {directory}",
            flush=True,
        )


if __name__ == "__main__":
    main()
