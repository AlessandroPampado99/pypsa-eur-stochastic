import logging

import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_population_weighted_energy_totals",
            kind="heat",
            clusters="adm",
            configfiles=["config/cutouts_prices_uncertainty/cutouts_det_capexp.yaml"],
            run="d_2024",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config["energy"]

    if snakemake.wildcards.kind == "heat":
        snapshots = get_snapshots(
            snakemake.params.snapshots, snakemake.params.drop_leap_day
        )
        data_years = pd.Index(snapshots.year.unique(), name="year")
    else:
        data_years = pd.Index([int(config["energy_totals_year"])], name="year")

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    # Ensure that the year level is numeric, so it can be compared with snapshot years.
    totals.index = pd.MultiIndex.from_arrays(
        [
            totals.index.get_level_values(0),
            totals.index.get_level_values(1).astype(int),
        ],
        names=totals.index.names,
    )

    available_years = totals.index.get_level_values(1).unique()
    max_available_year = int(available_years.max())

    # Use the latest available totals year whenever the requested year is beyond the data.
    effective_data_years = pd.Index(
        [min(int(year), max_available_year) for year in data_years],
        name="year",
    ).unique()

    capped_years = data_years[data_years.astype(int) > max_available_year]
    if len(capped_years) > 0:
        logger.warning(
            "Requested energy totals years %s are above the maximum available year %s. "
            "Using %s instead.",
            list(capped_years.astype(int)),
            max_available_year,
            max_available_year,
        )

    totals = totals.loc[idx[:, effective_data_years], :].groupby(level=0).mean()

    nodal_totals = totals.loc[pop_layout.ct].fillna(0.0)
    nodal_totals.index = pop_layout.index
    nodal_totals = nodal_totals.multiply(pop_layout.fraction, axis=0)

    nodal_totals.to_csv(snakemake.output[0])