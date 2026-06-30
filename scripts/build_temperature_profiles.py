# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build time series for air and soil temperatures per clustered model region.

Uses ``atlite.Cutout.temperature`` and ``atlite.Cutout.soil_temperature compute temperature ambient air and soil temperature for the respective cutout. The rule is executed in ``build_sector.smk``.


.. seealso::
    `Atlite.Cutout.temperature <https://atlite.readthedocs.io/en/master/ref_api.html#module-atlite.convert>`_
    `Atlite.Cutout.soil_temperature <https://atlite.readthedocs.io/en/master/ref_api.html#module-atlite.convert>`_

"""

import logging

import geopandas as gpd
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_temperature_profiles",
            clusters=50,
            configfiles=["config/cutouts_prices_uncertainty/cutouts_det_capexp.yaml"],
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nprocesses = int(snakemake.threads)
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    print("snapshots param:", snakemake.params.snapshots)
    print("drop_leap_day:", snakemake.params.drop_leap_day)
    print("n snapshots:", len(time))
    print("first snapshots:", time[:5])
    print("last snapshots:", time[-5:])
    print("cutout path:", snakemake.input.cutout)

    cutout = load_cutout(snakemake.input.cutout, time=time)

    print("cutout time size:", cutout.data.sizes.get("time"))
    print("cutout first times:", cutout.data.time.values[:5])
    print("cutout last times:", cutout.data.time.values[-5:])

    clustered_regions = (
        gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0)
    )

    I = cutout.indicatormatrix(clustered_regions)  # noqa: E741

    pop_layout = xr.open_dataarray(snakemake.input.pop_layout)

    stacked_pop = pop_layout.stack(spatial=("y", "x"))
    M = I.T.dot(np.diag(I.dot(stacked_pop)))

    nonzero_sum = M.sum(axis=0, keepdims=True)
    nonzero_sum[nonzero_sum == 0.0] = 1.0
    M_tilde = M / nonzero_sum

    temp_air = cutout.temperature(
        matrix=M_tilde.T,
        index=clustered_regions.index,
        dask_kwargs=dict(scheduler=client),
        show_progress=False,
    )

    temp_air.to_netcdf(snakemake.output.temp_air)

    temp_soil = cutout.soil_temperature(
        matrix=M_tilde.T,
        index=clustered_regions.index,
        dask_kwargs=dict(scheduler=client),
        show_progress=False,
    )

    temp_soil.to_netcdf(snakemake.output.temp_soil)
