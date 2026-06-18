# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create static energy balance maps for the defined carriers using`n.plot()`.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # points to /dati/pampado/pypsa-eur
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

# =============================================================================
# Batch execution settings
# =============================================================================

RUNS = [
    "agriculture_full_electric",
    "agriculture_machinery_full_oil",
    "electricity_optimistic",
    "industry_h2",
    "land_transport_linear_ev",
    "shipping_full_methanol",
    "urban_heat_full_central",
    # "stochastic_network",
]

CARRIERS = [
    "AC",
    "H2",
    "co2_stored",
    "gas",
    "urban_central_heat",
]

SCENARIO_LABEL_MAP = {
    "base": "BASE",
    "agriculture_full_electric": "AFE",
    "agriculture_machinery_full_oil": "AMFO",
    "electricity_optimistic": "EO",
    "industry_h2": "IH2",
    "land_transport_linear_ev": "LTLEV",
    "shipping_full_methanol": "SFM",
    "shipping_methanol": "SM",
    "urban_heat_full_central": "UHFC",
    "stochastic_network": "SP",
}


CARRIER_LABEL_MAP = {
    "AC": "AC",
    "H2": "H2",
    "co2_stored": "CO2",
    "co2 stored": "CO2",
    "gas": "Gas",
    "urban_central_heat": "UCH",
    "urban central heat": "UCH",
}

FAIL_FAST = False

def clean_scenario_name(run):
    """
    Convert a scenario name from snake_case to title case.
    """
    return " ".join(word.capitalize() for word in run.split("_"))


def get_plot_title(run, carrier, *, compact=True):
    """
    Build a clean plot title.

    If compact=True, use a short title such as 'AC - AFE'.
    If compact=False, use a long title such as 'Agriculture Full Electric (AFE)'.
    """
    scenario_label = SCENARIO_LABEL_MAP.get(run, run.upper())
    carrier_label = CARRIER_LABEL_MAP.get(carrier, carrier.replace("_", " ").title())

    if compact:
        return f"{carrier_label} - {scenario_label}"

    scenario_name = clean_scenario_name(run)
    return f"{scenario_name} ({scenario_label})"

def plot_one(snakemake):
    """
    Plot one balance map for a single run/carrier combination.
    """

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    sanitize_carriers(n, snakemake.config)

    pypsa.set_option("params.statistics.round", 8)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    config = snakemake.params.plotting
    carrier = snakemake.wildcards.carrier
    settings = snakemake.params.settings

    if settings is None:
        available = list((config or {}).get("balance_map", {}).keys())
        raise KeyError(
            f"plot_balance_map: no settings found for carrier='{carrier}' in plotting.balance_map. "
            f"Available keys: {available}. "
            "This usually means your wildcard 'carrier' includes a suffix like '__sc-...' or '__exp'."
        )

    show_legend = settings.get("show_legend", True)

    carrier = carrier.replace(
        "_", " "
    )  # needed for slurm environment where [space] is not allowed

    # Fill empty colors or "" with light grey
    mask = n.carriers.color.isna() | n.carriers.color.eq("")
    n.carriers["color"] = n.carriers.color.mask(mask, "lightgrey")

    # Set EU location with location from config
    eu_location = config["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

    # Get balance map plotting parameters
    boundaries = config["map"]["boundaries"]
    unit_conversion = settings["unit_conversion"]
    branch_color = settings.get("branch_color") or "darkseagreen"

    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier {carrier} is not in the network. "
            "Remove it from configuration `plotting: balance_map: bus_carriers`."
        )

    # For plotting change bus to location
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # Set bus coordinates from location
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    # Bus size according to energy balance of bus carrier
    eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])

    # Remove energy balance of transmission carriers which relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")

    # Keep only carriers that are also in the energy balance
    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]

    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()

    bus_size = eb.groupby(level=["bus", "carrier"]).sum().div(unit_conversion)
    bus_size = bus_size.sort_values(ascending=False)

    # Get colors for carriers
    n.carriers.update({"color": snakemake.params.plotting["tech_colors"]})
    carrier_colors = n.carriers.color.copy().replace("", "grey")

    colors = (
        bus_size.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(carrier_colors)
    )

    # Line and link widths according to optimal transmission
    flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(
        unit_conversion
    )

    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    # If there are no lines or links for the bus carrier, use fallback for plotting
    fallback = pd.Series(dtype=float)
    line_width = flow.get("Line", fallback).abs()
    link_width = flow.get("Link", fallback).abs()

    # Define maximal size of buses and branch width
    bus_size_factor = settings["bus_factor"]
    branch_width_factor = settings["branch_factor"]
    flow_size_factor = settings["flow_factor"]

    # Get prices per region as colormap
    buses = n.buses.query("carrier in @carrier").index
    weights = n.snapshot_weightings.generators
    prices = weights @ n.buses_t.marginal_price[buses] / weights.sum()

    level = "name" if PYPSA_V1 else "Bus"
    price = prices.rename(n.buses.location).groupby(level=level).mean()

    if carrier == "co2 stored" and "CO2Limit" in n.global_constraints.index:
        co2_price = n.global_constraints.loc["CO2Limit", "mu"]
        price = price - co2_price

    # If only one price is available, use this price for all regions
    if price.size == 1:
        regions["price"] = price.values[0]
        shift = round(abs(price.values[0]) / 20, 0)
    else:
        regions["price"] = price.reindex(regions.index).fillna(0)
        shift = 0

    vmin, vmax = regions.price.min() - shift, regions.price.max() + shift

    if settings["vmin"] is not None:
        vmin = settings["vmin"]

    if settings["vmax"] is not None:
        vmax = settings["vmax"]

    crs = load_projection(snakemake.params.plotting)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    line_flow = flow.get("Line")
    link_flow = flow.get("Link")
    transformer_flow = flow.get("Transformer")

    n.plot(
        bus_size=bus_size * bus_size_factor,
        bus_color=colors,
        bus_split_circle=True,
        line_width=line_width * branch_width_factor,
        link_width=link_width * branch_width_factor,
        line_flow=line_flow * flow_size_factor if line_flow is not None else None,
        link_flow=link_flow * flow_size_factor if link_flow is not None else None,
        link_color=branch_color,
        transformer_flow=transformer_flow * flow_size_factor
        if transformer_flow is not None
        else None,
        ax=ax,
        margin=0.2,
        geomap_color={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        boundaries=boundaries,
    )

    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="price",
        cmap=settings["cmap"],
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
    )

    run = getattr(snakemake.wildcards, "run", "")
    raw_carrier = snakemake.wildcards.carrier

    ax.set_title(
        get_plot_title(run, raw_carrier, compact=True),
        fontweight="bold",
    )

    # Add colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=settings["cmap"], norm=norm)
    price_unit = settings["region_unit"]

    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f"Average Marginal Price [{price_unit}]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    # Add legend
    if show_legend:
        legend_kwargs = {
            "loc": "upper left",
            "frameon": False,
            "alignment": "left",
            "title_fontproperties": {"weight": "bold"},
        }

        pad = 0.18
        n.carriers.loc["", "color"] = "None"

        # Get lists for supply and consumption carriers
        pos_carriers = bus_size[bus_size > 0].index.unique("carrier")
        neg_carriers = bus_size[bus_size < 0].index.unique("carrier")

        # Determine larger total absolute value for supply and consumption
        common_carriers = pos_carriers.intersection(neg_carriers)

        def get_total_abs(carrier_name, sign):
            values = bus_size.loc[:, carrier_name]
            return values[values * sign > 0].abs().sum()

        supp_carriers = sorted(
            set(pos_carriers) - set(common_carriers)
            | {
                c
                for c in common_carriers
                if get_total_abs(c, 1) >= get_total_abs(c, -1)
            }
        )

        cons_carriers = sorted(
            set(neg_carriers) - set(common_carriers)
            | {
                c
                for c in common_carriers
                if get_total_abs(c, 1) < get_total_abs(c, -1)
            }
        )

        # Add supply carriers
        add_legend_patches(
            ax,
            n.carriers.color[supp_carriers],
            supp_carriers,
            legend_kw={
                "bbox_to_anchor": (0, -pad),
                "ncol": 1,
                "title": "Supply",
                **legend_kwargs,
            },
        )

        # Add consumption carriers
        add_legend_patches(
            ax,
            n.carriers.color[cons_carriers],
            cons_carriers,
            legend_kw={
                "bbox_to_anchor": (0.5, -pad),
                "ncol": 1,
                "title": "Consumption",
                **legend_kwargs,
            },
        )

        # Add bus legend
        legend_bus_size = settings["bus_sizes"]
        carrier_unit = settings["unit"]

        if legend_bus_size is not None:
            add_legend_semicircles(
                ax,
                [
                    s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                    for s in legend_bus_size
                ],
                [f"{s} {carrier_unit}" for s in legend_bus_size],
                patch_kw={"color": "#666"},
                legend_kw={
                    "bbox_to_anchor": (0, 1),
                    **legend_kwargs,
                },
            )

        # Add branch legend
        legend_branch_sizes = settings["branch_sizes"]

        if legend_branch_sizes is not None:
            add_legend_lines(
                ax,
                [s * branch_width_factor for s in legend_branch_sizes],
                [f"{s} {carrier_unit}" for s in legend_branch_sizes],
                patch_kw={"color": "#666"},
                legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
            )

    output_path = Path(snakemake.output[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
    )

    png_output_path = output_path.with_suffix(".png")

    fig.savefig(
        png_output_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {png_output_path}")


if __name__ == "__main__":
    if "snakemake" in globals():
        plot_one(snakemake)

    else:
        from scripts._helpers import mock_snakemake

        for run in RUNS:
            for carrier in CARRIERS:
                print("=" * 100)
                print(f"Plotting run='{run}', carrier='{carrier}'")
                print("=" * 100)

                snakemake = mock_snakemake(
                    "plot_balance_map",
                    clusters="adm",
                    opts="",
                    sector_opts="",
                    planning_horizons="2050",
                    carrier=carrier,
                    configfiles="config/test_stochastic_scenarios/config.yaml",
                    run=run,
                    # stoch_scenario="normal",
                )

                try:
                    plot_one(snakemake)

                except Exception as exc:
                    print(
                        f"Failed for run='{run}', carrier='{carrier}': {type(exc).__name__}: {exc}"
                    )

                    if FAIL_FAST:
                        raise