"""Audit and plot demand-transition results. Run with pandas, openpyxl, numpy,
matplotlib and (only for NetCDF objective inputs) xarray installed.

Workbook values are selected by labels, never by column position. Missing rows
remain NaN; aliases are alternatives, not instructions to sum technologies.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

# =========================== USER SETTINGS ===========================
ROOT = Path(__file__).resolve().parents[2]
DETERMINISTIC_ENERGY_FILE = (
    ROOT
    / "results/demand_uncertainty_2035/analysis_output/csvs/analysis_networks_energy.xlsx"
)
DETERMINISTIC_CAPACITY_FILE = (
    ROOT
    / "results/demand_uncertainty_2035/analysis_output/csvs/analysis_networks_power.xlsx"
)
SENSITIVITY_ROOT = ROOT / "results/demand_uncertainty_sensitivity"
SENSITIVITY_FILES = {
    "E_HEAT": SENSITIVITY_ROOT
    / "analysis_output/ELEC_HEAT/csvs/analysis_networks_energy.xlsx",
    "E_IND": SENSITIVITY_ROOT
    / "analysis_output/ELEC_IND/csvs/analysis_networks_energy.xlsx",
    "E_ROAD": SENSITIVITY_ROOT
    / "analysis_output/ELEC_ROAD/csvs/analysis_networks_energy.xlsx",
    "F_HEAT": SENSITIVITY_ROOT
    / "analysis_output/OILGAS_HEAT/csvs/analysis_networks_energy.xlsx",
    "F_IND": SENSITIVITY_ROOT
    / "analysis_output/OILGAS_IND/csvs/analysis_networks_energy.xlsx",
    "F_ROAD": SENSITIVITY_ROOT
    / "analysis_output/OILGAS_TRANS/csvs/analysis_networks_energy.xlsx",
    "M_AIR": SENSITIVITY_ROOT
    / "analysis_output/MEOH_AIR/csvs/analysis_networks_energy.xlsx",
    "M_IND": SENSITIVITY_ROOT
    / "analysis_output/MEOH_IND/csvs/analysis_networks_energy.xlsx",
    "M_SHIP": SENSITIVITY_ROOT
    / "analysis_output/MEOH_SHIP/csvs/analysis_networks_energy.xlsx",
    "H_IND": SENSITIVITY_ROOT
    / "analysis_output/H2_IND/csvs/analysis_networks_energy.xlsx",
    "H_OIL": SENSITIVITY_ROOT
    / "analysis_output/H2_OIL/csvs/analysis_networks_energy.xlsx",
    "H_SHIP": SENSITIVITY_ROOT
    / "analysis_output/H2_TRANS/csvs/analysis_networks_energy.xlsx",
}
OUTPUT_DIR = SENSITIVITY_ROOT / "analysis_output/cross_family"
# Optional CSV/XLSX with scenario, perturbation_twh, objective, unit columns.
# Include scenario=BASE, perturbation_twh=0; units must be EUR/a.
COST_FILE: Path | None = None
COST_SHEET = "objectives"
# Used when COST_FILE is None. Read only the saved solver objective attribute.
OBJECTIVE_NETWORK_ROOT: Path | None = SENSITIVITY_ROOT
OBJECTIVE_NETWORK_NAME = "base_s_adm___2035.nc"
OBJECTIVE_ATTRIBUTE = "network__objective"
SKIP_MISSING_COST_FIGURE = True
MAKE_ELECTRICITY_RESPONSE_FIGURE = True
# False gives each cost panel its own scale; the choice is logged.
# H-family variations (<2%) are obscured by the E_ROAD excursion (~12%).
COST_SHARE_Y = False
# From config/demand_uncertainty_2035/scenarios/scenarios_determinstic_definition.yaml.
# These are exogenous targets, not realised final-demand changes.
NOMINAL_TARGET_TWH = {
    "E_HEAT": 400.0,
    "E_IND": 400.0,
    "E_ROAD": 400.0,
    "F_HEAT": 400.0,
    "F_IND": 400.0,
    "F_ROAD": 400.0,
    "M_AIR": 100.0,
    "M_IND": 100.0,
    "M_SHIP": 100.0,
    "H_IND": 100.0,
    "H_OIL": 100.0,
    "H_SHIP": 100.0,
}
# Confirmed from workbook producer scripts: MWh energy, MW power, MWh stores.
ENERGY_INPUT_UNIT = "MWh/a"
POWER_INPUT_UNIT = "MW"
STORAGE_INPUT_UNIT = "MWh"
BASE_CAPACITY_THRESHOLDS = {"GW": 1.0, "TWh": 0.01}
MATERIAL_TWH = 1.0
PLATEAU_TOL_TWH = 0.5
PLATEAU_MIN_POINTS = 2
SLOPE_CHANGE_FACTOR = 3.0
INCLUDE_INDUSTRY_BIOMASS = True

# Central paper-style configuration; widths are in inches.
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.2
SHOW_TITLES = False
PNG_DPI = 300
PLOT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.formatter.useoffset": False,
    "savefig.facecolor": "white",
}
TECH_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
LINE_STYLES = ("-", "--", "-.", ":")
FAMILY_COLORS = {"E": "#0072B2", "F": "#D55E00", "M": "#009E73", "H": "#CC79A7"}

# Explicit input-name aliases preserve the current public scenario names.
SOURCE_NAMES = dict(
    zip(
        NOMINAL_TARGET_TWH,
        (
            "ELEC_HEAT",
            "ELEC_IND",
            "ELEC_ROAD",
            "OILGAS_HEAT",
            "OILGAS_IND",
            "OILGAS_TRANS",
            "MEOH_AIR",
            "MEOH_IND",
            "MEOH_SHIP",
            "H2_IND",
            "H2_OIL",
            "H2_TRANS",
        ),
    )
)
SCENARIOS = tuple(NOMINAL_TARGET_TWH)
COST_SCENARIOS = SCENARIOS
FAMILY_TITLES = ("Electrification", "Fossil persistence", "Methanol", "Hydrogen")
# Primary fossil-oil supply upstream of refining.
PRIMARY_OIL_SUPPLY = ("oil primary", ("oil primary",))
FOSSIL_GAS_SUPPLY = ("gas", ("gas",))
BIOGAS_TO_GAS_SUPPLY = ("gas", ("biogas to gas",))
# Captured annual CO2 entering the stored-CO2 carrier, not net sequestration.
CO2_INPUT_UNIT = "tCO2/a"
CARBON_CAPTURE_TECHNOLOGIES = (
    "SMR CC", "process emissions CC", "solid biomass for industry CC",
    "DAC", "urban central gas CHP CC", "gas for industry CC",
    "urban central solid biomass CHP CC", "Methanol steam reforming CC",
)
CARBON_EXCLUDED_SUPPLY = {"CO2 pipeline": "Internal transport; counting it would duplicate captured carbon"}
CARBON_CONSUMPTION_TECHNOLOGIES = (
    "methanolisation", "co2 sequestered", "Fischer-Tropsch", "Sabatier",
)
# Classifications verified against prepare_sector_network.py:
# refining: 621-644; HVC stock release: 5050-5079; reserved H2 oil: 5882-5938.
CO2_OIL_TECHNOLOGIES = (
    "land transport oil", "kerosene for aviation", "shipping oil",
    "agriculture machinery oil", "oil refining", "oil from H2 emissions",
)
CO2_GAS_TECHNOLOGIES = (
    "OCGT", "CCGT", "rural gas boiler", "urban decentral gas boiler",
    "urban central gas boiler", "gas for industry", "gas for industry CC",
    "urban central gas CHP", "urban central gas CHP CC", "SMR", "SMR CC",
)
CO2_EMISSIONS_EXCLUDED = {
    "process emissions": "Non-combustion industrial process emissions",
    "process emissions CC": "Uncaptured process emissions, not fuel-gas combustion",
    "HVC to air": "Release from non-sequestered HVC stock; not current fuel combustion",
    "coal for industry": "Coal, not oil or gas",
    "industry methanol": "Methanol-derived carbon, kept separate",
    "shipping methanol": "Methanol-derived carbon, kept separate",
    "methanol-to-kerosene": "Methanol route, kept separate from ordinary oil kerosene",
    "Methanol steam reforming": "Methanol feedstock, not gas-fed SMR",
    "Methanol steam reforming CC": "Methanol feedstock, not gas-fed SMR",
    "OCGT methanol": "Methanol-fired generation, not gas OCGT",
}
METHANOL_CAPTURE_SOURCES = {
    "process_emissions_CC": "process emissions CC",
    "SMR_CC": "SMR CC",
    "solid_biomass_industry_CC": "solid biomass for industry CC",
}
# Display concept, component, carrier aliases, metric. Link power is input-side p_nom_opt.
CAPACITY_ROWS = [
    ("Distribution grid", "Link", ("electricity distribution grid",), "power_final"),
    ("Onshore wind", "Generator", ("onwind", "onshore wind"), "power_final"),
    ("Solar rooftop", "Generator", ("solar rooftop",), "power_final"),
    ("Solar-hsat", "Generator", ("solar-hsat",), "power_final"),
    ("Battery discharger", "Link", ("battery discharger",), "power_final"),
    ("Electrolysis", "Link", ("H2 Electrolysis",), "power_final"),
    ("SMR", "Link", ("SMR",), "power_final"),
    ("SMR CC", "Link", ("SMR CC",), "power_final"),
    ("H2 pipeline", "Link", ("H2 pipeline",), "power_final"),
    ("H2 storage", "Store", ("H2 Store",), "energy_final"),
    ("Methanolisation", "Link", ("methanolisation",), "power_final"),
    ("Biomass-to-methanol", "Link", ("biomass-to-methanol",), "power_final"),
    ("Biogas-to-gas", "Link", ("biogas to gas",), "power_final"),
    ("Gas CHP", "Link", ("urban central gas CHP",), "power_final"),
    ("Central water pit", "Store", ("urban central water pits",), "energy_final"),
]
HYDROGEN = {
    "Electrolysis": ("H2 Electrolysis",),
    "SMR": ("SMR",),
    "SMR CC": ("SMR CC",),
}
BIOMASS_BLOCKS = {
    "Heat": (
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
        "urban decentral biomass boiler",
        "rural biomass boiler",
    ),
    "Methanol": ("biomass-to-methanol",),
    "Synthetic oil": ("electrobiofuels", "biomass to liquid"),
    "Industry": ("solid biomass for industry", "solid biomass for industry CC"),
}
# ========================= END USER SETTINGS =========================

DIAGNOSTICS: list[str] = []
ENERGY_SCALE = {"MWh/a": 1e-6, "GWh/a": 1e-3, "TWh/a": 1.0}


def report(message: str, warning: bool = False) -> None:
    DIAGNOSTICS.append(("WARNING: " if warning else "") + message)
    if warning:
        warnings.warn(message, stacklevel=2)
    else:
        print(message)


def normalize(label: str) -> str:
    """Normalize only case and whitespace; retain technology distinctions."""
    return " ".join(str(label).strip().casefold().split())


def scenario_name(raw: str) -> str:
    if raw in ("__BASE__", "BASE"):
        return "BASE"
    for public, source in SOURCE_NAMES.items():
        if raw == source:
            return public
        if raw.startswith(source + "_"):
            return public + raw[len(source) :]
    return raw


def inspect_workbook(path: Path) -> list[str]:
    with pd.ExcelFile(path) as book:
        sheets = book.sheet_names
    report(f"INPUT: {path}\n  sheets: {sheets}")
    return sheets


def detect_sensitivity_levels(columns: pd.Index, scenario: str) -> dict[str, float]:
    levels = {}
    for name in columns:
        match = re.fullmatch(re.escape(scenario) + r"_(\d+(?:\.\d+)?)", str(name))
        if match:
            levels[str(name)] = float(match.group(1))
    if not levels or len(set(levels.values())) != len(levels):
        raise ValueError(
            f"Missing or ambiguous sensitivity levels for {scenario}: {list(columns)}"
        )
    return dict(sorted(levels.items(), key=lambda item: item[1]))


def load_excel_data(path: Path, capacity: bool = False) -> dict[str, pd.DataFrame]:
    sheets = inspect_workbook(path)
    wanted = (
        ["levels_by_component_carrier"]
        if capacity
        else ["levels_supply", "levels_consumption"]
    )
    indices = (
        ["component", "carrier", "metric"] if capacity else ["group", "technology"]
    )
    result = {}
    for sheet in wanted:
        if sheet not in sheets:
            raise ValueError(
                f"{path}: required sheet {sheet!r} missing; sheets={sheets}"
            )
        frame = pd.read_excel(path, sheet_name=sheet)
        if not set(indices).issubset(frame):
            raise ValueError(f"{path}/{sheet}: expected row labels {indices}")
        values = [c for c in frame if str(c).startswith("value__")]
        if not values:
            raise ValueError(f"{path}/{sheet}: no value__<scenario> columns")
        table = frame.set_index(indices)[values].rename(
            columns=lambda c: scenario_name(c[len("value__") :])
        )
        if table.index.has_duplicates or table.columns.has_duplicates:
            raise ValueError(
                f"Ambiguous duplicate row/scenario mapping in {path}/{sheet}"
            )
        table = table.apply(pd.to_numeric, errors="raise")
        if "BASE" not in table:
            raise ValueError(f"No BASE column in {path}/{sheet}")
        report(f"  {sheet}: scenarios={list(table.columns)}, rows={len(table)}")
        if table.isna().any().any():
            report(
                f"{path}/{sheet}: {int(table.isna().sum().sum())} missing cells retained as NaN",
                True,
            )
        if not capacity:
            if (table < -1e-8).any().any():
                raise ValueError(f"Expected positive magnitudes in {path}/{sheet}")
            table *= ENERGY_SCALE[ENERGY_INPUT_UNIT]
        result[sheet] = table
    report(
        f"  units: {POWER_INPUT_UNIT}/{STORAGE_INPUT_UNIT} capacity"
        if capacity
        else f"  units: {ENERGY_INPUT_UNIT} -> TWh/a (CO2 rows excluded from energy analyses)"
    )
    return result


def resolve_technology(
    labels: pd.Index, aliases: tuple[str, ...], context: str
) -> str | None:
    matches = [
        label
        for label in labels.unique()
        if normalize(label) in {normalize(a) for a in aliases}
    ]
    if len(matches) > 1:
        report(
            f"Ambiguous mapping {context}: aliases={aliases}, matches={matches}", True
        )
        raise ValueError(f"Explicit selection required for {context}")
    if not matches:
        report(f"Technology not found: {context}; requested aliases={aliases}", True)
        return None
    report(f"ALIAS {context}: {aliases} -> {matches[0]}")
    return str(matches[0])


def get_energy_series(
    data: dict[str, pd.DataFrame], side: str, group: str, aliases: tuple[str, ...]
) -> pd.Series:
    table = data[f"levels_{side}"]
    if group not in table.index.get_level_values("group"):
        report(f"Missing {side} carrier {group}", True)
        return pd.Series(np.nan, index=table.columns)
    part = table.xs(group, level="group")
    label = resolve_technology(part.index, aliases, f"{side}/{group}")
    return (
        part.loc[label] if label is not None else pd.Series(np.nan, index=table.columns)
    )


def get_supply_series(
    data: dict[str, pd.DataFrame], group: str, aliases: tuple[str, ...]
) -> pd.Series:
    return get_energy_series(data, "supply", group, aliases)


def get_consumption_series(
    data: dict[str, pd.DataFrame], group: str, aliases: tuple[str, ...]
) -> pd.Series:
    return get_energy_series(data, "consumption", group, aliases)


def get_capacity_series(
    table: pd.DataFrame, component: str, aliases: tuple[str, ...], metric: str
) -> pd.Series:
    mask = (table.index.get_level_values("component") == component) & (
        table.index.get_level_values("metric") == metric
    )
    part = table.loc[mask].droplevel(["component", "metric"])
    label = resolve_technology(part.index, aliases, f"capacity/{component}/{metric}")
    return (
        part.loc[label] if label is not None else pd.Series(np.nan, index=table.columns)
    )


def get_delta_vs_base(series: pd.Series) -> pd.Series:
    return series - series["BASE"]


def aggregate_technologies(
    data: dict[str, pd.DataFrame], group: str, technologies: tuple[str, ...]
) -> pd.Series:
    if len(set(technologies)) != len(technologies):
        raise ValueError(f"Duplicate technologies in aggregation: {technologies}")
    parts = [get_consumption_series(data, group, (tech,)) for tech in technologies]
    # A missing constituent invalidates the block rather than silently undercounting it.
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))


def save_figure(fig: plt.Figure, stem: str, title: str) -> None:
    if SHOW_TITLES:
        fig.suptitle(title)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            dpi=PNG_DPI,
            bbox_inches="tight",
            transparent=False,
        )
    plt.close(fig)


def separators(ax: plt.Axes) -> None:
    for x in (2.5, 5.5, 8.5):
        ax.axvline(x, color="0.4", linewidth=0.7)


def make_absolute_capacity_heatmap(result: pd.DataFrame) -> None:
    """Signed physical changes, with separate symmetric-log scales by unit."""
    units = ("GW", "TWh")
    linear_thresholds = {"GW": .1, "TWh": .01}
    row_labels = {
        unit: [label for label, _, _, _ in CAPACITY_ROWS
               if label in set(result.loc[result.unit == unit, "technology"])]
        for unit in units
    }
    fig, axes = plt.subplots(
        2, 1, figsize=(DOUBLE_COLUMN_WIDTH, 6.1), layout="constrained",
        gridspec_kw={"height_ratios": [len(row_labels[u]) for u in units]},
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    for ax, unit in zip(axes, units):
        matrix = result[result.unit == unit].pivot(
            index="technology", columns="scenario", values="absolute_delta"
        ).reindex(index=row_labels[unit], columns=SCENARIOS)
        finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
        threshold = linear_thresholds[unit]
        limit = max(np.max(np.abs(finite)) if len(finite) else 0., threshold)
        norm = SymLogNorm(linthresh=threshold, vmin=-limit, vmax=limit, base=10)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title("Power capacity [GW]" if unit == "GW" else "Storage energy capacity [TWh]")
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=55, ha="right")
        ax.tick_params(axis="x", labelbottom=unit == "TWh")
        separators(ax)
        for row in range(len(matrix)):
            for col in range(len(SCENARIOS)):
                value = matrix.iloc[row, col]
                if not np.isfinite(value):
                    text, color = "NA", "0.3"
                else:
                    text = "0" if abs(value) < 1e-6 else format(float(format(value, ".2g")), "g")
                    color = "white" if abs(float(norm(value)) - .5) > .32 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=5.5, color=color)
        powers = np.arange(int(np.log10(threshold)), int(np.floor(np.log10(limit))) + 1)
        ticks = 10. ** powers
        ticks = np.concatenate((-ticks[::-1], [0.], ticks))
        if unit == "TWh":
            ticks = [-10. ** np.floor(np.log10(limit)), 0., 10. ** np.floor(np.log10(limit))]
        bar = fig.colorbar(im, ax=ax, ticks=ticks, fraction=.035, pad=.025)
        bar.ax.set_yticklabels([f"{v:g}" for v in ticks])
        bar.set_label(f"Δ capacity [{unit}]", fontsize=7)
        bar.ax.tick_params(labelsize=6)
        report(f"Absolute capacity heatmap {unit}: symmetric-log colors, linear within ±{threshold:g} {unit}; range ±{limit:.6g} {unit}.")
    fig.suptitle("Capacity change relative to BASE; symmetric-log colors", fontsize=9)
    save_figure(fig, "01_capacity_response_heatmap_absolute_delta",
                "Absolute capacity changes relative to BASE")
    (OUTPUT_DIR / "01_capacity_response_heatmap_absolute_delta_caption.tex").write_text(
        r"Signed changes in optimal capacity relative to BASE across nominal scenarios. Power capacities are in GW and storage energy capacities in TWh, with separate symmetric logarithmic color scales. Red denotes increases and blue decreases; white denotes zero. Colors are linear within $\pm0.1$ GW and $\pm0.01$ TWh and logarithmic outside these intervals. Cell annotations show untransformed changes to two significant digits; magnitudes below $10^{-6}$ are displayed as zero. Link power capacities refer to input-side nominal capacity. Grey cells marked NA indicate missing values." + "\n", encoding="utf-8")


# Main useful output for the technologies used in the capacity heatmap.
# Multi-output links use bus1's energy carrier; storage uses discharge where available.
ENERGY_MAIN_OUTPUTS = {
    "Distribution grid": ("low voltage", ("electricity distribution grid",)),
    "Onshore wind": ("AC", ("onwind", "onshore wind")),
    "Solar rooftop": ("low voltage", ("solar rooftop",)),
    "Solar-hsat": ("AC", ("solar-hsat",)),
    "Battery discharger": ("AC", ("battery discharger",)),
    "Electrolysis": ("H2", ("H2 Electrolysis",)),
    "SMR": ("H2", ("SMR",)),
    "SMR CC": ("H2", ("SMR CC",)),
    "H2 pipeline": ("H2", ("H2 pipeline",)),
    "H2 storage": ("H2", ("H2 Store",)),
    "Methanolisation": ("methanol", ("methanolisation",)),
    "Biomass-to-methanol": ("methanol", ("biomass-to-methanol",)),
    "Biogas-to-gas": ("gas", ("biogas to gas",)),
    "Gas CHP": ("AC", ("urban central gas CHP",)),
    "Central water pit": ("urban central heat", ("urban central water pits discharger",)),
}


def make_energy_balance_heatmap(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One selected-technology heatmap, using main energy output only."""
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.lines import Line2D
    from matplotlib.textpath import TextPath

    labels = [row[0] for row in CAPACITY_ROWS]
    records = []
    for label in labels:
        carrier, aliases = ENERGY_MAIN_OUTPUTS[label]
        values = get_supply_series(data, carrier, aliases)
        for scenario in SCENARIOS:
            records.append(dict(technology=label, output_carrier=carrier,
                                source_technology=aliases[0], side="supply",
                                scenario=scenario, unit="TWh/a", base=values["BASE"],
                                level=values[scenario],
                                absolute_delta=values[scenario] - values["BASE"],
                                available=bool(np.isfinite(values[scenario]) and np.isfinite(values["BASE"]))))
    result = pd.DataFrame(records)
    stem = "09_energy_balance_heatmap_absolute_delta"
    result.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    matrix = result.pivot(index="technology", columns="scenario",
                          values="absolute_delta").reindex(index=labels, columns=SCENARIOS)
    matrix.to_csv(OUTPUT_DIR / f"{stem}_matrix.csv")
    finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    threshold = .1
    limit = max(np.max(np.abs(finite)) if len(finite) else 0., threshold)
    norm = SymLogNorm(linthresh=threshold, vmin=-limit, vmax=limit, base=10)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 5.5), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), [""] * len(labels))
    # Offset-box labels underline only the output carrier, without requiring LaTeX.
    for row, label in enumerate(labels):
        carrier = ENERGY_MAIN_OUTPUTS[label][0]
        width = TextPath((0, 0), carrier, size=6).get_extents().width + 1
        underline = DrawingArea(width, 1, 0, 0)
        underline.add_artist(Line2D([0, width], [.5, .5], color="black", linewidth=.45))
        carrier_box = VPacker(children=[TextArea(carrier, textprops={"size": 6}), underline],
                              align="center", pad=0, sep=0)
        label_box = HPacker(children=[TextArea(label + " → ", textprops={"size": 6}), carrier_box],
                            align="center", pad=0, sep=0)
        ax.add_artist(AnnotationBbox(label_box, (-.015, row),
                                    xycoords=ax.get_yaxis_transform(), box_alignment=(1, .5),
                                    frameon=False, annotation_clip=False, pad=0))
    separators(ax)
    for row in range(len(matrix)):
        for col in range(len(SCENARIOS)):
            value = matrix.iloc[row, col]
            text = ("NA" if not np.isfinite(value) else "0" if abs(value) < 1e-6
                    else format(float(format(value, ".2g")), "g"))
            color = ("white" if np.isfinite(value) and abs(float(norm(value)) - .5) > .32 else "black")
            ax.text(col, row, text, ha="center", va="center", fontsize=5.5, color=color)
    ticks = 10. ** np.arange(-1, int(np.floor(np.log10(limit))) + 1)
    ticks = np.concatenate((-ticks[::-1], [0.], ticks))
    bar = fig.colorbar(im, ax=ax, ticks=ticks, fraction=.035, pad=.025)
    bar.set_ticklabels([f"{v:g}" for v in ticks])
    bar.ax.tick_params(labelsize=6)
    bar.set_label("Δ main energy output [TWh/a]", fontsize=7)
    ax.set_title("Main energy output relative to BASE; symmetric-log colors", fontsize=9)
    fig.supxlabel("Underlined: output carrier. Storage: discharge; NA: not exported.", fontsize=6)
    save_figure(fig, stem, "Main-technology energy-output changes")
    pd.DataFrame([dict(page=1, scope="Selected capacity-heatmap technologies", rows=len(matrix),
                       pdf=f"{stem}.pdf", png=f"{stem}.png")]).to_csv(
        OUTPUT_DIR / f"{stem}_index.csv", index=False)
    caption = (
        "Signed changes in annual main energy output relative to BASE for the technologies selected "
        "in the capacity heatmap, across the 12 nominal scenarios. Underlined labels identify output "
        "carriers. Multi-output links use their main energy output only, excluding coproduct heat "
        "and carbon flows. Storage is represented by discharge: the central water pit uses its heat "
        "discharger; H2 storage is marked NA because no H2 Store supply row is exported in the workbook. "
        "Values are in TWh/a; red denotes increases and blue decreases. The symmetric-log color scale "
        "is linear within ±0.1 TWh/a. Annotations show two significant digits, with magnitudes below "
        "1e-6 displayed as zero. Network-transfer outputs are retained, so rows are not additive."
    )
    (OUTPUT_DIR / f"{stem}_caption.tex").write_text("% " + caption + "\n", encoding="utf-8")
    report(f"Selected energy heatmap: {len(labels)} technologies; main outputs only; "
           f"symmetric-log range ±{limit:.6g} TWh/a.")
    return result


def make_capacity_heatmap(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table = data["levels_by_component_carrier"]
    records = []
    for label, component, aliases, metric in CAPACITY_ROWS:
        series = get_capacity_series(table, component, aliases, metric)
        unit = "TWh" if metric == "energy_final" else "GW"
        scale = (
            {"MWh": 1e-6, "GWh": 1e-3, "TWh": 1.0}[STORAGE_INPUT_UNIT]
            if unit == "TWh"
            else {"MW": 1e-3, "GW": 1.0}[POWER_INPUT_UNIT]
        )
        series = series * scale
        base = series["BASE"]
        eligible = np.isfinite(base) and abs(base) > BASE_CAPACITY_THRESHOLDS[unit]
        if not eligible:
            report(
                f"Relative capacity excluded: {label}, BASE={base:g} {unit}, threshold={BASE_CAPACITY_THRESHOLDS[unit]}",
                True,
            )
        for scenario in SCENARIOS:
            value = series.get(scenario, np.nan)
            records.append(
                dict(
                    technology=label,
                    component=component,
                    source_label=aliases[0],
                    metric=metric,
                    unit=unit,
                    scenario=scenario,
                    base=base,
                    capacity=value,
                    absolute_delta=value - base,
                    relative_delta=100 * (value - base) / base if eligible else np.nan,
                )
            )
    result = pd.DataFrame(records)
    metadata = ["technology", "component", "source_label", "metric", "unit", "base"]
    for value, suffix in (
        ("absolute_delta", "absolute"),
        ("relative_delta", "relative"),
    ):
        wide = result.pivot(index=metadata, columns="scenario", values=value).reindex(
            columns=SCENARIOS
        )
        wide.reindex([r[0] for r in CAPACITY_ROWS], level="technology").to_csv(
            OUTPUT_DIR / f"01_capacity_response_heatmap_{suffix}_delta.csv"
        )
    result.to_csv(OUTPUT_DIR / "01_capacity_response_levels.csv", index=False)
    make_absolute_capacity_heatmap(result)
    matrix = result.pivot(
        index="technology", columns="scenario", values="absolute_delta"
    ).reindex(index=[r[0] for r in CAPACITY_ROWS], columns=SCENARIOS)
    denominators = matrix.abs().max(axis=1)
    normalized = matrix.div(denominators.replace(0, np.nan), axis=0)
    for label in matrix.index:
        if matrix.loc[label].isna().any():
            report(f"Row normalization incomplete: {label}; missing cells retained", True)
        elif denominators[label] == 0:
            normalized.loc[label] = 0.0
            report(f"Row normalization: {label}; all deltas zero, neutral row")
        else:
            report(f"Row normalization successful: {label}; max absolute delta={denominators[label]:.6g}; normalized max=1")
    normalized.to_csv(OUTPUT_DIR / "01_capacity_response_heatmap_row_normalized.csv")
    matrix = normalized
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.4), layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -1, 1)
    )
    ax.set_xticks(range(12), SCENARIOS, rotation=55, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    separators(ax)
    fig.colorbar(im, ax=ax, label="Row-normalized capacity change [−1, +1]", shrink=0.8)
    save_figure(
        fig,
        "01_capacity_response_heatmap_row_normalized",
        "Optimal-capacity response relative to BASE",
    )
    return result


def sensitivity_records(
    series: pd.Series, scenario: str, label: str, levels: dict[str, float]
) -> list[dict]:
    return [
        dict(
            scenario=scenario,
            perturbation_twh=x,
            technology=label,
            level_twh=series[name],
            base_twh=series["BASE"],
            delta_twh=series[name] - series["BASE"],
        )
        for name, x in {"BASE": 0.0, **levels}.items()
    ]


def line_panels(
    table: pd.DataFrame,
    scenarios: tuple[str, ...],
    labels: list[str],
    stem: str,
    ylabel: str,
    rows: int,
    highlight_nominal: bool = False,
) -> None:
    fig, axes = plt.subplots(
        rows,
        len(scenarios) // rows,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.8 if rows == 1 else 2.25 * rows),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, scenario in zip(axes.flat, scenarios):
        for j, label in enumerate(labels):
            part = table[
                (table.scenario == scenario) & (table.technology == label)
            ].sort_values("perturbation_twh")
            ax.plot(
                part.perturbation_twh,
                part.delta_twh,
                color=TECH_COLORS[j],
                linestyle=LINE_STYLES[j],
                marker="o",
                markersize=3,
                label=label,
            )
        if highlight_nominal:
            nominal = table[(table.scenario == scenario) & np.isclose(
                table.perturbation_twh, NOMINAL_TARGET_TWH[scenario])]
            if len(nominal) != len(labels):
                raise ValueError(f"Missing nominal values for {scenario}")
            ax.scatter(nominal.perturbation_twh, nominal.delta_twh, s=42,
                       facecolors="none", edgecolors="black", linewidths=.8, zorder=5)
        ax.set_title(scenario)
        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.grid(axis="y", alpha=0.18)
        ax.ticklabel_format(style="plain", axis="both")
        if not highlight_nominal or ax in axes[-1, :]:
            ax.set_xlabel("Perturbation [TWh]")
    if highlight_nominal:
        for ax, family in zip(axes[:, 0], "EFMH"):
            ax.annotate(family, xy=(0, .5), xycoords="axes fraction",
                        xytext=(-48, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=11, fontweight="bold",
                        color=FAMILY_COLORS[family])
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="outside lower center", ncol=len(labels), frameon=False,
        title="Outlined marker = nominal target" if highlight_nominal else None,
        title_fontsize=7,
    )
    save_figure(fig, stem, ylabel)


def make_hydrogen_pathway_figure(sensitivity: dict, levels: dict) -> pd.DataFrame:
    selected = SCENARIOS
    records = []
    for scenario in selected:
        for label, aliases in HYDROGEN.items():
            records += sensitivity_records(
                get_supply_series(sensitivity[scenario], "H2", aliases),
                scenario,
                label,
                levels[scenario],
            )
    table = pd.DataFrame(records)
    table["nominal_point"] = [np.isclose(r.perturbation_twh, NOMINAL_TARGET_TWH[r.scenario]) for r in table.itertuples()]
    table.to_csv(OUTPUT_DIR / "02_hydrogen_pathway_sensitivity.csv", index=False)
    line_panels(
        table,
        selected,
        list(HYDROGEN),
        "02_hydrogen_pathway_sensitivity",
        "Δ H2 production [TWh/a]",
        4,
        highlight_nominal=True,
    )
    return table


def make_biomass_reallocation_figure(sensitivity: dict, levels: dict) -> pd.DataFrame:
    selected = SCENARIOS
    blocks = {
        k: v
        for k, v in BIOMASS_BLOCKS.items()
        if INCLUDE_INDUSTRY_BIOMASS or k != "Industry"
    }
    flat = [tech for group in blocks.values() for tech in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Biomass blocks overlap")
    report(f"Biomass input mappings on solid biomass: {blocks}")
    records, components = [], []
    for scenario in selected:
        available = set(
            sensitivity[scenario]["levels_consumption"].xs("solid biomass").index
        )
        if available - set(flat):
            report(
                f"Unallocated solid biomass pathways in {scenario}: {sorted(available - set(flat))}",
                True,
            )
        for block, technologies in blocks.items():
            series = aggregate_technologies(
                sensitivity[scenario], "solid biomass", technologies
            )
            records += sensitivity_records(series, scenario, block, levels[scenario])
            for technology in technologies:
                rows = sensitivity_records(
                    get_consumption_series(
                        sensitivity[scenario], "solid biomass", (technology,)
                    ),
                    scenario,
                    technology,
                    levels[scenario],
                )
                components.extend(
                    dict(row, block=block, carrier="solid biomass") for row in rows
                )
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT_DIR / "03_biomass_reallocation_sensitivity.csv", index=False)
    pd.DataFrame(components).to_csv(
        OUTPUT_DIR / "03_biomass_allocation_components.csv", index=False
    )
    line_panels(
        table,
        selected,
        list(blocks),
        "03_biomass_reallocation_sensitivity",
        "Δ biomass input vs BASE [TWh/a]",
        4,
        highlight_nominal=True,
    )
    return table


def load_objectives(levels: dict) -> pd.DataFrame:
    """Load actual saved objectives, never reconstruct them from energy or costs."""
    if COST_FILE is not None:
        report(f"COST INPUT: {COST_FILE}")
        if COST_FILE.suffix.lower() == ".csv":
            table = pd.read_csv(COST_FILE)
        else:
            inspect_workbook(COST_FILE)
            table = pd.read_excel(COST_FILE, sheet_name=COST_SHEET)
        required = {"scenario", "perturbation_twh", "objective", "unit"}
        if not required.issubset(table):
            raise ValueError(f"Cost input needs columns {sorted(required)}")
        table["scenario"] = table.scenario.map(scenario_name)
        if not table.unit.eq("EUR/a").all():
            raise ValueError("Objective unit must be explicitly EUR/a")
        table["source"] = str(COST_FILE)
    elif OBJECTIVE_NETWORK_ROOT is not None:
        import xarray as xr

        records = []
        cases = [("BASE", 0.0, "BASE")]
        cases += [
            (s, x, f"{SOURCE_NAMES[s]}_{x:g}")
            for s in COST_SCENARIOS
            for x in levels[s].values()
        ]
        for scenario, magnitude, folder in cases:
            path = OBJECTIVE_NETWORK_ROOT / folder / "networks" / OBJECTIVE_NETWORK_NAME
            report(
                f"OBJECTIVE INPUT: {path}; scenario={scenario}, perturbation={magnitude:g} TWh; EUR/a"
            )
            try:
                with xr.open_dataset(path) as network:
                    objective = float(network.attrs[OBJECTIVE_ATTRIBUTE])
                    constant = float(network.attrs.get("network__objective_constant", np.nan))
                if not np.isfinite(objective):
                    raise ValueError("Nonfinite objective")
            except (OSError, KeyError, ValueError) as error:
                report(f"Missing objective {scenario} at {magnitude:g} TWh: {error}", True)
                continue
            records.append(
                dict(
                    scenario=scenario,
                    perturbation_twh=magnitude,
                    objective=objective,
                    objective_constant=constant,
                    source=str(path),
                    unit="EUR/a",
                )
            )
        table = pd.DataFrame(records)
    else:
        raise FileNotFoundError("No objective source configured")
    if table.duplicated(["scenario", "perturbation_twh"]).any():
        raise ValueError("Ambiguous duplicate objective rows")
    for column in ("perturbation_twh", "objective"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if not np.isfinite(table[column]).all():
            report(f"Nonfinite cost data in {column}; affected curves will be skipped", True)
    base_rows = table[(table.scenario == "BASE") & (table.perturbation_twh == 0)]
    if len(base_rows) != 1 or not np.isfinite(base_rows.objective.iloc[0]) or base_rows.objective.iloc[0] <= 0:
        raise ValueError(
            "Cost data need one positive BASE objective at perturbation_twh=0"
        )
    base = float(base_rows.objective.iloc[0])
    rows = []
    for scenario in COST_SCENARIOS:
        part = table[table.scenario == scenario].copy()
        expected = set(levels[scenario].values())
        missing = expected - set(part.loc[np.isfinite(part.objective), "perturbation_twh"])
        if missing:
            report(f"Objective availability {scenario}: missing levels {sorted(missing)}; skipping this curve only", True)
            continue
        report(f"Objective availability {scenario}: complete, {len(expected)} perturbations plus BASE")
        part = part[part.perturbation_twh.isin(expected)]
        part = pd.concat([base_rows.assign(scenario=scenario), part], ignore_index=True)
        rows.append(part)
    if not rows:
        raise ValueError("No complete scenario objective curves available")
    result = pd.concat(rows, ignore_index=True)
    result["base_objective"] = base
    result["absolute_delta_eur"] = result.objective - base
    result["relative_delta_percent"] = 100 * result.absolute_delta_eur / base
    return result


def make_cost_figure(table: pd.DataFrame) -> None:
    table.to_csv(OUTPUT_DIR / "04_cost_sensitivity_by_family.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 4.7),
                             sharey=COST_SHARE_Y, layout="constrained")
    report(f"Cost panels: shared y-axis={COST_SHARE_Y}. "
           + ("Identical percentage scale across families." if COST_SHARE_Y else
              "Independent percentage scales: H-family changes below 2% were visually compressed by the roughly 12% E_ROAD excursion. Zero lines and explicit tick labels retained."))
    for panel, (ax, family, title) in enumerate(zip(axes.flat, "EFMH", FAMILY_TITLES)):
        for i, scenario in enumerate(s for s in SCENARIOS if s.startswith(family)):
            part = table[table.scenario == scenario].sort_values("perturbation_twh")
            if part.empty:
                continue
            ax.plot(part.perturbation_twh, part.relative_delta_percent,
                    color=TECH_COLORS[i], marker=("o", "s", "^")[i], markersize=3,
                    linestyle=LINE_STYLES[i], label=scenario)
        ax.set_title(f"({chr(97+panel)}) {title}", loc="left")
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.grid(axis="y", alpha=.18)
        ax.set_xlabel("Perturbation [TWh]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=3,
                      columnspacing=.8, handlelength=1.7)
        else:
            ax.text(.5, .5, "No objective data", transform=ax.transAxes, ha="center")
    for ax in axes[:, 0]:
        ax.set_ylabel("Objective change vs BASE [%]")
    save_figure(fig, "04_cost_sensitivity_by_family", "Cost sensitivity by family")


def get_carbon_series(data: dict, side: str, technology: str, group: str = "co2 stored") -> pd.Series:
    """Undo the generic workbook scale, then convert carbon mass explicitly."""
    values = get_energy_series(data, side, group, (technology,))
    scale = {"tCO2/a": 1e-6, "ktCO2/a": 1e-3, "MtCO2/a": 1.0}[CO2_INPUT_UNIT]
    return values / ENERGY_SCALE[ENERGY_INPUT_UNIT] * scale


def build_carbon_indicator(data: dict, scenario: str, levels: dict) -> tuple[pd.Series, list[dict]]:
    """Sum production entering co2 stored once; export source and sink details."""
    for side in ("supply", "consumption"):
        table = data[f"levels_{side}"]
        for group in ("co2", "co2 stored", "co2 sequestered"):
            labels = (list(table.xs(group, level="group").index)
                      if group in table.index.get_level_values("group") else [])
            report(f"CARBON INSPECTION {scenario}/{side}/{group}: {labels}")
            if group != "co2 stored":
                report(f"Excluded from capture indicator: {scenario}/{side}/{group}: "
                       "atmospheric balance or downstream sequestration, not capture supply")
    table = data["levels_supply"]
    available = (set(table.xs("co2 stored", level="group").index)
                 if "co2 stored" in table.index.get_level_values("group") else set())
    unknown = available - set(CARBON_CAPTURE_TECHNOLOGIES) - set(CARBON_EXCLUDED_SUPPLY)
    if unknown:
        report(f"Ambiguous new co2 stored supply labels: {sorted(unknown)}", True)
        raise ValueError("Review CARBON_CAPTURE_TECHNOLOGIES before aggregating new carbon sources")
    report(f"CARBON MAPPING {scenario}: {CARBON_CAPTURE_TECHNOLOGIES}; "
           f"excluded supply={CARBON_EXCLUDED_SUPPLY}; units {CO2_INPUT_UNIT} -> MtCO2/a")
    report("All co2 stored consumption is excluded from the capture sum: methanolisation, "
           "Fischer-Tropsch and Sabatier are utilisation; co2 sequestered is a sink; CO2 pipeline is transport.")
    records, parts = [], []
    for side, technologies in (("supply", CARBON_CAPTURE_TECHNOLOGIES),
                               ("consumption", CARBON_CONSUMPTION_TECHNOLOGIES)):
        for technology in technologies:
            values = get_carbon_series(data, side, technology)
            if side == "supply":
                parts.append(values)
            for name, magnitude in {"BASE": 0., **levels}.items():
                records.append(dict(scenario=scenario, perturbation=magnitude,
                                    side=side, carrier="co2 stored", technology=technology,
                                    included_in_indicator=side == "supply",
                                    level_mtco2=values[name], base_mtco2=values["BASE"],
                                    delta_mtco2=values[name]-values["BASE"]))
    total = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
    return total, records


def carbon_interpretation(scatter: pd.DataFrame, components: pd.DataFrame) -> list[str]:
    """Generate transparent descriptive comparisons, not causal attribution."""
    valid = scatter.dropna(subset=["delta_co2_stored"])
    spread = valid.delta_co2_stored.max() - valid.delta_co2_stored.min()
    report(f"Carbon indicator across {len(valid)} sensitivity points: "
           f"{valid.delta_co2_stored.min():.3f} to {valid.delta_co2_stored.max():.3f} MtCO2/a; "
           f"informative variation >0.5 MtCO2/a: {spread > .5}")
    notes = [f"Gestione del carbonio -- Delta cattura fra {valid.delta_co2_stored.min():+.2f} e "
             f"{valid.delta_co2_stored.max():+.2f} MtCO2/a nei punti disponibili; esclusi trasporto e doppio conteggio della sequestrazione. "
             "Il pannello (c) mostra sorgenti di cattura lorda annuale, non stoccaggio permanente o emissioni nette."]
    selected = ("M_IND", "M_AIR", "M_SHIP")
    level_sets = [set(scatter.loc[scatter.scenario == name, "perturbation"]) for name in selected]
    common = set.intersection(*level_sets)
    if not common:
        report("No common perturbation for M_IND/M_AIR/M_SHIP carbon comparison", True)
        return notes
    target = max(common)
    comparison = []
    for name in selected:
        row = scatter[(scatter.scenario == name) & (scatter.perturbation == target)].iloc[0]
        part = components[(components.scenario == name) & (components.perturbation == target)]
        item = dict(scenario=name, perturbation=target, delta_oil_supply=row.delta_oil_supply,
                    delta_fossil_gas_supply=row.delta_fossil_gas_supply,
                    delta_co2_stored=row.delta_co2_stored, captured_co2=row.co2_stored,
                    delta_electrolysis_h2=row.delta_electrolysis_h2,
                    delta_smr_h2=row.delta_smr_h2, delta_smr_cc_h2=row.delta_smr_cc_h2)
        for technology, column, side in (
            ("SMR CC", "smr_cc_capture", "supply"),
            ("process emissions CC", "process_capture", "supply"),
            ("solid biomass for industry CC", "biomass_industry_capture", "supply"),
            ("methanolisation", "methanolisation_co2_use", "consumption"),
            ("co2 sequestered", "sequestration", "consumption")):
            match = part[(part.technology == technology) & (part.side == side)]
            item[column] = float(match.level_mtco2.iloc[0])
        comparison.append(item)
    frame = pd.DataFrame(comparison).set_index("scenario")
    frame["process_capture_share"] = frame.process_capture / frame.captured_co2
    frame["smr_cc_capture_share"] = frame.smr_cc_capture / frame.captured_co2
    frame.to_csv(OUTPUT_DIR / "05_carbon_management_comparison.csv")
    report(f"Carbon-management comparison at common target {target:g} TWh:\n{frame.to_string()}")
    ind = frame.loc["M_IND"]
    others = frame.loc[["M_AIR", "M_SHIP"]]
    supported = (others.delta_fossil_gas_supply.gt(ind.delta_fossil_gas_supply).all()
                 and others.smr_cc_capture.gt(ind.smr_cc_capture).all()
                 and others.process_capture.lt(ind.process_capture).all()
                 and others.process_capture_share.lt(ind.process_capture_share).all()
                 and others.delta_electrolysis_h2.lt(ind.delta_electrolysis_h2).all())
    report(f"M_IND distinction at {target:g} TWh supported={supported}: test smaller gas increase, "
           "less SMR-CC capture, more process capture and larger electrolysis increase than BOTH M_AIR/M_SHIP.")
    for name, row in frame.iterrows():
        notes.append(f"Confronto metanolo -- {name} a {target:g} TWh: delta oil {row.delta_oil_supply:+.1f}, "
                     f"delta gas {row.delta_fossil_gas_supply:+.1f} TWh/a; cattura totale {row.captured_co2:.2f} "
                     f"(delta {row.delta_co2_stored:+.2f}), process emissions CC {row.process_capture:.2f}, "
                     f"SMR CC {row.smr_cc_capture:.2f}, solid biomass for industry CC {row.biomass_industry_capture:.2f}, "
                     f"CO2 a metanolazione {row.methanolisation_co2_use:.2f} MtCO2/a; "
                     f"delta elettrolisi {row.delta_electrolysis_h2:+.1f} TWh/a; quota cattura di processo {100*row.process_capture_share:.1f} percento.")
    notes.append("Distinzione M_IND -- " + (
        "Al target comune la minore crescita del gas coincide con piu cattura di processo, meno cattura SMR CC e piu elettrolisi rispetto a M_AIR e M_SHIP."
        if supported else "Il contrasto ipotizzato non e supportato in tutte le sue componenti al target comune.")
        + " Associazione descrittiva, non attribuzione causale.")
    # Closest total-capture pair exposes composition differences hidden by a scalar.
    pairs = [(a, b) for i, a in enumerate(selected) for b in selected[i+1:]]
    a, b = min(pairs, key=lambda pair: abs(frame.loc[pair[0], "delta_co2_stored"]-frame.loc[pair[1], "delta_co2_stored"]))
    carbon_gap = abs(frame.loc[a, "delta_co2_stored"]-frame.loc[b, "delta_co2_stored"])
    gas_gap = abs(frame.loc[a, "delta_fossil_gas_supply"]-frame.loc[b, "delta_fossil_gas_supply"])
    report(f"Capture-total limitation: {a}/{b} at {target:g} TWh differ by only {carbon_gap:.3f} MtCO2/a "
           f"in delta capture but {gas_gap:.3f} TWh/a in delta fossil gas. The source-pathway panel resolves composition, "
           "but aggregate capture does not resolve capture-source composition; no causal attribution from the plotted association.")
    notes.append(f"Motivazione del pannello (c) -- {a} e {b} differiscono di {carbon_gap:.2f} MtCO2/a nella variazione di cattura "
                 f"ma di {gas_gap:.1f} TWh/a nella risposta del gas. Per distinguere le configurazioni "
                 "servono i componenti esportati, non solo il totale. La scomposizione non dimostra da sola causalita.")
    return notes


def atmospheric_emissions_tables(sensitivity: dict, levels: dict) -> tuple[pd.DataFrame, list[str]]:
    """Use positive atmospheric CO2 workbook entries, not emission factors."""
    if set(CO2_OIL_TECHNOLOGIES) & set(CO2_GAS_TECHNOLOGIES):
        raise ValueError("Oil and gas emissions mappings overlap")
    report(f"Atmospheric oil-related CO2 labels: {CO2_OIL_TECHNOLOGIES}")
    report(f"Atmospheric gas-related CO2 labels: {CO2_GAS_TECHNOLOGIES}")
    report(f"Atmospheric CO2 labels considered and excluded: {CO2_EMISSIONS_EXCLUDED}")
    report("Mapping evidence: prepare_sector_network.py add_carrier_buses creates oil refining "
           "from oil primary to oil with separate atmospheric CO2; add_oil_from_h2_demand reduces "
           "ordinary oil loads and adds oil from H2 emissions for the reserved share. Both are included "
           "once. HVC to air releases a separate industrial stock and is excluded.")
    report("Atmospheric CC rows are uncaptured residual emissions only; captured outputs on co2 stored "
           "are not added to panel b. Gross oil/gas carrier emissions include blended non-fossil fuels; "
           "negative atmospheric uptake, process emissions, coal and methanol emissions are outside this subtotal. "
           "The compensation diagonal is a benchmark, not the full model carbon constraint.")
    records, components = [], []
    for scenario, data in sensitivity.items():
        table = data["levels_supply"]
        available = set(table.xs("co2", level="group").index)
        unknown = available - set(CO2_OIL_TECHNOLOGIES) - set(CO2_GAS_TECHNOLOGIES) - set(CO2_EMISSIONS_EXCLUDED)
        if unknown:
            report(f"Unclassified atmospheric CO2 labels in {scenario}: {sorted(unknown)}", True)
            raise ValueError("Review atmospheric CO2 mappings before plotting")
        report(f"{scenario}: observed excluded atmospheric labels={sorted(available & set(CO2_EMISSIONS_EXCLUDED))}")
        totals = {}
        for category, technologies in (("oil", CO2_OIL_TECHNOLOGIES), ("gas", CO2_GAS_TECHNOLOGIES)):
            parts = []
            for technology in technologies:
                values = get_carbon_series(data, "supply", technology, group="co2")
                parts.append(values)
                for name, magnitude in {"BASE": 0., **levels[scenario]}.items():
                    components.append(dict(scenario=scenario, perturbation=magnitude,
                                           category=category, technology=technology,
                                           level_mtco2=values[name], base_mtco2=values["BASE"],
                                           delta_mtco2=values[name]-values["BASE"]))
            totals[category] = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        for name, magnitude in levels[scenario].items():
            oil, gas = totals["oil"], totals["gas"]
            records.append(dict(family=scenario[0], scenario=scenario, perturbation=magnitude,
                                delta_co2_oil=oil[name]-oil["BASE"],
                                delta_co2_gas=gas[name]-gas["BASE"],
                                co2_oil=oil[name], co2_gas=gas[name],
                                base_co2_oil=oil["BASE"], base_co2_gas=gas["BASE"],
                                nominal_point=bool(np.isclose(magnitude, NOMINAL_TARGET_TWH[scenario]))))
    result = pd.DataFrame(records)
    result["delta_co2_oil_plus_gas"] = result.delta_co2_oil + result.delta_co2_gas
    result.to_csv(OUTPUT_DIR / "06a_oil_gas_co2_substitution.csv", index=False)
    pd.DataFrame(components).to_csv(OUTPUT_DIR / "06a_co2_emission_components.csv", index=False)
    valid = result.dropna(subset=["delta_co2_oil", "delta_co2_gas"])
    magnitude = valid.delta_co2_oil.abs() + valid.delta_co2_gas.abs()
    material = valid[magnitude > 1.0]
    opposite = material.delta_co2_oil * material.delta_co2_gas < 0
    residual = material.delta_co2_oil_plus_gas.abs() / (
        material.delta_co2_oil.abs() + material.delta_co2_gas.abs())
    rho = valid.delta_co2_oil.corr(valid.delta_co2_gas)
    near = int((residual <= .1).sum())
    visible = rho <= -.5 and opposite.mean() >= .5
    report(f"Oil/gas CO2 compensation: n={len(valid)}, r={rho:.4f}; material points (>1 Mt/a total absolute change)="
           f"{len(material)}, opposite-sign={int(opposite.sum())}, within 10% normalized compensation residual={near}; "
           f"visible inverse compensation tendency={visible}, not necessarily exact compensation.")
    for family, part in valid.groupby("family", sort=False):
        report(f"CO2 compensation {family}: sum delta range {part.delta_co2_oil_plus_gas.min():.3f} "
               f"to {part.delta_co2_oil_plus_gas.max():.3f} MtCO2/a")
    notes = [f"Compensazione CO2 -- r={rho:+.3f}; {int(opposite.sum())}/{len(material)} punti materiali con variazioni "
             f"oil/gas di segno opposto; {near}/{len(material)} con residuo assoluto della somma entro il 10 percento "
             "della somma dei moduli. " + ("Tendenza inversa visibile." if visible else "Tendenza inversa non robusta secondo il criterio configurato.")
             + " La diagonale rappresenta compensazione perfetta solo fra questi due sottoinsiemi; processi, metanolo, "
             "carbone e assorbimenti restano fuori. Emissioni lorde dei carrier, non attribuzione esclusiva a combustibili fossili."]
    return result, notes


def methanol_capture_table(components: pd.DataFrame) -> pd.DataFrame:
    """Export plotted deltas plus absolute/base levels and the unplotted remainder."""
    records = []
    for scenario in ("M_AIR", "M_IND", "M_SHIP"):
        part = components[(components.scenario == scenario) & components.side.eq("supply")]
        for magnitude, rows in part.groupby("perturbation", sort=True):
            row = dict(scenario=scenario, perturbation=magnitude, unit="MtCO2/a",
                       plotted_metric="delta_vs_BASE")
            indexed = rows.set_index("technology")
            for column, technology in METHANOL_CAPTURE_SOURCES.items():
                row[column] = indexed.loc[technology, "delta_mtco2"]
                row[column+"_level"] = indexed.loc[technology, "level_mtco2"]
                row[column+"_base"] = indexed.loc[technology, "base_mtco2"]
            for suffix, source in (("", "delta_mtco2"), ("_level", "level_mtco2"), ("_base", "base_mtco2")):
                row["total_CO2_stored_supply"+suffix] = rows[source].sum(min_count=len(CARBON_CAPTURE_TECHNOLOGIES))
            row["other_capture_delta"] = row["total_CO2_stored_supply"] - sum(row[k] for k in METHANOL_CAPTURE_SOURCES)
            records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "06b_methanol_captured_co2.csv", index=False)
    report(f"Figure 6b captured-CO2 sources: {METHANOL_CAPTURE_SOURCES}; plotted values are changes vs BASE. "
           f"All 8 capture sources remain in total_CO2_stored_supply; maximum absolute unplotted remainder="
           f"{result.other_capture_delta.abs().max():.6g} MtCO2/a")
    return result


# Give the dense cluster the main panel; retain all points in an overview.
def add_full_range_inset(ax: plt.Axes, table: pd.DataFrame, xcol: str, ycol: str,
                     xlim: tuple[float, float], ylim: tuple[float, float],
                     bounds: tuple[float, float, float, float] = (.59, .61, .37, .34)) -> None:
    full_xlim, full_ylim = ax.get_xlim(), ax.get_ylim()
    ax.set(xlim=xlim, ylim=ylim)
    inset = ax.inset_axes(bounds)
    for family in "EFMH":
        for i, scenario in enumerate(s for s in SCENARIOS if s.startswith(family)):
            part = table[table.scenario == scenario].sort_values("perturbation")
            inset.plot(part[xcol], part[ycol], color=FAMILY_COLORS[family],
                       marker=("o", "s", "^")[i], markersize=2.1, linewidth=.5,
                       alpha=.85)
            nominal = part[part.nominal_point]
            inset.scatter(nominal[xcol], nominal[ycol], marker=("o", "s", "^")[i],
                          s=20, facecolors="none", edgecolors="black", linewidths=.45)
    inset.set_xlim(*full_xlim)
    inset.set_ylim(*full_ylim)
    inset.axvline(0, color="0.4", linewidth=.45)
    inset.axhline(0, color="0.4", linewidth=.45)
    inset.grid(alpha=.12, linewidth=.35)
    inset.tick_params(labelsize=6, pad=1)
    inset.xaxis.set_major_locator(plt.MaxNLocator(3))
    inset.yaxis.set_major_locator(plt.MaxNLocator(3))
    from matplotlib.patches import Rectangle
    inset.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0],
                              ylim[1] - ylim[0], fill=False,
                              edgecolor="0.35", linewidth=.7, linestyle="--"))
    inset.set_title("Full range", fontsize=6.5, pad=3)


def make_carbon_substitution_figure(sensitivity: dict, levels: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare primary oil with fossil gas, excluding downstream fuels and transfers."""
    scatter, biogas, components, carbon_components = [], [], [], []
    report(f"Primary oil mapping: {PRIMARY_OIL_SUPPLY}; excludes refining outputs and alternative oil routes.")
    report(f"Fossil gas mapping: {FOSSIL_GAS_SUPPLY}; excludes biogas, Sabatier and pipelines")
    report(f"Biogas-to-gas mapping: {BIOGAS_TO_GAS_SUPPLY}")
    for scenario in SCENARIOS:
        data = sensitivity[scenario]
        oil = get_supply_series(data, *PRIMARY_OIL_SUPPLY)
        if oil.isna().any():
            raise ValueError(f"{scenario}: missing primary-oil supply values")
        components += sensitivity_records(oil, scenario, "oil primary", levels[scenario])
        gas = get_supply_series(data, *FOSSIL_GAS_SUPPLY)
        carbon, carbon_rows = build_carbon_indicator(data, scenario, levels[scenario])
        carbon_components.extend(carbon_rows)
        h2 = {name: get_supply_series(data, "H2", aliases) for name, aliases in HYDROGEN.items()}
        for name, magnitude in levels[scenario].items():
            scatter.append(dict(family=scenario[0], scenario=scenario, perturbation=magnitude,
                                oil_supply=oil[name], base_oil_supply=oil["BASE"],
                                oil_supply_carrier=PRIMARY_OIL_SUPPLY[0],
                                oil_supply_technology=PRIMARY_OIL_SUPPLY[1][0],
                                fossil_gas_supply=gas[name], base_fossil_gas_supply=gas["BASE"],
                                delta_oil_supply=oil[name]-oil["BASE"],
                                delta_fossil_gas_supply=gas[name]-gas["BASE"],
                                co2_stored=carbon[name], base_co2_stored=carbon["BASE"],
                                delta_co2_stored=carbon[name]-carbon["BASE"],
                                delta_electrolysis_h2=h2["Electrolysis"][name]-h2["Electrolysis"]["BASE"],
                                delta_smr_h2=h2["SMR"][name]-h2["SMR"]["BASE"],
                                delta_smr_cc_h2=h2["SMR CC"][name]-h2["SMR CC"]["BASE"],
                                nominal_point=bool(np.isclose(magnitude, NOMINAL_TARGET_TWH[scenario]))))
        if scenario.startswith("F"):
            series = get_supply_series(data, *BIOGAS_TO_GAS_SUPPLY)
            for name, magnitude in {"BASE": 0., **levels[scenario]}.items():
                biogas.append(dict(scenario=scenario, perturbation=magnitude,
                                   biogas_to_gas=series[name], base_biogas_to_gas=series["BASE"],
                                   delta_biogas_to_gas=series[name]-series["BASE"]))
    scatter_table, biogas_table = pd.DataFrame(scatter), pd.DataFrame(biogas)
    panel_a = scatter_table.drop(columns=["co2_stored", "base_co2_stored", "delta_co2_stored",
                                          "delta_electrolysis_h2", "delta_smr_h2", "delta_smr_cc_h2"])
    panel_a.to_csv(OUTPUT_DIR / "05a_oil_gas_supply.csv", index=False)
    carbon_table = pd.DataFrame(carbon_components)
    carbon_table.to_csv(OUTPUT_DIR / "05_carbon_indicator_components.csv", index=False)
    emissions, emissions_notes = atmospheric_emissions_tables(sensitivity, levels)
    capture = methanol_capture_table(carbon_table)
    biogas_table.to_csv(OUTPUT_DIR / "05b_biogas_fossil_persistence.csv", index=False)
    carbon_notes = carbon_interpretation(scatter_table, carbon_table) + emissions_notes
    scatter_table.attrs["carbon_notes"] = carbon_notes
    scatter_table.to_csv(OUTPUT_DIR / "05_carbon_constrained_fuel_substitution_scatter.csv", index=False)
    biogas_table.to_csv(OUTPUT_DIR / "05_carbon_constrained_fuel_substitution_biogas.csv", index=False)
    pd.DataFrame(components).to_csv(OUTPUT_DIR / "05_oil_supply_components.csv", index=False)
    valid = scatter_table.dropna(subset=["delta_oil_supply", "delta_fossil_gas_supply"])
    if len(valid) != len(scatter_table):
        report(f"Scatter: {len(scatter_table)-len(valid)} points missing; retained as NaN in CSV", True)
    rho = valid.delta_oil_supply.corr(valid.delta_fossil_gas_supply)
    report(f"Oil-vs-fossil-gas scatter: n={len(valid)}, Pearson r={rho:.4f}; "
           + ("noticeable inverse pattern (r <= -0.5)" if rho <= -.5 else "no strong inverse pattern by r <= -0.5 criterion")
           + "; exploratory, dependent scenario samples, primary fossil oil and fossil-gas sources; no causal inference.")
    for family, part in valid.groupby("family", sort=False):
        report(f"Oil/gas within-family {family}: r={part.delta_oil_supply.corr(part.delta_fossil_gas_supply):.4f}, n={len(part)}")
    endpoints = []
    for scenario, part in biogas_table.groupby("scenario", sort=False):
        tail = part.sort_values("perturbation").tail(PLATEAU_MIN_POINTS)
        stable = tail.biogas_to_gas.notna().all() and np.ptp(tail.biogas_to_gas) <= PLATEAU_TOL_TWH
        if stable:
            endpoints.append(tail.biogas_to_gas.mean())
        report(f"Biogas plateau {scenario}: visible={stable}, last {len(tail)} points "
               f"({tail.perturbation.min():g}--{tail.perturbation.max():g} TWh), "
               f"production={tail.biogas_to_gas.min():.3f}--{tail.biogas_to_gas.max():.3f} TWh/a")
    common = len(endpoints) == 3 and np.ptp(endpoints) <= PLATEAU_TOL_TWH
    report(f"Common F-family biogas plateau: {common}; "
           + (f"approximately {np.mean(endpoints):.3f} TWh/a absolute production" if common else "not established"))
    fig_fuel, (ax_a, ax_d) = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.9), layout="constrained")
    fig_carbon = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 6.3), layout="constrained")
    grid = fig_carbon.add_gridspec(3, 2, width_ratios=(1.1, 1))
    ax_b = fig_carbon.add_subplot(grid[:, 0])
    capture_axes = []
    for row in range(3):
        capture_axes.append(fig_carbon.add_subplot(
            grid[row, 1], sharex=capture_axes[0] if row else None,
            sharey=capture_axes[0] if row else None))
    for family in "EFMH":
        for i, scenario in enumerate(s for s in SCENARIOS if s.startswith(family)):
            marker = ("o", "s", "^")[i]
            for ax, table, xcol, ycol in (
                (ax_a, scatter_table, "delta_oil_supply", "delta_fossil_gas_supply"),
                (ax_b, emissions, "delta_co2_oil", "delta_co2_gas")):
                part = table[table.scenario == scenario].sort_values("perturbation")
                ax.plot(part[xcol], part[ycol], color=FAMILY_COLORS[family], marker=marker,
                        markersize=3.2, linewidth=.65, alpha=.85, label=scenario)
                nominal = part[part.nominal_point]
                ax.scatter(nominal[xcol], nominal[ycol], marker=marker, s=45,
                           facecolors="none", edgecolors="black", linewidths=.7, zorder=4)
    ax_a.set(title="(a) Primary oil and fossil-gas supply", xlabel="Δ primary-oil supply [TWh/a]",
             ylabel="Δ fossil-gas supply [TWh/a]")
    ax_b.set(title="(a) Oil- and gas-related\nCO₂ emissions",
             xlabel="Δ oil-related CO₂ [MtCO₂/a]", ylabel="Δ gas-related CO₂ [MtCO₂/a]")
    for ax in (ax_a, ax_b):
        ax.axvline(0, color="0.4", linewidth=.7)
    xlim, ylim = ax_b.get_xlim(), ax_b.get_ylim()
    low, high = max(xlim[0], -ylim[1]), min(xlim[1], -ylim[0])
    ax_b.plot([low, high], [-low, -high], color="0.5", linestyle="--",
              linewidth=.75, zorder=0)
    ax_b.set(xlim=xlim, ylim=ylim)
    ax_b.plot([], [], color="0.5", linestyle="--", linewidth=.75,
              label="ΔCO₂ oil + ΔCO₂ gas = 0")
    ax_b.legend(handles=[ax_b.lines[-1]], loc="upper center",
                bbox_to_anchor=(.5, -.10), frameon=False, fontsize=6.5)

    add_full_range_inset(ax_a, scatter_table, "delta_oil_supply", "delta_fossil_gas_supply",
                     (-300, 30), (-40, 720))
    add_full_range_inset(ax_b, emissions, "delta_co2_oil", "delta_co2_gas",
                     (-75, 10), (-12, 120))
    report("Figures 5a and 6a: zoomed main axes with full-range overview insets.")
    handles, names = ax_a.get_legend_handles_labels()
    order = np.arange(len(SCENARIOS)).reshape(2, 6).T.ravel()
    for figure in (fig_fuel, fig_carbon):
        figure.legend([handles[i] for i in order], [names[i] for i in order],
                      loc="outside upper center", ncol=6, frameon=False,
                      title="Panel (a): outlined marker = nominal target; changes vs BASE",
                      title_fontsize=7, columnspacing=1, handletextpad=.3, handlelength=1.3)
    # Diverging stacks preserve any negative changes instead of hiding them.
    for ax, scenario in zip(capture_axes, ("M_AIR", "M_IND", "M_SHIP")):
        part = capture[capture.scenario == scenario].sort_values("perturbation")
        positive, negative = np.zeros(len(part)), np.zeros(len(part))
        for j, (column, label) in enumerate(METHANOL_CAPTURE_SOURCES.items()):
            values = part[column].to_numpy()
            ax.bar(part.perturbation, values, width=32,
                   bottom=np.where(values >= 0, positive, negative),
                   color=TECH_COLORS[j], label=label)
            positive += np.clip(values, 0, None)
            negative += np.clip(values, None, 0)
        ax.plot(part.perturbation, part.total_CO2_stored_supply,
                color="black", marker="o", markersize=2.5, linewidth=1,
                label="Total captured CO₂ (all sources)")
        ax.set_title(("(b) Captured-CO₂ supply\n" if ax is capture_axes[0] else "") + scenario)
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.grid(axis="y", alpha=.15)
        ax.tick_params(labelbottom=ax is capture_axes[-1])
        ax.set_xticks([0, 50, 100, 150, 200, 250])
    capture_axes[1].set_ylabel("Δ captured CO₂ supply [MtCO₂/a]")
    capture_axes[-1].set_xlabel("Perturbation [TWh]")
    stack_handles, stack_labels = capture_axes[0].get_legend_handles_labels()
    fig_carbon.legend(stack_handles, stack_labels, loc="outside lower center",
                      ncol=2, frameon=False, fontsize=7)

    for i, scenario in enumerate(("F_HEAT", "F_IND", "F_ROAD")):
        part = biogas_table[biogas_table.scenario == scenario].sort_values("perturbation")
        ax_d.plot(part.perturbation, part.delta_biogas_to_gas, color=TECH_COLORS[i],
                  linestyle=LINE_STYLES[i], marker=("o","s","^")[i], markersize=3, label=scenario)
    ax_d.set(title="(b) Biogas under fossil persistence",
             xlabel="Perturbation [TWh]", ylabel="Δ biogas-to-gas [TWh/a]")
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,-.23), ncol=3,
                columnspacing=.7, handlelength=1.5)
    for ax in (ax_a, ax_b, ax_d):
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.grid(axis="y", alpha=.15)
    save_figure(fig_fuel, "05_fuel_substitution_and_biogas", "Fuel substitution and biogas")
    save_figure(fig_carbon, "06_carbon_substitution_and_management", "Carbon substitution and management")
    return scatter_table, biogas_table


GAS_ELECTRICITY_TECHNOLOGIES = (
    "OCGT", "CCGT", "urban central gas CHP", "urban central gas CHP CC",
)
WIND_ELECTRICITY_TECHNOLOGIES = ("onwind",)


def make_gas_wind_generation_figure(sensitivity: dict, levels: dict) -> pd.DataFrame:
    """Compare onshore wind with gas electricity and electrolytic H2 output."""
    records, components = [], []
    mappings = {"gas": GAS_ELECTRICITY_TECHNOLOGIES,
                "wind": WIND_ELECTRICITY_TECHNOLOGIES}
    for scenario in SCENARIOS:
        totals = {}
        for category, technologies in mappings.items():
            parts = []
            for technology in technologies:
                values = get_supply_series(sensitivity[scenario], "AC", (technology,))
                if values.isna().any():
                    raise ValueError(f"{scenario}: missing AC output for {technology}")
                parts.append(values)
                components.extend(dict(row, category=category, carrier="AC") for row in
                                  sensitivity_records(values, scenario, technology, levels[scenario]))
            totals[category] = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        electrolysis = get_supply_series(sensitivity[scenario], "H2", HYDROGEN["Electrolysis"])
        if electrolysis.isna().any():
            raise ValueError(f"{scenario}: missing electrolytic H2 output")
        components.extend(dict(row, category="electrolysis", carrier="H2") for row in
                          sensitivity_records(electrolysis, scenario, "H2 Electrolysis", levels[scenario]))
        for name, magnitude in levels[scenario].items():
            row = dict(scenario=scenario, family=scenario[0], perturbation=magnitude,
                       nominal_point=bool(np.isclose(magnitude, NOMINAL_TARGET_TWH[scenario])))
            for category, values in totals.items():
                row[f"{category}_generation"] = values[name]
                row[f"base_{category}_generation"] = values["BASE"]
                row[f"delta_{category}_generation"] = values[name] - values["BASE"]
            row.update(electrolysis_h2=electrolysis[name],
                       base_electrolysis_h2=electrolysis["BASE"],
                       delta_electrolysis_h2=electrolysis[name] - electrolysis["BASE"])
            records.append(row)
    table = pd.DataFrame(records)
    stem = "08_gas_wind_generation_substitution"
    table.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    pd.DataFrame(components).to_csv(OUTPUT_DIR / "08_gas_wind_generation_components.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 4.2), layout="constrained")
    core = table[table.scenario != "E_ROAD"]
    def padded_range(values):
        low, high = min(0, values.min()), max(0, values.max())
        pad = max((high - low) * .08, 1.)
        return low - pad, high + pad
    ylim = padded_range(core.delta_wind_generation)
    for ax, xcol, title, xlabel in (
        (axes[0], "delta_gas_generation", "(a) Gas-fired electricity", "Δ gas-fired generation [TWh/a]"),
        (axes[1], "delta_electrolysis_h2", "(b) Electrolysis", "Δ electrolytic H₂ output [TWh/a]"),
    ):
        for family in "EFMH":
            for i, scenario in enumerate(s for s in SCENARIOS if s.startswith(family)):
                part = table[table.scenario == scenario].sort_values("perturbation")
                marker = ("o", "s", "^")[i]
                ax.plot(part[xcol], part.delta_wind_generation,
                        color=FAMILY_COLORS[family], marker=marker, markersize=3.2,
                        linewidth=.65, alpha=.85, label=scenario)
                nominal = part[part.nominal_point]
                ax.scatter(nominal[xcol], nominal.delta_wind_generation,
                           marker=marker, s=45, facecolors="none", edgecolors="black",
                           linewidths=.7, zorder=4)
        ax.set(title=title, xlabel=xlabel, ylabel="Δ onshore wind generation [TWh/a]")
        ax.axvline(0, color="0.4", linewidth=.7)
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.grid(axis="y", alpha=.15)
        add_full_range_inset(ax, table, xcol, "delta_wind_generation",
                             padded_range(core[xcol]), ylim,
                             bounds=(.59 if ax is axes[0] else .06, .61, .37, .34))
    handles, names = axes[0].get_legend_handles_labels()
    order = np.arange(len(SCENARIOS)).reshape(2, 6).T.ravel()
    fig.legend([handles[i] for i in order], [names[i] for i in order],
               loc="outside upper center", ncol=6, frameon=False,
               title="Outlined marker = nominal target; all changes vs BASE",
               title_fontsize=7, columnspacing=1, handletextpad=.3, handlelength=1.3)
    save_figure(fig, stem, "Onshore wind, gas-fired generation and electrolysis")
    correlation_rows = []
    for panel, xcol in (("a", "delta_gas_generation"), ("b", "delta_electrolysis_h2")):
        for group, part in [("All", table), *list(table.groupby("family", sort=False))]:
            rho = part[xcol].corr(part.delta_wind_generation)
            correlation_rows.append(dict(panel=panel, x_variable=xcol, group=group,
                                         n=len(part), pearson_r=rho))
            report(f"Onshore wind panel {panel}, {group}: n={len(part)}, Pearson r={rho:.4f}; dependent samples, no causal inference.")
    pd.DataFrame(correlation_rows).to_csv(OUTPUT_DIR / "08_gas_wind_generation_correlations.csv", index=False)
    (OUTPUT_DIR / f"{stem}_caption.tex").write_text(
        r"Changes in onshore wind generation relative to BASE versus (a) gas-fired electricity generation and (b) electrolytic hydrogen output. Gas-fired generation includes OCGT, CCGT, and gas CHP with and without carbon capture; only AC electricity output is counted. Wind includes only onwind, excluding offshore generation. Electrolysis is H2 output from H2 Electrolysis, consistent with Figure 2, rather than electricity input. Connected markers follow increasing perturbation levels; outlined markers identify nominal targets. Main axes magnify the central trajectories; full-range insets retain all scenarios, including E\_ROAD, with dashed boxes marking the main-axis limits. All values are changes in annual energy output in TWh/a. Associations do not establish causal substitution." + "\n",
        encoding="utf-8")
    return table


def make_electricity_response_ratio(data: dict) -> pd.DataFrame:
    ac = data["levels_supply"].xs("AC", level="group")
    totals = ac.clip(lower=0).sum(axis=0, min_count=len(ac))
    records = []
    for scenario in SCENARIOS:
        target = NOMINAL_TARGET_TWH[scenario]
        if not np.isfinite(target) or target <= 0:
            raise ValueError(
                f"Set a positive exogenous NOMINAL_TARGET_TWH for {scenario}"
            )
        value = totals[scenario]
        delta = value - totals["BASE"]
        records.append(
            dict(
                scenario=scenario,
                ac_supply_twh=value,
                base_ac_supply_twh=totals["BASE"],
                delta_ac_supply_twh=delta,
                target_perturbation_twh=target,
                response_ratio=delta / target,
            )
        )
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT_DIR / "07_electricity_response_ratio.csv", index=False)
    ac.to_csv(OUTPUT_DIR / "07_ac_supply_components_twh.csv")
    report(
        "Figure 7 uses gross positive AC supply, including AC/DC transfers and storage supply; it is not net generation. Low voltage is excluded. The existing balance plot merges AC and low voltage, so its total is different."
    )
    if not MAKE_ELECTRICITY_RESPONSE_FIGURE:
        report("Figure 7 disabled; underlying response-ratio tables still exported")
        return table
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 2.9), layout="constrained")
    ax.bar(
        range(12),
        table.response_ratio,
        color=[FAMILY_COLORS[s[0]] for s in SCENARIOS],
        width=0.72,
    )
    ax.set_xticks(range(12), SCENARIOS, rotation=55, ha="right")
    ax.set_ylabel("Electricity-system response ratio\n[TWh_AC / TWh_perturbation]")
    ax.axhline(0, color="0.2", linewidth=0.9)
    separators(ax)
    ax.grid(axis="y", alpha=0.15)
    save_figure(
        fig, "07_electricity_response_ratio", "Electricity-system response ratio"
    )
    return table


def detect_regimes(
    series: pd.Series, levels: dict[str, float], scenario: str, label: str
) -> list[dict]:
    """Flag sampled intervals, not exact physical thresholds or causal changes."""
    names = ["BASE", *levels]
    x = np.array([0.0, *levels.values()])
    y = series.reindex(names).to_numpy(dtype=float)
    flags = []
    if not np.isfinite(y).all():
        return flags
    slopes = np.diff(y) / np.diff(x)
    for i in range(1, len(x)):
        kind = None
        if abs(y[i - 1]) <= MATERIAL_TWH < abs(y[i]):
            kind = "activation"
        elif (
            i >= 2
            and abs(y[i] - y[i - 1]) <= PLATEAU_TOL_TWH
            and abs(y[i - 1] - y[i - 2]) > MATERIAL_TWH
        ):
            kind = "plateau onset candidate"
        elif (
            i >= 2
            and abs(y[i] - y[i - 1]) > MATERIAL_TWH
            and abs(y[i - 1] - y[i - 2]) > MATERIAL_TWH
        ):
            a, b = slopes[i - 2 : i]
            if a * b < 0 or max(abs(a), abs(b)) > SLOPE_CHANGE_FACTOR * max(
                min(abs(a), abs(b)), 0.01
            ):
                kind = "slope change / suspicious discontinuity candidate"
        if kind:
            record = dict(
                scenario=scenario,
                quantity=label,
                kind=kind,
                lower_twh=x[i - 1],
                upper_twh=x[i],
                value_twh=y[i],
            )
            flags.append(record)
            report(
                f"REGIME {scenario}/{label}: {kind} in ({x[i - 1]:g}, {x[i]:g}] TWh; value={y[i]:.3f} TWh/a"
            )
    return flags


def additional_evidence(
    sensitivity: dict, levels: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = [
        ("supply", "methanol", "biomass-to-methanol"),
        ("supply", "methanol", "methanolisation"),
        ("supply", "gas", "biogas to gas"),
        ("supply", "AC", "onwind"),
        ("supply", "AC", "solar-hsat"),
        ("supply", "AC", "urban central gas CHP"),
        ("supply", "AC", "CCGT"),
        ("supply", "AC", "OCGT"),
        ("consumption", "shipping methanol", "shipping methanol"),
        ("consumption", "land transport oil", "land transport oil"),
        ("consumption", "methanol", "methanol-to-kerosene"),
        ("consumption", "solid biomass", "electrobiofuels"),
        ("consumption", "gas", "gas for industry"),
        ("consumption", "co2 stored", "methanolisation"),
    ]
    records, flags = [], []
    for scenario, data in sensitivity.items():
        for side, group, tech in queries:
            series = get_energy_series(data, side, group, (tech,))
            label = f"{side}/{group}/{tech}"
            # CO2 workbook values are tonnes, not MWh; undo the energy conversion.
            unit = "MtCO2/a" if group == "co2 stored" else "TWh/a"
            if unit == "MtCO2/a":
                series = series / ENERGY_SCALE[ENERGY_INPUT_UNIT] * 1e-6
            rows = sensitivity_records(series, scenario, label, levels[scenario])
            records.extend(dict(row, unit=unit) for row in rows)
            if unit == "TWh/a":
                flags += detect_regimes(series, levels[scenario], scenario, label)
        for label, aliases in HYDROGEN.items():
            series = get_supply_series(data, "H2", aliases)
            records += [
                dict(row, unit="TWh/a")
                for row in sensitivity_records(
                    series, scenario, label, levels[scenario]
                )
            ]
            flags += detect_regimes(series, levels[scenario], scenario, label)
    evidence = pd.DataFrame(records).rename(
        columns={"level_twh": "level", "base_twh": "base", "delta_twh": "delta"}
    )
    flag_table = pd.DataFrame(
        flags,
        columns=["scenario", "quantity", "kind", "lower_twh", "upper_twh", "value_twh"],
    )
    evidence.to_csv(OUTPUT_DIR / "interpretation_evidence.csv", index=False)
    flag_table.to_csv(OUTPUT_DIR / "regime_candidates.csv", index=False)
    return evidence, flag_table


def write_latex_ideas(
    capacity: pd.DataFrame, hydrogen: pd.DataFrame, biomass: pd.DataFrame,
    ratio: pd.DataFrame, evidence: pd.DataFrame, flags: pd.DataFrame,
    costs: pd.DataFrame | None, scatter: pd.DataFrame, biogas: pd.DataFrame,
) -> None:
    """Write data-derived Italian comments, one physical line per paragraph."""
    notes = [
        "Figure 1 -- Le variazioni assolute di capacita sono normalizzate per il massimo modulo di ciascuna riga nei dodici scenari: colori confrontabili per segno e struttura, non per dimensione fisica tra tecnologie. SMR e SMR CC sono inclusi; le percentuali con BASE piccolo restano escluse solo dal CSV relativo.",
        "Figure 2 -- Produzione H2 lato offerta: confronto tra elettrolisi, SMR e SMR CC nei sei casi selezionati, ciascuno rispetto al proprio BASE.",
        "Figure 3 -- Input solid biomass ripartito fra calore, metanolo, olio sintetico e industria; i flussi di ingresso sono contati una volta, senza duplicare le uscite multiple dei CHP.",
        "Figure 4 -- Obiettivi effettivi per famiglia, rispetto a BASE. Le curve collegano esclusivamente punti campionati; target uguali non implicano servizi finali equivalenti. "
        + ("Scala percentuale comune ai quattro pannelli." if COST_SHARE_Y else "Scale percentuali indipendenti per rendere leggibili le variazioni interne alle famiglie.")
        + (" Dati disponibili per " + str(costs.scenario.nunique()) + " scenari." if costs is not None else " Figura non disponibile: obiettivi mancanti."),
        "Figure 5 -- (a,b) Traiettorie oil/gas con pannelli completi e zoom vicino all'origine; (b) emissioni atmosferiche lorde misurate sul carrier co2, con diagonale di compensazione perfetta oil+gas=0; (c) delta delle tre sorgenti principali di cattura e curva spessa della somma di tutte le otto sorgenti co2 stored nelle famiglie M; (d) plateau del biogas nelle famiglie F. Valori relativi al rispettivo BASE; nessun fattore emissivo generico applicato.",
        "Figure 6 -- Rapporto tra variazione di offerta lorda AC e target esogeno nominale; non e una misura di efficienza. Esclude low voltage e include trasferimenti e accumuli. Consigliato come supporto in appendice."
        + ("" if MAKE_ELECTRICITY_RESPONSE_FIGURE else " Grafico disabilitato; CSV disponibile."),
    ]
    def endpoint(table: pd.DataFrame, scenario: str, technology: str) -> float:
        part = table[(table.scenario == scenario) & (table.technology == technology)]
        return float(part.sort_values("perturbation_twh").delta_twh.iloc[-1])
    for scenario in ("E_ROAD", "F_ROAD", "M_AIR", "M_IND", "H_OIL", "H_SHIP"):
        maximum = hydrogen.loc[hydrogen.scenario == scenario, "perturbation_twh"].max()
        notes.append(f"Idrogeno -- {scenario} a {maximum:g} TWh: "
                     + ", ".join(f"delta {tech}={endpoint(hydrogen, scenario, tech):+.1f} TWh/a" for tech in HYDROGEN) + ".")
    er = endpoint(hydrogen, "E_ROAD", "Electrolysis")
    fr = endpoint(hydrogen, "F_ROAD", "Electrolysis")
    fossil_er = sum(endpoint(hydrogen, "E_ROAD", t) for t in ("SMR", "SMR CC"))
    notes.append("Bilancio carbonico -- "
                 + ("La riduzione di elettrolisi e crescita di SMR/SMR CC in E_ROAD, opposte alla crescita di elettrolisi in F_ROAD, sono compatibili con il ruolo dello spazio emissivo come accoppiamento settoriale."
                    if er < 0 and fr > 0 and fossil_er > 0 else
                    "Il contrasto atteso fra E_ROAD e F_ROAD non e confermato integralmente dai valori estremi.")
                 + " Per attribuzione causale servono anche emissioni, vincoli attivi e prezzi ombra; SMR CC non implica emissioni nulle.")
    for scenario in biomass.scenario.unique():
        notes.append("Biomassa -- " + scenario + " al massimo target: "
                     + ", ".join(f"{tech} {endpoint(biomass, scenario, tech):+.1f} TWh/a" for tech in biomass.technology.unique())
                     + "; verificare anche la traiettoria intermedia nella figura 3.")
    infrastructure = capacity[capacity.technology.isin(["H2 pipeline", "H2 storage"])]
    notes.append("H2 diretto e intermedio -- " + "; ".join(
        scenario + ": " + ", ".join(f"{row.technology} {row.absolute_delta:+.2f} {row.unit}" for row in infrastructure[infrastructure.scenario == scenario].itertuples())
        for scenario in ("H_IND", "H_SHIP", "H_OIL"))
        + ". Differenze a target nominale uguale; la conversione locale in combustibili e una spiegazione da verificare spazialmente, non una conclusione dai soli aggregati.")
    valid = scatter.dropna(subset=["delta_oil_supply", "delta_fossil_gas_supply"])
    rho = valid.delta_oil_supply.corr(valid.delta_fossil_gas_supply)
    notes.append(f"Sostituzione oil/gas -- {len(valid)} realizzazioni, correlazione Pearson r={rho:+.3f}. "
                 + ("Relazione inversa evidente secondo il criterio esplorativo r <= -0.5." if rho <= -.5 else "Non emerge una forte relazione inversa secondo il criterio esplorativo r <= -0.5.")
                 + " Punti dipendenti dalla famiglia e dal target; oil indica la fornitura primaria sul carrier oil primary, a monte della raffinazione. Nessuna regressione o causalita imposta.")
    for scenario, part in biogas.groupby("scenario", sort=False):
        tail = part.sort_values("perturbation").tail(PLATEAU_MIN_POINTS)
        stable = tail.biogas_to_gas.notna().all() and np.ptp(tail.biogas_to_gas) <= PLATEAU_TOL_TWH
        notes.append(f"Biogas -- {scenario}, ultimi {len(tail)} punti ({tail.perturbation.min():g}--{tail.perturbation.max():g} TWh): "
                     f"produzione {tail.biogas_to_gas.min():.2f}--{tail.biogas_to_gas.max():.2f} TWh/a, "
                     f"delta {tail.delta_biogas_to_gas.min():+.2f}--{tail.delta_biogas_to_gas.max():+.2f} TWh/a; "
                     + ("plateau compatibile" if stable else "plateau non evidente")
                     + f" con tolleranza {PLATEAU_TOL_TWH:g} TWh/a. Evidenza limitata alla gamma campionata.")
    if costs is not None:
        for scenario, part in costs.groupby("scenario", sort=False):
            row = part.loc[part.relative_delta_percent.idxmin()]
            notes.append(f"Costi -- {scenario}: minimo campionato a {row.perturbation_twh:g} TWh, delta obiettivo {row.relative_delta_percent:+.3f} percento; nessun minimo continuo stimato.")
    notes.append("Risposta elettrica -- " + "; ".join(
        f"{r.scenario}: R={r.response_ratio:+.3f}" for r in ratio.itertuples())
        + ". Valori positivi indicano espansione AC, negativi riduzione.")
    notes.extend([
        "Figura 2 -- Confronto sistematico dei percorsi H2: variazioni di electrolysis, SMR e SMR-CC rispetto a BASE in tutte le famiglie; i marcatori vuoti indicano il target nominale.",
        "Figura 5 -- Relazione inversa tra fornitura di petrolio primario (oil primary) e gas fossile (gas); saturazione della produzione di biogas negli scenari di persistenza fossile, entro la gamma di perturbazioni campionata.",
        "Figura 6 -- Compensazione tra oil-related e gas-related CO2, non necessariamente esatta; le barre mostrano la diversa origine della captured CO2 negli scenari methanol M_AIR, M_IND e M_SHIP, come variazioni della fornitura al carrier co2 stored rispetto a BASE. La linea nera include tutte le fonti catturate.",
    ])
    notes.extend(scatter.attrs.get("carbon_notes", []))
    content = "\n".join("% " + note.replace("\n", " ") for note in notes) + "\n"
    for filename in ("cross_family_figures_notes.tex", "cross_family_results_ideas.tex"):
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")


def write_diagnostics() -> None:
    (OUTPUT_DIR / "analysis_diagnostics.txt").write_text(
        "\n".join(DIAGNOSTICS) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_RC)
    try:
        deterministic = load_excel_data(DETERMINISTIC_ENERGY_FILE)
        capacity = load_excel_data(DETERMINISTIC_CAPACITY_FILE, capacity=True)
        sensitivity, levels = {}, {}
        for scenario in SCENARIOS:
            sensitivity[scenario] = load_excel_data(SENSITIVITY_FILES[scenario])
            levels[scenario] = detect_sensitivity_levels(
                sensitivity[scenario]["levels_supply"].columns, scenario
            )
            report(
                f"  {scenario}: sensitivity levels={list(levels[scenario].values())} TWh; BASE_CO2 excluded"
            )
            for side in ("supply", "consumption"):
                base = sensitivity[scenario][f"levels_{side}"]["BASE"]
                reference = deterministic[f"levels_{side}"]["BASE"]
                difference = (base - reference).abs().max()
                report(
                    f"BASE comparison {scenario}/{side}: maximum row discrepancy={difference:.6g} scaled units; each sensitivity uses its own BASE"
                )
        for data in (deterministic, capacity):
            for sheet, frame in data.items():
                missing = set(SCENARIOS) - set(frame.columns)
                if missing:
                    raise ValueError(f"Missing nominal scenarios in {sheet}: {missing}")
        costs, cost_error = None, None
        try:
            costs = load_objectives(levels)
        except (FileNotFoundError, KeyError, ValueError, ImportError, OSError) as error:
            cost_error = ValueError(
                f"Figure 4 unavailable: {error}. Provide COST_FILE CSV/XLSX with scenario, perturbation_twh, objective, unit=EUR/a and a BASE row, or configure saved objective networks."
            )
            report(str(cost_error), True)
        # All source files and detected schemas are printed before plotting.
        cap = make_capacity_heatmap(capacity)
        make_energy_balance_heatmap(deterministic)
        h2 = make_hydrogen_pathway_figure(sensitivity, levels)
        bio = make_biomass_reallocation_figure(sensitivity, levels)
        if costs is not None:
            make_cost_figure(costs)
        scatter, biogas = make_carbon_substitution_figure(sensitivity, levels)
        make_gas_wind_generation_figure(sensitivity, levels)
        ratios = make_electricity_response_ratio(deterministic)
        evidence, flags = additional_evidence(sensitivity, levels)
        write_latex_ideas(cap, h2, bio, ratios, evidence, flags, costs, scatter, biogas)
        report(f"Outputs written to {OUTPUT_DIR}")
        if cost_error is not None and not SKIP_MISSING_COST_FIGURE:
            raise cost_error
    finally:
        write_diagnostics()


if __name__ == "__main__":
    main()
