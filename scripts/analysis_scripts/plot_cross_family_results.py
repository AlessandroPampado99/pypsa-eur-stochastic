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
from matplotlib.colors import TwoSlopeNorm
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
OUTPUT_DIR = ROOT / "results/demand_uncertainty_2035/analysis_output/cross_family"
# Optional CSV/XLSX with scenario, perturbation_twh, objective, unit columns.
# Include scenario=BASE, perturbation_twh=0; units must be EUR/a.
COST_FILE: Path | None = None
COST_SHEET = "objectives"
# Used when COST_FILE is None. Read only the saved solver objective attribute.
OBJECTIVE_NETWORK_ROOT: Path | None = SENSITIVITY_ROOT
OBJECTIVE_NETWORK_NAME = "base_s_adm___2035.nc"
OBJECTIVE_ATTRIBUTE = "network__objective"
SKIP_MISSING_COST_FIGURE = True
COST_SCENARIOS = ("E_ROAD", "F_ROAD", "H_OIL")
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
            OUTPUT_DIR / f"01_capacity_response_{suffix}.csv"
        )
    result.to_csv(OUTPUT_DIR / "01_capacity_response_levels.csv", index=False)
    matrix = result.pivot(
        index="technology", columns="scenario", values="relative_delta"
    ).reindex(index=[r[0] for r in CAPACITY_ROWS], columns=SCENARIOS)
    matrix = matrix.dropna(how="all")
    if matrix.empty:
        raise ValueError("No capacities have a usable BASE for the heatmap")
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.4), layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit)
    )
    ax.set_xticks(range(12), SCENARIOS, rotation=55, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    separators(ax)
    fig.colorbar(im, ax=ax, label="Capacity change relative to BASE [%]", shrink=0.8)
    save_figure(
        fig,
        "01_capacity_response_heatmap",
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
) -> None:
    fig, axes = plt.subplots(
        rows,
        len(scenarios) // rows,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.5 if rows == 2 else 2.8),
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
        ax.set_title(scenario)
        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.grid(axis="y", alpha=0.18)
        ax.ticklabel_format(style="plain", axis="both")
        ax.set_xlabel("Perturbation [TWh]")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="outside lower center", ncol=len(labels), frameon=False
    )
    save_figure(fig, stem, ylabel)


def make_hydrogen_pathway_figure(sensitivity: dict, levels: dict) -> pd.DataFrame:
    selected = ("E_ROAD", "F_ROAD", "M_AIR", "M_IND", "H_OIL", "H_SHIP")
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
    table.to_csv(OUTPUT_DIR / "02_hydrogen_pathway_sensitivity.csv", index=False)
    line_panels(
        table,
        selected,
        list(HYDROGEN),
        "02_hydrogen_pathway_sensitivity",
        "Δ H2 production [TWh/a]",
        2,
    )
    return table


def make_biomass_reallocation_figure(sensitivity: dict, levels: dict) -> pd.DataFrame:
    selected = ("F_ROAD", "H_OIL", "H_SHIP")
    blocks = {
        k: v
        for k, v in BIOMASS_BLOCKS.items()
        if INCLUDE_INDUSTRY_BIOMASS or k != "Industry"
    }
    flat = [tech for group in blocks.values() for tech in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Biomass blocks overlap")
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
        "Δ biomass input [TWh/a]",
        1,
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
            with xr.open_dataset(path) as network:
                objective = float(network.attrs[OBJECTIVE_ATTRIBUTE])
                constant = float(
                    network.attrs.get("network__objective_constant", np.nan)
                )
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
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all():
            raise ValueError(f"Nonfinite cost data: {column}")
    base_rows = table[(table.scenario == "BASE") & (table.perturbation_twh == 0)]
    if len(base_rows) != 1 or base_rows.objective.iloc[0] <= 0:
        raise ValueError(
            "Cost data need one positive BASE objective at perturbation_twh=0"
        )
    base = float(base_rows.objective.iloc[0])
    rows = []
    for scenario in COST_SCENARIOS:
        part = table[table.scenario == scenario].copy()
        expected = set(levels[scenario].values())
        if not expected.issubset(set(part.perturbation_twh)):
            raise ValueError(
                f"Missing objective levels for {scenario}: {expected - set(part.perturbation_twh)}"
            )
        part = part[part.perturbation_twh.isin(expected)]
        part = pd.concat([base_rows.assign(scenario=scenario), part], ignore_index=True)
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    result["base_objective"] = base
    result["absolute_delta_eur"] = result.objective - base
    result["relative_delta_percent"] = 100 * result.absolute_delta_eur / base
    return result


def make_cost_figure(table: pd.DataFrame) -> None:
    table.to_csv(OUTPUT_DIR / "04_transition_cost_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.8), layout="constrained")
    for i, scenario in enumerate(COST_SCENARIOS):
        part = table[table.scenario == scenario].sort_values("perturbation_twh")
        ax.plot(
            part.perturbation_twh,
            part.relative_delta_percent,
            color=FAMILY_COLORS[scenario[0]],
            marker="o",
            markersize=3,
            linestyle=LINE_STYLES[i],
            label=scenario,
        )
        minimum = part.loc[part.relative_delta_percent.idxmin()]
        if (
            part.perturbation_twh.min()
            < minimum.perturbation_twh
            < part.perturbation_twh.max()
        ):
            ax.scatter(
                [minimum.perturbation_twh],
                [minimum.relative_delta_percent],
                s=60,
                facecolors="none",
                edgecolors="black",
                zorder=5,
            )
            report(
                f"Sampled interior cost minimum: {scenario}, {minimum.perturbation_twh:g} TWh; {minimum.relative_delta_percent:.3f}% (no fitted optimum)"
            )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.grid(axis="y", alpha=0.18)
    ax.set(xlabel="Perturbation [TWh]", ylabel="Objective change relative to BASE [%]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    save_figure(
        fig, "04_transition_cost_sensitivity", "Transition objective sensitivity"
    )


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
    table.to_csv(OUTPUT_DIR / "05_electricity_response_ratio.csv", index=False)
    ac.to_csv(OUTPUT_DIR / "05_ac_supply_components_twh.csv")
    report(
        "Figure 5 uses gross positive AC supply, including AC/DC transfers and storage supply; it is not net generation. Low voltage is excluded. The existing balance plot merges AC and low voltage, so its total is different."
    )
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
        fig, "05_electricity_response_ratio", "Electricity-system response ratio"
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
    capacity: pd.DataFrame,
    hydrogen: pd.DataFrame,
    biomass: pd.DataFrame,
    ratio: pd.DataFrame,
    evidence: pd.DataFrame,
    flags: pd.DataFrame,
    costs: pd.DataFrame | None,
) -> None:
    """Write Italian comments with one physical line per conceptual paragraph."""
    paragraphs = [
        "============================================================",
        "IDEE - PATTERN TRASVERSALI",
        "============================================================",
    ]

    def endpoint(table: pd.DataFrame, scenario: str, label: str, column: str) -> float:
        part = table[
            (table.scenario == scenario) & (table.technology == label)
        ].sort_values("perturbation_twh")
        return float(part[column].iloc[-1]) if not part.empty else np.nan

    def ev(scenario: str, label: str) -> pd.DataFrame:
        return evidence[
            (evidence.scenario == scenario) & (evidence.technology == label)
        ].sort_values("perturbation_twh")

    texts = []
    for s in ("E_ROAD", "F_ROAD", "M_AIR", "M_IND", "H_OIL", "H_SHIP"):
        texts.append(
            f"{s}: "
            + ", ".join(
                f"delta {t}={endpoint(hydrogen, s, t, 'delta_twh'):+.1f}"
                for t in HYDROGEN
            )
        )
    paragraphs.append(
        "1. Bilancio del carbonio e accoppiamento settoriale: agli estremi superiori delle rispettive sensibilita, variazioni di H2 in TWh/a: "
        + "; ".join(texts)
        + ". Questi confronti descrivono sostituzioni osservate; il ruolo causale dello spazio emissivo richiede anche vincolo CO2, emissioni e prezzi ombra. Non si deduce il rilascio del vincolo dalle sole produzioni; SMR CC non equivale a idrogeno privo di emissioni."
    )
    texts = []
    for s in ("H_IND", "H_SHIP", "H_OIL"):
        part = capacity[
            (capacity.scenario == s)
            & capacity.technology.isin(["H2 pipeline", "H2 storage"])
        ]
        texts.append(
            s
            + ": "
            + ", ".join(
                f"delta {r.technology}={r.absolute_delta:+.2f} {r.unit}"
                for r in part.itertuples()
            )
        )
    paragraphs.append(
        "2. Idrogeno diretto e intermedio: "
        + "; ".join(texts)
        + ". Confrontare queste variazioni a target nominale uguale (100 TWh). La conversione locale in combustibili e una possibile spiegazione, non dimostrata da aggregati di rete; condizionamento e distribuzione geografica non sono misurati nella figura 1."
    )
    texts = []
    for s in biomass.scenario.unique():
        texts.append(
            s
            + ": "
            + ", ".join(
                f"{t} {endpoint(biomass, s, t, 'delta_twh'):+.1f}"
                for t in biomass.technology.unique()
            )
        )
    paragraphs.append(
        "3. Biomassa come risorsa intersettoriale: variazioni all'ultimo livello, in TWh/a di input solid biomass: "
        + "; ".join(texts)
        + ". Si contano una sola volta gli ingressi a caldaie/CHP, metanolo, elettrobiocombustibili/biomass to liquid e industria; non le uscite energetiche multiple. Le traiettorie intermedie e i componenti sono nei CSV della figura 3."
    )
    texts = []
    for s in ("M_AIR", "M_IND", "M_SHIP"):
        parts = []
        for t in ("biomass-to-methanol", "methanolisation"):
            p = ev(s, f"supply/methanol/{t}")
            vals = p.level.to_numpy()
            plateau = len(vals) >= 3 and np.ptp(vals[-3:]) <= PLATEAU_TOL_TWH
            parts.append(
                f"{t}: {vals[0]:.1f}->{vals[-1]:.1f} TWh/a, plateau ultimi tre livelli={'si' if plateau else 'no'}"
            )
        co2 = ev(s, "consumption/co2 stored/methanolisation")
        parts.append(
            f"CO2 a metanolazione {co2.level.iloc[0]:.1f}->{co2.level.iloc[-1]:.1f} Mt/a"
        )
        texts.append(s + ": " + "; ".join(parts))
    paragraphs.append(
        "4. Metanolo, rigidita a valle e flessibilita a monte: "
        + " | ".join(texts)
        + ". Un plateau numerico non prova un vincolo di capacita attivo; la rigidita a valle richiede il confronto con i consumi finali. Per il contrasto fra M_AIR e M_IND usare congiuntamente le produzioni H2 della figura 2."
    )
    paragraphs.append(
        "5. Risposta elettrica e domanda finale: "
        + "; ".join(
            f"{r.scenario}: delta AC={r.delta_ac_supply_twh:+.1f} TWh/a, R={r.response_ratio:+.3f}"
            for r in ratio.itertuples()
        )
        + ". R positivo indica espansione, negativo riduzione; valori prossimi a zero indicano compensazioni aggregate. E un rapporto di risposta, non un'efficienza. AC indica offerta lorda positiva, inclusi trasferimenti e accumuli, esclusa low voltage; non coincide con il grafico esistente che fonde AC e low voltage."
    )
    plateaus, texts = [], []
    for s in ("F_HEAT", "F_IND", "F_ROAD"):
        p = ev(s, "supply/gas/biogas to gas")
        tail = p.tail(PLATEAU_MIN_POINTS)
        stable = np.ptp(tail.level) <= PLATEAU_TOL_TWH
        if stable:
            plateaus.append(float(tail.level.mean()))
        texts.append(
            f"{s}: livelli {tail.perturbation_twh.min():g}--{tail.perturbation_twh.max():g} TWh, produzione {tail.level.min():.2f}--{tail.level.max():.2f} TWh/a"
        )
    common = len(plateaus) == 3 and np.ptp(plateaus) <= PLATEAU_TOL_TWH
    paragraphs.append(
        "6. Saturazione del biogas: "
        + "; ".join(texts)
        + (
            f". Plateau comune compatibile con la tolleranza {PLATEAU_TOL_TWH:g} TWh/a, circa {np.mean(plateaus):.1f} TWh/a."
            if common
            else ". Non emerge un plateau comune secondo il criterio degli ultimi livelli entro la tolleranza configurata."
        )
        + " Criterio: ultimi punti configurati entro la tolleranza; evidenza limitata alla gamma campionata. Si tratta di produzione gas, non di input biogas."
    )
    correlations = []
    for s in SCENARIOS:
        wind = ev(s, "supply/AC/onwind").set_index("perturbation_twh").level
        solar = ev(s, "supply/AC/solar-hsat").set_index("perturbation_twh").level
        gas = pd.concat(
            [
                ev(s, f"supply/AC/{t}").set_index("perturbation_twh").level
                for t in ("urban central gas CHP", "CCGT", "OCGT")
            ],
            axis=1,
        ).sum(axis=1, min_count=3)
        valid = pd.concat(
            [(wind + solar).rename("renewables"), gas.rename("gas")], axis=1
        ).dropna()
        rho = (
            valid.renewables.corr(valid.gas)
            if len(valid) >= 3 and valid.std().min() > 1e-8
            else np.nan
        )
        correlations.append(dict(scenario=s, n=len(valid), pearson_r=rho))
    pd.DataFrame(correlations).to_csv(
        OUTPUT_DIR / "exploratory_correlations.csv", index=False
    )
    paragraphs.append(
        "7. Rinnovabili e gas dispacciabile: correlazione esplorativa fra onwind+solar-hsat e CHP gas+CCGT+OCGT, sulle sensibilita incluso BASE: "
        + "; ".join(
            f"{r['scenario']} r={r['pearson_r']:+.2f} (n={r['n']})"
            for r in correlations
        )
        + ". Campioni piccoli e dipendenti dalla stessa perturbazione: nessuna identificazione causale; solare rooftop e CHP con cattura sono esclusi da questa specifica metrica."
    )
    selected = flags[
        (flags.scenario.isin(["H_SHIP", "E_ROAD", "F_ROAD", "H_OIL"]))
        & (
            flags.quantity.str.contains(
                "shipping methanol|land transport oil|electrobiofuels"
            )
        )
    ]
    paragraphs.append(
        "8. Cambi di regime: "
        + (
            "; ".join(
                f"{r.scenario}, {r.quantity}: {r.kind} tra {r.lower_twh:g} e {r.upper_twh:g} TWh"
                for r in selected.itertuples()
            )
            if len(selected)
            else "nessuna soglia rilevata per i componenti finali selezionati"
        )
        + ". Intervalli campionati, non soglie esatte. Attivazione oltre 1 TWh/a, arresto entro 0.5 TWh/a dopo variazione maggiore di 1, o cambio forte di pendenza: candidati da verificare, non prova automatica della saturazione del settore."
    )
    recurring = []
    for tech, part in capacity.groupby("technology", sort=False):
        if part.capacity.notna().all() and part.relative_delta.notna().all():
            spread = float(part.relative_delta.abs().max())
            if spread <= 10:
                recurring.append(
                    f"{tech} (massima variazione assoluta {spread:.1f} percento)"
                )
    paragraphs.append(
        "9. Blocchi ricorrenti: tecnologie con capacita entro il 10 percento di BASE in tutti i dodici scenari: "
        + (", ".join(recurring) or "nessuna tra quelle selezionate")
        + ". La stabilita non implica resilienza a shock o indispensabilita. Reti e accumuli sono candidati abilitanti da confrontare per famiglia; SMR, elettrolisi e conversioni del metanolo sono tecnologie di conversione/sostituzione la cui risposta dipende dallo scenario. La sola dimensione non e un criterio di robustezza."
    )
    if costs is not None:
        minima = []
        for scenario in COST_SCENARIOS:
            part = costs[costs.scenario == scenario]
            row = part.loc[part.relative_delta_percent.idxmin()]
            minima.append(
                f"{scenario}: minimo campionato a {row.perturbation_twh:g} TWh, delta obiettivo {row.relative_delta_percent:+.3f} percento"
            )
        paragraphs.append(
            "10.0. Costi: "
            + "; ".join(minima)
            + ". Obiettivo salvato dal solver, senza aggiungere la costante; nessun ottimo continuo stimato."
        )
    for number, message in enumerate(
        [
            "Confrontare espansioni e contrazioni delle capacita nominali; righe con BASE piccolo escluse dai rapporti ma conservate nei CSV assoluti.",
            "Confrontare sostituzioni fra elettrolisi, SMR e SMR CC al variare della perturbazione usando produzione H2.",
            "Seguire la riallocazione della biomassa in ingresso fra funzioni settoriali senza duplicare i prodotti del CHP.",
            "Confrontare obiettivi effettivi senza interpolazione; i target rappresentano direzioni diverse, non servizi finali equivalenti, e le gamme si sovrappongono solo parzialmente."
            if costs is not None
            else "Figura costi non generata: obiettivi effettivi mancanti; consultare la diagnostica.",
            "Misurare la risposta AC aggregata per TWh di perturbazione esogena, distinta dalla sola elettrificazione finale.",
        ],
        1,
    ):
        paragraphs.append(f"10.{number}. Figura {number}: {message}")
    (OUTPUT_DIR / "cross_family_results_ideas.tex").write_text(
        "\n".join("% " + p.replace("\n", " ") for p in paragraphs) + "\n",
        encoding="utf-8",
    )


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
        h2 = make_hydrogen_pathway_figure(sensitivity, levels)
        bio = make_biomass_reallocation_figure(sensitivity, levels)
        if costs is not None:
            make_cost_figure(costs)
        ratios = make_electricity_response_ratio(deterministic)
        evidence, flags = additional_evidence(sensitivity, levels)
        write_latex_ideas(cap, h2, bio, ratios, evidence, flags, costs)
        report(f"Outputs written to {OUTPUT_DIR}")
        if cost_error is not None and not SKIP_MISSING_COST_FIGURE:
            raise cost_error
    finally:
        write_diagnostics()


if __name__ == "__main__":
    main()
