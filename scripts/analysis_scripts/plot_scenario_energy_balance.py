#!/usr/bin/env python3
"""Plot scenario energy balances by bus-carrier group from an analysis workbook."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import rename_techs  # noqa: E402

EXCEL_PATH = Path(
    "results/demand_uncertainty_/analysis_output/csvs/analysis_networks_energy.xlsx"
)
OUTPUT_DIR = Path(
    "results/demand_uncertainty_/analysis_output/graphs/scenario_energy_balance"
)
OUT_STEM = "scenario_energy_balance"
PLOTTING_YAML = Path("config/plotting.default.yaml")

SUPPLY_SHEET = "levels_supply"
CONSUMPTION_SHEET = "levels_consumption"
VALUE_PREFIX = "value__"
VALUE_SCALE = 1e6

EXCLUDED_SCENARIOS: set[str] = set()
INCLUDED_SCENARIOS: set[str] | None = None
INCLUDED_GROUPS: set[str] | None = None
EXCLUDED_GROUPS: set[str] = set()

SCENARIO_ORDER: list[str] | None = None
FAMILY_GROUPS: dict[str, list[str]] | None = None
FAMILY_ORDER: list[str] | None = None
INFER_FAMILY_FROM_NAME = True
FAMILY_GAP = 0.7
SHOW_FAMILY_SEPARATORS = True
SHOW_FAMILY_LABELS = True

ENERGY_THRESHOLD_TWH = 0.0
ANNOTATE_TOTALS = True
TOTAL_MODE = "positive"
TOTAL_DECIMALS = 0

# Labels inside technology segments. Threshold values use the plotted unit
# (TWh for energy groups and MtCO2 for CO2 groups).
ANNOTATE_SEGMENTS = True
LABELED_TECHNOLOGIES: set[str] | None = None
SEGMENT_LABEL_MODE = "percent"
SEGMENT_LABEL_MIN_VALUE = 1.0
SEGMENT_LABEL_MIN_PERCENT = 5.0
SEGMENT_LABEL_DECIMALS = 0

TITLE_PREFIX = "Energy balance"
FIGSIZE: tuple[float, float] | None = None
BAR_WIDTH = 0.78
CONSUMPTION_HATCH = "//"
SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
DPI = 300

CO2_GROUPS = {
    "co2",
    "co2 stored",
    "co2 sequestered",
    "process emissions",
    "non-sequestered HVC",
}


PREFERRED_ORDER = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def _family_mapping(scenarios: Sequence[str]) -> dict[str, str]:
    explicit: dict[str, str] = {}
    for family, members in (FAMILY_GROUPS or {}).items():
        for scenario in members:
            scenario = str(scenario)
            if scenario in explicit:
                raise ValueError(
                    f"Scenario '{scenario}' is assigned to both "
                    f"'{explicit[scenario]}' and '{family}'."
                )
            explicit[scenario] = str(family)

    mapping = {}
    for scenario in scenarios:
        if scenario in explicit:
            mapping[scenario] = explicit[scenario]
        elif INFER_FAMILY_FROM_NAME and "_" in scenario.strip("_"):
            mapping[scenario] = scenario.strip("_").split("_", 1)[0]
        else:
            mapping[scenario] = scenario.strip("_") or scenario
    return mapping


def order_and_group_scenarios(
    scenarios: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    """Order scenarios so members of each family are adjacent."""
    discovered = list(dict.fromkeys(map(str, scenarios)))
    if SCENARIO_ORDER:
        unknown = set(SCENARIO_ORDER) - set(discovered)
        if unknown:
            raise KeyError(
                f"SCENARIO_ORDER contains unknown scenarios: {sorted(unknown)}"
            )
        scenario_rank = {name: i for i, name in enumerate(SCENARIO_ORDER)}
        discovered.sort(
            key=lambda name: (scenario_rank.get(name, len(scenario_rank)), name)
        )

    family_by_scenario = _family_mapping(discovered)
    encountered = list(dict.fromkeys(family_by_scenario[name] for name in discovered))
    family_order = list(FAMILY_ORDER or [])
    family_order.extend(family for family in encountered if family not in family_order)
    family_rank = {family: i for i, family in enumerate(family_order)}
    scenario_rank = {name: i for i, name in enumerate(discovered)}
    ordered = sorted(
        discovered,
        key=lambda name: (
            family_rank[family_by_scenario[name]],
            scenario_rank[name],
        ),
    )
    return ordered, family_by_scenario


def _read_level_sheet(path: Path, sheet: str, sign: float) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    required = {"group", "technology"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing columns {sorted(missing)}.")
    value_columns = [
        column for column in frame.columns if str(column).startswith(VALUE_PREFIX)
    ]
    if not value_columns:
        raise ValueError(f"Sheet '{sheet}' has no '{VALUE_PREFIX}<scenario>' columns.")

    long = frame.melt(
        id_vars=["group", "technology"],
        value_vars=value_columns,
        var_name="scenario_column",
        value_name="value",
    )
    long["scenario"] = long["scenario_column"].str[len(VALUE_PREFIX) :]
    long["value"] = (
        pd.to_numeric(long["value"], errors="coerce").fillna(0.0) * sign / VALUE_SCALE
    )
    return long[["group", "technology", "scenario", "value"]]


def read_balance_workbook(path: Path) -> pd.DataFrame:
    """Read supply and consumption sheets into one signed long table."""
    if not path.exists():
        raise FileNotFoundError(f"Analysis workbook not found: {path}")

    supply = _read_level_sheet(path, SUPPLY_SHEET, sign=1.0)
    consumption = _read_level_sheet(path, CONSUMPTION_SHEET, sign=-1.0)
    balance = pd.concat([supply, consumption], ignore_index=True)
    balance["group"] = balance["group"].astype(str)
    balance["technology"] = balance["technology"].astype(str).map(rename_techs)
    balance["scenario"] = balance["scenario"].astype(str)

    excluded_scenarios = set(EXCLUDED_SCENARIOS or set())
    balance = balance[~balance["scenario"].isin(excluded_scenarios)]
    if INCLUDED_SCENARIOS is not None:
        balance = balance[balance["scenario"].isin(set(INCLUDED_SCENARIOS))]

    balance = balance[~balance["group"].isin(set(EXCLUDED_GROUPS or set()))]
    if INCLUDED_GROUPS is not None:
        balance = balance[balance["group"].isin(set(INCLUDED_GROUPS))]

    balance = balance.groupby(
        ["group", "scenario", "technology"], as_index=False, sort=False
    )["value"].sum()
    if balance.empty:
        raise ValueError("No workbook records remain after scenario/group filters.")
    return balance


def build_group_tables(balance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot each bus-carrier group to scenario-by-technology format."""
    tables = {}
    for group, records in balance.groupby("group", sort=True):
        table = records.pivot_table(
            index="scenario",
            columns="technology",
            values="value",
            aggfunc="sum",
            fill_value=0.0,
        )
        if not table.empty:
            tables[str(group)] = table
    if not tables:
        raise ValueError("No bus-carrier energy-balance groups were found.")
    return tables


def _technology_order(columns: pd.Index) -> list[str]:
    return (
        PREFERRED_ORDER.intersection(columns)
        .append(columns.difference(PREFERRED_ORDER, sort=False))
        .tolist()
    )


def _positions(
    scenarios: Sequence[str], family_by_scenario: Mapping[str, str]
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    x = []
    groups: list[tuple[str, int, int]] = []
    previous_family = None
    position = 0.0
    group_start = 0
    for i, scenario in enumerate(scenarios):
        family = family_by_scenario[scenario]
        if previous_family is not None and family != previous_family:
            groups.append((previous_family, group_start, i - 1))
            group_start = i
            position += FAMILY_GAP
        x.append(position)
        position += 1.0
        previous_family = family
    if scenarios:
        groups.append((str(previous_family), group_start, len(scenarios) - 1))
    return np.asarray(x), groups


def _colors(technologies: Sequence[str], plotting: Mapping[str, Any]) -> dict[str, Any]:
    configured = plotting.get("plotting", {}).get("tech_colors", {})
    cmap = plt.get_cmap("tab20")
    return {
        technology: configured.get(technology, cmap(i % cmap.N))
        for i, technology in enumerate(technologies)
    }


def calculate_totals(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate supply, consumption, net, and half-absolute throughput."""
    positive = table.clip(lower=0).sum(axis=1)
    consumption = -table.clip(upper=0).sum(axis=1)
    return pd.DataFrame(
        {
            "positive": positive,
            "consumption": consumption,
            "net": table.sum(axis=1),
            "throughput": 0.5 * table.abs().sum(axis=1),
        }
    )


def _annotation_values(totals: pd.DataFrame) -> pd.Series:
    modes = {"positive", "consumption", "net", "throughput"}
    if TOTAL_MODE not in modes:
        raise ValueError(
            f"TOTAL_MODE must be one of {sorted(modes)}, got {TOTAL_MODE!r}."
        )
    return totals[TOTAL_MODE]


def _label_segment(technology: str, value: float, side_total: float) -> bool:
    """Return whether a technology segment should show its signed value."""
    if not ANNOTATE_SEGMENTS or value == 0.0:
        return False
    if LABELED_TECHNOLOGIES is not None and technology not in LABELED_TECHNOLOGIES:
        return False

    absolute_match = abs(value) >= SEGMENT_LABEL_MIN_VALUE
    share = 100.0 * abs(value) / side_total if side_total > 0.0 else 0.0
    percent_match = share >= SEGMENT_LABEL_MIN_PERCENT
    modes = {
        "absolute": absolute_match,
        "percent": percent_match,
        "either": absolute_match or percent_match,
        "both": absolute_match and percent_match,
    }
    if SEGMENT_LABEL_MODE not in modes:
        raise ValueError(
            "SEGMENT_LABEL_MODE must be one of "
            f"{sorted(modes)}, got {SEGMENT_LABEL_MODE!r}."
        )
    return modes[SEGMENT_LABEL_MODE]


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "unnamed"


def _unit_for_group(group: str) -> str:
    return "MtCO2/a" if group in CO2_GROUPS else "TWh/a"


def _visible_table(table: pd.DataFrame) -> pd.DataFrame:
    if ENERGY_THRESHOLD_TWH <= 0:
        return table
    keep = table.abs().max(axis=0) >= ENERGY_THRESHOLD_TWH
    return table.loc[:, keep]


def plot_group(
    group: str,
    table: pd.DataFrame,
    plotting: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Write one scenario comparison plot and its source CSVs."""
    scenarios, family_by_scenario = order_and_group_scenarios(table.index)
    full_table = table.reindex(scenarios)
    totals = calculate_totals(full_table)
    table = _visible_table(full_table)
    if table.shape[1] == 0:
        print(
            f"[SKIP] {group}: no technologies exceed "
            f"{ENERGY_THRESHOLD_TWH:g} {_unit_for_group(group)}"
        )
        return pd.DataFrame()

    technologies = _technology_order(table.columns)
    table = table.loc[:, technologies]
    x, family_ranges = _positions(scenarios, family_by_scenario)
    colors = _colors(technologies, plotting)
    width = max(11.0, 0.72 * len(scenarios) + 4.0)
    fig, ax = plt.subplots(figsize=FIGSIZE or (width, 8.0))

    positive_bottom = np.zeros(len(scenarios))
    negative_bottom = np.zeros(len(scenarios))
    for technology in technologies:
        values = table[technology].to_numpy(dtype=float)
        positive = np.where(values > 0.0, values, 0.0)
        negative = np.where(values < 0.0, values, 0.0)
        ax.bar(
            x,
            positive,
            BAR_WIDTH,
            bottom=positive_bottom,
            color=colors[technology],
            linewidth=0,
        )
        ax.bar(
            x,
            negative,
            BAR_WIDTH,
            bottom=negative_bottom,
            color=colors[technology],
            edgecolor="black",
            linewidth=0.3,
            hatch=CONSUMPTION_HATCH,
        )
        for i, value in enumerate(values):
            side_total = (
                totals.iloc[i]["positive"]
                if value > 0.0
                else totals.iloc[i]["consumption"]
            )
            if _label_segment(technology, value, side_total):
                bottom = positive_bottom[i] if value > 0.0 else negative_bottom[i]
                ax.text(
                    x[i],
                    bottom + value / 2.0,
                    f"{value:.{SEGMENT_LABEL_DECIMALS}f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )
        positive_bottom += positive
        negative_bottom += negative

    annotation_values = _annotation_values(totals)
    span = max(positive_bottom.max() - negative_bottom.min(), 1.0)
    if ANNOTATE_TOTALS:
        for xpos, top, value in zip(x, positive_bottom, annotation_values):
            ax.text(
                xpos,
                top + 0.015 * span,
                f"{value:.{TOTAL_DECIMALS}f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                fontweight="bold",
            )

    for family_index, (family, first, last) in enumerate(family_ranges):
        if SHOW_FAMILY_LABELS:
            ax.text(
                (x[first] + x[last]) / 2.0,
                -0.25,
                family,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )
        if SHOW_FAMILY_SEPARATORS and family_index < len(family_ranges) - 1:
            next_first = family_ranges[family_index + 1][1]
            ax.axvline(
                (x[last] + x[next_first]) / 2.0,
                color="0.35",
                linewidth=0.8,
                alpha=0.8,
            )

    unit = _unit_for_group(group)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel(f"Energy balance [{unit}]", fontweight="bold")
    ax.set_xlabel("Scenario", fontweight="bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_title(f"{TITLE_PREFIX}: {group}")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    supply_technologies = [
        technology for technology in technologies if (table[technology] > 0.0).any()
    ]
    consumption_technologies = [
        technology for technology in technologies if (table[technology] < 0.0).any()
    ]
    heading = Patch(facecolor="none", edgecolor="none")
    legend_handles = [heading]
    legend_labels = ["SUPPLY"]
    legend_handles.extend(
        Patch(facecolor=colors[technology], edgecolor="none")
        for technology in supply_technologies
    )
    legend_labels.extend(supply_technologies)
    legend_handles.append(heading)
    legend_labels.append("CONSUMPTION")
    legend_handles.extend(
        Patch(
            facecolor=colors[technology],
            edgecolor="black",
            linewidth=0.3,
            hatch=CONSUMPTION_HATCH,
        )
        for technology in consumption_technologies
    )
    legend_labels.extend(consumption_technologies)
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    for text in legend.get_texts():
        if text.get_text() in {"SUPPLY", "CONSUMPTION"}:
            text.set_fontweight("bold")
    ax.set_ylim(
        negative_bottom.min() - 0.04 * span,
        positive_bottom.max() + 0.12 * span,
    )
    fig.subplots_adjust(bottom=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{OUT_STEM}_{_safe_filename(group)}"
    for enabled, suffix in (
        (SAVE_PNG, "png"),
        (SAVE_SVG, "svg"),
        (SAVE_PDF, "pdf"),
    ):
        if enabled:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(
                path,
                dpi=DPI if suffix == "png" else None,
                bbox_inches="tight",
            )
            print(f"[WRITE] {path}")
    plt.close(fig)

    full_table.to_csv(output_dir / f"{stem}_by_technology.csv")
    totals = totals.assign(group=group, unit=unit)
    totals.to_csv(output_dir / f"{stem}_totals.csv")
    return totals.reset_index(names="scenario")


def main() -> None:
    plotting = _load_yaml(Path(PLOTTING_YAML))
    balance = read_balance_workbook(Path(EXCEL_PATH))
    tables = build_group_tables(balance)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for group, table in tables.items():
        totals = plot_group(group, table, plotting, output_dir)
        if not totals.empty:
            summary.append(totals)

    if not summary:
        raise ValueError("No carrier plots were produced.")
    pd.concat(summary, ignore_index=True).to_csv(
        output_dir / f"{OUT_STEM}_all_group_totals.csv", index=False
    )
    print(f"[DONE] Wrote {len(summary)} carrier plot set(s) to {output_dir}")


if __name__ == "__main__":
    main()
