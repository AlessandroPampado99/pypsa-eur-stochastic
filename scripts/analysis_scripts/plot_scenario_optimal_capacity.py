#!/usr/bin/env python3
"""Plot optimal capacities by bus-carrier sector for multiple scenarios.

The input is the ``levels_by_component_carrier`` sheet written by
``analysis_network_powers.py``.  One stacked bar chart is produced for every
sector/metric pair so that power, apparent-power, and energy capacities are
never mixed.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import rename_techs  # noqa: E402


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

EXCEL_PATH = Path(
    "results/demand_uncertainty_/analysis_output/csvs/analysis_networks_power.xlsx"
)
OUTPUT_DIR = Path(
    "results/demand_uncertainty_/analysis_output/graphs/scenario_optimal_capacity"
)
PLOTTING_YAML = Path("config/plotting.default.yaml")
SHEET_NAME = "levels_by_sector_carrier"
VALUE_PREFIX = "value__"

EXCLUDED_SCENARIOS: set[str] = set()
INCLUDED_SCENARIOS: set[str] | None = None
INCLUDED_GROUPS: set[str] | None = None
EXCLUDED_GROUPS: set[str] = set()
EXCLUDED_COMPONENTS: set[str] = {"Load"}
INCLUDED_COMPONENTS: set[str] | None = None
EXCLUDED_CARRIERS: set[str] = {"<none>"}

SCENARIO_ORDER: list[str] | None = None
FAMILY_GROUPS: dict[str, list[str]] | None = None
FAMILY_ORDER: list[str] | None = None
INFER_FAMILY_FROM_NAME = True
FAMILY_GAP = 0.7

CAPACITY_THRESHOLD = 0.0  # in the plotted unit
ANNOTATE_TOTALS = True
TOTAL_DECIMALS = 0
ANNOTATE_SEGMENTS = True
SEGMENT_LABEL_MIN_PERCENT = 5.0
SEGMENT_LABEL_DECIMALS = 0

BAR_WIDTH = 0.78
FIGSIZE: tuple[float, float] | None = None
SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
DPI = 300
OUT_STEM = "scenario_optimal_capacity"


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
        "nuclear",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
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
    ]
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def _family_mapping(scenarios: Sequence[str]) -> dict[str, str]:
    explicit = {
        str(scenario): str(family)
        for family, members in (FAMILY_GROUPS or {}).items()
        for scenario in members
    }
    return {
        scenario: explicit.get(
            scenario,
            (
                scenario.strip("_").split("_", 1)[0]
                if INFER_FAMILY_FROM_NAME and "_" in scenario.strip("_")
                else scenario.strip("_") or scenario
            ),
        )
        for scenario in scenarios
    }


def order_and_group_scenarios(
    scenarios: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    discovered = list(dict.fromkeys(map(str, scenarios)))
    scenario_rank = {
        name: i for i, name in enumerate(SCENARIO_ORDER or discovered)
    }
    discovered.sort(key=lambda name: (scenario_rank.get(name, len(scenario_rank)), name))
    family_by_scenario = _family_mapping(discovered)
    encountered = list(dict.fromkeys(family_by_scenario[name] for name in discovered))
    family_order = list(FAMILY_ORDER or [])
    family_order.extend(family for family in encountered if family not in family_order)
    family_rank = {family: i for i, family in enumerate(family_order)}
    discovered.sort(
        key=lambda name: (family_rank[family_by_scenario[name]], scenario_rank[name])
    )
    return discovered, family_by_scenario


def _scale_and_unit(metric: str) -> tuple[float, str]:
    """Convert PyPSA's MW/MWh component values to plotting units."""
    if "energy" in metric.lower():
        return 1e3, "GWh"
    if "apparent_power" in metric.lower():
        return 1e3, "GVA"
    return 1e3, "GW"


def read_capacity_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analysis workbook not found: {path}")
    frame = pd.read_excel(path, sheet_name=SHEET_NAME)
    required = {"group", "component", "carrier", "metric"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sheet '{SHEET_NAME}' is missing {sorted(missing)}.")
    value_columns = [
        column for column in frame.columns if str(column).startswith(VALUE_PREFIX)
    ]
    if not value_columns:
        raise ValueError(f"No '{VALUE_PREFIX}<scenario>' columns found.")

    long = frame.melt(
        id_vars=["group", "component", "carrier", "metric"],
        value_vars=value_columns,
        var_name="scenario_column",
        value_name="value",
    )
    long["scenario"] = long["scenario_column"].astype(str).str[len(VALUE_PREFIX) :]
    long["carrier"] = long["carrier"].astype(str).map(rename_techs)
    long["group"] = long["group"].astype(str)
    long["component"] = long["component"].astype(str)
    long["metric"] = long["metric"].astype(str)
    long = long[~long["group"].isin(EXCLUDED_GROUPS or set())]
    if INCLUDED_GROUPS is not None:
        long = long[long["group"].isin(INCLUDED_GROUPS)]
    long["value"] = (
        pd.to_numeric(long["value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    long = long[~long["scenario"].isin(EXCLUDED_SCENARIOS or set())]
    if INCLUDED_SCENARIOS is not None:
        long = long[long["scenario"].isin(INCLUDED_SCENARIOS)]
    long = long[~long["component"].isin(EXCLUDED_COMPONENTS or set())]
    if INCLUDED_COMPONENTS is not None:
        long = long[long["component"].isin(INCLUDED_COMPONENTS)]
    long = long[~long["carrier"].isin(EXCLUDED_CARRIERS or set())]
    if long.empty:
        raise ValueError("No capacity records remain after applying filters.")
    return (
        long.groupby(
            ["group", "component", "metric", "scenario", "carrier"],
            as_index=False,
            sort=False,
        )["value"]
        .sum()
    )


def _positions(
    scenarios: Sequence[str], family_by_scenario: Mapping[str, str]
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    positions, ranges = [], []
    previous, start, position = None, 0, 0.0
    for i, scenario in enumerate(scenarios):
        family = family_by_scenario[scenario]
        if previous is not None and family != previous:
            ranges.append((previous, start, i - 1))
            start, position = i, position + FAMILY_GAP
        positions.append(position)
        position += 1.0
        previous = family
    if scenarios:
        ranges.append((str(previous), start, len(scenarios) - 1))
    return np.asarray(positions), ranges


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def plot_capacity(
    group: str,
    metric: str,
    records: pd.DataFrame,
    plotting: Mapping[str, Any],
    output_dir: Path,
) -> bool:
    scale, unit = _scale_and_unit(metric)
    table = records.pivot_table(
        index="scenario",
        columns="carrier",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    ) / scale
    scenarios, families = order_and_group_scenarios(table.index)
    table = table.reindex(scenarios)
    table = table.loc[:, table.abs().max() >= CAPACITY_THRESHOLD]
    if table.empty or table.shape[1] == 0:
        return False

    carriers = (
        PREFERRED_ORDER.intersection(table.columns)
        .append(table.columns.difference(PREFERRED_ORDER, sort=False))
        .tolist()
    )
    table = table[carriers]
    totals = table.sum(axis=1)
    x, family_ranges = _positions(scenarios, families)
    configured = plotting.get("plotting", {}).get("tech_colors", {})
    cmap = plt.get_cmap("tab20")
    colors = {
        carrier: configured.get(carrier, cmap(i % cmap.N))
        for i, carrier in enumerate(carriers)
    }

    width = max(11.0, 0.72 * len(scenarios) + 4.0)
    fig, ax = plt.subplots(figsize=FIGSIZE or (width, 8.0))
    bottom = np.zeros(len(scenarios))
    for carrier in carriers:
        values = table[carrier].to_numpy(float)
        ax.bar(
            x, values, BAR_WIDTH, bottom=bottom, color=colors[carrier],
            linewidth=0, label=carrier,
        )
        if ANNOTATE_SEGMENTS:
            for i, value in enumerate(values):
                share = 100.0 * abs(value) / abs(totals.iloc[i]) if totals.iloc[i] else 0
                if value and share >= SEGMENT_LABEL_MIN_PERCENT:
                    ax.text(
                        x[i], bottom[i] + value / 2,
                        f"{value:.{SEGMENT_LABEL_DECIMALS}f}",
                        ha="center", va="center", fontsize=7,
                    )
        bottom += values

    span = max(bottom.max(), 1.0)
    if ANNOTATE_TOTALS:
        for xpos, top in zip(x, bottom):
            ax.text(
                xpos, top + 0.015 * span, f"{top:.{TOTAL_DECIMALS}f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
    for family_index, (family, first, last) in enumerate(family_ranges):
        ax.text(
            (x[first] + x[last]) / 2, -0.25, family,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=10, fontweight="bold",
        )
        if family_index < len(family_ranges) - 1:
            next_first = family_ranges[family_index + 1][1]
            ax.axvline(
                (x[last] + x[next_first]) / 2,
                color="0.35", linewidth=0.8, alpha=0.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel(f"Optimal capacity [{unit}]", fontweight="bold")
    ax.set_xlabel("Scenario", fontweight="bold")
    ax.set_title(f"Optimal capacity: {group} — {metric.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_ylim(0, bottom.max() + 0.12 * span)
    fig.subplots_adjust(bottom=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{OUT_STEM}_{_safe_filename(group)}_{_safe_filename(metric)}"
    for enabled, suffix in (
        (SAVE_PNG, "png"), (SAVE_SVG, "svg"), (SAVE_PDF, "pdf")
    ):
        if enabled:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=DPI if suffix == "png" else None, bbox_inches="tight")
            print(f"[WRITE] {path}")
    plt.close(fig)
    table.to_csv(output_dir / f"{stem}_by_carrier.csv")
    totals.rename("total").to_csv(output_dir / f"{stem}_totals.csv")
    return True


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXCEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    plotting = _load_yaml(PLOTTING_YAML)
    capacity = read_capacity_workbook(EXCEL_PATH)
    count = 0
    for (group, metric), records in capacity.groupby(
        ["group", "metric"], sort=True
    ):
        count += plot_capacity(
            str(group), str(metric), records, plotting, OUTPUT_DIR
        )
    if not count:
        raise ValueError("No capacity plots were produced.")
    print(f"[DONE] Wrote {count} capacity plot set(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    args = _arguments()
    EXCEL_PATH = args.input
    OUTPUT_DIR = args.output_dir
    main()
