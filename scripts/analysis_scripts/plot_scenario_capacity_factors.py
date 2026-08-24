#!/usr/bin/env python3
"""Plot capacity-weighted mean availability for selected technologies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import rename_techs  # noqa: E402


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT_DIR = Path("results/cutouts_det_capexp_")
NETWORK_GLOB = "networks/base_s_adm___2050.nc"
BASE_NETWORK_PATH: Path | None = None
INCLUDE_BASE = False
BASE_LABEL = "__BASE__"
EXCLUDED_SCENARIOS: set[str] = set()
SCENARIO_NAME_MODE = "folder"

# Raw network carrier names. Set to None to include every discovered carrier.
TECHNOLOGIES: list[str] | None = [
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "solar",
    "solar rooftop",
    "solar-hsat",
    "ror",
]
COMPONENTS = ["Generator", "Link"]
DROP_ZERO_CAPACITY = True
ZERO_TOL = 1e-9

OUTPUT_DIR = Path(
    "results/cutouts_det_capexp_/analysis_output/graphs/scenario_capacity_factors"
)
OUTPUT_CSV = Path(
    "results/cutouts_det_capexp_/analysis_output/csvs/scenario_capacity_factors.csv"
)
SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
DPI = 300
FIGSIZE: tuple[float, float] | None = None
BAR_COLOR = "tab:blue"
ANNOTATE_VALUES = True
VALUE_DECIMALS = 1


def _scenario_label(path: Path) -> str:
    if SCENARIO_NAME_MODE == "folder":
        return path.parent.parent.name
    if SCENARIO_NAME_MODE == "filename":
        return path.stem
    if SCENARIO_NAME_MODE == "folder__filename":
        return f"{path.parent.parent.name}__{path.stem}"
    raise ValueError(f"Unknown SCENARIO_NAME_MODE: {SCENARIO_NAME_MODE}")


def _final_capacity(frame: pd.DataFrame, opt_col: str, base_col: str) -> pd.Series:
    base = pd.to_numeric(frame.get(base_col, 0.0), errors="coerce")
    if opt_col not in frame:
        return base.fillna(0.0)
    optimal = pd.to_numeric(frame[opt_col], errors="coerce")
    return optimal.where(np.isfinite(optimal), base).fillna(0.0)


def _component_records(
    network: pypsa.Network, component: str, scenario: str
) -> pd.DataFrame:
    """Return capacity-weighted p_max_pu means for one component."""
    if component == "Generator":
        frame = network.generators
        opt_col, base_col = "p_nom_opt", "p_nom"
    elif component == "Link":
        frame = network.links
        opt_col, base_col = "p_nom_opt", "p_nom"
    else:
        raise ValueError(f"Unsupported component: {component}")
    if frame.empty:
        return pd.DataFrame()

    capacity = _final_capacity(frame, opt_col, base_col)
    availability = network.get_switchable_as_dense(component, "p_max_pu").mean()
    availability = availability.reindex(frame.index).fillna(0.0)
    raw = pd.DataFrame(
        {
            "carrier": frame.carrier.fillna("").astype(str),
            "capacity": capacity,
            "weighted_availability": availability * capacity,
        }
    )
    if TECHNOLOGIES is not None:
        raw = raw[raw["carrier"].isin(TECHNOLOGIES)]
    if DROP_ZERO_CAPACITY:
        raw = raw[raw["capacity"].abs() > ZERO_TOL]
    if raw.empty:
        return pd.DataFrame()

    grouped = raw.groupby("carrier", as_index=False).sum(numeric_only=True)
    grouped["capacity_factor"] = (
        grouped["weighted_availability"] / grouped["capacity"]
    )
    grouped["scenario"] = scenario
    grouped["component"] = component
    return grouped[
        ["scenario", "component", "carrier", "capacity", "capacity_factor"]
    ]


def analyze_network(network: pypsa.Network, scenario: str) -> pd.DataFrame:
    records = [
        _component_records(network, component, scenario)
        for component in COMPONENTS
    ]
    records = [record for record in records if not record.empty]
    if not records:
        return pd.DataFrame(
            columns=["scenario", "component", "carrier", "capacity", "capacity_factor"]
        )
    return pd.concat(records, ignore_index=True)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _plot_technology(technology: str, records: pd.DataFrame) -> None:
    records = records.copy()
    records["weighted"] = records["capacity"] * records["capacity_factor"]
    records = records.groupby("scenario", as_index=False, sort=False).agg(
        capacity=("capacity", "sum"), weighted=("weighted", "sum")
    )
    records["capacity_factor"] = records["weighted"] / records["capacity"]
    records["label"] = records["scenario"]
    width = max(10.0, 0.55 * len(records) + 3.0)
    fig, ax = plt.subplots(figsize=FIGSIZE or (width, 6.0))
    x = np.arange(len(records))
    values = 100.0 * records["capacity_factor"].to_numpy(float)
    ax.bar(x, values, color=BAR_COLOR, width=0.75)
    if ANNOTATE_VALUES:
        for xpos, value in zip(x, values):
            ax.text(
                xpos,
                value + 0.01 * max(values.max(), 1.0),
                f"{value:.{VALUE_DECIMALS}f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(records["label"], rotation=45, ha="right")
    ax.set_ylabel("Capacity-weighted mean availability [%]")
    ax.set_xlabel("Scenario")
    ax.set_title(f"Average capacity factor: {rename_techs(technology)}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(values.max() * 1.15, 1.0))
    fig.subplots_adjust(bottom=0.28)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"scenario_capacity_factor_{_safe_filename(technology)}"
    for enabled, suffix in (
        (SAVE_PNG, "png"),
        (SAVE_SVG, "svg"),
        (SAVE_PDF, "pdf"),
    ):
        if enabled:
            path = OUTPUT_DIR / f"{stem}.{suffix}"
            fig.savefig(
                path, dpi=DPI if suffix == "png" else None, bbox_inches="tight"
            )
            print(f"[WRITE] {path}")
    plt.close(fig)


def main() -> None:
    candidates = sorted(Path(ROOT_DIR).glob(f"*/{NETWORK_GLOB}"))
    candidates = [
        path
        for path in candidates
        if path.parent.parent.name not in (EXCLUDED_SCENARIOS or set())
    ]
    network_inputs: list[tuple[str, Path]] = []
    if INCLUDE_BASE and BASE_NETWORK_PATH:
        network_inputs.append((BASE_LABEL, Path(BASE_NETWORK_PATH)))
    network_inputs.extend((_scenario_label(path), path) for path in candidates)
    if not network_inputs:
        raise FileNotFoundError("No scenario networks were found.")

    all_records = []
    for scenario, path in network_inputs:
        print(f"[SCENARIO={scenario}] Loading: {path}")
        all_records.append(analyze_network(pypsa.Network(path), scenario))
    results = pd.concat(all_records, ignore_index=True)
    if results.empty:
        raise ValueError("No selected technology capacity factors were found.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"[WRITE] {OUTPUT_CSV}")
    for technology, records in results.groupby("carrier", sort=True):
        _plot_technology(str(technology), records)
    print(f"[DONE] Wrote {results['carrier'].nunique()} technology plot set(s).")


if __name__ == "__main__":
    main()
