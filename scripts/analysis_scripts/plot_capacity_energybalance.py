#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot capacity and energy balance by technology across multiple PyPSA scenarios.

For each technology/carrier, the script saves one bar plot comparing all scenarios:
- left y-axis: optimal capacity [GW]
- right y-axis: energy balance [TWh]

The script also writes CSV tables with the extracted values.

Run:
    python scripts/plot_capacity_energy_by_technology.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa


ROOT = Path(__file__).resolve().parents[1]  # points to pypsa-eur root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from add_electricity import sanitize_carriers


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT_DIR = Path("results/prices_and_renewables")  # where to look for scenario folders
NETWORK_GLOB = "networks/base_s_adm___2040.nc"

BASE_NETWORK_PATH = Path("results/prices_and_renewables/base/networks/base_s_adm___2040.nc")
INCLUDE_BASE = True
BASE_LABEL = "__BASE__"

OUTPUT_DIR = Path("results/prices_and_renewables/2040/csvs/capacity_energy_by_technology")
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "plots"

CONFIG_YAML = Path("config/prices_renewables/config.yaml")
PLOTTING_YAML = Path("config/plotting.default.yaml")

EXCLUDED_SCENARIOS = {"base"}

SCENARIO_NAME_MODE = "folder"
# Options:
# - "folder"
# - "filename"
# - "folder__filename"


# Optional filtering on technologies/carriers.
# If None, all technologies found in either capacity or energy balance are plotted.
TECHNOLOGY_FILTER = None
# Example:
# TECHNOLOGY_FILTER = ["solar", "onwind", "H2 Electrolysis"]

DROP_ZERO_TECHNOLOGIES = True
ZERO_TOL = 1e-9

SAVE_PDF = False
SAVE_PNG = True
DPI = 300

FIGSIZE_BASE = (10.0, 5.5)
MAX_SCENARIO_LABEL_LENGTH = 28

CAPACITY_COLOR = "tab:red"
ENERGY_COLOR = "tab:blue"
CAPACITY_ALPHA = 0.75
ENERGY_ALPHA = 0.75

# Summary heatmaps
SAVE_HEATMAPS = True
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"

# If True, technologies are sorted by max absolute value across scenarios.
# If False, they are sorted alphabetically.
SORT_HEATMAP_TECHS_BY_VALUE = True

# Optional: limit number of technologies in heatmap.
# Set to None to include all technologies.
MAX_HEATMAP_TECHNOLOGIES = None

# Avoid unreadable huge figures. Height is computed from number of technologies.
HEATMAP_CELL_HEIGHT = 0.28
HEATMAP_MIN_HEIGHT = 6.0
HEATMAP_MAX_HEIGHT = 60.0
HEATMAP_WIDTH = 12.0


# =========================
# INTERNALS
# =========================

def _load_config(cfg_path: Path) -> dict:
    """Load YAML config."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config YAML did not parse to dict: {cfg_path}")

    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into dict a and return a new dict."""
    out = dict(a)

    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v

    return out


def _load_and_merge_configs(config_yaml: Path, plotting_yaml: Path) -> dict:
    """Load model config and plotting config, then merge them."""
    cfg = _load_config(config_yaml)
    plot = _load_config(plotting_yaml)
    return _deep_merge(cfg, plot)


def _filter_excluded_scenarios(
    candidates: list[Path],
    excluded_scenarios: set[str],
) -> list[Path]:
    """Remove candidate networks whose scenario folder is excluded."""
    if not excluded_scenarios:
        return candidates

    kept = []
    excluded = set(excluded_scenarios)

    for p in candidates:
        scen_folder = p.parent.parent.name
        if scen_folder not in excluded:
            kept.append(p)

    return kept


def _scenario_label(nc_path: Path, mode: str) -> str:
    """Build a scenario label from a .nc path."""
    scen_folder = nc_path.parent.parent.name
    fname = nc_path.stem

    if mode == "folder":
        return scen_folder
    if mode == "filename":
        return fname
    if mode == "folder__filename":
        return f"{scen_folder}__{fname}"

    raise ValueError(f"Unknown SCENARIO_NAME_MODE: {mode}")


def _clean_filename(text: str) -> str:
    """Create a filesystem-safe filename."""
    text = str(text).strip()
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def _shorten_label(label: str, max_len: int = MAX_SCENARIO_LABEL_LENGTH) -> str:
    """Shorten long scenario labels for plotting."""
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _series_to_frame_by_component_carrier(
    s: pd.Series,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert a PyPSA statistics Series to a DataFrame indexed by component/carrier.

    Expected input indexes:
    - optimal_capacity: component, carrier
    - energy_balance: component, carrier, bus_carrier

    For energy_balance, bus_carrier is aggregated away.
    """
    if s.empty:
        return pd.DataFrame(columns=["component", "carrier", value_name])

    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError(
            f"Expected a MultiIndex Series for {value_name}, got: {type(s.index)}"
        )

    index_names = list(s.index.names)

    required = {"component", "carrier"}
    missing = required - set(index_names)

    if missing:
        raise ValueError(
            f"Missing expected index levels in {value_name}: {missing}. "
            f"Got index levels: {index_names}"
        )

    df = s.rename(value_name).reset_index()

    out = (
        df.groupby(["component", "carrier"], as_index=False)[value_name]
        .sum()
    )

    out["component"] = out["component"].astype(str)
    out["carrier"] = out["carrier"].astype(str)

    return out


def _extract_optimal_capacity(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract optimal capacity from n.statistics.optimal_capacity().

    Output columns:
    - component
    - carrier
    - optimal_capacity
    """
    s = n.statistics.optimal_capacity()
    return _series_to_frame_by_component_carrier(s, "optimal_capacity")


def _extract_energy_balance(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract energy balance from n.statistics.energy_balance().

    Output columns:
    - component
    - carrier
    - energy_balance

    The original energy_balance has index:
    component, carrier, bus_carrier

    Here bus_carrier is summed away.
    """
    s = n.statistics.energy_balance()
    return _series_to_frame_by_component_carrier(s, "energy_balance")


def analyze_one_network(
    n: pypsa.Network,
    config: dict,
    scenario: str,
) -> pd.DataFrame:
    """
    Extract optimal capacity and energy balance directly from PyPSA statistics.

    Output columns:
    - scenario
    - component
    - carrier
    - technology
    - optimal_capacity
    - energy_balance
    - capacity_GW
    - energy_balance_TWh
    """
    sanitize_carriers(n, config)

    pypsa.options.params.statistics.nice_names = False
    pypsa.options.params.statistics.drop_zero = False

    cap = _extract_optimal_capacity(n)
    eb = _extract_energy_balance(n)

    out = pd.merge(
        cap,
        eb,
        on=["component", "carrier"],
        how="outer",
    ).fillna(0.0)

    out["scenario"] = scenario

    # Use component + carrier as unique plotting technology.
    # This avoids mixing, for example, Store oil with Link oil if both exist.
    out["technology"] = out["component"] + " | " + out["carrier"]

    out["capacity_GW"] = out["optimal_capacity"] / 1e3
    out["energy_balance_TWh"] = out["energy_balance"] / 1e6

    out = out[
        [
            "scenario",
            "component",
            "carrier",
            "technology",
            "optimal_capacity",
            "energy_balance",
            "capacity_GW",
            "energy_balance_TWh",
        ]
    ]

    if TECHNOLOGY_FILTER is not None:
        keep = set(map(str, TECHNOLOGY_FILTER))
        out = out[
            out["technology"].isin(keep)
            | out["carrier"].isin(keep)
        ].copy()

    return out


def _build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build a wide table with capacity and energy balance by scenario."""
    idx = ["component", "carrier", "technology"]

    cap = long_df.pivot_table(
        index=idx,
        columns="scenario",
        values="capacity_GW",
        aggfunc="sum",
        fill_value=0.0,
    )

    eb = long_df.pivot_table(
        index=idx,
        columns="scenario",
        values="energy_balance_TWh",
        aggfunc="sum",
        fill_value=0.0,
    )

    cap.columns = [f"capacity_GW__{c}" for c in cap.columns]
    eb.columns = [f"energy_balance_TWh__{c}" for c in eb.columns]

    out = pd.concat([cap, eb], axis=1).reset_index()

    cap_cols = [c for c in out.columns if c.startswith("capacity_GW__")]
    eb_cols = [c for c in out.columns if c.startswith("energy_balance_TWh__")]

    out["_max_capacity"] = out[cap_cols].abs().max(axis=1) if cap_cols else 0.0
    out["_max_energy"] = out[eb_cols].abs().max(axis=1) if eb_cols else 0.0

    out = (
        out.sort_values(
            ["_max_capacity", "_max_energy", "component", "carrier"],
            ascending=[False, False, True, True],
        )
        .drop(columns=["_max_capacity", "_max_energy"])
        .reset_index(drop=True)
    )

    return out


def _ordered_scenarios(long_df: pd.DataFrame) -> list[str]:
    """Return scenarios preserving their first appearance in the long table."""
    return list(dict.fromkeys(long_df["scenario"].tolist()))


def _plot_one_technology(
    tech_df: pd.DataFrame,
    technology: str,
    scenarios: list[str],
    output_dir: Path,
):
    """Save one capacity/energy-balance bar plot for one technology."""
    data = (
        tech_df.set_index("scenario")
        .reindex(scenarios)
        .fillna(0.0)
        .reset_index()
    )

    x = np.arange(len(scenarios))
    width = 0.38

    capacity = data["capacity_GW"].to_numpy()
    energy = data["energy_balance_TWh"].to_numpy()

    scenario_labels = [_shorten_label(s) for s in scenarios]

    fig_width = max(FIGSIZE_BASE[0], 0.55 * len(scenarios) + 4.0)
    fig, ax1 = plt.subplots(figsize=(fig_width, FIGSIZE_BASE[1]))

    ax2 = ax1.twinx()

    bars_capacity = ax1.bar(
        x - width / 2,
        capacity,
        width=width,
        color=CAPACITY_COLOR,
        alpha=CAPACITY_ALPHA,
        label="Optimal capacity [GW]",
    )

    bars_energy = ax2.bar(
        x + width / 2,
        energy,
        width=width,
        color=ENERGY_COLOR,
        alpha=ENERGY_ALPHA,
        label=f"Energy balance [TWh]",
    )

    ax1.set_title(str(technology))
    ax1.set_ylabel("Optimal capacity [GW]")
    ax2.set_ylabel("Energy balance [TWh]")

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels, rotation=45, ha="right")

    ax1.grid(axis="y", alpha=0.25)

    ax1.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax2.axhline(0.0, linewidth=0.8, alpha=0.5)

    handles = [bars_capacity, bars_energy]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left")

    ax1.set_xlim(-0.75, len(scenarios) - 0.25)

    fig.tight_layout()

    safe_name = _clean_filename(technology)

    if SAVE_PNG:
        fig.savefig(output_dir / f"{safe_name}.png", dpi=DPI, bbox_inches="tight")

    if SAVE_PDF:
        fig.savefig(output_dir / f"{safe_name}.pdf", bbox_inches="tight")

    plt.close(fig)


def make_plots(long_df: pd.DataFrame, output_dir: Path):
    """Create one plot per technology."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _ordered_scenarios(long_df)

    technologies = sorted(long_df["technology"].dropna().astype(str).unique())

    for tech in technologies:
        g = long_df[long_df["technology"] == tech].copy()

        max_capacity = g["capacity_GW"].abs().max()
        max_energy = g["energy_balance_TWh"].abs().max()

        if DROP_ZERO_TECHNOLOGIES and max(max_capacity, max_energy) <= ZERO_TOL:
            continue

        _plot_one_technology(
            tech_df=g,
            technology=tech,
            scenarios=scenarios,
            output_dir=output_dir,
        )

def _matrix_for_heatmap(
    long_df: pd.DataFrame,
    value_col: str,
    scenarios: list[str],
) -> pd.DataFrame:
    """
    Build a technology x scenario matrix for heatmap plotting.
    """
    mat = long_df.pivot_table(
        index="technology",
        columns="scenario",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )

    mat = mat.reindex(columns=scenarios, fill_value=0.0)

    if SORT_HEATMAP_TECHS_BY_VALUE:
        order = mat.abs().max(axis=1).sort_values(ascending=False).index
        mat = mat.loc[order]
    else:
        mat = mat.sort_index()

    if MAX_HEATMAP_TECHNOLOGIES is not None:
        mat = mat.head(MAX_HEATMAP_TECHNOLOGIES)

    return mat


def _plot_heatmap_matrix(
    mat: pd.DataFrame,
    title: str,
    cbar_label: str,
    output_path: Path,
    *,
    symmetric: bool = False,
):
    """
    Save one heatmap from a technology x scenario matrix.

    No explicit colors are set, so matplotlib default colormap is used.
    """
    if mat.empty:
        print(f"[WARNING] Empty heatmap skipped: {title}")
        return

    n_techs = len(mat.index)
    height = min(
        HEATMAP_MAX_HEIGHT,
        max(HEATMAP_MIN_HEIGHT, HEATMAP_CELL_HEIGHT * n_techs),
    )

    fig, ax = plt.subplots(figsize=(HEATMAP_WIDTH, height))

    values = mat.to_numpy(dtype=float)

    if symmetric:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
    else:
        vmin = None
        vmax = None

    im = ax.imshow(
        values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Technology")

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(
        [_shorten_label(c) for c in mat.columns],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_summary_heatmaps(long_df: pd.DataFrame, output_dir: Path):
    """
    Save two summary heatmaps:
    - optimal capacity [GW]
    - energy balance [TWh]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _ordered_scenarios(long_df)

    cap_mat = _matrix_for_heatmap(
        long_df=long_df,
        value_col="capacity_GW",
        scenarios=scenarios,
    )

    eb_mat = _matrix_for_heatmap(
        long_df=long_df,
        value_col="energy_balance_TWh",
        scenarios=scenarios,
    )

    cap_mat.to_csv(output_dir / "heatmap_capacity_GW_matrix.csv")
    eb_mat.to_csv(output_dir / "heatmap_energy_balance_TWh_matrix.csv")

    _plot_heatmap_matrix(
        mat=cap_mat,
        title="Optimal capacity by technology and scenario",
        cbar_label="Optimal capacity [GW]",
        output_path=output_dir / "heatmap_capacity_GW.png",
        symmetric=False,
    )

    _plot_heatmap_matrix(
        mat=eb_mat,
        title="Energy balance by technology and scenario",
        cbar_label="Energy balance [TWh]",
        output_path=output_dir / "heatmap_energy_balance_TWh.png",
        symmetric=True,
    )

def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    if INCLUDE_BASE and not BASE_NETWORK_PATH.exists():
        raise FileNotFoundError(f"BASE_NETWORK_PATH not found: {BASE_NETWORK_PATH}")

    cfg = _load_and_merge_configs(CONFIG_YAML, PLOTTING_YAML)

    candidates = sorted(ROOT_DIR.glob(f"*/{NETWORK_GLOB}"))
    candidates = _filter_excluded_scenarios(candidates, EXCLUDED_SCENARIOS)

    if not candidates:
        raise FileNotFoundError(
            f"No networks found under {ROOT_DIR} with pattern */{NETWORK_GLOB} "
            f"after excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}"
        )

    if EXCLUDED_SCENARIOS:
        print(f"[INFO] Excluding scenarios: {sorted(EXCLUDED_SCENARIOS)}")

    all_records = []

    if INCLUDE_BASE:
        print(f"[BASE] Loading: {BASE_NETWORK_PATH}")
        n_base = pypsa.Network(str(BASE_NETWORK_PATH))
        base_df = analyze_one_network(n_base, cfg, BASE_LABEL)
        all_records.append(base_df)

    for p in candidates:
        label = _scenario_label(p, SCENARIO_NAME_MODE)
        print(f"[SCENARIO={label}] Loading: {p}")

        n = pypsa.Network(str(p))
        df = analyze_one_network(n, cfg, label)
        all_records.append(df)

    long_df = pd.concat(all_records, ignore_index=True)

    if long_df.empty:
        raise ValueError("No capacity or energy-balance records were produced.")

    if DROP_ZERO_TECHNOLOGIES:
        nonzero_techs = (
            long_df.groupby("technology")[["capacity_GW", "energy_balance_TWh"]]
            .agg(lambda x: float(np.nanmax(np.abs(x))))
            .max(axis=1)
        )
        nonzero_techs = nonzero_techs[nonzero_techs > ZERO_TOL].index
        long_df = long_df[long_df["technology"].isin(nonzero_techs)].copy()

    wide_df = _build_wide_table(long_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_path = OUTPUT_DIR / "capacity_energy_by_technology_long.csv"
    wide_path = OUTPUT_DIR / "capacity_energy_by_technology_wide.csv"

    long_df.sort_values(["technology", "scenario"]).to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print(f"[INFO] Wrote: {long_path}")
    print(f"[INFO] Wrote: {wide_path}")

    make_plots(long_df, OUTPUT_PLOTS_DIR)

    if SAVE_HEATMAPS:
        make_summary_heatmaps(long_df, HEATMAPS_DIR)
        print(f"✔ Wrote heatmaps to: {HEATMAPS_DIR}")

    print(f"✔ Wrote plots to: {OUTPUT_PLOTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        raise