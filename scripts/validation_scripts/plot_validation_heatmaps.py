#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Build validation heatmap tables from PyPSA-Eur solved networks.

Expected folder structure:
    results/<prefix>/<scenario>/networks/

Supported files:
- diagonal (standard capacity expansion):
    base_s_adm___2050.nc
- diagonal for stochastic expected value:
    base_s_adm___2050__exp.nc
- off-diagonal validation:
    base_s_adm___2050__cap-<cap_source>__op-<op_source>.nc

The script creates:
- one figure with 3 heatmaps:
    1) objective
    2) load curtailment
    3) renewable curtailment
- one Excel file with the underlying matrices

Rows   = scenario providing capacities
Columns = scenario providing operations

"""

from pathlib import Path
import sys
import math
import warnings

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = Path("results/cutouts_det_capexp_")
OUTPUT_DIR = Path("results/cutouts_det_capexp_nols/analysis_output/validation_heatmaps")

# Diagonal standard solved network
DIAGONAL_FILENAME = "base_s_adm___2050.nc"

# Diagonal stochastic expected-value solved network
DIAGONAL_STOCHASTIC_FILENAME = "base_s_adm___2050.nc"

# Off-diagonal validation solved network
CROSS_FILENAME_TEMPLATE = "base_s_adm___2050__cap-{cap_source}__op-{op_source}.nc"

# Output names
OUTPUT_FIGURE = OUTPUT_DIR / "validation_heatmaps.png"
OUTPUT_EXCEL = OUTPUT_DIR / "validation_heatmaps.xlsx"

# Optionally export each in-memory deterministic expected view used to evaluate
# a stochastic network. Deterministic input networks are not exported.
EXPORT_EXPECTED_NETWORKS = True
EXPECTED_NETWORKS_DIR = OUTPUT_DIR / "expected_networks"

# Reuse the existing Excel tables instead of loading all solved networks again.
# Set to False to recompute the metrics and overwrite the workbook.
REUSE_EXISTING_EXCEL = True

# Scenario selection
EXCLUDED_SCENARIOS = set()
INCLUDE_STOCHASTIC_SCENARIOS = True

# These scenarios, if included, use __exp on the diagonal
STOCHASTIC_SCENARIOS = {} # {"stochastic_network"}

# Optional manual order. If None, folder order is used alphabetically.
SCENARIO_ORDER = None
# Example:
# SCENARIO_ORDER = [
#    "base",
#    "agriculture_full_electric",
#    "agriculture_machinery_full_oil",
#    "electricity_optimistic",
#    "industry_h2",
#    "land_transport_linear_ev",
#    "shipping_full_methanol",
#    "urban_heat_full_central",
#    "stochastic_network",
#]

# Optional pretty labels for plots
#SCENARIO_LABELS = {
#    "base": "BASE",
#    "agriculture_full_electric": "AFE",
#    "agriculture_machinery_full_oil": "AMFO",
#    "electricity_optimistic": "EO",
#    "industry_h2": "IH2",
#    "land_transport_linear_ev": "LTLEV",
#    "shipping_full_methanol": "SFM",
#    "urban_heat_full_central": "UHFC",
#    "stochastic_network": "SP",
#}
SCENARIO_LABELS = None

# Metrics
TOTAL_COST_SCALE = 1e9
TOTAL_COST_UNIT = "bn. €/a"

LOAD_CURTAILMENT_SCALE = 1e6
LOAD_CURTAILMENT_UNIT = "TWh"

RES_CURTAILMENT_SCALE = 1e6
RES_CURTAILMENT_UNIT = "TWh"

FIGSIZE = (18, 6)
DPI = 240

TOTAL_COST_FMT = "{:.0f}"
LOAD_CURTAILMENT_FMT = "{:.2f}"
RES_CURTAILMENT_FMT = "{:.2f}"

# Load curtailment definition:
# total annual generation from generators with carrier in this set
LOAD_CURTAILMENT_CARRIERS = {"load"}

# Renewable curtailment definition:
# sum over generators of available energy minus actual dispatch
RENEWABLE_CARRIERS = {
    "solar",
    "solar rooftop",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "ror",
}

# If True, missing files become NaN and are left blank in the heatmap.
# If False, the script raises an error.
ALLOW_MISSING_FILES = True

INVALID_TEXT = "NaN"


# =========================
# INTERNAL HELPERS
# =========================

def _network_has_invalid_solution(n: pypsa.Network) -> bool:
    """
    Detect whether the exported network contains NaN values in typical solved outputs.

    This is used to flag invalid validation runs that would otherwise appear
    as zero total cost because PyPSA statistics can collapse to zero.
    """
    static_checks = [
        ("generators", "p_nom_opt"),
        ("links", "p_nom_opt"),
        ("stores", "e_nom_opt"),
        ("storage_units", "p_nom_opt"),
        ("lines", "s_nom_opt"),
        ("transformers", "s_nom_opt"),
    ]

    for comp_name, col in static_checks:
        df = getattr(n, comp_name, None)
        if df is None or df.empty or col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any() and vals.isna().any():
            return True

    time_series_checks = [
        ("generators_t", "p"),
        ("links_t", "p0"),
        ("links_t", "p1"),
        ("stores_t", "e"),
        ("storage_units_t", "p"),
        ("storage_units_t", "state_of_charge"),
    ]

    for ts_name, attr in time_series_checks:
        ts_obj = getattr(n, ts_name, None)
        if ts_obj is None or not hasattr(ts_obj, attr):
            continue
        df = getattr(ts_obj, attr)
        if df is None or df.empty:
            continue
        if df.notna().any().any() and df.isna().any().any():
            return True

    return False

def _scenario_display_name(name: str) -> str:
    """Return the scenario label used in the plot."""
    return SCENARIO_LABELS.get(name, name) if SCENARIO_LABELS else name


def _list_scenarios(root_dir: Path) -> list[str]:
    """List scenario folders under ROOT_DIR."""
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root_dir}")

    scenarios = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "networks").exists():
            continue
        scenarios.append(p.name)

    if EXCLUDED_SCENARIOS:
        scenarios = [s for s in scenarios if s not in EXCLUDED_SCENARIOS]

    if not INCLUDE_STOCHASTIC_SCENARIOS:
        scenarios = [s for s in scenarios if s not in STOCHASTIC_SCENARIOS]

    if SCENARIO_ORDER is not None:
        wanted = [s for s in SCENARIO_ORDER if s in scenarios]
        remaining = [s for s in scenarios if s not in wanted]
        scenarios = wanted + remaining

    if not scenarios:
        raise ValueError("No scenarios found after applying filters.")

    return scenarios


def _diagonal_network_path(root_dir: Path, scenario: str) -> Path:
    """Return the diagonal solved network path for one scenario."""
    fname = (
        DIAGONAL_STOCHASTIC_FILENAME
        if scenario in STOCHASTIC_SCENARIOS
        else DIAGONAL_FILENAME
    )
    return root_dir / scenario / "networks" / fname


def _cross_network_path(root_dir: Path, cap_source: str, op_source: str) -> Path:
    """Return the off-diagonal validation network path."""
    fname = CROSS_FILENAME_TEMPLATE.format(
        cap_source=cap_source,
        op_source=op_source,
    )
    return root_dir / cap_source / "networks" / fname


def _pair_network_path(root_dir: Path, cap_source: str, op_source: str) -> Path:
    """Return the network path for a matrix cell."""
    if cap_source == op_source:
        return _diagonal_network_path(root_dir, cap_source)
    return _cross_network_path(root_dir, cap_source, op_source)


def _get_snapshot_weightings(n: pypsa.Network) -> pd.Series:
    """Return a robust weighting series for time aggregation."""
    sw = n.snapshot_weightings

    if isinstance(sw, pd.DataFrame):
        for col in ["objective", "generators", "stores"]:
            if col in sw.columns:
                return sw[col]
        return sw.iloc[:, 0]

    if isinstance(sw, pd.Series):
        return sw

    raise TypeError(f"Unsupported snapshot_weightings type: {type(sw)}")


def _network_is_stochastic(n: pypsa.Network) -> bool:
    """Return whether the network contains PyPSA stochastic scenarios."""
    if hasattr(n, "has_scenarios"):
        return bool(n.has_scenarios)

    # Compatibility fallback for PyPSA versions without has_scenarios.
    try:
        columns = n.loads_t.p_set.columns
        return isinstance(columns, pd.MultiIndex) and columns.nlevels >= 2
    except (AttributeError, TypeError):
        return False


def _build_expected_network(n: pypsa.Network) -> pypsa.Network:
    """Build an in-memory deterministic expected view of a stochastic network."""
    from scripts.export_stochastic_views import build_view

    scenario_weightings = n.scenario_weightings
    if "weight" not in scenario_weightings.columns:
        raise ValueError("Stochastic network has no 'weight' scenario weighting column.")

    probabilities = scenario_weightings["weight"].astype(float).to_dict()
    return build_view(n, mode="expected", probs=probabilities, scenario=None)


def _compute_total_cost(
    n: pypsa.Network,
    expected_network_export_path: Path | None = None,
) -> float:
    """
    Compute the total objective value.

    Stochastic networks are first converted to an in-memory deterministic
    expected view. CAPEX + OPEX are then computed from PyPSA statistics for
    both stochastic and deterministic inputs.

    If the resulting total cost is zero (or numerically close to zero),
    treat the solution as invalid and return NaN.
    """
    if _network_is_stochastic(n):
        evaluation_network = _build_expected_network(n)
        if expected_network_export_path is not None:
            expected_network_export_path.parent.mkdir(parents=True, exist_ok=True)
            evaluation_network.export_to_netcdf(expected_network_export_path)
            print(f"[INFO] Exported expected network: {expected_network_export_path}")
    else:
        evaluation_network = n

    capex = float(evaluation_network.statistics.capex().sum())
    opex = float(evaluation_network.statistics.opex().sum())
    total = capex + opex

    if not np.isfinite(total):
        return np.nan

    if np.isclose(total, 0.0, atol=1e-12, rtol=0.0):
        return np.nan

    return total

def _compute_load_curtailment_from_generators(n: pypsa.Network) -> float:
    """
    Compute annual load curtailment as generation from generators
    with carrier in LOAD_CURTAILMENT_CARRIERS.
    """
    if n.generators.empty:
        return 0.0

    gens = n.generators.index[n.generators.carrier.isin(LOAD_CURTAILMENT_CARRIERS)]
    if len(gens) == 0:
        return 0.0

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return 0.0

    p = n.generators_t.p.reindex(columns=gens, fill_value=0.0)
    w = _get_snapshot_weightings(n).reindex(p.index)

    total = p.mul(w, axis=0).sum().sum()
    return float(total)


def _get_generator_nominal_power(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    """Return generator nominal capacity using p_nom_opt if available, else p_nom."""
    gdf = n.generators.loc[gens]

    if "p_nom_opt" in gdf.columns:
        p_nom = gdf["p_nom_opt"].fillna(gdf["p_nom"])
    else:
        p_nom = gdf["p_nom"]

    return p_nom.astype(float)


def _compute_renewable_curtailment(n: pypsa.Network) -> float:
    """
    Compute renewable curtailment from generators.

    Curtailment = max(available - dispatch, 0), aggregated over time.
    """
    if n.generators.empty:
        return 0.0

    gens = n.generators.index[n.generators.carrier.isin(RENEWABLE_CARRIERS)]
    if len(gens) == 0:
        return 0.0

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return 0.0

    if not hasattr(n.generators_t, "p_max_pu"):
        return 0.0

    p = n.generators_t.p.reindex(columns=gens, fill_value=0.0)
    p_max_pu = n.generators_t.p_max_pu.reindex(index=p.index, columns=gens, fill_value=0.0)
    p_nom = _get_generator_nominal_power(n, gens)

    available = p_max_pu.mul(p_nom, axis=1)
    curtailed = (available - p).clip(lower=0.0)

    w = _get_snapshot_weightings(n).reindex(curtailed.index)
    total = curtailed.mul(w, axis=0).sum().sum()

    return float(total)


def _extract_metrics(
    n: pypsa.Network,
    expected_network_export_path: Path | None = None,
) -> dict[str, float]:
    """Extract all metrics from one network."""
    return {
        "total_cost": _compute_total_cost(n, expected_network_export_path)
        / TOTAL_COST_SCALE,
        "load_curtailment": _compute_load_curtailment_from_generators(n) / LOAD_CURTAILMENT_SCALE,
        "renewable_curtailment": _compute_renewable_curtailment(n) / RES_CURTAILMENT_SCALE,
    }


def _safe_load_network(path: Path) -> pypsa.Network | None:
    """Load a PyPSA network if the file exists."""
    if not path.exists():
        if ALLOW_MISSING_FILES:
            warnings.warn(f"Missing network file: {path}")
            return None
        raise FileNotFoundError(f"Missing network file: {path}")

    return pypsa.Network(str(path))


def _build_metric_matrices(
    root_dir: Path,
    scenarios: list[str],
    existing_matrices: dict[str, pd.DataFrame] | None = None,
    existing_invalid_masks: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Build one matrix per metric and one boolean mask per metric for invalid
    existing networks.

    Missing files stay as NaN in the metric matrix and False in the invalid mask.
    Existing files with invalid metrics are marked True in the invalid mask.
    """
    metric_names = ["total_cost", "load_curtailment", "renewable_curtailment"]

    matrices = {}
    invalid_masks = {}
    for m in metric_names:
        if existing_matrices is None:
            matrices[m] = pd.DataFrame(index=scenarios, columns=scenarios, dtype=float)
        else:
            matrices[m] = existing_matrices[m].reindex(index=scenarios, columns=scenarios)

        if existing_invalid_masks is None:
            invalid_masks[m] = pd.DataFrame(
                False, index=scenarios, columns=scenarios, dtype=bool
            )
        else:
            invalid_masks[m] = existing_invalid_masks[m].reindex(
                index=scenarios, columns=scenarios, fill_value=False
            ).astype(bool)

    for cap_source in scenarios:
        for op_source in scenarios:
            already_computed = all(
                pd.notna(matrices[m].loc[cap_source, op_source])
                or invalid_masks[m].loc[cap_source, op_source]
                for m in metric_names
            )
            if already_computed:
                print(f"[INFO] Reusing {cap_source} vs {op_source} from Excel")
                continue

            path = _pair_network_path(root_dir, cap_source, op_source)
            print(f"[INFO] Loading {cap_source} vs {op_source}: {path}")

            n = _safe_load_network(path)
            if n is None:
                for m in metric_names:
                    matrices[m].loc[cap_source, op_source] = np.nan
                    invalid_masks[m].loc[cap_source, op_source] = False
                continue

            expected_network_export_path = None
            if EXPORT_EXPECTED_NETWORKS:
                expected_network_export_path = EXPECTED_NETWORKS_DIR / (
                    f"expected__cap-{cap_source}__op-{op_source}.nc"
                )

            vals = _extract_metrics(n, expected_network_export_path)

            for m in metric_names:
                val = vals[m]
                matrices[m].loc[cap_source, op_source] = val
                invalid_masks[m].loc[cap_source, op_source] = pd.isna(val)

    return matrices, invalid_masks


def _format_annotation(
    val: float,
    fmt: str,
    is_invalid: bool = False,
    invalid_text: str = "NaN",
) -> str:
    """Format cell annotation distinguishing invalid existing files from missing files."""
    if pd.isna(val):
        return invalid_text if is_invalid else ""
    return fmt.format(val)


def _plot_single_heatmap(
    ax,
    df: pd.DataFrame,
    invalid_mask: pd.DataFrame,
    title: str,
    unit: str,
    cmap: str,
    fmt: str,
    invalid_text: str = "NaN",
):
    """Plot one annotated heatmap with matplotlib only."""
    data = df.to_numpy(dtype=float)

    masked = np.ma.masked_invalid(data)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")

    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj)

    ax.set_title(
        f"{title} ({unit})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))

    ax.set_xticklabels(
        [_scenario_display_name(c) for c in df.columns],
        rotation=45,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_yticklabels(
        [_scenario_display_name(i) for i in df.index],
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("Operations on", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Capacities from", fontsize=11, fontweight="bold", labelpad=10)

    # Minor grid to mimic a table
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove external spines for a cleaner table-like look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell annotations
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size == 0:
        threshold = 0.0
    else:
        threshold = 0.5 * (np.nanmin(finite_vals) + np.nanmax(finite_vals))

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = data[i, j]
            is_invalid = bool(invalid_mask.iloc[i, j])
            txt = _format_annotation(
                val,
                fmt,
                is_invalid=is_invalid,
                invalid_text=invalid_text,
            )
            if txt == "":
                continue

            if pd.isna(val):
                color = "black"
            else:
                color = "white" if val >= threshold else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(unit, rotation=90, fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)


def _write_excel(
    matrices: dict[str, pd.DataFrame],
    invalid_masks: dict[str, pd.DataFrame],
    output_excel: Path,
) -> None:
    """Write metric matrices and invalid-cell metadata to Excel."""
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for name, df in matrices.items():
            df.to_excel(writer, sheet_name=name[:31])
            invalid_masks[name].to_excel(writer, sheet_name=f"invalid_{name}"[:31])


def _read_excel(
    output_excel: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read previously computed metric matrices from Excel."""
    metric_names = ["total_cost", "load_curtailment", "renewable_curtailment"]
    sheets = pd.read_excel(output_excel, sheet_name=None, index_col=0)

    missing_sheets = [name for name in metric_names if name not in sheets]
    if missing_sheets:
        raise ValueError(
            f"{output_excel} is missing required sheets: "
            + ", ".join(missing_sheets)
        )

    matrices = {}
    invalid_masks = {}
    for name in metric_names:
        df = sheets[name]
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        matrices[name] = df.apply(pd.to_numeric, errors="coerce")

        mask_name = f"invalid_{name}"[:31]
        if mask_name in sheets:
            mask = sheets[mask_name]
            mask.index = mask.index.map(str)
            mask.columns = mask.columns.map(str)
            invalid_masks[name] = mask.reindex(
                index=matrices[name].index,
                columns=matrices[name].columns,
                fill_value=False,
            ).fillna(False).astype(bool)
        else:
            # Older workbooks do not distinguish a missing file from an
            # invalid solution. Empty cells are therefore retried once.
            invalid_masks[name] = pd.DataFrame(
                False,
                index=matrices[name].index,
                columns=matrices[name].columns,
                dtype=bool,
            )

    return matrices, invalid_masks


def _metric_plot_settings(metric_name: str) -> dict:
    """Return plot settings for each metric."""
    settings = {
        "total_cost": {
            "title": "Total cost",
            "unit": TOTAL_COST_UNIT,
            "cmap": "Reds",
            "fmt": TOTAL_COST_FMT,
            "filename": "validation_heatmap_total_cost.png",
        },
        "load_curtailment": {
            "title": "Load curtailment",
            "unit": LOAD_CURTAILMENT_UNIT,
            "cmap": "Blues",
            "fmt": LOAD_CURTAILMENT_FMT,
            "filename": "validation_heatmap_load_curtailment.png",
        },
        "renewable_curtailment": {
            "title": "Renewable curtailment",
            "unit": RES_CURTAILMENT_UNIT,
            "cmap": "Purples",
            "fmt": RES_CURTAILMENT_FMT,
            "filename": "validation_heatmap_renewable_curtailment.png",
        },
    }
    return settings[metric_name]


def _plot_metric_heatmap(
    metric_name: str,
    df: pd.DataFrame,
    invalid_mask: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create one figure per metric."""
    cfg = _metric_plot_settings(metric_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg["filename"]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    _plot_single_heatmap(
        ax=ax,
        df=df,
        invalid_mask=invalid_mask,
        title=cfg["title"],
        unit=cfg["unit"],
        cmap=cfg["cmap"],
        fmt=cfg["fmt"],
        invalid_text=INVALID_TEXT,
    )

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    scenarios = _list_scenarios(ROOT_DIR)

    print("[INFO] Scenarios used:")
    for s in scenarios:
        print(f"  - {s}")

    if REUSE_EXISTING_EXCEL and OUTPUT_EXCEL.exists():
        print(f"[INFO] Updating existing Excel file: {OUTPUT_EXCEL}")
        existing_matrices, existing_invalid_masks = _read_excel(OUTPUT_EXCEL)
        matrices, invalid_masks = _build_metric_matrices(
            ROOT_DIR, scenarios, existing_matrices, existing_invalid_masks
        )
        excel_status = f"Excel updated at: {OUTPUT_EXCEL}"
    else:
        matrices, invalid_masks = _build_metric_matrices(ROOT_DIR, scenarios)
        excel_status = f"Excel written to: {OUTPUT_EXCEL}"

    _write_excel(matrices, invalid_masks, OUTPUT_EXCEL)

    for metric_name, df in matrices.items():
        _plot_metric_heatmap(
            metric_name,
            df,
            invalid_masks[metric_name],
            OUTPUT_DIR,
        )

    print(f"✔ {excel_status}")
    print(f"✔ Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ ERROR: {exc}", file=sys.stderr)
        raise
