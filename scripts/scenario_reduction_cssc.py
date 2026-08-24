#!/usr/bin/env python3
"""
Reduce solved scenarios with Cost-Space Scenario Clustering (CSSC).

This script only reads an already-created validation heatmap workbook.  It does
not load or solve PyPSA networks.  Rows of the cost matrix are capacity/design
scenarios and columns are operation/realization scenarios.

python scripts/scenario_reduction_cssc.py \
    --results-prefix results/cutouts_det_capexp_ \
    --network-name base_s_adm___2050.nc \
    --k 3 \
    --solver gurobi
"""



from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import linopy
import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger("cssc")
DEFAULT_WORKBOOK = Path("analysis_output/validation_heatmaps/validation_heatmaps.xlsx")


def natural_sort_key(value: str) -> list[object]:
    """Return a deterministic, case-insensitive natural-sort key."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def discover_scenarios(results_dir: Path, deterministic_filename: str) -> list[str]:
    """Discover canonical scenarios from immediate deterministic result folders."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    scenarios = [
        child.name
        for child in results_dir.iterdir()
        if child.is_dir() and (child / "networks" / deterministic_filename).is_file()
    ]
    scenarios.sort(key=natural_sort_key)
    if not scenarios:
        raise ValueError(
            f"No scenarios contain networks/{deterministic_filename} below {results_dir}"
        )
    return scenarios


def load_opportunity_cost_matrix(
    workbook: Path,
    scenarios: Sequence[str],
    sheet_name: str = "total_cost",
    workbook_cost_scale: float = 1e9,
) -> pd.DataFrame:
    """Load, orient, filter, and restore raw objective units from a workbook."""
    if not workbook.is_file():
        raise FileNotFoundError(f"Validation workbook does not exist: {workbook}")
    if not np.isfinite(workbook_cost_scale) or workbook_cost_scale <= 0:
        raise ValueError("workbook_cost_scale must be finite and greater than zero")

    raw = pd.read_excel(workbook, sheet_name=sheet_name, index_col=0)
    raw.index = raw.index.map(str)
    raw.columns = raw.columns.map(str)
    if not raw.index.is_unique or not raw.columns.is_unique:
        raise ValueError(f"Sheet {sheet_name!r} contains duplicate scenario labels")

    canonical = set(scenarios)
    extra_rows = sorted(set(raw.index) - canonical, key=natural_sort_key)
    extra_columns = sorted(set(raw.columns) - canonical, key=natural_sort_key)
    if extra_rows:
        LOGGER.warning(
            "Ignoring non-canonical workbook rows: %s", ", ".join(extra_rows)
        )
    if extra_columns:
        LOGGER.warning(
            "Ignoring non-canonical workbook columns: %s", ", ".join(extra_columns)
        )

    missing_rows = [scenario for scenario in scenarios if scenario not in raw.index]
    missing_columns = [
        scenario for scenario in scenarios if scenario not in raw.columns
    ]
    if missing_rows or missing_columns:
        details = []
        if missing_rows:
            details.append("capacity rows: " + ", ".join(missing_rows))
        if missing_columns:
            details.append("operation columns: " + ", ".join(missing_columns))
        raise ValueError(
            "Workbook is missing canonical scenarios (" + "; ".join(details) + ")"
        )

    matrix = raw.loc[list(scenarios), list(scenarios)].apply(
        pd.to_numeric, errors="coerce"
    )
    matrix = matrix.astype(float) * float(workbook_cost_scale)
    matrix.index.name = "capacities_from"
    matrix.columns.name = "operations_on"
    validate_opportunity_cost_matrix(matrix, scenarios)
    return matrix


def validate_opportunity_cost_matrix(
    matrix: pd.DataFrame, scenarios: Sequence[str]
) -> None:
    """Require a square, consistently ordered, entirely finite CSSC matrix."""
    expected = list(scenarios)
    if matrix.shape != (len(expected), len(expected)):
        raise ValueError(
            f"Expected matrix shape {(len(expected),) * 2}, got {matrix.shape}"
        )
    if list(matrix.index) != expected or list(matrix.columns) != expected:
        raise ValueError(
            "Matrix rows and columns do not use the canonical scenario ordering"
        )
    values = matrix.to_numpy(dtype=float)
    missing = np.argwhere(~np.isfinite(values))
    if missing.size:
        examples = [
            f"capacities from {expected[i]}, operations on {expected[j]}"
            for i, j in missing[:10]
        ]
        suffix = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
        raise ValueError(
            "Missing or non-finite validation result: " + "; ".join(examples) + suffix
        )


@dataclass(frozen=True)
class CSSCResult:
    """Validated result of one CSSC solve."""

    assignments: pd.DataFrame
    representatives: pd.DataFrame
    objective: float
    runtime: float
    solver_status: str
    termination_condition: str


def _solver_options(
    solver: str, time_limit: float | None, mip_gap: float | None
) -> dict:
    """Translate solver-independent CLI options to common solver option names."""
    options: dict[str, float] = {}
    solver = solver.lower()
    if time_limit is not None:
        options["TimeLimit" if solver == "gurobi" else "time_limit"] = time_limit
    if mip_gap is not None:
        if solver == "gurobi":
            options["MIPGap"] = mip_gap
        elif solver == "highs":
            options["mip_rel_gap"] = mip_gap
        else:
            options["mip_gap"] = mip_gap
    return options


def solve_cssc(
    matrix: pd.DataFrame,
    k: int,
    solver: str = "gurobi",
    time_limit: float | None = None,
    mip_gap: float | None = None,
    tolerance: float = 1e-6,
) -> CSSCResult:
    """Solve the exact equiprobable CSSC partitioning MILP with linopy."""
    scenarios = list(matrix.index)
    n = len(scenarios)
    if not 1 <= k <= n:
        raise ValueError(f"K must be between 1 and N={n}, got {k}")
    validate_opportunity_cost_matrix(matrix, scenarios)

    scenario_coord = pd.Index(scenarios, name="scenario")
    representative_coord = pd.Index(scenarios, name="representative")
    model = linopy.Model()
    x = model.add_variables(
        coords=[scenario_coord, representative_coord], binary=True, name="assignment"
    )
    u = model.add_variables(coords=[representative_coord], binary=True, name="selected")
    t = model.add_variables(
        lower=0.0, coords=[representative_coord], name="absolute_deviation"
    )

    # delta[i, j] = V[j, i] - V[j, j]: representative j evaluated on scenario i.
    values = matrix.to_numpy(dtype=float)
    delta = xr.DataArray(
        values.T - np.diag(values)[None, :],
        coords={"scenario": scenario_coord, "representative": representative_coord},
        dims=("scenario", "representative"),
    )
    cluster_deviation = (x * delta).sum("scenario")
    model.add_constraints(t >= cluster_deviation, name="absolute_deviation_positive")
    model.add_constraints(t >= -cluster_deviation, name="absolute_deviation_negative")
    model.add_constraints(x <= u, name="assign_only_to_selected")
    diagonal = x.sel(scenario=x.coords["representative"])
    model.add_constraints(diagonal == u, name="representative_self_assignment")
    model.add_constraints(x.sum("representative") == 1, name="assign_once")
    model.add_constraints(u.sum() == k, name="select_k")
    model.add_objective(t.sum() / n)

    started = time.perf_counter()
    status, termination = model.solve(
        solver_name=solver, **_solver_options(solver, time_limit, mip_gap)
    )
    runtime = time.perf_counter() - started
    if status not in {"ok", "warning"} or model.solution is None:
        raise RuntimeError(
            f"CSSC solve failed: status={status}, termination={termination}"
        )

    x_value = x.solution.to_pandas().reindex(index=scenarios, columns=scenarios)
    u_value = u.solution.to_pandas().reindex(scenarios)
    selected = [s for s in scenarios if u_value.loc[s] > 0.5]
    assigned_to: dict[str, str] = {}
    for scenario in scenarios:
        choices = [rep for rep in scenarios if x_value.loc[scenario, rep] > 0.5]
        if len(choices) != 1:
            raise RuntimeError(
                f"Scenario {scenario} has {len(choices)} rounded assignments"
            )
        assigned_to[scenario] = choices[0]

    cluster_sizes = (
        pd.Series(assigned_to).value_counts().reindex(selected, fill_value=0)
    )
    probabilities = cluster_sizes.astype(float) / n
    assignments = pd.DataFrame(
        {
            "scenario": scenarios,
            "representative": [assigned_to[s] for s in scenarios],
            "is_representative": [s in selected for s in scenarios],
            "original_probability": np.full(n, 1.0 / n),
            "reduced_probability": [
                probabilities.loc[s] if s in selected else np.nan for s in scenarios
            ],
        }
    )
    representatives = pd.DataFrame(
        {
            "representative": selected,
            "probability": probabilities.to_numpy(dtype=float),
            "cluster_size": cluster_sizes.to_numpy(dtype=int),
        }
    )
    solver_objective = float(model.objective.value)
    recomputed = validate_solution(
        matrix, k, assignments, representatives, solver_objective, tolerance
    )
    return CSSCResult(
        assignments=assignments,
        representatives=representatives,
        objective=recomputed,
        runtime=runtime,
        solver_status=status,
        termination_condition=termination,
    )


def validate_solution(
    matrix: pd.DataFrame,
    k: int,
    assignments: pd.DataFrame,
    representatives: pd.DataFrame,
    solver_objective: float,
    tolerance: float = 1e-6,
) -> float:
    """Validate integrality-derived invariants and independently recompute CSSC."""
    scenarios = list(matrix.index)
    n = len(scenarios)
    selected = set(representatives["representative"])
    mapping = assignments.set_index("scenario")["representative"].to_dict()
    if len(selected) != k:
        raise RuntimeError(f"Expected {k} representatives, found {len(selected)}")
    if len(mapping) != n or set(mapping) != set(scenarios):
        raise RuntimeError("Not every original scenario is assigned exactly once")
    if any(rep not in selected for rep in mapping.values()):
        raise RuntimeError(
            "At least one assignment points to an unselected representative"
        )
    if any(mapping[rep] != rep for rep in selected):
        raise RuntimeError("At least one representative is not assigned to itself")
    if int(representatives["cluster_size"].sum()) != n:
        raise RuntimeError("Cluster sizes do not sum to N")
    if not np.isclose(
        representatives["probability"].sum(), 1.0, atol=tolerance, rtol=0
    ):
        raise RuntimeError("Reduced probabilities do not sum to one")

    total = 0.0
    for rep in selected:
        members = [
            scenario
            for scenario, assigned_rep in mapping.items()
            if assigned_rep == rep
        ]
        total += abs(
            sum(
                matrix.loc[rep, scenario] - matrix.loc[rep, rep] for scenario in members
            )
        )
    recomputed = total / n
    objective_atol = tolerance * max(1.0, abs(recomputed), abs(solver_objective))
    if not np.isclose(
        recomputed, solver_objective, atol=objective_atol, rtol=tolerance
    ):
        raise RuntimeError(
            f"Recomputed objective {recomputed} disagrees with solver objective {solver_objective}"
        )

    if k == n:
        if selected != set(scenarios) or not np.isclose(
            recomputed, 0.0, atol=objective_atol
        ):
            raise RuntimeError("K=N sanity test failed")
    if k == 1:
        direct = {
            rep: abs(
                sum(
                    matrix.loc[rep, scenario] - matrix.loc[rep, rep]
                    for scenario in scenarios
                )
            )
            / n
            for rep in scenarios
        }
        minimum = min(direct.values())
        chosen = next(iter(selected))
        if not np.isclose(direct[chosen], minimum, atol=objective_atol, rtol=tolerance):
            raise RuntimeError(
                f"K=1 sanity test failed: selected {chosen}, direct cost {direct[chosen]}, minimum {minimum}"
            )
    return recomputed


def save_result(
    result: CSSCResult, output_dir: Path, n: int, k: int, solver: str
) -> None:
    """Write assignments, representatives, and a machine-readable summary."""
    result.assignments.to_csv(output_dir / f"cssc_K{k}_assignments.csv", index=False)
    result.representatives.to_csv(
        output_dir / f"cssc_K{k}_representatives.csv", index=False
    )
    summary = {
        "N": n,
        "K": k,
        "solver": solver,
        "solver_status": result.solver_status,
        "termination_condition": result.termination_condition,
        "objective": result.objective,
        "runtime": result.runtime,
        "representatives": result.representatives["representative"].tolist(),
    }
    (output_dir / f"cssc_K{k}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-prefix",
        type=Path,
        required=True,
        help="Path below which scenario directories live",
    )
    parser.add_argument(
        "--network-name",
        default="base_s_adm___2050.nc",
        help="Deterministic network filename used only for scenario discovery",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        help="Validation workbook (default: RESULTS_PREFIX/analysis_output/validation_heatmaps/validation_heatmaps.xlsx)",
    )
    parser.add_argument(
        "--sheet-name",
        default="total_cost",
        help="Workbook sheet containing the cost matrix",
    )
    parser.add_argument(
        "--workbook-cost-scale",
        type=float,
        default=1e9,
        help="Multiply workbook values by this factor to restore raw objective units (default: 1e9)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        required=True,
        help="One or more representative counts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: RESULTS_PREFIX/analysis_output/cssc)",
    )
    parser.add_argument(
        "--solver",
        default="gurobi",
        help="MILP solver passed to linopy (default: gurobi)",
    )
    parser.add_argument("--time-limit", type=float, help="Solver time limit in seconds")
    parser.add_argument("--mip-gap", type=float, help="Relative MIP optimality gap")
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Post-solve numerical tolerance"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    return parser.parse_args()


def main() -> None:
    """Run workbook import once and solve CSSC for every requested K."""
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    results_dir = args.results_prefix.resolve()
    workbook = (args.workbook or results_dir / DEFAULT_WORKBOOK).resolve()
    output_dir = (args.output_dir or results_dir / "analysis_output" / "cssc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(results_dir, args.network_name)
    scenario_to_index = {scenario: i for i, scenario in enumerate(scenarios)}
    matrix = load_opportunity_cost_matrix(
        workbook, scenarios, args.sheet_name, args.workbook_cost_scale
    )
    matrix.to_csv(output_dir / "opportunity_cost_matrix.csv")
    (output_dir / "scenario_to_index.json").write_text(
        json.dumps(scenario_to_index, indent=2) + "\n", encoding="utf-8"
    )

    n = len(scenarios)
    LOGGER.info("Number of scenarios: %d", n)
    LOGGER.info("Expected matrix entries: %d", n * n)
    LOGGER.info("Loaded diagonal entries: %d", n)
    LOGGER.info("Loaded validation entries: %d", n * n - n)
    LOGGER.info("Missing entries: 0")
    LOGGER.info("Minimum cost: %.12g", matrix.to_numpy().min())
    LOGGER.info("Maximum cost: %.12g", matrix.to_numpy().max())

    if len(set(args.k)) != len(args.k):
        LOGGER.warning(
            "Duplicate K values supplied; each distinct K will be solved once"
        )
    for k in dict.fromkeys(args.k):
        LOGGER.info("Solving CSSC for K=%d with %s", k, args.solver)
        result = solve_cssc(
            matrix, k, args.solver, args.time_limit, args.mip_gap, args.tolerance
        )
        save_result(result, output_dir, n, k, args.solver)
        LOGGER.info(
            "K=%d objective %.12g; representatives: %s",
            k,
            result.objective,
            ", ".join(result.representatives["representative"]),
        )


if __name__ == "__main__":
    main()
