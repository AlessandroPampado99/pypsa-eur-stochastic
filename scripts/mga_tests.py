#!/usr/bin/env python3
"""
Run minimum/maximum carrier-capacity MGA experiments with PyPSA.

Edit the configuration block below, then run::

    conda run -n pypsa-eur python mga_tests.py

The input network must be ready for optimization. It may already contain a
cost-optimal solution; otherwise set ``BASELINE_MODE`` to ``"solve"`` (or use
the default ``"auto"``) and the script will solve it before running MGA.
"""

from pathlib import Path

import pandas as pd
import pypsa

# =============================================================================
# USER CONFIGURATION
# =============================================================================

NETWORK = Path(
    "results/demand_uncertainty_2035/ELEC_HEAT/networks/base_s_adm___2035.nc"
)
OUTPUT_FOLDER = Path("results/mga_tests")

# Carrier patterns whose combined extendable nominal capacity is the MGA
# objective. Multiple entries are allowed, e.g. ("nuclear", "coal").
CARRIERS = ("urban central gas CHP")

# "exact", "contains", or "regex". With "contains", the example above matches
# carriers such as "urban central gas boiler" and "urban decentral gas boiler".
CARRIER_MATCH_MODE = "contains"
CASE_SENSITIVE = False

# Used in output filenames and log messages.
OBJECTIVE_NAME = "gas_boilers"

# Components whose matching assets form the combined MGA objective.
# Gas boilers are Links; nuclear plants are usually Generators.
COMPONENTS = ("Link",)

# Run both alternatives, or choose only one of "min" / "max".
SENSES = ("min", "max")

# Maximum cost increase relative to the cost-optimal solution (0.05 = 5%).
SLACK = 0.03

# "auto": solve only if the file has no finite objective; "solve": always
# recompute the cost optimum; "reuse": require a solved input network.
BASELINE_MODE = "auto"

# Export the baseline in addition to the MGA networks.
EXPORT_BASELINE = False

# Solver and options passed to both n.optimize() and n.optimize.optimize_mga().
SOLVER_NAME = "gurobi"
SOLVER_OPTIONS = {
    "threads": 32,
    "method": 2,
    "crossover": 0,
    "BarConvTol": 1.0e-05,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
    "GURO_PAR_BARDENSETHRESH": 200,
}

# Set to True for networks with multiple investment periods.
MULTI_INVESTMENT_PERIODS = False

# Optional subset of snapshots. Use None to optimize all network snapshots.
SNAPSHOTS = None

# False accepts feasible non-optimal incumbents (for example "suboptimal" or
# "time_limit"). Solver errors, infeasible, and unbounded runs are still rejected.
REQUIRE_OPTIMAL_SOLUTION = False


# =============================================================================
# IMPLEMENTATION
# =============================================================================

NOMINAL_ATTRIBUTES = {
    "Generator": "p_nom",
    "Link": "p_nom",
    "Store": "e_nom",
    "StorageUnit": "p_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
}


def selected_assets(n: pypsa.Network) -> dict[str, pd.Index]:
    """Return carrier-matching assets for each configured component."""
    selected = {}
    for component in COMPONENTS:
        if component not in NOMINAL_ATTRIBUTES:
            valid = ", ".join(NOMINAL_ATTRIBUTES)
            raise ValueError(
                f"Unsupported component {component!r}; choose from {valid}."
            )

        table = n.static(component)
        if "carrier" not in table:
            raise ValueError(f"Component {component!r} has no 'carrier' column.")
        carriers = table.carrier.fillna("").astype(str)
        if CARRIER_MATCH_MODE == "exact":
            patterns = (
                CARRIERS
                if CASE_SENSITIVE
                else tuple(carrier.casefold() for carrier in CARRIERS)
            )
            values = carriers if CASE_SENSITIVE else carriers.str.casefold()
            mask = values.isin(patterns)
        elif CARRIER_MATCH_MODE in {"contains", "regex"}:
            mask = pd.Series(False, index=table.index)
            for pattern in CARRIERS:
                mask |= carriers.str.contains(
                    pattern,
                    case=CASE_SENSITIVE,
                    regex=CARRIER_MATCH_MODE == "regex",
                    na=False,
                )
        else:
            raise ValueError(
                "CARRIER_MATCH_MODE must be 'exact', 'contains', or 'regex'."
            )
        selected[component] = table.index[mask]

    if not any(len(index) for index in selected.values()):
        available = sorted(
            {
                str(carrier)
                for component in COMPONENTS
                for carrier in n.static(component).carrier.dropna().unique()
            }
        )
        raise ValueError(
            f"No assets matching {CARRIERS!r} ({CARRIER_MATCH_MODE}) in "
            f"{COMPONENTS}. "
            f"Available carriers include: {', '.join(available)}"
        )
    return selected


def mga_weights(n: pypsa.Network) -> dict[str, dict[str, pd.Series]]:
    """Build native PyPSA MGA weights for matching extendable capacities."""
    weights = {}
    for component, assets in selected_assets(n).items():
        attribute = NOMINAL_ATTRIBUTES[component]
        extendable = n.get_extendable_i(component)
        names = assets.intersection(extendable)
        if len(names):
            # PyPSA reindexes this Series to all extendable assets. Explicit
            # zeros keep unrelated assets out of the alternative objective.
            coefficients = pd.Series(0.0, index=n.static(component).index)
            coefficients.loc[names] = 1.0
            weights[component] = {attribute: coefficients}

    if not weights:
        raise ValueError(
            f"Assets matching {CARRIERS!r} exist, but none of their nominal "
            "capacities are extendable. MGA can only vary optimization variables."
        )
    return weights


def has_solution(n: pypsa.Network) -> bool:
    """Check whether the network contains a usable baseline objective."""
    try:
        return pd.notna(float(n.objective))
    except (AttributeError, TypeError, ValueError):
        return False


def validate_solution(status: str, condition: str, label: str) -> None:
    """Accept any usable incumbent, optionally requiring proven optimality."""
    if status != "ok":
        raise RuntimeError(
            f"{label} failed: solver status={status!r}, condition={condition!r}."
        )
    if REQUIRE_OPTIMAL_SOLUTION and condition != "optimal":
        raise RuntimeError(
            f"{label} was not proven optimal: condition={condition!r}. "
            "Set REQUIRE_OPTIMAL_SOLUTION = False to accept this solution."
        )
    if condition != "optimal":
        print(f"Accepting {label} with termination condition {condition!r}.")


def capacity_summary(n: pypsa.Network) -> dict[str, float]:
    """Sum optimized and existing nominal capacity of selected assets."""
    result = {}
    total = 0.0
    for component, assets in selected_assets(n).items():
        table = n.static(component)
        attribute = NOMINAL_ATTRIBUTES[component]
        optimized = f"{attribute}_opt"
        values = (
            table.loc[assets, optimized]
            if optimized in table
            else table.loc[assets, attribute]
        )
        value = float(values.fillna(0.0).sum())
        result[f"{component}_{attribute}"] = value
        total += value
    result["total_nominal_capacity"] = total
    return result


def solve_baseline(n: pypsa.Network) -> None:
    status, condition = n.optimize(
        snapshots=SNAPSHOTS,
        multi_investment_periods=MULTI_INVESTMENT_PERIODS,
        solver_name=SOLVER_NAME,
        solver_options=SOLVER_OPTIONS,
    )
    validate_solution(status, condition, "Baseline optimization")


def main() -> None:
    if not CARRIERS or any(not carrier for carrier in CARRIERS):
        raise ValueError("CARRIERS must contain at least one non-empty pattern.")
    if BASELINE_MODE not in {"auto", "solve", "reuse"}:
        raise ValueError("BASELINE_MODE must be 'auto', 'solve', or 'reuse'.")
    if not SENSES or any(sense not in {"min", "max"} for sense in SENSES):
        raise ValueError("SENSES must contain 'min', 'max', or both.")
    if SLACK < 0:
        raise ValueError("SLACK must be non-negative.")
    if not NETWORK.is_file():
        raise FileNotFoundError(f"Input network not found: {NETWORK}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    baseline = pypsa.Network(NETWORK)
    selected_assets(baseline)  # Fail early on a misspelled carrier/component.

    should_solve = BASELINE_MODE == "solve" or (
        BASELINE_MODE == "auto" and not has_solution(baseline)
    )
    if should_solve:
        print("Solving cost-optimal baseline ...")
        solve_baseline(baseline)
    elif not has_solution(baseline):
        raise ValueError(
            "The input has no finite objective. Use BASELINE_MODE = 'solve' or 'auto'."
        )
    else:
        print(f"Reusing baseline objective: {baseline.objective:g}")

    if EXPORT_BASELINE:
        baseline.export_to_netcdf(OUTPUT_FOLDER / "baseline.nc")

    records = []
    for sense in SENSES:
        print(f"Running {sense}imum-{OBJECTIVE_NAME} MGA with {SLACK:.1%} slack ...")
        alternative = baseline.copy()
        status, condition = alternative.optimize.optimize_mga(
            snapshots=SNAPSHOTS,
            multi_investment_periods=MULTI_INVESTMENT_PERIODS,
            weights=mga_weights(alternative),
            sense=sense,
            slack=SLACK,
            solver_name=SOLVER_NAME,
            solver_options=SOLVER_OPTIONS,
        )
        validate_solution(status, condition, f"{sense}imum-{OBJECTIVE_NAME} MGA")

        output = OUTPUT_FOLDER / f"mga_{OBJECTIVE_NAME}_{sense}.nc"
        alternative.export_to_netcdf(output)
        records.append(
            {
                "sense": sense,
                "carrier_patterns": " | ".join(CARRIERS),
                "carrier_match_mode": CARRIER_MATCH_MODE,
                "slack": SLACK,
                "baseline_objective": float(baseline.objective),
                "output_network": str(output),
                **capacity_summary(alternative),
            }
        )

    summary = pd.DataFrame.from_records(records).set_index("sense")
    summary.to_csv(OUTPUT_FOLDER / "summary.csv")
    print(f"\nResults written to {OUTPUT_FOLDER.resolve()}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
