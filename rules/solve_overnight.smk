# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


# --- Add this rule somewhere near solve_sector_network (same file where you define it) ---
from pathlib import Path
import yaml

def _stoch_cfg():
    return (config.get("stochastic_scenarios", {}) or {})

def _stoch_enabled():
    return bool(_stoch_cfg().get("enable", False))

def _stoch_active_scenario():
    return _stoch_cfg().get("active_scenario", None)

def _stoch_preprocess_enabled():
    return _stoch_enabled() or _stoch_active_scenario() is not None

def _stoch_cfg_for_wildcards(w):
    return config_provider("stochastic_scenarios", default={"enable": False})(w) or {}

def _stoch_preprocess_enabled_for_wildcards(w):
    stoch = _stoch_cfg_for_wildcards(w)
    return bool(stoch.get("enable", False)) or stoch.get("active_scenario") is not None

def _stoch_file():
    return _stoch_cfg().get("file", None)

def _stoch_scenario_names():
    """Read scenario names from the YAML file at parse time."""
    p = _stoch_file()
    if not p:
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sc = data.get("scenarios", {}) or {}
    return list(sc.keys())

def input_sector_network(w):
    if _stoch_preprocess_enabled_for_wildcards(w):
        return (
            RESULTS
            + f"networks/base_s_stoch_{w.clusters}_{w.opts}_{w.sector_opts}_{w.planning_horizons}.nc"
        )
    return resources(
        "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
    )


STOCH_SCENARIOS = _stoch_scenario_names() if _stoch_enabled() else []


rule stochasticify_sector_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        stochastic_scenarios=config_provider("stochastic_scenarios", default={"enable": False}),
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
        ),
    output:
        network=RESULTS
        + "networks/base_s_stoch_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        config=RESULTS
        + "configs/config.base_s_stoch_{clusters}_{opts}_{sector_opts}_{planning_horizons}.yaml",
    shadow:
        shadow_config
    log:
        python=RESULTS
        + "logs/stochasticify/base_s_stoch_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: 1
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="1h"),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/stochasticify_network.py"


# --- Then modify solve_sector_network to take the stochasticified network as input ---
# Replace:
#   input: network=resources("networks/base_s_...nc")
# With:

rule solve_sector_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=input_sector_network,

    output:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        config=RESULTS
        + "configs/config.base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.yaml",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_sector_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"

if _stoch_enabled():

    rule export_stochastic_expected:
        message:
            "Exporting expected deterministic view from stochastic solution (__exp)"
        params:
            scenarios_file=lambda w: _stoch_file(),
            mode="expected",
        input:
            network=RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        output:
            expected=RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}__exp.nc",
        threads: 1
        resources:
            mem_mb=8000,
        log:
            RESULTS
            + "logs/export_stochastic_views/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}__exp.log",
        script:
            "../scripts/export_stochastic_views.py"


    rule export_stochastic_scenario:
        message:
            "Exporting scenario deterministic view from stochastic solution (__sc-{wildcards.stoch_scenario})"
        params:
            scenarios_file=lambda w: _stoch_file(),
            mode="scenario",
            scenario=lambda w: w.stoch_scenario,
        input:
            network=RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        output:
            scenario=RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}__sc-{stoch_scenario}.nc",
        threads: 1
        resources:
            mem_mb=8000,
        log:
            RESULTS
            + "logs/export_stochastic_views/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}__sc-{stoch_scenario}.log",
        script:
            "../scripts/export_stochastic_views.py"


VALIDATION_CFG_BLOCK = config.get("validation", {}) or {}
VALIDATION_ENABLED = bool(VALIDATION_CFG_BLOCK.get("enable", False))
VALIDATION_CFG_PATH = VALIDATION_CFG_BLOCK.get("config", "config/validation.yaml")

if VALIDATION_ENABLED:
    with open(VALIDATION_CFG_PATH, encoding="utf-8") as f:
        VALIDATION_CFG = yaml.safe_load(f) or {}
    VALIDATION_RUNS = VALIDATION_CFG.get("validation_runs", {}) or {}
else:
    VALIDATION_CFG = {}
    VALIDATION_RUNS = {}


def _current_run_prefix() -> str:
    """
    Return the current run prefix from the main config.

    Falls back to an empty string if no prefix is defined.
    """
    return str((config.get("run", {}) or {}).get("prefix", "") or "")


def _normalize_validation_ref(ref, field_name: str) -> dict:
    """
    Normalize a validation reference into a stable internal representation.

    Supported user formats
    ----------------------
    1) Short form:
        capacities_from: scenario_A

    2) Structured form:
        capacities_from:
          name: scenario_A
          prefix: my_prefix

    3) Explicit path form:
        capacities_from:
          path: /full/or/relative/path/to/network.nc

    Returned format
    ---------------
    {
        "name": str | None,
        "prefix": str,
        "path": str | None,
    }
    """
    default_prefix = _current_run_prefix()

    if isinstance(ref, str):
        return {
            "name": ref,
            "prefix": default_prefix,
            "path": None,
        }

    if not isinstance(ref, dict):
        raise ValueError(
            f"Validation field '{field_name}' must be either a string or a mapping."
        )

    if "path" in ref:
        path = ref["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"Validation field '{field_name}.path' must be a non-empty string."
            )
        return {
            "name": ref.get("name"),
            "prefix": str(ref.get("prefix", default_prefix) or default_prefix),
            "path": path,
        }

    name = ref.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Validation field '{field_name}' must define either 'path' "
            f"or a non-empty 'name'."
        )

    return {
        "name": name,
        "prefix": str(ref.get("prefix", default_prefix) or default_prefix),
        "path": None,
    }


def _validation_pairs():
    """
    Parse validation runs into a stable list of normalized entries.

    Each entry contains explicit information for:
    - label
    - capacity source
    - operation source
    """
    pairs = []

    for label, entry in VALIDATION_RUNS.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Validation entry '{label}' must be a mapping.")

        if "capacities_from" not in entry or "operations_on" not in entry:
            raise ValueError(
                f"Validation entry '{label}' must contain both "
                f"'capacities_from' and 'operations_on'."
            )

        cap_ref = _normalize_validation_ref(
            entry["capacities_from"], f"{label}.capacities_from"
        )
        op_ref = _normalize_validation_ref(
            entry["operations_on"], f"{label}.operations_on"
        )

        pairs.append(
            {
                "label": str(label),
                "cap_source": cap_ref["name"] or str(label),
                "op_source": op_ref["name"] or str(label),
                "cap_prefix": cap_ref["prefix"],
                "op_prefix": op_ref["prefix"],
                "cap_path": cap_ref["path"],
                "op_path": op_ref["path"],
            }
        )

    return pairs


VALIDATION_PAIRS = _validation_pairs()


def _validation_pair_map():
    """
    Build lookup table for validation entries by (cap_source, op_source).
    """
    by_cap_op = {}

    for pair in VALIDATION_PAIRS:
        key = (pair["cap_source"], pair["op_source"])

        if key in by_cap_op:
            raise ValueError(
                "Validation runs must be unique by (capacities_from, operations_on). "
                f"Duplicate pair found for cap='{key[0]}', op='{key[1]}'."
            )

        by_cap_op[key] = pair

    return by_cap_op


VALIDATION_PAIR_BY_CAP_OP = _validation_pair_map()


def _get_validation_pair(w):
    """
    Resolve the validation pair for the current wildcards using
    (cap_source, op_source).
    """
    cap_source = getattr(w, "cap_source", None)
    op_source = getattr(w, "op_source", None)

    if cap_source is None or op_source is None:
        raise ValueError(
            "Validation rules require both 'cap_source' and 'op_source' wildcards."
        )

    key = (cap_source, op_source)
    if key not in VALIDATION_PAIR_BY_CAP_OP:
        raise ValueError(
            f"No validation entry found in {VALIDATION_CFG_PATH} for "
            f"capacities_from='{cap_source}', operations_on='{op_source}'."
        )

    return VALIDATION_PAIR_BY_CAP_OP[key]


def _run_base_dir(root: str, prefix: str, name: str) -> Path:
    """
    Build the base directory of a run.

    The canonical structure is assumed to be:
        <root>/<prefix>/<name>/

    If prefix is empty, it falls back to:
        <root>/<name>/
    """
    base = Path(root)
    if prefix:
        return base / prefix / name
    return base / name


def validation_capacity_network(w):
    """
    Path to the solved capacity-expansion network used as source of capacities.

    Resolution priority:
    1. explicit path from validation.yaml
    2. canonical path under results/<prefix>/<name>/networks/
    """
    pair = _get_validation_pair(w)

    if pair["cap_path"]:
        return pair["cap_path"]

    return str(
        _run_base_dir("results", pair["cap_prefix"], pair["cap_source"])
        / "networks"
        / f"base_s_{w.clusters}_{w.opts}_{w.sector_opts}_{w.planning_horizons}.nc"
    )


def validation_operation_network(w):
    """
    Path to the operation network used as operational environment.

    Resolution priority:
    1. explicit path from validation.yaml
    2. canonical path under resources/<prefix>/<name>/networks/

    The stochasticified resource is used only when stochastic scenarios are enabled.
    """
    pair = _get_validation_pair(w)

    if pair["op_path"]:
        return pair["op_path"]

    network_prefix = "base_s_stoch" if _stoch_enabled() else "base_s"

    return str(
        _run_base_dir("resources", pair["op_prefix"], pair["op_source"])
        / "networks"
        / f"{network_prefix}_{w.clusters}_{w.opts}_{w.sector_opts}_{w.planning_horizons}.nc"
    )


def validation_targets():
    """
    Build all validation target networks from validation.yaml.

    Validation outputs are stored under:
        results/<current_prefix>/<cap_source>/...
    """
    targets = []

    current_prefix = _current_run_prefix()

    for pair in VALIDATION_PAIRS:
        targets += expand(
            "results/{run_prefix}/{cap_source}/networks/"
            + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
            + "__cap-{cap_source}__op-{op_source}.nc",
            run_prefix=[current_prefix],
            cap_source=[pair["cap_source"]],
            op_source=[pair["op_source"]],
            **config["scenario"],
        )

    return targets


rule solve_validation_operations_network:
    message:
        "Solving validation dispatch with capacities from {wildcards.cap_source} and operations on {wildcards.op_source}"
    params:
        options=config_provider("solving", "options"),
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        capacity_network=validation_capacity_network,
        operation_network=validation_operation_network,
    output:
        network="results/{run_prefix}/{cap_source}/networks/"
        + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        + "__cap-{cap_source}__op-{op_source}.nc",
    shadow:
        shadow_config
    log:
        solver="results/{run_prefix}/{cap_source}/logs/solve_validation_operations_network/"
        + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        + "__cap-{cap_source}__op-{op_source}_solver.log",
        python="results/{run_prefix}/{cap_source}/logs/solve_validation_operations_network/"
        + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        + "__cap-{cap_source}__op-{op_source}_python.log",
        memory="results/{run_prefix}/{cap_source}/logs/solve_validation_operations_network/"
        + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        + "__cap-{cap_source}__op-{op_source}_memory.log",
    benchmark:
        "results/{run_prefix}/{cap_source}/benchmarks/solve_validation_operations_network/"
        + "base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        + "__cap-{cap_source}__op-{op_source}"
    threads: 4
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    wildcard_constraints:
        run_prefix=".+",
        cap_source=".+",
        op_source=".+",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_validation_operations_network.py"


rule all_validation:
    input:
        validation_targets()
    default_target: False