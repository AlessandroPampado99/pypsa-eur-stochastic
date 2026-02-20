# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


# --- Add this rule somewhere near solve_sector_network (same file where you define it) ---
import yaml

def _stoch_cfg():
    return (config.get("stochastic_scenarios", {}) or {})

def _stoch_enabled():
    return bool(_stoch_cfg().get("enable", False))

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
    if _stoch_enabled():
        return RESULTS + f"networks/base_s_stoch_{w.clusters}_{w.opts}_{w.sector_opts}_{w.planning_horizons}.nc"
    return resources(f"networks/base_s_{w.clusters}_{w.opts}_{w.sector_opts}_{w.planning_horizons}.nc")


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
