# CSSC-driven stochastic scenario reduction and capacity-expansion solve.
# snakemake cssc_stochastic_networks \
#   --configfile config/cutouts_prices_uncertainty/config_cssc.yaml --cores 32

from pathlib import Path

import pandas as pd


CSSC_CFG = config["cssc_scenario_reduction"]
CSSC_ENABLED = CSSC_CFG["enable"]


if CSSC_ENABLED:
    CSSC_RESULTS_PREFIX = str(Path(CSSC_CFG["results_prefix"]))
    CSSC_NETWORK_NAME = CSSC_CFG["network_name"]
    CSSC_OUTPUT_ROOT = str(Path(CSSC_CFG["output_dir"]))
    CSSC_WORKBOOK = str(Path(CSSC_CFG["workbook"]))
    CSSC_K_VALUES = [int(k) for k in CSSC_CFG["k"]]


    def _cssc_optional_solver_args():
        args = []
        if CSSC_CFG["time_limit"] is not None:
            args.extend(["--time-limit", str(CSSC_CFG["time_limit"])])
        if CSSC_CFG["mip_gap"] is not None:
            args.extend(["--mip-gap", str(CSSC_CFG["mip_gap"])])
        return " ".join(args)


    checkpoint cssc_select_representatives:
        message:
            "Selecting {wildcards.k} CSSC representative scenarios"
        input:
            workbook=CSSC_WORKBOOK,
        output:
            representatives=CSSC_OUTPUT_ROOT + "/{k}/cssc_K{k}_representatives.csv",
        params:
            results_prefix=CSSC_RESULTS_PREFIX,
            network_name=CSSC_NETWORK_NAME,
            output_dir=CSSC_OUTPUT_ROOT,
            solver=CSSC_CFG["solver"],
            workbook_cost_scale=CSSC_CFG["workbook_cost_scale"],
            optional_solver_args=_cssc_optional_solver_args(),
            log_dir=lambda w: str(Path(CSSC_OUTPUT_ROOT) / w.k / "logs"),
        threads: 1
        resources:
            mem_mb=CSSC_CFG["mem_mb"],
            runtime=CSSC_CFG["runtime"],
        log:
            CSSC_OUTPUT_ROOT + "/{k}/logs/cssc.log",
        conda:
            "../envs/environment.yaml"
        shell:
            "mkdir -p {params.log_dir:q} && "
            "python scripts/scenario_reduction_cssc.py "
            "--results-prefix {params.results_prefix:q} "
            "--network-name {params.network_name:q} "
            "--workbook {input.workbook:q} "
            "--workbook-cost-scale {params.workbook_cost_scale} "
            "--output-dir {params.output_dir:q} "
            "--solver {params.solver:q} "
            "--k {wildcards.k} {params.optional_solver_args} "
            "> {log:q} 2>&1"


    def _cssc_representative_networks(wildcards):
        representatives_file = checkpoints.cssc_select_representatives.get(
            k=wildcards.k
        ).output.representatives
        table = pd.read_csv(representatives_file)
        if "representative" not in table:
            raise ValueError(f"Missing representative column in {representatives_file}")
        return [
            str(Path(CSSC_RESULTS_PREFIX) / scenario / "networks" / CSSC_NETWORK_NAME)
            for scenario in table["representative"].astype(str)
        ]


    rule build_stochastic_network_cutouts:
        message:
            "Building the K={wildcards.k} CSSC stochastic network"
        input:
            representatives=lambda w: checkpoints.cssc_select_representatives.get(
                k=w.k
            ).output.representatives,
            scenario_networks=_cssc_representative_networks,
        output:
            network=resources(
                "networks/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
            ),
            config=resources(
                "configs/config.base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.yaml"
            ),
        params:
            solving=config_provider("solving"),
            foresight=config_provider("foresight"),
            sector=config_provider("sector"),
            co2_sequestration_potential=config_provider(
                "sector", "co2_sequestration_potential", default=200
            ),
        log:
            python=logs(
                "build_stochastic_network_cutouts/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log"
            ),
        threads: 1
        resources:
            mem_mb=config_provider("solving", "mem_mb"),
            runtime=config_provider("solving", "runtime", default="2h"),
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/build_stochastic_network_cutouts.py"


    rule solve_cssc_stochastic_network:
        message:
            "Solving the K={wildcards.k} CSSC stochastic network"
        input:
            network=resources(
                "networks/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
            ),
        output:
            network=RESULTS
            + "networks/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            config=RESULTS
            + "configs/config.base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.yaml",
            model=(
                RESULTS
                + "models/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
                if config["solving"]["options"]["store_model"]
                else []
            ),
        params:
            solving=config_provider("solving"),
            foresight=config_provider("foresight"),
            sector=config_provider("sector"),
            co2_sequestration_potential=config_provider(
                "sector", "co2_sequestration_potential", default=200
            ),
            custom_extra_functionality=input_custom_extra_functionality,
        log:
            solver=RESULTS
            + "logs/solve_cssc/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
            memory=RESULTS
            + "logs/solve_cssc/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
            python=RESULTS
            + "logs/solve_cssc/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
        benchmark:
            RESULTS
            + "benchmarks/solve_cssc/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        threads: solver_threads
        resources:
            mem_mb=config_provider("solving", "mem_mb"),
            runtime=config_provider("solving", "runtime", default="6h"),
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/solve_network.py"


    rule cssc_stochastic_networks:
        input:
            expand(
                RESULTS
                + "networks/base_s_cssc_K{k}_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
                k=CSSC_K_VALUES,
                **config["scenario"],
            )
