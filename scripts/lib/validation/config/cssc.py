# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Configuration model for CSSC-driven stochastic scenario reduction."""

from pydantic import Field, model_validator

from scripts.lib.validation.config._base import ConfigModel


class CsscScenarioReductionConfig(ConfigModel):
    """Configuration for CSSC selection and the reduced stochastic solve."""

    enable: bool = Field(
        False,
        description="Enable the CSSC scenario-selection and stochastic-solve rules.",
    )
    results_prefix: str = Field(
        "results/cutouts_det_capexp_",
        description="Directory containing deterministic scenario result folders.",
    )
    network_name: str = Field(
        "base_s_adm___2050.nc",
        description="Deterministic network filename used for scenario discovery and stochastic inputs.",
    )
    workbook: str = Field(
        "results/cutouts_det_capexp_/analysis_output/validation_heatmaps/validation_heatmaps.xlsx",
        description="Validation workbook containing the total-cost opportunity matrix.",
    )
    output_dir: str = Field(
        "results/cutouts_det_capexp_/analysis_output/cssc",
        description="Root output directory; each K is written to its own child directory.",
    )
    k: list[int] = Field(
        default_factory=lambda: [3],
        min_length=1,
        description="Numbers of representative scenarios to select.",
    )
    solver: str = Field(
        "gurobi",
        min_length=1,
        description="MILP solver passed to Linopy for the CSSC partitioning problem.",
    )
    workbook_cost_scale: float = Field(
        1e9,
        gt=0,
        description="Factor restoring workbook presentation values to raw objective units.",
    )
    time_limit: float | None = Field(
        None,
        gt=0,
        description="Optional CSSC solver time limit in seconds.",
    )
    mip_gap: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Optional relative CSSC MILP optimality gap.",
    )
    mem_mb: int = Field(
        4000,
        gt=0,
        description="Memory requested by the CSSC selection rule in MB.",
    )
    runtime: str = Field(
        "1h",
        min_length=1,
        description="Runtime requested by the CSSC selection rule.",
    )

    @model_validator(mode="after")
    def validate_k_values(self):
        """Require positive, unique representative counts."""
        if any(k <= 0 for k in self.k):
            raise ValueError("cssc_scenario_reduction.k values must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("cssc_scenario_reduction.k values must be unique")
        return self
