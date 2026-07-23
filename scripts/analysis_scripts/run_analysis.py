#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central runner for analysis scripts.

This runner imports each analysis script, translates a compact YAML
configuration into each script's module-level settings, and then calls the
script's main() function.

The original scripts remain executable on their own.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


PATH_KEYS = {
    "ROOT_DIR",
    "PREFIX_DIR",
    "BASE_NETWORK_PATH",
    "OUTPUT_EXCEL",
    "OUTPUT_DIR",
    "OUTPUT_PLOTS_DIR",
    "OUT_DIR",
    "EXCEL_PATH",
    "CONFIG_YAML",
    "PLOTTING_YAML",
    "HEATMAPS_DIR",
    "ZIP_NAME",
}


OUTPUT_DIRECTORY_KEYS = {
    "OUTPUT_DIR",
    "OUTPUT_PLOTS_DIR",
    "OUT_DIR",
    "HEATMAPS_DIR",
}

OUTPUT_FILE_KEYS = {
    "OUTPUT_EXCEL",
    "ZIP_NAME",
}


SET_KEYS = {
    "EXCLUDED_SCENARIOS",
    "EXCLUDE_SCENARIOS",
    "INCLUDED_SCENARIOS",
    "EXCLUDED_COMPONENTS",
    "INCLUDED_COMPONENTS",
    "INCLUDED_METRICS",
    "EXCLUDED_METRICS",
    "INCLUDED_CARRIERS",
    "EXCLUDED_CARRIERS",
    "INCLUDED_GROUPS",
    "EXCLUDED_GROUPS",
    "LABELED_TECHNOLOGIES",
    "EXCLUDED_TECHNOLOGY_SUBSTRINGS",
    "EXCLUDED_ITEM_LABEL_SUBSTRINGS",
}


PARAMETER_ALIASES = {
    "scenario_name_mode": "SCENARIO_NAME_MODE",
    "save_png": "SAVE_PNG",
    "save_pdf": "SAVE_PDF",
    "dpi": "DPI",
    "zero_tol": "ZERO_TOL",
}


DEFAULT_SCRIPTS: dict[str, dict[str, Any]] = {
    "analysis_network_batch": {
        "file": "analysis_network_batch.py",
        "outputs": {"OUTPUT_EXCEL": "csvs/analysis_networks_energy.xlsx"},
    },
    "analysis_network_powers": {
        "file": "analysis_network_powers.py",
        "outputs": {"OUTPUT_EXCEL": "csvs/analysis_networks_power.xlsx"},
    },
    "analysis_stores_batch": {
        "file": "analysis_stores_batch.py",
        "outputs": {"OUTPUT_EXCEL": "csvs/analysis_stores.xlsx"},
    },
    "demand_comparison": {
        "file": "demand_comparison.py",
        "outputs": {"OUT_DIR": "postprocess_demand_compare"},
        "defaults": {"OUT_STEM": "demand_compare", "GROUPBY": "carrier"},
    },
    "objective_comparison": {
        "file": "objective_comparison.py",
        "outputs": {"OUT_DIR": "_postprocess_objectives"},
        "defaults": {"OUT_STEM": "objectives", "TRY_LOG": True, "SORT_BY_TOTAL": False},
    },
    "plot_capacity_energybalance": {
        "file": "plot_capacity_energybalance.py",
        "outputs": {
            "OUTPUT_DIR": "csvs/capacity_energy_by_technology",
            "OUTPUT_PLOTS_DIR": "csvs/capacity_energy_by_technology/plots",
            "HEATMAPS_DIR": "csvs/capacity_energy_by_technology/heatmaps",
        },
        "defaults": {"SAVE_HEATMAPS": True},
    },
    "plot_scenario_energy_balance": {
        "file": "plot_scenario_energy_balance.py",
        "outputs": {
            "EXCEL_PATH": "csvs/analysis_networks_energy.xlsx",
            "OUTPUT_DIR": "graphs/scenario_energy_balance",
        },
        "defaults": {
            "OUT_STEM": "scenario_energy_balance",
            "TOTAL_MODE": "positive",
        },
    },
    "plot_demand_bars": {
        "file": "plot_demand_bars.py",
        "outputs": {
            "EXCEL_PATH": "postprocess_demand_compare/demand_compare_levels.xlsx",
            "OUT_DIR": "postprocess_demand_compare/demand_level_plots",
        },
        "defaults": {"SHEET_NAME": "levels", "EXCLUDE_BASE": True},
    },
    "plot_heatmap_vs_base": {
        "file": "plot_heatmap_vs_base.py",
        "outputs": {
            "EXCEL_PATH": "csvs/analysis_networks_energy.xlsx",
            "OUT_DIR": "heatmaps_vs_base_energy",
            "ZIP_NAME": "heatmaps_vs_base_energy.zip",
        },
    },
    "plot_heatmap_vs_base_size": {
        "file": "plot_heatmap_vs_base_size.py",
        "outputs": {
            "EXCEL_PATH": "csvs/analysis_networks_power.xlsx",
            "OUT_DIR": "heatmaps_vs_base_power",
            "ZIP_NAME": "heatmaps_vs_base_power.zip",
        },
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML dictionary, got {type(data)}")

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    out = dict(base)

    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value

    return out


def normalize_value(key: str, value: Any) -> Any:
    """
    Convert YAML values to the types expected by the existing scripts.
    """
    if value is None:
        return None

    if key in PATH_KEYS and isinstance(value, str):
        return Path(value)

    if key in SET_KEYS:
        if value is None:
            return None
        if isinstance(value, set):
            return value
        if isinstance(value, list):
            return set(value)
        if isinstance(value, tuple):
            return set(value)
        return {value}

    return value


def output_path(output_dir: Path, relative_path: str | Path) -> str:
    """Build a script output path from the shared output directory."""
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(output_dir / path)


def compact_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalize friendly compact parameter names to script setting names."""
    return {PARAMETER_ALIASES.get(key, key): value for key, value in parameters.items()}


def compact_common_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Translate compact top-level config keys to shared script settings."""
    input_dir = Path(config.get("input_dir", config.get("root_dir", "results/prices_and_renewables")))
    network_file = config.get("network_file", "base_s_adm___2040.nc")
    network_glob = config.get("network_glob", f"networks/{network_file}")
    base_network_path = config.get(
        "base_network_path",
        str(input_dir / "base" / "networks" / network_file),
    )

    excluded = config.get("excluded_scenarios", ["base"])
    common = {
        "ROOT_DIR": str(input_dir),
        "PREFIX_DIR": str(input_dir),
        "NETWORK_GLOB": network_glob,
        "NETWORK_PICKER": config.get("network_picker", network_file),
        "BASE_NETWORK_PATH": base_network_path,
        "BASE_NAME": config.get("base_name", "__BASE__"),
        "BASE_LABEL": config.get("base_label", "__BASE__"),
        "CONFIG_YAML": config.get("model_config", config.get("config_yaml", "config/prices_renewables/config.yaml")),
        "PLOTTING_YAML": config.get("plotting_config", config.get("plotting_yaml", "config/plotting.default.yaml")),
        "EXCLUDED_SCENARIOS": excluded,
        "EXCLUDE_SCENARIOS": excluded,
        "SCENARIO_NAME_MODE": config.get("scenario_name_mode", "folder"),
        "SAVE_PNG": config.get("save_png", True),
        "SAVE_PDF": config.get("save_pdf", False),
        "DPI": config.get("dpi", 300),
        "ZERO_TOL": config.get("zero_tol", 1.0e-9),
    }

    common = deep_merge(common, compact_parameters(config.get("parameters", {})))
    return deep_merge(common, compact_parameters(config.get("common", {})))


def compact_scripts(config: dict[str, Any], output_dir: Path) -> dict[str, dict[str, Any]]:
    """Build script entries and derived output overrides."""
    configured = config.get("scripts", {})
    list_selects_subset = isinstance(configured, list)
    if configured is None:
        configured = {}
    if list_selects_subset:
        configured = {name: {} for name in configured}
    if not isinstance(configured, dict):
        raise ValueError("'scripts' must be a mapping or a list of script names.")

    names = list(configured) if list_selects_subset else list(DEFAULT_SCRIPTS)
    scripts: dict[str, dict[str, Any]] = {}

    for name in names:
        if name not in DEFAULT_SCRIPTS:
            raise KeyError(f"Unknown script '{name}'. Available scripts: {list(DEFAULT_SCRIPTS)}")

        base = DEFAULT_SCRIPTS[name]
        user_entry = configured.get(name, {}) or {}
        if isinstance(user_entry, bool):
            user_entry = {"enabled": user_entry}
        if not isinstance(user_entry, dict):
            raise ValueError(f"Script entry for '{name}' must be a mapping or boolean.")

        overrides = dict(base.get("defaults", {}))
        overrides.update({key: output_path(output_dir, value) for key, value in base.get("outputs", {}).items()})
        overrides = deep_merge(overrides, user_entry.get("overrides", {}))

        scripts[name] = {
            "enabled": user_entry.get("enabled", True),
            "file": user_entry.get("file", base["file"]),
            "overrides": overrides,
        }

    return scripts


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Support both detailed legacy config and compact analysis config."""
    if "output_dir" not in config:
        return config

    output_dir = Path(config["output_dir"])
    return {
        "common": compact_common_settings(config),
        "scripts": compact_scripts(config, output_dir),
    }


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config paths from the current directory first, then this script directory."""
    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path

    cwd_path = config_path.resolve()
    if cwd_path.exists():
        return cwd_path

    return HERE / config_path


def import_script(script_path: Path):
    """Import a Python script as a module."""
    script_path = script_path.resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    module_name = f"_analysis_{script_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def apply_overrides(module, settings: dict[str, Any]) -> None:
    """
    Apply configuration values as module-level variables.

    This intentionally overrides the USER SETTINGS block variables in each
    analysis script without editing the script itself.
    """
    for key, value in settings.items():
        normalized = normalize_value(key, value)
        setattr(module, key, normalized)


def create_output_folders(settings: dict[str, Any]) -> None:
    """Create configured output directories before a script writes files."""
    for key in OUTPUT_DIRECTORY_KEYS:
        value = settings.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

    for key in OUTPUT_FILE_KEYS:
        value = settings.get(key)
        if value:
            Path(value).parent.mkdir(parents=True, exist_ok=True)


def selected_script_names(
    config: dict[str, Any],
    only: list[str] | None,
) -> list[str]:
    """Return the list of script names to run."""
    scripts = config.get("scripts", {})

    if only:
        missing = [name for name in only if name not in scripts]
        if missing:
            raise KeyError(
                f"Unknown script(s): {missing}. "
                f"Available scripts: {list(scripts)}"
            )
        return only

    return [
        name
        for name, entry in scripts.items()
        if entry.get("enabled", True)
    ]


def run_one(
    script_name: str,
    entry: dict[str, Any],
    common_settings: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Run one configured analysis script."""
    script_file = entry.get("file")
    if not script_file:
        raise KeyError(f"Missing 'file' for script '{script_name}'")

    script_path = Path(script_file)
    if not script_path.is_absolute():
        script_path = HERE / script_path

    overrides = entry.get("overrides", {})
    settings = deep_merge(common_settings, overrides)

    print("\n" + "=" * 80)
    print(f"[RUN] {script_name}")
    print(f"[FILE] {script_path}")
    print("=" * 80)

    if dry_run:
        print("[DRY-RUN] Settings that would be applied:")
        for key in sorted(settings):
            print(f"  {key}: {settings[key]}")
        return

    module = import_script(script_path)

    if not hasattr(module, "main"):
        raise AttributeError(
            f"Script '{script_name}' does not define a main() function."
        )

    apply_overrides(module, settings)
    create_output_folders(settings)
    module.main()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="analysis_config.yaml",
        help="Path to the central analysis YAML config.",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only selected scripts by config name.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected scripts and exit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing scripts.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running the next scripts after an error.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the central analysis workflow."""
    args = parse_args()

    config_path = resolve_config_path(args.config)

    config = normalize_config(load_yaml(config_path))
    common_settings = config.get("common", {})
    scripts = config.get("scripts", {})

    names = selected_script_names(config, args.only)

    if args.list:
        print("Selected scripts:")
        for name in names:
            print(f"  - {name}")
        return

    failed = []

    for name in names:
        try:
            run_one(
                script_name=name,
                entry=scripts[name],
                common_settings=common_settings,
                dry_run=args.dry_run,
            )
        except Exception:
            failed.append(name)
            print(f"\n[ERROR] Script failed: {name}")
            traceback.print_exc()

            if not args.continue_on_error:
                raise

    if failed:
        raise RuntimeError(f"Some scripts failed: {failed}")

    print("\nAll selected analysis scripts completed successfully.")


if __name__ == "__main__":
    main()


# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml

# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml \
#  --only objective_comparison

# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml \
#  --only analysis_network_batch plot_heatmap_vs_base

# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml \
#  --list

# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml \
#  --dry-run

# python scripts/analysis_scripts/run_analysis.py \
#  --config scripts/analysis_scripts/analysis_config.yaml \
#  --continue-on-error