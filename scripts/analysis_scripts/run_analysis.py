#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central runner for analysis scripts.

This runner imports each analysis script, overrides selected module-level
settings from a YAML configuration file, and then calls the script's main()
function.

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
    "EXCLUDED_TECHNOLOGY_SUBSTRINGS",
    "EXCLUDED_ITEM_LABEL_SUBSTRINGS",
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

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = HERE / config_path

    config = load_yaml(config_path)
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