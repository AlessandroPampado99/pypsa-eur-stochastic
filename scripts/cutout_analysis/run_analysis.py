#!/usr/bin/env python3
"""Run collection and plotting for the cutout uncertainty analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/cutouts_det_capexp_"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cutout_analysis/output"))
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--one-case", type=Path, help="Optionally write a separate detailed one-case extraction")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.one_case:
        run(
            [
                sys.executable,
                str(HERE / "analyze_one_case.py"),
                str(args.one_case),
                "--output-dir",
                str(args.output_root / "one_case"),
            ]
        )
    if not args.skip_collection:
        command = [
            sys.executable,
            str(HERE / "collect_scenarios.py"),
            "--results-root",
            str(args.results_root),
            "--output-dir",
            str(args.output_root / "collected"),
        ]
        if args.scenarios:
            command.extend(["--scenarios", *args.scenarios])
        run(command)
    if not args.skip_plots:
        run(
            [
                sys.executable,
                str(HERE / "plot_analysis.py"),
                "--data-dir",
                str(args.output_root / "collected"),
                "--output-dir",
                str(args.output_root / "plots"),
            ]
        )


if __name__ == "__main__":
    main()
