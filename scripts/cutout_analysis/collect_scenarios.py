#!/usr/bin/env python3
"""Collect explanatory metrics across deterministic capacity-expansion scenarios."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from analyze_one_case import analyze_network, parse_case_ids


DIAGONAL_NAME = "base_s_adm___2050.nc"

def year_key(value: str) -> int:
    match = re.search(r"(\d{4})", value)
    return int(match.group(1)) if match else 9999


def discover(root: Path) -> list[Path]:
    """Return only exact capacity-expansion outputs, never validation files."""
    paths = root.glob(f"d_*/networks/{DIAGONAL_NAME}")
    return sorted(paths, key=lambda p: (year_key(parse_case_ids(p)[0]), p.name))


def collect(root: Path, output_dir: Path, scenarios: set[str] | None) -> None:
    paths = discover(root)
    if scenarios:
        paths = [p for p in paths if set(parse_case_ids(p)) & scenarios]
    if not paths:
        raise FileNotFoundError(f"No capacity-expansion networks named {DIAGONAL_NAME} found below {root}")

    accumulated: dict[str, list[pd.DataFrame]] = {}
    failures = []
    for number, path in enumerate(paths, 1):
        cap_source, op_source = parse_case_ids(path)
        case_id = f"{cap_source}__{op_source}"
        print(f"[{number}/{len(paths)}] {case_id}: {path}")
        try:
            tables = analyze_network(path)
        except Exception as exc:
            failures.append({"case_id": case_id, "network": str(path), "error": repr(exc)})
            print(f"  FAILED: {exc}")
            continue
        for name, table in tables.items():
            if name == "demand_profiles":
                continue
            part = table.copy()
            part.insert(0, "operation_scenario", op_source)
            part.insert(0, "capacity_scenario", cap_source)
            part.insert(0, "case_id", case_id)
            accumulated.setdefault(name, []).append(part)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, parts in accumulated.items():
        pd.concat(parts, ignore_index=True).to_csv(output_dir / f"all_{name}.csv", index=False)
    pd.DataFrame(failures, columns=["case_id", "network", "error"]).to_csv(
        output_dir / "failures.csv", index=False
    )
    print(f"Wrote collected tables to {output_dir}; failures: {len(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/cutouts_det_capexp_"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cutout_analysis/output/collected"))
    parser.add_argument("--scenarios", nargs="*", help="Optional d_YYYY scenario filter")
    args = parser.parse_args()
    collect(args.results_root, args.output_dir, set(args.scenarios or []) or None)


if __name__ == "__main__":
    main()
