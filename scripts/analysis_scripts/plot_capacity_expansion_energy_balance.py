#!/usr/bin/env python3
"""Compare annual balances of expansion solutions, including expected stochastic balances."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pypsa

import plot_scenario_energy_balance as plotting


def network_order(path: Path) -> tuple[int, int, str]:
    folder = path.parent.parent.name
    year = re.fullmatch(r"d_(\d{4})", folder)
    cluster = re.fullmatch(r"stochastic_K(\d+)", folder)
    return (0, int(year[1]), folder) if year else (1, int(cluster[1]) if cluster else 0, folder)


def extract(path: Path, label: str) -> tuple[pd.DataFrame, dict]:
    n = pypsa.Network(path)
    hours = float(n.snapshot_weightings.generators.sum())
    if not np.isfinite(hours) or hours <= 0:
        raise ValueError(f"Invalid snapshot weights in {path}")
    # Keep bus and component detail until signs have been recorded, matching
    # analysis_network_batch's signed annual balances rather than clipping
    # every timestep to positive/negative values.
    values = n.statistics.energy_balance(
        groupby=["bus_carrier", "bus", "carrier"],
        nice_names=False, drop_zero=False, round=9,
    )
    records = values.rename("value").reset_index()
    probabilities = {}
    if n.has_scenarios:
        probabilities = n.scenario_weightings["weight"].astype(float).to_dict()
        if not np.isclose(sum(probabilities.values()), 1) or min(probabilities.values()) < 0:
            raise ValueError(f"Invalid scenario probabilities in {path}")
        if "scenario" not in records:
            raise ValueError(f"Statistics omitted stochastic scenario dimension in {path}")
        weights = records["scenario"].map(probabilities)
        if weights.isna().any():
            raise ValueError(f"Unknown stochastic scenarios in {path}")
        records["value"] *= weights
    if not np.isfinite(records["value"]).all():
        raise ValueError(f"Nonfinite balance values in {path}")
    records["value"] *= 8760 / hours / plotting.VALUE_SCALE
    records = records.rename(columns={"bus_carrier": "group", "carrier": "technology"})
    records["technology"] = records["technology"].map(
        lambda name: plotting.rename_techs(str(name), preserve_chp=True)
    )
    records["scenario"] = label
    records = records[["group", "technology", "scenario", "value"]]
    manifest = {
        "scenario": label, "network": str(path), "stochastic": bool(n.has_scenarios),
        "snapshot_hours": hours, "probabilities": probabilities,
    }
    del n
    gc.collect()
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/cutouts_det_capexp_nols"))
    parser.add_argument("--reuse-records", action="store_true", help="Reuse this script's extracted balances")
    args = parser.parse_args()
    output = args.root / "analysis_output/graphs/scenario_energy_balance"
    data_dir = args.root / "analysis_output/csvs/capacity_expansion_energy_balance"
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (p for p in args.root.glob("*/networks/*.nc")
         if "__cap-" not in p.name and "__op-" not in p.name),
        key=network_order,
    )
    if not candidates:
        raise ValueError(f"No expansion networks in {args.root}")
    frames, manifests, labels = [], [], []
    for i, path in enumerate(candidates, 1):
        folder = path.parent.parent.name
        label = folder.removeprefix("d_")
        if label in labels:
            raise ValueError(f"Multiple expansion networks for {label}")
        labels.append(label)
        csv_path = data_dir / f"{label}.csv"
        json_path = data_dir / f"{label}.json"
        print(f"[{i}/{len(candidates)}] {label}: {path}", flush=True)
        if args.reuse_records and csv_path.exists() and json_path.exists():
            records = pd.read_csv(csv_path, dtype={"scenario": str})
            manifest = json.loads(json_path.read_text())
        else:
            records, manifest = extract(path, label)
            records.to_csv(csv_path, index=False)
            json_path.write_text(json.dumps(manifest, indent=2) + "\n")
        frames.append(records)
        manifests.append(manifest)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(data_dir / "balances_long.csv", index=False)
    (data_dir / "networks.json").write_text(json.dumps(manifests, indent=2) + "\n")
    data = plotting._merge_electricity_groups(data)
    data["group"] = data["group"].replace(plotting.GROUP_ALIASES)
    plotting.SCENARIO_ORDER = labels
    plotting.FAMILY_GROUPS = {"all": labels}
    plotting.FAMILY_GAP = 0
    plotting.SHOW_FAMILY_LABELS = False
    plotting.SHOW_FAMILY_SEPARATORS = False
    tables = plotting.build_group_tables(data)
    config = plotting._load_yaml(plotting.PLOTTING_YAML)
    summary, count = [], 0
    for group, table in tables.items():
        # Include every solution even when this carrier is absent in one.
        totals, written = plotting.plot_group(group, table.reindex(labels, fill_value=0), config, output)
        summary.append(totals)
        count += int(written)
    pd.concat(summary, ignore_index=True).to_csv(output / "scenario_energy_balance_all_group_totals.csv", index=False)
    print(f"[DONE] {count} carrier graphs for {len(labels)} expansion networks: {output}", flush=True)


if __name__ == "__main__":
    main()
