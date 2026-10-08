#!/usr/bin/env python3
"""Plot sector capacities for expansion solutions without a base or families."""
from pathlib import Path
import gc
import json
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pypsa

import plot_scenario_optimal_capacity as plot
from plot_capacity_expansion_energy_balance import network_order

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_network_powers import _component_metric_sector_records
from scripts.export_stochastic_views import build_view


def main():
    root = Path("results/cutouts_det_capexp_nols")
    output = root / "analysis_output/graphs/scenario_optimal_capacity"
    data_dir = root / "analysis_output/csvs/capacity_expansion_optimal_capacity"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in root.glob("*/networks/*.nc")
                    if "__cap-" not in p.name and "__op-" not in p.name), key=network_order)
    frames, labels, manifest = [], [], []
    for i, path in enumerate(paths, 1):
        label = path.parent.parent.name.removeprefix("d_")
        if label in labels:
            raise ValueError(f"Duplicate expansion solution: {label}")
        labels.append(label)
        print(f"[{i}/{len(paths)}] {label}", flush=True)
        n = pypsa.Network()
        n.import_from_netcdf(path, skip_time=True)
        probabilities = {}
        if n.has_scenarios:
            probabilities = n.scenario_weightings.weight.to_dict()
            # Expansion capacities are shared across scenarios. Check this
            # before collapsing duplicated scenario rows to one asset table.
            for component in n.components:
                df = component.static
                if not isinstance(df.index, pd.MultiIndex):
                    continue
                for col in ("p_nom_opt", "e_nom_opt", "s_nom_opt"):
                    if col in df:
                        grouped = df[col].groupby(level="name")
                        if not np.allclose(grouped.max(), grouped.min(), rtol=1e-6, atol=1e-3):
                            raise ValueError(f"Unshared expansion capacity in {label}: {component.name} {col}")
            view = build_view(n, "expected", probabilities, None)
        else:
            view = n
        records = _component_metric_sector_records(view)
        records["scenario"] = label
        records.to_csv(data_dir / f"{label}.csv", index=False)
        frames.append(records)
        manifest.append({"scenario": label, "network": str(path), "probabilities": probabilities})
        del view, n
        gc.collect()
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(data_dir / "capacities_long.csv", index=False)
    (data_dir / "networks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data = raw.copy()
    data["group"] = data["group"].replace(plot.GROUP_ALIASES)
    data = data[~data.carrier.isin(plot.EXCLUDED_CARRIERS)]
    data["carrier"] = data.carrier.map(lambda name: plot.rename_techs(name, preserve_chp=True))
    plot.SCENARIO_ORDER = labels
    plot.FAMILY_GROUPS = {"all": labels}
    plot.FAMILY_GAP = 0
    plot.SHOW_FAMILY_LABELS = False
    plot.SHOW_FAMILY_SEPARATORS = False
    plot.ALWAYS_LEGEND_CARRIERS = {"DAC"}
    config = plot._load_yaml(plot.PLOTTING_YAML)
    count = 0
    for (group, metric), records in data.groupby(["group", "metric"], sort=True):
        # Supply explicit zero records so absent sectors retain all case labels.
        zeros = pd.DataFrame({"scenario": labels, "carrier": records.carrier.iloc[0], "value": 0.0})
        if records.value.abs().max() > 1e-9:
            count += int(plot.plot_capacity(group, metric, pd.concat([records, zeros]), config, output))
    print(f"[DONE] {count} capacity plots for {len(labels)} expansion solutions: {output}", flush=True)


if __name__ == "__main__":
    main()
