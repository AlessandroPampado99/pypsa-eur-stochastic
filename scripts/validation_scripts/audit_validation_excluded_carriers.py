#!/usr/bin/env python3
"""Audit actual exported capacity flags for validation's excluded carriers."""
from pathlib import Path
import pandas as pd
import xarray as xr

ROOT = Path("results/demand_uncertainty_2035")
OUT = ROOT / "analysis_output/validation_heatmaps/diagnostics/family_cost_explanation"
CARRIERS = {
    "co2 stored", "co2 sequestered", "co2 atmosphere", "oil primary", "gas",
    "gas for industry", "solid biomass for industry", "coal for industry",
    "land transport oil", "industry methanol", "naphtha for industry", "shipping oil",
    "shipping methanol", "kerosene for aviation", "agriculture machinery oil", "process emissions",
}


def main():
    costs = pd.read_excel(ROOT / "analysis_output/validation_heatmaps/validation_heatmaps.xlsx",
                          sheet_name="total_cost", index_col=0)
    records = []
    for capacity in costs.index:
        for operation in costs.columns:
            if capacity == operation:
                continue
            path = ROOT / capacity / "networks" / f"base_s_adm___2035__cap-{capacity}__op-{operation}.nc"
            with xr.open_dataset(path) as ds:
                for component, nominal in (("generators", "p_nom"), ("links", "p_nom"),
                                           ("stores", "e_nom"), ("storage_units", "p_nom"),
                                           ("lines", "s_nom"), ("transformers", "s_nom")):
                    key = f"{component}_carrier"
                    if key not in ds:
                        continue
                    carrier = ds[key].to_series()
                    ids = carrier.index[carrier.isin(CARRIERS)]
                    if not len(ids):
                        continue
                    table = pd.DataFrame(dict(capacity=capacity, operation=operation,
                                               component=component, carrier=carrier.loc[ids], asset=ids))
                    for attr, default in ((f"{nominal}_extendable", False), (nominal, 0),
                                          (f"{nominal}_opt", 0), (f"{nominal}_min", 0),
                                          (f"{nominal}_max", float("inf")),
                                          ("capital_cost", 0), ("marginal_cost", 0)):
                        variable = f"{component}_{attr}"
                        table[attr] = ds[variable].to_series().reindex(ids) if variable in ds else default
                    table["extendable"] = table[f"{nominal}_extendable"].astype(bool)
                    table["finite_nominal_max"] = table[f"{nominal}_max"] < float("inf")
                    records.append(table.reset_index(drop=True))
            print(f"Audited {capacity} -> {operation}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    assets = pd.concat(records, ignore_index=True)
    assets.to_csv(OUT / "excluded_carrier_asset_audit.csv", index=False)
    summary = assets.groupby(["component", "carrier", "extendable", "finite_nominal_max"]).agg(
        asset_cases=("asset", "size"), unique_assets=("asset", "nunique"),
        capital_cost_min=("capital_cost", "min"), capital_cost_max=("capital_cost", "max"))
    summary.to_csv(OUT / "excluded_carrier_audit_summary.csv")
    presence = assets.groupby(["capacity", "operation", "carrier"]).size().unstack(fill_value=0).reindex(columns=sorted(CARRIERS), fill_value=0)
    presence.to_csv(OUT / "excluded_carrier_presence.csv")
    (OUT / "excluded_carrier_audit.md").write_text(
        "# Exported validation capacity flags\n\n"
        "Audited all 156 off-diagonal networks against the supplied carrier list. "
        "Exclusion preserves the original flag; it does not force an originally fixed asset to become extendable. "
        "Absent carriers are exported in the presence table. Component defaults are used for attributes omitted from NetCDF. "
        "An infinite nominal maximum does not establish that other constraints are nonbinding.\n\n"
        + summary.to_string() + "\n")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
