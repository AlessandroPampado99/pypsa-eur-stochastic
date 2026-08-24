#!/usr/bin/env python3
"""Extract explanatory metrics from one solved PyPSA-Eur network."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


MAIN_COUNTRIES = ("IT", "DE", "ES", "FR")
RENEWABLE_CARRIERS = {
    "solar", "solar rooftop", "solar-hsat", "onwind", "offwind-ac",
    "offwind-dc", "offwind-float", "ror",
}


def weights(n: pypsa.Network, kind: str = "generators") -> pd.Series:
    sw = n.snapshot_weightings
    if isinstance(sw, pd.Series):
        return sw.astype(float)
    for col in (kind, "objective", "generators", "stores"):
        if col in sw:
            return sw[col].astype(float)
    return sw.iloc[:, 0].astype(float)


def weighted_stats(frame: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Column-wise weighted statistics, treating weights as represented hours."""
    frame = frame.apply(pd.to_numeric, errors="coerce")
    w = w.reindex(frame.index).fillna(0.0)
    valid = frame.notna()
    denom = valid.mul(w, axis=0).sum()
    mean = frame.mul(w, axis=0).sum().div(denom.replace(0, np.nan))
    variance = frame.sub(mean).pow(2).mul(w, axis=0).sum().div(
        denom.replace(0, np.nan)
    )
    return pd.DataFrame(
        {
            "mean": mean,
            "std": np.sqrt(variance),
            "min": frame.min(),
            "p05": frame.quantile(0.05),
            "p10": frame.quantile(0.10),
            "median": frame.median(),
            "p90": frame.quantile(0.90),
            "p95": frame.quantile(0.95),
            "max": frame.max(),
        }
    )


def bus_country(n: pypsa.Network) -> pd.Series:
    countries = n.buses.get("country", pd.Series("", index=n.buses.index)).fillna("")
    fallback = n.buses.index.to_series().str.extract(r"^([A-Z]{2})(?:\b| )", expand=False)
    return countries.astype(str).where(countries.astype(str).ne(""), fallback).fillna("")


def component_country(n: pypsa.Network, table: pd.DataFrame, bus_col: str = "bus") -> pd.Series:
    bc = bus_country(n)
    if bus_col not in table:
        return pd.Series("", index=table.index)
    return table[bus_col].map(bc).fillna("")


def demand_group(carrier: str) -> str:
    c = str(carrier).lower()
    if "land transport" in c or ("ev" in c and "battery" not in c):
        return "land_transport_ev"
    if "heat" in c:
        if "urban" in c and ("decentral" in c or "decentralised" in c):
            return "urban_decentral_heat"
        if "urban" in c:
            return "urban_heat"
        if "rural" in c:
            return "rural_heat"
        if "decentral" in c or "decentralised" in c:
            return "decentral_heat"
    if c in {"ac", "electricity", "low voltage"} or "electricity demand" in c:
        return "electricity"
    return "other"


def load_timeseries(n: pypsa.Network) -> pd.DataFrame:
    static = n.loads.get("p_set", pd.Series(0.0, index=n.loads.index)).astype(float)
    dynamic = getattr(n.loads_t, "p_set", pd.DataFrame(index=n.snapshots))
    return dynamic.reindex(index=n.snapshots, columns=n.loads.index).fillna(
        pd.DataFrame(
            np.broadcast_to(static.to_numpy(), (len(n.snapshots), len(static))),
            index=n.snapshots,
            columns=static.index,
        )
    )


def extract_demand(n: pypsa.Network) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = load_timeseries(n)
    meta = pd.DataFrame(index=n.loads.index)
    meta["carrier"] = n.loads.carrier.fillna("")
    meta["demand_group"] = meta.carrier.map(demand_group)
    meta["country"] = component_country(n, n.loads)
    w = weights(n)
    records, profiles = [], {}
    for (group, country), names in meta.groupby(["demand_group", "country"]).groups.items():
        if group == "other":
            continue
        profile = p[list(names)].sum(axis=1)
        key = f"{group}|{country or 'ALL'}"
        profiles[key] = profile
        records.append(
            {
                "demand_group": group,
                "country": country,
                "energy_mwh": float(profile.mul(w).sum()),
                "mean_mw": float(profile.mul(w).sum() / w.sum()),
                "peak_mw": float(profile.max()),
                "minimum_mw": float(profile.min()),
                "load_factor": float(profile.mul(w).sum() / w.sum() / profile.max())
                if profile.max() else np.nan,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(profiles, index=n.snapshots)


def extract_capacity_factors(n: pypsa.Network) -> pd.DataFrame:
    gens = n.generators.index[n.generators.carrier.isin(RENEWABLE_CARRIERS)]
    if gens.empty:
        return pd.DataFrame()
    cf = n.generators_t.p_max_pu.reindex(index=n.snapshots, columns=gens)
    countries = component_country(n, n.generators.loc[gens])
    carriers = n.generators.loc[gens, "carrier"]
    rows = []
    for carrier in sorted(set(carriers)):
        for country in ("ALL", *MAIN_COUNTRIES):
            selected = gens[(carriers == carrier) & ((countries == country) if country != "ALL" else True)]
            if selected.empty:
                continue
            # Equal weighting across sites describes weather, not the optimized build.
            profile = cf[selected].mean(axis=1)
            stats = weighted_stats(profile.to_frame("value"), weights(n)).loc["value"]
            row = {"carrier": carrier, "country": country, "n_generators": len(selected)}
            row.update(stats.to_dict())
            row["hours_below_0_05"] = float(
                weights(n).where(profile < 0.05, 0.0).sum()
            )
            row["hours_below_0_10"] = float(
                weights(n).where(profile < 0.10, 0.0).sum()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def extract_generation(n: pypsa.Network) -> pd.DataFrame:
    p = n.generators_t.p.reindex(columns=n.generators.index).fillna(0.0)
    meta = pd.DataFrame(
        {
            "carrier": n.generators.carrier.fillna(""),
            "country": component_country(n, n.generators),
        }
    )
    w = weights(n)
    rows = []
    for (carrier, country), names in meta.groupby(["carrier", "country"]).groups.items():
        profile = p[list(names)].sum(axis=1)
        rows.append(
            {
                "carrier": carrier,
                "country": country,
                "generation_mwh": float(profile.mul(w).sum()),
                "peak_mw": float(profile.max()),
            }
        )
    return pd.DataFrame(rows)


def extract_capacities(n: pypsa.Network) -> pd.DataFrame:
    specs = [
        ("Generator", n.generators, "p_nom_opt", "p_nom", "MW", "bus"),
        ("Link", n.links, "p_nom_opt", "p_nom", "MW", "bus0"),
        ("Store", n.stores, "e_nom_opt", "e_nom", "MWh", "bus"),
        ("StorageUnit", n.storage_units, "p_nom_opt", "p_nom", "MW", "bus"),
        ("Line", n.lines, "s_nom_opt", "s_nom", "MVA", "bus0"),
    ]
    rows = []
    for component, table, opt_col, base_col, unit, bus_col in specs:
        if table.empty:
            continue
        values = table.get(opt_col, table.get(base_col, pd.Series(0.0, index=table.index)))
        if base_col in table:
            values = values.fillna(table[base_col])
        meta = pd.DataFrame(
            {
                "carrier": table.get("carrier", pd.Series(component, index=table.index)).fillna(component),
                "country": component_country(n, table, bus_col),
                "value": pd.to_numeric(values, errors="coerce").fillna(0.0),
            }
        )
        for (carrier, country), part in meta.groupby(["carrier", "country"]):
            rows.append(
                {
                    "component": component,
                    "carrier": carrier,
                    "country": country,
                    "capacity": float(part.value.sum()),
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def extract_costs(n: pypsa.Network) -> pd.DataFrame:
    rows = []
    for kind, values in (("capex", n.statistics.capex()), ("opex", n.statistics.opex())):
        series = values if isinstance(values, pd.Series) else values.sum(axis=1)
        for idx, value in series.items():
            labels = idx if isinstance(idx, tuple) else (idx,)
            rows.append(
                {
                    "cost_type": kind,
                    "component": str(labels[0]),
                    "carrier": str(labels[-1]),
                    "cost_eur": float(value),
                }
            )
    return pd.DataFrame(rows)


def extract_diagnostics(n: pypsa.Network) -> pd.DataFrame:
    w = weights(n)
    rows = []
    load_gens = n.generators.index[n.generators.carrier.eq("load")]
    shedding = (
        n.generators_t.p.reindex(columns=load_gens, fill_value=0.0).sum(axis=1)
        if len(load_gens) else pd.Series(0.0, index=n.snapshots)
    )
    rows.append({"metric": "load_shedding_mwh", "value": float(shedding.mul(w).sum())})
    res = n.generators.index[n.generators.carrier.isin(RENEWABLE_CARRIERS)]
    if len(res):
        p_nom = n.generators.loc[res].get("p_nom_opt", n.generators.loc[res, "p_nom"])
        p_nom = p_nom.fillna(n.generators.loc[res, "p_nom"])
        available = n.generators_t.p_max_pu.reindex(columns=res).mul(p_nom, axis=1)
        actual = n.generators_t.p.reindex(columns=res, fill_value=0.0)
        curtailment = (available - actual).clip(lower=0.0).sum(axis=1)
        rows.append({"metric": "renewable_curtailment_mwh", "value": float(curtailment.mul(w).sum())})
    rows.extend(
        [
            {"metric": "capex_eur", "value": float(n.statistics.capex().sum())},
            {"metric": "opex_eur", "value": float(n.statistics.opex().sum())},
            {"metric": "objective_eur", "value": float(getattr(n, "objective", np.nan))},
        ]
    )
    prices = getattr(n.buses_t, "marginal_price", pd.DataFrame())
    if not prices.empty:
        ac = n.buses.index[n.buses.carrier.isin(["AC", "electricity", "low voltage"])]
        profile = prices.reindex(columns=ac).mean(axis=1)
        rows.extend(
            [
                {"metric": "mean_electricity_price_eur_per_mwh", "value": float(profile.mul(w).sum() / w.sum())},
                {"metric": "p95_electricity_price_eur_per_mwh", "value": float(profile.quantile(0.95))},
                {"metric": "max_electricity_price_eur_per_mwh", "value": float(profile.max())},
            ]
        )
    return pd.DataFrame(rows)


def parse_case_ids(path: Path) -> tuple[str, str]:
    match = re.search(r"__cap-(d_\d{4})__op-(d_\d{4})", path.name)
    if match:
        return match.group(1), match.group(2)
    scenario = path.parents[1].name if path.parent.name == "networks" else path.stem
    return scenario, scenario


def analyze_network(network_path: Path) -> dict[str, pd.DataFrame]:
    n = pypsa.Network(str(network_path))
    demand, profiles = extract_demand(n)
    return {
        "demand": demand,
        "demand_profiles": profiles,
        "renewable_capacity_factors": extract_capacity_factors(n),
        "generation": extract_generation(n),
        "capacities": extract_capacities(n),
        "costs": extract_costs(n),
        "diagnostics": extract_diagnostics(n),
    }


def write_case(network_path: Path, output_dir: Path) -> None:
    cap_source, op_source = parse_case_ids(network_path)
    tables = analyze_network(network_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=name == "demand_profiles")
    metadata = {
        "network": str(network_path),
        "capacity_scenario": cap_source,
        "operation_scenario": op_source,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cutout_analysis/output/one_case"))
    args = parser.parse_args()
    write_case(args.network, args.output_dir)
    print(f"Wrote one-case analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
