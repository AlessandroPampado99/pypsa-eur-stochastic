#!/usr/bin/env python3
"""Show DAC dispatch and capacity separately using exported expansion tables."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_scenario_optimal_capacity as capacity


def main():
    root = Path("results/cutouts_det_capexp_nols/analysis_output")
    energy_dir = root / "graphs/scenario_energy_balance"
    capacity_dir = root / "graphs/scenario_optimal_capacity"
    out = root / "graphs/DAC_detail"
    out.mkdir(parents=True, exist_ok=True)
    def energy(group):
        return pd.read_csv(energy_dir / f"scenario_energy_balance_{group}_by_technology.csv", index_col=0)["DAC"]
    def nominal(group):
        return pd.read_csv(capacity_dir / f"scenario_optimal_capacity_{group}_power_final_by_carrier.csv", index_col=0)["DAC"]
    captured = -energy("co2")
    electricity = -energy("electricity")
    np.testing.assert_allclose(captured, energy("co2_stored"), atol=1e-8)
    ratio = captured / electricity
    np.testing.assert_allclose(ratio, np.full(len(ratio), ratio.iloc[0]), rtol=1e-7)
    data = pd.DataFrame({
        "CO2_captured_Mt_per_year": captured,
        "electricity_TWh_per_year": electricity,
        "central_heat_TWh_per_year": -energy("urban_central_heat"),
        "decentral_heat_TWh_per_year": -energy("urban_decentral_heat"),
        "electricity_input_capacity_GW": nominal("electricity"),
        "central_DAC_input_capacity_GW": nominal("urban_central_heat"),
        "decentral_DAC_input_capacity_GW": nominal("urban_decentral_heat"),
        "capture_capacity_ktCO2_per_hour": nominal("electricity") * ratio,
    })
    np.testing.assert_allclose(data.electricity_input_capacity_GW,
                              data.central_DAC_input_capacity_GW + data.decentral_DAC_input_capacity_GW)
    data.to_csv(out / "DAC_detail.csv")
    x = np.arange(len(data))
    def draw(stem, panels):
        fig, axes = plt.subplots(len(panels), 1, figsize=(23, 4 * len(panels)), sharex=True)
        for ax, (columns, title, unit) in zip(np.atleast_1d(axes), panels):
            bottom = np.zeros(len(data))
            for i, (column, label) in enumerate(columns):
                ax.bar(x, data[column], bottom=bottom, color=["#ff5270", "#673b80"][i], label=label)
                bottom += data[column].to_numpy()
            ax.set_ylabel(unit)
            ax.set_title(title, fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        np.atleast_1d(axes)[-1].set_xticks(x, data.index.astype(str), rotation=45, ha="right")
        np.atleast_1d(axes)[-1].set_xlabel("Capacity-expansion solution")
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    draw("DAC_energy_balance", [
        ([("CO2_captured_Mt_per_year", "Atmospheric CO₂ captured")], "DAC: annual CO₂ captured", "MtCO₂/a"),
        ([("electricity_TWh_per_year", "Electricity consumed")], "DAC: electricity use", "TWh/a"),
        ([("central_heat_TWh_per_year", "Central heat"), ("decentral_heat_TWh_per_year", "Decentral heat")], "DAC: heat use", "TWh/a"),
    ])
    draw("DAC_optimal_capacity", [
        ([("central_DAC_input_capacity_GW", "DAC at central heat buses"), ("decentral_DAC_input_capacity_GW", "DAC at decentral heat buses")], "DAC: installed electricity-input capacity", "GW electricity input"),
        ([("capture_capacity_ktCO2_per_hour", "CO₂ capture capacity")], "DAC: installed CO₂ capture rate", "ktCO₂/h"),
    ])
    # Redraw the five existing sector plots containing DAC with a forced legend entry.
    raw = pd.read_csv(root / "csvs/capacity_expansion_optimal_capacity/capacities_long.csv", dtype={"scenario": str})
    raw["group"] = raw["group"].replace(capacity.GROUP_ALIASES)
    raw = raw[~raw.carrier.isin(capacity.EXCLUDED_CARRIERS)]
    raw["carrier"] = raw.carrier.map(lambda name: capacity.rename_techs(name, preserve_chp=True))
    labels = data.index.astype(str).tolist()
    capacity.SCENARIO_ORDER = labels
    capacity.FAMILY_GROUPS = {"all": labels}
    capacity.SHOW_FAMILY_LABELS = capacity.SHOW_FAMILY_SEPARATORS = False
    capacity.ALWAYS_LEGEND_CARRIERS = {"DAC"}
    config = capacity._load_yaml(capacity.PLOTTING_YAML)
    for (group, metric), records in raw.groupby(["group", "metric"]):
        if "DAC" in records.carrier.values:
            zeros = pd.DataFrame({"scenario": labels, "carrier": "DAC", "value": 0.0})
            capacity.plot_capacity(group, metric, pd.concat([records, zeros]), config, capacity_dir)
    print(f"[DONE] DAC detail plots and CSV: {out}")


if __name__ == "__main__":
    main()
