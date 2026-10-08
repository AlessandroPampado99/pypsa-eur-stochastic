#!/usr/bin/env python3
"""Plot CSSC scenario probabilities and spatially averaged weather availability."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import xarray as xr

TECHS = ["Solar", "Onshore wind", "Offshore wind", "Hydro"]


def save(fig, output):
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def probabilities(root, cases, out):
    rows = []
    introduction_order = {}
    for case in sorted(cases, key=lambda case: int(case.split("K")[-1])):
        k = int(case.split("K")[-1])
        path = root / f"analysis_output/cssc/{k}/cssc_K{k}_representatives.csv"
        frame = pd.read_csv(path).sort_values("representative")
        if len(frame) != k or not np.isclose(frame.probability.sum(), 1):
            raise ValueError(f"Invalid scenario probabilities: {path}")
        for year in frame.representative:
            if year not in introduction_order:
                introduction_order[year] = len(introduction_order)
        frame["introduction_order"] = frame.representative.map(introduction_order)
        frame = frame.sort_values("introduction_order")
        frame["case"] = case
        rows.append(frame)
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(out / "stochastic_scenario_probabilities.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = draw_probabilities(ax, rows)
    fig.colorbar(im, ax=ax, label="Scenario probability", format="%.2f")
    save(fig, out / "stochastic_scenario_probabilities")


def draw_probabilities(ax, rows):
    """Use the same matrix and annotations in standalone and combined figures."""
    matrix = np.full((len(rows), max(map(len, rows))), np.nan)
    for i, frame in enumerate(rows):
        matrix[i, : len(frame)] = frame.probability
    norm = Normalize(0, np.nanmax(matrix))
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for i, frame in enumerate(rows):
        for j, row in enumerate(frame.itertuples()):
            year = row.representative.removeprefix("d_")
            ax.text(
                j,
                i,
                f"{year}\n{row.probability:.1%}",
                ha="center",
                va="center",
                color="white" if norm(row.probability) > 0.55 else "black",
                fontsize=10,
            )
    ax.set_yticks(range(len(rows)), [r.case.iloc[0] for r in rows])
    ax.set_xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
    ax.set_xlabel("Selected scenario (ordered by first appearance as K increases)")
    ax.set_ylabel("Stochastic case")
    ax.set_title("Scenario years and probabilities by stochastic case")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def probabilities_with_profiles(out, *, show_mean=False):
    """Align each matrix column with its year’s four spatial-mean CF profiles."""
    data = pd.read_csv(out / "stochastic_scenario_probabilities.csv")
    cases = sorted(data.case.unique(), key=lambda case: int(case.split("K")[-1]))
    rows = [
        data.loc[data.case == case].sort_values("introduction_order") for case in cases
    ]
    selected = rows[-1].representative.tolist()
    # A single profile per column is meaningful only if the years align in every row.
    for row in rows:
        if row.representative.tolist() != selected[: len(row)]:
            raise ValueError(
                "Scenario sets must be nested to align profiles under matrix columns."
            )
    years = [int(year.removeprefix("d_")) for year in selected]
    series = pd.read_csv(
        out / "deterministic_capacity_factors_timeseries.csv",
        parse_dates=["timestamp"],
        index_col=["weather_year", "timestamp"],
    )
    frames = {year: series.xs(year).sort_index() for year in years}
    if show_mean:
        # These annual means retain the network snapshot-duration weighting.
        means = pd.read_csv(
            out / "deterministic_capacity_factors_mean.csv", index_col=0
        )
        if not np.isfinite(means.loc[TECHS, list(map(str, years))].to_numpy()).all():
            raise ValueError("Missing annual mean capacity factors for selected years.")
    if any(frame[TECHS].isna().any().any() for frame in frames.values()):
        raise ValueError("Missing capacity-factor values for selected years.")
    fig = plt.figure(figsize=(20, 14), layout="none")
    # Zero column spacing aligns the profile panel edges exactly with matrix cells.
    grid = fig.add_gridspec(
        5,
        len(years),
        height_ratios=[5.4, 1, 1, 1, 1],
        left=0.09,
        right=0.91,
        bottom=0.06,
        top=0.94,
        hspace=0.42,
        wspace=0,
    )
    ax = fig.add_subplot(grid[0, :])
    im = draw_probabilities(ax, rows)
    ax.set_xticks(range(len(years)), years)
    ax.set_xlabel("Scenario year (ordered by first appearance as K increases)")
    position = ax.get_position()
    color_ax = fig.add_axes([0.925, position.y0, 0.012, position.height])
    fig.colorbar(im, cax=color_ax, label="Scenario probability", format="%.2f")
    colors = ["#d99b13", "#287d8e", "#4664b0", "#469760"]
    for t, tech in enumerate(TECHS):
        limit = max(frame[tech].max() for frame in frames.values()) * 1.10
        for col, year in enumerate(years):
            profile_ax = fig.add_subplot(grid[t + 1, col])
            raw = frames[year][tech]
            daily = raw.resample("D").mean()
            profile_ax.plot(
                raw.index.dayofyear + raw.index.hour / 24,
                raw,
                color=colors[t],
                alpha=0.18,
                lw=0.25,
            )
            profile_ax.plot(daily.index.dayofyear, daily, color=colors[t], lw=0.8)
            if show_mean:
                mean = means.loc[tech, str(year)]
                profile_ax.axhline(mean, color="black", linestyle="--", lw=1, zorder=4)
                profile_ax.text(
                    0.96,
                    0.94,
                    f"Mean {mean:.3f}",
                    transform=profile_ax.transAxes,
                    ha="right",
                    va="top",
                    color="black",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1,
                    },
                    zorder=5,
                )
            profile_ax.set_xlim(-12, 379)
            profile_ax.set_ylim(0, limit)
            profile_ax.set_xticks([15, 182, 350], ["Jan", "Jul", "Dec"])
            profile_ax.tick_params(labelsize=8, axis="both", length=2)
            profile_ax.spines[["top", "right"]].set_visible(False)
            profile_ax.spines["left"].set_alpha(0.2)
            profile_ax.grid(alpha=0.15)
            if col == 0:
                profile_ax.set_ylabel(f"{tech}\nCF", fontsize=10)
            else:
                profile_ax.tick_params(labelleft=False)
            if t == 0:
                profile_ax.set_title(str(year), fontsize=11)
    fig.text(
        0.5,
        0.015,
        "Spatial mean capacity factors — faint: model snapshots; solid: daily mean. "
        "Hydro: natural availability including reservoir inflow."
        + (
            "\nBlack dashed line and label: annual mean capacity factor (snapshot-duration weighted)."
            if show_mean
            else ""
        ),
        ha="center",
        fontsize=10,
    )
    name = "stochastic_scenario_probabilities_with_capacity_factors"
    save(fig, out / (name + "_with_means" if show_mean else name))


def spatial_profiles(path):
    """Equal bus means; combine hydro resources by nominal capacity within bus."""
    with xr.open_dataset(path) as ds:
        gen = pd.DataFrame(
            {
                key: ds[f"generators_{key}"].to_series()
                for key in ("carrier", "bus", "p_nom")
            }
        )
        profiles = ds.generators_t_p_max_pu.to_pandas()
        profiles.index = pd.DatetimeIndex(ds.snapshots_snapshot.values)
        result = {}
        for tech, carriers in [
            ("Solar", ["solar"]),
            ("Onshore wind", ["onwind"]),
            ("Offshore wind", ["offwind-ac", "offwind-dc", "offwind-float"]),
        ]:
            selected = gen.index[gen.carrier.isin(carriers)].intersection(
                profiles.columns
            )
            if selected.empty:
                raise ValueError(f"No {tech} profiles in {path}")
            # Average offshore variants within each bus before averaging buses.
            result[tech] = (
                profiles[selected].T.groupby(gen.loc[selected, "bus"]).mean().mean()
            )
        ror = gen.loc[(gen.carrier == "ror") & (gen.p_nom > 0)]
        ror_power = profiles[ror.index].mul(ror.p_nom).T.groupby(ror.bus).sum().T
        units = pd.DataFrame(
            {
                key: ds[f"storage_units_{key}"].to_series()
                for key in ("carrier", "bus", "p_nom")
            }
        )
        hydro = units.loc[(units.carrier == "hydro") & (units.p_nom > 0)]
        inflow = ds.storage_units_t_inflow.to_pandas()[hydro.index]
        inflow.index = profiles.index
        power = ror_power.add(inflow.T.groupby(hydro.bus).sum().T, fill_value=0)
        capacity = (
            ror.groupby("bus")
            .p_nom.sum()
            .add(hydro.groupby("bus").p_nom.sum(), fill_value=0)
        )
        result["Hydro"] = power.div(capacity).mean(axis=1)
        frame = pd.DataFrame(result)
        weights = ds.snapshots_generators.values
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Missing/nonfinite capacity factors in {path}")
        annual = pd.Series(
            np.average(frame, axis=0, weights=weights), index=frame.columns
        )
    return frame, annual


def weather(root, years, out):
    annual, series = {}, {}
    sources = []
    for year in years:
        path = root / f"d_{year}/networks/base_s_adm___2050.nc"
        frame, annual[year] = spatial_profiles(path)
        series[year] = frame
        sources.append(str(path))
        print(f"Read capacity factors: {year}", flush=True)
    means = pd.DataFrame(annual).reindex(TECHS)
    means.to_csv(
        out / "deterministic_capacity_factors_mean.csv", index_label="technology"
    )
    pd.concat(series, names=["weather_year", "timestamp"]).to_csv(
        out / "deterministic_capacity_factors_timeseries.csv"
    )
    fig, ax = plt.subplots(figsize=(19, 4.5), constrained_layout=True)
    im = ax.imshow(means, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(years)), years, rotation=90)
    ax.set_yticks(range(len(TECHS)), TECHS)
    for i in range(len(TECHS)):
        for j in range(len(years)):
            v = means.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if im.norm(v) > 0.55 else "black",
            )
    ax.set_xlabel("Deterministic weather year")
    ax.set_title("Annual mean capacity factor — equal spatial weighting")
    fig.colorbar(im, ax=ax, label="Capacity factor / hydro inflow per unit capacity")
    save(fig, out / "deterministic_capacity_factors_mean")

    # Ten year columns per block keep the annual time series legible on export.
    blocks = (len(years) + 9) // 10
    fig, axes = plt.subplots(
        4 * blocks,
        min(10, len(years)),
        figsize=(22, 6 * blocks),
        squeeze=False,
        constrained_layout=True,
    )
    colors = ["#d99b13", "#287d8e", "#4664b0", "#469760"]
    for position, year in enumerate(years):
        block, col = divmod(position, 10)
        frame = series[year]
        for t, tech in enumerate(TECHS):
            ax = axes[block * 4 + t, col]
            raw = frame[tech]
            daily = raw.resample("D").mean()
            ax.plot(
                raw.index.dayofyear + raw.index.hour / 24,
                raw,
                color=colors[t],
                alpha=0.18,
                lw=0.25,
            )
            ax.plot(daily.index.dayofyear, daily, color=colors[t], lw=0.7)
            limit = max(f[tech].max() for f in series.values())
            ax.set_ylim(0, max(1, limit) * 1.02)
            ax.set_xlim(1, 366)
            ax.set_xticks([1, 182, 365], ["Jan", "Jul", "Dec"])
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if t == 0:
                ax.set_title(str(year), fontsize=10)
            if col == 0:
                ax.set_ylabel(tech, fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            ax.grid(alpha=0.15)
    for position in range(len(years), blocks * axes.shape[1]):
        block, col = divmod(position, 10)
        for t in range(4):
            axes[block * 4 + t, col].set_visible(False)
    fig.suptitle(
        "Spatial mean capacity factor through each year\nFaint: model snapshots; solid: daily mean",
        fontsize=15,
    )
    save(fig, out / "deterministic_capacity_factors_timeseries")
    (out / "weather_plot_method.txt").write_text(
        "Solar: fixed-tilt solar carrier. Onshore: onwind. Offshore: mean of AC, DC and floating profiles within each bus.\n"
        "Spatial weighting: equal weight per available model bus, without optimized-capacity weighting.\n"
        "Hydro: run-of-river available power plus reservoir natural inflow, divided by their combined nominal capacity within each bus, then equal mean across buses. Excludes pumped storage.\n"
        "Hydro is a resource-availability proxy, not dispatched generation; reservoir inflow per unit capacity can exceed one and is not clipped.\n"
        "Annual means use generator snapshot-duration weights. Time-series CSV preserves model snapshots; plots also show daily means.\n"
        "Scenario matrix rows are stochastic cases; selected years are ordered by first appearance as K increases (chronological tie-break for years introduced together); probabilities come from CSSC representative CSVs.\n\nSources:\n"
        + "\n".join(sources)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/cutouts_det_capexp_nols")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.results_root
    out = (
        args.output_dir
        or root / "analysis_output/validation_heatmaps/cost_distributions"
    )
    out.mkdir(parents=True, exist_ok=True)
    costs = pd.read_excel(
        root / "analysis_output/validation_heatmaps/validation_heatmaps.xlsx",
        sheet_name="total_cost",
        index_col=0,
    )
    cases = sorted(
        (s for s in costs.index if s.startswith("stochastic_K")),
        key=lambda s: int(s.split("K")[-1]),
    )
    years = sorted(int(s[2:]) for s in costs.index if s.startswith("d_"))
    probabilities(root, cases, out)
    weather(root, years, out)
    probabilities_with_profiles(out)
    probabilities_with_profiles(out, show_mean=True)
    print(f"Written weather plots to {out}")


if __name__ == "__main__":
    main()
