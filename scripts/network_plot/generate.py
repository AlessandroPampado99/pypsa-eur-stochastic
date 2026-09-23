#!/usr/bin/env python3
"""Render a verified structural schematic without solving or filtering capacities.

Run from any directory with the project's PyPSA environment. See README.md.
"""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pypsa
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INK = "#283846"
CARBON = "#936279"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def text(ax, x, y, s, size=7, **kw):
    return ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK),
                   va="center", **kw)


def box(ax, x, y, w, h, fill="white", edge="#cad1d7"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=.6))


def arrow(ax, a, b, carbon=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6,
                               lw=.7, color=CARBON if carbon else INK,
                               linestyle="--" if carbon else "-", shrinkA=0, shrinkB=0))


def canvas(height=238):
    fig = plt.figure(figsize=(180 / 25.4, height / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 180), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def save(fig, out, stem):
    check_bounds(fig)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=450, facecolor="white")


def check_bounds(fig):
    """Reject clipped text at the actual publication page size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.texts:
            bounds = item.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise ValueError(f"Text outside page: {item.get_text()}")


def audit_links(n):
    """Inspect every populated terminal, including time-dependent efficiencies.

    Injection is -p0 at bus0 and efficiency_i*p0 at every other port.
    Reverse-only links (notably heat pumps) require reversing those signs.
    """
    groups = {}
    for name, row in n.links.iterrows():
        lo = n.links_t.p_min_pu[name] if name in n.links_t.p_min_pu else pd.Series([row.p_min_pu])
        hi = n.links_t.p_max_pu[name] if name in n.links_t.p_max_pu else pd.Series([row.p_max_pu])
        reverse = hi.max() <= 0 and lo.min() < 0
        bidirectional = lo.min() < 0 and hi.max() > 0
        ports = []
        for col in sorted((c for c in n.links if c.startswith("bus") and c[3:].isdigit()), key=lambda c: int(c[3:])):
            bus = row[col]
            if not isinstance(bus, str) or not bus:
                continue
            i = int(col[3:])
            attr = "efficiency" if i == 1 else f"efficiency{i}"
            values = pd.Series([-1.]) if i == 0 else (
                n.links_t[attr][name] if name in n.links_t[attr] else pd.Series([row[attr]]))
            injections = values * (-1 if reverse else 1)
            a, b = float(injections.min()), float(injections.max())
            role = "input" if b < 0 else "output" if a > 0 else "zero" if a == b == 0 else "variable"
            if role == "variable":
                raise ValueError(f"Time-varying sign requires manual treatment: {name} {col}")
            ports.append({"port": col, "carrier": n.buses.at[bus, "carrier"],
                          "role": role, "injection_min": a, "injection_max": b})
        key = (row.carrier, reverse, bidirectional,
               tuple((p["port"], p["carrier"], p["role"]) for p in ports))
        if key not in groups:
            groups[key] = {"carrier": row.carrier, "reverse_only": bool(reverse),
                           "bidirectional": bool(bidirectional), "ports": ports,
                           "examples": [], "count": 0, "candidate_count": 0}
        g = groups[key]
        g["count"] += 1
        g["candidate_count"] += int(row.p_nom_extendable)
        if len(g["examples"]) < 2:
            g["examples"].append(name)
        for old, p in zip(g["ports"], ports):
            old["injection_min"] = min(old["injection_min"], p["injection_min"])
            old["injection_max"] = max(old["injection_max"], p["injection_max"])
    return list(groups.values())


def scenario_loads(n, catalogue):
    """Mark only selectors referenced by active catalogue definitions."""
    affected = set()
    entries = []
    for name, definition in catalogue["scenario_definitions"].items():
        if not isinstance(definition, dict) or "demand_transition" not in definition:
            continue
        action = definition["demand_transition"]
        family = catalogue["families"][action["family"]]
        for key in action["priority"]:
            entry = family["entries"][key]
            entries.append({"scenario": name, "entry": key, **entry})
            for side in ("source", "target"):
                selector = entry[side]
                if isinstance(selector, dict):
                    names = selector.get("name", [])
                    names = [names] if isinstance(names, str) else names
                    affected.update(n.loads.loc[n.loads.index.intersection(names), "carrier"])
                else:
                    affected.update([selector] if isinstance(selector, str) else selector)
    assert affected <= set(n.loads.carrier), affected - set(n.loads.carrier)
    return affected, entries


def supplement(groups, out):
    """A faceted process graph avoids a dense, illegible all-to-all graph."""
    per_page = 10
    with PdfPages(out / "system_supplement.pdf") as pdf:
        for page in range(math.ceil(len(groups) / per_page)):
            fig, ax = canvas(250)
            text(ax, 5, 7, f"S1 · Full link topology · {page + 1}", 11, weight="bold")
            text(ax, 5, 14, "All populated ports; repeated regional instances grouped. No capacity or dispatch filtering.", 7)
            text(ax, 5, 19, "Dashed purple: carbon accounting / CO2. Solid: energy. Repeated labels denote the same carrier.", 7)
            for i, g in enumerate(groups[page * per_page:(page + 1) * per_page]):
                y = 25 + 21 * i
                box(ax, 5, y, 170, 19, "#fcfcfd")
                ins = [p for p in g["ports"] if p["role"] == "input"]
                outs = [p for p in g["ports"] if p["role"] == "output"]
                box(ax, 61, y + 3, 58, 13, "#eaf0f5")
                label = "\n".join(textwrap.wrap(g["carrier"], 30))
                text(ax, 90, y + 8, label, 7.5, ha="center")
                text(ax, 90, y + 14, f"{g['count']} links; {g['candidate_count']} extendable", 6, ha="center")
                for side, ports in (("in", ins), ("out", outs)):
                    for j, p in enumerate(ports):
                        yy = y + 9.5 + (j - (len(ports) - 1) / 2) * 4.5
                        carbon = p["carrier"] in ("co2", "co2 stored", "co2 sequestered", "process emissions", "non-sequestered HVC")
                        label = p["carrier"].replace("electricity", "electric").replace("urban central", "central").replace("urban decentral", "decentral")
                        label = label.replace("methanol kerosene for aviation", "methanol-route kerosene")
                        text(ax, 8 if side == "in" else 128, yy, label, 6.6, color=CARBON if carbon else INK)
                        arrow(ax, (53, yy) if side == "in" else (119.5, yy),
                              (60.5, yy) if side == "in" else (127, yy), carbon)
                if g["bidirectional"]:
                    text(ax, 90, y + 18, "Reversible (ports shown for positive p0)", 5.5, ha="center")
            text(ax, 5, 239, "Air CO2 inputs to biomass routes encode biogenic carbon bookkeeping, not DAC energy supply.", 7)
            text(ax, 5, 244, "Heat-pump arrows use reverse-only bounds and time-dependent efficiency; co-inputs stay coupled.", 7)
            check_bounds(fig)
            pdf.savefig(fig)
            fig.savefig(out / f"system_supplement_{page + 1:02}.svg")
            fig.savefig(out / f"system_supplement_{page + 1:02}.png", dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=ROOT / "results/demand_uncertainty_2035/BASE/networks/base_s_adm___2035.nc")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = yaml.safe_load((HERE / "mapping.yaml").read_text())
    n = pypsa.Network(args.network)
    plt.rcParams["figure.autolayout"] = False
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    catalogue_path = ROOT / n.meta["stochastic_scenarios"]["file"]
    catalogue = yaml.safe_load(catalogue_path.read_text())
    affected, transitions = scenario_loads(n, catalogue)
    for c in mapping["carriers"].values():
        assert set(c["buses"]) <= set(n.buses.carrier)
    for s in mapping["sources"]:
        assert set(s["evidence"]) <= set(getattr(n, s["component"]).carrier)
    for d in mapping["demands"]:
        for r in d["rows"]:
            assert set(r["loads"]) <= set(n.loads.carrier)
    for p in mapping["processes"]:
        assert {p["carrier"], *p.get("extra", [])} <= set(n.links.carrier)
    groups = audit_links(n)
    args.output.joinpath("link_ports.json").write_text(json.dumps(groups, indent=2))
    args.output.joinpath("resolved_config.json").write_text(json.dumps(n.meta, indent=2))
    args.output.joinpath("uncertainty_selectors.json").write_text(json.dumps({"marked_load_carriers": sorted(affected), "transitions": transitions}, indent=2))
    for component in ("buses", "generators", "links", "stores", "storage_units", "loads", "lines"):
        df = getattr(n, component)
        cols = [c for c in df if c in ("carrier", "bus", "sign", "p_nom_extendable", "e_nom_extendable") or c.startswith("bus")]
        df[cols].to_csv(args.output / f"{component}_inventory.csv")
    with args.network.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    provenance = {"network": str(args.network.resolve()), "sha256": digest,
                  "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "pypsa_runtime": pypsa.__version__, "model_version_in_network": n.meta.get("version"),
                  "scenario_catalogue": str(catalogue_path), "catalogue_sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest(),
                  "links_inspected": len(n.links), "port_signatures": len(groups)}
    args.output.joinpath("provenance.json").write_text(json.dumps(provenance, indent=2))
    from integrated import render
    render(args.output)
    supplement(groups, args.output)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
