# Reference-network schematic

Run from the repository root using existing dependencies (PyPSA, pandas,
PyYAML, Matplotlib):

```bash
/home/pampado/miniconda3/envs/pypsa-eur/bin/python scripts/network_plot/generate.py
```

No workflow execution, downloads, optimisation or input modification occurs.
`--network PATH` and `--output DIRECTORY` override defaults. The mapping is
study-specific: another network must satisfy component assertions and requires
fresh scientific review. `mapping.yaml` separates verified component selectors and uncertainty groups
from rendering. `integrated.py` renders the deliberately simplified main
figure; `generate.py` performs the audit and renders the detailed supplement.

Outputs: `output/system_main.{pdf,svg,png}` (180 × 180 mm, 450 dpi PNG, editable
SVG text, embedded PDF fonts); multipage `system_supplement.pdf` with per-page
SVG and 300 dpi PNG; component CSV inventories; complete port-sign audit;
resolved metadata, uncertainty selectors and SHA-256 provenance. `figure.tex`
contains the English caption and LaTeX inclusion snippet. The supplement is a
faceted graph: repeated carrier labels connect process panels without crossings.

## Authoritative evidence

- Requested [reference network](../../results/demand_uncertainty_2035/BASE/networks/base_s_adm___2035.nc).
- [Saved resolved configuration](../../results/demand_uncertainty_2035/BASE/configs/config.base_s_adm___2035.yaml), consistent with network metadata: BASE, 2035, administrative clustering level 0, three-hour resolution, model version `v2025.07.0`.
- Current repository HEAD: `1887287f9800eb2d87ba47d4c424639c3a19795a`, described as `v2025.07.0-225-g1887287f`. This is NOT proof of the network's build commit. Inspection runtime: PyPSA 0.34.1.
- [Snakefile](../../Snakefile) loads default settings; study [deterministic settings](../../config/demand_uncertainty_2035/config_deterministic.yaml) and [run overrides](../../config/demand_uncertainty_2035/scenarios/scenarios_deterministic.yaml) identify the corresponding study. The current deterministic YAML is locally modified: CO2 budget 0.25 versus 0.30 in the reference network/saved config. `config_base.yaml` names `BASE_2032_lesstechs`, a different run. Network metadata takes precedence. Existing modifications were preserved.
- No AGENTS.md was found in the repository or ancestor directories. The general [README](../../README.md) describes a broader technology menu than this network.

Construction: [prepare_sector_network rule](../../rules/build_sector.smk#L1565)
uses [prepare_sector_network.py](../prepare_sector_network.py). Scenario
preprocessing is selected in [solve_overnight.smk](../../rules/solve_overnight.smk#L42),
including deterministic active scenarios; BASE is a no-op.

## Connections and terminal verification

All 3,508 links are inspected, including every populated bus port, temporal
efficiencies and operating bounds; they yield 92 port signatures. For positive
p0, bus0 withdraws p0 and other ports inject efficiency_i × p0. Negative
coefficients denote co-inputs. Reverse-only links reverse the orientation.
Heat pumps use reverse-only bounds and time-dependent inverse COPs: their
arrows correctly run from electricity to heat despite reversed bus order.
No capacity or dispatch thresholds enter selection. Candidate counts are
reported independently; all installed and candidate components remain eligible.

| Displayed connection | Network / code evidence |
|---|---|
| Sources → electricity and heat | Wind, PV, run-of-river, nuclear generators; reservoir hydro StorageUnits; solar-thermal generators. Nuclear has no explicit uranium bus. |
| Fossil / biomass supply | Gas, oil-primary, coal, solid-biomass, biogas and unsustainable biomass/biogas/bioliquid generators. Oil refining and bioliquids-to-oil links. The coal arrow supplies only coal, not ammonia. |
| Electricity → hydrogen + heat | `H2 Electrolysis`: AC input, H2 and central-heat outputs; `add_h2_gas_infrastructure` and `add_waste_heat`. |
| Hydrogen → power / heat | `H2 Fuel Cell`: joint electricity and heat; `H2 turbine`: electricity only. |
| Hydrogen + electricity + CO2 → methanol + heat | `methanolisation`: negative electricity and captured-CO2 ports, positive methanol and heat ports; construction around lines 4910–4940, `add_waste_heat`. |
| Methanol → H2 / power | `Methanol steam reforming`, its CC variant, `OCGT methanol`; `add_methanol_reforming` at line 1163 and CC at 1186. |
| Methanol + H2 → aviation kerosene | `methanol-to-kerosene`: H2 co-input, dedicated aviation product bus and atmospheric carbon output. It does not feed the general oil pool. |
| Gas → H2; H2 + CO2 → gas | `SMR`, `SMR CC`, `Sabatier`; Sabatier also produces heat. |
| H2 + CO2 → liquids + heat | `Fischer-Tropsch`, including separate copies feeding `oil from H2`. |
| Biomass + H2 → liquids | `electrobiofuels`; atmospheric CO2 input is biogenic bookkeeping, not a DAC requirement. Biomass-to-liquid and biomass-to-methanol also exist. |
| H2 + electricity → ammonia + heat; ammonia → H2 | `Haber-Bosch`, `ammonia cracker`; EU ammonia store and NH3 industry Load. Water/nitrogen are not explicit buses and are not invented. |
| Other power / heat | Gas turbines, gas/biomass boilers and CHP, capture variants, heat pumps and resistive heaters. CHP co-products stay coupled. |
| Carriers → end-use sectors | Actual Load selectors in `mapping.yaml` and `loads_inventory.csv`; upstream conversion/emissions links in S1. EV charging/V2G and H2 liquefaction are retained there. |
| Carbon management | DAC consumes electricity, heat and atmospheric CO2. Capture at gas/methanol reforming, gas/biomass CHP, industrial gas/biomass use and process emissions. CO2 pipelines and sequestration links/stores exist. |

`output/link_ports.json` records exact component examples, terminal carriers,
directions and coefficient ranges. S1 renders every signature as one process
with all co-inputs/products; it does not turn each terminal into an independent
conversion option. Carbon arrows are dashed purple. Atmospheric `co2` is
separate from captured `co2 stored`. Biogenic atmospheric inputs are carbon
accounting, not an energy source. S1 includes residual emissions and
non-sequestered HVC accounting, omitted from the main figure for clarity.

Spatial/storage distinctions: regional AC/DC/distribution, gas and H2 networks;
regional heat categories and tank stores, with pits for central heat; EU
methanol supply/store with regional end-use buses; EU oil, dedicated H2-oil,
ammonia and coal pools. No methanol grid is inferred. Biomass is regional
without transport. Reservoir/pumped hydro and stationary/home/EV batteries
are verified in component tables. Ammonia and coal remain explicit in the industry inlet label; their separate
EU buses and conversion links are documented in S1 and the inventories.

Disabled/absent alternatives are excluded: H2-retrofitted pipes, biomass
transport, BioSNG, biomass-to-H2, municipal-waste pathways, biomass-to-methanol
CC, biomass-to-liquid CC, biogas-upgrading CC, methanol CCGT/Allam, energy
import links and enhanced geothermal. Heat vents are sinks, not sources;
they remain in inventories rather than the main source panel.

## Demand uncertainty and custom adaptations

Markers are derived from active source/target selectors in the current
[scenario catalogue](../../config/demand_uncertainty_2035/scenarios/scenarios_determinstic_definition.yaml).
[_apply_shift_like_entry](../stochasticify_network.py#L1355) reduces source
Load profiles and increases target profiles using imposed efficiency factors,
priority order, caps and family targets before optimisation. Markers indicate
categories eligible for shifts, not that every load changes in BASE or every
scenario. Applied amounts in other solved scenarios were not reconstructed.

Only split gas/electric building heat is marked, not baseline electricity or
residual heat. Industrial electricity/gas/biomass/H2/methanol/naphtha are
marked, not coal/ammonia/low-temperature heat. Road and shipping fuels,
aviation kerosene routes and agricultural machinery energy are marked; other
agricultural electricity/heat is not. No upstream conversion is marked uncertain.

Custom heat splitting: `add_heat`, lines 2942–3000.
[add_oil_from_h2_demand](../prepare_sector_network.py#L5882) reserves 1% of BASE
fixed oil demand (road, naphtha, aviation, shipping, agricultural machinery),
reduces original loads and duplicates FT/electrobiofuel links and storage onto
a separate bus and Load. It has no return connection to the general oil pool.
The `H2.oil_from_H2` scenario entry shifts road, shipping and agricultural oil
demand to that dedicated load. Its size is externally imposed; the choice
between eligible production technologies remains endogenous. The main L*
annotation represents this pooled demand rather than inventing sector outlets.

## Verification limits

The network proves component structure. Current code/catalogue explain it but
are not guaranteed byte-identical to the historical build. The exact build
commit and historical scenario hash are unavailable; current hashes and HEAD
are separately recorded in `provenance.json`. Current selectors are checked
against network loads; no historical scenario shift magnitude is claimed.

PyPSA import emits a local PROJ diagnostic but loads successfully; this script
performs no geographic transformations. Generation and visual review cover
the main figure and supplementary multi-terminal processes. Every page has
text-bound checks at its actual 180 mm width. Main labels are 7–8 pt and minor
annotations 6.4–6.8 pt. Use full two-column width. Original inputs and results
are unchanged.

The main figure was simplified following author feedback: a single integrated
view replaces the earlier process catalogue. The central frame represents the
energy system as a whole; arrows leaving its boundary carry only the carrier
bundles explicitly listed in the sector boxes, not every carrier. It omits
secondary conversion routes and carbon coproducts; S1 retains them. Liquids
aggregate general oil, dedicated H2-oil and methanol-route aviation pools
without asserting interchangeability. The methanol-to-kerosene arrow explicitly
requires additional hydrogen. The refinery arrow carries oil, the gas arrow gas;
coal demand remains explicit in industry. Grid and storage annotations are
compact summaries, with spatial distinctions documented above.
