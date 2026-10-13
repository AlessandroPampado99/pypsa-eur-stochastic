#!/usr/bin/env python3
"""Quantify existing E->F solutions without reoptimisation; write a LaTeX report."""
from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_validation_heatmaps import _get_snapshot_weightings, _scenario_display_name
from plot_validation_diagnostics import load_shedding_decomposition

ROOT = Path("results/demand_uncertainty_2035")
PARENT = ROOT / "analysis_output/validation_heatmaps/diagnostics/family_cost_explanation"
OUT = PARENT / "fuel_persistence_diagnosis"
E = ["ELEC_HEAT", "ELEC_IND", "ELEC_ROAD"]
F = ["OILGAS_HEAT", "OILGAS_IND", "OILGAS_TRANS"]


def nominal(table, attr):
    return table[attr].where(~table[f"{attr}_extendable"], table[f"{attr}_opt"])


def capacities(n, capacity, operation):
    rows = []
    for comp, attr in (("links", "p_nom"), ("generators", "p_nom")):
        table = getattr(n, comp)
        values = nominal(table, attr)
        keep = table.carrier.ne("load") & np.isfinite(values)
        for carrier, value in values[keep].groupby(table.loc[keep, "carrier"]).sum().items():
            rows.append(dict(capacity=capacity, operation=operation, component=comp,
                             carrier=carrier, nominal_GW=value/1e3))
    return rows


def globals_row(n, capacity, operation):
    weights = _get_snapshot_weightings(n)
    emitted = n.carriers.index[n.carriers.co2_emissions.ne(0)]
    store_bus_carrier = n.stores.bus.map(n.buses.carrier)
    ids = n.stores.index[store_bus_carrier.isin(emitted) & ~n.stores.e_cyclic]
    final_emissions = n.stores_t.e.loc[n.snapshots[-1], ids].sum()
    glc = n.global_constraints.loc["CO2Limit"]
    sequestration = n.stores.index[n.stores.carrier.eq("co2 sequestered")]
    # Mirrors PyPSA operational_limit: initial energy minus terminal energy for noncyclic stores.
    sequestration = sequestration[~n.stores.loc[sequestration, "e_cyclic"]]
    seq = (n.stores_t.e.loc[n.snapshots[-1], sequestration] - n.stores.loc[sequestration, "e_initial"]).sum()
    seq_constraint = n.global_constraints.loc["co2_sequestration_limit"]
    row = dict(capacity=capacity, operation=operation,
               case=f"{_scenario_display_name(capacity)} → {_scenario_display_name(operation)}",
               slack_TWh=load_shedding_decomposition(n).sum(),
               co2_budget_Mt=glc.constant/1e6, co2_used_Mt=final_emissions/1e6,
               co2_headroom_t=glc.constant-final_emissions,
               co2_shadow_EUR_per_t=-float(glc.mu),
               sequestration_Mt=seq/1e6, sequestration_limit_Mt=-seq_constraint.constant/1e6,
               sequestration_shadow_EUR_per_t=float(seq_constraint.mu))
    coal = n.generators.index[n.generators.carrier.eq("coal")]
    coalcap = nominal(n.generators, "p_nom").loc[coal]
    coalavail = n.get_switchable_as_dense("Generator", "p_max_pu").loc[:, coal].mul(coalcap, axis=1).sum(axis=1)
    coalp = n.generators_t.p.loc[:, coal].sum(axis=1)
    slack_ids = n.generators.index[n.generators.carrier.eq("load")]
    slack = n.generators_t.p.loc[:, slack_ids]
    bus_carrier = n.generators.loc[slack_ids, "bus"].map(n.buses.carrier)
    coal_slack = slack.loc[:, bus_carrier.eq("coal for industry")].sum(axis=1)
    active = coal_slack > 1e-3
    row["coal_capacity_GW"] = coalcap.sum()/1e3
    row["coal_peak_dispatch_GW"] = coalp.max()/1e3
    row["coal_mean_utilisation_during_coal_slack"] = float((coalp[active]/coalavail[active].replace(0,np.nan)).mean())
    row["coal_saturated_weight_share_during_coal_slack"] = float(weights[active & (coalp >= .99*coalavail)].sum()/weights[active].sum()) if active.any() else np.nan
    row["coal_slack_TWh"] = float(coal_slack.mul(weights).sum()/1e6)
    delivery = n.links.index[n.links.carrier.eq("coal for industry")]
    link = delivery[0]
    carbon_ports = [f"bus{i}" for i in range(1,5) if f"bus{i}" in n.links and n.links.at[link,f"bus{i}"] == "co2 atmosphere"]
    emission_factor = sum(n.links.at[link,f"efficiency{port[3:]}"] for port in carbon_ports)
    row["coal_emission_factor_t_per_MWh"] = emission_factor/n.links.at[link,"efficiency"]
    row["coal_CO2_opportunity_cost_EUR_per_MWh"] = row["co2_shadow_EUR_per_t"]*row["coal_emission_factor_t_per_MWh"]
    row["coal_delivery_extendable"] = bool(n.links.at[link,"p_nom_extendable"])
    # Renewable or conversion saturation conditioned on slack at the asset's output bus.
    slack_by_bus = slack.T.groupby(n.generators.loc[slack_ids,"bus"]).sum().T
    assets = []
    for comp, ts, port in (("generators", "p", "bus"), ("links", "p0", "bus1")):
        table = getattr(n,comp)
        cap = nominal(table,"p_nom")
        fixed = ~table.p_nom_extendable
        eligible = fixed & table.carrier.ne("load") & (cap > 1e-6) & np.isfinite(cap)
        eligible &= table[port].isin(slack_by_bus.columns)
        ids = table.index[eligible]
        if not len(ids):
            continue
        dispatch = getattr(getattr(n,comp+"_t"),ts).loc[:,ids]
        upper = n.get_switchable_as_dense("Generator" if comp=="generators" else "Link","p_max_pu").loc[:,ids].mul(cap[ids],axis=1)
        for asset in ids:
            local = slack_by_bus[table.at[asset,port]]
            available = upper[asset] > 1e-6
            shedding = local > 1e-3
            hit = available & (dispatch[asset] >= .99*upper[asset])
            denom = local[shedding].mul(weights[shedding]).sum()
            assets.append(dict(capacity=capacity,operation=operation,component=comp,
                carrier=table.at[asset,"carrier"],asset=asset,output_bus=table.at[asset,port],
                nominal_GW=cap[asset]/1e3,local_slack_TWh=denom/1e6,
                slack_energy_share_with_asset_saturated=float(local[shedding & hit].mul(weights[shedding & hit]).sum()/denom) if denom>0 else np.nan,
                saturation_weight_share_during_local_slack=float(weights[shedding & hit].sum()/weights[shedding].sum()) if shedding.any() else np.nan))
    return row, assets


def tex_table(frame, caption, label):
    return ("\\begin{table}[htbp]\n\\centering\n\\small\n"+
            "\\resizebox{\\linewidth}{!}{%\n"+
            frame.to_latex(index=False,escape=True,na_rep="--",float_format=lambda x:f"{x:,.2f}")+"}\n"+
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    globs, assets, caps = [], [], []
    cases = [] if "--render-only" in sys.argv else [(e,f) for e in E for f in F] + [(f,f) for f in F]
    for capacity, operation in cases:
        print(f"Analysing {capacity} -> {operation}",flush=True)
        filename = "base_s_adm___2035.nc" if capacity==operation else f"base_s_adm___2035__cap-{capacity}__op-{operation}.nc"
        n=pypsa.Network(ROOT/capacity/"networks"/filename)
        row, details=globals_row(n,capacity,operation)
        globs.append(row); assets.extend(details); caps.extend(capacities(n,capacity,operation))
    if "--render-only" in sys.argv:
        g=pd.read_csv(OUT/"global_and_coal_diagnostics.csv")
        a=pd.read_csv(OUT/"fixed_assets_conditional_saturation.csv")
        c=pd.read_csv(OUT/"technology_capacity_by_case.csv")
    else:
        g=pd.DataFrame(globs); a=pd.DataFrame(assets); c=pd.DataFrame(caps)
    g.to_csv(OUT/"global_and_coal_diagnostics.csv",index=False)
    a.to_csv(OUT/"fixed_assets_conditional_saturation.csv",index=False)
    c.to_csv(OUT/"technology_capacity_by_case.csv",index=False)
    refs=c[c.capacity.eq(c.operation)].set_index(["operation","component","carrier"]).nominal_GW
    cross=c[~c.capacity.eq(c.operation)].copy()
    cross["reference_GW"]=[refs.get((r.operation,r.component,r.carrier),0) for r in cross.itertuples()]
    cross["difference_GW"]=cross.nominal_GW-cross.reference_GW
    cross["ratio_to_reference"]=cross.nominal_GW/cross.reference_GW.replace(0,np.nan)
    cross.to_csv(OUT/"capacity_comparison_to_F_optimum.csv",index=False)
    mean=cross.groupby(["component","carrier"])[["nominal_GW","reference_GW","difference_GW"]].mean()
    mean.to_csv(OUT/"mean_capacity_comparison.csv")
    ef=g[~g.capacity.eq(g.operation)].copy()
    # Do not interpret utilisation conditional on numerically negligible coal slack.
    ef.loc[ef.coal_slack_TWh < 1e-3,"coal_mean_utilisation_during_coal_slack"]=np.nan
    view=ef[["case","slack_TWh","co2_used_Mt","co2_shadow_EUR_per_t","sequestration_Mt","coal_mean_utilisation_during_coal_slack"]].copy()
    view.iloc[:,0]=view.iloc[:,0].str.replace(" → "," / ")
    view.columns=["Test","Slack (TWh)","CO2 (Mt)","CO2 dual (EUR/t)","Storage (Mt)","Coal use"]
    coalview=ef[["case","coal_slack_TWh","coal_capacity_GW","coal_peak_dispatch_GW","coal_CO2_opportunity_cost_EUR_per_MWh"]].copy()
    coalview.iloc[:,0]=coalview.iloc[:,0].str.replace(" → "," / ")
    coalview.columns=["Test","Coal slack (TWh)","Coal cap. (GW)","Coal peak (GW)","CO2 cost (EUR/MWh)"]
    selected=["DAC","SMR CC","SMR","Fischer-Tropsch","oil refining","urban central gas CHP CC",
              "urban central gas boiler","urban decentral gas boiler","rural gas boiler","gas for industry CC",
              "solid biomass for industry CC","process emissions CC","H2 Electrolysis","coal"]
    comparison=mean.loc[mean.index.get_level_values("carrier").isin(selected)].reset_index()
    comparison.columns=["Component","Technology","E capacity (GW)","F reference (GW)","Difference (GW)"]
    comparison.to_csv(OUT/"selected_capacity_comparison.csv",index=False)
    # Match individual fixed assets to each operations scenario's diagonal optimum.
    reference_assets = {}
    for f in F:
        nref=pypsa.Network()
        nref.import_from_netcdf(ROOT/f/"networks/base_s_adm___2035.nc",skip_time=True)
        for comp in ("links","generators"):
            refvalues=nominal(getattr(nref,comp),"p_nom")/1e3
            reference_assets.update({(f,comp,asset):value for asset,value in refvalues.items()})
    a["reference_GW"]=[reference_assets.get((r.operation,r.component,r.asset),np.nan) for r in a.itertuples()]
    a.to_csv(OUT/"fixed_assets_conditional_saturation.csv",index=False)
    boilers=a[(~a.capacity.eq(a.operation)) & a.asset.str.endswith("gas-only") & a.carrier.str.contains("gas boiler")].copy()
    boiler_rows=[]
    for (cap,op,carrier), block in boilers.groupby(["capacity","operation","carrier"]):
        if not block.output_bus.is_unique:
            raise ValueError("Gas-only boiler outputs overlap; cannot sum local slack")
        energy=block.local_slack_TWh.sum()
        boiler_rows.append(dict(capacity=cap,operation=op,carrier=carrier,
            capacity_GW=block.nominal_GW.sum(),reference_GW=block.reference_GW.sum(),
            local_slack_TWh=energy,
            slack_share_with_boiler_saturated=(block.local_slack_TWh*block.slack_energy_share_with_asset_saturated.fillna(0)).sum()/energy if energy>1e-6 else np.nan))
    boilersummary=pd.DataFrame(boiler_rows)
    boilersummary.to_csv(OUT/"gas_only_boiler_diagnosis.csv",index=False)
    heat=boilersummary[(boilersummary.capacity=="ELEC_HEAT") & (boilersummary.operation=="OILGAS_HEAT")].copy()
    heat["slack_share_with_boiler_saturated"]*=100
    heat=heat[["carrier","capacity_GW","reference_GW","local_slack_TWh","slack_share_with_boiler_saturated"]]
    heat.columns=["Technology","E-HEAT (GW)","F-HEAT (GW)","Local slack (TWh)","Saturation share (%)"]
    plotted_heat=heat[heat["Local slack (TWh)"]>1].copy()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    x=np.arange(len(plotted_heat))
    axes[0].bar(x-.18,plotted_heat["E-HEAT (GW)"],width=.36,label="E-HEAT capacities")
    axes[0].bar(x+.18,plotted_heat["F-HEAT (GW)"],width=.36,label="F-HEAT own optimum")
    axes[0].set_ylabel("Gas-only boiler input capacity [GW]")
    axes[0].legend(fontsize=8)
    bars=axes[1].bar(x,plotted_heat["Local slack (TWh)"],color="#b45849")
    axes[1].bar_label(bars,fmt="%.1f")
    axes[1].set_ylim(0,plotted_heat["Local slack (TWh)"].max()*1.2)
    axes[1].set_ylabel("Local load slack [TWh/year]")
    for ax in axes:
        ax.set_xticks(x,["Urban central", "Urban decentral"],fontweight="bold")
    fig.suptitle("E-HEAT capacities tested on F-HEAT demand")
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"heat_boiler_diagnosis.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)
    ecapacity=cross.groupby(["capacity","carrier"])[["nominal_GW","reference_GW"]].mean()
    ecapacity.to_csv(OUT/"capacity_comparison_by_E_scenario.csv")
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    labels=ef.case
    axes[0].barh(labels,ef.co2_shadow_EUR_per_t/1e3,color="#a64b3c")
    axes[0].set_xlabel("CO2 shadow value [thousand EUR/t]")
    axes[1].barh(labels,ef.coal_mean_utilisation_during_coal_slack*100,color="#547a94")
    axes[1].set_xlabel("Coal supply utilisation during coal slack [%]")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.2)
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"carbon_and_coal_diagnosis.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)
    # Copy report figures so the TeX file can travel with its adjacent PDFs.
    for stem in ("mean_cost_by_capacity_family","mean_cost_by_operation_family","carrier_slack_by_family_pair"):
        shutil.copy2(PARENT/f"{stem}.pdf",OUT/f"{stem}.pdf")
    binding=np.isclose(ef.co2_used_Mt,ef.co2_budget_Mt,atol=1e-3,rtol=1e-7).sum()
    seqbinding=np.isclose(ef.sequestration_Mt,ef.sequestration_limit_Mt,atol=1e-3,rtol=1e-7).sum()
    audit=pd.read_csv(PARENT/"excluded_carrier_audit_summary.csv")
    family=pd.read_csv(PARENT/"mean_cost_by_capacity_family.csv")
    op=pd.read_csv(PARENT/"mean_cost_by_operation_family.csv")
    family["penalty_share"]*=100
    ft=family[["capacity_family","total_cost","load_slack_TWh","penalty_share"]].copy()
    ft.columns=["Family","Mean cost (bn EUR/a)","Slack (TWh)","Penalty share (%)"]
    report=r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[italian]{babel}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,hyperref}
\graphicspath{{./}}
\title{Diagnosi quantitativa del load shedding:\\elettrificazione e persistenza dei combustibili}
\author{}
\date{12 ottobre 2026}
\begin{document}
\maketitle
\section{Risultato principale}
L'elevato costo della matrice deriva soprattutto dalla penalit\`a dello slack.
Nei test delle capacit\`a E su domanda F, la diagnosi dei network gi\`a risolti
indica una scarsit\`a di spazio emissivo e, in alcuni casi, di capacit\`a
di conversione o abbattimento. Non basta quindi interpretare il risultato
come insufficienza generalizzata delle taglie o dei link finali dei combustibili.
Non sono state eseguite nuove ottimizzazioni.
\section{Dati e metodo}
Le famiglie sono E (ELEC), F (OILGAS), M (MEOH), H (H2) e BASE.
Le statistiche complessive usano tutti i 156 test non diagonali, con uguale peso
per coppia di scenari; non usano le probabilit\`a IARSO. Le famiglie hanno dimensioni
diverse. La diagnosi dettagliata considera tutti i nove test E $\rightarrow$ F
e i tre ottimi diagonali F come riferimento. Costi e slack sono stati riconciliati
alla matrice ricalcolata. Lo slack \`e la produzione ponderata dei generatori
con carrier \texttt{load}, aggregata per carrier del bus: comprende vettori
intermedi e non equivale automaticamente a domanda finale non servita.
Il costo senza penalit\`a \`e una scomposizione contabile della soluzione con slack,
non il costo di una soluzione fattibile senza slack.
"""
    report+=tex_table(ft,"Medie per famiglia delle capacit\`a, escluse le diagonali.","tab:family")
    report+=r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{mean_cost_by_capacity_family.pdf}
\caption{Scomposizione del costo per famiglia delle capacit\`a.}
\end{figure}
\section{Budget emissivo e sequestro}
Il vincolo CO2 del codice locale limita la somma delle energie terminali degli
store non ciclici associati ai carrier emissivi. Si confronta questa grandezza
con la costante di \texttt{CO2Limit}; non si sostituisce con un inventario
indipendente delle emissioni lorde. Per il sequestro si confronta l'incremento
terminale degli store \texttt{co2 sequestered} con il limite operativo salvato.
"""
    report+=f"Il budget CO2 risulta raggiunto in {binding} test su 9; il limite di sequestro in {seqbinding} su 9. "
    report+=f"Lo scarto massimo dal budget CO2 \`e circa {abs(ef.co2_headroom_t).max():.0f} t, rispetto a circa {ef.co2_budget_Mt.mean():.0f} milioni di t: le piccole eccedenze numeriche non cambiano questa classificazione. "
    report+=f"Il valore di scarsit\`a CO2, definito come $-\\mu$ per il vincolo $\\leq$, varia fra {ef.co2_shadow_EUR_per_t.min():,.0f} e {ef.co2_shadow_EUR_per_t.max():,.0f} EUR/tCO2. "
    report+=r"""Questi valori sono duali locali della soluzione penalizzata: non sono
previsioni del prezzo di mercato della CO2 e non autorizzano un'estrapolazione
lineare a un grande rilassamento del budget. La degenerazione pu\`o rendere
non univoci i singoli duali.
"""
    report+=tex_table(view,"Vincoli globali e utilizzo del carbone nei nove test E su F. Coal use \`e la frazione media della disponibilit\`a primaria nelle ore con slack sul carbone industriale; non riportata per slack inferiore a 0.001 TWh.","tab:global")
    report+=r"""Occorre distinguere due regimi. In otto test il duale CO2 \`e fra
circa 297,500 e 369,117 EUR/t: la scarsit\`a emissiva \`e economicamente molto forte.
In E-HEAT su F-HEAT, invece, il budget \`e raggiunto ma il duale \`e solo circa
31 EUR/t e il sequestro \`e circa 78.6 Mt su 100 Mt disponibili. Essere al limite
non basta dunque a spiegare un grande slack: in questo caso emerge un limite
fisico locale delle caldaie, analizzato pi\`u avanti. Gli ottimi diagonali F
non hanno slack e mostrano duali CO2 di circa 430--545 EUR/t.
"""
    report+=r"""\section{Il carbone industriale: consegna libera e scarsit\`a emissiva}
L'audit di tutti i 156 network conferma che il link \texttt{coal for industry}
\`e espandibile, con CAPEX nullo e massimo nominale infinito. La capacit\`a
ottimizzata di un link gratuito pu\`o coincidere con il massimo dispatch senza
essere un collo di bottiglia economico: poteva essere aumentata nel solve.
Il generatore primario di carbone \`e invece fisso. Si confrontano dispatch e
disponibilit\`a per verificare se tale capacit\`a sia effettivamente satura.
Il link emette 0.3361 tCO2/MWh di carbone consegnato. Pertanto
\[
c_{\mathrm{CO2,coal}}=(-\mu_{\mathrm{CO2}})\,0.3361.
\]
Quando questa componente si avvicina alla penalit\`a dello slack di
100,000 EUR/MWh, l'uso di carbone compete con la mancata fornitura attraverso
il vincolo emissivo, anche se esiste capacit\`a primaria inutilizzata.
Questa \`e evidenza di scarsit\`a marginale emissiva nella soluzione esistente;
non quantifica il beneficio totale di eliminare il vincolo.
"""
    report+=tex_table(coalview,"Slack sul carbone industriale, capacit\`a primaria e costo implicito della CO2. Il picco del carbone riguarda l'intero anno.","tab:coal")
    report+=r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{carbon_and_coal_diagnosis.pdf}
\caption{Valore duale CO2 e utilizzo del carbone primario durante il relativo slack.}
\end{figure}
\section{Quali capacit\`a differiscono dal riferimento F?}
Si confrontano le capacit\`a effettive nei nove test con le capacit\`a dell'ottimo
di espansione F corrispondente alla domanda operativa. La tabella riporta
medie sulle nove coppie. Per i link, i GW sono riferiti alla porta di ingresso;
non sono direttamente sommabili ai GW di uscita o fra tecnologie diverse.
Il riferimento F non \`e una misura della capacit\`a minima necessaria:
un divario non prova da solo un collo di bottiglia. Le capacit\`a espandibili
nei test, come i link di consegna, sono quelle risultanti dal solve operativo.
"""
    report+=tex_table(comparison,"Capacit\`a medie E nei test e riferimento ottimale F per tecnologie selezionate.","tab:capacity")
    report+=r"""\subsection{Un limite fisico documentato: E-HEAT su F-HEAT}
In questo test lo slack complessivo \`e circa 575.37 TWh. Circa 208.20 TWh
sono sui nodi \texttt{gas urban central heat} e 367.17 TWh su
\texttt{gas urban decentral heat}: quasi tutto il deficit riguarda quindi
il servizio termico a gas. La tabella seguente considera soltanto le caldaie
con suffisso \texttt{gas-only}, che alimentano questi specifici nodi, senza
confonderle con le caldaie su nodi termici generici.
La quota di saturazione pesa l'energia di slack locale: quantifica quanta
energia non servita coincide con una caldaia al limite, non quanta energia
sarebbe recuperata espandendola. Le uscite degli asset sono distinte.
"""
    report+=tex_table(heat,"Caldaie gas-only: capacit\`a di ingresso, riferimento F e coincidenza tra slack locale e saturazione nel test E-HEAT su F-HEAT.","tab:boilers")
    report+=r"""Le caldaie centralizzate gas-only hanno circa 0.54 GW di ingresso,
contro 60.39 GW nel riferimento F-HEAT; le decentralizzate circa 59.84 contro
296.22 GW. In entrambi i gruppi il 100\% dell'energia di slack locale coincide
con la caldaia al limite superiore. Questi dati danno una misura quantitativa
della carenza di conversione termica locale, pur senza stimare un recupero
controfattuale del servizio mediante espansione.
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{heat_boiler_diagnosis.pdf}
\caption{Taglie delle caldaie gas-only e deficit termico nel test E-HEAT su F-HEAT.}
\end{figure}
"""
    report+=r"""\subsection{E-ROAD: una configurazione diversa dalle altre E}
La capacit\`a di elettrolisi di E-ROAD nei test \`e circa 0.00023 GW,
contro 55.73 GW medi nei riferimenti F; la cattura delle emissioni di processo
\`e circa 7.06 contro 13.83 GW di ingresso. E-HEAT ed E-IND hanno invece
circa 31.97 e 23.83 GW di elettrolisi e circa 12.68 GW di cattura di processo.
Questi divari individuano filiere molto diverse, ma non dimostrano che
l'elettrolisi mancante sia la causa diretta dello slack nei combustibili:
gli ottimi F scelgono una diversa combinazione di tecnologie e flussi.
Per contro, E-ROAD non ha meno capacit\`a in ogni tecnologia: le caldaie
decentralizzate aggregate sono circa 276.74 GW contro 233.53 GW nel riferimento.
Non \`e quindi corretto descrivere E-ROAD come uniformemente sottodimensionato.
"""
    report+=r"""La saturazione condizionata viene esportata per ogni generatore e link
fisso con capacit\`a positiva e slack sul proprio bus di uscita. Si misura la
quota di energia di slack locale coincidente con dispatch almeno al 99\%
del limite superiore effettivo, includendo disponibilit\`a temporali.
Il CSV distingue questa quota dalla frazione di ore di slack con saturazione.
Gli asset a capacit\`a zero sono visibili nel confronto delle taglie, ma non
ricevono una frazione di saturazione artificiale. Non vengono verificati
limiti di energia degli accumuli, vincoli CHP accoppiati o tutti i flussi di rete;
la saturazione simultanea di asset diversi non \`e additiva e non prova causalit\`a.
\section{Interpretazione per famiglia}
La famiglia E investe per una diversa distribuzione della domanda tra vettori.
Quando persiste la domanda di combustibili F, la libert\`a di espandere i link
finali non garantisce approvvigionamento, conversione e abbattimento sufficienti
entro il budget emissivo. I vincoli globali salvati e la disponibilit\`a inutilizzata
del carbone permettono di distinguere almeno questo meccanismo da una semplice
carenza della taglia primaria del carbone.
Per gli altri carrier, i divari di capacit\`a e la saturazione temporale forniscono
indizi quantitativi, ma non una attribuzione esclusiva dello slack a una tecnologia.
La famiglia E non \`e omogenea: E-ROAD, E-HEAT ed E-IND hanno filiere scoperte diverse.
Il caso E-HEAT su F-HEAT documenta una carenza della conversione termica locale;
negli altri otto casi, i duali elevati e l'inutilizzo di capacit\`a primaria
del carbone documentano una forte scarsit\`a marginale dello spazio emissivo.
In E-IND su F-IND, per esempio, il carbone primario usa circa il 3\% della
capacit\`a durante lo slack industriale; il costo CO2 implicito del carbone
\`e circa 99,993 EUR/MWh, quasi pari alla penalit\`a dello slack.
Nei test con E-ROAD tale costo sale a circa 124,060 EUR/MWh e la produzione
primaria di carbone \`e praticamente nulla pur con circa 26 GW disponibili.
Questi dati sostengono l'interpretazione della persistenza dei combustibili,
precisando che l'incompatibilit\`a pu\`o essere emissiva oppure legata a specifiche
conversioni, e non semplicemente alla capacit\`a dei link di consegna.
\begin{figure}[p]\centering
\includegraphics[width=0.98\linewidth,height=0.84\textheight,keepaspectratio]{carrier_slack_by_family_pair.pdf}
\caption{Slack medio per carrier e passaggio tra famiglie.}
\end{figure}
\section{Materiale per verifiche successive}
Nella stessa cartella sono disponibili:
\begin{itemize}
\item \texttt{global\_and\_coal\_diagnostics.csv}: budget, utilizzo, duali e carbone;
\item \texttt{capacity\_comparison\_to\_F\_optimum.csv}: confronto completo delle tecnologie;
\item \texttt{fixed\_assets\_conditional\_saturation.csv}: saturazione locale per asset;
\item \texttt{mean\_capacity\_comparison.csv}: medie per tecnologia;
\item \texttt{gas\_only\_boiler\_diagnosis.csv}: taglie termiche e saturazione locale;
\item PDF e PNG delle figure, con il sorgente di questo rapporto.
\end{itemize}
L'audit completo dei carrier esclusi dal congelamento delle capacit\`a \`e nella
cartella superiore. I generatori gas rimangono fissi, mentre gli store gas
sono espandibili; gli store di sequestro hanno massimi finiti. L'esclusione
preserva i flag originali e non rende espandibile un asset inizialmente fisso.
\end{document}
"""
    (OUT/"diagnosi_fuel_persistence.tex").write_text(report)
    print(g.to_string(index=False),flush=True)
    print(f"Report: {OUT/'diagnosi_fuel_persistence.tex'}",flush=True)


if __name__=="__main__":
    main()
