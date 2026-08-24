# Cutout uncertainty diagnostics

This first-pass pipeline analyzes only deterministic capacity-expansion
networks named `base_s_adm___2050.nc`. Files whose names continue with
`__cap-...__op-...` are validation networks and are deliberately excluded.

Run all available capacity-expansion scenarios:

```bash
conda run -n pypsa-eur python scripts/cutout_analysis/run_analysis.py
```

Analyze one capacity-expansion network in detail:

```bash
conda run -n pypsa-eur python scripts/cutout_analysis/analyze_one_case.py \
  results/cutouts_det_capexp_/d_2000/networks/base_s_adm___2050.nc \
  --output-dir results/cutout_analysis/output/one_case
```

Use `--scenarios d_1995 d_1996 ...` to restrict a trial run. Missing/invalid networks are
recorded in `output/collected/failures.csv` without stopping the batch.

The collected CSVs contain demand, renewable weather capacity factors,
generation, installed capacity, cost composition, load shedding, renewable
curtailment, and electricity-price diagnostics. The plots are deliberately
exploratory. Correlations are associations and should not be interpreted as
causal effects.
