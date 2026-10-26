# Experiment 28 — $\hat d_f$, $d=3$, axis anchor (slab $k=1$): does the slow drift continue past $i=512$?

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: d off by +7.9%. No planned run yet.**

## Acceptance criterion

Exploratory: local slopes to $i=1024$ and a two-correction diagnostic. Explains or removes the axis/face_far disagreement in `exponents_report.md` Table 1b.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=2.52293(10)$. Wikipedia also lists 2.523(4), 2.530(4) (both derived), 2.5230(1), 2.5226(1); 2.5226(1) is about 2.3 combined errors below 2.52293(10).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 1.26(23), 1.6334, 1.62(13), 1.61(5). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.64(2), 0.65(2), 0.77(3) give $\omega\approx1.61(5)$, 1.64(5), 1.94(8); the other $\Omega$ entries are 0.50(9), 0.73(8), 0.60(8), 0.64(5).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/28_perc_df_d3_axis_deep/recipes/samples_pilot_perc_df_d3_axis_deep.json \
    --study pilot --replicates 5 --data-root experiments/28_perc_df_d3_axis_deep/data
```

Run 2026-10-08 on niagara, 300 s drawn, ladder 4..512. d = 3.237 against the declared 3.000 (+7.9%); ω₁ = 1.71 ± 0.071; a₁ = -0.682 ± 0.036; eq. (232) fit converged; pilot γ̂ = 2.5585. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +7.9%).
