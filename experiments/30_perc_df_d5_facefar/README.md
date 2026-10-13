# Experiment 30 — $\hat d_f$, $d=5$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: d off by +6.6%. No planned run yet.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.21$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=3.5260(14)$. Wikipedia also lists 3.54(4), 3.69(2), 3.528, 3.524(2).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 0.96(26), 0.7178. Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.27(7), 0.2034, 0.210(2) give $\omega\approx0.95(25)$, 0.717, 0.740(7).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/30_perc_df_d5_facefar/recipes/samples_pilot_perc_df_d5_facefar.json \
    --study pilot --replicates 5 --data-root experiments/30_perc_df_d5_facefar/data
```

Run 2026-10-08 on niagara, 247 s drawn, ladder 4..64. d = 5.329 against the declared 5.000 (+6.6%); ω₁ = 1.35 ± 0.16; a₁ = -3.08 ± 0.29; eq. (232) fit converged; pilot γ̂ = 3.6293. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +6.6%).
