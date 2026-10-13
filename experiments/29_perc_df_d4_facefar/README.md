# Experiment 29 — $\hat d_f$, $d=4$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: d off by +5.6%. No planned run yet.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.087$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=3.0446(7)$. Wikipedia also lists 3.12(2), 3.05(5), 3.003, 3.0472(14), 3.046(7), 3.046(5), 3.0479, 3.0437(11).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 0.94(15), 1.2198, 1.13(10), 1.0(2). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.31(5), 0.5(1), 0.37(4), 0.4008 give $\omega\approx0.94(15)$, 1.5(3), 1.13(12), 1.220.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/29_perc_df_d4_facefar/recipes/samples_pilot_perc_df_d4_facefar.json \
    --study pilot --replicates 5 --data-root experiments/29_perc_df_d4_facefar/data
```

Run 2026-10-08 on niagara, 222 s drawn, ladder 4..128. d = 4.223 against the declared 4.000 (+5.6%); ω₁ = 1.26 ± 0.096; a₁ = -2.01 ± 0.14; eq. (232) fit converged; pilot γ̂ = 3.0475. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +5.6%).
