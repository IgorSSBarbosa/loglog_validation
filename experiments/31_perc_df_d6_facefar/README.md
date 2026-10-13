# Experiment 31 — $\hat d_f$ at the upper critical dimension $d=6$: a deliberate failure of Assumption 1

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); ready to plan. No planned run yet.**

## Acceptance criterion

Exploratory: log corrections violate eq. (232). Record whether the pilot refuses to plan ($\omega_1\to0$ / unidentified) or returns a confident wrong answer.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=4$ (mean field, Wikipedia's $6+$ column) with log corrections at $d=6$. Wikipedia does not mention the log corrections; that part is from the literature, not this source.
- Correction: Wikipedia's $6+$ column lists $\omega=0$ and no $\Omega$, consistent with Assumption 1 failing here.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/31_perc_df_d6_facefar/recipes/samples_pilot_perc_df_d6_facefar.json \
    --study pilot --replicates 5 --data-root experiments/31_perc_df_d6_facefar/data
```

Run 2026-10-08 on niagara, 264 s drawn, ladder 2..32. d = 6.198 against the declared 6.000 (+3.3%); ω₁ = 1.67 ± 0.015; a₁ = -4.43 ± 0.019; eq. (232) fit converged; pilot γ̂ = 4.3556. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **GREEN** (ok). The pilot does not refuse: it reports a tight ω₁ and d_f ≈ 4.36 against the mean-field 4. This is the confident-wrong outcome Assumption 1's failure can produce.
