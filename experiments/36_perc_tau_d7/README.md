# Experiment 36 — $\hat\tau$, $d=7$ (above the upper critical dimension), CPU

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-09); not ready: fit not converged, d off by -16.7%. No planned run yet.**

## Acceptance criterion

eq. (720) bound; known value in the next section. Step 0: p_c check, as in 35.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=5/2$ (mean field; Wikipedia has no $d=7$ column, its $6+$ column applies).
- Correction: Wikipedia gives no $\Omega$ for $d\ge6$ (its $\omega$ entry there is 0), so it supplies no correction exponent for $d=7$.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/36_perc_tau_d7/recipes/samples_pilot_perc_tau_d7.json \
    --study pilot --replicates 5 --data-root experiments/36_perc_tau_d7/data
```

Run 2026-10-09 on niagara, 863 s drawn, ladder 8..1024. d = 1.566 against the declared 1.880 (-16.7%); ω₁ = 13.5 ± 2.5; a₁ = 1.68e+10 ± 1.1e+10; eq. (232) fit NOT converged; pilot γ̂ = -1.4451. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (fit not converged, d off by -16.7%). The fitted exponent is 1 − τ, so τ̂ = 2.445.
