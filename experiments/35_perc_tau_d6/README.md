# Experiment 35 — $\hat\tau$, $d=6$ (upper critical dimension), CPU

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-09); not ready: fit not converged, d off by -19.0%. No planned run yet.**

## Acceptance criterion

Exploratory (log corrections at $d=6$). Step 0, before any planned run: p_c check. 05 H0 was inconclusive at $d=6$ and flatness of the spanning probability is not the right criterion at $d\ge6$, so this needs a decision before it runs.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=5/2$ (mean field, Wikipedia's $6+$ column) with log corrections at $d=6$. Wikipedia does not mention the log corrections; that part is from the literature, not this source.
- Correction: Wikipedia gives no $\Omega$ for $d\ge6$ (its $\omega$ entry there is 0).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/35_perc_tau_d6/recipes/samples_pilot_perc_tau_d6.json \
    --study pilot --replicates 5 --data-root experiments/35_perc_tau_d6/data
```

Run 2026-10-09 on niagara, 842 s drawn, ladder 8..1024. d = 1.097 against the declared 1.354 (-19.0%); ω₁ = 0.0347 ± 0.061; a₁ = -12.6 ± 4.2; eq. (232) fit NOT converged; pilot γ̂ = -1.7827. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (fit not converged, d off by -19.0%). The fitted exponent is 1 − τ, so τ̂ = 2.783.
