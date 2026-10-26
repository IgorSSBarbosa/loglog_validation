# Experiment 40 — $\widehat{\gamma/\nu}$, $d=3$, origin anchor

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: fit not converged. No planned run yet.**

## Acceptance criterion

First eq. (720) bound on $\gamma/\nu$ in $d=3$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma/\nu=2-\eta=2.0431(14)$, from $\eta=-0.0431(14)$. Wikipedia's other $\eta$ values give $\gamma/\nu=$ 2.046(8), 2.059(9), 2.07(5), 2.0470, 2.03(1).
- Correction (length ladder): $\omega$ = 1.26(23), 1.6334, 1.62(13), 1.61(5) (Wikipedia's $\omega$ cell, $d=3$).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/40_perc_gammanu_d3/recipes/samples_pilot_perc_gammanu_d3.json \
    --study pilot --replicates 5 --data-root experiments/40_perc_gammanu_d3/data
```

Run 2026-10-08 on niagara, 65 s drawn, ladder 4..256. d = 3.024 against the declared 3.000 (+0.8%); ω₁ = 10.4 ± 0.76; a₁ = -4.62e+04 ± 5.1e+04; eq. (232) fit NOT converged; pilot γ̂ = 2.0097. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (fit not converged).
