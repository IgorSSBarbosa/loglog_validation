# Experiment 32 — $\hat\tau$, $d=3$, CPU

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-09); not ready: d off by -7.0%. No planned run yet.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.13$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=2.18909(5)$. Wikipedia also lists 2.186(2), 2.1888, 2.189(2), 2.190(2), 2.189(1), 2.18906(8), 2.1892(1), 2.1938(12); 2.1938(12) is about 4 of its quoted errors from 2.18909(5).
- Correction (cluster-size ladder): Wikipedia's $\Omega$ cell lists 0.50(9), 0.64(2), 0.73(8), 0.65(2), 0.60(8), 0.77(3), 0.64(5).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/32_perc_tau_d3/recipes/samples_pilot_perc_tau_d3.json \
    --study pilot --replicates 5 --data-root experiments/32_perc_tau_d3/data
```

Run 2026-10-09 on niagara, 671 s drawn, ladder 8..4096. d = 1.120 against the declared 1.204 (-7.0%); ω₁ = 0.49 ± 0.085; a₁ = -0.617 ± 0.038; eq. (232) fit converged; pilot γ̂ = -1.1957. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by -7.0%). The fitted exponent is 1 − τ, so τ̂ = 2.196.
