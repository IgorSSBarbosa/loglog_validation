# Experiment 34 — $\hat\tau$, $d=5$, CPU

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-09); not ready: omega1 unidentified, d off by -15.2%. No planned run yet.**

## Acceptance criterion

Same as 33.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=2.4175(2)$. Wikipedia also lists 2.33, 2.412(4), 2.4171, 2.419(1).
- Correction (cluster-size ladder): Wikipedia's $\Omega$ cell lists 0.27(7), 0.2034, 0.210(2).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/34_perc_tau_d5/recipes/samples_pilot_perc_tau_d5.json \
    --study pilot --replicates 5 --data-root experiments/34_perc_tau_d5/data
```

Run 2026-10-09 on niagara, 988 s drawn, ladder 8..4096. d = 1.259 against the declared 1.484 (-15.2%); ω₁ = 0.0633 ± 0.047; a₁ = -5.1 ± 4.8; eq. (232) fit converged; pilot γ̂ = -1.5948. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (omega1 unidentified, d off by -15.2%). The fitted exponent is 1 − τ, so τ̂ = 2.595.
