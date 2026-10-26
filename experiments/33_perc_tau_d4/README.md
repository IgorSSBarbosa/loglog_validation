# Experiment 33 — $\hat\tau$, $d=4$, CPU

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-09); not ready: d off by +6.9%. No planned run yet.**

## Acceptance criterion

$\omega_1$ identified (se $<\omega_1/2$); then an eq. (720) bound. If not, report "not plannable" with the statistical term only.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=2.3142(5)$. Wikipedia also lists 2.26, 2.313(3), 2.3127(6), 2.313(2), 2.3124, 2.3150(8).
- Correction (cluster-size ladder): Wikipedia's $\Omega$ cell lists 0.31(5), 0.5(1), 0.37(4), 0.4008.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/33_perc_tau_d4/recipes/samples_pilot_perc_tau_d4.json \
    --study pilot --replicates 5 --data-root experiments/33_perc_tau_d4/data
```

Run 2026-10-09 on niagara, 706 s drawn, ladder 8..4096. d = 1.362 against the declared 1.274 (+6.9%); ω₁ = 0.233 ± 0.048; a₁ = -0.995 ± 0.21; eq. (232) fit converged; pilot γ̂ = -1.3519. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +6.9%). The fitted exponent is 1 − τ, so τ̂ = 2.352.
