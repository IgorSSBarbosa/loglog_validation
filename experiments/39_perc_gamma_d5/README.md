# Experiment 39 — Susceptibility exponent $\hat\gamma$, $d=5$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: a1 unidentified. No planned run yet.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.80$, with $a_1$ identified.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma=1.1792(7)$. Wikipedia also lists 1.3(1), 1.18(7), 1.185(5), 1.1817.
- Correction ($\varepsilon=|p-p_c|$ ladder): $\Delta_1=\omega\nu$. With $\nu=0.5739(1)$ and Wikipedia's $\omega$ entries 0.96(26), 0.7178, $\Delta_1\approx0.55(15)$, 0.412.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/39_perc_gamma_d5/recipes/samples_pilot_perc_gamma_d5.json \
    --study pilot --replicates 5 --data-root experiments/39_perc_gamma_d5/data
```

Run 2026-10-08 on niagara, 905 s drawn, ladder 4..64. d = 2.892 against the declared 2.839 (+1.9%); ω₁ = 0.941 ± 0.46; a₁ = -0.311 ± 1.5; eq. (232) fit converged; pilot γ̂ = 1.2050. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **YELLOW** (a1 unidentified).
