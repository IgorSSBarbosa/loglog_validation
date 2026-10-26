# Experiment 38 — Susceptibility exponent $\hat\gamma$, $d=4$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: omega1 unidentified. No planned run yet.**

## Acceptance criterion

$\omega_1$ identified, then an eq. (720) bound. If not, "not plannable".

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma=1.430(6)$. Wikipedia also lists 1.6(1), 1.48(8), 1.422(16), 1.4500, 1.435(15).
- Correction ($\varepsilon=|p-p_c|$ ladder): $\Delta_1=\omega\nu$. With $\nu=0.6842(16)$ and Wikipedia's $\omega$ entries 0.94(15), 1.2198, 1.13(10), 1.0(2), $\Delta_1\approx0.64(10)$, 0.835, 0.77(7), 0.68(14).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/38_perc_gamma_d4/recipes/samples_pilot_perc_gamma_d4.json \
    --study pilot --replicates 5 --data-root experiments/38_perc_gamma_d4/data
```

Run 2026-10-08 on niagara, 3611 s drawn, ladder 2..128. d = 2.773 against the declared 2.690 (+3.1%); ω₁ = 2.45 ± 1.6; a₁ = -0.145 ± 4.1; eq. (232) fit converged; pilot γ̂ = 1.4278. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (omega1 unidentified).
