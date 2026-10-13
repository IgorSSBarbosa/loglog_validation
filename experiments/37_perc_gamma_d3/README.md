# Experiment 37 — Susceptibility exponent $\hat\gamma$, $d=3$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot run (2026-10-08); not ready: omega1 unidentified. No planned run yet.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.040$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma=1.793(3)$ [also 1.819(3)]. Wikipedia's full list: 1.6, 1.80(5), 1.66(7), 1.793(3), 1.805(20), 1.8357, 1.819(3), 1.78(3). The two precise values 1.793(3) and 1.819(3) disagree by about 6 combined errors.
- Correction ($\varepsilon=|p-p_c|$ ladder): $\Delta_1=\omega\nu$ (Wikipedia: $\Delta_1=\Omega\beta\delta=\omega\nu$). With $\omega=1.61(5)$ and $\nu=0.8762(7)$, $\Delta_1\approx1.41(4)$; Wikipedia's other $\omega$ entries (1.26(23), 1.6334, 1.62(13)) give 1.10 to 1.43.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/37_perc_gamma_d3/recipes/samples_pilot_perc_gamma_d3.json \
    --study pilot --replicates 5 --data-root experiments/37_perc_gamma_d3/data
```

Run 2026-10-08 on niagara, 411 s drawn, ladder 4..256. d = 2.716 against the declared 2.626 (+3.5%); ω₁ = 0.804 ± 2.3; a₁ = 0.356 ± 1.4e+05; eq. (232) fit converged; pilot γ̂ = 1.7986. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (omega1 unidentified).
