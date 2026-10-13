# Experiment 21 — CLT of eq. (583) for $\hat d_f$, $d=3$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot and T1 run (2026-10-08). CLT holds in its exact finite-m form, not in the asymptotic one.**

## Acceptance criterion

Same as 20, with $R=200$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=2.52293(10)$. Wikipedia also lists 2.523(4), 2.530(4) (both derived), 2.5230(1), 2.5226(1); 2.5226(1) is about 2.3 combined errors below 2.52293(10).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 1.26(23), 1.6334, 1.62(13), 1.61(5). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.64(2), 0.65(2), 0.77(3) give $\omega\approx1.61(5)$, 1.64(5), 1.94(8); the other $\Omega$ entries are 0.50(9), 0.73(8), 0.60(8), 0.64(5).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/21_perc_df_d3_clt/recipes/samples_pilot_perc_df_d3_clt.json \
    --study pilot --replicates 5 --data-root experiments/21_perc_df_d3_clt/data
```

Run 2026-10-08 on niagara (GPU), 190 s drawn. d = 3.139 ± 0.007 (+4.6 % on the declared 3), ω₁ = 1.672 ± 0.081, a₁ = −1.28 ± 0.10, cv = 0.91. Pipeline ready to plan.

## Step 2 — T1, the CLT (2026-10-08)

```bash
python3 src/budget/allocation_experiment.py -meta experiments/21_perc_df_d3_clt/recipes/sweep_clt.json \
    -o experiments/21_perc_df_d3_clt/data --tag clt
python3 calibration/score_theorems.py --sweep experiments/21_perc_df_d3_clt/data/clt \
    --pilot experiments/21_perc_df_d3_clt/data/pilot
```

B = 2·10¹¹ per replicate, tuned m₀ = 2 (ladder 8..256), n = 4 876 per scale, **R = 200**, seed
2026100921. 36 min on niagara.

| quantity | value |
|---|---|
| mean γ̂ − 2.52293 | +0.0043 (≈ 0.94 sd: a finite-size bias, as the pilot's a₁ < 0 predicts in sign) |
| sd observed | 0.00454 |
| sd predicted, eq. (583) with σ∞² = cv² at the top scale | 0.00403 → Var ratio **1.27 [1.05, 1.47]**: FAIL |
| Var ratio, exact finite-m form (`sigma_se_per_scale`, per-scale cv) | **1.04** |
| KS of standardized γ̂ vs N(0,1) | p = 0.14: normal |
| eq. (720) half-width | 0.059; coverage 200/200 = **1.000 [0.981, 1.000]**, bound/RMSE = **9.5** (dominant term B_bad) |

**Reading.** γ̂ is Gaussian, and its variance is what the exact per-scale formula predicts. The
asymptotic eq. (583) underpredicts it by 27 % because Assumption 6 has not converged on
8..256: cv falls from 1.17 at the bottom of the ladder to 0.83 at the top, so σ∞² taken
from the top scale is too small for the lower rungs. The eq. (720) bound covers, but is
9.5× wider than the actual error, almost all of it the stray-mean term B_bad.
