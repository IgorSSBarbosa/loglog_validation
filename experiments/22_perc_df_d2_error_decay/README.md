# Experiment 22 — Error decay eq. (941) under the allocation eqs. (945)-(946), and eq. (720) coverage, $\hat d_f$, $d=2$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot and T2+T3 run (2026-10-08/09). RATE passes; tuned allocation within 1.10× of the best; eq. (720) covers everywhere; prop:opt's untuned m₀ unaffordable.**

## Acceptance criterion

Slope of $\log$ RMSE against $\log B$ over $B=10^9\ldots10^{12}$ sites ($\times4$ apart, $R=50$ each) matches $-\omega_1/(d+2\omega_1)$ within its CI; the optimal allocation beats a naive one at equal $B$; eq. (720) coverage $\ge95\%$ (Wilson CI on the coverage). Also record the empirical-optimum $m_0$ next to eq. (946)'s (`plans/three_experiment_ladder.md` §4 found an offset of about 3 on srw).

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=91/48$ (exact).
- Correction (length ladder): Wikipedia lists $\omega=3/2$ directly; equivalently $\omega=\Omega\,d_f=(72/91)(91/48)=3/2$. Its $\Omega$ cell also lists 0.70(2), 0.77(4), 0.77(2), 0.44(9).
- Predicted error-decay slope with $\omega=3/2$: $-1.5/(2+3)=-0.30$.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/22_perc_df_d2_error_decay/recipes/samples_pilot_perc_df_d2_error_decay.json \
    --study pilot --replicates 5 --data-root experiments/22_perc_df_d2_error_decay/data
```

First pilot (10¹⁰, 8..512) left ω₁ unidentified; re-piloted at 4·10¹¹ on 8..1024 (188 s): d = 2.022 (+1.1 %), ω₁ = 0.629 ± 0.019, a₁ = −0.075 ± 0.001, cv = 0.38. ω₁ is the effective correction on this ladder, not the leading ω = 3/2 (a₁ is tiny on the cylinder).

## Step 2 — T2 (rate, eq. 941; point, eq. 946) and T3 (eq. 720 coverage), 2026-10-08/09

```bash
python3 src/budget/allocation_experiment.py -meta experiments/22_perc_df_d2_error_decay/recipes/sweep_error_decay.json \
    -o experiments/22_perc_df_d2_error_decay/data --tag error_decay
python3 calibration/score_theorems.py --sweep experiments/22_perc_df_d2_error_decay/data/error_decay \
    --pilot experiments/22_perc_df_d2_error_decay/data/pilot
```

6 budgets B = 10⁹ … 10¹² (×4), m₀ = 3..7, m = 6, R = 20 per cell (30 cells, 600 estimates), seed
2026100922, 4 h 26 min on niagara. The driver exited 1 *after* writing its artifact (the
prop:opt-None print crash, fixed 2026-10-08 after this run had started).

**RATE.** Theory −ω₁/(d+2ω₁) = **−0.192**. Measured: tuned arm **−0.174 [−0.228, −0.122]**, best
arm **−0.185 [−0.236, −0.149]** (bootstrap 95 %). Theory inside both.

**POINT.** eq. (946)'s untuned prop:opt m₀ is 10–12: a ladder up to i = 2¹⁸ with n < 2 below
B = 1.6·10¹⁰, so it is unaffordable at every budget here (the srw over-shoot of
`plans/three_experiment_ladder.md` §4, made extreme by a₁ ≈ −0.07). The tuned allocation
(constant restored from a₁, cv) picks the best m₀ at 4 of 6 budgets and costs at most 1.10×
the best RMSE:

| B | best m₀ | best RMSE | prop:opt m₀ | tuned m₀ | tuned / best |
|---|---|---|---|---|---|
| 10⁹ | 3 | 3.33e-3 | unaffordable | 3 | 1.00 |
| 4·10⁹ | 4 | 2.72e-3 | unaffordable | 4 | 1.00 |
| 1.6·10¹⁰ | 4 | 1.65e-3 | 10 | 4 | 1.00 |
| 6.4·10¹⁰ | 4 | 1.96e-3 | 10 | 5 | 1.05 |
| 2.56·10¹¹ | 5 | 1.24e-3 | 11 | 5 | 1.00 |
| 10¹² | 5 | 8.58e-4 | 12 | 6 | 1.10 |

The mechanism is visible directly: at fixed m₀ = 3 the bias stays at +0.0027 from 10⁹ to 10¹²
while the sd falls 37×, so the error floors and the optimum moves up the ladder with B.

**T3.** eq. (720) covers in **all 30 cells** (20/20 each, Wilson lower end 0.84), with
bound/RMSE 2.9–9.5 (B_fs dominant at low m₀, B_bad at high m₀).

**CLT, as a by-product.** 27 of 30 cells pass at α = 0.05 per test; 3 fail with ratio < 1
(0.56–0.65). With two tests per cell the expected false-fail count is about 2–3, so this is
consistent with the CLT holding.

Figure: `data/error_decay/theorem_scores.png` (QQ, RMSE vs B with the predicted slope, coverage
per cell). Candidate for `images/` once reviewed.
