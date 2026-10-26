# Experiment 23 — Same as 22, $d=3$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot and T2+T3 run (2026-10-09). RATE passes; untuned prop:opt costs 5.6–22× in RMSE, tuned matches the best m₀; eq. (720) covers 479/480.**

## Acceptance criterion

Same as 22, $R=30$-$50$ per budget.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=2.52293(10)$. Wikipedia also lists 2.523(4), 2.530(4) (both derived), 2.5230(1), 2.5226(1); 2.5226(1) is about 2.3 combined errors below 2.52293(10).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 1.26(23), 1.6334, 1.62(13), 1.61(5). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.64(2), 0.65(2), 0.77(3) give $\omega\approx1.61(5)$, 1.64(5), 1.94(8); the other $\Omega$ entries are 0.50(9), 0.73(8), 0.60(8), 0.64(5).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/23_perc_df_d3_error_decay/recipes/samples_pilot_perc_df_d3_error_decay.json \
    --study pilot --replicates 5 --data-root experiments/23_perc_df_d3_error_decay/data
```

Run 2026-10-08, 189 s drawn: d = 3.139 (+4.6 %), ω₁ = 1.793 ± 0.042, a₁ = −1.48 ± 0.08, cv = 0.91.

## Step 2 — T2 (rate, eq. 941; point, eq. 946) and T3 (eq. 720 coverage), 2026-10-09

```bash
python3 src/budget/allocation_experiment.py -meta experiments/23_perc_df_d3_error_decay/recipes/sweep_error_decay.json \
    -o experiments/23_perc_df_d3_error_decay/data --tag error_decay
python3 calibration/score_theorems.py --sweep experiments/23_perc_df_d3_error_decay/data/error_decay \
    --pilot experiments/23_perc_df_d3_error_decay/data/pilot
```

6 budgets B = 10¹⁰ … 10¹² (×2.5), m₀ = 2..5, m = 5, R = 20 per cell (24 cells, 480 estimates),
seed 2026100923, 2 h 40 min on niagara.

**RATE.** Theory −ω₁/(d+2ω₁) = **−0.267**. Tuned arm **−0.304 [−0.381, −0.232]**, best arm
**−0.304 [−0.378, −0.231]**: theory inside both. The untuned prop:opt arm measures −0.354
[−0.440, −0.275], outside, because it starts far above the optimum (below).

**POINT.** Here eq. (946)'s m₀ is affordable, so its cost can be measured, and it is large:

| B | best m₀ | best RMSE | prop:opt m₀ | prop:opt / best | tuned m₀ | tuned / best |
|---|---|---|---|---|---|---|
| 10¹⁰ | 2 | 1.21e-2 | 4 | **5.6×** | 2 | 1.00 |
| 2.5·10¹⁰ | 2 | 7.36e-3 | 5 | **22.3×** | 2 | 1.00 |
| 6.3·10¹⁰ | 2 | 8.21e-3 | 5 | **11.7×** | 3 | 1.01 |
| 1.6·10¹¹ | 3 | 5.94e-3 | 5 | **10.8×** | 3 | 1.00 |
| 4·10¹¹ | 3 | 5.10e-3 | 5 | **6.5×** | 3 | 1.00 |
| 10¹² | 3 | 2.26e-3 | 5 | **8.6×** | 3 | 1.00 |

Same direction as srw (`plans/three_experiment_ladder.md` §4: m₀ ≈ 3 too high, 2.2–2.4×) but
much costlier: the discarded constant matters more when cv ≈ 0.9 and d ≈ 3. The tuned
allocation (constant restored from a₁ and cv) hits the empirical best at every budget.

**T3.** eq. (720) covers **479 of 480** estimates; every cell is consistent with ≥ 95 %;
bound/RMSE 2.7–27 (it widens with m₀). The one miss is instructive: at B = 10¹⁰, m₀ = 5
(ladder 64..1024) the allocation leaves n = 3 per scale, and one replicate had a top-rung mean
of exactly 0, so γ̂ = −∞. That is the article's "bad event" (Assumption 2 failing at small n),
which the B_bad term exists to price.

**CLT.** 22 of 24 cells pass (2 false fails expected at this many tests).
