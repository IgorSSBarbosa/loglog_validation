# Experiment 20 — CLT of eq. (583) for $\hat d_f$, $d=2$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot and T1 run (2026-10-08). CLT holds.**

## Acceptance criterion

Var($\hat\gamma$) over $R=1000$ fresh replicates at fixed $(n,m,m_0)$ matches $12\sigma_\infty^2/(nm^3\log^2\rho)$ within a bootstrap CI, and a KS test of the standardized $\hat\gamma$ against $\mathcal N(0,1)$ passes.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=91/48$ (exact).
- Correction (length ladder): Wikipedia lists $\omega=3/2$ directly; equivalently $\omega=\Omega\,d_f=(72/91)(91/48)=3/2$. Its $\Omega$ cell also lists 0.70(2), 0.77(4), 0.77(2), 0.44(9).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/20_perc_df_d2_clt/recipes/samples_pilot_perc_df_d2_clt.json \
    --study pilot --replicates 5 --data-root experiments/20_perc_df_d2_clt/data
```

First pilot (budget 10¹⁰, 8..512): ω₁ unidentified (0.38 ± 0.21, a₁ ≈ 0, 5 s drawn). Re-piloted at 4·10¹¹ on 8..1024 (188 s): d = 2.022 (+1.1 %), ω₁ = 0.581 ± 0.047, a₁ = −0.073 ± 0.002, cv = 0.38. The cylinder leaves only a tiny correction, so the fitted ω₁ is an effective exponent on this ladder, well below the leading ω = 3/2.

## Step 2 — T1, the CLT (2026-10-08)

```bash
python3 src/budget/allocation_experiment.py -meta experiments/20_perc_df_d2_clt/recipes/sweep_clt.json \
    -o experiments/20_perc_df_d2_clt/data --tag clt
python3 calibration/score_theorems.py --sweep experiments/20_perc_df_d2_clt/data/clt \
    --pilot experiments/20_perc_df_d2_clt/data/pilot
```

B = 10¹⁰ per replicate, tuned m₀ = 4 (ladder 32..1024), n = 6 191, **R = 1000**, seed 2026100920,
13.4 min on niagara. (The driver exited 1 *after* writing its artifact: a print of the unaffordable
prop:opt m₀ = None crashed. Fixed in `src/budget/allocation_experiment.py` the same day.)

| quantity | value |
|---|---|
| mean γ̂ − 91/48 | +0.00166 (≈ 1.0 sd) |
| Var ratio, eq. (583) | **1.036 [0.946, 1.128]**: pass; exact finite-m form 0.971 |
| KS vs N(0,1) | p = 0.976 |
| eq. (720) | half-width 0.0122; coverage **1000/1000 [0.996, 1]**; bound/RMSE = 5.3 (dominant B_fs) |

**Reading.** At d = 2 the article's CLT holds as stated, asymptotic σ∞² included (cv is nearly flat
on the cylinder). The bound covers and is 5× conservative.
