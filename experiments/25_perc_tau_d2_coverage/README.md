# Experiment 25 — eq. (720) coverage for $\hat\tau$, $d=2$, tail observable

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-13). With box_exponent = 48/91 the eq. (720) bound covers 100/100 and the bias falls 5–10×; at 1/2 it covered 50 %.**The eq. (720) bound under-covers: 50 %.****

## Acceptance criterion

Coverage $\ge95\%$ (Wilson CI) over $R=50$ replicates at one planned budget.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\tau=187/91$ (exact).
- Correction (cluster-size ladder): $\Omega=72/91$ (exact). Wikipedia's $\Omega$ cell also lists 0.70(2), 0.77(4), 0.77(2), 0.44(9).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/25_perc_tau_d2_coverage/recipes/samples_pilot_perc_tau_d2_coverage.json \
    --study pilot --replicates 5 --data-root experiments/25_perc_tau_d2_coverage/data
```

Run 2026-10-08, 13 s drawn: d = 1.004 (+0.7 % on 0.997), ω₁ = 0.752 ± 0.066, a₁ = −0.522 ± 0.053, cv = 0.43.

## Step 2 — T3, eq. (720) coverage (2026-10-08)

```bash
python3 src/budget/allocation_experiment.py -meta experiments/25_perc_tau_d2_coverage/recipes/sweep_coverage.json \
    -o experiments/25_perc_tau_d2_coverage/data --tag coverage
python3 calibration/score_theorems.py --sweep experiments/25_perc_tau_d2_coverage/data/coverage \
    --pilot experiments/25_perc_tau_d2_coverage/data/pilot
```

B = 10⁸, tuned m₀ = 7 (ladder s = 256..8192), n = 6 009, R = 50, seed 2026100925, 13 min.

| quantity | value |
|---|---|
| mean γ̂ − (1 − 187/91) | **+0.0099** (≈ 4.8 sd) |
| Var ratio, eq. (583) | 1.14 [0.71, 1.61]: pass; KS p = 0.73 |
| eq. (720) | half-width 0.0100, dominant B_fs; coverage **25/50 = 0.50 [0.37, 0.63]: below nominal**; bound/RMSE = 0.99 |

**Reading.** The noise is as predicted; the bias is not. The bound's finite-size term is
built from the pilot's (ω₁, a₁), fitted on s = 8..2048, and the planned ladder runs to 8192 —
outside the range the correction was fitted on (`calibration/score_theorems.py`, ambiguity 2). The
actual bias sits exactly at the bound's edge, so half the replicates fall outside. Two
candidate causes, not yet separated: (a) the one-correction fit misdescribes the slow drift
at large s (the effective-exponent effect of `06_susceptibility`); (b) the tail observable's
finite box (`box_factor` = 16, `box_exponent` = 1/2) cuts off the largest clusters. Next
check: rerun the pilot on 256..8192 and compare its ω₁, a₁ with these.

## Step 3 — diagnosing the bias (2026-10-09)

**Re-pilot on the run's own ladder** (`recipes/samples_pilot_deep.json`, s = 256..8192, study
`pilot_deep`, 5.5 min): γ̂ = −1.0453 against −1.0549, the same +0.0096 offset, and the
log-log is **straight** there: eq. (232) has nothing to fit (ω₁ → 3.8 ± 0.6, a₁ ≈ −1.7·10⁶).
So the bias is not a correction to scaling that a deeper ladder outruns. cv rises 0.433 →
0.460 across the ladder.

**Hypothesis**, already documented in `models/percolation_tau.py` (lines 81-93): with
`box_exponent` = 1/2 the cluster-size cutoff relative to the box drifts as
s^(1 − d_f/2) = s^(5/96), about 13 % per decade, and the mean count per box drifts as
s^(τ−2), so Assumption 6 is also slowly violated (the cv rise above). Both bend the slope
by an amount that does not decay with s. Test queued: `recipes/samples_pilot_cutoff.json`,
the same ladder at `box_exponent` = 0.6 (> 1/d_f, so the cutoff recedes; no d_f imported).

**Result of the test (2026-10-09, study `pilot_cutoff`, 13 min).** At `box_exponent` = 0.6 the
log-log estimator on the same ladder (OLS slope on 256..8192) gives γ̂ = **−1.0577**, now
**−0.0028** off (the other sign; at 1/2 it was +0.0101 by the same measure). cv falls
0.280 → 0.222, and eq. (232) again finds no correction (ω₁ = 0.06 ± 1.3, fit not converged).
So the slope tracks the box rule, not the estimator: below 1/d_f the cutoff closes in, above
it the cutoff recedes as s^(−0.14), and in both cases the finite-box correction decays too
slowly to look like a correction on this ladder.

*Correction (2026-10-26).* This paragraph first reported γ̂ = −1.0881 (−0.033 off) and, by
interpolating, a zero crossing of the bias near `box_exponent` ≈ 0.52 = 48/91. That γ̂ was the
non-converged eq. (232) fit's γ, not the log-log estimator. On the estimator itself the bias
is +0.0101 at 1/2, +0.0025 at 48/91 (`pilot_box_df`, same window) and −0.0028 at 0.6: it
decreases monotonically through the three and does not vanish at 1/d_f. What 48/91 does do is
freeze the cutoff, which makes cv flat (Assumption 6) and leaves a bias that the bound
absorbs (Step 4). Figure: `images/box_rule.png`, drawn by
`experiments/plot_percolation_section.py`.

**Conclusion.** The eq. (720) coverage failure in Step 2 is a property of the τ observable's
box rule, not of the bound: eq. (232) cannot see a bias that does not decay on the ladder.
Using `box_exponent` = 48/91 would remove it, but imports d_f as a design constant; that
choice is the user's (ground rule 4). **Decision pending.**

## Step 4 — rerun at box_exponent = 48/91 (user decision 2026-10-13)

`box_exponent` = 48/91 = 1/d_f(2) is a design constant: it sizes the box and never reaches the
estimator (same category as `DF_LOWER` in `models/percolation_tau_zd.py`).

```bash
python3 src/study/pilot.py -meta experiments/25_perc_tau_d2_coverage/recipes/samples_pilot_box_df.json \
    --study pilot_box_df --replicates 5 --data-root experiments/25_perc_tau_d2_coverage/data
python3 src/budget/allocation_experiment.py -meta experiments/25_perc_tau_d2_coverage/recipes/sweep_coverage_box_df.json \
    -o experiments/25_perc_tau_d2_coverage/data --tag coverage_box_df
python3 calibration/score_theorems.py --sweep experiments/25_perc_tau_d2_coverage/data/coverage_box_df \
    --pilot experiments/25_perc_tau_d2_coverage/data/pilot_box_df
```

**Pilot** (8..8192, 330 s): d = 1.0625 (+0.7 % on the declared 2·48/91), ω₁ = 0.557 ± 0.024,
a₁ = −0.425 ± 0.003, **cv flat at 0.39 on every rung** (it rose 0.42 → 0.46 at 1/2): Assumption 6
now holds. Triage GREEN.

**Coverage** (B = 10⁸, R = 50 per cell, seed 2026101326, 25 min):

| m₀ | ladder | bias | Var ratio | KS p | eq. (720) half-width | coverage | bound/RMSE |
|---|---|---|---|---|---|---|---|
| 7 | 256..8192 (the failed run's ladder) | **+0.0022** (was +0.0099) | 0.75 [0.49, 1.01] | 0.73 | 0.019 | **50/50** | 6.7 |
| 8 (tuned) | 512..16384 | **+0.0010** | 0.99 [0.54, 1.51] | 0.76 | 0.019 | **50/50** | 5.7 |

**Conclusion.** With the box tied to the cluster size, the bias decays like a correction to
scaling again, eq. (232) identifies it, and eq. (720) covers. The 50 % failure at
`box_exponent` = 1/2 was the observable, not the bound.
