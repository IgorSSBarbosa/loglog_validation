# Experiment 39 — Susceptibility exponent $\hat\gamma$, $d=5$

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: 10 h long run done 2026-11-08. FAILS: the eq. (720) bound from report.py excludes 1.1792 (pinned refit underestimates B_fs when the bias is large). The long-run AIC fit gives 1.1743 +- 0.0041.**

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

## AIC-quality check on all existing data (user, 2026-11-07)

Every dataset with these exact params was pooled (duplicates dropped; the sets agree at shared scales, chi2 p = 0.37) and fitted by `tools/local_slope.omega_ranges` (CLT se). Pooled 4..64 (39 + 09): 5 points, 1 dof, omega 0.72 +- 0.31, gamma 1.192 +- 0.019: too short to judge.

### Pilot `aic_explore_d5` (launched 2026-11-07 on kolyma (CPU, 32 threads), seed 2026110839)

`percolation_susceptibility_explore`, dim 5 enabled for this (user, 2026-11-07; key range +-2047 sites, edge raises), x = 4..512.

```bash
.venv/bin/python src/study/pilot.py -meta experiments/39_perc_gamma_d5/recipes/samples_gsusc_d5_explore.json \
    --study aic_explore_d5 --data-root experiments/39_perc_gamma_d5/data --replicates 5 --seed 2026110839
```

Result (638.9 s drawn, 5 replicates): AIC (CLT se, 4..512) omega 0.44 +- 0.25 (stat) +- 0.16 (range), gamma 1.1783 +- 0.0188, containing 1.1792. Pipeline constants: omega1 0.440 +- 0.157, a1 -0.384 +- 2.286, d 1.433 +- 0.061, cv 6.7 (x=4) to 42.2 (x=512). Triage still flags a1 as unidentified, but that is the error method, not the fit: the pooled fit gives a1 -0.38 +- 0.12 (Fisher; +- 0.036 with omega known), and the +- 2.29 is the spread of 5 single-replicate fits, one of which ran into the omega -> 0 valley (omega 0.03, a1 -11.9). The 4 fit parameters are 95-99.8 % correlated at omega ~ 0.44 over 4..512 (condition number 218).

### Long run `aic_explore_d5`, 10 h (user, 2026-11-08), xingu

Budget is bias-limited: eq. (720) half-width 0.126 at 5 h, 0.1258 at 10 h, 0.125 at 48 h (B_fs 0.124 from the pilot's a1, omega1; 1.96 sigma_se 0.0014 at 10 h, about 2x the literature +- 0.0007). The planner's actual bias is 0.025. The ladder stops at 512 because of the key range (+-2047 sites at dim 5), not memory. Run anyway so AIC on the long-run data can be compared with the literature precision (user).

```bash
D=experiments/39_perc_gamma_d5/data
.venv/bin/python src/study/plan.py --study aic_explore_d5 --data-root $D --time 10h --replicates 5 --max-scale 512 --accept
.venv/bin/python src/study/run.py --study aic_explore_d5 --data-root $D --seed 2026110855 && \
  .venv/bin/python src/study/report.py --study aic_explore_d5 --data-root $D
for r in 1 2 3 4 5; do   # x = 4, 8 for the AIC refit, se(log Ybar) matched to x = 512
  .venv/bin/python src/generate/generate.py -meta experiments/39_perc_gamma_d5/recipes/samples_aic_explore_d5_small.json \
      --tag aic_explore_d5_small/rep$r --seed 20261108550$r
done
```

Plan: m0 = 3, scales 16..512, n = 704,073,267 per scale x 5 replicates. Log: `data/aic_explore_d5_run10h.log`.

Result (584.6 min drawn, predicted 600.0, ratio 0.97x):

| | value |
|---|---|
| gamma-hat (eq. 526, 16..512) | 1.20269 (replicates 1.20303, 1.20281, 1.20289, 1.20177, 1.20294) |
| Student t(4) | [1.20204, 1.20333] |
| eq. (720) bound, `report.py` | [1.19358, 1.21180] (+- 0.0091): **excludes 1.1792** |
| B_fs, `report.py` | 0.0080, from the pinned AIC refit of the pilot on 4..256: omega1 1.03 +- 0.33, a1 -0.341 |
| 1.96 sigma_se | 0.0011 (about 1.5x the literature +- 0.0007) |
| actual bias gamma-hat - 1.1792 | +0.0235 (the plan predicted 0.0251) |

Why it fails: the Step 4 refit pins gamma to gamma-hat. That assumes the bias of gamma-hat is small next to its se, which holds in d = 4 (B_fs 8e-5) but not here, where the bias is 20x the se term. Forced to the slope 1.2027, the fit needs a faster correction (omega1 1.03, against 0.44 free on the same pilot), and B_fs comes out about 11x too small. With omega1, a1 from the long-run AIC fit below (free gamma), the same eq. (720) gives B_fs 0.105, bound [1.0968, 1.3086], containing 1.1792; over omega1 +- 1 se, B_fs 0.086..0.130, all containing it. Correction to the budget table above: it used m = 5; report.py's m is the number of scales (6), which gives B_fs 0.089 from the pilot constants, not 0.124.

AIC on the whole long run (`tools/local_slope.omega_ranges`, gamma free, min_points 6, CLT se, x = 4..512: 16..512 from run.py plus 4, 8 from `data/aic_explore_d5_small/rep1..5`): best range 4..256 (chi2 0.93/3, weight 0.29; 4..512 weight 0.27), **omega 0.421 +- 0.029 (stat) +- 0.032 (range) = +- 0.043 (10 %)**, a1 -0.414, **gamma 1.1743 +- 0.0030 (stat) +- 0.0028 (range) = +- 0.0041**: 1.2 sigma below 1.1792, error about 6x the literature's. omega is in x = 1/eps units, so it is Delta_1 directly: 0.42 against 0.412 from Wikipedia's omega x nu.

## Wilson interval with the joint refit (2026-11-11)

B_fs now takes $\omega_1, a_1$ from one AIC range fit on the pilot **and** the long run
together ($\gamma$ free, $\ge 6$ consecutive scales, a scale in both merged by inverse
variance; `report.joint_refit`, user 2026-11-11), not from the pilot refitted with $\gamma$
pinned to $\hat\gamma$, whose pin carries $\hat\gamma$'s scatter but not its bias (see
`src/study/README.md`). $\hat\gamma$ is unchanged; only B_fs and the eq. (720) half-width
move. Old = the 2026-11-10 pinned average (still in `answer.json` as `pilot_refit`), new =
`joint_refit`; reports rerun 2026-11-11. "best" is the lowest-AIC range of the union and
its share of the kept weight; its $\chi^2$ gives the warning (⚠). ✓/✗: the interval contains / misses the reference (one mark per reference; scoring only, ground rule 4).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `aic_explore_d5` | 1.202687 | 1.23 | 0.00632 ✗ | 0.453 ± 0.0821 | 4..512 (0.32) | -0.397 | 0.0866 | 0.0877 ✓ |

Coverage of every reference: old 0/1, new 1/1.
