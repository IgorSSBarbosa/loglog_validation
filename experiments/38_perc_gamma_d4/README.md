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

## AIC-quality check on all existing data (user, 2026-11-07)

Every dataset with these exact params was pooled (duplicates dropped; the sets agree at shared scales, chi2 p = 0.53) and fitted by `tools/local_slope.omega_ranges` (CLT se). Pooled 2..256 (38 + 09): every closed range is rejected (chi2 61.5/4); the local slopes zig-zag 1.4307, 1.4367, 1.4212, 1.4264, 1.4385 at 5-8 sigma, which no smooth correction fits. Suspected (not verified): the ceil'd torus side L = ceil(4 x^0.69).

### Pilot `aic_explore_d4` (launched 2026-11-07 on niagara (CPU, 32 threads), seed 2026110838)

`percolation_susceptibility_explore`, no box, x = 4..512. A first launch on tarim failed at import (tarim runs Python 3.12; the .venv is 3.10).

```bash
.venv/bin/python src/study/pilot.py -meta experiments/38_perc_gamma_d4/recipes/samples_gsusc_d4_explore.json \
    --study aic_explore_d4 --data-root experiments/38_perc_gamma_d4/data --replicates 5 --seed 2026110838
```

Result (2026-11-07): weighted linear fit over 4..512, gamma = 1.4306 +- 0.0010, chi2 = 6.31/6 (p = 0.39); residuals flat (max |res/se| 1.86), so AIC finds no omega (every range open or on a bound). The torus zig-zag is absent. p_c's error (0.19688561(3)) biases this ladder by < 1e-4 at 3 sigma.

### Long run `aic_explore_d4`, 3 h on the pilot's rungs (user, 2026-11-07)

Same 8 rungs, to see whether a finite-size correction appears at ~6x the pilot's precision (expected se(gamma) ~ 0.0004). Launched on missouri, seed 2026110738.

```bash
.venv/bin/python src/study/plan.py --study aic_explore_d4 --data-root experiments/38_perc_gamma_d4/data \
    --time 3h --replicates 5 --m 8 --max-scale 512 --accept      # n = 32,477,449 per scale
.venv/bin/python src/study/run.py --study aic_explore_d4 --data-root experiments/38_perc_gamma_d4/data --seed 2026110738
```

plan.py's |bias| 0.0192 comes from the pilot's eq. (232) fit, whose omega is unidentified; the flat residuals contradict it, and this run tests it.

Result (2026-11-08, 5 x ~3285 s on missouri, 4.6 h, not the planned 3 h: the plan's throughput was niagara's): **a correction appears.** The pure power law is rejected, chi2 175/6 (gamma = 1.4313 +- 0.0002 there means nothing); residuals are concave, local slopes 1.4401, 1.4341, 1.4258, 1.4289, 1.4264, 1.4313, 1.4278: the drift is at x <= 32, flat after. CLT se agrees with the replicate spread. AIC: best closed range 4..512, omega = 1.65 +- 0.41 stat +- 0.24 syst, **gamma = 1.4276 +- 0.0007 +- 0.0009**, chi2 7.9/4; 16..512 and 8..512 carry most weight but have open omega (correction already spent there), gamma 1.4291 +- 0.0025 and 1.4278 +- 0.0006. Literature 1.430(6): inside. omega in x (= omega*nu) above the literature's 0.64-0.84. Figure: data/aic_explore_d4/loglog_residuals_run3h.png.

## Ladder robustness: is the flat log-log a lucky ladder? (user, 2026-11-07)

The pilot's ladder is eps = p_c - p = 0.09 / x. The question: would another first p (eps0) or another exponent, x = (eps/eps0)^(-k), still give no visible correction? `percolation_susceptibility_explore` gained `ladder_exponent` k (default 1, the old ladder), so eps = eps0 x^(-1/k) and log Ybar has slope gamma/k in log x. Every gamma below is k x slope. Seven ladders on the same x = 4..512, one independent replicate each (seeds 2026110741..47 in run order base, half, ugly, large, k1.2, k1.1_e0.15, k0.9). n is 2x the pilot's per-replicate n (1x for k0.9). Drawn on kolyma (30 threads), 22:25-23:28, 63 min in total.

```bash
for name in base half ugly large k1.2 k1.1_e0.15 k0.9; do       # seed 2026110741, +1 per ladder
  NUMBA_NUM_THREADS=30 .venv/bin/python src/generate/generate.py \
      -meta experiments/38_perc_gamma_d4/recipes/samples_ladder_$name.json --tag ladder/$name --seed $SEED
done
.venv/bin/python experiments/38_perc_gamma_d4/ladder_sweep.py    # tables below + both figures
```

| ladder | eps0 | k | eps window | straight-line gamma | chi2/6 | p |
|---|---|---|---|---|---|---|
| pilot (5 reps) | 0.09 | 1 | 1.8e-4 - 2.3e-2 | 1.4306 +- 0.0010 | 6.3 | 0.39 |
| base | 0.09 | 1 | 1.8e-4 - 2.3e-2 | 1.4299 +- 0.0016 | 12.8 | 0.05 |
| half | 0.0636 | 1 | 1.2e-4 - 1.6e-2 | 1.4260 +- 0.0017 | 6.0 | 0.42 |
| ugly | 0.1234 | 1 | 2.4e-4 - 3.1e-2 | 1.4311 +- 0.0014 | 13.2 | 0.04 |
| large | 0.18 | 1 | 3.5e-4 - 4.5e-2 | 1.4333 +- 0.0013 | 18.5 | 0.01 |
| k1.2 | 0.09 | 1.2 | 5.0e-4 - 2.8e-2 | 1.4311 +- 0.0015 | 11.9 | 0.06 |
| k1.1_e0.15 | 0.15 | 1.1 | 5.2e-4 - 4.3e-2 | 1.4322 +- 0.0013 | 7.9 | 0.24 |
| k0.9 | 0.09 | 0.9 | 8.8e-5 - 1.9e-2 | 1.4251 +- 0.0023 | 7.7 | 0.26 |

Result: taken alone, every ladder looks unbiased. AIC finds no omega on 7 of 8 (`ugly`: omega 1.21 [0.50, 2.55]). Taken together, they do not agree:

- The 8 gammas are inhomogeneous: chi2 19.2/7, p = 0.008. They follow the window: gamma against the window's mean log eps has slope 0.0045 +- 0.0012 (3.7 sigma). Windows nearer p_c give the lower gamma.
- One pure power law in eps for all ladders (one amplitude per ladder) is rejected: chi2 103.7/55, p = 8e-5. Adding a eps^Delta with Delta FIXED (the data cannot fit it) lowers chi2 by 34 for one parameter:

  | Delta (fixed) | gamma | a | chi2/54 | p |
  |---|---|---|---|---|
  | 0.5 | 1.4232 +- 0.0014 | -0.24 +- 0.04 | 72.5 | 0.05 |
  | 0.7 | 1.4250 +- 0.0011 | -0.37 +- 0.06 | 69.4 | 0.08 |
  | 0.85 | 1.4258 +- 0.0009 | -0.54 +- 0.09 | 67.3 | 0.11 |
  | 1.0 | 1.4263 +- 0.0009 | -0.82 +- 0.13 | 65.5 | 0.13 |

- At Delta = 0.7 that correction biases the pilot ladder's straight-line gamma by +0.0047. The pilot's 1.4306 is a ladder where this bias, too small to curve 8 points at se 0.001, lands on the average. Its flat residuals were not luck in the sense of a special eps0: `base` (same ladder, new seed) agrees with it. They are the ladder's lack of power.

Caveats: Delta comes from the literature's omega nu (0.64-0.84, "Known values"), not from these data. It is a diagnostic and enters no estimator. Each new ladder is one replicate with CLT se, and their chi2 run high (sum 78/42 against the pilot's 6.3/6), so those se may be slightly small, which would weaken the 3.7 sigma. Both 1.4306 and ~1.425 lie inside the literature's 1.430(6). The 3 h long run on the pilot's ladder (above) should show the curvature directly if it is real.

![local slopes on 8 ladders](images/ladder_local_slopes.png)

Each panel: one ladder's local slopes x k (CLT se) against the rung's mean eps. Black line: the pooled Delta = 0.7 fit's prediction gamma - a Delta eps^Delta. Band: literature 1.430(6).

![straight-line gamma against window](images/ladder_gamma_vs_window.png)

Filled: each ladder's straight-line gamma at the geometric mean eps of its window. Open: what the pooled correction predicts that ladder's straight line gives. Dashed: the pooled fit's gamma.

## Wilson interval with the joint refit (2026-11-11)

B_fs now takes $\omega_1, a_1$ from one AIC range fit on the pilot **and** the long run
together ($\gamma$ free, $\ge 6$ consecutive scales, a scale in both merged by inverse
variance; `report.joint_refit`, user 2026-11-11), not from the pilot refitted with $\gamma$
pinned to $\hat\gamma$, whose pin carries $\hat\gamma$'s scatter but not its bias (see
`src/study/README.md`). $\hat\gamma$ is unchanged; only B_fs and the eq. (720) half-width
move. Old = the 2026-11-10 pinned average (still in `answer.json` as `pilot_refit`), new =
`joint_refit`; reports rerun 2026-11-11. "best" is the lowest-AIC range of the union and
its share of the kept weight; its $\chi^2$ gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `aic_explore_d4` | 1.429813 | — | — | 1.4 ± 0.495 | 4..128 (0.53) | -0.11 | 0.00307 | 0.005 ✓ |
