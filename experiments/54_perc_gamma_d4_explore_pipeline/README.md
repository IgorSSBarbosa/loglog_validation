# Experiment 54 — the article's pipeline on $\gamma$, $d=4$, by exploration, pilot on $x = 2..64$

**Status: done (2026-11-08). eq. (720) bound [1.42652, 1.43069] contains the literature 1.430(6).**

## Question (user, 2026-11-08)

Run the full pipeline (pilot → plan → run → report) for the susceptibility exponent $\gamma$ in
$d=4$ with `percolation_susceptibility_explore` (no box), with a pilot on the small scales
$x = 2..64$, where experiment 38's 3 h run showed the finite-size correction lives (local slopes
drift only for $x \lesssim 32$). The pilot must pin $\omega_1$ to 10%, since $\omega_1$ sets B_fs
in the eq. (720) bound; 38's own pilot, on 4..512 at lower precision, saw no correction and
left the bound unformed.

## Setup

| | |
|---|---|
| model | `percolation_susceptibility_explore`, dim 4, $\varepsilon_0 = 0.09$ ($p = p_c - \varepsilon_0/x$, $p_c = 0.19688561(3)$), one seed |
| pilot ladder | $x = 2, 4, 8, 16, 32, 64$ (6 rungs: 5 is the pinned refit's minimum, but 6 lets the $\gamma$-free AIC run and halves $\sigma(\omega)$) |
| pilot n | $1.5\cdot10^8$ per rung per replicate, 5 replicates (~1 h, user) |
| seed | 2026110854 |

**Sizing** (2026-11-08). Fisher information of eq. (232) at $J=1$,
$\log\overline Y = A + \gamma\log x + a_1 x^{-\omega}$, with se$(\log\overline Y)$ = cv$/\sqrt N$,
cv from 38's run (4.68 at $x=2$ to 12.6 at 64), $a_1 = -0.131$ and $\omega = 1.65 \pm 0.47$ from
38's 3 h run's AIC fit. Samples per rung, in total over replicates, for $\sigma(\omega)/\omega = 10\%$:

| $\omega$ | $\gamma$ free | $\gamma$ pinned |
|---|---|---|
| 1.19 | $1.4\cdot10^8$ | $1.7\cdot10^7$ |
| 1.65 | $1.5\cdot10^8$ | $2.9\cdot10^7$ |
| 2.12 | $2.3\cdot10^8$ | $6.9\cdot10^7$ |

The pilot draws $7.5\cdot10^8$ per rung: Fisher is a lower bound for an exact model, and AIC's
cuts and $\omega_2$ at $x=2$ cost more. Memory ~7 GB per rung (48 bytes a sample).

```bash
.venv/bin/python src/study/pilot.py -meta experiments/54_perc_gamma_d4_explore_pipeline/recipes/samples_pilot.json \
    --study gsusc_d4 --data-root experiments/54_perc_gamma_d4_explore_pipeline/data --replicates 5 --seed 2026110854
```

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma = 1.430(6)$. Pass: the eq. (720) 95% interval contains it.
- Correction on the $\varepsilon$ ladder: $\Delta_1 = \omega\nu \approx 0.64$–$0.84$ (see 38's README).

## Results

Pilot `gsusc_d4` (2026-11-08, missouri, 5 x ~344 s = 29 min drawn, cost probe 44 s of an
861 s cap). Local slopes 1.4623, 1.4404, 1.4307, 1.4292, 1.4280 (se 4e-5 to 9e-5): the
correction is plain.

| fit | range | $\omega_1$ | $a_1$ | $\chi^2$/dof | $\gamma$ |
|---|---|---|---|---|---|
| eq. (232), pilot (replicate spread) | 2..64 | **1.464 ± 0.068 (4.7%)** | −0.1063 ± 0.0021 | | 1.4270 |
| AIC, $\gamma$ free, ≥ 6 points | 2..64 (only range) | 1.464 ± 0.076 (5.2%) | −0.106 | 3.7/2 | 1.4270 ± 0.0005 |
| AIC, ≥ 5 points (as the pinned refit uses) | 4..64 (w 0.56) | 2.25 ± 0.70 (31%) | −0.240 | 0.35/1 | 1.4283 ± 0.0010 |

The 10% target is met by the pilot fit that plans the run. The ≥ 5-point AIC, which is how
report.py's pinned refit chooses its range for B_fs, prefers 4..64 (dropping x = 2) and gives
$\omega_1$ to only 31%: with 5 points that range has 1 dof. $d = 1.597 \pm 0.018$.

plan.py preview (scratch copy, not accepted), --time 3h --max-scale 512: m0 = 3, scales 16..512,
n = 3.45e7, |bias| 4.6e-4, sd 4.6e-4 per replicate, 5.1e-4 on the answer.

### Long run (user, 2026-11-08: 10 h)

plan.py's throughput (5.07e8, from the 2..64 cost probe) is ~1.6x optimistic at 256..512, where
the explore model hits its cache wall: 38's run on missouri drew 32.5M per scale up to 512 in
3285 s per replicate. `--throughput` is set from that measurement so 10 h is wall clock.

```bash
D=experiments/54_perc_gamma_d4_explore_pipeline/data
.venv/bin/python src/study/plan.py --study gsusc_d4 --data-root $D --time 10h --replicates 5 \
    --max-scale 512 --throughput 314057929 --accept     # m0 = 3, 16..512, n = 71,374,323
.venv/bin/python src/study/run.py --study gsusc_d4 --data-root $D --seed 2026110810
.venv/bin/python src/study/report.py --study gsusc_d4 --data-root $D
```

Predicted eq. (720) half-width (B_fs from the pilot fit): 0.0026 (B_fs 0.0006, 1.96 sigma_se 0.0020).
Result (602.7 min drawn against 600.0 predicted, ratio 1.00x with the missouri throughput):

| | value |
|---|---|
| gamma-hat (eq. 526, pooled) | 1.42861 |
| per replicate | 1.42811, 1.42816, 1.42871, 1.42838, 1.42967 |
| Student t(4), scatter only | [1.42781, 1.42940] (+- 0.00080) |
| **eq. (720) bound** | **[1.42652, 1.43069] (+- 0.00208)**, dominated by the se term |
| 1.96 sigma_se | 0.00199 (sigma_se 0.00102 closed form; replicate sd 0.00064) |
| B_fs | 8.0e-05, pinned AIC refit of the pilot on 4..64 (omega1 2.49 +- 0.40, a1 -0.316, chi2 0.60/2, p 0.74) |
| B_good | 1.1e-05 |

Contains gamma = 1.430(6) (literature central value inside). The t interval excludes 1.430, so the scatter-only interval would have failed if 1.430 were exact; the bound did not. B_fs over omega1 +- 1 se spans 2.5e-05..2.6e-04: even its top is 13 % of the se term, so the bound is insensitive to omega1 here (half-width <= 0.0023). Predicted half-width 0.0026; measured 0.0021 because cv at 512 came out lower than the plan assumed. The pinned refit chose 4..64, not 2..64 (x = 2 cut by AIC). Bound marked incomplete: the omega2 piece of B_fs (needs phi_plus) is missing. Reading the final data alone, the eq. (232) fit does not converge (omega1 -> 0.03): over 16..512 the correction is below the noise, consistent with B_fs ~ 1e-4.

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
| `gsusc_d4` | 1.428606 | 2.47 | 0.00209 ✓ | 2.67 ± 0.554 | 4..512 (0.56) | -0.393 | 5.93e-05 | 0.00206 ✓ |

Coverage of every reference: old 1/1, new 1/1.
