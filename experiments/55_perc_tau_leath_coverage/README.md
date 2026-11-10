# Experiment 55 — coverage of eq. (720) for $\hat\tau$ by Leath growth, $d=2,3$, with Assumption sigma_converges broken

**Status: done (2026-11-08). Eq. (720) covers at both windows when $\sigma^2_\infty$ comes from the top of the window or above (as the pipeline's report.py takes it); with a pilot that stops below the window, the variance is underestimated by the predicted factor and coverage drops to 0.88 (d=2, below nominal) and 0.92 (d=3).**

## Question (user, 2026-11-08)

Measure $\tau$ with Mertens & Moore's sampler (arXiv:1806.08067, Sec. V: Leath growth from
the origin at $p_c$), and see what the estimator does when its assumptions fail. The
coverage test comes first; it can also serve as a pilot for a later pipeline run.

## Model

`percolation_tau_leath` (`models/percolation_tau_leath.py`). One sample at scale $s$ grows
$K=\lceil c\,s^{\kappa}\rceil$ fresh clusters from the open origin and returns the
fraction that reach $s$ sites, so $\mathbb E Y_s = Q(s) = P(|C(0)|\ge s)\sim s^{2-\tau}$ and
the fitted exponent is $\gamma = 2-\tau$ (the box models of 25/49 fit $1-\tau$).

Here $\kappa=0$, $K=1$: $Y_s$ is Bernoulli($Q(s)$) and
$\sigma_s^2=\mathrm{Var}(Y_s/Q)=(1-Q)/Q\to\infty$, which breaks Assumption
`sigma_converges` (`main.tex`, $\sigma_i^2\to\sigma_\infty^2\in(0,\infty)$).
Benchmark (purus, 2026-11-08, $2\cdot10^5$ samples a scale):

| | $\sigma_s^2$, $s=16 \to 2^{16}$ | reads exponent | local slope of $\log Q$ at $2^{16}$ |
|---|---|---|---|
| $d=2$ | 0.115 → 0.737 (0.875 at $2^{18}$) | 0.948 | −0.0544 |
| $d=3$ | 0.51 → 6.08 | 0.813 | −0.1916 |

Wall time grows faster than reads above $2^{16}$ in $d=2$ (×6.4 from $2^{16}$ to $2^{18}$
against ×3.7 in reads), likely cache misses once the table outgrows cache: a strain on
Assumption `cost_is_power_law` too.

## Design

`score_theorems.py` takes $\sigma^2_\infty$ as the pilot's $\mathrm{cv}^2$ at its LARGEST
scale. With $\sigma_s^2$ increasing, a sweep window inside the pilot's ladder is scored
with an overestimate; the broken assumption bites only once the window passes the
pilot's top scale. So each sweep has two $m_0$: one window ending at the pilot's top
($2^{16}$) and one two rungs past it ($2^{18}$), as experiment 25's $m_0=7,8$.

1. Pilot, 5 replicates, ladder $2^4..2^{16}$, $n=10^6$ (recipes `samples_pilot_d{2,3}.json`).
2. Sweep `sweep_coverage_d{2,3}.json`: $m=6$, $\rho=2$, $R=50$, $m_0\in\{10,12\}$,
   constants from the pilot's `constants.json`; budget chosen from the pilot's timing.
3. `calibration/score_theorems.py`: Var ratio (thm:CLT), coverage of eq. (720)
   (cor:wilson) with its Wilson CI, bound/RMSE.

```bash
D=experiments/55_perc_tau_leath_coverage
python3 src/study/pilot.py -meta $D/recipes/samples_pilot_d2.json --study pilot_d2 --replicates 5 --data-root $D/data
python3 src/study/pilot.py -meta $D/recipes/samples_pilot_d3.json --study pilot_d3 --replicates 5 --data-root $D/data
```

Host: purus (idle at launch, 2026-11-08; missouri was at load 30), pilots in sequence.

Short pilots and sweeps (user, 2026-11-08: two hosts). d = 2 on purus (load 0.08 at launch,
4.7 h), d = 3 on kolyma (load 0.04, 2.7 h):

```bash
python3 src/study/pilot.py -meta $D/recipes/samples_pilot_short_d2.json --study pilot_short_d2 --replicates 5 --data-root $D/data
python3 src/budget/allocation_experiment.py -meta $D/recipes/sweep_coverage_d2.json -o $D/data --tag coverage_d2
python3 calibration/score_theorems.py --sweep $D/data/coverage_d2/allocation_sweep.json --pilot $D/data/pilot_d2
# same sweep, short pilot: a symlinked copy, so the fixed-name theorem_scores.json is not overwritten
python3 calibration/score_theorems.py --sweep $D/data/coverage_d2_vs_short_pilot/allocation_sweep.json --pilot $D/data/pilot_short_d2
```

Short pilot (`samples_pilot_short_d{2,3}.json`): the full pilot's recipe cut to $2^4..2^{12}$,
seed + 100. It draws nothing for the sweep; it only changes what the sweep is scored against.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d=2$: $\tau=187/91$, so $2-\tau=-5/91=-0.0549451$ (exact). $\Omega=72/91$.
- $d=3$: $\tau=2.18909(5)$, so $2-\tau=-0.18909(5)$. Mertens & Moore Table 3: 2.1892(1).

Pass: coverage $\ge 95\%$ (Wilson CI) over $R=50$. Expected to fail at the $m_0=12$ window
if the broken assumption matters; the point is to measure by how much.

## Results

### Pilots (purus, 5 replicates, $n=10^6$)

| pilot | $d$ (timed) | $\omega_1$ | $a_1$ | $\sigma^2$ at top | pilot $\gamma$ |
|---|---|---|---|---|---|
| `pilot_d2` ($2^4..2^{16}$) | 0.966 ± 0.009 | 0.718 ± 0.073 | −0.085 ± 0.013 | 0.742 | −0.0551 |
| `pilot_short_d2` ($2^4..2^{12}$) | 1.001 ± 0.017 | 0.764 ± 0.068 | −0.087 ± 0.013 | 0.495 | −0.0549 |
| `pilot_d3` ($2^4..2^{16}$) | 0.843 ± 0.004 | 0.640 ± 0.019 | −0.221 ± 0.008 | 6.04 | −0.1897 |
| `pilot_short_d3` ($2^4..2^{12}$) | 0.844 ± 0.015 | 0.665 ± 0.077 | −0.234 ± 0.027 | 3.17 | −0.1895 |

Timed $d$ sits above the reads exponent $3-\tau$ (0.945 / 0.811), from cache misses at large $s$.
Every cost probe hit its cap, so each se(d) rests on one probe.

### Coverage sweeps ($m=6$, $\rho=2$, $R=50$)

Bias and sd are of $\hat\gamma = 2-\hat\tau$. Var ratio = observed / predicted Var($\hat\gamma$)
(thm:CLT), with a bootstrap 95% CI; coverage of eq. (720) with a Wilson 95% CI. In every cell
the bound is dominated by its se term.

| d | B | $m_0$ (window) | n | bias | sd obs | scored against | $\sigma^2_\infty$ | Var ratio | coverage | bound/RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2e10 | 10 ($2^{11}..2^{16}$) | 220,356 | +1.3e-4 | 5.1e-4 | full | 0.742 | 0.66 [0.47, 0.84] | 50/50 [0.93, 1] | 2.73 |
| 2 | 2e10 | 12 ($2^{13}..2^{18}$) | 57,720 | +1.9e-4 | 1.25e-3 | full | 0.742 | 1.02 [0.68, 1.37] | 48/50 [0.87, 0.99] | 2.01 |
| 2 | 2e10 | 10 | 220,356 | | | short | 0.495 | 0.99 [0.70, 1.26] | 49/50 [0.90, 1.00] | 2.18 |
| 2 | 2e10 | 12 | 57,720 | | | short | 0.495 | **1.53 [1.02, 2.06]** | **44/50 [0.76, 0.94] below nominal** | 1.63 |
| 3 | 1e10 | 10 ($2^{11}..2^{16}$) | 397,353 | −3.0e-4 | 1.09e-3 | full | 6.04 | 0.66 [0.41, 0.90] | 50/50 [0.93, 1] | 3.23 |
| 3 | 1e10 | 12 ($2^{13}..2^{18}$) | 123,504 | −3.8e-4 | 2.20e-3 | full | 6.04 | 0.83 [0.59, 1.07] | 50/50 [0.93, 1] | 2.35 |
| 3 | 1e10 | 10 | 397,353 | | | short | 3.17 | 1.26 [0.79, 1.71] | 50/50 [0.93, 1] | 2.44 |
| 3 | 1e10 | 12 | 123,504 | | | short | 3.17 | **1.58 [1.12, 2.03]** | 46/50 [0.81, 0.97] | 1.71 |

Mean $\hat\tau$ over the 50 replicates (± se of the mean):

| d | $m_0=10$ | $m_0=12$ | reference |
|---|---|---|---|
| 2 | 2.054812 ± 0.000073 | 2.054755 ± 0.000177 | 187/91 = 2.054945 |
| 3 | 2.189394 ± 0.000154 | 2.189468 ± 0.000311 | 2.18909(5); Mertens & Moore 2.1892(1) |

### Reading

- **The broken assumption acts through $\sigma^2_\infty$ only, and by the predicted amount.**
  thm:CLT's variance uses one $\sigma^2_\infty$, the scoring pilot's top-scale $\mathrm{cv}^2$. With
  $\sigma_s^2$ growing, the ratio should be about (window mean of $\sigma_s^2$) / $\sigma^2_\infty$.
  For $m_0=12$ against the short pilot that is $0.71/0.495 = 1.44$ (d=2) and $5.76/3.17 = 1.82$
  (d=3), with the window's $\sigma^2$ at $2^{17}, 2^{18}$ extrapolated from the pilot's $Q$.
  Measured: 1.53 and 1.58. For $m_0=10$ against the full pilot the prediction is
  $0.59/0.74 = 0.79$ and $4.20/6.04 = 0.70$; measured 0.66 and 0.66. Both CLT "FAIL"s in that
  row are this effect in the conservative direction.
- **Coverage tolerates it only partly.** The bound has slack (bound/RMSE 2–3 when $\sigma^2$
  is not underestimated), so a variance 1.5× too small costs only some coverage: 0.88 in d=2
  (Wilson CI excludes 0.95) and 0.92 in d=3 (consistent with 0.95).
- **The pipeline is on the safe side.** `src/study/report.py` takes $\sigma^2_\infty$ from the
  LONG RUN's cv at its own largest scale, the top of the window. With $\sigma_s^2$ increasing,
  that overestimates the window's variance, like the "full" rows: the se term is conservative.
  Only the short-pilot rows, which score a window against a $\sigma^2$ measured below it,
  can undercover.
- **Bias is small.** $|{\rm bias}| \le 3.8\cdot10^{-4}$, below the sd in every cell; B_fs is
  never the dominant term. $\hat\tau$ is 1.8 se below 187/91 in d=2 ($m_0=10$) and 2.0 se above
  2.18909 in d=3, where it is closer to Mertens & Moore's 2.1892(1).
- **Per budget.** At $m_0=10$, d=3 gives sd $1.1\cdot10^{-3}$ in 2.7 h of 32 threads, against
  experiment 49's box sampler, whose 95% radius is $7.3\cdot10^{-3}$ after 8.9 h. The two are not
  the same statistic (one sd at a fixed window vs a pipeline radius), so this is indicative only.
