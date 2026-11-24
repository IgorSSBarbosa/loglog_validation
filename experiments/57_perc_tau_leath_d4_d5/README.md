# Experiment 57 — the article's pipeline on $\tau$ by Leath growth, $d=4,5$

**Status: done (2026-11-12). Both eq. (wilson) intervals contain the literature value, but they are 10× (d=4) and 100× (d=5) wider than the literature's errors. The planner predicted 3× and 7× narrower scatter than measured: it plans on the pilot's MEAN cv, and $\sigma_s^2$ grows (Assumption `sigma_converges` broken).**

## Question (user, 2026-11-11)

Experiment 56's pipeline (pilot, allocation, long run, Step 4 + eq. (wilson)), 9 h per study,
for $d=4$ and then $d=5$, on purus only. The box sampler gave no bound in $d=4,5$
(`exponents_report.md`: $(\omega_1, a_1)$ unidentified by the pilots).

## Setup

| study | host | params | pilot ladder | seed |
|---|---|---|---|---|
| `tau_leath_d4_9h` | purus | dim 4, p = 0.19688561, K = 1 | $2^4..2^{16}$, $n=10^6$ | 2026111158 |
| `tau_leath_d5_9h` | purus, after d4 | dim 5, p = 0.14079633, K = 1 | $2^4..2^{16}$, $n=10^6$ | 2026111159 |

```bash
nohup setsid bash experiments/57_perc_tau_leath_d4_d5/queue.sh > /dev/null 2>&1 &
```

Tests: `tools/tests/test_percolation_tau_leath.py` extended to dims 4, 5 (exact $Q_2$, $Q_3$,
reference BFS, degenerate p); 19 pass (2026-11-11).

Benchmark (purus, 2026-11-11, 30 threads):

| d | s | $Q$ | $\sigma_s^2$ | µs/sample (wall) | peak / sample |
|---|---|---|---|---|---|
| 4 | $2^{12}$ / $2^{16}$ / $2^{20}$ | 0.0951 / 0.0400 / 0.0159 | 9.5 / 24 / 62 | 6.7 / 45.5 / 660 | 32 MiB at $2^{20}$ |
| 5 | $2^{12}$ / $2^{16}$ / $2^{20}$ | 0.0465 / 0.0150 / 0.0033 | 20 / 66 / 302 | 4.5 / 25.4 / 207 | 32 MiB at $2^{20}$ |

Memory: $32\,s$ bytes per thread; `--max-scale` $2^{24}$ caps it at 15 GiB (purus: 107 GB
available, load 0.01 at launch). Key range $\pm2^{14}$ (d=4), $\pm2^{11}$ (d=5) per axis, against a
typical radius $s^{1/d_f}\approx 110$ at $s=2^{24}$, d=5; an edge hit raises.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d=4$: $\tau = 2.3142(5)$, $2-\tau = -0.3142(5)$.
- $d=5$: $\tau = 2.4175(2)$, $2-\tau = -0.4175(2)$.

Pass: the eq. (wilson) 95% interval contains the value.

## Results

Queue finished 2026-11-12 05:03; `report.md` of each study. Both **PASS**.

| | d = 4 | d = 5 |
|---|---|---|
| pilot | 18.7 min, 1 round | 12 min, 1 round |
| ω₁, a₁ pilot (γ free) | 0.418 ± 0.008, −0.350 ± 0.007 | 0.315 ± 0.026, −0.514 ± 0.015 |
| window, n × R | $2^{16}..2^{21}$ ($m_0=15$), 2,843,688 × 5 | $2^{19}..2^{24}$ ($m_0=18$, at `--max-scale`), 1,551,466 × 5 |
| $\hat\tau$ (eq. (wilson) 95%) | **2.3144 ± 0.0050** ∋ 2.3142(5) | **2.4150 ± 0.0255** ∋ 2.4175(2) |
| replicate t(4) 95%, $\hat\tau$ | [2.31262, 2.31619] ∋ | [2.40914, 2.42081] ∋ |
| $B_{fs}$ / 1.96σ_se | 1.5e-3 (ω₁ = 0.468 AIC avg) / 3.4e-3 | 1.12e-2 (ω₁ = 0.30, pilot + long run) / 1.38e-2 |
| sd of one estimate: plan / closed form / replicate spread | 5.4e-4 / 1.75e-3 / 1.44e-3 | — / 7.0e-3 / 4.7e-3 |
| plan's error on the answer | 7.8e-4 | 1.6e-3 |
| σ²∞ used: plan (pilot mean cv²) / report (top of window) | 6.9 / 73 | 14.6 / 646 |
| wall clock, long run | 506 min | 475 min |

Reading:

- **Accurate, not precise.** $\hat\tau$ is $2\cdot10^{-4}$ (d=4) and $2.5\cdot10^{-3}$ (d=5) from the literature,
  inside both intervals. The half-widths are 10× and 100× the literature's errors.
- **The broken assumption now costs the plan, not the coverage.** `plan.py` takes $\sigma^2_\infty$ as
  the pilot's mean cv² over its 13 scales; report.py takes the long run's cv² at the top of the window.
  With $\sigma_s^2\sim s^{\tau-2}$ growing fast in $d=4,5$, the two differ by 10× and 44×. The
  measured replicate spread sides with the report (closed form 1.2× and 1.5× the spread, conservative,
  as experiment 55 found), so the plan's predicted scatter was 2.7× (d=4) too optimistic and its
  allocation was chosen on the wrong variance. In $d=2,3$ the same gap was small (σ² grows like
  $s^{0.055}$, $s^{0.19}$), which is why experiment 56's plans held.
- **Slow corrections.** ω₁ ≈ 0.42–0.47 (d=4) and 0.30 (d=5), against 0.70 in d=3, with larger
  $|a_1|$: the plan pushed the window up, to the $2^{24}$ cap in d=5, and $B_{fs}$ is no longer
  negligible (30% of the half-width in d=4, 45% in d=5).
- Cost: the planned windows sat above the pilot's ladder, so the autopilot timed them first
  (d=4: 1.65× the predicted time, $d=0.91$) and re-planned; d=5 cut n from 2.73e6 to 1.55e6.
- `plot_pipeline.py` failed on d=4 with experiment 56's `AssertionError`; d=5 wrote its figure.

## d=4 re-reported with the joint refit (2026-11-12)

The d=4 report was written by the code loaded at launch, before commit `bb5de3e`, so its
B_fs came from the pilot refitted with $\gamma$ pinned. `report.py` rerun on purus: B_fs now
from the joint pilot + long-run AIC fit, as in d=5 and the article's Step 5. $\hat\gamma$ is unchanged.

| | old (pinned pilot) | new (joint) |
|---|---|---|
| $\hat\omega_1$ | 0.468 ± 0.058 | 0.459 ± 0.123, best 16..65536 (0.14), χ² 1.5 / 9 |
| $\hat a_1$ | −0.359 | −0.360 |
| B_fs | 1.5e-3 | 1.68e-3 |
| radius | 0.0050 ✓ | 0.00515 ✓ |

The new report warns that B_fs spans [3.5e-4, 8.7e-3] over $\hat\omega_1$ ± 1 se (×24.5);
d=5 spans [5.3e-3, 2.4e-2] (×4.5). Both rows are in the article's percolation tables.

Coverage count (src/study/README.md, 2026-11-11: 64 of 67 joint, 59 pinned): both runs cover
their reference under the joint and the pinned fit, so it is now 66 of 69 against 61. The
article's Step 5 paragraph quotes the new count.
