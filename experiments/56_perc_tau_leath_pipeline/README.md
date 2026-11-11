# Experiment 56 — the article's pipeline on $\tau$ by Leath growth, $d=2,3$

**Status: done (2026-11-11). Both eq. (wilson) intervals contain the reference value; in d=2 the replicate interval excludes it by about $B_{fs}$, so the finite-size bias is visible and the bound absorbs it.**

## Question (user, 2026-11-10)

Run the four-step pipeline (pilot, allocation, long run, Step 4 + eq. (wilson) interval), 9 h
per study, on `percolation_tau_leath` (experiment 55's sampler, one cluster a sample, so
Assumption `sigma_converges` is broken), for comparison with experiment 49's box sampler.

## Setup

| study | host | model | params | pilot ladder | seed |
|---|---|---|---|---|---|
| `tau_leath_d2_9h` | purus | `percolation_tau_leath` | dim 2, p = 0.5927460507921, K = 1 | $2^4..2^{16}$, $n=10^6$ | 2026111056 |
| `tau_leath_d3_9h` | missouri | `percolation_tau_leath` | dim 3, p = 0.3116077, K = 1 | $2^4..2^{16}$, $n=10^6$ | 2026111057 |

```bash
bash experiments/56_perc_tau_leath_pipeline/run_study.sh tau_leath_d2 9h 2026111056 9h --max-scale 16777216
bash experiments/56_perc_tau_leath_pipeline/run_study.sh tau_leath_d3 9h 2026111057 9h --max-scale 16777216
```

Hosts at launch (2026-11-10): purus load 1.5, missouri load 0.7, ~108 GB available on each.

Memory (measured 2026-11-10): a sample's peak is $32\,s$ bytes (table of 12-byte slots at
load $\le 1/2$, plus an 8-byte BFS queue), on every thread at once in the worst case. With
experiment 55's constants and throughput, 9 h gives $B\approx 6.7\cdot10^{11}$ (d=2) and
$4.5\cdot10^{11}$ (d=3) per replicate; prop:opt's window would end at $2^{22}$ / $2^{24}$
(3.8 / 15 GiB at 30 threads), the tuned one at $2^{16}$ / $2^{18}$ (0.1 / 0.2 GiB). Sample
arrays add $\le 8 n$ bytes a scale (59 MB at $n = 7.4\cdot10^6$). `--max-scale` $2^{24}$ caps
the window at 15 GiB.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d=2$: $2-\tau = -5/91 = -0.0549451$ (exact).
- $d=3$: $2-\tau = -0.18909(5)$ ($\tau = 2.18909(5)$); Mertens & Moore 2.1892(1).

Pass: the eq. (wilson) 95% interval contains the value. Compared with experiment 49
(box sampler, $1-\tau$): radius $\pm0.00172$ (d=2), $\pm0.00725$ (d=3).

## Results

Autopilot answers final (2026-11-10, 19:01 / 19:36); the Step 4 local-slope (A1) top-ups
were still drawing when these were recorded. Both **PASS**.

| | d = 2 | d = 3 |
|---|---|---|
| window, n × R | $2^{11}..2^{16}$, 7,542,414 × 5 | $2^{13}..2^{18}$, 5,714,618 × 5 |
| $\hat\gamma = 2-\hat\tau$ | −0.054872 ± 0.00028 | −0.18935 ± 0.00100 |
| eq. (wilson) 95% | [−0.05515, −0.05460] ∋ −0.054945 | [−0.19035, −0.18836] ∋ −0.18909 |
| $B_{fs}$ / 1.96σ_se | 6.4e-5 / 2.1e-4 | 1.86e-4 / 8.1e-4 |
| $B_{fs}$ at ω₁ ± 1 se | [3.8e-6, 1.25e-3] (×329) | [3.3e-5, 1.09e-3] (×32.5) |
| ω₁ refit (AIC range) | 0.91 ± 0.34 on $2^6..2^{16}$, χ² 1.7/8 | 0.73 ± 0.17 on $2^4..2^{16}$, χ² 8.6/10 |
| ω₁, a₁ pilot | 0.736 ± 0.041, −0.084 ± 0.008 | 0.658 ± 0.033, −0.225 ± 0.014 |
| $\hat\sigma^2_{\max}$ | 0.740 | 8.17 |
| pilot + long run | 1.65 h + 7.30 h, CPU 30 threads | 0.72 h + 8.67 h (+6 min cost re-fit), CPU 30 threads |

Radius against experiment 49 (box sampler, same 9 h): 5.8× (d=2) and 7.4× (d=3) narrower.
Interval incomplete (no ω₂ piece of $B_{fs}$), as for every run so far.

Final (2026-11-11, both jobs exited by 01:18; `report.md` after the Step 4 top-ups, numbers
unchanged from the table above):

| | d = 2 | d = 3 |
|---|---|---|
| $\hat\tau$ | 2.05487 ± 0.00028 | 2.18935 ± 0.00100 |
| reference | 187/91 = 2.054945 | 2.18909(5); M&M 2.1892(1) |
| Student-t replicate 95% ($\hat\gamma$) | [−0.05492, −0.05482] **∌ −0.054945** | [−0.18993, −0.18877] ∋ −0.18909 |
| closed-form sd / replicate sd | 1.08e-4 / 4.1e-5 | 4.12e-4 / 4.65e-4 |
| plan's prediction, error on the answer | 8.6e-5 | 1.9e-4 (exp 55 constants) |

- **d=2: the bias is real and the bound carries it.** $\hat\tau$ sits $7.5\cdot10^{-5}$ below 187/91,
  four replicate se; experiment 55's $m_0=10$ mean (2.054812 ± 0.000073) sits on the same side.
  The replicate interval excludes the truth; eq. (wilson) contains it, with $B_{fs}=6.4\cdot10^{-5}$
  matching the offset. The plan predicted $|{\rm bias}| = 7.9\cdot10^{-5}$.
- **The closed-form sd overestimates in d=2** by 2.6× (replicate spread from 5 carries ~35%):
  report.py's $\sigma^2_\infty$ is the top-of-window cv², and $\sigma_s^2$ grows (experiment 55), so the
  se term is conservative, as predicted. In d=3 the two agree.
- **d=3** is $2.6\cdot10^{-4}$ above 2.18909 and $1.5\cdot10^{-4}$ above Mertens & Moore, within the
  interval. With experiment 55's two windows (2.189394 ± 0.000154, 2.189468 ± 0.000311) all three
  sit near 2.1894, about 2 se above 2.18909 each; not resolved at this precision.
- **p_c systematic** (not in the bound): the tabulated $p_c=0.3116077(2)$, with the measured
  response $d\ln Q/dp \approx 3.0\,s^{0.4522}$ (kolyma, 2026-11-10, $s=2^8..2^{16}$), moves
  $\hat\gamma$ on $2^{13}..2^{18}$ by $3.8\cdot10^{-5}$ per $2\cdot10^{-7}$ of $p_c$ error: negligible here,
  but it grows like $s^{\sigma}$ with the window.
- `run_study.sh`'s last step, experiment 46's `plot_pipeline.py`, fails on both studies with
  `AssertionError: eq. (m0-star) disagrees with the plan` (its `tradeoff` recomputes $m_0^*$ and
  asserts it equals the plan's). The report is written before it and unaffected; no figure.
