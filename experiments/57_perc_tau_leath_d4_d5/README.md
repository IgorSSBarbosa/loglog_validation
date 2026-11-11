# Experiment 57 — the article's pipeline on $\tau$ by Leath growth, $d=4,5$

**Status: running (2026-11-11).**

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

Not run yet.
