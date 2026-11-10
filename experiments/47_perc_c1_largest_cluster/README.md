# Experiment 47 — largest cluster $C_1$, Zhang et al.'s observable, for $d_f$

**Status: A1–A3 pass (2026-10-31). A4–A5 (5D) not run.**

## Question

Experiment 43 found the log-log slope fails D3 in $d=5$ on `face_far`: its leading
correction is slow ($\omega_1\approx0.4$–$0.6$) and large ($a_1/a_0\approx-1.1$), so
the bias at $L\le60$ is 0.09–0.24. Zhang, Hou, Fang, Hu & Deng, arXiv:2004.11289 (2020),
measure $d_f$ from $C_1$, the mean largest cluster on a periodic box, whose correction
they fit as $y_1=-2.0(2)$ in 5D. Does the log-log estimator, and the article's
allocation, work on $C_1$?

**Departure from PLAN.md ground rule 7** (face-connected count): the user's decision
for this experiment (2026-10-31).

## Model

`models/percolation_c1_gpu.py`, registered as `MODELS["percolation_c1_gpu"]`.
$Y_i$ = sites in the largest open cluster of an $i^{\rm dim}$ lattice at $p_c$,
`geometry` = `torus` by default. The draw, labelling, wrap merge and block size are
`percolation_zd_gpu`'s, imported. Only the count is new.

Differences from Zhang et al., by design:

- **Site, not bond percolation.** `cupyx.scipy.ndimage.label` is site-based, and
  labelling is ~90% of a sample's cost in 5D (experiment 43, 2026-10-31). Same
  universality class, so same $d_f$, but the correction amplitudes differ: $y_1$ is
  measured here, never imported.
- $p_c$ is `percolation_zd`'s site table (5D: 0.14079633(4), Mertens & Moore 2018).

Cost, dim 5, torus, purus RTX 5090, ms per sample:

| $L$ | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| `percolation_c1_gpu` | 0.005 | 0.172 | 9.92 | 267 |
| `percolation_zd_gpu` (`face_far`) | 0.005 | 0.143 | 5.94 | 293 |

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f(2)=91/48=1.89583$ (exact).
- $d_f(5)=3.5260(14)$ (Zhang et al. 2020, bond; same universality class).

## Acceptance

| id | check | criterion | status |
|---|---|---|---|
| A1 | count is exact | `largest_cluster` = an independent BFS on the same lattices: dim 2–5 × box/cylinder/torus × $p\in\{0,\tfrac12,1,2,4\}p_c$, 40 lattices per cell stacked in one block | **pass** (`tools/tests/test_percolation_c1_gpu.py`). A mismatched geometry fails 31–40 of 40 in 3D and 5D, so the test can tell |
| A2 | closed forms | $p=0\Rightarrow Y=0$, $p=1\Rightarrow Y=i^{\rm dim}$, every geometry | **pass** |
| A3 | known $d_f$, dim 2 | torus, the article's pipeline (`autopilot.py`): the eq. (720) 95% interval contains $91/48$ | **pass**, see below |
| A4 | 5D correction | pilot on Zhang's 5D ladder ($L$ = 4..60): fit eq. (232) and Zhang's eq. (5) and report $\omega_1$ / $y_1$ with errors. This is the measurement the experiment exists for, not a pass/fail | not run |
| A5 | 5D $d_f$ | Zhang's ansatz recovers 3.5260 within 2 se on the A4 data | not run |

```bash
.venv/bin/python -m pytest -q tools/tests/test_percolation_c1_gpu.py   # A1, A2
```

## A3 — result (`a3_c1_d2_1h`, missouri, 2026-10-31, seed 20261031, 71.0 min)

```bash
D=experiments/47_perc_c1_largest_cluster/data
python3 -u src/study/autopilot.py -meta experiments/47_perc_c1_largest_cluster/recipes/samples_pilot_c1_d2.json \
    --study a3_c1_d2_1h --data-root $D --time 1h --replicates 3 --seed 20261031 \
    --max-scale 32768 --yes --no-progress > $D/a3_c1_d2_1h.log 2>&1
```

| step | outcome |
|---|---|
| pilot | 3 replicates on 8..1024. $\hat d=2.188\pm0.006$, $\hat\omega_1=1.68\pm0.10$, $\hat a_1=-0.20\pm0.03$ |
| plan | $m_0=5$, scales 64..2048, $n=1{,}170{,}764$ ×3, 56.1 min, predicted bias $4.6\times10^{-5}$ |
| estimate | $\hat\gamma=1.89589$, eq:wilson 95% **[1.89571, 1.89608]**, contains $91/48=1.89583$ (error $+6\times10^{-5}$). Three-term bound of 2026-11-04; the four-term one gave [1.89570, 1.89609] |
| replicate-$t$ | [1.89579, 1.89600], also contains it |

The correction on the 2-D torus is fast ($\omega_1\approx1.7$) and small ($a_1\approx-0.2$),
so the plan is variance-limited and both intervals cover. Local slopes 64..1024 sit
within $3.5\times10^{-4}$ of $91/48$ (local-slope pass, `local_slope.json`).

## The article's d_f runs on C1 (user, 2026-11-05)

The article switched its $d_f$ observable from `face_far` to $C_1$. Queued in
`local/queue_47_c1_d4.txt`, launched by `local/after_53_launch_c1_d4.sh` once experiment 53
finishes on purus; every run follows the pipeline (pilot, plan, long run, Step 4, eq. (wilson)).

| study | recipe | pilot | long run | seed |
|---|---|---|---|---|
| `pipe_c1_d3_9h` | `samples_pipe_c1_d3.json` | 4..512, Neyman $2\cdot10^{11}$ (49's df_d3 design) | 9 h, `--max-scale 1024` | 2026110547 |
| `pipe_c1_d4_9h` | `samples_pipe_c1_d4.json` | 4, 6, 8, ..., 128 ($\sqrt2$-spaced), Neyman $2\cdot10^{12}$ | 9 h, `--max-scale 128` | 2026110549 |
| `pipe_c1_d5_9h` | `samples_pipe_c1_d5.json` | 4, 6, 8, ..., 64 ($\sqrt2$-spaced), Neyman $2\cdot10^{12}$ | 9 h, `--max-scale 64 --m 5` | 2026110548 |
| `zhang_c1_d5` | `samples_pipe_c1_d5.json` (pilot), `samples_zhang_c1_d5.json` (long run) | 4, 6, 8, ..., 64 ($\sqrt2$-spaced), Neyman $2\cdot10^{12}$ | **Zhang et al.'s 5D allocation**, 4..60, as 5 replicates of $n/5$ (pooled = Zhang's $10^6$/$10^5$), then `fixed_run.py report/plot` | 2026110550 / 51 |

The last repeats experiment 43's step 1 (face count) on $C_1$: Zhang's allocation is the long
run, so the two observables are compared on the same published allocation. Its ladder is not
$\rho^k$, which the pipeline's drivers require, so `fixed_run.py` does Step 3 and Step 4 on it:
OLS $\hat\gamma$ per window (eq. (526) is the OLS slope on a $\rho^k$ window; the headline is
the top 6 scales, the pipeline's $m$), eq. (232) with a cutoff scan, the pilot refit pinned to
$\hat\gamma$, and local slopes in pair form (`tools/local_slope.py`'s `*_pairs` functions,
user 2026-11-05). No eq. (wilson) interval: the theorem is proved for $\rho^k$ ladders only.
Checked before launch: on a $\rho^k$ ladder the pair functions equal the geometric ones (slopes,
se, CLT se, n-mismatch, and `fit_power`'s $\gamma$, $\omega$, $\chi^2$); on the noiseless Zhang
ladder they recover a planted $\gamma=1.9$, $a_1=-0.8$, $\omega=0.9$ to $10^{-8}$ ($\chi^2\sim10^{-22}$);
a 2D smoke run (`data/smoke_fixed`) went through draw, report and plot. $256^4$ and $128^5$
exceed the int32 labels, hence the caps. Known values: $d_f(3)=2.52293(10)$,
$d_f(4)=3.0446(7)$, $d_f(5)=3.5260(14)$.

A first launch of `pipe_c1_d3_9h` and `pipe_c1_d5_9h` (2026-11-05 12:57) was stopped by the
user after ~10 min to wait for 53; its partial output is `data/*_aborted*`.

## Wilson interval with the AIC pilot refit (2026-11-07)

B_fs now takes $\omega_1, a_1$ from the pilot refitted with $\gamma$ pinned to $\hat\gamma$ on a range of
$\ge 5$ consecutive pilot scales chosen by AIC with data cuts (Jay & Neil 2021;
`local_slope.omega_ranges`, `report.pinned_refit`), not on every pilot scale as in the
2026-11-02 refit. $\hat\gamma$ is unchanged; only the eq. (720) half-width moves. The
$\omega_1$ error is the profile error $\oplus$ the spread over ranges. Old = all-scales pinned
refit (kept in `answer.json` as `pilot_refit_all_scales`), new = AIC refit (`pilot_refit`);
reports rerun 2026-11-07. A warning is $\chi^2$ $p < 0.01$ (old: or $\omega_1$ on its bound).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old $\chi^2$/dof | old radius | new $\hat\omega_1$ | range | new $\chi^2$/dof | new radius |
|---|---|---|---|---|---|---|---|---|
| `a3_c1_d2_1h` | 1.895895 | 1.67 ± 0.048 | 1.02/5 | 0.000185 | 1.67 ± 0.11 | 8..1024 | 1.02/5 | 0.000185 |
| `pipe_c1_d3_9h` | 2.523407 | 2.2 ± 0.067 | 1.13/5 | 0.00124 | 2.2 ± 0.096 | 4..512 | 1.13/5 | 0.00124 |
| `pipe_c1_d4_9h` | 3.067902 | 3.89 ± 0.06 | 1.03e+03/8 ⚠ | 0.0147 | 3.54 ± 0.053 | 4..16 | 71/2 ⚠ | 0.0152 |
| `pipe_c1_d5_9h` | 3.587979 | 4.3 ± 0.14 | 304/6 ⚠ | 0.0291 | 4.18 ± 0.13 | 4..16 | 60.1/2 ⚠ | 0.0288 |

## Wilson interval with the AIC-weighted refit (2026-11-10)

$\omega_1$ is now the AIC-weighted average over the kept ranges (Jay & Neil 2021, eq. 15),
not the single highest-weight range; its error is their eq. (18). $a_1$ is refitted on the
highest-weight range with $\omega_1$ fixed at that average, so B_fs takes one coherent
$(a_1, \omega_1)$ (averaging $a_1$ separately pairs values from different fits). $\hat\gamma$
is unchanged; only B_fs and the eq. (720) half-width move.
Old = the 2026-11-07 highest-weight range, new = the average; reports rerun 2026-11-10.
"best" is the highest-weight range and its share of the kept weight; its $\chi^2$ still
gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `a3_c1_d2_1h` | 1.895895 | 1.67 | 0.000185 | 1.67 ± 0.336 | 8..1024 (0.40) | -0.2 | 5.77e-05 | 0.000185 |
| `pipe_c1_d3_9h` | 2.523407 | 2.2 | 0.00124 | 2.24 ± 0.33 | 4..512 (0.42) | -1.12 | 0.000584 | 0.0012 |
| `pipe_c1_d4_9h` | 3.067902 | 3.54 | 0.0152 | 3.54 ± 0.053 | 4..16 (1.00) ⚠ | -8.51 | 0.0141 | 0.0152 |
| `pipe_c1_d5_9h` | 3.587979 | 4.18 | 0.0288 | 4.18 ± 0.13 | 4..16 (1.00) ⚠ | -27.3 | 0.0255 | 0.0288 |
