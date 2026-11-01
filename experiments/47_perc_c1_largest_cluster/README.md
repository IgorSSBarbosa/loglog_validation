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
| estimate | $\hat\gamma=1.89589$, eq. (720) 95% **[1.89570, 1.89609]**, contains $91/48=1.89583$ (error $+6\times10^{-5}$) |
| replicate-$t$ | [1.89579, 1.89600], also contains it |

The correction on the 2-D torus is fast ($\omega_1\approx1.7$) and small ($a_1\approx-0.2$),
so the plan is variance-limited and both intervals cover. Local slopes 64..1024 sit
within $3.5\times10^{-4}$ of $91/48$ (local-slope pass, `local_slope.json`).
