# Experiment 53 — the article's pipeline on percolation: susceptibility exponent $\gamma$, $d=2$

**Status: running (launched 2026-11-05 on purus).**

## Question (user, 2026-11-05)

Fill the $d=2$ cell of the article's percolation table: experiment 51's pipeline
(pilot, allocation, long run, Step 4 + eq. (wilson) interval) on $\gamma_{\rm susc}$ in $d=2$.

## Setup

| study | model | params | pilot ladder | seed |
|---|---|---|---|---|
| `gsusc_d2_x4_4h` | `percolation_susceptibility_gpu` (x-ladder) | dim 2, $\varepsilon_0=0.5$, `box_factor` 4, $\nu_{\rm box}=4/3$, torus, moment 1 | x = 4..128, $n=10^4$ per rung | 2026110553 |

```bash
bash experiments/53_perc_gamma_susc_d2/run_study.sh gsusc_d2_x4 4h 2026110553 4h --max-scale 512
```

`--max-scale 512`: $L(1024)=41{,}285$, i.e. $1.7\cdot10^9$ sites, does not fit on a 32 GB card.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma = 43/18 = 2.38\overline{8}$ (exact).

Pass: the eq. (wilson) 95% interval contains 43/18.

## Results

Not yet.
