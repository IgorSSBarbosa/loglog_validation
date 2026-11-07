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
| `gsusc_d2_x4_4h` | 2.370206 | 1.19 ± 0.055 | 18.1/3 ⚠ | 0.0185 | 1.1 ± 0.22 | 4..64 | 7.73/2 | 0.0216 |
