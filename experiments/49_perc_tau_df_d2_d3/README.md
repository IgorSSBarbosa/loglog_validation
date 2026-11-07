# Experiment 49 — the article's pipeline on percolation: $\tau$ and $d_f$ in $d=2,3$

**Status: queued (2026-10-31).**

## Question (user, 2026-10-31)

Measure $\tau$ and $d_f$ in $d=2$ and $d=3$ with the four-step pipeline of the
article's Section "Simulations" (pilot, budget allocation, long run, Step 4 checks +
eq. (wilson) interval), 9 h per study. Four studies, each on its own host.

## Setup

| study | model | params | pilot ladder | from |
|---|---|---|---|---|
| `df_d2` | `percolation_zd_gpu` | dim 2, `face_far`, cylinder | 8..1024, neyman $4\times10^{11}$ | 27's design at dim 2, 20's ladder |
| `df_d3` | `percolation_zd_gpu` | dim 3, `face_far`, cylinder | 4..512, neyman $2\times10^{11}$ | 27's pilot recipe |
| `tau_d2` | `percolation_tau_gpu` | tail, torus, `box_exponent` 48/91 | 8..8192, neyman $4\times10^8$ | 25's `box_df` recipe (the one whose bound covered) |
| `tau_d3` | `percolation_tau_zd` (CPU) | dim 3, bin, torus, default box rule | 8..4096, neyman $2.88\times10^7$ | 32's pilot recipe |

`tau_*` fit $\gamma=1-\tau$. Pipeline: `run_study.sh <study> 9h <seed>`, i.e.
`autopilot.py --time 9h --replicates 5`, then `46/plot_pipeline.py`. `df_d3` keeps 27's
`--max-scale 1024` (one sample must fit on the card).

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f(2)=91/48=1.895833$, $\tau(2)=187/91=2.054945$ (exact).
- $d_f(3)=2.52293(10)$, $\tau(3)=2.18909(5)$ (literature; see 27 and 32).

Pass: the eq. (wilson) 95% interval contains the value. Reported beside it: the
replicate-$t$ interval, and each Step 4 check.

## Results

Not run yet.

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
| `df_d2_9h_refit` | 1.896069 | 0.733 ± 0.025 | 7.45/5 | 0.00126 | 0.72 ± 0.055 | 8..128 | 0.0421/2 | 0.00131 |
| `df_d2_smoke` | 1.896441 | 0.69 ± 0.036 | 8.24/5 | 0.00275 | 0.759 ± 0.38 | 8..128 | 0.12/2 | 0.00227 |
| `df_d3_9h` | 2.521996 | 2.03 ± 0.072 | 3.94/5 | 0.00346 | 2.03 ± 0.14 | 4..512 | 3.94/5 | 0.00346 |
| `df_d3_smoke` | 2.523749 | 1.93 ± 0.073 | 20.2/5 ⚠ | 0.012 | 3.8 ± 2.9 | 8..128 | 4.89/2 | 0.0102 |
| `tau_d2_9h_rerun` | -1.054830 | 0.643 ± 0.01 | 37.2/8 ⚠ | 0.0021 | 0.729 ± 0.15 | 16..8192 | 6.98/7 | 0.00172 |
| `tau_d2_smoke` | -1.053865 | 0.548 ± 0.017 | 51.1/8 ⚠ | 0.0112 | 0.88 ± 0.12 | 32..2048 | 4.55/4 | 0.00581 |
| `tau_d3_9h` | -1.188980 | 0.596 ± 0.027 | 11.7/7 | 0.0103 | 0.828 ± 0.27 | 32..4096 | 6.04/5 | 0.00725 |
| `tau_d3_smoke` | -1.178067 | 0.689 ± 0.052 | 17.1/7 | 0.0279 | 1.48 ± 0.51 | 32..4096 | 3.1/5 | 0.0209 |
