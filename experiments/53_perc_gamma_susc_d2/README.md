# Experiment 53 — the article's pipeline on percolation: susceptibility exponent $\gamma$, $d=2$

**Status: torus study done (2026-11-05), interval misses 43/18; exploration study stopped (2026-11-07) to investigate its cost.**

## Question (user, 2026-11-05)

Fill the $d=2$ cell of the article's percolation table: experiment 51's pipeline
(pilot, allocation, long run, Step 4 + eq. (wilson) interval) on $\gamma_{\rm susc}$ in $d=2$.

## Setup

| study | model | params | pilot ladder | seed |
|---|---|---|---|---|
| `gsusc_d2_x4_4h` | `percolation_susceptibility_gpu` (x-ladder) | dim 2, $\varepsilon_0=0.5$, `box_factor` 4, $\nu_{\rm box}=4/3$, torus, moment 1 | x = 4..128, $n=10^4$ per rung | 2026110553 |
| `gsusc_d2_explore_9h_missouri` | `percolation_susceptibility_explore` (x-ladder, CPU) | dim 2, $\varepsilon_0=0.5$, one seed on $\mathbb Z^2$, no box, `max_sites` $10^9$ | x = 4..256, $n=2\cdot10^5$ per rung | 2026110753 |

```bash
bash experiments/53_perc_gamma_susc_d2/run_study.sh gsusc_d2_x4 4h 2026110553 4h --max-scale 512
```

`--max-scale 512`: $L(1024)=41{,}285$, i.e. $1.7\cdot10^9$ sites, does not fit on a 32 GB card.

Exploration study (user, 2026-11-07): 9 h total, `--max-scale 2048`, on missouri (30 threads,
idle at launch). A first launch on purus (`gsusc_d2_explore_9h`) was stopped after 11 min,
inside the first pilot replicate, to move to an idle machine (user, 2026-11-07); it had
written only its log, since deleted.

```bash
bash experiments/53_perc_gamma_susc_d2/run_study.sh gsusc_d2_explore 9h 2026110753 9h_missouri --max-scale 2048
```

Why: the torus study's window is capped at x = 512 by GPU memory, and there more budget
buys no bias (plan.py at the cap: |bias| 0.0307 at every budget). The exploration sampler
has no box. Probe before launch (x = 16 / 64 / 128): cv 2.20 / 2.46 / 2.60, 478 s per
$10^6$ samples at x = 128, mean 703 at x = 16 against the torus's 699.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma = 43/18 = 2.38\overline{8}$ (exact).

Pass: the eq. (wilson) 95% interval contains 43/18.

## Results

`gsusc_d2_x4_4h` (2026-11-05): $\hat\gamma = 2.37021 \pm 0.01850$, 95% [2.35171, 2.38871]:
**misses** 43/18 by $1.8\cdot10^{-4}$. B_fs 0.0146 from the pinned refit ($\omega_1 = 1.19$,
$\chi^2 = 18.1/3$, p = 4e-4: the refit warning) against an actual bias of 0.0187. Flags
visible without the answer: the refit warning; the plan's predicted bias (0.0307, pilot
$\omega_1 = 0.73$) twice the report's B_fs; $\omega_1$ moved 0.46 in the refit; the
gamma-free AIC fit, $2.3979 \pm 0.0044$, 0.028 above $\hat\gamma$. The AIC fit also misses
43/18, from above (by $3\cdot10^{-4}$ at 95%); its $\gamma$ falls toward 43/18 as the left
cut grows (16..512: $2.3924 \pm 0.0058$, $\omega = 0.99$).

`gsusc_d2_explore_9h_missouri`: stopped by the user (2026-11-07, 18:45) after 6 h 05 m, inside
the cost probe. Pilot round 1 had finished (5 x ~935 s); the probe then ran 4 h 46 m with no
output. No study files were written, only the log. Stopped to find out why cost per sample
grows like x^3 rather than x^gamma.

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
| `gsusc_d2_x4_4h` | 2.370206 | 1.1 | 0.0216 | 1.25 ± 0.235 | 4..64 (0.68) | 1.16 | 0.013 | 0.017 |
