# Experiment 51 — the article's pipeline on percolation: susceptibility exponent $\gamma$, $d=3$

**Status: done (2026-11-03).**

## Question (user, 2026-11-02)

Run the four-step pipeline of the article's Section "Simulations" (pilot, budget
allocation, long run, Step 4 checks + eq. (wilson) interval) on $\gamma_{\rm susc}$ in
$d=3$, as experiment 49 did for $\tau$ and $d_f$. 4 h total.

Why a new pilot design: experiment 37's pilot (2026-10-08, x = 4..256, Neyman $10^9$,
411 s) was triaged RED with $\omega_1 = 0.80 \pm 2.3$. The pooled fit itself pins
$\omega_1$ to roughly 0.45..1.25 ($\Delta\chi^2 = 1$ profile). The $\pm2.3$ is
`pilot.py`'s se, the spread of the per-replicate fits: with only 284 samples on the top
rung, two of five replicates fell into the degenerate corners
($\omega_1 = 0.08$, and $\omega_1 = 12$ with $a_1 = 7\cdot10^5$, not converged). This
design gives every replicate $10^4$ samples on every rung so each fit stands on its own.

## Setup

| study | model | params | pilot ladder | seed |
|---|---|---|---|---|
| `gsusc_d3_x4_4h` | `percolation_susceptibility_gpu` (x-ladder) | dim 3, $\varepsilon_0=0.15$, `box_factor` 4, $\nu_{\rm box}=0.88$, torus, moment 1 | x = 4..128, $n=10^4$ per rung | 2026110252 |

```bash
PY=.venv-py312/bin/python bash experiments/51_perc_gamma_susc_d3/run_study.sh gsusc_d3_x4 4h 2026110252 4h
```

Run on **orinoco** (RTX 5090, driver 580, Ubuntu 24.04), launched 2026-11-02 19:2x, because
purus's GPU was taken by another user. orinoco has only Python 3.12, so it uses its own
venv, `.venv-py312` (gitignored): `requirements.txt`'s pins plus `cupy-cuda12x[ctk]==14.2.0`,
since orinoco has no system CUDA toolkit (NVRTC/cuRAND 12.9 from pip, against 12.8 on
purus). Smoke check before launch: one 4-sample draw per rung, means in line with 37's.

i.e. `autopilot.py --time 4h --replicates 5`, then `46/plot_pipeline.py`. Same x-ladder and
$c=4$ as 37, so the ceiling-rounding shift of experiment 09 Step 3 is still in the design.

### Aborted first attempt: x = 2..64 (`gsusc_d3_4h`, seed 2026110251)

`recipes/samples_gsusc_d3.json`, $n=10^4$ per rung. Round 1 of the pilot (240 s) gave
$\omega_1 = 0.069 \pm 0.155$, $\hat\gamma = 2.16$, B_fs span 614x: gate failed. Stopped by
the user in round 2. Cause: **x = 2 is pre-asymptotic.** Pooled local slopes
1.719, 1.728, 1.755, 1.780, 1.786 (each $\pm0.006$) step by +0.009, +0.027, +0.025, +0.006;
one correction term needs geometrically shrinking steps, and the first is too small, so the
only one-term fit that accepts x = 2 has $\omega_1 \to 0$, where $a_1 2^{-\omega_1 i}$ is
nearly linear in $i$ and trades with $\gamma i + a_0$. Profile $\chi^2$ over $\omega_1$:
all six rungs, minimum at the lower edge ($\chi^2 = 2.4$ at 0.07, 11.1 at 0.8); x = 2
dropped, minimum at $\omega_1 \approx 0.6$–$0.8$ ($\chi^2 = 0.37$ / 1 dof). Doubling $n$
cannot fix a misspecified rung, so the ladder was changed rather than the budget.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma = 1.793(3)$ and $1.819(3)$ (literature, about 6 combined errors apart; see 37).
- Correction: $\Delta_1 = \omega\nu \approx 1.41(4)$ (see 37).

Pass: the eq. (wilson) 95% interval contains a literature value; which one is reported.
Reported beside it: the replicate-$t$ interval, and each Step 4 check.

## Results (2026-11-03, `gsusc_d3_x4_4h`)

Pilot: round 1 ($n=10^4$) $\omega_1 = 1.37\pm1.02$, gate failed; round 2 ($2\cdot10^4$)
$\omega_1 = 1.004\pm0.292$, B_fs span 7.66x at $m_0=3$, passed (76.5 min, over the 60 min
cap). Plan "usable but soft" (constants move $m_0$ by 1.05 steps). Long run: $m_0=3$,
x = 16..512, $n=1{,}366$ per scale x 5 replicates, 158.7 min (predicted 161.4).

| interval | $\hat\gamma$ | 95% | contains 1.793 | contains 1.819 |
|---|---|---|---|---|
| eq:wilson bound (incomplete: no $\omega_2$ piece) | 1.7876 ± 0.0201 | [1.7674, 1.8077] | yes | no |
| Student t(4), 5 replicates, scatter only | 1.7876 ± 0.0024 | [1.7810, 1.7941] | yes | no |

eq:wilson terms (three-term form of the 2026-11-04 article revision): B_fs 0.0068 (pinned
refit $\omega_1 = 1.087\pm0.161$, $\chi^2 = 1.27/3$), B_good 0.0005, $1.96\sigma_{se}$ 0.0129.
The old four-term bound (B_good 0.0013, B_bad 0.0103, $1.96\sigma_{se}$ 0.0127) gave
[1.7565, 1.8187], which missed 1.819 by 0.0004; the new one misses it by 0.011. Per-replicate $\hat\gamma$:
1.7887, 1.7839, 1.7881, 1.7955, 1.7818; spread 0.0053 vs closed-form sd 0.0065.

**Pass** on $\gamma = 1.793(3)$. Step 4 (`pipeline_checks.png`): local slopes flat over
the window within errors (x = 4, 8 lie below, as in the aborted attempt); moments (A4–A6)
show no trend; cost model matches the long run (0.98x); pilot extrapolation lies on the
long-run means. The local-slope fits cannot separate a power-law correction
($\gamma = 1.7916$, $\chi^2 = 3.59/4$) from a $1/\ln i$ one ($\gamma = 1.8157\pm0.013$,
$\chi^2 = 4.14/5$), so the data alone do not decide between the two literature values'
extrapolations. Caveats: declared cost exponent 2.623 vs measured 2.676 (MISMATCH, timing
only); the pooled eq. (232) fit of the long run alone degenerates ($\omega_1 = 8.9$),
because x = 16..512 sits past most of the correction.

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
| `gsusc_d3_explore` | 1.786715 | 0.782 ± 0.11 | 2.05/3 | 0.0255 | 0.782 ± 0.14 | 4..128 | 2.05/3 | 0.0255 |
| `gsusc_d3_x4_4h` | 1.787570 | 1.09 ± 0.16 | 1.27/3 | 0.0201 | 1.09 ± 0.16 | 4..128 | 1.27/3 | 0.0201 |
