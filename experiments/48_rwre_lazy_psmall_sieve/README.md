# Experiment 48 — the article's pipeline on `rwre_gpu_lazy`, a sieve in $p$ near 0

**Status: queued (2026-10-31).**

## Question (user, 2026-10-31)

Run the four-step pipeline of the article's Section "Simulations" (pilot, budget
allocation, long run, Step 4 checks + eq. (wilson) interval) on the RWRE for
$p\in\{0.02,0.04,\dots,0.14\}$, 3 h per $p$. Experiments 19 and 41 found $\omega_1$
small and ladder-dependent near $p=0$ (0.45 on 4..1024 against 0.15 from the tail); this
asks how the pipeline's own plan and interval behave there.

## Setup

| | |
|---|---|
| model | `rwre_gpu_lazy`, 18's defaults ($\alpha=\tfrac12$, `env_sweeps` 4, bit layout, `cap` $2^{15}$), $Y_k=\lvert X_k\rvert$ |
| pilot recipe | `recipes/samples_sieve_p<p>.json`: 19's `samples_autopilot_p0.json` (4..1024, snr rule, budget $10^9$) with only `p` changed |
| pipeline | `autopilot.py --time 3h --replicates 5 --seed 2026103148<j>` ($j$ = index of $p$), no `--max-scale`, then `46/plot_pipeline.py` for the Step 1–4 figures |
| cap fallback | 41's rule: on "needed more than cap", redo the study at `cap` $2^{17}$, $2^{18}$, 393,216 (recipes `_cap<c>`), same seed. `cap` changes no draw |
| hosts | `local/dispatch_pool.sh`: at most 4 simultaneous jobs, one per idle host |

```bash
bash experiments/48_rwre_lazy_psmall_sieve/run_p.sh 0.02 3h 202610314800
```

## Known values — acceptance criteria, not inputs (ground rule 4)

None proven for $p\ne\tfrac12$. For comparison only: 41 measured $\hat\gamma=0.5061$ at
$p=0$ (2¹⁹..2²⁴), 19's `ap2h_fix` sweep gives the 2¹⁵..2²⁰ values at $p=0.05,0.1,0.15$.
Checks that do apply: (A2) $Y>0$ fraction, (A4)–(A6) flat across the window, and the
long run's time against the plan (A8).

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
| `sieve_p0.02_3h_cap131072` | 0.515547 | 0.382 ± 0.00094 | 1.56e+04/6 ⚠ | 0.00744 | 0.212 ± 0.0012 | 32..512 | 2.42/2 | 0.0951 |
| `sieve_p0.04_3h_cap131072` | 0.517032 | 0.296 ± 0.00066 | 2.07e+04/6 ⚠ | 0.0244 | 0.21 ± 0.00092 | 32..512 | 7.9/2 | 0.0991 |
| `sieve_p0.06_3h_cap131072` | 0.514772 | 0.367 ± 0.0011 | 8.91e+03/6 ⚠ | 0.00749 | 0.19 ± 0.002 | 32..512 | 2.93/2 | 0.129 |
| `sieve_p0.08_3h_cap131072` | 0.516271 | 0.324 ± 0.0011 | 1e+04/6 ⚠ | 0.0127 | 0.182 ± 0.0012 | 16..256 | 13.8/2 ⚠ | 0.15 |
| `sieve_p0.12_3h_cap131072` | 0.513606 | 0.27 ± 0.0013 | 9.61e+03/6 ⚠ | 0.0223 | 0.142 ± 0.0012 | 16..256 | 1.57/2 | 0.325 |
| `sieve_p0.14_3h_cap131072` | 0.514370 | 0.264 ± 0.0015 | 8.21e+03/6 ⚠ | 0.0223 | 0.124 ± 0.0019 | 16..256 | 3.37/2 | 0.49 |
| `sieve_p0.1_3h_cap131072` | 0.515666 | 0.272 ± 0.001 | 1.26e+04/6 ⚠ | 0.0264 | 0.162 ± 0.00095 | 16..256 | 12.6/2 ⚠ | 0.222 |
