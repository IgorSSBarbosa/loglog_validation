# 01_srw — Simple Random Walk

This folder holds three separate, independent uses of the same `srw()` simulator
(`models/srw.py`, registered as `MODELS["srw"]` in `tools/models.py`). Don't
conflate them. As of this session, the scripts that drive all three are shared with
every other experiment (`src/generate.py`, `src/plot_loglog.py`,
`src/measure_cost.py`, `src/plot_cost.py` — see `src/README.md`); this folder now
holds only recipes, README, `data/` (gitignored), and `images/` (committed evidence).

## Known ground truth for $Y_k=\lvert S_k\rvert$ — acceptance criteria

The article's `appendix-SimpleRandomWalk` is still an empty section header, but for the
observable $Y_k=\lvert S_k\rvert$ the mean is known exactly:

$$\mathbb{E}\lvert S_k\rvert = k\binom{k-1}{\lfloor (k-1)/2\rfloor}2^{-(k-1)}
=\sqrt{\tfrac{2}{\pi}}\;k^{1/2}\exp\!\Bigl(-\tfrac14 k^{-1}+\tfrac1{24}k^{-3}+O(k^{-5})\Bigr).$$

Verified two independent ways (2026-08-20): exhaustively against
$2^{-k}\sum_j\binom kj\lvert 2j-k\rvert$ for every $k=1,\dots,200$ (exact to machine
precision), and by Monte Carlo at $2\times10^6$ samples across
$k\in\{1,\dots,101\}$ (within 1.6 SE everywhere). Matching against article eq. (232),
$\mathbb{E} Y_i = a_0 i^\gamma\exp(a_1 i^{-\omega_1}+\cdots)$:

| object | exact value |
|---|---|
| $a_0$ | $\sqrt{2/\pi}\approx0.797885$ |
| $\gamma$ | $1/2$ |
| $\omega_1$ | $1$ |
| $a_1$ | $-1/4$ |
| $\omega_2$ | $3$ (the $k^{-2}$ term cancels identically) |

**These are acceptance criteria, not inputs.** By explicit decision (user, 2026-08-20 —
see `plans/three_experiment_ladder.md` D1/D2) they are deliberately *not* wired into
the code: `MODELS["srw"]` still has no `target_fn`/`true_gamma_key`, so
`src/plot_loglog.py` still overlays no reference curve, reports no `true_gamma`, and
still prints its "exploratory, not validated" note. The estimators are never handed
the answer they are supposed to be measuring — the numbers above are checked by hand
against a finished run instead. Do not add `target_fn` here without revisiting that
decision.

That $\omega_2=3$ sits three full orders below $\omega_1=1$ makes this an unusually
clean testbed for measuring $\omega_1$: contamination from the next correction term is
negligible over any usable scale range.

### The scale grid must not mix parities

$\mathbb{E}\lvert S_k\rvert$ is a **staircase**, not a smooth curve:

$$\mathbb{E}\lvert S_{2m-1}\rvert=\mathbb{E}\lvert S_{2m}\rvert\quad\text{exactly, for every }m$$

(verified for $m=1,\dots,300$) — a walk of odd length cannot get as close to the origin
as the even length above it. The asymptotic expansion above describes the *even* branch.
Any scale grid mixing odd and even $k$ therefore presents a zig-zag that eq. (232)'s
smooth model fits as curvature, and the failure is **silent** — the fit converges and
reports a small residual. Measured on exact means with zero sampling noise, a
$\rho=\sqrt2$ grid over $8\dots256$ returns $\hat\omega_1\approx17.8$ instead of $1$.

Powers of 2 are safe, which is what every recipe here uses. Pinned by
`tools/tests/test_correction.py::test_mixed_parity_grid_breaks_the_fit_on_a_staircase_mean`.

## Cost-model probe — not blocked, done

Separate from the above: measures the computational-cost exponent $d$ (article
Assumption `cost_is_power_law`, $\mathrm{cost}(i) = i^d$), needed for the budget
allocation rule (`tools/allocation.py`, Definition `def:alloc`). `MODELS["synthetic"]`
can't be used for this — drawing samples from a closed-form formula costs the same
regardless of scale, so it would just measure $d\approx0$. `srw(k)` is used here purely
as a fixture whose cost genuinely grows with $k$: generating $k$ i.i.d. $\pm1$ steps and
summing them is $\Theta(k)$, i.e. $d=1$ — a known ground truth to validate the
*measurement procedure* against, before ever pointing it at a real (and expensive)
simulator.

- `models/srw.py` — the simulator, `srw(k, n=1, q=0.5, rng=None, block_n=None)`,
  returning `n` i.i.d. realizations of $|S_k|$ as an array (vectorized over row-blocks
  of $n$; $k$ steps are genuinely drawn per sample, keeping the cost $\Theta(nk)$).
  `src/measure_cost.py` calls it at the default `n=1` -- Assumption
  `cost_is_power_law` defines $\mathrm{cost}(i)$ as the cost of simulating *one*
  sample -- `n>1` is what "Sample generation" below uses.
- `tools/cost_model.py` — generic, experiment-agnostic estimator: `cost(i)=c\cdot i^d`
  has the same log-log-linear form as $\mathbb{E} Y_i=a_0 i^\gamma$ (eq. 232), so
  `estimate_cost_exponent` reuses `tools/loglog.py`'s OLS-slope machinery, behind a
  name-keyed registry (`COST_ESTIMATORS`) so a different estimation approach can be
  added later without touching callers.
- `src/measure_cost.py` — the shared driver: times `MODELS[model].simulate(k, n=1,
  ...)` at a small grid of scales (`cost_probe_config.json`: `[256, 1024, 4096, 16384,
  65536, 262144, 1048576]`), 20 repeats each, aggregated by **median** (not mean --
  repeated timings all target the same true deterministic quantity, so noise only ever
  adds delay; this is the one place this experiment departs from the sample-mean
  framing used elsewhere for genuinely stochastic $Y_i$). `min` is the classic
  microbenchmark choice and remains available, but `median` is the default because it
  resists the same one-sided jitter *and* carries a distribution-free confidence
  interval (`tools/cost_model.py`'s `median_ci`), which a minimum does not; empirically
  the two differ by $<0.005$ in $\hat d$.
- **Two cost models are fitted, and the pure one is the wrong one at small $k$.** A
  single `simulate(k, n=1, ...)` call pays a fixed $\approx11$–$24\,\mu s$ of
  Python/NumPy dispatch that does not scale with $k$ at all. At the smallest scales
  that overhead *is* the measurement (88% of it at $k=256$), so a pure
  $\mathrm{cost}(i)=c\,i^d$ fit attributes overhead to the power law and returns a badly
  biased $\hat d$. `tools/cost_model.py`'s `estimate_cost_affine` fits
  $\mathrm{cost}(i)=a+b\,i^d$ instead and recovers $d$ correctly; acceptance is checked
  against it. Measured 2026-08-20:

  | run | pure-power $\hat d$ | affine $\hat d$ | overhead $a$ |
  |---|---|---|---|
  | `cost_probe` ($k=256\dots10^6$) | 0.771 | **1.006** | 11.2 µs |
  | `time_measure` ($k=2\dots1024$) | 0.103 | **0.948** | 23.7 µs |

  The `gamma_drop_leading`-style view (drop the first $m_0$ scales) is kept as an
  independent finite-overhead diagnostic — the same shape as the article's own $m_0$
  finite-size correction. It agrees: the local slope climbs monotonically from 0.771
  (all 7 scales) to 0.998 ($m_0=5$), converging on the same $d=1$ the affine fit
  reports directly and the $\Theta(k)$ ground truth predicts.
- `src/plot_cost.py` — separate from `measure_cost.py` (generation and plotting are
  always distinct scripts in this repo). Reads `measure_cost.py`'s saved
  `<tag>/result.json` directly (`elapsed_all` is already `{scale: array}`,
  `tools/loglog_plot.py`'s exact required shape) and hands it to the same generic
  `loglog_plot`.

Numeric acceptance criterion (`tools/tests/test_measure_cost.py`, local-only — see
`tools/README.md`): **affine** $\hat d \in [0.8, 1.2]$ over the default grid/repeats.
A companion test asserts that the fitted overhead is strictly positive and that the
pure-power $\hat d$ comes back *lower* than the affine one — so if a future change ever
removes the fixed per-call overhead, that test fails loudly and the simpler pure fit can
be restored rather than silently kept for no reason.
`images/cost_probe.png` (committed snapshot, per ground rule 1 — supplement to the
numeric check, not a replacement for it; copied in manually from a
`data/cost_probe/plot.png` run, since `plot_cost.py`'s default output now lives
alongside the data, not auto-written to `images/`) shows a clean log-log line once
past the small-$k$ overhead-dominated points.

Run (from `src/`):
```
python3 measure_cost.py -meta ../experiments/01_srw/cost_probe_config.json --tag cost_probe
python3 plot_cost.py -data ../experiments/01_srw/data/cost_probe
```

**Not done here:** checkpoint 0.5 itself (the actual error-decay-law experiment using
`tools/allocation.py`) — separate future work, once $d$-measurement is trusted on a
real (not toy) simulator.

## Sample generation — not blocked, exploratory (not Phase 1)

`src/generate.py` -- the second consumer of `tools/persistence.py`'s save/load
pattern (extracted from `experiments/00_synthetic/generator.py`, now the shared driver
for every model) -- draws $n$ i.i.d. $|S_k|$ samples per scale $k$ via
`MODELS["srw"].simulate` (i.e. `srw(k, n, q, rng)`), seeded and timed the same way, to
the same run-directory shape every model uses. Recipe (`example_config.json`) mirrors
`00_synthetic`'s exactly, with `"model": "srw"` and `"params": {"q": 0.5}` standing in
for `SyntheticParams`:

```
cd ../../src    # or wherever src/ is relative to your cwd
python3 generate.py -meta ../experiments/01_srw/example_config.json --tag demo_run
python3 plot_loglog.py -data ../experiments/01_srw/data/demo_run --estimates
```

`plot_loglog.py` hands the saved samples to `tools/loglog_plot.py`'s generic
`loglog_plot` (same tool 00_synthetic uses) and always runs `tools/loglog.py`'s
four $\hat\gamma$ estimators (`compare_methods`), writing `results.json` --
comparing estimators against each other doesn't need a known ground truth, only
comparing against one does. What it does **not** do: overlay a reference curve, or
claim any of these numbers are checked against a known target -- `MODELS["srw"]`
has no `target_fn` (see "Gamma-estimation ladder" above, still blocked), so
`plot_loglog.py` prints an explicit note that the $\hat\gamma$ values are
exploratory, not a checkpoint result. Don't mistake this for Phase 1 progress: the
numbers are real estimates, just not yet validated against an article-sanctioned
closed form. What this *does* establish: the log-log plot of $\overline{|S_k|}$ vs
$k$ is visibly a clean straight line, and all four estimators agree closely around
$\hat\gamma\approx0.5$ — consistent with the classical $\mathbb{E}|S_k| \sim
\sqrt{2k/\pi}$ asymptotic — useful groundwork for whenever Phase 1's design
question is resolved, but not itself a validated result.

`tools/tests/test_generate.py` checks output shapes match the recipe and that
`reproduce()` (regenerate from recorded metadata) matches a saved run exactly, for
every registered model including this one (see `tools/README.md`).

## Experiment B — measuring $\omega_1$

`plans/three_experiment_ladder.md` §3. Estimates the correction-to-scaling exponent,
which `tools/allocation.py`'s budget rule needs as an input, against the known
$\omega_1=1$.

```bash
cd src
python3 generate.py -meta ../experiments/01_srw/omega1_config.json --tag omega1
python3 estimate_omega1.py -data ../experiments/01_srw/data/omega1 \
        --expect-omega1 1.0 --expect-gamma 0.5
```

`estimate_omega1.py` writes `<run_dir>/omega1.json` and applies both of
`tools/correction.py`'s estimators — the direct fit of eq. (232)'s one-correction
truncation, and the bias-decay fit of how `gamma_drop_leading` converges. `--expect-*`
is **reporting only**; the estimators never receive those values.

**Allocation.** The recipe uses `"n": {"rule": "snr", ...}`, not a flat $n$ and not
Neyman. Details and the derivation are in `tools/allocation.py`'s `snr_allocation`; the
short version is that $\omega_1$ is estimated from the *correction term*, whose size
shrinks with $i$, so equalizing the error of $\overline Y_i$ (Neyman) over-samples the
small scales and starves the large ones. Measured on a first Neyman run, the
correction's SNR ran 460 at $k=2$ down to 0.25 at $k=1024$; the `snr` rule
($n_i\propto i^{2\omega_1}$) flattens it to a constant.

**Choosing the window.** Two effects pull in opposite directions, so the grid is a
tradeoff, not a free choice. Fitting the one-correction model to the *exact* means
(zero sampling noise) isolates the truncation bias from the neglected $\omega_2$ term
and caps what any sample size can achieve:

| smallest scale kept | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| $\hat\omega_1$ ceiling | 0.959 | 0.987 | 0.996 | 0.999 | 1.000 |

But dropping small scales shrinks the correction signal, so the allocation has to
compensate with more samples at the larger scales. `omega1_config.json` uses
$8\dots256$ (ceiling $\approx0.995$).

**Precision is the binding constraint, and it is variance, not bias.** The estimator is
unbiased but has a wide sampling distribution: with $B=5\times10^{10}$ over this grid,
simulating the fit against noise of the measured size gives
$\hat\omega_1 = 1.005 \pm 0.155$ over 400 replicates. So a single run's deviation from
$1$ is dominated by noise — report a replicate-based standard error, never a lone point
estimate. Ten times the budget brings the standard deviation to $\approx0.05$.

### Result (2026-08-20) — PASS

Five **independent** replicates (ground rule 2: separate seeds, no shared samples),
$B=4\times10^{10}$ each, scales $8\dots256$, `snr` allocation. Mean $\pm$ standard error
across replicates:

| quantity | measured | known truth | $z$ |
|---|---|---|---|
| $\omega_1$ | $1.0155 \pm 0.1050$ | $1$ | $+0.15$ |
| $\gamma$ | $0.5000 \pm 0.0003$ | $1/2$ | $+0.11$ |
| $a_1$ | $-0.2748 \pm 0.0597$ | $-1/4$ | $-0.41$ |
| $a_0$ | $0.7979 \pm 0.0017$ | $\sqrt{2/\pi}\approx0.797885$ | $+0.01$ |

All four within half a standard error, and $\hat\omega_1\in[0.85,1.15]$ meets the
acceptance criterion. Note the enormous spread in precision across parameters: $\gamma$
is pinned to $3\times10^{-4}$ while $\omega_1$ is only known to $\pm0.1$ from the same
data — a correction exponent is intrinsically far harder to measure than the leading
one, which is precisely why it gets its own experiment and its own allocation rule.

Reproduce (each replicate ~4 min, ~1.4 GB of samples; delete them afterwards, the
`omega1.json` is the result):

```bash
cd src
for r in 1 2 3 4 5; do
  sed "s/20260820/2026082$r/; s/5e10/4e10/" ../experiments/01_srw/omega1_config.json > /tmp/rep$r.json
  python3 generate.py -meta /tmp/rep$r.json -o ../experiments/01_srw/data --tag omega1_rep$r
  python3 estimate_omega1.py -data ../experiments/01_srw/data/omega1_rep$r
  rm -rf ../experiments/01_srw/data/omega1_rep$r/samples   # keep omega1.json!
done
```

**Not done:** the bias-decay cross-check did not run on this grid — with only 6 scales,
`gamma_drop_leading` leaves fewer than 4 windows of $\ge4$ scales, and
`estimate_omega1.py` reports that rather than fitting 3 parameters to 3 points. It did
run on the earlier 10-scale grid (giving $0.78$ against the direct fit's $0.94$), and it
agrees with the direct fit on planted data in `tools/tests/test_correction.py`. Getting
both estimators on one grid needs $\ge10$ scales, i.e. extending the window upward at
$\Theta(i^3)$ cost — deferred, not attempted.

## Experiment C — does the budget-allocation rule deliver?

`plans/three_experiment_ladder.md` §4. Tests Proposition `prop:opt` (eq. 945-946) on a
testbed where $\gamma=1/2$, $d=1$ and $\omega_1=1$ are all known.

```bash
cd src
python3 allocation_experiment.py -meta ../experiments/01_srw/allocation_config.json --tag allocation
```

The theorem makes two claims that can come apart, and they are tested separately:

- **RATE** — the error falls like $B^{-\omega_1/(d+2\omega_1)}$, i.e. $B^{-1/3}$ here.
- **POINT** — the specific $m_0=\theta_2\log_\rho B$ it names minimizes the error at a
  given finite $B$.

A rate theorem is asymptotic *up to constants*, so POINT can fail while RATE holds — and
measuring only the allocation the theorem names, with no comparison, could never tell
the difference. `prop:opt` already fixes $n$ uniform across scales, so what it really
chooses is $m_0$; the honest control arm is therefore **every other $m_0$ at the same
budget**, which is what the script sweeps.

Estimator is the article's own closed-form $w_{k,m}$ weights (`gamma_closed_form`,
eq. 523-526) — the one `prop:opt` is stated for, and this ladder is exactly the
consecutive $\rho^k$ grid it requires. Generic OLS is recorded alongside. Every
$(B,m_0,\text{replicate})$ cell draws fresh independent randomness
(`SeedSequence.spawn`, ground rule 2). Samples are not persisted — the sweep draws far
more than is worth storing, and the per-cell $\hat\gamma$ values in `result.json` are
the result; the base seed is recorded and spawning is deterministic, so any cell
regenerates exactly.

### Result (2026-08-20) — RATE passes, POINT fails by a constant

$m=6$, $\rho=2$, $d=\omega_1=1$, 40 replicates per cell, $m_0\in\{2,\dots,11\}$,
$B\in\{10^7,10^8,10^9\}$ (48 min).

**RATE — passes.** $\mathrm d\log\mathrm{RMSE}/\mathrm d\log B$ measured $-0.364$ at
`prop:opt`'s own $m_0$ and $-0.384$ at the empirically best $m_0$, against the predicted
$-\omega_1/(d+2\omega_1) = -1/3$. The promised $B^{-1/3}$ error decay is real.

**POINT — fails, by a constant offset of about 3 in $m_0$.**

| $B$ | `prop:opt` $m_0$ | best $m_0$ | RMSE at `prop:opt` | best RMSE | penalty |
|---|---|---|---|---|---|
| $10^7$ | 7 | 3 | $1.04\times10^{-2}$ | $4.75\times10^{-3}$ | $2.18\times$ |
| $10^8$ | 8 | 5 | $4.77\times10^{-3}$ | $2.19\times10^{-3}$ | $2.18\times$ |
| $10^9$ | 9 | 6 | $1.94\times10^{-3}$ | $8.12\times10^{-4}$ | $2.39\times$ |

**It is an offset, not a wrong trend.** The empirical argmin tracks
$\theta_2\log_\rho B$ in slope — 0.45 over these three integer-valued points, and 0.30
across five decades in the analytic version below, against $\theta_2=1/3$ — but sits
about $3.3$ lower. So the theorem's $\log_\rho B$ coefficient is right and only the
additive constant it drops is missing.

**The mechanism: `prop:opt` over-corrects for bias.** Its derivation balances bias
against standard deviation, and at the *empirical* optimum they are indeed balanced —
$\lvert\mathrm{bias}\rvert/\mathrm{sd} = 1.68,\,0.77,\,0.60$ at the three budgets. At
the $m_0$ the formula actually names, that ratio is $0.09,\,0.12,\,0.30$: the bias has
been driven far below the noise floor, and the budget spent buying that unnecessary
bias reduction would have bought more accuracy as samples. Raising $m_0$ costs $\rho^d$
per step in affordable $n$, which is expensive.

**Independent confirmation.** Computing the bias exactly (the same $w_{k,m}$ weights
applied to the *exact* $\mathbb{E}\lvert S_k\rvert$, no sampling at all) and the
standard deviation analytically from $n$ and the half-normal CV $\sqrt{\pi/2-1}$
predicts the same thing without drawing a single sample — argmin $m_0 = 5$ at $10^8$
and $6$ at $10^9$ with penalties $2.4\times$ and $2.3\times$, against the measured $5$,
$6$, $2.18\times$, $2.39\times$. Two independent routes, same conclusion.

**How to read this.** None of it contradicts the theorem: `prop:opt` is a *rate*
result, correct up to constants, and the rate is confirmed. What the experiment adds is
that the dropped constant is not negligible at usable budgets — it costs a factor
$\approx2.2$–$2.4$ in RMSE, equivalently a factor of $\approx2^{3.3}\approx10$ in
budget. A practitioner following the formula literally pays that.

### Tuning the constant (2026-08-21)

The offset *is* derivable in closed form, from exactly the constants the rate argument
discards. Write the two error sources of $\hat\gamma=\frac{1}{\log\rho}\sum_k w_{k,m}
\log\overline Y_{\rho^k}$ separately. Using the weight identities $\sum_k w_k=0$ and
$\sum_k w_k k=1$, the leading term passes through exactly and only the correction
survives, so with $j=k-m_0$:

$$\lvert\mathrm{bias}\rvert = C_b\,\rho^{-m_0\omega_1},\qquad
C_b=\frac{\bigl\lvert a_1\sum_{j=1}^m w_j\rho^{-j\omega_1}\bigr\rvert}{\log\rho}$$

$$\mathrm{sd} = C_s\,n^{-1/2},\qquad
C_s=\frac{c_v\lVert w\rVert}{\log\rho},\qquad
\lVert w\rVert^2=\frac{12}{m(m^2-1)}$$

with $c_v=\mathrm{sd}(Y_i)/\mathbb{E}Y_i$ the observable's coefficient of variation
(scale-free for $\lvert S_k\rvert$, $=\sqrt{\pi/2-1}$). The budget is
$B=n\rho^{dm_0}G$, $G=\rho^d\frac{\rho^{dm}-1}{\rho^d-1}$ (Lemma `lem:budget`).
Minimizing $C_b^2\rho^{-2m_0\omega_1}+C_s^2G\rho^{dm_0}/B$ over $m_0$ gives

$$\boxed{\;m_0^{\ast}=\theta_2\Bigl(\log_\rho B+\log_\rho\kappa\Bigr),\qquad
\kappa=\frac{2\omega_1 C_b^2}{d\,C_s^2\,G}\;}$$

`prop:opt` is this with the $\log_\rho\kappa$ term dropped. Two consequences worth
noting:

- The optimum has $\lvert\mathrm{bias}\rvert/\mathrm{sd}=\sqrt{d/(2\omega_1)}$ —
  $0.707$ here. **Measured: 0.60, 0.77, 1.68.** The two larger budgets bracket it.
- For this testbed the offset evaluates to $\theta_2\log_\rho\kappa = -3.94$, against
  the $-3.3$ estimated by eye from the sweep.

Validation against Experiment C's own cells: predicted RMSE lands within **4–16%** of
measured at every $(B,m_0)$ tested, and the predicted penalties (2.07, 2.37, 2.29) match
the measured ones (2.18, 2.18, 2.39). The tuned $m_0$ reproduces the measured argmins
$3,5,6$ as $4,5,6$ — within one step everywhere.

Implemented as `tools/allocation.py`'s `allocation_constants` / `tuned_allocation` /
`predict_error`. Note `tuned_allocation` **rounds** $m_0$ where `optimal_allocation`
floors it: $n$ is recomputed and floored from whichever integer $m_0$ is chosen, so the
budget guarantee holds either way, leaving rounding free to pick the nearer candidate
(flooring would give $3,4,5$ against the measured $3,5,6$).

### Planning table — what precision, for how long?

`src/allocation_table.py` builds it from the measured results rather than assumed
constants ($a_1$ from Experiment B, $c_v$ from its samples, throughput from Experiment
C's wall clock):

```bash
cd src
python3 allocation_table.py --compare          # add --csv table.csv to save it
```

Each row is one integer $m_0$, at the budget where it is optimal:

| $m_0$ | $n$ per scale | wall clock | RMSE($\hat\gamma$) |
|---|---|---|---|
| 3 | 1,816 | 12 ms | $7.7\times10^{-3}$ |
| 6 | 116,242 | 6.1 s | $9.6\times10^{-4}$ |
| 9 | 7,439,547 | 52 min | $1.2\times10^{-4}$ |
| 12 | 476,131,015 | 445 h | $1.5\times10^{-5}$ |
| 15 | 30,472,384,995 | 26 yr | $1.9\times10^{-6}$ |
| 18 | 1,950,232,639,725 | 13,306 yr | $2.4\times10^{-7}$ |

The $B^{-1/3}$ rate is brutal in this direction: each extra decimal digit of accuracy
costs $10^3$ in time. Three digits ($10^{-3}$) is seconds; six digits is millennia.

The `--compare` column shows a steady $3.19\times$ RMSE penalty for using `prop:opt`'s
uncalibrated $m_0$ — larger than the $2.18$–$2.39\times$ Experiment C measured, because
the table compares at the exact continuous offset ($3.94$ steps) while those particular
budgets happened to round to integer gaps of 3.

### "I ran it twice and got two different tables"

Both were right, and the difference is mostly an artefact of how the table is indexed.

**First, a real bug, now fixed.** Pointing `--data-root` at a *run* directory
(`.../data/Huge_test/`) rather than the parent of runs (`.../data/`) found nothing and
silently substituted hardcoded fallback constants, while still printing a confident
table. It now accepts a run directory directly, and any fallback prints an unmissable
warning naming which input was not measured.

**Second, the indexing.** In the default table each row is an integer $m_0$ and the
question is *"at what budget is this $m_0$ optimal"*. That budget depends on the offset,
so any change in $a_1$ or $c_v$ slides **every row** along the budget axis and two honest
tables look wildly different. Use `--by-budget` to index by budget instead — *"given this
much time, what should I run"* — which is the comparable view. Across the two data roots
above it gives **identical $m_0$ and $n$ at every budget**, with RMSE differing by under
2%. In general the guarantee is weaker but still strong: $m_0$ agrees **within one step**,
because the continuous optimum can fall either side of a rounding boundary.

**Third, and most reassuring: the optimum is flat.** RMSE is quadratic in the $m_0$
error, so

| error in $m_0$ | 0.25 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| RMSE penalty | 1.4% | 5.3% | 19% | 64% | 131% |

The offset moves as $2\theta_2\log_\rho\lvert a_1'/a_1\rvert$ — *logarithmically*, so
the 24% disagreement in $a_1$ between those two tables shifts $m_0$ by only 0.234, costing
**1.2%** in RMSE. Scored against the exact truth ($a_1=-1/4$, $c_v=\sqrt{\pi/2-1}$), all
three constant sets recommend the *same* $m_0$ and $n$ at $B=10^8,10^{10},10^{12}$, with
identical RMSE.

So: **the tuned allocation is not sensitive to these constants at all.** What *is*
sensitive is $\omega_1$ itself, which enters the exponent $\theta_2=1/(d+2\omega_1)$
rather than a logarithm — which is exactly why Experiment B exists and why it is worth
running with replicates.

### Replicates: run them, but pool them rather than averaging the fits

Replicates are the right protocol, for two separate reasons — they are the only source
of a standard error, and they cut the variance. But **how they are combined matters**,
and the obvious choice is the wrong one.

`fit_correction` is a nonlinear function of the data, so $\mathbb{E}[\hat a_1]\neq a_1$.
Averaging $R$ separate fits therefore converges to $\mathbb{E}[\hat a_1]$, **not** to
$a_1$ — a bias that no number of replicates removes. Pooling the sample means first
(weighted by $n$) and fitting *once* feeds the nonlinear step data with $\sqrt R$ less
noise, so the bias shrinks with $R$ too. Measured on this experiment's own
configuration, 250 trials, truth $a_1=-1/4$:

| $R$ | mean-of-fits bias | pooled-fit bias | RMSE (mean / pooled) |
|---|---|---|---|
| 1 | $-0.0046$ | $-0.0046$ | 0.0787 / 0.0787 |
| 5 | $-0.0094$ | $+0.0027$ | 0.0383 / 0.0348 |
| 20 | $-0.0114$ | $+0.0038$ | 0.0212 / **0.0171** |

Mean-of-fits is heading for $-0.262$, and by $R=20$ its bias is already comparable to
its own spread — it would dominate entirely at larger $R$. $\omega_1$ is barely affected
either way (0.0393 vs 0.0395 at $R=20$); $a_1$ is what the allocation offset depends on.

`allocation_table.py` now pools. It works from the `y_bar`/`n`/`sigma_log` that
`estimate_omega1.py` records in `omega1.json`, so **the samples need not be kept** — but
the JSON does. The standard error still comes from the spread of the individual fits
divided by $\sqrt R$ (that spread estimates a single fit's sd, and the pooled
estimator's sd is close to it over $\sqrt R$: measured 0.0166 against
$0.0785/\sqrt{20}=0.0176$).

Two guards worth knowing: runs are only pooled when their scale grids match, and
unrelated runs in the same `data/` are not silently treated as replicates — `data/`
holds both `omega1` (scales $8\dots256$) and `Huge_test` (scales $2\dots1024$), and
averaging *those* would be meaningless rather than merely imprecise.

**How to be sure, in order of cost:**

1. Run $\ge2$ replicates of Experiment B, with **identical configuration** so they can
   be pooled (`omega1_rep*` tags are picked up automatically). The script then prints $a_1$'s standard error
   and converts it into an $m_0$ range and an RMSE penalty, instead of reporting a bare
   number with no uncertainty.
2. Compare with `--by-budget`, never the $m_0$-indexed default.
3. Sanity-check $c_v$'s printed per-scale spread. It is assumed scale-free; for
   $\lvert S_k\rvert$ it tends to $\sqrt{\pi/2-1}$, but small $k$ inflates it
   (at $k=2$, $\lvert S_2\rvert\in\{0,2\}$ gives $c_v=1$), so a run whose grid starts
   very low will report a biased average.
4. The final arbiter is simulation: `allocation_experiment.py` sweeps $m_0$ at a fixed
   budget and measures the argmin directly, assuming none of this.

**Keep `omega1.json` when cleaning up.** The reproduce block below deletes only
`samples/`; deleting the whole run directory loses Experiment B's fit, and
`allocation_table.py` then falls back to a single-replicate $a_1$ (which is what
produced $-0.2217$ instead of the 5-replicate $-0.2748$, shifting the offset by 0.2).

### Fixed-memory generation for large `n`

`Huge_test.json` (`n=100,000,000`, scales up to `1024`) used to have to be killed for
exhausting memory: a single unblocked `srw(k, n, ...)` call built one `(n, k)` matrix
in RAM (819 GiB at `k=1024, n=1e8` with the old `int64` dtype). Fixed at two layers
(see `tools/README.md`'s `persistence.py`/`models.py` entries for the mechanics):
`models/srw.py` now blocks its internal draw over `n` to a fixed byte budget, and
`src/generate.py` streams any run whose total estimated size exceeds a byte budget
straight to on-disk per-scale arrays instead of assembling everything in RAM first.
Both are exact for any block/chunk size (bit-identical to the unblocked path, same
seed — see `models/srw.py`'s module docstring for why), not just "close enough".
Smoke-tested this session at `n=20,000,000` (20x `larger_test.json`, which already ran
fine) — RSS stayed a few GiB throughout, no `psutil`-triggered flush warnings, and the
chunked output matched the in-RAM path exactly. `Huge_test.json` itself is expected to
run (slowly — ~100M samples x 10 scales) without OOM, but hasn't been run end-to-end
in this session (would take a long time).
