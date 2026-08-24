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

### The two plots, and the RATE check against measured $\omega_1$ *and* $d$

`src/plot_allocation.py` writes `<run_dir>/plot.png`:

```bash
cd src && python3 plot_allocation.py -data ../experiments/01_srw/data/allocation
```

**Left — the POINT claim.** RMSE of $\hat\gamma$ against $m_0$, one curve per budget. The
U-shape *is* the bias-variance tradeoff `prop:opt` reasons about: at small $m_0$ the
window still contains correction-contaminated scales (bias), at large $m_0$ the same
budget buys fewer samples (variance). Each curve carries two markers — ● the empirical
argmin, ✕ `prop:opt`'s choice. **Every ✕ sits systematically right of its ●**: that
displacement is the POINT failure, and the vertical gap between them is the penalty
factor (`rmse_at_prop_opt_m0 / best_rmse`, i.e. how much more error the theorem's ladder
gives for identical compute).

**Right — the RATE claim**, and the plot the paper's $B^{-\omega_1/(d+2\omega_1)}$ law calls
for: RMSE against budget on log-log, fitted as a power law, against the exponent
predicted from **measurements of both $\omega_1$ and $d$** — Experiment B's pooled
$\omega_1$ and the cost probe's affine $\hat d$, neither taken from the recipe:

$$\theta = -\frac{\omega_1}{d+2\omega_1},\qquad
\sigma_\theta^2 = \frac{d^2\sigma_{\omega_1}^2 + \omega_1^2\sigma_d^2}{(d+2\omega_1)^4}$$

```
omega_1 = 0.9836 +/- 0.1113   (6 replicates, pooled)
d       = 1.0069 +/- 0.0023   (9 cost probes, affine fit)

predicted -omega1/(d+2*omega1) = -0.3307 +/- 0.0127
  error budget (omega1: 0.01266, d: 0.00026)
measured slope se = 0.0343   (RMSE over R=40 draws carries ~1/sqrt(2R) relative sd)

  at prop:opt's m0   -0.3637 +/- 0.0343   delta -0.0330 +/- 0.0366   z = -0.90  consistent
  at the best m0     -0.3837 +/- 0.0343   delta -0.0530 +/- 0.0366   z = -1.45  consistent
```

**The two experiments agree.** Three points are worth extracting:

1. **The measured slope needs its own error bar.** An RMSE estimated from $R$ replicates
   is itself random — relative sd $\approx1/\sqrt{2R}$, so $\mathrm{sd}(\log\mathrm{RMSE})
   \approx1/\sqrt{2R}$ and $\mathrm{se(slope)} = \mathrm{sd}(\log\mathrm{RMSE})/\sqrt{S_{xx}}$.
   Derived from the known noise rather than fit residuals on purpose: with 3 budgets a
   residual-based estimate has 1 degree of freedom. **Omitting it made a consistent
   result read as a 3-sigma discrepancy** — the error that this section corrects.
2. **$d$'s uncertainty is negligible — 48x smaller than $\omega_1$'s** (0.00026 vs
   0.01266). Both partials share $(d+2\omega_1)^{-2}$, so the error budget is decided
   purely by which input is measured worse. $d$ comes from timings and is easy; $\omega_1$
   is a correction exponent and is intrinsically hard. Effort spent tightening $d$ is
   wasted; effort spent on $\omega_1$ replicates is not.
3. **Both measured slopes are slightly steeper than predicted**, and consistently so
   ($-0.033$ and $-0.053$). Within noise at three budgets, but if it is real it would
   mean the error falls a little *faster* than the theorem promises. Distinguishing it
   needs more budgets (the span enters $\mathrm{se(slope)}$ through $S_{xx}$) and more
   Experiment B replicates, not more replicates in C.

Measured $d$ comes from `data/cost_probe_reps/rep*` (8 independent probes, seconds each):

```bash
cd src && for r in 0 1 2 3 4 5 6 7; do
  python3 measure_cost.py -meta ../experiments/01_srw/cost_probe_config.json \
      -o ../experiments/01_srw/data --tag cost_probe_reps/rep$r
done
```

### Extending to lower budgets (2026-08-21) — the RATE law tightens to 0.3 sigma

`allocation_wide_config.json` repeats the sweep over **six** budgets,
$10^4\dots10^9$, with $m_0\in\{0,\dots,11\}$ and 40 replicates (~58 min).
Widening the span is the right lever: $\mathrm{se(slope)}$ carries
$1/\sqrt{S_{xx}}$, and going from 3 budgets over 2 decades to 6 over 5 decades
shrinks it **3x**, from $0.0343$ to $0.0116$ — whereas more replicates at the same
span would only have bought $1/\sqrt R$.

```
predicted -omega1/(d+2*omega1) = -0.3307 +/- 0.0127
measured slope se = 0.0116   (6 budgets, R=40)

  at prop:opt's m0   -0.3502 +/- 0.0116   delta -0.0195 +/- 0.0172   z = -1.14  consistent
  at the best m0     -0.3259 +/- 0.0116   delta +0.0049 +/- 0.0172   z = +0.28  consistent
```

**The apparent "consistently steeper" trend was an artefact of the narrow range.**
Over 3 budgets both slopes came out steeper than predicted ($-0.364$, $-0.384$); over 6
they move *toward* the prediction ($-0.350$, $-0.326$), and the best-$m_0$ curve now
agrees to $0.28\sigma$ — essentially exact. Nothing was wrong with the earlier numbers;
two decades simply cannot pin a slope to better than $\pm0.034$.

The POINT failure is unchanged and, if anything, clearer: `prop:opt` overshoots $m_0$ at
every one of the six budgets, at a cost of $2.0$–$2.7\times$ in RMSE.

Note the left panel switches to a **sequential** color ramp beyond three budgets.
Budget is an ordered variable, and cycling the 3-slot categorical palette gave
$B=10^4$ and $B=10^7$ the same blue — not a nitpick but a misreading, since the eye
groups them as one series. The ramp also encodes the ordering itself, so "darker = more
budget" is legible without the legend.

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

### Are the error bars themselves calibrated? (2026-08-22) — they were not

Everything above reports *estimate $\pm$ uncertainty*, and the estimates have been
checked against known truth. The uncertainties never had been. "$\omega_1 = 1.0155 \pm
0.1050$" is also a claim: that re-running the experiment many times would land within
$0.1050$ of truth about 68% of the time. `src/check_coverage.py` measures that
(PLAN.md checkpoint 0.4) by replaying Experiment B's exact configuration — scales
$8..256$, the real per-scale $n$, $R=5$ — 2000 times against srw's known truth and
counting interval hits.

**Result: every "95%" interval this repo published was an 88% interval.**

| quantity | centre | $q$ = normal | $q$ = $t(4)$ |
|---|---|---|---|
| $\omega_1$ | pooled | **0.880** [0.866, 0.894] | 0.948 [0.937, 0.956] |
| $\omega_1$ | mean-of-fits | **0.877** | 0.946 |
| $a_1$ | pooled | **0.880** | 0.950 |
| $a_1$ | mean-of-fits | **0.878** | 0.955 |
| $\gamma$ | pooled | **0.891** | 0.952 |
| $\gamma$ | mean-of-fits | **0.878** | 0.950 |

The cause is the quantile, and the diagnostics show it rather than assume it: the
`se_ratio` (stated se over the estimates' actual scatter) is $0.94$–$1.03$ for the
pooled estimates, so the error bar is the right *size*. What is wrong is multiplying
it by $1.960$ when an sd built from 5 points needs $2.776$. An sd from $R$ draws is
both noisy and downward-biased ($c_4(5)=0.940$), and $t(R-1)$ is exactly the
correction. Nothing about the model, the estimator or the pooling was at fault.

Two things fell out that were not the point of the exercise:

**The pooling decision is confirmed at 2000 trials.** $\hat a_1$'s bias is $-0.0155$
for mean-of-fits against $-0.0019$ pooled — 8x smaller. That choice had been argued
from a 250-trial run; it holds.

**A "mismatched pair" worry turned out to be backwards.** The pipeline takes its
centre from the pooled refit but its width from the spread of the *unpooled* fits,
which describe different estimators. Measured, the mismatch is benign and mildly
favourable: pooling shrinks the centre's scatter ($0.0349$ vs $0.0390$ for $a_1$),
moving it *closer* to the stated se — `se_ratio` improves from $0.870$ to $0.974$.

#### Calibration is not free at any $n$

Sweeping the sample size shows where it stops working (300 trials each, $t(4)$):

| $n$-scale | $n$ at $k=256$ | $\omega_1$ coverage | $\omega_1$ bias |
|---|---|---|---|
| 1.0 | 170,899,089 | 0.953 | $-0.013$ |
| 0.3 | 51,269,726 | 0.947 | $-0.021$ |
| 0.1 | 17,089,908 | 0.947 | $-0.035$ |
| 0.03 | 5,126,972 | 0.917 | $-0.055$ |
| 0.01 | 1,708,990 | 0.937 | $+0.275$ |
| 0.003 | 512,697 | 0.863 | $+1.400$ |

Below $n$-scale $\approx0.1$ the *estimator* fails, not the interval: $(a_1,\omega_1)$
stops being identified once the noise swamps the correction, and at $0.003$ the fit
returns $\hat a_1\approx-2\times10^{10}$. So the calibration established here belongs
to the regime Experiment B actually ran in, and does not transfer automatically to a
cheaper one. **Check coverage again before trusting an error bar at a smaller budget.**

#### Validating the planting instead of re-running srw

The coverage arm draws $\overline Y_i$ from $\mathcal N(\mu_i,\sigma_i^2/n_i)$ with the
exact moments of $\lvert S_k\rvert$ rather than simulating, which is what makes 2000
trials affordable. Doing the same with real srw draws is not affordable in the regime
that works — $\approx160$ s per trial at $n$-scale $0.1$, i.e. hours for a coverage
interval too wide to conclude anything.

So the Gaussian assumption is tested directly instead, by `--arm planting`:
many independent real srw $\overline Y_i$, one-sample KS against the planted normal.
Run at **small** $n$ on purpose — $\overline Y$ is a sample mean, so its normality can
only improve as $n$ grows, and a pass at $n=10^4$ implies a pass at Experiment B's
$1.7\times10^5$–$1.7\times10^8$. At $n=10^4$, 3000 draws per scale:

| $k$ | exact se | observed sd | KS | $p$ |
|---|---|---|---|---|
| 8 | 1.793e-2 | 1.766e-2 | 0.0273 | 0.022 |
| 16 | 2.475e-2 | 2.456e-2 | 0.0151 | 0.496 |
| 32 | 3.456e-2 | 3.435e-2 | 0.0169 | 0.354 |
| 64 | 4.855e-2 | 4.874e-2 | 0.0178 | 0.296 |
| 128 | 6.843e-2 | 6.818e-2 | 0.0161 | 0.412 |
| 256 | 9.661e-2 | 9.584e-2 | 0.0135 | 0.638 |

#### What changed as a result

The printed $\pm$ values are unchanged — a standard error is a fine thing to report.
What changed is every place an se becomes a *decision*. `src/plot_allocation.py`'s
"consistent / DISCREPANT" rule used $\lvert z\rvert<2$, which is only right when the
se is known exactly; here it is built partly from $\omega_1$'s replicate se. It now
combines components with `tools/coverage.py`'s `combine_se` (Welch–Satterthwaite for
the effective dof) and cuts at $t(\mathrm{dof}_{\mathrm{eff}})$:

```
consistency cut-off |z| < 2.111 (t at 16.9 effective dof, not the normal's 1.960)
  at prop:opt's m0   -0.3502 +/- 0.0116   delta -0.0195 +/- 0.0172   z = -1.14  consistent
  at the best m0     -0.3259 +/- 0.0116   delta +0.0049 +/- 0.0172   z = +0.28  consistent
```

Both verdicts are unchanged — the correction widens the cut-off rather than moving any
conclusion, which is the outcome to hope for from a calibration fix. But the rule is
now the one it always claimed to be.

### The article's own Wilson interval, eq. (720) — for $\gamma$ (2026-08-24)

The calibration above fixed the replicate interval by widening its quantile. The
article already contains a different answer, Theorem `thm:wilson`:

$$\lvert\hat\beta-\beta\rvert \le \underbrace{\tfrac{6}{m(m+1)}\Big[\tfrac{a_1\rho^{-\omega_1m_0}}{\rho^{\omega_1}-1}+\phi^{+}\tfrac{\rho^{-\omega_2m_0}}{\rho^{\omega_2}-1}\Big]}_{\mathcal B_{\mathrm{fs}}} + \underbrace{\tfrac{6(c_0+2)\sigma^2_{\max,m_0}}{n(m+1)}}_{\mathcal B_{\mathrm{good}}} + \mathcal B_{\mathrm{bad}} + \Phi(\alpha)\underbrace{\sqrt{\tfrac{12\sigma_\infty^2}{nm^3}}}_{\sigma_{\mathrm{se}}}$$

with $c_0=4\log 2-2$ and $\beta=\gamma\log\rho$. Implemented as `tools/wilson.py`,
**for $\gamma$ only** — that is the theorem's own scope, not a shortcut: eq. (720)
describes $\hat\beta=\sum_k w_{k,m}\log\overline Y_{\rho^k}$ and says nothing about
$\omega_1$ or $a_1$, which come from `correction.py`'s nonlinear fit the article does
not analyse.

**Why bother, when $t_4$ already restored calibration.** The replicate interval
estimates its own width from 5 numbers, which forces $t_4=2.776$ instead of $1.960$.
Eq. (720)'s fourth term is a *closed form* in $\sigma_\infty^2$, and $\sigma_\infty^2$
is estimated from the raw samples — $1.7\times10^8$ of them at $k=256$, relative error
$\sim1/\sqrt{2n}\approx5\times10^{-5}$. At that precision $\sigma_{\mathrm{se}}$ is
*known*, not estimated, and $\Phi(\alpha)=1.960$ is legitimate. No replicates required.

**Both derivations check out against measurement** (planted arm, Experiment B's grid,
$m_0=2$, 3000 draws):

| quantity | closed form | measured | ratio |
|---|---|---|---|
| bias, our $C_b\rho^{-\omega_1m_0}$ (Part I of the derivations note) | $8.091\times10^{-3}$ | $8.069\times10^{-3}$ | 0.997 |
| sd, eq. (720)'s $\sigma_{\mathrm{se}}$ | $4.313\times10^{-4}$ | $4.267\times10^{-4}$ | 0.989 |
| eq. (720)'s $\mathcal B_{\mathrm{fs}}$ *bound* | $1.288\times10^{-2}$ | $8.069\times10^{-3}$ | 1.60 — conservative, as a bound must be |

#### The decisive comparison

1500 trials, **equal total sampling budget**, on the article's own estimator
`gamma_closed_form` — the replicate route spends it as $R=5$ passes of $n=N/5$, the
Wilson route as one pass of $n=N$:

| $m_0$ | replicate $t_4$ coverage | half-width | Wilson coverage | half-width |
|---|---|---|---|---|
| 2 | **0.000** | $1.00\times10^{-3}$ | 1.000 | $1.36\times10^{-2}$ |
| 4 | **0.021** | $9.73\times10^{-4}$ | 1.000 | $3.94\times10^{-3}$ |
| 6 | **0.835** | $9.66\times10^{-4}$ | 0.996 | $1.52\times10^{-3}$ |
| 8 | 0.945 | $9.65\times10^{-4}$ | 0.982 | $9.17\times10^{-4}$ |
| 10 | 0.955 | $9.64\times10^{-4}$ | 0.966 | $\mathbf{7.66\times10^{-4}}$ |

Two things to read off.

**The replicate interval does not merely undercover on shallow ladders — it fails
completely**, because it contains no bias term at all. At $m_0=2$ the estimator's true
bias is $8.07\times10^{-3}$, eight half-widths from truth, so no quantile could rescue
it. This was invisible in the coverage work above only because that measured
`fit_correction`'s $\gamma$, which fits the correction away and so carries bias
$-1.8\times10^{-4}$ instead. **Two different $\gamma$ estimators with a 44x difference
in bias** — on this grid the nonlinear fit is 9.5x better in RMSE (it trades 3.8x
variance for the bias), though at Experiment C's tuned $m_0$ the ranking reverses.

**Where the bias is negligible the bound is both valid and narrower** — $7.66$ against
$9.64\times10^{-4}$ at $m_0=10$, a 20% gain, rising to 29% asymptotically
($1.960/(t_4/\sqrt5)=0.706$). Spending the budget on one deep pass beats splitting it
into five, once you no longer need replicates to learn the width.

#### Caveats recorded in the module

- **It is a bound, not an interval**: it adds $|\text{bias}|$ rather than recentring, so
  it overcovers. That is the price of validity, and it is measured above rather than
  assumed.
- **Uniform $n$.** Eq. (720) assumes it; Experiment B's snr allocation spans
  $1.7\times10^5$ to $1.7\times10^8$. Substituting a mean $n$ gives $4.2\times10^{-5}$
  against the correct $4.3\times10^{-4}$ — **wrong by a factor of ten**.
  `sigma_se_per_scale` generalises the fourth term; the three bias terms would need
  rederiving, so the module refuses rather than guessing.
- **$\mathcal B_{\mathrm{bad}}$ needs $(\Lambda,\delta)$ and the $\omega_2$ piece needs
  $\phi^{+}$**, none of which we have measured. They are omitted only behind a
  `complete=False` flag that `format_interval` prints in the first line — a bound
  missing a term is not a bound, and the reported half-width is then a *lower* estimate
  of the true one. `moment_bounds` will estimate $\Lambda$, $\sigma_\infty^2$ and
  $\sigma^2_{\max}$ from real samples when we want the complete version.

### Replicate groups on disk, and how to check the predictions

Same-configuration replicates live in one folder, one subfolder per replicate
(`--tag` may contain `/`, and `--seed` overrides the recipe's, so the loop is a
one-liner):

```bash
cd src
for r in 0 1 2 3 4; do
  python3 generate.py -meta ../experiments/01_srw/omega1_config.json \
      --tag omega1_b5e10/rep$r --seed $((20260900+r))
  python3 estimate_omega1.py -data ../experiments/01_srw/data/omega1_b5e10/rep$r
  rm -rf ../experiments/01_srw/data/omega1_b5e10/rep$r/samples   # keep omega1.json!
done
```

`allocation_table.py --list` shows what it found, grouped by `(model, scales, n)` —
only runs agreeing on all three are replicates of one configuration and may be pooled:

```
  #  group                      reps  scales                 n per scale
  1  omega1                        6  8..256 (6)             166,893..170,899,089
  2  Huge_test                     1  2..1024 (10)           100,000,000..100,000,000
```

`--group <name>` picks one; without it the group with the most replicates wins, and if
several tie *and* stdin is a terminal you are asked. It never prompts in a pipeline —
`--no-prompt` forces that off entirely.

#### Are single runs unstable? Yes — badly, in $a_1$ and $\omega_1$; not at all in $\gamma$

Six independent replicates of the *identical* configuration ($B=5\times10^{10}$,
scales $8\dots256$):

| run | $\hat a_1$ | $\hat\omega_1$ | $\hat\gamma$ | offset | recommended $m_0$ at $B=10^{12}$ |
|---|---|---|---|---|---|
| `omega1` | $-0.2217$ | — | — | $-4.141$ | 9 |
| `rep0` | $-0.3857$ | 1.198 | 0.5005 | $-3.609$ | 10 |
| `rep1` | $-0.2475$ | 0.965 | 0.4998 | $-4.035$ | 9 |
| `rep2` | $-0.1104$ | 0.486 | 0.4971 | $-4.812$ | 8 |
| `rep3` | $-0.3838$ | 1.254 | 0.5008 | $-3.613$ | 10 |
| `rep4` | $-0.2291$ | 0.987 | 0.5001 | $-4.109$ | 9 |
| **pooled (6)** | $\mathbf{-0.2353\pm0.0432}$ | | | $-4.084$ | **9** |
| **truth** | $-0.25$ | 1 | 0.5 | $-3.998$ | |

$\hat a_1$ spans a factor of 3.5 and $\hat\omega_1$ ranges 0.49–1.25, while $\hat\gamma$
never leaves $0.497$–$0.501$. **Never trust a single run's $a_1$ or $\omega_1$.**

But the *recommendation* is far more stable than its inputs, because the offset depends
on $a_1$ only logarithmically: the six runs disagree by at most 1–2 steps in $m_0$, and
the pooled estimate lands $0.086$ steps from the truth — under 1% in RMSE. Even the worst
single run (`rep2`) is one step off, costing 19%.

#### Are the predictions themselves right? The timing, essentially exactly

`src/verify_prediction.py` runs each tuned ladder for real and compares
(~4 min at the defaults; keep the machine idle, it measures wall clock):

```bash
cd src && python3 verify_prediction.py --m0 3 4 5 6 7 --replicates 3
```

| $m_0$ | $n$ | predicted | measured | ratio | pred RMSE | meas RMSE | ratio |
|---|---|---|---|---|---|---|---|
| 3 | 2,478 | 0.016 s | 0.015 s | 0.94× | $6.59\times10^{-3}$ | $7.24\times10^{-3}$ | 1.10× |
| 4 | 9,912 | 0.130 s | 0.125 s | 0.96× | $3.30\times10^{-3}$ | $2.54\times10^{-3}$ | 0.77× |
| 5 | 39,649 | 1.042 s | 1.041 s | 1.00× | $1.65\times10^{-3}$ | $2.30\times10^{-3}$ | 1.40× |
| 6 | 158,598 | 8.337 s | 8.347 s | 1.00× | $8.24\times10^{-4}$ | $7.83\times10^{-4}$ | 0.95× |
| 7 | 634,393 | 66.69 s | 66.56 s | 1.00× | $4.12\times10^{-4}$ | $4.39\times10^{-4}$ | 1.06× |

**Timing: 0.94×–1.00×, median 1.00×, over four orders of magnitude.** The wall-clock
column of the planning table can be taken at face value on this machine — unsurprising
once the cost model is right, since cost is genuinely $\Theta(nk)$ and throughput is a
single calibrated constant. Re-derive it on other hardware
(`--throughput`, or rerun Experiment C).

**Accuracy: 0.77×–1.40×, median 1.06×** — consistent, but note an RMSE from $R=3$
replicates has a relative sd of $1/\sqrt{2R}\approx41\%$ on its own, so this column is an
order-of-magnitude check rather than a calibration. Raise `--replicates` to tighten it.

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
