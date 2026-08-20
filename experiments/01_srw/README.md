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
