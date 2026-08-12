# 01_srw — Simple Random Walk

This folder holds three separate, independent uses of the same `srw()` simulator
(`models/srw.py`, registered as `MODELS["srw"]` in `tools/models.py`). Don't
conflate them. As of this session, the scripts that drive all three are shared with
every other experiment (`src/generate.py`, `src/plot_loglog.py`,
`src/measure_cost.py`, `src/plot_cost.py` — see `src/README.md`); this folder now
holds only recipes, README, `data/` (gitignored), and `images/` (committed evidence).

## Gamma-estimation ladder — blocked on design

The article's `appendix-SimpleRandomWalk` is currently an empty section header — no
closed-form $\mathbb{E} Y_i$, $\gamma$, or $\omega_1$ is written down for this testbed
yet. Needs discussion (which observable $Y_i$: return probability at step $i$, range
after $i$ steps, something else?) before any code is written for *this* purpose. See
`PLAN.md` "Open questions before Phase 1". Phase 1 proper has not started. This is
exactly why `MODELS["srw"]` in `tools/models.py` has no `target_fn`/`true_gamma_key` —
that absence is what keeps `src/plot_loglog.py` from running gamma-hat estimators
against this model at all (see "Sample generation" below), not a special case coded
into the driver.

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

- `models/srw.py` — the simulator, `srw(k, n=1, q=0.5, rng=None)`, returning `n`
  i.i.d. realizations of $|S_k|$ as an array (vectorized: one $(n,k)$ matrix of $\pm1$
  steps, summed along the $k$ axis). `src/measure_cost.py` calls it at the default
  `n=1` -- Assumption `cost_is_power_law` defines $\mathrm{cost}(i)$ as the cost of
  simulating *one* sample -- `n>1` is what "Sample generation" below uses.
- `tools/cost_model.py` — generic, experiment-agnostic estimator: `cost(i)=c\cdot i^d`
  has the same log-log-linear form as $\mathbb{E} Y_i=a_0 i^\gamma$ (eq. 232), so
  `estimate_cost_exponent` reuses `tools/loglog.py`'s OLS-slope machinery, behind a
  name-keyed registry (`COST_ESTIMATORS`) so a different estimation approach can be
  added later without touching callers.
- `src/measure_cost.py` — the shared driver: times `MODELS[model].simulate(k, n=1,
  ...)` at a small grid of scales (`cost_probe_config.json`: `[256, 1024, 4096, 16384,
  65536, 262144, 1048576]`), 20 repeats each, aggregated by **minimum** (not mean --
  repeated timings all target the same true deterministic quantity, so noise only ever
  adds delay; this is the one place this experiment departs from the sample-mean
  framing used elsewhere for genuinely stochastic $Y_i$). Also prints a
  `gamma_drop_leading`-style view (drop the first $m_0$ scales) as a finite-overhead
  diagnostic — fixed per-call overhead dominates at the smallest $k$, the same shape as
  the article's own $m_0$ finite-size correction; empirically the local slope climbs
  from $\approx 0.90$ (all 7 scales) to $\approx 1.0$–$1.09$ once the first scale or two
  are dropped.
- `src/plot_cost.py` — separate from `measure_cost.py` (generation and plotting are
  always distinct scripts in this repo). Reads `measure_cost.py`'s saved
  `<tag>/result.json` directly (`elapsed_all` is already `{scale: array}`,
  `tools/loglog_plot.py`'s exact required shape) and hands it to the same generic
  `loglog_plot`.

Numeric acceptance criterion (`tools/tests/test_measure_cost.py`, local-only — see
`tools/README.md`): $\hat d \in [0.8, 1.2]$ over the default grid/repeats.
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
cd ../../tools    # or wherever tools/ is relative to your cwd
python3 generate.py -meta ../experiments/01_srw/example_config.json --tag demo_run
python3 plot_loglog.py -data ../experiments/01_srw/data/demo_run
```

`plot_loglog.py` hands the saved samples straight to `tools/loglog_plot.py`'s generic
`loglog_plot` — same tool 00_synthetic uses. **Deliberately does not** run
`tools/loglog.py`'s $\hat\gamma$ estimators or overlay a reference curve, unlike
00_synthetic's data: `MODELS["srw"]` has no `target_fn` (see "Gamma-estimation ladder"
above, still blocked), so any $\hat\gamma$ computed from this data would be an
unvalidated number, not a checkpoint result — don't mistake this for Phase 1 progress.
What this *does* establish: the log-log plot of $\overline{|S_k|}$ vs $k$ is visibly a
clean straight line (slope $\approx 0.5$, consistent with the classical
$\mathbb{E}|S_k| \sim \sqrt{2k/\pi}$ asymptotic) — useful groundwork for whenever
Phase 1's design question is resolved, but not itself a validated result.

`tools/tests/test_generate.py` checks output shapes match the recipe and that
`reproduce()` (regenerate from recorded metadata) matches a saved run exactly, for
every registered model including this one (see `tools/README.md`).
