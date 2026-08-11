# 01_srw — Simple Random Walk

This folder holds three separate, independent uses of the same `srw()` simulator. Don't
conflate them.

## Gamma-estimation ladder — blocked on design

The article's `appendix-SimpleRandomWalk` is currently an empty section header — no
closed-form $\mathbb{E} Y_i$, $\gamma$, or $\omega_1$ is written down for this testbed
yet. Needs discussion (which observable $Y_i$: return probability at step $i$, range
after $i$ steps, something else?) before any code is written for *this* purpose. See
`PLAN.md` "Open questions before Phase 1". Phase 1 proper has not started.

## Cost-model probe — not blocked, done

Separate from the above: measures the computational-cost exponent $d$ (article
Assumption `cost_is_power_law`, $\mathrm{cost}(i) = i^d$), needed for the budget
allocation rule (Definition `def:alloc`) but not yet implemented in this repo
(`tools/allocation.py`). `experiments/00_synthetic/generator.py` can't be used for this
— drawing samples from a closed-form formula costs the same regardless of scale, so it
would just measure $d\approx0$. `srw(k)` (`srw.py`) is used here purely as a fixture
whose cost genuinely grows with $k$: generating $k$ i.i.d. $\pm1$ steps and summing them
is $\Theta(k)$, i.e. $d=1$ — a known ground truth to validate the *measurement
procedure* against, before ever pointing it at a real (and expensive) simulator.

- `srw.py` — the simulator, `srw(k, n=1, q=0.5, rng=None)`, returning `n` i.i.d.
  realizations of $|S_k|$ as an array (vectorized: one $(n,k)$ matrix of $\pm1$ steps,
  summed along the $k$ axis). `measure_cost.py` calls it at the default `n=1` --
  Assumption `cost_is_power_law` defines $\mathrm{cost}(i)$ as the cost of simulating
  *one* sample -- `n>1` is what "Sample generation" below uses.
- `tools/cost_model.py` — generic, experiment-agnostic estimator: `cost(i)=c\cdot i^d`
  has the same log-log-linear form as $\mathbb{E} Y_i=a_0 i^\gamma$ (eq. 232), so
  `estimate_cost_exponent` reuses `tools/loglog.py`'s OLS-slope machinery, behind a
  name-keyed registry (`COST_ESTIMATORS`, mirrors `generator.py`'s `NOISE_FAMILIES`) so
  a different estimation approach can be added later without touching callers.
- `measure_cost.py` — the driver: times `srw(k)` at a small grid of scales (default
  `[256, 1024, 4096, 16384, 65536, 262144, 1048576]`), 20 repeats each, aggregated by
  **minimum** (not mean — repeated timings all target the same true deterministic
  quantity, so noise only ever adds delay; this is the one place this experiment departs
  from the sample-mean framing used elsewhere for genuinely stochastic $Y_i$). Also
  prints a `gamma_drop_leading`-style view (drop the first $m_0$ scales) as a
  finite-overhead diagnostic — fixed per-call overhead dominates at the smallest $k$,
  the same shape as the article's own $m_0$ finite-size correction; empirically the
  local slope climbs from $\approx 0.90$ (all 7 scales) to $\approx 1.0$–$1.09$ once the
  first scale or two are dropped.
- `test_cost_probe.py` — the numeric acceptance criterion, executable: $\hat d \in
  [0.8, 1.2]$ over the default grid/repeats. Verified: `python3 -m pytest
  experiments/01_srw/` passes; `images/cost_probe.png` (committed, per ground rule 1 —
  supplement to the numeric check, not a replacement for it) shows a clean log-log line
  once past the small-$k$ overhead-dominated points.

Run directly: `python3 measure_cost.py` (writes `data/<tag>.json`, gitignored). Pass
`--plot` to also save the log-log plot to `images/<tag>.png` -- off by default so a
routine rerun doesn't silently overwrite the committed evidence figure.

**Not done here:** `tools/allocation.py` (Definition `def:alloc`'s $n$, $m_0$ formulas)
and checkpoint 0.5 itself — separate future work, once $d$-measurement is trusted on a
real (not toy) simulator.

## Sample generation — not blocked, exploratory (not Phase 1)

`generate.py` is the second consumer of `tools/persistence.py` (the sample+metadata
save/load pattern extracted from `experiments/00_synthetic/generator.py` once this
became a real second use case): draws $n$ i.i.d. $|S_k|$ samples per scale $k$ via
`srw(k, n, q, rng)`, seeded and timed the same way, to the identical `.npz`+`.json`
output shape. Recipe shape mirrors `generator.py`'s exactly, with `"params": {"q":
0.5}` standing in for `SyntheticParams`:

```
python3 generate.py -meta example_config.json --tag demo_run
python3 plot_loglog.py -data data/demo_run.npz
```

`plot_loglog.py` hands the saved samples straight to `tools/loglog_plot.py`'s generic
`loglog_plot` — same tool 00_synthetic uses. **Deliberately does not** run
`tools/loglog.py`'s $\hat\gamma$ estimators or overlay a reference curve, unlike
00_synthetic's version: there is still no article-sanctioned closed-form
$\mathbb{E} Y_i$/$\gamma$/$\omega_1$ for SRW (see "Gamma-estimation ladder" above,
still blocked), so any $\hat\gamma$ computed from this data would be an unvalidated
number, not a checkpoint result — don't mistake this for Phase 1 progress. What this
*does* establish: the log-log plot of $\overline{|S_k|}$ vs $k$ is visibly a clean
straight line (slope $\approx 0.5$, consistent with the classical $\mathbb{E}|S_k|
\sim \sqrt{2k/\pi}$ asymptotic) — useful groundwork for whenever Phase 1's design
question is resolved, but not itself a validated result.

`test_generate.py` checks output shapes match the recipe and that `reproduce()`
(regenerate from recorded metadata) matches a saved run exactly, the same
independent correctness check `generator.py`'s `reproduce` provides.
