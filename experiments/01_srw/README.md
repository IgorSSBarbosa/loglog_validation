# 01_srw — Simple Random Walk

This folder holds two separate, independent uses of the same `srw()` simulator. Don't
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

- `srw.py` — the simulator, `srw(k, q=0.5, rng=None)`, returning one realization of
  $|S_k|$.
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
