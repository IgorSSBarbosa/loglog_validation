# src

The scripts a human runs directly — as opposed to `tools/`, which holds the functions
those scripts (and each other) call (user's own framing, 2026-08-12). One shared copy
of each action, used by every experiment via a recipe's `"model"` field (dispatched
through `tools/models.py`'s registry), instead of each experiment keeping its own copy.

Twelve drivers in five layers, named for the question they answer:

```
src/
  generate/   generate.py                                 draw samples
  estimate/   measure_cost.py  estimate_omega1.py         measure the constants
  budget/     allocation_experiment.py  allocation_table.py    spend a budget well
  report/     plot_loglog.py  plot_cost.py  plot_allocation.py    say what happened
  study/      pilot.py  plan.py  run.py  report.py    the whole thing, end to end
```

**If you just want $\hat\gamma$ for a model, use `src/study/` and read
`src/study/README.md`.** The four layers below are the pieces it orchestrates;
running them by hand is for when you want a specific piece, not the whole answer.

Scripts whose subject is **this repo's own machinery** rather than a model live in
`calibration/` instead — `check_coverage.py` (are the error bars honest?) and
`verify_prediction.py` (does a predicted runtime predict?). See
`calibration/README.md`.

Run everything from the repo root; the paths below assume it.

```bash
# 1. Generate samples
python3 src/generate/generate.py -meta experiments/00_synthetic/recipes/samples_example.json --tag demo_run
python3 src/generate/generate.py -meta experiments/01_srw/recipes/samples_example.json --tag demo_run

# 2. Plot them (log-log) -- add --estimates for the 4-estimator comparison chart
python3 src/report/plot_loglog.py -data experiments/00_synthetic/data/demo_run
python3 src/report/plot_loglog.py -data experiments/01_srw/data/demo_run --estimates

# 3. Measure the cost-model exponent d (only meaningful for models whose cost
#    genuinely grows with scale, e.g. srw)
python3 src/estimate/measure_cost.py -meta experiments/01_srw/recipes/cost_probe.json --tag cost_probe

# 4. Plot that timing data
python3 src/report/plot_cost.py -data experiments/01_srw/data/cost_probe

# 5. Estimate the correction-to-scaling exponent omega_1 from a run (Experiment B)
python3 src/generate/generate.py -meta experiments/01_srw/recipes/samples_omega1.json --tag omega1
python3 src/estimate/estimate_omega1.py -data experiments/01_srw/data/omega1 --expect-omega1 1.0

# 6. Test the budget-allocation rule prop:opt (Experiment C)
python3 src/budget/allocation_experiment.py -meta experiments/01_srw/recipes/sweep_allocation.json --tag allocation

# 7. Plan a run: precision vs wall-clock, using the tuned allocation
python3 src/budget/allocation_table.py --list          # which run groups are available
python3 src/budget/allocation_table.py --compare

# 8. Check those predictions against reality -- calibration/, not src/
#    (a few minutes; keep the machine idle)
python3 calibration/verify_prediction.py --m0 3 4 5 6 7 --replicates 3

# 9. Plot Experiment C: the m0 tradeoff, and measured vs predicted decay rate
python3 src/report/plot_allocation.py -data experiments/01_srw/data/allocation

# 10. Are the error bars themselves calibrated? -- calibration/, checkpoint 0.4
python3 calibration/check_coverage.py --arm planted --trials 2000 --centre both
python3 calibration/check_coverage.py --arm planting --trials 3000   # is the planting faithful?
python3 calibration/check_coverage.py --arm wilson --trials 1000     # eq. (720) as a bound on gamma
```

---

## `generate/`

`generate.py` — the shared sample-generator CLI/API (`generate`, `reproduce`).
Recipe: `{"kind": "samples", "model": ..., "params": {...}, "scales": [...],
"n": ..., "seed": null}`. `"n"` is a scalar, an explicit per-scale list, or an
**allocation rule** — `{"rule": "snr", "budget": 5e10, "d": 1.0, "omega1": 1.0}` or
`{"rule": "neyman", "budget": ..., "d": ...}` — which keeps a recipe self-describing
about *why* those sample counts were chosen (see `tools/allocation.py`; `snr` is the
right one for measuring $\omega_1$, `neyman` the documented wrong one).

A rule's `d` and `omega1` are **design inputs**: they decide how the budget is *split*
across scales and never reach an estimator — the fit sees only the drawn samples.
They are still not allowed to be silent. `d` is taken from the model's own declared
`cost_hint` when it has one (srw's says $\Theta(k)$, so $d=1$ by construction and the
key can be omitted); an explicit `"d"` overrides it. `omega1` has no such source and
must be stated — omitting it is an error naming what the input is, not a `KeyError`.
Recipes are also checked for **starved scales**: under `snr`, $n_i\propto i^{2\omega_1}$
spans $(i_{\max}/i_{\min})^{2\omega_1}$ across the ladder, so a wide ladder on a small
budget floors the smallest scales at $n=1$ — which has no standard error, and
$\sigma_{\log}$ is what the log-log fit weights by. The allocation warns, and
`pilot.py` refuses before drawing anything.

`seed` accepts an int, a `SeedSequence`, or `rng.seed_record`'s dict. The
`SeedSequence` spelling is what lets a caller obeying ground rule 2 hand over one of
`spawn`'s children — never as `child.entropy`, which would give every sibling the same
stream (`tools/rng.py`). `reduce` applies a statistic to each scale as it is drawn and
drops the samples: `reduce=np.mean` retains one scale's array instead of the whole
ladder's (381 MB → 63 MB at Experiment C's largest cell), and is what makes it
practical for the sweep to call this function rather than repeat its loop. It is
refused together with `out_dir`, which would write summaries to a file named samples.

Default output directory is the **experiment's** `data/`, resolved by
`tools/artifacts.py`'s `default_out_dir` — since recipes live in
`experiments/<exp>/recipes/`, that is the recipe's grandparent, not its sibling. Runs
whose total estimated size (`sum(n) * 8` bytes) exceeds `max_chunk_bytes` (default
~1 GiB) are streamed straight to on-disk per-scale arrays
(`tools.persistence.open_scale_writer`) in chunks, instead of assembled fully in RAM
and saved once at the end — the fix for OOM on very large `n` (see
`experiments/01_srw/README.md`'s "Fixed-memory generation" section); a `psutil`-based
backstop (`mem_flush_pct`, default 90%) shrinks the chunk size further if system
memory gets tight mid-run. Ordinary-sized runs are unaffected — same `samples.npz`
output as always. Verified: `tools/tests/test_generate.py` (shapes match the recipe,
`reproduce()` exact-match — both parametrized across every registered model — plus the
chunked path matching the in-RAM path exactly for the same seed).

## `estimate/`

`measure_cost.py` — the standalone cost-model-exponent probe, same registry dispatch
as `generate.py` (times `MODELS[model].simulate(i, n=1, ...)` instead of drawing real
samples). The timing and the fitting are `tools/cost_model.py`'s `time_over_scales`
and `fit_cost_probe`, shared with the pilot's probe in `src/study/pilot.py` — the two
differ only in choosing their scales, and this one's ladder *is* the experiment: a
named grid, its per-scale confidence intervals, and the acceptance check below. Only
meaningful for models whose cost genuinely grows with scale (e.g.
`"srw"`); pointed at `"synthetic"` it will just measure $d\approx0$, an expected,
uninteresting result. Repeated timings at each scale are collapsed by the aggregator
named in the recipe's optional `"aggregator"` key (`tools/cost_model.py`'s
`AGGREGATORS`), defaulting to `median`. Two cost models are fitted and both reported:
the pure $\mathrm{cost}(i)=c\,i^d$ of Assumption `cost_is_power_law`, and
$\mathrm{cost}(i)=a+b\,i^d$, which additionally accounts for the fixed per-call
dispatch overhead a single `simulate(k, n=1, ...)` pays. **Prefer the affine one
whenever the printed overhead share at the smallest scale is non-trivial** — the
script says so explicitly when it exceeds 20%, and the acceptance check uses the
affine $\hat d$. The result is also checked against the model's own declared
`cost_hint` when it has one, so a simulator that claims $\Theta(k)$ has to prove it.
Verified: `tools/tests/test_measure_cost.py` (real-timing acceptance check, affine
$\hat d\in[0.8,1.2]$ for `srw`; that the overhead is detected as strictly positive and
is what biases the pure fit low; and that the aggregator is selectable and recorded).

`estimate_omega1.py` — Experiment B's analysis driver: reads a run directory and
applies both of `tools/correction.py`'s $\omega_1$ estimators to it, writing
`<run_dir>/omega1.json`. Never draws samples (generation and analysis are always
separate scripts here). `--expect-omega1`/`--expect-gamma` print a PASS/FAIL against
known values for reporting only — those values are never passed to the estimators, so
the measurement stays blind to the answer it is checking. See
`experiments/01_srw/README.md` for the recipe, the allocation rule, and the result.

## `budget/`

`allocation_experiment.py` — Experiment C's driver: sweeps a (budget × $m_0$) grid,
drawing $R$ independent replicates per cell (`tools/rng.py`'s `spawn`, ground rule 2)
and estimating $\gamma$ with the article's own closed-form $w_{k,m}$ weights, to test
Proposition `prop:opt`. Runs **paired arms** — the same grid scored against `prop:opt`'s
$m_0$ and against the tuned $m_0$ that puts the dropped constant back — so the two
rules are compared on identical draws rather than on separate runs. Reports per-cell
bias/sd/RMSE, the empirically best $m_0$ against the one each rule names, and the
measured error-decay exponent against $-\omega_1/(d+2\omega_1)$. Does **not** persist
samples — the sweep draws far more than is worth storing, and the per-cell
$\hat\gamma$ values in `<run_dir>/allocation_sweep.json` are the result (the base seed
is recorded and spawning is deterministic, so any cell regenerates exactly).
Verified: `tools/tests/test_allocation_experiment.py`. Result and interpretation in
`experiments/01_srw/README.md`.

`allocation_table.py` — the budget-planning table: given the measured constants, how
long must a run take for a target precision on $\hat\gamma$? One row per integer $m_0$,
at the budget where that $m_0$ is optimal, showing $n$, wall-clock cost and predicted
RMSE. Reads its inputs from this repo's own results rather than hardcoding them ($a_1$
from Experiment B's `omega1.json`, the coefficient of variation from that run's
samples, throughput from Experiment C's wall clock), each overridable by flag and each
falling back to the last measured value if the run is absent. `--compare` adds what
`prop:opt`'s uncalibrated $m_0$ would have cost; `--csv` saves the table.
**`--by-budget` indexes rows by budget instead of by $m_0$, and is the view to use when
comparing two runs of this script** — the default view asks "at what budget is this
$m_0$ optimal", so any change in the measured constants slides every row along the
budget axis (see `experiments/01_srw/README.md`, "I ran it twice and got two different
tables"). A run directory may be passed directly to `--data-root`, and any input that
falls back to a hardcoded constant prints a loud warning rather than quietly pretending
to be a measurement. Replicates are **pooled and refitted once**, not averaged
fit-by-fit: the fit is nonlinear, so averaging separate fits converges to
$\mathbb{E}[\hat a_1]$ rather than $a_1$, a bias no number of replicates removes (see
`measured_a1`'s docstring for the measured numbers). Pooling uses the
`y_bar`/`n`/`sigma_log` stored in `omega1.json`, so the samples need not be kept.
Verified: `tools/tests/test_allocation_table.py` (20 cases).

## `report/`

`plot_loglog.py` — the shared log-log plotter, reading a run directory's own
`samples_meta.json` to look up its model in `tools/models.py`'s registry. The raw-data
plot (`<run_dir>/plot.png`) always overlays the all-points OLS fit (solid line,
$\hat\gamma$ in the legend) — needs no known ground truth, it's computed from the data
itself — and additionally overlays the known $\mathbb{E} Y_i$ curve (dashed) when the
model has a `target_fn` (currently only `"synthetic"`). `tools/loglog.py`'s
four-estimator comparison (`compare_methods`) always runs and is always written to
`<run_dir>/gamma_estimates.json`, for every model — comparing estimators against each
other doesn't need a known truth, only comparing against one does; when `true_gamma` is
unknown, a printed note makes clear the numbers are exploratory, not a validated
checkpoint result (see `experiments/01_srw/README.md`). The comparison *chart*
(`estimates.png`) is opt-in via `--estimates`, since unlike `gamma_estimates.json` it
is a supplementary figure (ground rule 1), not the numeric result itself.

`plot_cost.py` — separate from `measure_cost.py` (generation and plotting are always
separate scripts here). Writes `<run_dir>/plot.png` next to `cost_probe.json`, same
one-run-one-folder convention as `plot_loglog.py`.

`plot_allocation.py` — Experiment C's two panels, `<run_dir>/plot.png`. Left: RMSE vs
$m_0$ per budget, with the empirical argmin (●) and `prop:opt`'s choice (✕) marked, so
the systematic gap between them — the POINT failure — is visible. Right: RMSE vs budget
on log-log with the fitted power law, against the exponent $-\omega_1/(d+2\omega_1)$
predicted from **measured** $\omega_1$ (Experiment B, pooled) and **measured** $d$ (the
cost probe's affine fit), banded by the propagated uncertainty — the cross-experiment
check that B's $\omega_1$ predicts C's decay rate. Both the prediction and the fitted
slope carry standard errors, and the comparison is reported as a z-score — omitting the
slope's own error once made a consistent result read as a 3σ discrepancy. Budgets whose
`prop:opt` $m_0$ was not swept are dropped from the right panel and named on stderr,
rather than silently leaving it empty. Palette is `tools/loglog_plot.py`'s unchanged.
Verified: `tools/tests/test_plot_allocation.py` (5 cases — the chart itself is not
unit-tested, per the convention for plots; the series selection and fitted exponents
are).

---

## Conventions these eight share

**Recipes are inputs and declare their kind.** A recipe lives in
`experiments/<exp>/recipes/<kind>_<name>.json` and carries a `"kind"` field
(`samples`, `cost_probe`, `sweep`). Drivers load it through
`tools/artifacts.py`'s `load_recipe(path, expect=...)`, which refuses a mismatch with
an error naming the mistake — recipes all start with `model` and `params`, so a wrong
one used to travel a long way before failing on a missing key.

**Outputs are named for what they are.** Never `result.json`: `samples_meta.json`,
`cost_probe.json`, `omega1.json`, `allocation_sweep.json`, `prediction_check.json`,
`gamma_estimates.json`, `coverage.json`. `tools/artifacts.py` is the single source of
truth for those names, and stamps provenance (`produced_by`, `recipe`, `created`)
*inside* each file rather than in its name — data outlives the script that wrote it.
Everything about one run lives in the same folder as `samples.npz`, and since `data/`
is gitignored, nothing here is auto-committed — copy a specific plot into the
experiment's `images/` folder when you want to keep it as evidence (ground rule 1/6 —
committed deliberately, one at a time).

**The draw itself is `generate()`'s, everywhere it can be.** `allocation_experiment`
and `verify_prediction` used to hand-roll the same three-line ladder loop; both now
call `generate(..., reduce=np.mean)`, pinned bit-for-bit by
`test_run_cell_draws_through_generate` / `test_run_ladder_draws_through_generate` so
the published sweep stays reproducible. Two draw sites are deliberately *not*
converted: `measure_cost` times a single `simulate` call and must not pay `generate`'s
bookkeeping, and `check_coverage._srw_replicate` is a callback the coverage harness
drives from one long-lived `Generator`, so it takes an rng rather than a seed — its
equivalence to `generate` is asserted by `test_srw_replicate_matches_generate` instead
of by a comment.

**Each driver bootstraps its own `sys.path`.** Two levels up from `src/<layer>/x.py` is
the repo root, so each inserts `ROOT/"tools"` (plus a sibling layer where it needs one)
before importing `tools/*.py` by bare name — which is what lets them be run from
anywhere without an install step. `tools/tests/test_artifacts.py` compiles every file
under `src/`, `tools/` and `models/` and runs `--help` on every entry point, after a
`from __future__` ordering bug shipped past 288 tests because nothing imported the two
modules it broke.
