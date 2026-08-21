# src

The scripts a human runs directly (`python3 src/generate.py ...`) — as opposed to
`tools/`, which holds the functions those scripts (and each other) call (user's own
framing, 2026-08-12). One shared copy of each action, used by every experiment via a
recipe's `"model"` field (dispatched through `tools/models.py`'s registry), instead
of each experiment keeping its own copy.

```bash
cd src

# 1. Generate samples
python3 generate.py -meta ../experiments/00_synthetic/example_config.json --tag demo_run
python3 generate.py -meta ../experiments/01_srw/example_config.json --tag demo_run

# 2. Plot them (log-log) -- add --estimates for the 4-estimator comparison chart
python3 plot_loglog.py -data ../experiments/00_synthetic/data/demo_run
python3 plot_loglog.py -data ../experiments/01_srw/data/demo_run --estimates

# 3. Measure the cost-model exponent d (only meaningful for models whose cost
#    genuinely grows with scale, e.g. srw)
python3 measure_cost.py -meta ../experiments/01_srw/cost_probe_config.json --tag cost_probe

# 4. Plot that timing data
python3 plot_cost.py -data ../experiments/01_srw/data/cost_probe

# 5. Estimate the correction-to-scaling exponent omega_1 from a run (Experiment B)
python3 generate.py -meta ../experiments/01_srw/omega1_config.json --tag omega1
python3 estimate_omega1.py -data ../experiments/01_srw/data/omega1 --expect-omega1 1.0

# 6. Test the budget-allocation rule prop:opt (Experiment C)
python3 allocation_experiment.py -meta ../experiments/01_srw/allocation_config.json --tag allocation

# 7. Plan a run: precision vs wall-clock, using the tuned allocation
python3 allocation_table.py --list          # which run groups are available
python3 allocation_table.py --compare

# 8. Check those predictions against reality (a few minutes; keep the machine idle)
python3 verify_prediction.py --m0 3 4 5 6 7 --replicates 3

# 9. Plot Experiment C: the m0 tradeoff, and measured vs predicted decay rate
python3 plot_allocation.py -data ../experiments/01_srw/data/allocation
```

`generate.py` — the shared sample-generator CLI/API (`generate`, `reproduce`).
Recipe: `{"model": ..., "params": {...}, "scales": [...], "n": ..., "seed": null}`.
`"n"` is a scalar, an explicit per-scale list, or an **allocation rule** —
`{"rule": "snr", "budget": 5e10, "d": 1.0, "omega1": 1.0}` or
`{"rule": "neyman", "budget": ..., "d": ...}` — which keeps a recipe self-describing
about *why* those sample counts were chosen (see `tools/allocation.py`; `snr` is the
right one for measuring $\omega_1$, `neyman` the documented wrong one).
Default output directory is `data/` next to the recipe file itself (not this
script's own location, since it's shared across experiments), so each experiment's
runs still land under that experiment's own `data/`. Runs whose total estimated size
(`sum(n) * 8` bytes) exceeds `max_chunk_bytes` (default ~1 GiB) are streamed straight
to on-disk per-scale arrays (`tools.persistence.open_scale_writer`) in chunks, instead
of assembled fully in RAM and saved once at the end — the fix for OOM on very large
`n` (see `experiments/01_srw/README.md`'s "Fixed-memory generation" section); a
`psutil`-based backstop (`mem_flush_pct`, default 90%) shrinks the chunk size further
if system memory gets tight mid-run. Ordinary-sized runs are unaffected — same
`samples.npz` output as always. Verified: `tools/tests/test_generate.py` (shapes
match the recipe, `reproduce()` exact-match — both parametrized across every
registered model — plus the chunked path matching the in-RAM path exactly for the
same seed).

`plot_loglog.py` — the shared log-log plotter, reading a run directory's own
`metadata.json` to look up its model in `tools/models.py`'s registry. The raw-data
plot (`<run_dir>/plot.png`) always overlays the all-points OLS fit (solid line,
$\hat\gamma$ in the legend) — needs no known ground truth, it's computed from the
data itself — and additionally overlays the known $\mathbb{E} Y_i$ curve (dashed)
when the model has a `target_fn` (currently only `"synthetic"`).
`tools/loglog.py`'s four-estimator comparison (`compare_methods`) always runs and
is always written to `<run_dir>/results.json`, for every model — comparing
estimators against each other doesn't need a known truth, only comparing against
one does; when `true_gamma` is unknown, a printed note makes clear the numbers are
exploratory, not a validated checkpoint result (see
`experiments/01_srw/README.md`). The four-estimator comparison *chart*
(`estimates.png`) is opt-in via `--estimates`, since unlike `results.json` it's a
supplementary figure (ground rule 1), not the numeric result itself. Everything
about one run lives in the same folder as `samples.npz`/`metadata.json`, and since
`data/` is gitignored, nothing here is auto-committed — copy a specific plot into
the experiment's `images/` folder when you want to keep it as evidence (ground
rule 1/6 — committed deliberately, one at a time).

`estimate_omega1.py` — Experiment B's analysis driver: reads a run directory and
applies both of `tools/correction.py`'s $\omega_1$ estimators to it, writing
`<run_dir>/omega1.json`. Never draws samples (generation and analysis are always
separate scripts here). `--expect-omega1`/`--expect-gamma` print a PASS/FAIL against
known values for reporting only — those values are never passed to the estimators, so
the measurement stays blind to the answer it is checking. See
`experiments/01_srw/README.md` for the recipe, the allocation rule, and the result.

`allocation_experiment.py` — Experiment C's driver: sweeps a (budget x $m_0$) grid,
drawing $R$ independent replicates per cell (`SeedSequence.spawn`, ground rule 2) and
estimating $\gamma$ with the article's own closed-form $w_{k,m}$ weights, to test
Proposition `prop:opt`. Reports per-cell bias/sd/RMSE, the empirically best $m_0$ against
the one the theorem names, and the measured error-decay exponent against
$-\omega_1/(d+2\omega_1)$. Does **not** persist samples -- the sweep draws far more than
is worth storing, and the per-cell $\hat\gamma$ values in `<run_dir>/result.json` are the
result (the base seed is recorded and spawning is deterministic, so any cell regenerates
exactly). `true_gamma` from the recipe scores finished estimates only and never reaches an
estimator. Verified: `tools/tests/test_allocation_experiment.py` (10 cases). Result and
interpretation in `experiments/01_srw/README.md`.

`allocation_table.py` — the budget-planning table: given the measured constants, how
long must a run take for a target precision on $\hat\gamma$? One row per integer $m_0$,
at the budget where that $m_0$ is optimal, showing $n$, wall-clock cost and predicted
RMSE. Reads its inputs from this repo's own results rather than hardcoding them ($a_1$
from Experiment B's `omega1.json`, the coefficient of variation from that run's samples,
throughput from Experiment C's wall clock), each overridable by flag and each falling
back to the last measured value if the run is absent. `--compare` adds what `prop:opt`'s
uncalibrated $m_0$ would have cost; `--csv` saves the table. **`--by-budget` indexes rows
by budget instead of by $m_0$, and is the view to use when comparing two runs of this
script** -- the default view asks "at what budget is this $m_0$ optimal", so any change in
the measured constants slides every row along the budget axis (see
`experiments/01_srw/README.md`, "I ran it twice and got two different tables"). A run
directory may be passed directly to `--data-root`, and any input that falls back to a
hardcoded constant prints a loud warning rather than quietly pretending to be a
measurement. Replicates are **pooled and refitted once**, not averaged fit-by-fit: the fit is nonlinear, so averaging separate fits converges to $\mathbb{E}[\hat a_1]$ rather than $a_1$, a bias no number of replicates removes (see `measured_a1`'s docstring for the measured numbers). Pooling uses the `y_bar`/`n`/`sigma_log` stored in `omega1.json`, so the samples need not be kept. Verified: `tools/tests/test_allocation_table.py` (20 cases).

`verify_prediction.py` — runs the tuned ladders for real and compares wall clock and
RMSE against what `allocation_table.py` predicted. Measured 0.94x-1.00x on timing across
four orders of magnitude, and 1.06x median on RMSE (noisy at small `--replicates`, since
an RMSE over $R$ draws carries $1/\sqrt{2R}$ relative sd itself). Skips ladders needing
more than `--max-n` samples per scale: budget is derived *from* $m_0$ here, so $n$ grows
like $\rho^{2m_0}$ and a stray `--m0 20` would otherwise ask for $10^{25}$ samples.
Verified: `tools/tests/test_verify_prediction.py` (7 cases).

`plot_allocation.py` — Experiment C's plot, `<run_dir>/plot.png`. Left panel: RMSE
against $m_0$, one curve per budget, with `prop:opt`'s $m_0$ and the empirical argmin
marked — the U-shape *is* the bias-variance tradeoff, and the systematic gap between the
two markers is the POINT failure made visible. Right panel: RMSE against budget on
log-log with fitted decay exponents, compared against $-\omega_1/(d+2\omega_1)$
**predicted from Experiment B's measured $\omega_1$**, banded by that measurement's own
standard error — the cross-experiment check that B's $\omega_1$ predicts C's decay rate.
Budgets whose `prop:opt` $m_0$ was not swept are dropped from the right panel and named
on stderr, rather than silently leaving it empty. Palette is `tools/loglog_plot.py`'s
unchanged. Verified: `tools/tests/test_plot_allocation.py` (5 cases — the chart itself is
not unit-tested, per the convention for plots; the series selection and fitted exponents
are).

`measure_cost.py`/`plot_cost.py` — the shared cost-model-exponent probe and its
plot, same registry dispatch as `generate.py`/`plot_loglog.py` (times
`MODELS[model].simulate(i, n=1, ...)` instead of drawing real samples). Only
meaningful for models whose cost genuinely grows with scale (e.g. `"srw"`); pointed
at `"synthetic"` it will just measure $d\approx0$, an expected, uninteresting
result. Repeated timings at each scale are collapsed by the aggregator named in the
recipe's optional `"aggregator"` key (`tools/cost_model.py`'s `AGGREGATORS`),
defaulting to `median`. Two cost models are fitted and both reported: the pure
$\mathrm{cost}(i)=c\,i^d$ of Assumption `cost_is_power_law`, and
$\mathrm{cost}(i)=a+b\,i^d$, which additionally accounts for the fixed per-call
dispatch overhead a single `simulate(k, n=1, ...)` pays. **Prefer the affine one
whenever the printed overhead share at the smallest scale is non-trivial** — the
script says so explicitly when it exceeds 20%, and the acceptance check uses the
affine $\hat d$. `plot_cost.py` writes `<run_dir>/plot.png` next to `result.json`,
same one-run-one-folder convention as `plot_loglog.py`. Verified:
`tools/tests/test_measure_cost.py` (real-timing acceptance check, affine $\hat
d\in[0.8,1.2]$ for `srw`; that the overhead is detected as strictly positive and is
what biases the pure fit low; and that the aggregator is selectable and recorded).

Each of these nine inserts `tools/` onto `sys.path` itself (`sys.path.insert(0,
str(Path(__file__).resolve().parent.parent / "tools"))`) so their bare imports of
`tools/*.py` helper modules work regardless of where they're invoked from.
