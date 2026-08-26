# tools

Helper functions: code meant to be *called by other code*, not run directly (user's
own framing, 2026-08-12) — `src/` holds the scripts a human runs
(`python3 src/generate/generate.py ...`); everything here is imported by those scripts, or by
each other, or by tests. May not import from `experiments/`. Every function here
needs a passing unit test in `tools/tests/` (checked against a closed form, not just
"runs") before any experiment is allowed to depend on it. `tools/tests/` is
gitignored (local verification only, not tracked in git — user request,
2026-08-12): the files are kept on disk and run normally via `python3 -m pytest`, but
a fresh checkout or `EnterWorktree` won't have them.

Model-specific simulation code lives in `models/<name>.py` (a sibling of `tools/`,
not a subfolder of it — see `models/README.md`) + one entry in `tools/models.py`'s
registry; `tools/models.py` itself is purely an importer/registry module, not where
the simulation logic lives.

Thirteen modules. The dependency graph is shallow on purpose — `loglog.py` is the only
one several others import, because it owns the article's weight definition:

| module | what it owns | imports |
|---|---|---|
| `loglog.py` | four $\hat\gamma$ estimators + eq. (526) weights — **the canonical definition** | — |
| `correction.py` | two $\omega_1$ estimators (direct fit of eq. 232, and bias decay) | — |
| `allocation.py` | `prop:opt` (eq. 945–946), `lem:budget` costs, the tuned constant $\kappa$, `snr`/`neyman`, the `ladder` itself and its decay rate | `loglog` |
| `cost_model.py` | cost exponent $d$: the two timing probes, pure + affine fits, aggregators, declared-vs-measured | `loglog` |
| `wilson.py` | eq. (720)'s four-term bound, **for $\gamma$ only** | `loglog` |
| `coverage.py` | do our stated error bars actually cover? | — |
| `artifacts.py` | what every file on disk is called, in or out | — |
| `rng.py` | seeding, and how a seed is recorded so a run regenerates | — |
| `constants.py` | meta-constants with their errors and provenance; no fallbacks | — |
| `persistence.py` | run directories, samples, metadata, content hashing | — |
| `models.py` | the `MODELS` registry — a pure importer | `models/` |
| `summary.py` | what a replicate *is* once its draws are gone: $\overline Y_i$, its log-scale se, and the cv | — |
| `loglog_plot.py` | the two charts | — |

Full detail, one section per module, below.

---

`loglog_plot.py` — two charts. `loglog_plot`: generic log-log plot of
$\overline Y_i$ vs $i$ (any `{scale: samples}` dict, from any experiment) with
$\pm 1$ SE error bars, an optional overlay of a known $\mathbb{E} Y_i$ curve
(`target_fn`, dashed gray, only when ground truth is known), and an optional
overlay of an arbitrary fitted curve (`fit_fn`, solid colored -- e.g. the
all-points OLS fit, which needs no known ground truth, just the data itself;
`src/report/plot_loglog.py` always passes one). `estimates_plot`: compares the four $\hat\gamma$ estimators from
`loglog.py`'s `compare_methods` — `two_point`/`drop_leading` (each a sequence
of estimates) plotted against the smallest scale in their window, so the
chart shows convergence as small, more finite-size-biased scales are dropped;
`all_points`/`mle` (single estimates, no window) as horizontal reference
lines; `true_gamma` (if known) as a muted dashed reference. Colors follow the
dataviz skill's validated default palette — categorical slots 1-3, `mle`'s
untrustworthy state uses the reserved status-critical color instead of a 4th
competing hue (the palette's series-count ladder caps an all-pairs-visible
chart at three). Neither chart has a unit test yet (they're plots, not
numeric claims) — visually verified against `experiments/00_synthetic/`,
including the untrustworthy-MLE styling path.

`wilson.py` — the article's Wilson confidence interval, Theorem `thm:wilson`
(eq. 720): the four-term bound $\mathcal B_{\mathrm{fs}}+\mathcal B_{\mathrm{good}}
+\mathcal B_{\mathrm{bad}}+\Phi(\alpha)\sigma_{\mathrm{se}}$ on
$\lvert\hat\beta-\beta\rvert$. **For $\gamma$ only** — that is the theorem's own
scope: it describes $\hat\beta=\sum_k w_{k,m}\log\overline Y_{\rho^k}$ and says
nothing about $\omega_1$ or $a_1$, so do not reach for it there. Its value over a
replicate interval is the fourth term: $\sigma_{\mathrm{se}}=\sqrt{12\sigma_\infty^2/
(nm^3)}$ is a closed form, estimable from the raw samples, so no 5-point variance
estimate is involved and $\Phi(\alpha)=1.960$ is honest where replicates force
$t_4=2.776$. It is a **bound**, so it overcovers — measured $0.966$–$1.000$ across
$m_0=2..10$, against a replicate interval that collapses to $0.000$ at $m_0=2$ for want
of any bias term. `moment_bounds` reads $\sigma_\infty^2$/$\sigma_{\max}^2$/$\Lambda$
off real samples; `sigma_se_per_scale` handles the non-uniform $n$ eq. (720) does not
cover. Terms whose constants are unmeasured are dropped only behind a loud
`complete=False` — a bound missing a term is not a bound. Driven by
`calibration/check_coverage.py --arm wilson`.

`coverage.py` — calibration testing for error bars (PLAN.md checkpoint 0.4).
Answers a question none of the other tools do: this repo reports every result
as *estimate ± uncertainty*, and while the estimates are checked against known
truth, the **uncertainties are themselves untested claims**. `coverage_test`
replays a whole pipeline many times against a known truth and counts how often
its stated interval actually contains it; `coverage_multi` does the same but
scores many quantities from one pass, which matters because $\omega_1$ and
$a_1$ come out of a single `least_squares` call — re-running the trials per
quantity would both multiply the cost and score each on different draws, so
two results could not be compared. `interval(estimate, se, dof=)` builds the
interval, with `dof=None` reproducing the naive normal quantile on purpose so
a test can measure what it costs. `wilson_score_interval` puts an honest CI on
a measured coverage (score, not Wald: a calibrated 95% test measures coverage
near 1, exactly where Wald runs past 1). `se_ratio` separates the two causes of
undercoverage — an interval too narrow vs. an estimator off-centre — because
the fix differs. **Not article eq. (720)**: the article's Wilson interval is a
four-term bound on $\hat\gamma$ and lives in `tools/wilson.py` (above); the one
here is the textbook binomial score interval, used only for counting
proportions. The distinction is the whole reason both exist -- `coverage.py`
*scores* intervals, `wilson.py` *builds* one. Driven by `calibration/check_coverage.py`.

`loglog.py` — four $\hat\gamma$ estimators. Three as the general OLS slope of
$\log\overline Y_i$ vs $\log i$ (equivalent to the article's closed-form
$w_{k,m}$ weights, eq. 523-526, on a consecutive $\rho^k$ grid, but not tied to
one): `gamma_all_points` (every scale), `gamma_two_point` (one estimate per
adjacent pair, $m=2$), `gamma_drop_leading` (one estimate per $m_0$, dropping
the first $m_0$ scales). Verified against a noiseless planted power law
(exact recovery, float precision) and against unsorted input.

The 4th, `gamma_mle`, is a genuinely different estimator (not OLS on
logs): the MLE under $\overline Y_k \sim \mathcal N(\mu_k, \sigma^2\mu_k^2/n_k)$
(CLT on the sample mean itself, not its log) — full derivation, including a
second-order (concavity) analysis, in `derivations/mle_gamma_estimator.tex`.
Implemented as direct joint optimization over $(\gamma,\log a_0,\log\sigma^2)$
from the OLS-on-log starting point; **its result dict must be checked for
`trustworthy` before using `gamma_hat`** — the joint likelihood is not
established to be globally concave (see the derivation), so this is a real
diagnostic, not decoration. Validated: 200-replicate Monte Carlo matches the
derivation's numbers exactly (`trustworthy=False` in 5/200 trials, always via
the optimizer's own non-convergence rather than a bad estimate; bias/RMSE
matching `gamma_all_points` in an equal-$n_k$ design, as the derivation
predicts). Known edge case: exactly-zero-residual data (e.g. a truly
noiseless synthetic run) makes $\hat\sigma^2\to0$ and is numerically
degenerate — not a bug, a structural feature of this particular MLE (the OLS
methods don't have it, since they never divide by an estimated variance).

`compare_methods` bundles all four into one JSON-serializable dict (takes `n`,
the per-scale sample counts, in addition to `scales`/`y_bar`, since
`gamma_mle` needs it; `true_gamma` is genuinely optional -- comparing
estimators against each other doesn't require a known ground truth, only
comparing against one does). `all_points` also carries `a0_hat` (the OLS
fit's intercept, exp'd) alongside `gamma_hat` -- needed to actually draw the
fitted line (`src/report/plot_loglog.py`'s `fit_fn`), not just report its slope.
`src/report/plot_loglog.py` runs this, and writes its output to `gamma_estimates.json`, for
every model unconditionally, not just ones with a known closed form -- see
that script's module docstring for why. Not wired into `compare_methods`
(which is deliberately grid-agnostic): the article's exact closed-form $w_{k,m}$ weights,
`closed_form_weights`/`gamma_closed_form` — checkpoint 0.2's acceptance
criterion. Unlike the other four, this one requires `scales` to be exactly the
consecutive grid $\rho^k$, $k=m_0+1,\dots,m_0+m$ (raises otherwise), so it
isn't a drop-in fifth method for arbitrary scale sets. `tools/tests/test_loglog.py`
(first test file in the repo) checks all five weight identities (a)-(e) of
Lemma "Elementary identities" for a spread of $(m,m_0)$, exact recovery on
noiseless data, agreement with `gamma_all_points` on a consecutive grid, and
the rejection path.

`persistence.py` — sample+metadata save/load shared across every model. One run,
one `<out_dir>/<tag>/` directory (`run_dir`), holding `samples.npz`
(`save_samples`/`load_samples`, `{scale: array}` compressed) and `samples_meta.json`
(`write_metadata`/`load_metadata` — now also records `model`, the registry name
used, alongside `params`/`scales`/`n`/`seed`/`timing_seconds`), plus `content_id`
for deterministic hash-based tags and `normalize_scales_n` for scalar-or-sequence
`n`. Nested by tag deliberately (not `<out_dir>/<tag>.npz` + `<out_dir>/<tag>.json`
flat files) so `data/` stays navigable once there are dozens of runs. Named
`persistence.py`, not `io.py` — that name would shadow the stdlib `io` module once
`tools/` is on `sys.path`, as every file in `tools/tests/` already puts it.
Verified: `tools/tests/test_persistence.py` (10 cases — save/load roundtrip,
content-hash determinism, missing-file handling, the one-run-one-folder shape).

`open_scale_writer(run_dir, scale, n, dtype)` is the alternative for runs too large to
build entirely in RAM: returns a writable on-disk memmap at
`<run_dir>/samples/<scale>.npy`, filled in slices by the caller instead of assembled in
memory and saved all at once. `load_samples` reads either layout transparently (flat
`samples.npz` first, falling back to `samples/*.npy` loaded with `mmap_mode="r"`) — see
`src/generate/generate.py`'s chunked path below and `experiments/01_srw/README.md`'s
"Fixed-memory generation" section for why this exists (fixes the OOM on
`Huge_test.json`-sized runs). Verified: 2 more cases in `test_persistence.py`
(round-trip through `open_scale_writer`, and that a stray `samples/` dir never
shadows an existing flat `samples.npz`).

`models.py` — the `MODELS` registry (`ModelSpec`: `simulate(i, n, params, rng)`,
optional `target_fn(i, params)` and `true_gamma_key`) that `src/generate/generate.py`,
`src/estimate/measure_cost.py`, and `src/report/plot_loglog.py` dispatch through via a run's
`"model"` name, instead of each experiment hardcoding its own simulator. This file
is purely an importer: it adds the sibling `models/` directory to `sys.path` and
imports each `models/<name>.py` as a bare top-level name (`srw`, `synthetic`) --
deliberately never through the literal name `models`, since this file is itself
`tools/models.py` and would otherwise self-referentially collide with its own
module identity (see its docstring). The actual per-model logic lives in
`models/<name>.py`, one level up — see `models/README.md`:
- `synthetic.py` — the closed-form model (`SyntheticParams`, `NOISE_FAMILIES`,
  `mean_Y`, article eq. 232). Has both `target_fn` and `true_gamma_key="gamma"`,
  since the ground truth is planted and known — the only model that currently does.
- `srw.py` — `srw(k, n, q, rng, block_n=None)`, $n$ i.i.d. realizations of $|S_k|$.
  Draws steps in `(block_n, k)` float32 blocks over the $n$ axis and accumulates
  row sums, rather than one $(n,k)$ matrix — bounds peak transient memory to a fixed
  byte budget regardless of how large $n$ gets (the old unblocked, `int64` version
  needed 819 GiB at $n=10^8,\,k=1024$). Blocking over $n$, not $k$, is deliberate:
  splitting the leading axis into sequential row ranges consumes numpy's row-major RNG
  stream in the same order a single unblocked call would, so results are bit-identical
  for the same seed at any block size (splitting over $k$ would not have this
  property — see the module's own docstring). The draw is
  `rng.random(dtype=float32) < q`, ~4.4x faster than the `rng.choice(..., p=)` it
  replaced; note the still-faster `rng.integers(0,2,dtype=int8)` was tried and
  **rejected** because numpy's bit-packing discards leftover bits per call and breaks
  exactly that block-invariance, and `rng.binomial` was rejected for sampling every
  scale in ~constant time (which would destroy the $\Theta(k)$ cost this model exists
  to provide). No `target_fn`/`true_gamma_key` — a deliberate choice, not a gap:
  $\mathbb{E}\lvert S_k\rvert$ *is* known exactly, but the user's decision
  (2026-08-20) is to keep $\gamma=1/2$, $\omega_1=1$ out of the code path and state
  them as README acceptance criteria instead, so the estimators are never handed the
  answer they are measuring (see `experiments/01_srw/README.md`). That absence is
  what keeps `src/report/plot_loglog.py` from overlaying a reference curve or reporting a
  `true_gamma` for this model — not a special case in the driver. The gamma-hat
  estimators still run (comparing estimators against each other doesn't need a known
  truth); an explicit "exploratory" note is printed instead.

Verified: `tools/tests/test_models.py` (registry shape, unknown-name error),
`tools/tests/test_srw.py` (shape/bounds/parity, classical $\mathbb E|S_k|
\sim\sqrt{2k/\pi}$ asymptotic, `block_n` exact-equivalence with the unblocked
path, and a large-$(n,k)$ case that would be gigabytes unblocked). The four
scripts that dispatch through this
registry (`generate.py`, `plot_loglog.py`, `measure_cost.py`, `plot_cost.py`) live
in `src/`, not here — see `src/README.md`.

`cost_model.py` — `estimate_cost_exponent`, recovering the cost-model exponent
$d$ from Assumption `cost_is_power_law` ($\mathrm{cost}(i)=i^d$), which has the
same log-log-linear form as $\mathbb{E} Y_i=a_0 i^\gamma$ (eq. 232) — reuses
`loglog.py`'s `gamma_all_points` internally, behind a name-keyed registry
(`COST_ESTIMATORS`) so a different estimation approach can be swapped in later.

`estimate_cost_affine` fits $\mathrm{cost}(i)=a+b\,i^d$ instead, and is the one
to trust whenever the fitted overhead $a$ is not small next to the timings at
the smallest scales. A real timing probe pays a fixed per-call cost that does
not scale with $i$ at all (~11–24 µs of Python/NumPy dispatch on this machine);
the pure power law has nowhere to put it and folds it into $d$, biasing the
estimate downward — badly, when small scales are included ($\hat d=0.10$ on
`time_measure`, where the truth is $1$; the affine fit recovers $0.95$). Fitted
in log space, since costs span orders of magnitude across a scale grid.

The **timing** half lives here too, because there turned out to be exactly two ways
this repo measures $\mathrm{cost}(i)$, differing only in *which* scales get timed.
`time_over_scales` times a ladder you name — `src/estimate/measure_cost.py`'s
strategy, where the ladder is the experiment. `climb_to_target` doubles the scale
until one call takes `PROBE_TARGET_SECONDS` — `src/study/pilot.py`'s, where the
ladder is not free: a pilot's scales are chosen so the *correction* term is visible,
which means small, and timing srw's own $8..256$ returned $d=8.0\pm280$. $d$ is a
property of the model, not of the window, so measuring it further out costs nothing.
Both return the same payload and both feed `fit_cost_probe`, which does the two fits,
the `overhead_share` diagnostic and the declared-vs-measured cross-check — written
once instead of once per caller. `time_at_scale` throws away a warm-up call before
timing anything, and times calls individually rather than as a batch divided by
`repeats`, so the aggregator can do its job.

`AGGREGATORS`/`aggregate`/`median_ci` collapse repeated timings at one scale.
Repeated timings target the same deterministic quantity, so their noise is
one-sided (jitter only ever adds delay) — which is why `min` is the classic
microbenchmark choice and was this project's original one. The registry
(`min`/`median`/`mean`/`q95`/`iqmean`, selectable per-recipe) defaults to
**`median`**: equally resistant to one-sided jitter, but unlike a minimum it
has a distribution-free confidence interval from the order statistics
(`median_ci`), so cost curves get honest error bars. The choice barely moves
the estimate (min 0.884 / median 0.888 / iqmean 0.887 / mean 0.850 / q95 0.811
on `cost_probe`) — it buys the interval, not a different number.

Verified: `tools/tests/test_cost_model.py` — the two probes' shared payload shape,
that the climb doubles, stops at its target, never stops below the affine fit's
four-point minimum, and honours its wall-clock budget rather than hanging; then
exact recovery of $d$ on
noiseless power-law curves; exact recovery of *all three* of $(a,b,d)$ on
planted affine curves, alongside a check that the pure fit is genuinely biased
low on the same data; aggregator behaviour on a sample with a planted outlier;
and `median_ci`'s nominal coverage measured empirically (95.6% against a
nominal 95%) plus its $1/\sqrt N$ width shrinkage — not a single-draw bracket
assertion, which would fail 5% of the time by construction. Empirically
validated against a real $\Theta(k)$ simulator in `experiments/01_srw/`
(affine $\hat d=1.006$ against ground truth $d=1$).

`allocation.py` — four allocation rules; they answer different questions and
must not be swapped. `optimal_allocation` (Proposition `prop:opt`, eq. 945-946)
and `total_cost` (Lemma `lem:budget`'s closed-form cost) are the article's.

`tuned_allocation` is `prop:opt` with its dropped multiplicative constant put
back. `prop:opt` is a *rate* result -- $m_0=\theta_2\log_\rho B$ is right up
to an additive constant in $m_0$, which the rate argument discards and which
Experiment C measured to be worth a factor $2.2$-$2.4$ in RMSE. Writing
$\lvert\mathrm{bias}\rvert=C_b\rho^{-m_0\omega_1}$ and
$\mathrm{sd}=C_s n^{-1/2}$ and minimizing, the constant is
$\theta_2\log_\rho\kappa$ with $\kappa=2\omega_1C_b^2/(d\,C_s^2G)$;
`allocation_constants` computes $C_b$, $C_s$, $G$ and $\kappa$ in closed form
from the article's weights plus two *measured* inputs -- $a_1$ (Experiment B)
and the observable's coefficient of variation -- so it is a calibration rather
than extra theory. `predict_error` gives the bias/sd/RMSE of any $(n,m_0)$.
Validated against Experiment C: predicted RMSE within 4-16% of measured at
every cell, and the optimum's $\lvert\mathrm{bias}\rvert/\mathrm{sd}=
\sqrt{d/(2\omega_1)}=0.707$ against measured 0.60-0.77. Note it **rounds**
$m_0$ where `optimal_allocation` floors: $n$ is refloored from whichever
integer is chosen, so the budget bound holds either way, and rounding lands
nearer the continuous optimum. Given a budget $B$
and $(d,\omega_1,\rho,m)$, returns the rate-optimal exponents
$\theta_1,\theta_2$ and the sample count/scale-offset $n$, $m_0$ an experiment
can actually run. The theorem treats $n$, $m_0$ as continuous; this module
floors both to integers, which is provably safe (cost stays $\le B$) whenever
the continuous $n_{\mathrm{exact}}\ge1$ — when it isn't (budget too small for
this configuration), `n`/`m0`/`cost` come back `None` and `integer_feasible`
is `False` rather than silently overspending, the same "diagnostic the caller
must check" pattern `gamma_mle`'s `trustworthy` uses. The other two are for Experiment B, which measures $\omega_1$ rather than
$\gamma$ and therefore needs the *opposite* treatment of the small scales --
`prop:opt` slides its window upward to escape the correction term, while
Experiment B has to keep the scales where that term is still visible.
`neyman_allocation` ($n_i\propto i^{-d/2}$) minimizes the variance of
$\overline Y_i$ itself. **It is the wrong rule for $\omega_1$ and is kept only
as the documented wrong answer**: $\omega_1$ is estimated from the correction
$a_1i^{-\omega_1}$, whose size shrinks with $i$, so equalizing the error of
$\overline Y_i$ over-samples the small scales, where the correction is already
resolved hundreds of times over, and starves the large ones, where it has sunk
below the noise (measured SNR 460 at $k=2$ down to 0.25 at $k=1024$).
`snr_allocation` ($n_i\propto s_i^2 i^{2\omega_1}$, *increasing* in $i$)
equalizes the correction term's signal-to-noise ratio across scales instead,
and is what `experiments/01_srw/recipes/samples_omega1.json` uses. `ladder` builds Definition `def:alloc`'s scale set $\rho^k$, $k=m_0+1..m_0+m$, and
refuses a grid whose rounding has collided ($\rho=1.5$, $m_0=0$, $m=6$ rounds to
$[2,2,3,5,8,11]$ -- $m-1$ distinct points where every downstream formula assumes
$m$). `n_for_budget` inverts `total_cost`. `rate_exponent`/`rate_exponent_se`
measure eq. (941)/(966)'s decay exponent from any (budget, error) series, the second
deriving its error bar from the known $1/\sqrt{2R}$ noise in an RMSE rather than
from 3-4 fit residuals. These four lived in `src/budget/allocation_experiment.py`
until 2026-08-25, where `check_coverage` and `plot_allocation` had to reach sideways
into a driver to use them (CATALOG.md §5.1); they are generic allocation math with
no dependence on the sweep. Verified:
`tools/tests/test_allocation.py` (47 cases — $\theta_1+d\theta_2=1$ exactly,
continuous allocation costs exactly $B$, discretized allocation never exceeds
$B$ when feasible, the small-$B$ infeasibility case, both new rules' closed-form
weights, that `snr_allocation` really does flatten the correction SNR, that the
two new rules trend in *opposite* directions so they can never be silently
conflated, and parameter validation).

`correction.py` — the correction-to-scaling exponent $\omega_1$ itself, which
`allocation.py` needs as an input and which `loglog.py` treats purely as a
nuisance. Two deliberately different functionals, so agreement between them is
evidence rather than bookkeeping: `fit_correction` fits eq. (232)'s
one-correction truncation $\log\overline Y_i = \log a_0+\gamma\log i+a_1
i^{-\omega_1}$ with all four parameters free (optionally weighted by each
$\mathrm{sd}(\log\overline Y_i)$, which matters because Experiment B's
allocation deliberately makes $n_i$ differ across scales); `omega1_from_bias_decay`
never looks at $\overline Y_i$ at all, and instead fits how a *sequence* of
$\hat\gamma$ estimates (e.g. `gamma_drop_leading`'s, one per $m_0$) converges as
the most contaminated small scales are dropped. Both restart the optimizer from
several $\omega_1$ seeds — the objective is not convex in $\omega_1$ and a
single start does stop in local minima. **Caution, documented at length in the
module: the scale grid must not mix odd and even $k$ for a lattice observable.**
$\mathbb E|S_{2m-1}|=\mathbb E|S_{2m}|$ exactly, so the mean is a staircase, and
a $\rho=\sqrt2$ grid returns $\hat\omega_1\approx17.8$ on *exact* means — a
silent failure, since the fit converges with a small residual. Verified:
`tools/tests/test_correction.py` (21 cases — exact recovery of all four planted
parameters, a $(\gamma,\omega_1)$ grid of truths, agreement between the two
estimators on planted data, robustness to small multiplicative noise, and the
mixed-parity failure mode pinned down explicitly).

`cost_model.py` also carries the **budget-unit cross-check**: `declared_exponent`
reads $d$ off a model's own `cost_hint`, `compare_cost_models` sets it against the
affine fit of the wall clock, and `format_cost_comparison` prints both. Keeping both
is the point — the declared count is exact where the clock is not (throughput varied
8.6x across scales on this machine, making a time-based $\hat d$ 22.9% wrong globally
and 49.4% wrong at the small scales), but only the clock can notice that a simulator
has stopped being compute-bound. Agreement needs either $\lvert z\rvert\le3$ **or**
a relative gap $\le5\%$: srw's declared $d=1$ is exact by construction, and against a
measured $1.0028\pm0.0020$ the sigma test alone would flag a 0.3% discrepancy as a
disagreement and train you to ignore the warning. Driven by
`src/estimate/measure_cost.py`.

`artifacts.py` — the naming registry: one place that decides what a run directory's
files are called, and what a recipe is called. `result.json` used to be written by
three producers with three schemas and `results.json` by a fourth, so telling them
apart meant duck-typing a schema key or hardcoding a directory name — and two of the
three genuinely cannot be separated by schema (both carry `cells`). `ARTIFACTS` maps
kind to filename, `write_artifact` stamps provenance (`produced_by`, `recipe`,
`created`) *inside* the file, and `read_artifact` still reads the old names behind a
`DeprecationWarning`. `python3 tools/artifacts.py --migrate <root>` renames legacy
files in place; `--list` prints the table. `RECIPES` does the same job for inputs:
each recipe declares its `kind`, and `load_recipe(path, expect=...)` refuses a
mismatch with an error that names the mistake rather than the missing key.
`default_out_dir` resolves a driver's default output location to the *experiment's*
`data/` — the recipe's grandparent now that recipes live in `recipes/`.

`rng.py` — seeding, and the one way to record a seed. `as_seed_sequence` accepts
an int, a `SeedSequence`, `None`, or a recorded dict; `seed_record` turns any of
them back into something JSON-safe; `spawn` is ground rule 2's primitive. It exists
because of one sharp trap, which it pins as a test: **a spawned child carries its
parent's entropy**, not its own —

```
kid = SeedSequence(12345).spawn(4)[2]
kid.entropy == 12345        # the PARENT's
kid.spawn_key == (2,)       # the only distinguishing part
```

— so handing a child to an `int`-typed seed parameter as `kid.entropy`, the obvious
workaround, silently collapses every replicate onto one identical stream. That is
`presentation18-05-2026`'s pool-reuse bug in a new disguise: plausible numbers, no
error, and comparisons between configurations that are quietly invalid.
`SeedSequence(kid)` doesn't work either — it raises `TypeError`. Hence seeds travel
as `SeedSequence` objects and are recorded as `{"entropy": ..., "spawn_key": [...]}`
— except when `spawn_key` is empty, where the record stays a bare int so every
`samples_meta.json` already on disk still loads unchanged. Verified:
`tools/tests/test_rng.py` (9 cases, including the collapse itself, asserted in both
directions).

`constants.py` — every number that reaches an allocation decision ($d$, $\omega_1$,
$a_1$, cv, throughput) is an *estimate*, and this module refuses to let one travel
anonymously. A constant is **measured** (value, error where one exists, and a
provenance string naming the file), a **user override** (stamped, and printed
`<-- NOT MEASURED`), or **absent** — and absent is a `SystemExit` naming both the flag
that supplies it and the command that would measure it.

It exists because of a real defect. `allocation_table.py` used to define
`FALLBACK_D = 1.0` and `FALLBACK_CV = sqrt(pi/2 - 1)` — srw's **exact truths** — so on
srw the table printed the right answer whether or not anything had been measured. Worse,
`--d` and `--omega1` *defaulted* to `1.0` with no provenance marker at all, so every
table this repo published silently used $\omega_1=1$ while Experiment B's own runs
measured $0.907$, $0.986$, $1.198$, $0.486$. The agreement was partly the default.
`se=None` (one replicate, no spread) is deliberately distinct from absent, and
`format_table` renders the two differently. Verified: `tools/tests/test_constants.py`
(9 cases).

Not started yet — planned module (see `PLAN.md` repo layout): `bootstrap.py`.

---

`summary.py` — the three numbers a replicate is reduced to, defined once.

`summarize_scale(draws)` returns $(\overline Y_i,\ \sigma_{\log},\ \mathrm{cv})$ for
one scale in a single pass: the sample mean, its standard error *on the log scale*
$\mathrm{sd}/(\sqrt n\,\overline Y_i)$ — which is what the log-log fit weights by —
and $\mathrm{sd}/\overline Y_i$, which is what an allocation needs.
`replicate_summary(stats, scales)` transposes `{scale: triple}` into one list per
name, ordered by `scales` rather than by the mapping's key order.

Those three are the entire interface between drawing and fitting: nothing downstream
(`correction.fit_correction`, `report.analyse`, `plan.py`) ever looks at a raw draw.
That is what lets `run.py` pass `summarize_scale` straight to
`generate(..., reduce=)` and never write a sample to disk — a planned run is
routinely hundreds of MB per replicate.

The module exists because the definition was written twice, verbatim, in
`src/study/pilot.py` and `src/study/run.py`. Deliberately free of any dependency on
`src/`: a summary is a fact about an array, not about how the array was obtained.
Verified: `tools/tests/test_summary.py` (5 cases — the closed forms, that
$\sigma_{\log}$ falls like $1/\sqrt n$ while cv does not, plain-`float` output for
JSON, and scale ordering).
