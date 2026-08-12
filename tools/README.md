# tools

Shared, experiment-agnostic utilities. May not import from `experiments/`. Every
function here needs a passing unit test in `tools/tests/` (checked against a closed
form, not just "runs") before any experiment is allowed to depend on it.
`tools/tests/` is gitignored (local verification only, not tracked in git — user
request, 2026-08-12): the files are kept on disk and run normally via
`python3 -m pytest`, but a fresh checkout or `EnterWorktree` won't have them.

This directory also holds the single, shared CLI drivers (`generate.py`,
`plot_loglog.py`, `measure_cost.py`, `plot_cost.py`) — one copy each, used by every
experiment, instead of each experiment keeping its own. What's model-specific lives
in `tools/model_<name>.py` + one entry in `tools/models.py`'s registry, not in the
drivers themselves.

`loglog_plot.py` — two charts. `loglog_plot`: generic log-log plot of
$\overline Y_i$ vs $i$ (any `{scale: samples}` dict, from any experiment) with
$\pm 1$ SE error bars and an optional overlay of a known $\mathbb{E} Y_i$
curve. `estimates_plot`: compares the four $\hat\gamma$ estimators from
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

`compare_methods` bundles all four into one JSON-serializable dict (now takes
`n`, the per-scale sample counts, in addition to `scales`/`y_bar`, since
`gamma_mle` needs it). Not wired into `compare_methods` (which is deliberately
grid-agnostic): the article's exact closed-form $w_{k,m}$ weights,
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
(`save_samples`/`load_samples`, `{scale: array}` compressed) and `metadata.json`
(`write_metadata`/`load_metadata` — now also records `model`, the registry name
used, alongside `params`/`scales`/`n`/`seed`/`timing_seconds`), plus `content_id`
for deterministic hash-based tags and `normalize_scales_n` for scalar-or-sequence
`n`. Nested by tag deliberately (not `<out_dir>/<tag>.npz` + `<out_dir>/<tag>.json`
flat files) so `data/` stays navigable once there are dozens of runs. Named
`persistence.py`, not `io.py` — that name would shadow the stdlib `io` module once
`tools/` is on `sys.path`, as every file in `tools/tests/` already puts it.
Verified: `tools/tests/test_persistence.py` (10 cases — save/load roundtrip,
content-hash determinism, missing-file handling, the one-run-one-folder shape).

`models.py` — the `MODELS` registry (`ModelSpec`: `simulate(i, n, params, rng)`,
optional `target_fn(i, params)` and `true_gamma_key`) that `generate.py`,
`measure_cost.py`, and `plot_loglog.py` dispatch through via a run's `"model"`
name, instead of each experiment hardcoding its own simulator. Each entry's actual
logic lives in its own `tools/model_<name>.py`:
- `model_synthetic.py` — the closed-form model (`SyntheticParams`, `NOISE_FAMILIES`,
  `mean_Y`, article eq. 232). Has both `target_fn` and `true_gamma_key="gamma"`,
  since the ground truth is planted and known — the only model that currently does.
- `model_srw.py` — `srw(k, n, q, rng)`, $n$ i.i.d. realizations of $|S_k|$
  (vectorized $(n,k)$ step matrix). No `target_fn`/`true_gamma_key`: no
  article-sanctioned closed form for SRW yet (see `experiments/01_srw/README.md`),
  which is exactly what keeps `plot_loglog.py` from running gamma-hat estimators
  against it — not a special case in the driver, just an absence in the registry.

Verified: `tools/tests/test_models.py` (registry shape, unknown-name error),
`tools/tests/test_model_srw.py` (shape/bounds/parity, classical $\mathbb E|S_k|
\sim\sqrt{2k/\pi}$ asymptotic).

`generate.py` — the shared sample-generator CLI/API (`generate`, `reproduce`).
Recipe: `{"model": ..., "params": {...}, "scales": [...], "n": ..., "seed": null}`.
Default output directory is `data/` next to the recipe file itself (not the
script's own location, since this script is no longer inside any one experiment
folder), so each experiment's runs still land under that experiment's own `data/`.
Verified: `tools/tests/test_generate.py` (shapes match the recipe, `reproduce()`
exact-match, both parametrized across every registered model).

`plot_loglog.py` — the shared log-log plotter, reading a run directory's own
`metadata.json` to look up its model in the registry. Only overlays a reference
curve / runs `loglog.py`'s four $\hat\gamma$ estimators when that model has a
`target_fn` (currently only `"synthetic"`) — computing a $\hat\gamma$ with nothing
known to validate it against would be an unvalidated number, easy to mistake for a
checked result (see `experiments/01_srw/README.md`). Default image path is
`images/<tag>.png` next to the run's experiment folder (sibling to its `data/`).

`measure_cost.py`/`plot_cost.py` — the shared cost-model-exponent probe and its
plot, same registry dispatch as `generate.py`/`plot_loglog.py` (times
`MODELS[model].simulate(i, n=1, ...)` instead of drawing real samples). Only
meaningful for models whose cost genuinely grows with scale (e.g. `"srw"`);
pointed at `"synthetic"` it will just measure $d\approx0$, an expected,
uninteresting result. Verified: `tools/tests/test_measure_cost.py` (real-timing
acceptance check, $\hat d\in[0.8,1.2]$ for `srw`).

`cost_model.py` — `estimate_cost_exponent`, recovering the cost-model exponent
$d$ from Assumption `cost_is_power_law` ($\mathrm{cost}(i)=i^d$), which has the
same log-log-linear form as $\mathbb{E} Y_i=a_0 i^\gamma$ (eq. 232) — reuses
`loglog.py`'s `gamma_all_points` internally, behind a name-keyed registry
(`COST_ESTIMATORS`) so a different estimation approach can be swapped in later.
Verified: `tools/tests/test_cost_model.py` on synthetic noiseless cost curves;
empirically validated against a real $\Theta(k)$ simulator in
`experiments/01_srw/` (recovers $\hat d\approx0.90$–$1.09$ against ground truth
$d=1$).

`allocation.py` — `optimal_allocation` (Proposition `prop:opt`, eq. 945-946)
and `total_cost` (Lemma `lem:budget`'s closed-form cost). Given a budget $B$
and $(d,\omega_1,\rho,m)$, returns the rate-optimal exponents
$\theta_1,\theta_2$ and the sample count/scale-offset $n$, $m_0$ an experiment
can actually run. The theorem treats $n$, $m_0$ as continuous; this module
floors both to integers, which is provably safe (cost stays $\le B$) whenever
the continuous $n_{\mathrm{exact}}\ge1$ — when it isn't (budget too small for
this configuration), `n`/`m0`/`cost` come back `None` and `integer_feasible`
is `False` rather than silently overspending, the same "diagnostic the caller
must check" pattern `gamma_mle`'s `trustworthy` uses. Verified:
`tools/tests/test_allocation.py` (15 cases — $\theta_1+d\theta_2=1$ exactly,
continuous allocation costs exactly $B$, discretized allocation never exceeds
$B$ when feasible, the small-$B$ infeasibility case, parameter validation).

Not started yet — planned modules (see `PLAN.md` repo layout): `wilson.py`,
`bootstrap.py`, `rng.py`.
