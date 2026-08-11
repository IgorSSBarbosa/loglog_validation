# tools

Shared, experiment-agnostic utilities. May not import from `experiments/`. Every
function here needs a passing unit test in `tools/tests/` (checked against a closed
form, not just "runs") before any experiment is allowed to depend on it.

`loglog_plot.py` — generic log-log plot of $\overline Y_i$ vs $i$ (any `{scale:
samples}` dict, from any experiment) with $\pm 1$ SE error bars and an optional
overlay of a known $\mathbb{E} Y_i$ curve. No unit test yet (it's a plot, not a
numeric claim) — visually verified against `experiments/00_synthetic/`.

`loglog.py` — a few $\hat\gamma$ estimators, all as the general OLS slope of
$\log\overline Y_i$ vs $\log i$ (equivalent to the article's closed-form
$w_{k,m}$ weights, eq. 523-526, on a consecutive $\rho^k$ grid, but not tied to
one): `gamma_all_points` (every scale), `gamma_two_point` (one estimate per
adjacent pair, $m=2$), `gamma_drop_leading` (one estimate per $m_0$, dropping
the first $m_0$ scales). `compare_methods` bundles all three into one
JSON-serializable dict. Verified against a noiseless planted power law
(exact recovery, float precision) and against unsorted input. **Not yet
implemented**: the article's exact closed-form $w_{k,m}$ weights and their
algebraic identities (eq. 542) — that's checkpoint 0.2's specific acceptance
criterion, still open. Also not yet implemented: a Hill-estimator-style
method — flagged to the user, since the classical Hill estimator targets a
different problem (tail index of one heavy-tailed sample) than ours
(cross-scale scaling exponent from many low-$n$ per-scale samples); needs
discussion before guessing a formula.

Not started yet — planned modules (see `PLAN.md` repo layout): `wilson.py`,
`allocation.py`, `bootstrap.py`, `rng.py`, `io.py`.
