# tools

Shared, experiment-agnostic utilities. May not import from `experiments/`. Every
function here needs a passing unit test in `tools/tests/` (checked against a closed
form, not just "runs") before any experiment is allowed to depend on it.

`loglog_plot.py` — generic log-log plot of $\overline Y_i$ vs $i$ (any `{scale:
samples}` dict, from any experiment) with $\pm 1$ SE error bars and an optional
overlay of a known $\mathbb{E} Y_i$ curve. No unit test yet (it's a plot, not a
numeric claim) — visually verified against `experiments/00_synthetic/`.

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
`gamma_mle` needs it). **Not yet implemented**: the article's exact
closed-form $w_{k,m}$ weights and their algebraic identities (eq. 542) —
that's checkpoint 0.2's specific acceptance criterion, still open.

Not started yet — planned modules (see `PLAN.md` repo layout): `wilson.py`,
`allocation.py`, `bootstrap.py`, `rng.py`, `io.py`.
