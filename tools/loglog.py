"""Critical-exponent estimators from a log-log plot.

Given per-scale sample means Y_bar_i (e.g. from `tools.loglog_plot.loglog_points`),
estimate gamma from E[Y_i] ~ a0 * i**gamma (article eq. 232's leading term):
taking logs, log(Y_bar_i) ~ log(a0) + gamma*log(i), so gamma is recoverable as an
ordinary-least-squares slope in (log i, log Y_bar_i) space -- for *any* set of
scales, not just a rho^k grid. This is mathematically the same estimator as the
article's closed-form weighted sum eq. (523)-(526) whenever the scales *are* a
consecutive rho^k grid (the weights w_{k,m} are exactly the OLS slope formula
specialized to equally-spaced integer k); the general form is used here because
not every experiment's scales are on such a grid (e.g. the current
`example_config.json` spans 2**10, 2**15, 2**20 -- not consecutive).

The exact closed-form weights (eq. 526), `closed_form_weights`/`gamma_closed_form`
below, are checkpoint 0.2's own acceptance criterion (see PLAN.md / TODO.md);
their algebraic identities (eq. 542, Lemma "Elementary identities") are checked
in `tests/test_loglog.py`, not asserted at runtime.

`gamma_mle` is a 4th, genuinely different estimator: the maximum-likelihood
estimate under Y_bar_k ~ N(mu_k, sigma^2 mu_k^2/n_k) (a CLT approximation of
the per-scale sample mean itself, not of its log). See
derivations/mle_gamma_estimator.tex for the full derivation, including a
second-order analysis showing the joint likelihood is not established to be
globally concave -- gamma_mle's result carries diagnostics that should be
checked, not just trusted.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def ols_slope(x, y) -> tuple[float, float]:
    """OLS fit y = slope*x + intercept. Returns (slope, intercept)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("need at least 2 points for a linear fit")
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(slope), float(intercept)


def _sorted_log(scales, y_bar) -> tuple[np.ndarray, np.ndarray]:
    scales = np.asarray(scales)
    y_bar = np.asarray(y_bar)
    order = np.argsort(scales)
    return np.log(scales[order].astype(np.float64)), np.log(y_bar[order].astype(np.float64))


def gamma_all_points(scales, y_bar) -> float:
    """Method 1: OLS slope of log(Y_bar_i) vs log(i), using every scale."""
    log_i, log_y = _sorted_log(scales, y_bar)
    slope, _ = ols_slope(log_i, log_y)
    return slope


def gamma_two_point(scales, y_bar) -> list[dict]:
    """Method 2: two-point slope between each pair of adjacent scales (sorted by i).

    One estimate per adjacent pair -- the m=2 case discussed in the article
    (Remark "Fixed window", eq. 888).
    """
    scales_sorted = np.sort(np.asarray(scales))
    log_i, log_y = _sorted_log(scales, y_bar)
    estimates = []
    for j in range(len(scales_sorted) - 1):
        gamma_hat = (log_y[j + 1] - log_y[j]) / (log_i[j + 1] - log_i[j])
        estimates.append(
            {
                "scales": [int(scales_sorted[j]), int(scales_sorted[j + 1])],
                "gamma_hat": float(gamma_hat),
            }
        )
    return estimates


def gamma_drop_leading(scales, y_bar) -> list[dict]:
    """Method 3: OLS slope over all remaining scales after dropping the first m0
    (sorted by i), for m0 = 0, 1, ..., len(scales)-2. One estimate per m0."""
    scales_sorted = np.sort(np.asarray(scales))
    log_i, log_y = _sorted_log(scales, y_bar)
    estimates = []
    for m0 in range(len(scales_sorted) - 1):
        gamma_hat, _ = ols_slope(log_i[m0:], log_y[m0:])
        estimates.append(
            {
                "m0": m0,
                "scales_used": [int(s) for s in scales_sorted[m0:]],
                "gamma_hat": float(gamma_hat),
            }
        )
    return estimates


def closed_form_weights(m: int, m0: int = 0) -> np.ndarray:
    """Regression weights w_{k,m}, article eq. (526), for k = m0+1, ..., m0+m.

    Shift-invariant in m0 (Lemma "Elementary identities" (a)): the value only
    depends on the local index j = k - m0 in {1, ..., m}, so `m0` doesn't
    change the returned array -- kept as a parameter purely to document which
    k the array indexes (weights[0] is w_{m0+1,m}).
    """
    j = np.arange(1, m + 1, dtype=np.float64)
    return 12.0 * (j - (m + 1) / 2.0) / (m * (m**2 - 1))


def gamma_closed_form(scales, y_bar, rho: float, m0: int = 0) -> float:
    """Closed-form beta_hat/gamma_hat via the article's own weights (eq. 523-526),
    rather than generic OLS -- an independent check that the two forms agree.

    `scales` must be exactly the consecutive rho^k grid k = m0+1, ..., m0+m
    (any order; sorted internally) -- that grid structure is what the closed
    form assumes. Raises ValueError otherwise; use `gamma_all_points` for
    scales off a single consecutive grid (e.g. a mix of rho^k for several rho).
    """
    log_i, log_y = _sorted_log(scales, y_bar)
    m = len(log_i)
    k = np.rint(log_i / np.log(rho)).astype(np.int64)
    expected_k = np.arange(m0 + 1, m0 + m + 1)
    if not np.array_equal(k, expected_k):
        raise ValueError(
            f"scales must be the consecutive grid rho**k for k={m0 + 1}..{m0 + m} (rho={rho}); "
            f"got k={k.tolist()}"
        )
    w = closed_form_weights(m, m0)
    beta_hat = float(np.dot(w, log_y))
    return beta_hat / np.log(rho)


def _neg_loglik(params, log_i, n, y_bar):
    """-log-likelihood (additive constants dropped), eq. (loglik) of the derivation.

    params = (gamma, beta, log_sigma2), with a0 = exp(beta), sigma2 = exp(log_sigma2)
    (unconstrained reparametrization, so a0, sigma2 > 0 automatically).
    """
    gamma, beta, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)
    mu = np.exp(beta + gamma * log_i)
    K = len(log_i)
    return K / 2 * log_sigma2 + np.sum(np.log(mu)) + np.sum(n * (y_bar - mu) ** 2 / mu**2) / (2 * sigma2)


def _numerical_hessian(f, x0, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Hessian of scalar f at x0 (d-dimensional)."""
    x0 = np.asarray(x0, dtype=np.float64)
    d = len(x0)
    H = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            x1, x2, x3, x4 = (x0.copy() for _ in range(4))
            x1[a] += eps
            x1[b] += eps
            x2[a] += eps
            x2[b] -= eps
            x3[a] -= eps
            x3[b] += eps
            x4[a] -= eps
            x4[b] -= eps
            H[a, b] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * eps * eps)
    return H


def gamma_mle(scales, y_bar, n) -> dict:
    """Method 4: maximum-likelihood estimate of gamma (see module docstring).

    Direct joint optimization over (gamma, log a0, log sigma2) from the
    OLS-on-log estimate as a starting point (derivations/mle_gamma_estimator.tex,
    Section "Second-order conditions": the joint likelihood is not established
    to be globally concave, so this can in principle converge to a spurious
    point -- the returned dict's diagnostics are not decorative:

    - converged: the optimizer's own convergence flag.
    - region_ok: mu_hat_k < 2*Ybar_k for every k (the condition under which the
      likelihood, at fixed sigma2, is provably concave -- see the derivation).
    - hessian_pd: the numerical Hessian of -log-likelihood at the solution,
      over all three parameters, is positive definite (confirms a genuine
      local maximum of the likelihood, not a saddle).
    - trustworthy: all three of the above.

    Empirically (see the derivation): from a good starting point this recovers
    gamma about as well as gamma_all_points, but non-convergence and Hessian
    failures do occur (order a few percent of runs in testing) and should not
    be silently ignored.
    """
    scales = np.asarray(scales, dtype=np.float64)
    y_bar = np.asarray(y_bar, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    log_i = np.log(scales)

    gamma0, beta0 = ols_slope(log_i, np.log(y_bar))
    x0 = np.array([gamma0, beta0, 0.0])  # log_sigma2=0 -> sigma2=1, a generic starting scale

    res = minimize(_neg_loglik, x0, args=(log_i, n, y_bar), method="L-BFGS-B")
    gamma_hat, beta_hat, log_sigma2_hat = res.x
    a0_hat = float(np.exp(beta_hat))
    sigma2_hat = float(np.exp(log_sigma2_hat))
    mu_hat = a0_hat * scales**gamma_hat

    region_ok = bool(np.all(mu_hat < 2 * y_bar))
    H = _numerical_hessian(lambda p: _neg_loglik(p, log_i, n, y_bar), res.x)
    hessian_pd = bool(np.all(np.linalg.eigvalsh(H) > 0))

    return {
        "gamma_hat": float(gamma_hat),
        "a0_hat": a0_hat,
        "sigma2_hat": sigma2_hat,
        "converged": bool(res.success),
        "region_ok": region_ok,
        "hessian_pd": hessian_pd,
        "trustworthy": bool(res.success and region_ok and hessian_pd),
    }


def compare_methods(scales, y_bar, n, *, true_gamma: float | None = None) -> dict:
    """Bundle methods 1-4 into one JSON-serializable comparison dict."""
    log_i, log_y = _sorted_log(scales, y_bar)
    all_points_slope, all_points_intercept = ols_slope(log_i, log_y)
    result = {
        "scales": [int(s) for s in np.sort(np.asarray(scales))],
        "methods": {
            "all_points": {
                "description": "OLS slope of log(Y_bar_i) vs log(i), using every scale",
                "gamma_hat": float(all_points_slope),
                "a0_hat": float(np.exp(all_points_intercept)),
                "n_points": int(len(scales)),
            },
            "two_point": {
                "description": "OLS slope between each pair of adjacent scales (m=2)",
                "estimates": gamma_two_point(scales, y_bar),
            },
            "drop_leading": {
                "description": "OLS slope over all remaining scales after dropping the "
                "first m0 (m0 = 0..len(scales)-2)",
                "estimates": gamma_drop_leading(scales, y_bar),
            },
            "mle": {
                "description": "Maximum-likelihood estimate under Y_bar_k ~ N(mu_k, "
                "sigma^2 mu_k^2/n_k); see derivations/mle_gamma_estimator.tex. Check "
                "'trustworthy' before using -- see gamma_mle's docstring.",
                **gamma_mle(scales, y_bar, n),
            },
        },
    }
    if true_gamma is not None:
        result["true_gamma"] = true_gamma
    return result
