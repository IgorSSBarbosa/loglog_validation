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

The exact closed-form weights (eq. 526) and their algebraic identities (eq. 542)
are checkpoint 0.2's own acceptance criterion and are a separate, still-open
item (see PLAN.md / TODO.md) -- not implemented here.
"""

from __future__ import annotations

import numpy as np


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


def compare_methods(scales, y_bar, *, true_gamma: float | None = None) -> dict:
    """Bundle methods 1-3 into one JSON-serializable comparison dict."""
    result = {
        "scales": [int(s) for s in np.sort(np.asarray(scales))],
        "methods": {
            "all_points": {
                "description": "OLS slope of log(Y_bar_i) vs log(i), using every scale",
                "gamma_hat": gamma_all_points(scales, y_bar),
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
        },
    }
    if true_gamma is not None:
        result["true_gamma"] = true_gamma
    return result
