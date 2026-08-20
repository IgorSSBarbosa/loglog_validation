"""Correction-to-scaling exponent omega_1: article eq. (232),

    E Y_i = a_0 * i**gamma * exp(a_1 * i**-omega_1 + ... + phi_J(i) * i**-omega_J).

tools/loglog.py estimates gamma while *treating the correction as a nuisance*
-- gamma_drop_leading exists precisely to throw away the scales where it
bites. This module does the opposite: it estimates omega_1 itself, which is
what the budget-allocation rule (tools/allocation.py, Proposition prop:opt)
needs as an input.

Two independent estimators, deliberately not variants of one another, so
agreement between them is evidence rather than bookkeeping:

- `fit_correction` fits the one-term truncation of eq. (232) directly to
  log(Y_bar) vs log(i), with (log a0, gamma, a1, omega1) all free. It uses
  every scale and every sample mean, and is the more efficient of the two
  when the model is right.
- `omega1_from_bias_decay` never looks at Y_bar. It takes a *sequence of
  gamma-hat estimates* (e.g. tools/loglog.py's gamma_drop_leading, one per
  m0) and fits gamma_hat(i) = gamma_inf + a * i**-omega1 to how that
  sequence converges as the most-contaminated small scales are dropped. This
  measures the bias decay of a specific estimator, which is the quantity
  that actually justifies a choice of m0 -- and it is a different functional
  of the data, so it fails differently when the model is wrong.

Both are plain nonlinear least squares over 3-4 parameters. Neither is taken
from the article as a named estimator -- the article specifies the model
(eq. 232) and the gamma estimators, not a procedure for omega_1 -- so these
are validated against planted ground truth in tools/tests/test_correction.py
rather than against an equation number (ground rule 1/4).

CAUTION -- the scale grid must not mix parities for a lattice observable.
Both fits assume E Y_i varies smoothly in i, as eq. (232) does. A lattice
walk need not: for srw's |S_k| the exact mean is a STAIRCASE,

    E|S_{2m-1}| = E|S_{2m}|   exactly, for every m

(verified for m = 1..300), because a walk of odd length cannot return as
close to the origin as the even length above it. A grid of even scales -- in
particular the powers of 2 this project uses, and any rho**k grid with rho a
power of 2 anchored at one -- samples one consistent branch of that staircase
and the smooth model applies. A grid that mixes odd and even k does not, and
the resulting zig-zag is fitted as if it were curvature: measured on EXACT
means with zero sampling noise, a rho = sqrt(2) grid over 8..256 returns
omega1 ~ 17.8 instead of 1. That failure is silent -- the fit converges and
reports a small residual -- so it is the caller's job to choose the grid, and
this is why the failure mode is documented here rather than guarded against
(the parity that matters is a property of the model, which this module never
sees).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Bounds on omega1 for both fits. The lower bound keeps the optimizer away
# from omega1 -> 0, where i**-omega1 -> 1 becomes degenerate with the
# intercept (log a0) and the problem loses identifiability; the upper bound
# keeps it away from omega1 -> inf, where the correction vanishes at every
# scale and a1 is unidentifiable. Neither is a claim about the true value.
_OMEGA1_BOUNDS = (1e-3, 20.0)


def fit_correction(
    scales: Sequence,
    y_bar: Sequence,
    *,
    sigma_log: Sequence | None = None,
    omega1_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict:
    """Fit log(E Y_i) = log a0 + gamma*log i + a1 * i**-omega1 by least squares.

    Returns {'gamma', 'omega1', 'a0', 'a1', 'rel_rmse', 'converged'}.

    `sigma_log`, if given, is the standard error of each log(y_bar[i]) and is
    used to weight the residuals (the natural weighting: log Y_bar has
    variance ~ sigma_i^2/(n_i * mu_i^2), which differs across scales whenever
    n_i does -- and Experiment B's allocation deliberately makes n_i differ).

    The (a1, omega1) pair is only weakly identified when the correction is
    small, and the objective is not convex in omega1, so the fit is restarted
    from several omega1 seeds (`omega1_grid`) and the best objective kept --
    a single start from an arbitrary guess is genuinely liable to stop in a
    local minimum here.
    """
    from scipy.optimize import least_squares

    i = np.asarray(scales, dtype=float)
    y = np.asarray(y_bar, dtype=float)
    if i.size != y.size:
        raise ValueError(f"scales and y_bar differ in length: {i.size} vs {y.size}")
    if i.size < 5:
        raise ValueError("need at least 5 scales to fit 4 free parameters")
    if np.any(i <= 0) or np.any(y <= 0):
        raise ValueError("scales and y_bar must be strictly positive")

    log_i, log_y = np.log(i), np.log(y)
    if sigma_log is None:
        w = np.ones_like(log_y)
    else:
        s = np.asarray(sigma_log, dtype=float)
        if s.shape != log_y.shape:
            raise ValueError("sigma_log must match scales in length")
        if np.any(s <= 0):
            raise ValueError("sigma_log must be strictly positive")
        w = 1.0 / s

    def residual(theta):
        log_a0, gamma, a1, omega1 = theta
        model = log_a0 + gamma * log_i + a1 * i ** (-omega1)
        return w * (model - log_y)

    # Start gamma/log a0 from the plain OLS-on-logs fit, which ignores the
    # correction entirely -- a deliberately correction-blind starting point.
    gamma0, log_a00 = np.polyfit(log_i, log_y, 1)

    best = None
    for omega0 in omega1_grid:
        try:
            fit = least_squares(
                residual,
                x0=[log_a00, gamma0, 0.0, omega0],
                bounds=(
                    [-np.inf, -np.inf, -np.inf, _OMEGA1_BOUNDS[0]],
                    [np.inf, np.inf, np.inf, _OMEGA1_BOUNDS[1]],
                ),
            )
        except Exception:  # pragma: no cover - optimizer blow-up on pathological input
            continue
        if best is None or fit.cost < best.cost:
            best = fit

    if best is None:  # pragma: no cover
        raise RuntimeError("all least_squares restarts failed")

    log_a0, gamma, a1, omega1 = best.x
    return {
        "gamma": float(gamma),
        "omega1": float(omega1),
        "a0": float(np.exp(log_a0)),
        "a1": float(a1),
        "rel_rmse": float(np.sqrt(np.mean((best.fun / w) ** 2))),
        "converged": bool(best.success),
    }


def omega1_from_bias_decay(
    scales: Sequence,
    gamma_hats: Sequence,
    *,
    omega1_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict:
    """Fit gamma_hat(i) = gamma_inf + a * i**-omega1 to a sequence of estimates.

    `scales[j]` is the smallest scale retained by the window that produced
    `gamma_hats[j]` -- for tools/loglog.py's `gamma_drop_leading` output that
    is `est['scales_used'][0]`, and `gamma_hats[j]` is `est['gamma_hat']`.

    Returns {'gamma_inf', 'omega1', 'a', 'rel_rmse', 'converged'}, where
    `gamma_inf` is the extrapolated bias-free estimate: the value the
    estimator would converge to if the smallest usable scale went to infinity.

    Note these inputs are strongly correlated with each other (nested windows
    over the same samples), so `rel_rmse` describes fit quality only -- it is
    NOT a standard error for omega1. Get that from replicate runs with
    independent seeds (ground rule 2), not from this fit.
    """
    from scipy.optimize import least_squares

    i = np.asarray(scales, dtype=float)
    g = np.asarray(gamma_hats, dtype=float)
    if i.size != g.size:
        raise ValueError(f"scales and gamma_hats differ in length: {i.size} vs {g.size}")
    if i.size < 4:
        raise ValueError("need at least 4 estimates to fit 3 free parameters")
    if np.any(i <= 0):
        raise ValueError("scales must be strictly positive")

    def residual(theta):
        gamma_inf, a, omega1 = theta
        return gamma_inf + a * i ** (-omega1) - g

    best = None
    for omega0 in omega1_grid:
        try:
            fit = least_squares(
                residual,
                x0=[g[-1], g[0] - g[-1], omega0],
                bounds=(
                    [-np.inf, -np.inf, _OMEGA1_BOUNDS[0]],
                    [np.inf, np.inf, _OMEGA1_BOUNDS[1]],
                ),
            )
        except Exception:  # pragma: no cover
            continue
        if best is None or fit.cost < best.cost:
            best = fit

    if best is None:  # pragma: no cover
        raise RuntimeError("all least_squares restarts failed")

    gamma_inf, a, omega1 = best.x
    return {
        "gamma_inf": float(gamma_inf),
        "omega1": float(omega1),
        "a": float(a),
        "rel_rmse": float(np.sqrt(np.mean(best.fun**2))),
        "converged": bool(best.success),
    }
