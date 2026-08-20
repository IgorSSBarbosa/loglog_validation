"""Cost-model exponent estimation: article Assumption cost_is_power_law states
cost(i) = i**d, "the computational complexity of simulating one sample of
distribution Y_i". This has exactly the same log-log-linear form as
E[Y_i] = a0 * i**gamma (eq. 232), so d is recoverable the same way gamma is --
`_ols_cost_exponent` just calls `loglog.gamma_all_points` with elapsed time
standing in for Y_i.

`COST_ESTIMATORS` is a name -> function registry (mirrors
experiments/00_synthetic/generator.py's NOISE_FAMILIES) so a different
estimation approach can be added later as one more entry, without touching
callers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Self-contained: works whether this module is reached as `tools.cost_model`
# (repo root on sys.path, e.g. experiments/*/measure_cost.py) or as a bare
# `cost_model` (tools/ itself on sys.path, e.g. tools/tests/test_loglog.py's
# convention) -- either way, `loglog` needs its own directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loglog import gamma_all_points  # noqa: E402


def _ols_cost_exponent(scales: Sequence, elapsed: Sequence) -> float:
    return gamma_all_points(scales, elapsed)


COST_ESTIMATORS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "ols": _ols_cost_exponent,
}


def estimate_cost_exponent(scales: Sequence, elapsed: Sequence, method: str = "ols") -> float:
    """Estimate d from cost(i) = c * i**d, given elapsed[i] measured at each scales[i]."""
    if method not in COST_ESTIMATORS:
        raise ValueError(f"unknown method {method!r}; known: {list(COST_ESTIMATORS)}")
    return COST_ESTIMATORS[method](scales, elapsed)


# --------------------------------------------------------------------------
# Aggregating repeated timings at one scale
# --------------------------------------------------------------------------
# Repeated timings of the same deterministic quantity differ only by OS and
# interpreter jitter, which is one-sided: noise can only ever ADD delay. `min`
# is therefore the standard microbenchmark choice (Python's own timeit) and is
# what this project used originally. Measured on the existing cost_probe /
# time_measure runs (2026-08-20) the choice barely moves d_hat -- min 0.884,
# median 0.888, iqmean 0.887, mean 0.850, q95 0.811 -- so this registry exists
# to allow an aggregator with a usable spread/CI (min has none), not to change
# the estimate. Default is `median` (user's choice, 2026-08-20): robust to the
# one-sided jitter like min, but with a distribution-free confidence interval
# from order statistics, so the cost curve can carry honest error bars.


def _iqmean(times: np.ndarray) -> float:
    """Mean of the central 50% (inter-quartile mean): trims both tails."""
    lo, hi = np.quantile(times, 0.25), np.quantile(times, 0.75)
    central = times[(times >= lo) & (times <= hi)]
    return float(central.mean()) if central.size else float(np.median(times))


AGGREGATORS: dict[str, Callable[[np.ndarray], float]] = {
    "min": lambda t: float(t.min()),
    "median": lambda t: float(np.median(t)),
    "mean": lambda t: float(t.mean()),
    "q95": lambda t: float(np.quantile(t, 0.95)),
    "iqmean": _iqmean,
}

DEFAULT_AGGREGATOR = "median"


def aggregate(times: Sequence[float], method: str = DEFAULT_AGGREGATOR) -> float:
    """Collapse repeated timings at one scale to a single cost estimate."""
    if method not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {method!r}; known: {list(AGGREGATORS)}")
    arr = np.asarray(times, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot aggregate an empty timing sample")
    return AGGREGATORS[method](arr)


def median_ci(times: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Distribution-free CI for the median, from the order statistics.

    The number of observations below the true median is Binomial(N, 1/2), so
    the r-th and (N+1-r)-th order statistics bracket it with a probability
    that depends on no distributional assumption at all -- which is the reason
    `median` is the default aggregator over `min` (a minimum has no comparable
    interval). Falls back to (min, max) when N is too small to do better.
    """
    from scipy.stats import binom  # local import: keeps module import cheap

    arr = np.sort(np.asarray(times, dtype=float))
    N = arr.size
    if N < 2:
        return (float(arr[0]), float(arr[0])) if N else (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    # Largest r with P(r <= #below < N+1-r) >= confidence.
    r = int(binom.ppf(alpha / 2.0, N, 0.5))
    r = max(1, min(r, N // 2))
    return float(arr[r - 1]), float(arr[N - r])


# --------------------------------------------------------------------------
# Affine-plus-power cost model
# --------------------------------------------------------------------------


def estimate_cost_affine(scales: Sequence, elapsed: Sequence) -> dict:
    """Fit cost(i) = a + b * i**d, returning {'a', 'b', 'd', 'rel_rmse', 'converged'}.

    Assumption cost_is_power_law (eq. 353) states cost(i) = i**d, and
    `estimate_cost_exponent` fits exactly that. But a real timing probe also
    pays a fixed per-call overhead that is independent of i -- on this machine
    ~22us of Python/NumPy dispatch, which at k=2 is essentially 100% of the
    measurement. Fitting a pure power law to a + b*i**d data returns a badly
    biased d (measured: 0.10 on `time_measure`, where the truth is 1). This
    estimator separates the two, so `a` can be reported as a diagnostic and `d`
    read off the part that actually scales.

    Fitted in log space (relative, not absolute, error) because costs span
    orders of magnitude across the scale grid.
    """
    from scipy.optimize import least_squares  # local import: keeps module import cheap

    i = np.asarray(scales, dtype=float)
    y = np.asarray(elapsed, dtype=float)
    if i.size != y.size:
        raise ValueError(f"scales and elapsed differ in length: {i.size} vs {y.size}")
    if i.size < 4:
        raise ValueError("affine fit needs at least 4 scales (3 free parameters)")
    if np.any(y <= 0) or np.any(i <= 0):
        raise ValueError("scales and elapsed must be strictly positive")

    log_y = np.log(y)

    def residual(theta):
        log_a, log_b, d = theta
        return np.log(np.exp(log_a) + np.exp(log_b) * i ** d) - log_y

    # Start from the pure-power fit with the overhead guessed as the cheapest
    # observed timing (an upper bound on the true floor).
    d0 = float(gamma_all_points(i, y))
    a0 = max(y.min() * 0.5, 1e-15)
    b0 = max((y.max() - a0) / (i.max() ** max(d0, 1e-3)), 1e-15)
    fit = least_squares(
        residual,
        x0=[np.log(a0), np.log(b0), d0],
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 8.0]),
    )
    log_a, log_b, d = fit.x
    return {
        "a": float(np.exp(log_a)),
        "b": float(np.exp(log_b)),
        "d": float(d),
        "rel_rmse": float(np.sqrt(np.mean(fit.fun ** 2))),
        "converged": bool(fit.success),
    }
