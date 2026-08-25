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

    # Standard error of d from the Gauss-Newton covariance,
    # sigma^2 (J^T J)^-1, with sigma^2 the residual variance on the log scale.
    # Without it `compare_cost_models` cannot say whether a declared and a
    # measured exponent actually disagree -- it would only ever report a gap.
    d_se = None
    dof = len(i) - 3
    if dof > 0:
        try:
            jtj = fit.jac.T @ fit.jac
            cov = np.linalg.inv(jtj) * float(fit.fun @ fit.fun) / dof
            var_d = float(cov[2, 2])
            if np.isfinite(var_d) and var_d > 0:
                d_se = float(np.sqrt(var_d))
        except np.linalg.LinAlgError:      # singular: d not identified here
            d_se = None

    return {
        "a": float(np.exp(log_a)),
        "b": float(np.exp(log_b)),
        "d": float(d),
        "d_se": d_se,
        "rel_rmse": float(np.sqrt(np.mean(fit.fun ** 2))),
        "converged": bool(fit.success),
    }


# --------------------------------------------------------------------------
# Declared vs measured cost: the cross-check
# --------------------------------------------------------------------------

def declared_exponent(scales: Sequence, cost_hint, params: dict | None = None
                      ) -> float:
    """The exponent d implied by a model's own cost_hint, by OLS on logs.

    Exact for a true power law (srw's cost_hint(i) = i returns 1.0 to float
    precision). A model whose cost is not a power law -- say a + b*i -- will
    return the local slope over `scales`, which is the honest answer: that IS
    the effective d over the window being planned for.
    """
    i = np.asarray(scales, dtype=float)
    c = np.asarray([float(cost_hint(int(k), params or {})) for k in i])
    if np.any(c <= 0):
        raise ValueError(f"cost_hint must be positive; got {c.tolist()}")
    if np.allclose(c, c[0]):
        return 0.0                      # constant cost, e.g. the synthetic model
    return float(np.polyfit(np.log(i), np.log(c), 1)[0])


def compare_cost_models(scales: Sequence, elapsed: Sequence, cost_hint,
                        params: dict | None = None,
                        tolerance_sigma: float = 3.0,
                        tolerance_rel: float = 0.05) -> dict:
    """Cross-check a model's declared cost against the wall clock.

    Returns the declared d, the measured d (affine fit, which is the only one
    that survives per-call overhead), their gap in units of the measured
    stderr, and `agree`. The declared value is what an allocation should use --
    it is exact where the clock is not -- but a real disagreement usually means
    one of two things worth knowing:

      * the cost_hint is wrong (the exploration is not the complexity you
        thought), or
      * the machine has stopped being compute-bound at large i (cache, memory
        bandwidth, swap), so wall clock genuinely grows faster than work does.

    Neither is detectable from either measurement alone, which is why both are
    kept rather than picking one.
    """
    declared = declared_exponent(scales, cost_hint, params)
    try:
        aff = estimate_cost_affine(scales, elapsed)
        measured, se, how = aff["d"], aff.get("d_se"), "affine a + b*i**d"
    except ValueError as exc:                  # too few scales for 3 parameters
        measured = estimate_cost_exponent(scales, elapsed)
        se, how = None, f"pure power law ({exc})"

    z = (measured - declared) / se if se else None
    rel = abs(measured - declared) / abs(declared) if declared else float("inf")

    # Two tolerances, and agreement needs only one of them. The sigma test
    # alone is too strict here: `estimate_cost_affine`'s standard error is
    # residual-based and treats timing noise as i.i.d., which it is not --
    # cache behaviour, clock scaling and the affine model's own approximation
    # error are all systematic. On srw, whose declared d = 1 is EXACT by
    # construction, the measured 1.0063 +/- 0.0023 sits 2.76 sigma away: a
    # 0.6% discrepancy inflated by an over-tight se. Flagging that as a
    # disagreement would train the user to ignore the warning, which is worse
    # than not having it. The relative floor says what actually matters for an
    # allocation -- the offset moves only logarithmically in d, so sub-percent
    # differences are irrelevant regardless of their significance.
    agree = True
    if z is not None:
        agree = bool(abs(z) <= tolerance_sigma or rel <= tolerance_rel)

    return {
        "declared_d": declared,
        "measured_d": measured,
        "measured_se": se,
        "measured_via": how,
        "z": z,
        "rel_gap": rel,
        "agree": agree,
        "tolerance_sigma": tolerance_sigma,
        "tolerance_rel": tolerance_rel,
    }


def format_cost_comparison(cmp: dict) -> str:
    """One block, warning first when the two disagree."""
    lines = []
    if not cmp["agree"]:
        lines.append(
            f"WARNING: declared and measured cost exponents differ by "
            f"{abs(cmp['z']):.1f} sigma.")
        lines.append("  Either cost_hint is wrong, or the machine is no longer "
                     "compute-bound at the largest scales.")
        lines.append("  Allocation uses the DECLARED value; "
                     "pass --trust-measured to override.")
    se = cmp["measured_se"]
    lines.append(f"  declared d = {cmp['declared_d']:.4f}   (model's cost_hint)")
    lines.append(f"  measured d = {cmp['measured_d']:.4f}"
                 + (f" +/- {se:.4f}" if se else "")
                 + f"   ({cmp['measured_via']})")
    if cmp["z"] is not None:
        lines.append(f"  gap = {cmp['z']:+.2f} sigma, {100 * cmp['rel_gap']:.2f}% relative"
                     + ("" if cmp["agree"] else "   -> DISAGREE"))
    return "\n".join(lines)
