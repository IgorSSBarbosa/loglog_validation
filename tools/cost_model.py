"""Cost-model exponent estimation: article Assumption cost_is_power_law states
cost(i) = i**d, "the computational complexity of simulating one sample of
distribution Y_i". This has exactly the same log-log-linear form as
E[Y_i] = a0 * i**gamma (eq. 232), so d is recoverable the same way gamma is --
`_ols_cost_exponent` just calls `loglog.gamma_all_points` with elapsed time
standing in for Y_i.

`COST_ESTIMATORS` is a name -> function registry (mirrors
models/synthetic.py's NOISE_FAMILIES) so a different
estimation approach can be added later as one more entry, without touching
callers.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

# Self-contained: works whether this module is reached as `tools.cost_model`
# (repo root on sys.path, e.g. experiments/*/measure_cost.py) or as a bare
# `cost_model` (tools/ itself on sys.path, e.g. tools/tests/test_loglog.py's
# convention) -- either way, `loglog` needs its own directory on the path.

from tools.loglog import gamma_all_points

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

def work_units(scales, counts, cost_hint, params: dict | None = None) -> float:
    """Actual work of drawing `counts[i]` samples at each scale, in cost_hint units.

    The unit is whatever the model's `cost_hint` counts -- lattice sites, walk
    steps, seconds of burn. `counts` may be a scalar (the same n at every
    scale) or one entry per scale.
    """
    import numpy as np

    i = np.atleast_1d(np.asarray(scales))
    n = np.broadcast_to(np.atleast_1d(np.asarray(counts, dtype=float)), i.shape)
    return float(sum(float(c) * float(cost_hint(int(k), params or {}))
                     for k, c in zip(i, n)))


def cost_unit_ratio(scales, counts, d: float, cost_hint,
                    params: dict | None = None) -> float:
    """The constant between the ALLOCATION's cost unit and the model's own.

    tools/allocation.py charges Assumption 7's cost(i) = i**d -- the SCALE
    raised to d -- while a model's `cost_hint` reports the work it actually
    does. For srw (cost_hint(i) = i, d = 1) and percolation2d (i**2, d = 2)
    those coincide and this returns 1.0. They are NOT the same thing in
    general: percolation_tau's cost_hint is L(s)**2 = box_factor**2 * s, so
    the ratio is box_factor**2 = 256, and synthetic's burn carries
    `cost_scale`.

    Dividing a budget in allocation units by a throughput measured in
    cost_hint units -- which is exactly what src/study/plan.py did until
    2026-09-06 -- is therefore wrong by this factor, and silently right for
    every model that had been tried. It predicted 740 s for a run that was
    going to take 42 hours.

        seconds = cost_in_allocation_units * cost_unit_ratio / throughput
        budget  = seconds * throughput / cost_unit_ratio

    Returns Sum(n_i cost_hint(i)) / Sum(n_i i**d) over the ladder given, so it
    is exact for the ladder it is asked about (including the `ceil` wobble in
    a box rule), not merely asymptotic.
    """
    import numpy as np

    i = np.atleast_1d(np.asarray(scales, dtype=float))
    n = np.broadcast_to(np.atleast_1d(np.asarray(counts, dtype=float)), i.shape)
    declared = float((n * i ** float(d)).sum())
    if declared <= 0:
        raise ValueError(f"allocation cost must be positive; got {declared}")
    return work_units(scales, counts, cost_hint, params) / declared


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

    `agree` is None when the two could not be compared at all -- fewer than
    four scales leaves the affine fit without a standard error, and a gap with
    no error bar is not evidence either way. Check `comparable` before reading
    `agree`; `format_cost_comparison` says so in words.
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
    # Whether a comparison was possible AT ALL, which is not the same question
    # as whether it agreed. Without a standard error the sigma test below is
    # skipped, and `agree` used to keep its initial True -- indistinguishable
    # in the returned dict from a real agreement, and silent in
    # `format_cost_comparison`. It is now None, and the caller has to say so.
    comparable = se is not None and se > 0

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
    agree = (bool(abs(z) <= tolerance_sigma or rel <= tolerance_rel)
             if comparable else None)

    return {
        "declared_d": declared,
        "measured_d": measured,
        "measured_se": se,
        "measured_via": how,
        "z": z,
        "rel_gap": rel,
        "comparable": comparable,
        "agree": agree,
        "tolerance_sigma": tolerance_sigma,
        "tolerance_rel": tolerance_rel,
    }

def format_cost_comparison(cmp: dict) -> str:
    """One block, warning first when the two disagree -- or cannot be compared."""
    lines = []
    if cmp.get("agree") is None:
        lines.append(
            "NOT COMPARED: no standard error for the measured d, so the "
            "declared value\n  cannot be scored against it. This is not "
            "agreement -- it is the absence of\n  a test. Time at least "
            "4 scales so the affine fit has a degree of freedom.")
    elif not cmp["agree"]:
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
    elif cmp["declared_d"]:
        lines.append(f"  gap = {100 * cmp['rel_gap']:.2f}% relative, "
                     f"significance unknown")
    return "\n".join(lines)

# --------------------------------------------------------------------------
# Timing a model: the two probes
# --------------------------------------------------------------------------
#
# There are exactly two ways this repo measures cost(i), and they differ only
# in WHICH scales get timed:
#
#   time_over_scales  times a ladder you name  -- src/estimate/measure_cost.py,
#                     the standalone probe, where the ladder is the experiment.
#   probe_window      MEASURES the per-call overhead, then places a window just
#                     above it -- src/study/pilot.py, where the ladder is not
#                     free: the pilot's own scales are chosen so the CORRECTION
#                     term is visible, which means small, and a single
#                     simulate() call there may be almost entirely dispatch.
#   climb_to_target   doubles the scale from a start until one call is slow
#                     enough. What pilot.py used before probe_window, kept
#                     because it is the right strategy when no overhead
#                     measurement is available, and still exercised by
#                     calibration/exercise_all.py.
#
# All three hand the same {scales, elapsed, times, repeats, aggregator} to
# `fit_cost_probe`, so the fitting, the overhead diagnostic and the
# declared-vs-measured cross-check are written once. Before this split the two
# callers had their own timing loop and their own fit, and only one of them
# had learned that the pure power law is unusable at small k.

#: Repeats per scale. Small on purpose: the affine fit needs the SHAPE of
#: cost(i) across scales, not a precise absolute time at any one of them.
PROBE_REPEATS = 5

#: How slow a single call must get before the climb stops. Below roughly this,
#: the fixed ~10-25us of per-call dispatch is a visible share of the
#: measurement; timing srw's own pilot ladder (8..256) returned d = 8.0 +/- 280.
PROBE_TARGET_SECONDS = 2e-3

#: Ceilings on the climb, so a model with a steep cost cannot hang its caller.
PROBE_MAX_DOUBLINGS = 24
PROBE_TIME_BUDGET = 20.0

#: Fewest rungs worth fitting: `estimate_cost_affine` has 3 free parameters.
PROBE_MIN_SCALES = 4

#: How many times a rung's WORK must exceed the per-call overhead before that
#: rung is worth fitting. This is what PROBE_TARGET_SECONDS was a proxy for,
#: and the proxy was wrong in both directions because it is an ABSOLUTE time
#: while the thing it guards against is a RATIO. Measured on this machine
#: (2026-09-14), fitting cost(i) = a + b*i**d over windows placed by this rule
#: against windows placed by the old one:
#:
#:   model                     window        cost      d_hat           truth
#:   percolation_zd dim=3      128..1024     280 s     3.086 +/- 0.042   3
#:   percolation_zd dim=3       32..256       12 s     2.999 +/- 0.033   3
#:   percolation_zd dim=3       16..128        3 s     2.953 +/- 0.013   3
#:   srw                      1024..8192    0.04 s     1.000 +/- 0.014   1
#:   srw                     16384..131072  0.07 s     1.027 +/- 0.013   1
#:
#: and BELOW the floor the fit is not merely noisy but wrong: srw over 8..64
#: fails to converge at all, over 32..256 returns 3.42 +/- 0.69, and
#: percolation_zd over 2..256 returns 2.913. kappa = 3 is where every model in
#: the repo lands on a window whose d_hat is within 0.05 of its declared cost.
PROBE_OVERHEAD_FACTOR = 3.0

#: Seconds ONE probe may spend, including its overhead measurement and its
#: walk up to the floor. A ceiling on a diagnostic, not a precision knob: the
#: error budget prices se(d) = 0.042 at 1.0006x in RMSE (plan.py's
#: `input_sensitivity`), so a probe has nothing to buy with more time. When the
#: doubling window does not fit, the rungs are packed closer together rather
#: than dropped -- a shorter lever arm widens se(d), which costs nothing, while
#: fewer than PROBE_MIN_SCALES rungs means no affine fit at all.
#:
#: Deliberately NOT scaled with the pilot's own draw time, which was the
#: obvious generalisation and is measured to buy nothing. On percolation_zd
#: dim=5, where the budget is what binds:
#:
#:   window          cost     d_hat              declared
#:   8..16 (packed)   1.7 s   4.849 +/- 0.051    5
#:   8..32 (wide)    33.5 s   4.799 +/- 0.005    5
#:
#: 20x the compute moves d_hat by 0.05 and AWAY from the declaration. The
#: shortfall in high d is systematic, not a lever-arm artefact -- see TODO.md,
#: "the measured cost exponent falls increasingly short of the declared one as
#: d grows" -- so a wider window measures the same wrong thing more precisely.
PROBE_WINDOW_BUDGET = 6.0

def time_at_scale(spec, i: int, params: dict, rng, repeats: int = PROBE_REPEATS,
                  aggregator: str = DEFAULT_AGGREGATOR) -> tuple[float, list[float]]:
    """Time `spec.simulate(i, 1, params, rng)` `repeats` times.

    Returns (aggregated seconds, the raw timings). One warm-up call is thrown
    away first: the first call through a code path pays import, branch
    prediction and allocator costs that no later call does, and at these
    durations that single outlier moves a mean noticeably.

    Timed one call at a time rather than as a batch divided by `repeats`, so
    the aggregator (median by default) can do its job -- jitter here is
    one-sided, and a mean hands the whole of any hiccup to the estimate.
    """
    spec.simulate(i, 1, params, rng)                     # warm the code path
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        spec.simulate(i, 1, params, rng)
        times.append(time.perf_counter() - t0)
    return aggregate(times, aggregator), times

def _probe(scales, times_by_scale, repeats, aggregator, **extra) -> dict:
    """The common probe payload both timing strategies return."""
    return {"scales": [int(k) for k in scales],
            "elapsed": [aggregate(times_by_scale[k], aggregator) for k in scales],
            "elapsed_all": {str(k): times_by_scale[k] for k in scales},
            "repeats": repeats, "aggregator": aggregator, **extra}

def time_over_scales(spec, scales, params: dict, rng, repeats: int = PROBE_REPEATS,
                     aggregator: str = DEFAULT_AGGREGATOR) -> dict:
    """Time a ladder of scales you choose. The standalone probe's strategy."""
    times_by_scale = {int(k): time_at_scale(spec, int(k), params, rng, repeats,
                                            aggregator)[1]
                      for k in scales}
    return _probe([int(k) for k in scales], times_by_scale, repeats, aggregator)

def climb_to_target(spec, params: dict, rng, start: int,
                    repeats: int = PROBE_REPEATS,
                    aggregator: str = DEFAULT_AGGREGATOR,
                    target_seconds: float = PROBE_TARGET_SECONDS,
                    max_doublings: int = PROBE_MAX_DOUBLINGS,
                    time_budget: float = PROBE_TIME_BUDGET) -> dict:
    """Double the scale from `start` until one call takes `target_seconds`.

    The pilot's strategy, and the reason it exists: a probe must be run where
    the WORK dominates the per-call overhead, and that is generally nowhere
    near the scales a log-log ladder wants to sample. d is a property of the
    model, not of the window, so measuring it further out costs nothing.

    Stops on whichever comes first: the target, `max_doublings` rungs, or
    `time_budget` seconds -- but never before `PROBE_MIN_SCALES` rungs, which
    is what the affine fit needs. `reached_target` reports which happened.

    That floor now holds on every exit, not just the target one. The time
    budget used to be able to cut the climb short at three rungs, leaving
    `fit_cost_probe` with no affine fit and the pilot with no measured d --
    the guarantee this docstring states, but two of the three branches did
    not keep. A `max_doublings` below the floor is a contradiction and is
    refused up front rather than silently honoured.

    A FOURTH stopping rule: the model REFUSING the scale. A climb doubles
    blindly, and a model whose per-sample working set grows as i**dim runs out
    of addressable memory long before it runs out of clock -- at dim = 3 the
    climb reached i = 2048, i.e. 8.6e9 sites and 80 GiB, and
    models/percolation_zd.py's guard raised, correctly. The climb used to let
    that propagate, so a pilot that had already spent nine minutes drawing
    replicates died at the cost probe with a ValueError about a scale nobody
    asked for (user, 2026-09-06). A refusal is now a CEILING: the climb stops,
    keeps the rungs it has, and records `refused_at`.

    A ceiling reached too early is not an error either, because the probe does
    not care WHERE the rungs are -- d is a property of the model, not of the
    window (`src/study/pilot.py`). pilot.py starts the climb at the ladder's
    LARGEST scale, so on percolation_zd at dim = 3 the ceiling arrives after
    three rungs, one short of PROBE_MIN_SCALES. The climb then fills DOWNWARD
    (`extended_down`), halving below the smallest rung it has, which always has
    room -- rather than restarting, which would re-time the expensive rungs it
    already measured. Only running out of room at i = 1 is fatal, and that
    means the parameters cannot be simulated at any useful scale.
    """
    if max_doublings < PROBE_MIN_SCALES:
        raise ValueError(
            f"max_doublings={max_doublings} cannot reach PROBE_MIN_SCALES="
            f"{PROBE_MIN_SCALES} rungs, which the affine fit needs")
    scales, times_by_scale = [], {}
    refused_at = None
    i = int(start)
    t_start = time.perf_counter()
    for _ in range(max_doublings):
        try:
            agg, times = time_at_scale(spec, i, params, rng, repeats, aggregator)
        except (ValueError, MemoryError) as exc:
            # The model says this scale is past what it can build. That is a
            # ceiling on the climb, not a failure of the probe -- unless we do
            # not yet have enough rungs to fit anything.
            refused_at = i
            break
        scales.append(i)
        times_by_scale[i] = times
        if len(scales) < PROBE_MIN_SCALES:
            i *= 2
            continue                # the floor outranks both stopping rules
        if agg >= target_seconds:
            break
        if time.perf_counter() - t_start > time_budget:
            break
        i *= 2
    # The ceiling can arrive before PROBE_MIN_SCALES rungs -- pilot.py starts
    # the climb at the ladder's LARGEST scale, so on percolation_zd at dim = 3
    # it gets three rungs and then 80 GiB. Fill downward rather than upward:
    # d does not depend on where the window is, the rungs already measured are
    # good and are kept, and halving always has room. Only a refusal at i = 1
    # means the parameters cannot be simulated at all.
    extended_down = []
    while len(scales) < PROBE_MIN_SCALES:
        j = (min(scales) if scales else int(start)) // 2
        if j < 1:
            raise ValueError(
                f"the cost probe cannot reach {PROBE_MIN_SCALES} rungs: the "
                f"model refused scale {refused_at} and there is nothing below "
                f"{min(scales) if scales else start} left to measure. These "
                f"parameters are not simulable at any useful scale.")
        try:
            _, times = time_at_scale(spec, j, params, rng, repeats, aggregator)
        except (ValueError, MemoryError) as exc:
            # Refused going DOWN as well: this is not a ceiling, the model
            # cannot run these parameters at all.
            raise ValueError(
                f"the cost probe cannot reach {PROBE_MIN_SCALES} rungs: the "
                f"model refused scale {j} on the way down as well as "
                f"{refused_at} on the way up. These parameters are not "
                f"simulable at any useful scale. Underlying refusal: {exc}"
            ) from exc
        scales.insert(0, j)
        times_by_scale[j] = times
        extended_down.append(j)

    out = _probe(scales, times_by_scale, repeats, aggregator,
                 target_seconds=target_seconds)
    out["reached_target"] = bool(out["elapsed"] and
                                 out["elapsed"][-1] >= target_seconds)
    out["refused_at"] = refused_at
    out["extended_down"] = extended_down or None
    return out

def measure_overhead(spec, params: dict, rng, repeats: int = PROBE_REPEATS,
                     aggregator: str = DEFAULT_AGGREGATOR,
                     ceiling: int | None = None) -> tuple[float, int]:
    """Time one call at the smallest scale the model will accept.

    This is `a` of the affine model cost(i) = a + b*i**d, measured rather than
    fitted, and it is the quantity the whole window rule is built on. It has to
    be measured because the FITTED `a` is unreliable exactly where it matters:
    on percolation_zd dim=3 the affine fit over 128..1024 reported a = 3.2 ms
    against a directly measured 0.21 ms, a factor of 15, because over a short
    top-heavy window `a` absorbs curvature that belongs to `d`. Reading the
    window off that number would have said "no rung below 128 is usable", and
    the truth is that everything from 32 up is.

    Nearly free: at i = 1 there is no work to do, so this costs
    `repeats + 1` dispatches -- 0.5 ms on percolation_zd, 0.06 ms on srw.

    Returns (seconds, the scale it was measured at). Climbs if the model
    refuses i = 1, since some parameter sets have a minimum size.
    """
    i = 1
    while ceiling is None or i <= max(1, int(ceiling)):
        try:
            agg, _ = time_at_scale(spec, i, params, rng, repeats, aggregator)
            return float(agg), int(i)
        except (ValueError, MemoryError):
            i *= 2
    raise ValueError(
        f"the model refused every scale up to {ceiling}, so the per-call "
        f"overhead cannot be measured and no probe window can be placed")


def _space(bottom: int, top: int, count: int) -> list[int]:
    """`count` geometrically spaced integer rungs from `bottom` to `top`.

    Returns FEWER than `count` when the integers collide -- the window is too
    narrow to hold that many distinct scales -- and the caller widens rather
    than silently fitting a shorter ladder.
    """
    bottom, top = int(bottom), int(top)
    if count < 2 or top <= bottom:
        return [bottom]
    r = (top / bottom) ** (1.0 / (count - 1))
    return sorted({int(round(bottom * r ** k)) for k in range(count)})


def probe_window(spec, params: dict, rng, ladder,
                 repeats: int = PROBE_REPEATS,
                 aggregator: str = DEFAULT_AGGREGATOR,
                 min_scales: int = PROBE_MIN_SCALES,
                 overhead_factor: float = PROBE_OVERHEAD_FACTOR,
                 seconds_budget: float = PROBE_WINDOW_BUDGET,
                 max_doublings: int = PROBE_MAX_DOUBLINGS) -> dict:
    """Place the timing window just above the measured per-call overhead.

    THE RULE, in three steps:

      1. measure the overhead a0 (`measure_overhead`, ~0.5 ms);
      2. walk up from the sample ladder's bottom, doubling, to the first rung
         whose WORK clears the floor: t(i) - a0 >= PROBE_OVERHEAD_FACTOR * a0;
      3. take `min_scales` rungs doubling from there, and pack them closer
         together if the predicted cost does not fit `seconds_budget`.

    Why this replaces `climb_to_target` in the pilot. That strategy starts at
    the ladder's LARGEST scale and only ever goes up, with the rung floor
    outranking both of its stopping rules -- so on a model whose top rung is
    already slow it pays three unconditional doublings. Measured on
    percolation_zd dim=3 with a 2..128 ladder (user's run, 2026-09-14): the
    climb started at 128, where one call already took 40 ms, i.e. 20x the
    target it was climbing toward, and still went to 1024. Cost 585 units of
    the start scale, 280 s of a 20-minute study -- 43% of the work of the
    final run, spent on draws that are timed and discarded.

    Both ends of the window matter, and they fail differently:

      TOO LOW and the fit is not noisy but WRONG -- the measurement is mostly
      dispatch. srw over 8..64 does not converge at all; over 32..256 it
      returns 3.42 +/- 0.69 against a truth of 1.
      TOO HIGH and it measures a machine regime the study never enters. On
      percolation_zd dim=3 the per-site cost is flat at 2.03e-8 s across
      64..512 and rises 14% at i = 1024, where the working set leaves cache;
      fitting through that upturn is what returned d = 3.086 instead of 3,
      while the final run's largest box was i = 128. A bigger probe was not a
      better measurement of the run's own d.

    The rule reaches opposite conclusions for the two models in the repo, which
    is the point of measuring a0 rather than assuming a scale: percolation_zd
    dim=3 has a0 = 0.2 ms and clears the floor at i = 32, so the window comes
    DOWN from 128..1024 to 32..256 (12 s, d = 2.999 +/- 0.033). srw has
    a0 = 8.7 us and work of 3 ns per step, so it does not clear the floor until
    i = 16384 and the window goes far ABOVE the ladder, as it always did
    (0.07 s, d = 1.027 +/- 0.013). Cheap models were never the problem.

    Returns the same payload as the other two probes, plus `overhead_seconds`,
    `overhead_scale`, `window_bottom`, `predicted_seconds` and `over_budget`.
    """
    ladder = sorted(int(x) for x in ladder) or [1]
    a0, a0_scale = measure_overhead(spec, params, rng, repeats, aggregator,
                                    ceiling=ladder[-1])
    floor_t = a0 * (1.0 + float(overhead_factor))

    # Step 2: walk up to the floor. Every rung below the one we keep is
    # cheaper than it, so the whole walk costs less than the window's own
    # bottom rung -- and on a model whose ladder already sits above the floor
    # (percolation_zd) it stops on the first or second try.
    walk: dict[int, list[float]] = {}
    bottom, refused_at = None, None
    i = max(1, ladder[0])
    for _ in range(max_doublings):
        try:
            t, times = time_at_scale(spec, i, params, rng, repeats, aggregator)
        except (ValueError, MemoryError):
            refused_at = i
            break
        walk[i] = times
        if t >= floor_t:
            bottom = i
            break
        i *= 2
    below_floor = bottom is None
    if below_floor:
        # Never cleared the floor: the model refused a scale first, or ran out
        # of doublings. Use the largest rung reached -- d measured there is
        # overhead-contaminated, which `fit_cost_probe`'s overhead_share
        # reports and `_resolve_d` turns into a warning.
        if not walk:
            raise ValueError(
                f"the cost probe timed no scale at all: the model refused "
                f"{refused_at}, the smallest scale on the ladder")
        bottom = max(walk)

    # Step 3: min_scales rungs doubling up, packed closer if too expensive.
    # `cost_hint` prices the candidates exactly when the model declares one;
    # otherwise the walk's own two slowest rungs give a slope. Predicting the
    # cost is what lets the budget bind BEFORE the expensive rung is timed --
    # the old climb checked its time budget only after paying.
    predict_one = _rung_predictor(spec, params, walk, bottom, a0, aggregator)

    def predicted(rungs):
        return sum((repeats + 1) * predict_one(k) for k in rungs)

    rungs = [bottom * 2 ** k for k in range(min_scales)]
    top = rungs[-1]
    while predicted(rungs) > seconds_budget and top > bottom * 2:
        cand = _space(bottom, max(bottom * 2, int(top // 2)), min_scales)
        if len(cand) < min_scales:
            break
        rungs, top = cand, cand[-1]
    over_budget = predicted(rungs) > seconds_budget

    # Time them, reusing anything the walk already measured.
    times_by_scale = {k: walk[k] for k in rungs if k in walk}
    for k in rungs:
        if k in times_by_scale:
            continue
        try:
            _, times = time_at_scale(spec, k, params, rng, repeats, aggregator)
        except (ValueError, MemoryError):
            # A ceiling inside the window: keep what is below it and re-space
            # the rest underneath, exactly as the climb's downward fill does.
            refused_at = k
            break
        times_by_scale[k] = times

    # A ceiling INSIDE the chosen window -- the model refused a rung the budget
    # was happy to pay for. Fill downward by halving, as the climb's own
    # downward fill does: the rungs already timed are good, and halving always
    # has room. These rungs are below the overhead floor by construction (the
    # bottom was the FIRST scale above it), so the overhead share rises and
    # `_resolve_d` says so -- which is the honest outcome, since a model that
    # cannot be run where it should be timed leaves nowhere better to look.
    scales = sorted(times_by_scale)
    extended_down = []
    while len(scales) < min_scales:
        k = scales[0] // 2
        if k < 1:
            raise ValueError(
                f"the cost probe cannot reach {min_scales} rungs: the model "
                f"refused {refused_at} going up and there is nothing below "
                f"{scales[0]} left to measure. These parameters are not "
                f"simulable at any useful scale.")
        try:
            _, times = time_at_scale(spec, k, params, rng, repeats, aggregator)
        except (ValueError, MemoryError) as exc:
            raise ValueError(
                f"the cost probe cannot reach {min_scales} rungs: the model "
                f"refused {refused_at} going up and {k} going down. These "
                f"parameters are not simulable at any useful scale."
            ) from exc
        times_by_scale[k] = times
        extended_down.append(k)
        scales = sorted(times_by_scale)

    out = _probe(scales, times_by_scale, repeats, aggregator,
                 target_seconds=PROBE_TARGET_SECONDS)
    out["reached_target"] = bool(out["elapsed"] and
                                 out["elapsed"][-1] >= PROBE_TARGET_SECONDS)
    out["refused_at"] = refused_at
    out["extended_down"] = extended_down or None
    out["overhead_seconds"] = a0
    out["overhead_scale"] = a0_scale
    # The share of the cheapest rung that is dispatch, from the MEASURED
    # overhead rather than the affine fit's `a`. Both are reported and they
    # disagree: on srw the fit says 51% where the clock says 25%, and on
    # percolation_zd dim=3 the fit said 3.2 ms against a measured 0.21 ms.
    # `a` is fitted jointly with `d` over a short window and absorbs curvature
    # that belongs to the exponent, which is the whole reason this rule
    # measures the overhead instead of reading it off the fit.
    out["overhead_share_measured"] = (float(a0 / out["elapsed"][0])
                                      if out.get("elapsed") else None)
    out["overhead_factor"] = float(overhead_factor)
    out["window_bottom"] = int(bottom)
    out["below_overhead_floor"] = bool(below_floor)
    out["walk_scales"] = sorted(walk)
    out["predicted_seconds"] = float(predicted(scales))
    out["seconds_budget"] = float(seconds_budget)
    out["over_budget"] = bool(over_budget)
    return out


def _rung_predictor(spec, params: dict, walk: dict, bottom: int, a0: float,
                    aggregator: str):
    """t_hat(i) for a rung not yet timed: a0 + work(bottom) * cost(i)/cost(bottom).

    The ratio comes from the model's declared `cost_hint` when it has one --
    exact, and free. With no declaration it comes from the slope between the
    two slowest rungs the walk already timed, which is the best available
    evidence at that moment and only ever used to decide whether to pack the
    window tighter.
    """
    t_bottom = aggregate(walk[bottom], aggregator) if bottom in walk else a0
    work = max(t_bottom - a0, 1e-12)

    if spec.cost_hint is not None:
        base = float(spec.cost_hint(bottom, params)) or 1.0
        return lambda i: a0 + work * float(spec.cost_hint(int(i), params)) / base

    timed = sorted(walk)
    if len(timed) >= 2:
        lo, hi = timed[-2], timed[-1]
        w_lo = max(aggregate(walk[lo], aggregator) - a0, 1e-12)
        w_hi = max(aggregate(walk[hi], aggregator) - a0, 1e-12)
        d_guess = float(np.log(w_hi / w_lo) / np.log(hi / lo))
    else:
        d_guess = 1.0
    return lambda i: a0 + work * (float(i) / bottom) ** d_guess


def fit_cost_probe(probe: dict, cost_hint=None, params: dict | None = None) -> dict:
    """Fit d from a probe, both ways, plus the overhead diagnostic.

    Adds to `probe` (and returns it):

      d_hat            the pure power law cost(i) = c*i**d, Assumption
                       cost_is_power_law (eq. 353) taken literally
      affine           cost(i) = a + b*i**d, or {"error": ...} on too few rungs
      overhead_share   a / elapsed[0]: how much of the cheapest measurement was
                       dispatch rather than work. Above ~0.2, `d_hat` is
                       measuring the overhead and only `affine["d"]` is usable
      declared_d       what the model's own cost_hint implies, when it has one.
                       Reported for CROSS-CHECKING only -- src/study/pilot.py
                       always takes d from the affine fit and scores any
                       declaration as z = (d_hat - declared)/se(d_hat). It used
                       to be preferred over the measurement; that is exactly the
                       leak `_resolve_d` was written to close.
    """
    scales, elapsed = probe["scales"], probe["elapsed"]
    probe["d_hat"] = estimate_cost_exponent(scales, elapsed)
    try:
        probe["affine"] = estimate_cost_affine(scales, elapsed)
    except ValueError as exc:              # too few scales for 3 parameters
        probe["affine"] = {"error": str(exc)}
    a = probe["affine"].get("a")
    probe["overhead_share"] = (float(a / elapsed[0])
                               if a is not None and elapsed and elapsed[0] else None)
    if cost_hint is not None:
        probe["declared_d"] = declared_exponent(scales, cost_hint, params or {})
    return probe
