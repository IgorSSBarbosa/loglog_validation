"""Calibration testing for error bars: do our stated uncertainties cover truth?

Every result this repo reports is a pair -- an estimate and an uncertainty.
The estimate is checked against known ground truth (srw has one). The
uncertainty is ALSO a claim, and an untested one: "1.0155 +/- 0.1050" asserts
that re-running the whole experiment many times would land within 0.1050 of
truth about 68% of the time. If the true figure were 45%, every "consistent"
/ "z = -1.14" verdict built on it would be worthless.

This module measures that. `coverage_test` runs a whole experiment many times,
forms the interval each time exactly as the pipeline would, and counts how
often it contains the truth. PLAN.md checkpoint 0.4.

NOT article eq. (720). `wilson_score_interval` here is the textbook binomial
score interval, used only to put an honest CI on a measured COVERAGE (a
proportion). The article's Wilson interval of eq. (720) is a different object
-- a four-term bound on gamma-hat -- and belongs in `tools/wilson.py`, which
is still unwritten. Do not conflate them.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def wilson_score_interval(successes: int, trials: int, level: float = 0.95
                          ) -> tuple[float, float]:
    """Binomial score interval for a proportion -- for the coverage count itself.

    Score rather than Wald: a measured coverage near 1 (which is exactly where
    a well-calibrated 95% interval lives) makes Wald's sqrt(p(1-p)/N) collapse
    and produce bounds above 1. The score interval stays inside [0, 1] and
    keeps sensible width at the boundary -- the entire reason to prefer it here.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive; got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}]; got {successes}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2))
    p, n = successes / trials, trials
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def interval(estimate: float, se: float, *, level: float = 0.95,
             dof: int | None = None) -> tuple[float, float]:
    """estimate +/- q * se, with q from Student-t when `dof` is given.

    `dof` matters far more than it looks. This repo's error bars come from the
    spread of R independent replicates (se = sd/sqrt(R)) with R as small as 5,
    where the normal quantile 1.960 should be Student's 2.776 on 4 dof -- a 42%
    too-narrow interval. Passing dof=None reproduces the naive normal choice on
    purpose, so a coverage test can measure what that costs.
    """
    if se < 0:
        raise ValueError(f"se must be non-negative; got {se}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    from scipy.stats import norm, t

    if dof is None:
        q = float(norm.ppf(0.5 + level / 2))
    else:
        if dof < 1:
            raise ValueError(f"dof must be >= 1; got {dof}")
        q = float(t.ppf(0.5 + level / 2, dof))
    return estimate - q * se, estimate + q * se


def coverage_test(
    experiment: Callable[[np.random.Generator], tuple[float, float]],
    truth: float,
    *,
    trials: int = 1000,
    level: float = 0.95,
    dof: int | None = None,
    seed: int | None = None,
    progress: bool = False,
) -> dict:
    """Run `experiment` `trials` times; count how often its interval holds `truth`.

    `experiment(rng) -> (estimate, se)` must perform a COMPLETE, independent
    replay of the pipeline whose error bar is under test -- including drawing
    its own replicates. Handing it shared draws would measure the interval
    against a distribution narrower than the real one and report a coverage
    that cannot be reproduced.

    Every trial gets its own stream via SeedSequence.spawn (ground rule 2).
    Returns the measured coverage with a Wilson interval on that coverage, the
    mean interval width, and the estimator's own bias -- the last because
    undercoverage has two quite different causes, an interval that is too
    narrow and an estimator that is off-centre, and the fix differs.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1; got {trials}")
    seed_seq = np.random.SeedSequence(seed)
    streams = seed_seq.spawn(trials)

    hits = 0
    ests, widths, ses = [], [], []
    for t_i, ss in enumerate(streams):
        est, se = experiment(np.random.default_rng(ss))
        if not (math.isfinite(est) and math.isfinite(se)):
            continue
        lo, hi = interval(est, se, level=level, dof=dof)
        hits += int(lo <= truth <= hi)
        ests.append(est)
        ses.append(se)
        widths.append(hi - lo)
        if progress and (t_i + 1) % 50 == 0:
            import sys
            print(f"\r  {t_i + 1}/{trials} trials, coverage so far "
                  f"{hits / (t_i + 1):.3f}", end="", file=sys.stderr, flush=True)
    if progress:
        import sys
        print(file=sys.stderr)

    n_ok = len(ests)
    if n_ok == 0:
        raise RuntimeError("every trial returned a non-finite estimate or se")
    lo, hi = wilson_score_interval(hits, n_ok, level=level)
    ests = np.asarray(ests)
    return {
        "coverage": hits / n_ok,
        "coverage_ci": (lo, hi),
        "nominal": level,
        "calibrated": lo <= level <= hi,
        "trials": n_ok,
        "trials_requested": trials,
        "hits": hits,
        "dof": dof,
        "truth": truth,
        "mean_estimate": float(ests.mean()),
        "bias": float(ests.mean() - truth),
        "sd_of_estimates": float(ests.std(ddof=1)) if n_ok > 1 else float("nan"),
        "mean_stated_se": float(np.mean(ses)),
        "mean_width": float(np.mean(widths)),
        "seed": seed_seq.entropy,
        "values": [float(e) for e in ests],
        "ses": [float(x) for x in ses],
    }


def coverage_multi(
    experiment: Callable[[np.random.Generator], dict],
    truths: dict,
    *,
    trials: int = 1000,
    level: float = 0.95,
    dofs: Sequence = (None,),
    seed: int | None = None,
    progress: bool = False,
) -> dict:
    """`coverage_test` for many quantities scored from ONE pass of the trials.

    `experiment(rng) -> {name: (estimate, se)}`. Every name is scored against
    `truths[name]` at every quantile in `dofs`, reusing the same trial. Not
    merely a speed convenience: the different quantities are usually different
    READINGS of one fit (omega1 and a1 come out of a single least_squares
    call), so re-running the trials per quantity would both multiply the cost
    and score each quantity on a different set of draws -- leaving two results
    that cannot be compared to each other.

    Returns {f"{name} [q=...]": result}, each shaped like `coverage_test`'s.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1; got {trials}")
    seed_seq = np.random.SeedSequence(seed)
    streams = seed_seq.spawn(trials)

    rows: dict = {}
    for t_i, ss in enumerate(streams):
        got = experiment(np.random.default_rng(ss))
        for name, (est, se) in got.items():
            rows.setdefault(name, []).append((est, se))
        if progress and (t_i + 1) % 25 == 0:
            import sys
            print(f"\r  {t_i + 1}/{trials} trials", end="", file=sys.stderr,
                  flush=True)
    if progress:
        import sys
        print(file=sys.stderr)

    out = {}
    for name, pairs in rows.items():
        if name not in truths:
            raise KeyError(f"no truth given for {name!r}")
        truth = truths[name]
        good = [(e, s) for e, s in pairs if math.isfinite(e) and math.isfinite(s)]
        if not good:
            raise RuntimeError(f"every trial for {name!r} was non-finite")
        ests = np.array([e for e, _ in good])
        ses = np.array([s for _, s in good])
        for dof in dofs:
            hits, widths = 0, []
            for e, s in good:
                lo, hi = interval(e, s, level=level, dof=dof)
                hits += int(lo <= truth <= hi)
                widths.append(hi - lo)
            clo, chi = wilson_score_interval(hits, len(good), level=level)
            q = "normal" if dof is None else f"t({dof})"
            out[f"{name} [q={q}]"] = {
                "coverage": hits / len(good),
                "coverage_ci": (clo, chi),
                "nominal": level,
                "calibrated": clo <= level <= chi,
                "trials": len(good),
                "trials_requested": trials,
                "hits": hits,
                "dof": dof,
                "truth": truth,
                "mean_estimate": float(ests.mean()),
                "bias": float(ests.mean() - truth),
                "sd_of_estimates": float(ests.std(ddof=1)) if len(good) > 1
                else float("nan"),
                "mean_stated_se": float(ses.mean()),
                "mean_width": float(np.mean(widths)),
                "seed": seed_seq.entropy,
                "values": [float(e) for e in ests],
                "ses": [float(x) for x in ses],
            }
    return out


def combine_se(components: Sequence[tuple]) -> tuple[float, float]:
    """Add independent error components in quadrature, tracking effective dof.

    `components` is a sequence of (se, dof) pairs; use dof=None (or a large
    number) for an se that is known rather than estimated from a handful of
    replicates. Returns (combined_se, effective_dof) via Welch-Satterthwaite:

        dof_eff = (sum se_j^2)^2 / sum (se_j^4 / dof_j)

    Why this is needed rather than just hypot: a z-score built from an se that
    was itself estimated from R = 5 replicates is not standard normal, so the
    familiar "|z| < 2 means consistent" is wrong -- the honest threshold is
    t(dof_eff), which at 4 dof is 2.776, not 1.960. When one component is
    known and the other comes from 5 replicates, dof_eff interpolates between
    the two according to how much each contributes, which is exactly the
    question "how much does the noisy component dominate here".
    """
    if not components:
        raise ValueError("need at least one (se, dof) component")
    var = 0.0
    denom = 0.0
    for se, dof in components:
        if se is None:
            continue
        if se < 0:
            raise ValueError(f"se must be non-negative; got {se}")
        v = float(se) ** 2
        var += v
        if dof is not None and dof > 0:
            denom += v * v / float(dof)
    if var == 0:
        return 0.0, float("inf")
    dof_eff = float("inf") if denom == 0 else var * var / denom
    return math.sqrt(var), dof_eff


def consistency_threshold(dof: float, level: float = 0.95) -> float:
    """The |z| cut-off that really corresponds to `level`, given `dof`.

    1.960 only when dof is infinite. At the R = 5 replicates this repo uses it
    is 2.776 -- so a discrepancy reported as "2.5 sigma, DISCREPANT" on 4 dof
    is in fact consistent at 95%.
    """
    from scipy.stats import norm, t

    if not math.isfinite(dof):
        return float(norm.ppf(0.5 + level / 2))
    if dof < 1:
        raise ValueError(f"dof must be >= 1; got {dof}")
    return float(t.ppf(0.5 + level / 2, dof))


def rescore(result: dict, *, level: float | None = None,
            dof: int | None = ...) -> dict:
    """Re-evaluate a finished coverage result at a different level or quantile.

    The expensive part of a coverage test is the trials, not the counting, and
    the trials do not depend on either choice -- so re-running the pipeline to
    ask "and what about +/- 1 SE?" would waste hours and, worse, answer on a
    different set of draws. This re-scores the stored per-trial (estimate, se)
    pairs instead, so every level and quantile is compared on identical data.

    Pass `dof=None` explicitly to force the normal quantile; omitting `dof`
    keeps whatever the original result used (which is why the default is the
    Ellipsis sentinel, not None -- None is a meaningful value here).
    """
    if "values" not in result or "ses" not in result:
        raise KeyError("result has no stored per-trial values; it predates rescore()")
    lvl = result["nominal"] if level is None else level
    d = result["dof"] if dof is Ellipsis else dof
    truth = result["truth"]

    hits, widths = 0, []
    for e, se in zip(result["values"], result["ses"]):
        lo, hi = interval(e, se, level=lvl, dof=d)
        hits += int(lo <= truth <= hi)
        widths.append(hi - lo)
    n = len(result["values"])
    clo, chi = wilson_score_interval(hits, n, level=0.95)
    out = dict(result)
    out.update({
        "coverage": hits / n,
        "coverage_ci": (clo, chi),
        "nominal": lvl,
        "calibrated": clo <= lvl <= chi,
        "hits": hits,
        "dof": d,
        "mean_width": float(np.mean(widths)),
    })
    return out


def se_ratio(result: dict) -> float:
    """mean stated se / actual sd of the estimates.

    The direct diagnosis of a width problem: 1.0 means the pipeline's error bar
    is the right size on average, < 1 means it systematically understates its
    own scatter. Separates "interval too narrow" from "estimator biased", which
    `coverage_test` reports alongside as `bias`.
    """
    sd = result["sd_of_estimates"]
    if not math.isfinite(sd) or sd == 0:
        return float("nan")
    return result["mean_stated_se"] / sd


def format_result(result: dict, name: str = "") -> str:
    """One-block human summary, with the verdict spelled out."""
    lo, hi = result["coverage_ci"]
    verdict = "CALIBRATED" if result["calibrated"] else (
        "UNDERCOVERS" if result["coverage"] < result["nominal"] else "OVERCOVERS")
    q = "normal" if result["dof"] is None else f"t({result['dof']})"
    return (
        f"{name or 'coverage test'}: {verdict}\n"
        f"  nominal {result['nominal']:.0%} using the {q} quantile\n"
        f"  measured {result['coverage']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
        f"({result['hits']}/{result['trials']} trials)\n"
        f"  truth {result['truth']:+.4f}   mean estimate {result['mean_estimate']:+.4f}"
        f"   bias {result['bias']:+.4f}\n"
        f"  stated se {result['mean_stated_se']:.4f} vs actual sd "
        f"{result['sd_of_estimates']:.4f}   ratio {se_ratio(result):.3f}\n"
        f"  mean interval width {result['mean_width']:.4f}"
    )
