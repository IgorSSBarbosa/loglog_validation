"""The article's Wilson confidence interval, Theorem cor:wilson (eq:wilson).

A three-term bound on |beta_hat - beta| with beta = gamma*log(rho):

    |beta_hat - beta| <= Phi^-1(1 - alpha/2)*sigma_se + B_fs + B_good

    sigma_se = sqrt(12*sigma_inf2/(n*m*(m^2 - 1)))       standard error
    B_fs     = 6/(m(m+1)) * [ a1*rho^(-omega1*m0)/(rho^omega1 - 1)
                              + phi_plus*rho^(-omega2*m0)/(rho^omega2 - 1) ]
                                                          bound on |R(m,m0)|, eq:biasbound
    B_good   = 6*sigma_max2 / (n*(m+1))                   Taylor remainder, bound on
                                                          E[|Q_{n,m}| 1_G], lem:Q-prob (b)

Article revision of 2026-11-04 (user): the logarithmic-moment assumption
(old Assumption 5, E|log xi|^(2+delta) <= L) was dropped, and with it the
B_bad term and its Lambda/delta inputs. B_good lost its (c0+2) factor, and
sigma_se now carries the exact m(m^2-1) of Theorem thm:CLT instead of m^3.
Old answer.json files with B_bad/Lambda keys predate this.

GAMMA ONLY. The theorem is a statement about beta_hat, the article's linear
estimator sum_k w_{k,m} log Ybar_{rho^k} (eq. 523-526), and nothing else. It
does NOT cover omega1 or a1: those come from `tools/correction.py`'s nonlinear
four-parameter fit, which the article does not analyse. Do not reach for this
function to put an interval on them -- use replicates (with the t quantile, see
tools/coverage.py) or a bootstrap.

Why this exists, given that replicate intervals already work. The replicate
interval estimates its width from R = 5 numbers, which forces the Student
quantile 2.776 instead of 1.960 and makes every interval 42% wider than
necessary (see experiments/01_srw/README.md, and derivations/
allocation_constant_and_coverage.tex Part II). This bound's standard error is a
CLOSED FORM in sigma_inf2, estimable from the raw samples -- 1.7e8 of them at
k = 256, so its relative error is ~1/sqrt(2n) ~ 5e-5. At that precision
sigma_se is known rather than estimated, and Phi(alpha) = 1.960 is legitimate.

What you pay for it: this is a BOUND, not an interval with exact coverage. It
adds |bias| bounds rather than recentring, so it overcovers -- by how much is a
measurement, not a guess (`calibration/check_coverage.py --arm wilson`).

Validated against measurement (planted arm, Experiment B's grid, m0 = 2):
sigma_se predicts the closed-form estimator's scatter to 1.1% (4.313e-4 against
4.267e-4 measured), and B_fs bounds its true bias by a factor 1.60 (1.288e-2
against 8.069e-3) -- correctly conservative, as a bound must be.

UNIFORM n. The theorem assumes the same n at every scale, as does def:alloc.
Experiment B deliberately does not (snr allocation spans 1.7e5 to 1.7e8), and
substituting a mean n is badly wrong there -- it gives 4.2e-5 against the
correct 4.3e-4, a factor of ten. `sigma_se_per_scale` generalises the standard
error for that case; the two bias terms would need rederiving, so this module
refuses rather than guessing (pass `se_override` yourself).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tools.loglog import closed_form_weights


def sigma_se(n: float, m: int, rho: float, sigma_inf2: float,
             *, for_gamma: bool = True) -> float:
    """The standard error of eq:wilson, sqrt(12*sigma_inf2/(n*m*(m^2 - 1))).

    `sigma_inf2` is Var(xi_k) with xi_k = Y_{rho^k}/E Y_{rho^k}, i.e. the
    squared coefficient of variation (its limit is Assumption
    sigma_converges). For srw's |S_k| the limit is pi/2 - 1 = 0.5708.

    With `for_gamma` the result is divided by log(rho), converting from
    beta = gamma*log(rho) to gamma itself.
    """
    if n <= 0 or m < 2 or rho <= 1 or sigma_inf2 < 0:
        raise ValueError(f"bad input: n={n}, m={m}, rho={rho}, sigma_inf2={sigma_inf2}")
    se = math.sqrt(12.0 * sigma_inf2 / (n * m * (m**2 - 1)))
    return se / math.log(rho) if for_gamma else se


def sigma_se_per_scale(n_per_scale: Sequence, m: int, rho: float,
                       cv2_per_scale: Sequence, *, for_gamma: bool = True) -> float:
    """The standard error with per-scale n, sqrt(sum_j w_j^2 cv2_j / n_j).

    NOT in the article: eq:wilson assumes uniform n. This is the exact variance
    of the same linear estimator when n differs across scales, which is what
    Experiment B's snr allocation produces.

    At uniform n and uniform cv2 it equals `sigma_se` exactly, since
    sum_j w_j^2 = 12/(m(m^2-1)). (Before the 2026-11-04 article revision the
    theorem carried the asymptotic 12/m^3 and the two differed by
    sqrt(m^2/(m^2-1)); that gap is gone.)

    Only the VARIANCE generalises this easily. The bias terms of eq:wilson are
    stated for uniform n and are not adapted here.
    """
    n = np.asarray(n_per_scale, dtype=float)
    cv2 = np.asarray(cv2_per_scale, dtype=float)
    if n.size != m or cv2.size != m:
        raise ValueError(f"expected {m} entries; got n:{n.size}, cv2:{cv2.size}")
    if np.any(n <= 0) or np.any(cv2 < 0):
        raise ValueError("n must be positive and cv2 non-negative")
    w = closed_form_weights(m)          # eq. (526), one definition (tools/loglog.py)
    se = math.sqrt(float(np.sum(w**2 * cv2 / n)))
    return se / math.log(rho) if for_gamma else se


def sigma_se_ols(scales: Sequence, n_per_scale: Sequence,
                 cv2_per_scale: Sequence) -> float:
    """Finite-m standard error of the log-log OLS slope on ANY grid of scales.

    NOT in the article, which states the estimator on a consecutive rho^k grid.
    Published finite-size-scaling protocols use grids like L = 8, 12, 16, 24, ...
    (experiments 42, 43), so the slope is the plain OLS slope of log Ybar_j on
    x_j = log L_j, gamma_hat = sum_j v_j log Ybar_j with
    v_j = (x_j - xbar) / sum_k (x_k - xbar)^2, and to first order
    Var log Ybar_j = cv_j^2 / n_j, so

        se = sqrt(sum_j v_j^2 cv2_j / n_j).

    On a rho^k grid v_j = w_{j,m} / log(rho), so this equals
    `sigma_se_per_scale(..., for_gamma=True)` exactly (tested). The result is
    already in units of gamma; there is no rho to divide by.
    """
    x = np.log(np.asarray(scales, dtype=float))
    n = np.asarray(n_per_scale, dtype=float)
    cv2 = np.asarray(cv2_per_scale, dtype=float)
    if not (x.size == n.size == cv2.size):
        raise ValueError(f"length mismatch: scales {x.size}, n {n.size}, cv2 {cv2.size}")
    if x.size < 2:
        raise ValueError("need at least 2 scales for a slope")
    if np.any(n <= 0) or np.any(cv2 < 0):
        raise ValueError("n must be positive and cv2 non-negative")
    dx = x - x.mean()
    v = dx / float(np.sum(dx**2))
    return math.sqrt(float(np.sum(v**2 * cv2 / n)))


def finite_size_bias(m: int, m0: int, rho: float, a1: float, omega1: float,
                     *, omega2: float | None = None, phi_plus: float = 0.0,
                     for_gamma: bool = True) -> float:
    """B_fs, the finite-size term of eq:wilson (the bound eq:biasbound on |R(m,m0)|).

    The omega2 piece is dropped when `omega2` is None or `phi_plus` is 0. That
    makes the result no longer a bound on the full expansion -- callers get
    told, loudly, via `wilson_interval`'s `complete` flag. It is dropped by
    default because phi_plus is an assumption-level constant (Assumption
    phi_is_unif_bounded) that we have not measured, and inventing a value would
    be worse than declaring the gap.
    """
    if m < 1 or rho <= 1 or omega1 <= 0:
        raise ValueError(f"bad input: m={m}, rho={rho}, omega1={omega1}")
    if m0 < 0:
        raise ValueError(f"m0 must be >= 0; got {m0}")
    lead = 6.0 / (m * (m + 1))
    total = abs(a1) * rho ** (-omega1 * m0) / (rho**omega1 - 1)
    if omega2 is not None and phi_plus:
        if omega2 <= omega1:
            raise ValueError(f"omega2 must exceed omega1; got {omega2} <= {omega1}")
        total += abs(phi_plus) * rho ** (-omega2 * m0) / (rho**omega2 - 1)
    b = lead * total
    return b / math.log(rho) if for_gamma else b


def good_event_bias(n: float, m: int, sigma_max2: float, rho: float,
                    *, for_gamma: bool = True) -> float:
    """B_good = 6 sigma_max2 / (n(m+1)), the Taylor-remainder term of eq:wilson.

    Lemma lem:Q-prob (b): E[|Q_{n,m}| 1_G] <= 6 sigma_max2/(n(m+1)) on the good
    event G where every sample mean is >= 1/2 of its expectation. The article
    notes it is of smaller order than the standard error and keeps it as a
    finite-n heuristic.
    """
    if n <= 0 or m < 1 or sigma_max2 < 0:
        raise ValueError(f"bad input: n={n}, m={m}, sigma_max2={sigma_max2}")
    b = 6.0 * sigma_max2 / (n * (m + 1))
    return b / math.log(rho) if for_gamma else b


def moment_bounds(samples_by_scale: dict) -> dict:
    """Estimate sigma_inf2 and sigma_max2 from real samples.

    `samples_by_scale` is {scale: array}, the shape `tools/persistence.py`'s
    `load_samples` returns. With xi = Y/E Y (E Y estimated by the sample mean):

        sigma_k^2 = Var(xi_k),   sigma_max2 = max_k sigma_k^2,
        sigma_inf2 = sigma_k^2 at the LARGEST scale (the limit Assumption
                     sigma_converges asserts exists; using the largest
                     available k is the honest finite-sample stand-in).

    Returns the pieces plus per-scale detail, since that convergence is worth
    seeing rather than assuming.
    """
    if not samples_by_scale:
        raise ValueError("no samples given")
    rows = []
    for k in sorted(samples_by_scale, key=float):
        y = np.asarray(samples_by_scale[k], dtype=float)
        mu = float(y.mean())
        if mu <= 0:
            raise ValueError(f"non-positive mean at scale {k}: {mu}")
        xi = y / mu
        rows.append({
            "scale": float(k),
            "sigma2": float(np.var(xi, ddof=1)),
            "n": int(y.size),
            "zero_fraction": float(np.mean(xi <= 0)),
        })
    return {
        "sigma_inf2": rows[-1]["sigma2"],
        "sigma_max2": max(r["sigma2"] for r in rows),
        "per_scale": rows,
    }


def wilson_interval(gamma_hat: float, n: float, m: int, m0: int, rho: float,
                    *, sigma_inf2: float, sigma_max2: float,
                    a1: float, omega1: float,
                    omega2: float | None = None, phi_plus: float = 0.0,
                    level: float = 0.95,
                    se_override: float | None = None) -> dict:
    """Theorem cor:wilson (eq:wilson) evaluated for gamma.

    Returns the three terms, the total half-width, the interval, and a
    `complete` flag that is False whenever a term was omitted for want of a
    constant. Read that flag: a bound missing a term is not a bound, and the
    half-width is then a LOWER estimate of the true one.

    `se_override` substitutes a standard error computed elsewhere -- the reason
    it exists is `sigma_se_per_scale`, for the non-uniform-n case eq:wilson
    does not cover.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    from scipy.stats import norm

    b_fs = finite_size_bias(m, m0, rho, a1, omega1,
                            omega2=omega2, phi_plus=phi_plus)
    b_good = good_event_bias(n, m, sigma_max2, rho)
    se = se_override if se_override is not None else sigma_se(n, m, rho, sigma_inf2)

    q = float(norm.ppf(0.5 + level / 2))
    half = b_fs + b_good + q * se

    missing = []
    if omega2 is None or not phi_plus:
        missing.append("the omega2 piece of B_fs (needs phi_plus)")

    return {
        "gamma_hat": float(gamma_hat),
        "half_width": float(half),
        "interval": (float(gamma_hat - half), float(gamma_hat + half)),
        "B_fs": float(b_fs),
        "B_good": float(b_good),
        "sigma_se": float(se),
        "quantile": q,
        "se_term": float(q * se),
        "level": level,
        "complete": not missing,
        "missing_terms": missing,
        "dominant": max(
            [("B_fs", b_fs), ("B_good", b_good), ("se_term", q * se)],
            key=lambda kv: kv[1])[0],
    }


def format_interval(res: dict) -> str:
    """Human-readable breakdown, with the incompleteness warning up front."""
    lines = []
    if not res["complete"]:
        lines.append("INCOMPLETE BOUND -- omitted: " + "; ".join(res["missing_terms"]))
        lines.append("  the half-width below is a LOWER estimate of the true bound.")
    lo, hi = res["interval"]
    lines.append(f"gamma_hat = {res['gamma_hat']:.6f}   "
                 f"{res['level']:.0%} bound [{lo:.6f}, {hi:.6f}]  "
                 f"(+/- {res['half_width']:.3e})")
    total = res["half_width"]
    for name in ("B_fs", "B_good", "se_term"):
        v = res[name]
        mark = "  <-- dominant" if name == res["dominant"] else ""
        lines.append(f"    {name:<8} {v:>11.4e}  {100 * v / total:>5.1f}%{mark}")
    return "\n".join(lines)
