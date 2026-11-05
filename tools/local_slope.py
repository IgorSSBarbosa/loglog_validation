"""Consecutive-point slopes s_k and how they approach their limit (plans/local_slope.md).

    s_k = (log Ybar_{rho^{k+1}} - log Ybar_{rho^k}) / log rho

is the article's gamma-hat with m = 2 and m0 = k (eqs. 523-526: with m = 2 the
weights of eq. (526) are -1, +1). `tools.loglog.gamma_two_point` computes the same
number from one mean per scale; this module works from R independent replicate
means per scale instead, because the question here -- power law, logarithm, or a
crossover -- is read off DIFFERENCES of s_k, and those need a standard error.

Prop. (820) puts the bias of s_k at rho^{-omega_1 k}, i.e. i^{-omega_1} in the scale
i = rho^k, when Assumption 1 (eq. 232) holds. The truth-free diagnostic is the local
exponent of the successive differences Delta s_k = s_{k+1} - s_k,

    w_k = -log(Delta s_{k+1} / Delta s_k) / log rho,

flat at omega_1 under a power law and drifting to 0 under s - gamma ~ c / ln i
(Delta s_k ~ 1/k^2 there). Two fits are reported side by side, compared by chi2 and
p-value only: this module never declares a winner.

No I/O and no knowledge of any model: `src/study/local_slope.py` reads the study,
draws what is missing and calls `analyse`.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

#: |Delta s_k| must exceed this many se for the point to be read (plan section 4.3).
RESOLVED_Z = 2.0


def merge_means(n1: int, y1: float, cv1: float,
                n2: int, y2: float, cv2: float) -> tuple[int, float, float]:
    """(n, y_bar, cv) of the union of two independent samples, from their summaries alone.

    Exact, not approximate: the mean is the n-weighted mean, and the pooled sample
    variance (ddof = 1, as tools/summary.py computes it) is recovered from each part's
    sd = cv * y_bar plus the between-part term n1 n2 / n (y1 - y2)^2.
    """
    n = n1 + n2
    y = (n1 * y1 + n2 * y2) / n
    ss = ((n1 - 1) * (cv1 * y1) ** 2 + (n2 - 1) * (cv2 * y2) ** 2
          + n1 * n2 / n * (y1 - y2) ** 2)
    return n, float(y), float(np.sqrt(ss / (n - 1)) / y)


def scale_moments(y_bars) -> tuple[float, float, int]:
    """(mean of log Ybar_r, sd_r / sqrt(R), R) over one scale's replicate means."""
    v = np.log(np.asarray(y_bars, dtype=float))
    R = v.size
    if R < 2:
        raise ValueError(f"a scale needs >= 2 replicate means for an se; got {R}")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(R)), R


def _check_grid(scales, rho: float) -> np.ndarray:
    i = np.asarray(scales, dtype=float)
    ratio = i[1:] / i[:-1]
    if i.size < 2 or not np.allclose(ratio, rho, rtol=1e-9):
        raise ValueError(f"scales must be consecutive rho^k (rho={rho}); got {list(scales)}")
    return i


def _propagate(J: np.ndarray, se: np.ndarray) -> np.ndarray:
    """sd of J @ L for independent L_j with sd se_j: scales are independent draws."""
    return np.sqrt((J ** 2) @ (se ** 2))


def local_slopes(scales, log_y, se_log, rho: float) -> dict:
    """s_k, Delta s_k and w_k, each with its se and (for the last two) resolved flag.

    Errors are propagated from the per-scale se of mean log Ybar through the exact
    linear (s, Delta s) or linearized (w) map. Scales are independent draws, so no
    replicate pairing enters (plan section 4.2); Delta s_k and w_k reuse a scale in
    more than one term, which the Jacobian accounts for.
    """
    i = _check_grid(scales, rho)
    L, se = np.asarray(log_y, float), np.asarray(se_log, float)
    K, lr = L.size, np.log(rho)

    D1 = (np.eye(K, k=1) - np.eye(K))[:-1] / lr          # s = D1 @ L
    s, se_s = D1 @ L, _propagate(D1, se)

    D2 = D1[1:] - D1[:-1]                                 # Delta s = D2 @ L
    ds, se_ds = D2 @ L, _propagate(D2, se)
    ds_ok = np.abs(ds) > RESOLVED_Z * se_ds

    w = np.full(max(K - 3, 0), np.nan)
    se_w = np.full_like(w, np.nan)
    for k in range(w.size):
        r = ds[k + 1] / ds[k]
        if r > 0:
            w[k] = -np.log(r) / lr
            # d w / d L = -(1/lr) (D2[k+1]/ds[k+1] - D2[k]/ds[k])
            g = -(D2[k + 1] / ds[k + 1] - D2[k] / ds[k]) / lr
            se_w[k] = float(np.sqrt((g ** 2) @ (se ** 2)))
    w_ok = ds_ok[:-1] & ds_ok[1:] & np.isfinite(w)

    return {"scales": i.astype(int).tolist(),
            "s": s.tolist(), "se_s": se_s.tolist(),
            "ds": ds.tolist(), "se_ds": se_ds.tolist(), "ds_resolved": ds_ok.tolist(),
            "w": [None if not np.isfinite(v) else float(v) for v in w],
            "se_w": [None if not np.isfinite(v) else float(v) for v in se_w],
            "w_resolved": w_ok.tolist()}


def clt_se(cv, n, R, rho: float) -> np.ndarray:
    """se(s_k) the CLT predicts, eq. (583) at m = 2 with the exact weights of eq. (526).

    thm:CLT's 12/(m(m^2 - 1)) = sum_k w_k^2 is 2 at m = 2; per scale the variance of log Ybar is cv^2/n (delta method, sigma_inf^2
    read as cv^2), and mean log Ybar over R replicates divides it by R.
    """
    v = np.asarray(cv, float) ** 2 / (np.asarray(n, float) * np.asarray(R, float))
    return np.sqrt(v[:-1] + v[1:]) / np.log(rho)


def n_mismatch(cv, n, rho: float) -> np.ndarray:
    """The spurious term n-mismatch adds to s_k (plan section 3), per consecutive pair.

    E log Ybar ~ log E Y - cv^2/(2n), so unequal n on the two scales of a pair shifts
    s_k by cv^2/(2 log rho) (1/n_k - 1/n_{k+1}); cv here is the pair's mean cv.
    Identically 0 when n is equal at every scale, which is what this checks.
    """
    cv, n = np.asarray(cv, float), np.asarray(n, float)
    c2 = ((cv[:-1] + cv[1:]) / 2) ** 2
    return c2 / (2 * np.log(rho)) * (1 / n[:-1] - 1 / n[1:])


def _chi2_summary(r: np.ndarray, n_par: int) -> dict:
    chi2 = float(r @ r)
    dof = int(r.size - n_par)
    return {"chi2": chi2, "dof": dof,
            "p_value": float(stats.chi2.sf(chi2, dof)) if dof > 0 else None}


def fit_log(scales, s, se) -> dict:
    """(L) s = gamma + c / ln i, weighted least squares. Linear, so solved exactly."""
    i, s, se = (np.asarray(v, float) for v in (scales, s, se))
    A = np.vstack([np.ones_like(i), 1 / np.log(i)]).T / se[:, None]
    b = s / se
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    return {"model": "gamma + c / ln i", "gamma": float(coef[0]), "c": float(coef[1]),
            "se_gamma": float(np.sqrt(cov[0, 0])), "se_c": float(np.sqrt(cov[1, 1])),
            **_chi2_summary(A @ coef - b, 2)}


def fit_power(scales, s, se, omega_bounds=(1e-3, 10.0)) -> dict:
    """(P) s = gamma + c i^-omega, weighted least squares.

    gamma and c are linear given omega, so omega is found by profiling the chi2 over
    a log grid and refining the best cell; curve_fit then supplies the covariance
    from that start. An omega on a bound is reported, not hidden: it means the data
    do not bracket one.
    """
    i, s, se = (np.asarray(v, float) for v in (scales, s, se))

    def profile(om: float) -> tuple[float, np.ndarray]:
        A = np.vstack([np.ones_like(i), i ** -om]).T / se[:, None]
        coef, *_ = np.linalg.lstsq(A, s / se, rcond=None)
        r = A @ coef - s / se
        return float(r @ r), coef

    grid = np.geomspace(*omega_bounds, 400)
    chi = np.array([profile(om)[0] for om in grid])
    j = int(np.argmin(chi))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    om = float(optimize.minimize_scalar(lambda o: profile(o)[0], bounds=(lo, hi),
                                        method="bounded", options={"xatol": 1e-10}).x)
    _, (g, c) = profile(om)
    out = {"model": "gamma + c i^-omega", "gamma": float(g), "c": float(c), "omega": om,
           "omega_on_bound": bool(np.isclose(om, omega_bounds[0], rtol=1e-3)
                                  or np.isclose(om, omega_bounds[1], rtol=1e-3))}
    if i.size > 3:
        popt, pcov = optimize.curve_fit(lambda x, g_, c_, o_: g_ + c_ * x ** -o_,
                                        i, s, p0=[g, c, om], sigma=se,
                                        absolute_sigma=True, maxfev=20000)
        out.update(gamma=float(popt[0]), c=float(popt[1]), omega=float(popt[2]))
        err = np.sqrt(np.diag(pcov))
        out.update(se_gamma=float(err[0]), se_c=float(err[1]), se_omega=float(err[2]))
    r = (out["gamma"] + out["c"] * i ** -out["omega"] - s) / se
    return {**out, **_chi2_summary(r, 3)}


def analyse(scales, y_bars_by_scale, rho: float, *, cv=None, n=None) -> dict:
    """Everything of plan section 4 from R_k replicate means per scale.

    `y_bars_by_scale[k]` holds scale k's replicate means (R may differ by scale).
    `cv` and `n` (per scale) are optional and only feed the CLT cross-check and the
    n-mismatch term.
    """
    mom = [scale_moments(y) for y in y_bars_by_scale]
    L = np.array([m[0] for m in mom])
    se = np.array([m[1] for m in mom])
    R = np.array([m[2] for m in mom])
    out = {"rho": float(rho), "log_y": L.tolist(), "se_log_y": se.tolist(),
           "R": R.tolist(), **local_slopes(scales, L, se, rho)}
    lower = np.asarray(scales, float)[:-1]        # s_k is placed at its lower scale rho^k
    s, se_s = np.array(out["s"]), np.array(out["se_s"])
    out["fits"] = {"power": fit_power(lower, s, se_s) if s.size >= 3 else None,
                   "log": fit_log(lower, s, se_s) if s.size >= 2 else None}
    if cv is not None and n is not None:
        out["se_s_clt"] = clt_se(cv, n, R, rho).tolist()
        out["n_mismatch"] = n_mismatch(cv, n, rho).tolist()
    return out


def _window_fit(x: np.ndarray, L: np.ndarray, se: np.ndarray, rho: float,
                omega_bounds: tuple[float, float]) -> dict:
    """Weighted LS of L = a + b x + c rho^(-omega x) on one window (x = 0, 1, ...).

    (a, b, c) are linear given omega, so omega is profiled over a log grid, refined
    in the best cell, and its 68% interval read off the profile at chi2_min + 1. The
    profile, not curve_fit's linearized se, is the error bar: on 5 points the chi2 in
    omega is far from quadratic.
    """
    def profile(om: float) -> tuple[float, np.ndarray]:
        A = np.vstack([np.ones_like(x), x, rho ** (-om * x)]).T / se[:, None]
        coef, *_ = np.linalg.lstsq(A, L / se, rcond=None)
        r = A @ coef - L / se
        return float(r @ r), coef

    grid = np.geomspace(*omega_bounds, 600)
    chi = np.array([profile(om)[0] for om in grid])
    j = int(np.argmin(chi))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    om = float(optimize.minimize_scalar(lambda o: profile(o)[0], bounds=(lo, hi),
                                        method="bounded", options={"xatol": 1e-12}).x)
    chi2, (a, b, c) = profile(om)
    excess = lambda o: profile(o)[0] - chi2 - 1.0   # noqa: E731

    def edge(side: np.ndarray, bound: float) -> tuple[float, bool]:
        # the nearest grid point outside the interval brackets the crossing; none means
        # the profile never rises by 1 on that side, so the interval is open there
        out = side[chi[np.isin(grid, side)] > chi2 + 1.0]
        if out.size == 0:
            return bound, True
        far = out.max() if bound < om else out.min()
        return float(optimize.brentq(excess, *sorted((far, om)), xtol=1e-12)), False

    lo_e, lo_open = edge(grid[grid < om], omega_bounds[0])
    hi_e, hi_open = edge(grid[grid > om], omega_bounds[1])
    on_bound = bool(np.isclose(om, omega_bounds[0], rtol=1e-3)
                    or np.isclose(om, omega_bounds[1], rtol=1e-3))
    return {"omega": om, "a": float(a), "b": float(b), "c": float(c),
            "omega_lo": lo_e, "omega_hi": hi_e, "interval_open": lo_open or hi_open,
            "omega_on_bound": on_bound, "chi2": chi2, "dof": int(x.size - 4),
            "p_value": float(stats.chi2.sf(chi2, x.size - 4)) if x.size > 4 else None}


def omega_windows(scales, log_y, se_log, rho: float, width: int = 5,
                  omega_bounds=(1e-2, 10.0)) -> list[dict]:
    """omega from a + b k + c rho^(-omega k) fitted on each run of `width` consecutive
    points of the log-log plot (user, 2026-11-04): m - width + 1 estimates.

    By eq. (232), log E Y_i = log a0 + gamma log i + a1 i^-omega1 + ..., which on the
    grid i = rho^k is exactly this model at J = 1, so omega reads omega1, b / log rho
    reads gamma and c rho^(omega k0) reads a1 (k0 = log_rho of the window's first
    scale). At width = 4 the fit is an interpolation and omega equals w_k of
    `local_slopes` identically; width = 5 leaves one dof, hence a chi2.
    """
    i = _check_grid(scales, rho)
    L, se = np.asarray(log_y, float), np.asarray(se_log, float)
    if width < 4:
        raise ValueError(f"the model has 4 parameters; width must be >= 4, got {width}")
    out = []
    x = np.arange(width, dtype=float)
    for k in range(L.size - width + 1):
        f = _window_fit(x, L[k:k + width], se[k:k + width], rho, omega_bounds)
        f.update(scales=i[k:k + width].astype(int).tolist(),
                 gamma=f["b"] / np.log(rho),
                 a1=f["c"] * float(i[k]) ** f["omega"])
        out.append(f)
    return out
