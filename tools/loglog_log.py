"""The loglog-log plot: omega_1 from how the local slopes approach gamma-hat.

Method (user, 2026-10-16). Take the consecutive slopes of the log-log plot,

    gamma_k = (log Ybar_{rho^{k+1}} - log Ybar_{rho^k}) / log rho

(`tools/local_slope`'s s_k: the article's gamma-hat at m = 2, eqs. 523-526), and a
"best" gamma-hat. Under Assumption 1 (eq. 232) with J = 1,

    gamma_k - gamma = a_1 (rho^{-omega_1} - 1) / log rho * rho^{-omega_1 k}

exactly (Prop. (820)'s bias order, without the O(.)), so log|gamma_k - gamma-hat| against
log rho^k is a straight line of slope -omega_1. Its fitted slope is the estimator.

This module only fits. Which data give gamma_k and which give gamma-hat is the
driver's choice: `src/estimate/omega_loglog.py` (general case: gamma_k from the pilot,
gamma-hat from the independent final run) and
`experiments/19_rwre_lazy_omega/omega_loglog_top.py` (pilot only: gamma-hat is the top
pair's slope, so it shares a scale with the last gamma_k).

Errors (user, 2026-10-16: delta method). Per scale, mean log Ybar over R replicates has
variance cv^2 / (n R), as in `local_slope.clt_se`, and scales are independent draws.
D_k = gamma_k - gamma-hat is linear in those means (plus an independent gamma-hat when it
is external), so Cov(D) is exact; log|D_k| is then linearized. Every D_k shares
gamma-hat and neighbouring gamma_k share a scale, so the line is a GLS on that full
covariance rather than a diagonal WLS, which would understate se(omega).

Points (user, 2026-10-16): only resolved ones, |D_k| > RESOLVED_Z sd(D_k), with the sign
most resolved points have. Near gamma-hat, log|D_k| is the log of noise. RESOLVED_Z was 2
at first; at exp 19 p = 1/2 one of ~12 points whose true D is ~0 passed 2 sd by chance
(~43% odds for that many) and pulled omega from ~1 to 0.53. The user raised it to 3
(2026-10-16) rather than add a contiguity rule: no rules tuned to get the right answer. The cut is on
the data, so it biases |D| upward where D is barely resolved; the fitted points and
their z are recorded so that can be judged.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: |D_k| must exceed this many sd(D_k) for the point to enter the fit.
RESOLVED_Z = 3.0
#: A line with dof >= 1.
MIN_POINTS = 3


def ladder_ratio(scales) -> float:
    """rho of a geometric ladder of >= 3 rungs; raises if it isn't one."""
    i = np.asarray(scales, float)
    ratio = i[1:] / i[:-1]
    if i.size < 3 or not np.allclose(ratio, ratio[0], rtol=1e-9):
        raise ValueError(f"scales must be a geometric ladder of >= 3 rungs; got {list(scales)}")
    return float(ratio[0])


def scale_means(y_bar_by_rep, cv_by_rep, n, R: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean over replicates of log Ybar, its CLT variance cv^2 / (n R)) per scale.

    `y_bar_by_rep[r][j]` and `cv_by_rep[r][j]` are replicate r's mean and cv at scale j.
    """
    L = np.log(np.asarray(y_bar_by_rep, float)).mean(axis=0)
    cv = np.asarray(cv_by_rep, float).mean(axis=0)
    return L, cv ** 2 / (np.asarray(n, float) * R)


def slope_matrix(n_scales: int, rho: float) -> np.ndarray:
    """J with (J @ L)_k = gamma_k, k = 0 .. n_scales - 2."""
    J = np.zeros((n_scales - 1, n_scales))
    idx = np.arange(n_scales - 1)
    J[idx, idx], J[idx, idx + 1] = -1.0, 1.0
    return J / np.log(rho)


def differences(L, var_L, rho: float, *, gamma_ext: tuple[float, float] | None = None,
                gamma_weights=None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(D, Cov D, gamma-hat, se gamma-hat) for D_k = gamma_k - gamma-hat.

    Exactly one of: `gamma_ext` = (value, se) from an independent draw, or
    `gamma_weights` = g with gamma-hat = g @ L (the same draw as the gamma_k).
    """
    if (gamma_ext is None) == (gamma_weights is None):
        raise ValueError("give exactly one of gamma_ext and gamma_weights")
    L, var_L = np.asarray(L, float), np.asarray(var_L, float)
    J = slope_matrix(L.size, rho)
    if gamma_weights is not None:
        g = np.asarray(gamma_weights, float)
        A = J - g[None, :]
        return A @ L, A @ np.diag(var_L) @ A.T, float(g @ L), float(np.sqrt((g ** 2) @ var_L))
    gamma_hat, se_g = float(gamma_ext[0]), float(gamma_ext[1])
    return J @ L - gamma_hat, J @ np.diag(var_L) @ J.T + se_g ** 2, gamma_hat, se_g


def select(D, cov, window=None) -> tuple[np.ndarray, int]:
    """(mask of fitted points, their sign): resolved, majority sign, inside `window`.

    `window` is a boolean mask of the candidate k (default: all). Ties go to +.
    """
    D = np.asarray(D, float)
    sd = np.sqrt(np.diag(cov))
    cand = np.ones(D.size, bool) if window is None else np.asarray(window, bool)
    resolved = cand & (np.abs(D) > RESOLVED_Z * sd)
    sign = 1 if (D[resolved] > 0).sum() >= (D[resolved] < 0).sum() else -1
    return resolved & (np.sign(D) == sign), sign


def gls_line(x, D, cov) -> dict:
    """GLS of ln|D| = c - omega x, with the delta-method covariance of ln|D|."""
    y = np.log(np.abs(D))
    Jy = np.diag(1.0 / D)
    Ci = np.linalg.inv(Jy @ cov @ Jy)
    X = np.vstack([np.ones_like(x), x]).T
    covb = np.linalg.inv(X.T @ Ci @ X)
    b = covb @ X.T @ Ci @ y
    r = y - X @ b
    chi2, dof = float(r @ Ci @ r), int(x.size - 2)
    return {"omega": float(-b[1]), "se": float(np.sqrt(covb[1, 1])), "c": float(b[0]),
            "chi2": chi2, "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof))}


def omega_loglog(scales, L, var_L, *, gamma_ext=None, gamma_weights=None, window=None) -> dict:
    """The whole method on one ladder: D_k, its covariance, the point cut and the line.

    x is ln of each pair's LOWER scale, so the slope is -omega whatever rho is. `window`
    masks the candidate pairs (length len(scales) - 1). Plain lists out, for the JSON.
    """
    rho = ladder_ratio(scales)
    D, cov, gamma_hat, se_g = differences(L, var_L, rho, gamma_ext=gamma_ext,
                                          gamma_weights=gamma_weights)
    mask, sign = select(D, cov, window)
    sd = np.sqrt(np.diag(cov))
    out = {"rho": rho, "gamma_hat": gamma_hat, "gamma_hat_se": se_g,
           "lower_scale": [int(s) for s in scales[:-1]], "gamma_k": (D + gamma_hat).tolist(),
           "D": D.tolist(), "sd_D": sd.tolist(), "fitted": mask.tolist(), "sign": sign,
           "resolved_z": RESOLVED_Z, "n_points": int(mask.sum())}
    if mask.sum() < MIN_POINTS:
        return {**out, "omega": None, "se": None, "c": None, "chi2": None, "dof": None,
                "p_value": None,
                "why_none": f"{int(mask.sum())} resolved point(s); a line needs {MIN_POINTS}"}
    x = np.log(np.asarray(scales[:-1], float))
    return {**out, **gls_line(x[mask], D[mask], cov[np.ix_(mask, mask)]), "why_none": None}


def plot_panel(ax, res: dict, color: str, label: str | None = None) -> None:
    """log_rho|gamma_k - gamma-hat| (± delta-method se) against log_rho k, with the line.

    Filled markers were fitted; open ones were not (unresolved, wrong sign or outside
    the window).
    """
    rho = res["rho"]
    lr = np.log(rho)
    x = np.log(res["lower_scale"]) / lr
    D, sd, fit = np.asarray(res["D"]), np.asarray(res["sd_D"]), np.asarray(res["fitted"])
    ok = D != 0
    y = np.log(np.abs(D[ok])) / lr
    e = sd[ok] / np.abs(D[ok]) / lr
    for sel, face in ((fit[ok], color), (~fit[ok], "none")):
        ax.errorbar(x[ok][sel], y[sel], yerr=e[sel], color=color, mfc=face, marker="o",
                    ms=5, lw=0, elinewidth=1, capsize=2)
    if res["omega"] is not None:
        xf = x[fit]
        xx = np.linspace(xf.min() - 0.3, xf.max() + 0.3, 50)
        # ln|D| = c - omega ln(scale), in log_rho units on both axes.
        ax.plot(xx, (res["c"] - res["omega"] * xx * lr) / lr, color=color, lw=1,
                label=label)
