"""The L_min rule of published finite-size-scaling fits, applied to the log-log line.

Published percolation studies (Wang et al. 2013, PRE 87 052107, Sec. III; Zhang et
al. 2020, arXiv:2004.11289, Sec. II) drop the smallest system sizes until the fit is
acceptable, in these words (Wang 2013):

    "our preferred fit for any given ansatz corresponds to the smallest Lmin for which
    chi^2/DF is O(1), and for which subsequent increases in Lmin do not cause the chi^2
    value to drop by vastly more than one unit per degree of freedom."

The papers apply it to a nonlinear ansatz with correction terms. Here the "ansatz" is
the article's straight log-log line, log Ybar_L = c + gamma log L, because arm B of
`plans/literature_fss_protocols.md` keeps the papers' ALLOCATION and swaps in this
repo's ESTIMATOR (user, 2026-10-26). The two vague phrases were made numeric by the
user the same day:

- "chi^2/DF is O(1)"  ->  chi^2/DF <= 1 + 2 sqrt(2/DF), i.e. within two standard
  deviations of the chi^2_DF law (whose mean/DF is 1 and sd/DF is sqrt(2/DF));
- "does not drop by vastly more than one unit per DF"  ->  for EVERY larger L_min'
  that still leaves DF >= 1, chi^2(L_min) - chi^2(L_min') <= 2 (DF(L_min) - DF(L_min')).

chi^2 is that of the WEIGHTED least-squares line, weights 1/sigma_log_j^2, which is the
statistic that is chi^2_{m-2} distributed when the line is right. The gamma estimate
itself stays the article's UNWEIGHTED slope (eqs. 523-531, `tools/loglog.py`); the
weighted fit exists only to test linearity.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: chi^2/DF acceptance: within this many sd of the chi^2_DF law (user, 2026-10-26).
ACCEPT_SDS = 2.0
#: Stability clause: allowed chi^2 drop per DF removed (user, 2026-10-26).
DROP_PER_DF = 2.0


def line_chi2(scales: Sequence, y_bar: Sequence, sigma_log: Sequence) -> tuple[float, int]:
    """chi^2 and DF of the weighted straight-line fit of log Ybar on log L."""
    x = np.log(np.asarray(scales, dtype=float))
    y = np.log(np.asarray(y_bar, dtype=float))
    s = np.asarray(sigma_log, dtype=float)
    if not (x.size == y.size == s.size):
        raise ValueError(f"length mismatch: {x.size}, {y.size}, {s.size}")
    if x.size < 3:
        raise ValueError("need at least 3 points for a chi^2 with DF >= 1")
    if np.any(s <= 0):
        raise ValueError("sigma_log must be strictly positive")
    w = 1.0 / s**2
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * y))
    r = (y - A @ coef) / s
    return float(r @ r), int(x.size - 2)


def acceptable(chi2: float, dof: int) -> bool:
    return chi2 / dof <= 1.0 + ACCEPT_SDS * math.sqrt(2.0 / dof)


def scan(scales: Sequence, y_bar: Sequence, sigma_log: Sequence) -> list[dict]:
    """One row per L_min (in increasing order) that leaves at least 3 scales."""
    order = np.argsort(np.asarray(scales, dtype=float))
    L = np.asarray(scales)[order]
    y = np.asarray(y_bar, dtype=float)[order]
    s = np.asarray(sigma_log, dtype=float)[order]
    rows = []
    for k in range(L.size - 2):
        chi2, dof = line_chi2(L[k:], y[k:], s[k:])
        rows.append({"lmin": int(L[k]), "chi2": chi2, "dof": dof,
                     "acceptable": acceptable(chi2, dof)})
    return rows


def select_lmin(scales: Sequence, y_bar: Sequence, sigma_log: Sequence) -> dict:
    """The preferred L_min under the rule above, with the scan that chose it.

    Returns {"lmin": int or None, "scan": [...], "why": str}. `lmin` is None when no
    L_min passes, which the caller must report rather than paper over.
    """
    rows = scan(scales, y_bar, sigma_log)
    for i, r in enumerate(rows):
        if not r["acceptable"]:
            continue
        stable = all(r["chi2"] - later["chi2"] <= DROP_PER_DF * (r["dof"] - later["dof"])
                     for later in rows[i + 1:])
        if stable:
            return {"lmin": r["lmin"], "scan": rows,
                    "why": f"smallest acceptable and stable L_min (chi2/DF = {r['chi2']:.2f}/{r['dof']})"}
    return {"lmin": None, "scan": rows, "why": "no L_min is both acceptable and stable"}
