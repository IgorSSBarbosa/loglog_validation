"""Pre-registered predictions for experiment 50's long runs, from data drawn BEFORE them.

Written and run before the long runs were launched (user, 2026-11-02), so that the runs
test them out of sample. For each p, the local slope

    s_k = log2(Ybar_2k / Ybar_k) = gamma + a1 (2^-omega - 1) / ln 2 * k^-omega

that eq. (pilot-fit) implies, at the lower scales k of the long run's pairs
(2^18..2^23; the 2^18..2^23 window reaches k = 2^22), under two models fitted on the same source data as build_pilots.py:

  free     gamma, a1, omega all fitted (p = 0.04); at p = 0.3, omega held at p = 0.04's.
  half     gamma held at 1/2, a1 and omega fitted.

The se is the delta method on the fit's covariance. Nothing is drawn; the numbers go
into README.md's "Pre-registered predictions" table.

    python3 experiments/50_rwre_lazy_long_omega/predict.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.stats import chi2

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from build_pilots import SRC_004, SRC_03, TAIL_004, TAIL_03, constants_004  # noqa: E402

KS = [2 ** j for j in range(18, 24)]
LN2 = math.log(2)


def fit(scales, ly, se, *, gamma=None, omega=None):
    """ln Ybar = A + gamma ln i + a1 i^-omega; returns (params dict, cov, chi2, dof)."""
    j = np.log2(scales)
    names = ["A"] + (["gamma"] if gamma is None else []) + ["a1"] + (["omega"] if omega is None else [])

    def full(th):
        d = dict(zip(names, th))
        return (d["A"], d.get("gamma", gamma), d["a1"], d.get("omega", omega))

    def resid(th):
        A, g, a1, w = full(th)
        return (A + g * j * LN2 + a1 * 2.0 ** (-w * j) - ly) / se

    best = None
    for w0 in (0.05, 0.1, 0.2, 0.4):
        x0 = {"A": ly[0] - 0.5 * j[0] * LN2, "gamma": 0.5, "a1": -1.0, "omega": w0}
        lb = [1e-3 if n == "omega" else -np.inf for n in names]
        r = optimize.least_squares(resid, [x0[n] for n in names], bounds=(lb, np.inf))
        if best is None or r.cost < best.cost:
            best = r
    cov = np.linalg.inv(best.jac.T @ best.jac)
    return names, best.x, cov, 2 * best.cost, len(j) - len(names), full


def predict(names, x, cov, full, ks=KS):
    """s_k at `ks` and their delta-method covariance (the k share one fit, so it is full)."""
    def s(th):
        _, g, a1, w = full(th)
        return np.array([g + a1 * (2.0 ** -w - 1) / LN2 * k ** -w for k in ks])
    h = 1e-6 * np.maximum(1.0, np.abs(x))
    jac = np.column_stack([(s(x + h[i] * np.eye(len(x))[i]) - s(x - h[i] * np.eye(len(x))[i]))
                           / (2 * h[i]) for i in range(len(x))])
    return s(x), jac @ cov @ jac.T


def models():
    """The four pre-registered (p, label, fit) rows."""
    ls = json.loads((SRC_004 / "local_slope.json").read_text())["analysis"]
    sc = np.asarray(ls["scales"], float)
    t = sc >= TAIL_004
    d004 = (sc[t], np.asarray(ls["log_y"])[t], np.asarray(ls["se_log_y"])[t])
    w004 = constants_004()[0]["omega1"].value

    pil = json.loads((SRC_03 / "pilot.json").read_text())
    sc3 = np.asarray(pil["scales"], float)
    t3 = sc3 >= TAIL_03
    d03 = (sc3[t3], np.log(np.asarray(pil["y_bar"]))[t3], np.asarray(pil["sigma_log"])[t3])

    return [("0.04", "free", fit(*d004)), ("0.04", "half", fit(*d004, gamma=0.5)),
            ("0.3", "free (omega = p=0.04's)", fit(*d03, omega=w004)),
            ("0.3", "half (omega free)", fit(*d03, gamma=0.5))]


def main() -> int:
    rows = models()
    print("| p | model | fitted | chi2 p | " + " | ".join(f"s at 2^{int(math.log2(k))}" for k in KS) + " |")
    print("|---|---|---|---|" + "---|" * len(KS))
    for p, lab, (names, x, cov, ch, dof, full) in rows:
        d = dict(zip(names, x))
        se = dict(zip(names, np.sqrt(np.diag(cov))))
        par = ", ".join(f"{n} = {d[n]:.4f} ± {se[n]:.4f}" for n in ("gamma", "omega") if n in d)
        v, c = predict(names, x, cov, full)
        print(f"| {p} | {lab} | {par} | {chi2.sf(ch, dof):.2g} | "
              + " | ".join(f"{a:.4f} ± {e:.4f}" for a, e in zip(v, np.sqrt(np.diag(c)))) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
