"""Experiment 43, step 1 diagnostics: refit the 5D replicate with Zhang et al.'s
correction-to-scaling ansatz, and test the article's Assumptions 1, 6 and 7 on it.

    .venv/bin/python experiments/43_lit_zhang2020_df_d4_d5/diagnose_d5.py \
        experiments/43_lit_zhang2020_df_d4_d5/data/zhang2020_d5

The literature d_f is used only to print errors, never inside a fit (ground rule 4). Reads protocol_arm.json and the run log written beside the data dir.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

LITERATURE = 3.5260  # Zhang et al. 2020, for the err column only
ACCEPT_SDS = 2.0     # same chi2 rule as tools/lmin_rule.py


def fit_scan(name: str, f, L: np.ndarray, obs: np.ndarray, s: np.ndarray, p0: list[float]) -> None:
    """Weighted LS for every L_m. Each fit is started from the previous L_m's solution
    and from p0, and the smallest finite-covariance chi2 is kept: from p0 alone the
    fits at L_m >= 12 fall into spurious minima with y1 ~ -10."""
    print(f"-- {name}")
    p = p0
    for lm in L[:-4]:
        k = L >= lm
        best = None
        for start in (p, p0, [p0[0], p[1], p[2], 1.0], [p0[0], p[1], p[2], 0.25]):
            try:
                q, c = curve_fit(f, L[k], obs[k], p0=start, sigma=s[k], absolute_sigma=True, maxfev=200000)
            except RuntimeError:
                continue
            chi2 = float(np.sum(((obs[k] - f(L[k], *q)) / s[k]) ** 2))
            if np.all(np.isfinite(np.diag(c))) and (best is None or chi2 < best[2]):
                best = (q, c, chi2)
        if best is None:
            print(f"  Lm={lm:>2.0f} no finite fit")
            continue
        q, c, chi2 = best
        p = list(q)
        se = np.sqrt(np.diag(c))
        dof = int(k.sum()) - len(q)
        ok = chi2 / dof <= 1 + ACCEPT_SDS * np.sqrt(2 / dof)
        print(f"  Lm={lm:>2.0f} d_f={q[0]:.4f}({se[0]:.4f}) err={q[0] - LITERATURE:+.4f} "
              f"exponent={q[3]:+.2f}({se[3]:.2f}) chi2/DF={chi2:.1f}/{dof} {'accept' if ok else 'reject'}")


def main(data: Path) -> None:
    warnings.simplefilter("ignore")  # curve_fit's OptimizeWarning on the degenerate high-L_m fits
    a = json.loads((data / "protocol_arm.json").read_text())
    r = a["per_replicate"][0]
    L = np.array(a["scales"], float)
    n = np.array(a["n"], float)
    y = np.array(r["y_bar"])
    sl = np.array(r["sigma_log"])

    print("== local slopes (Assumption 1: do they settle?) ==")
    slope = np.diff(np.log(y)) / np.diff(np.log(L))
    err = np.hypot(sl[1:], sl[:-1]) / np.diff(np.log(L))
    for i in range(len(slope)):
        print(f"  {L[i]:.0f}-{L[i + 1]:.0f}: {slope[i]:.4f} +- {err[i]:.4f}")

    print("== correction-to-scaling fits ==")
    fit_scan("Zhang eq. (5): Y = L^d_f (a0 + a1 L^y1), y1 free",
             lambda x, g, a0, a1, y1: x ** g * (a0 + a1 * x ** y1), L, y, y * sl, [3.5, 1, -1, -0.5])
    fit_scan("article eq. (232), J=1: log Y = log a0 + d_f log L + a1 L^-omega1",
             lambda x, g, la0, a1, w: la0 + g * np.log(x) + a1 * x ** (-w), L, np.log(y), sl, [3.5, 0, -1, 0.5])

    print("== Assumption 7: GPU ns per site, from the run log ==")
    ms = np.array([float(m) for m in re.findall(r"\(([\d.]+) ms\)", data.with_suffix(".log").read_text())])
    for Li, t in zip(L, ms / 1e3 / n / L ** 5):
        print(f"  L={Li:>3.0f} {t * 1e9:6.3f}")
    k = L >= 8
    print(f"  OLS cost exponent over L>=8: {np.polyfit(np.log(L[k]), np.log(ms[k] / n[k]), 1)[0]:.3f} (declared 5)")

    print("== Assumption 6: cv^2 = Var Y / (E Y)^2 ==")
    cv2 = np.array(r["cv2"])
    print("  " + " ".join(f"{Li:.0f}:{c:.3f}" for Li, c in zip(L, cv2)))
    q, c = curve_fit(lambda x, s, b, w: s + b * x ** (-w), L, cv2, p0=[0.1, 3, 1], maxfev=100000)
    print(f"  unweighted fit s_inf + b L^-w: s_inf={q[0]:.3f}({np.sqrt(c[0, 0]):.3f}) w={q[2]:.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
