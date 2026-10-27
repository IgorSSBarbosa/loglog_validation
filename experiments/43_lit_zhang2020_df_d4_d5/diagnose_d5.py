"""Experiment 43, step 1 diagnostics: refit the 5D replicate with Zhang et al.'s
correction-to-scaling ansatz, and test the article's Assumptions 1, 6 and 7 on it.

    .venv/bin/python experiments/43_lit_zhang2020_df_d4_d5/diagnose_d5.py \
        experiments/43_lit_zhang2020_df_d4_d5/data/zhang2020_d5 \
        [--plot experiments/43_lit_zhang2020_df_d4_d5/images/d5_loglog.png]

The literature d_f is used only to print errors and as a reference line, never inside
a fit (ground rule 4). Reads protocol_arm.json and the run log beside the data dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loglog_plot import _AQUA, _BLUE, _GRIDLINE, _INK, _MUTED, _ORANGE  # noqa: E402

LITERATURE = 3.5260  # Zhang et al. 2020, for the err column and reference line only
ACCEPT_SDS = 2.0     # same chi2 rule as tools/lmin_rule.py
ZHANG_LM = 6         # smallest accepted L_m of the y1-free Zhang fit, drawn in panel (b)


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


def ols(L: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of log y on log L: the log-log estimator on any grid."""
    g, c = np.polyfit(np.log(L), np.log(y), 1)
    return float(g), float(c)


def plot(out: Path, L: np.ndarray, y: np.ndarray, sl: np.ndarray, lmin_window: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.3))
    g_full, c_full = ols(L, y)
    w = L >= lmin_window
    g_win, c_win = ols(L[w], y[w])
    x = np.geomspace(L[0], L[-1], 200)

    ax.plot(L, y, "o", ms=7, color=_INK, mec="white", mew=1.5, zorder=3, label=r"$\bar Y_L$ (13 sizes)")
    ax.plot(x, np.exp(c_full) * x ** g_full, lw=2, color=_BLUE, label=rf"log-log, all sizes: $\hat d_f$={g_full:.4f}")
    xw = np.geomspace(L[w][0] / 1.6, L[-1], 50)
    ax.plot(xw, np.exp(c_win) * xw ** g_win, lw=2, color=_ORANGE,
            label=rf"log-log, $L\geq{lmin_window:.0f}$: $\hat d_f$={g_win:.4f}")
    ax.set(xscale="log", yscale="log", xlabel="$L$", ylabel=r"$\bar Y_L$ (far-half sites joined to the face)",
           title="(a) the log-log plot")
    ax.set_xticks(L[::2], [f"{v:.0f}" for v in L[::2]])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9)

    mid = np.sqrt(L[1:] * L[:-1])
    slope = np.diff(np.log(y)) / np.diff(np.log(L))
    err = 1.96 * np.hypot(sl[1:], sl[:-1]) / np.diff(np.log(L))
    bx.errorbar(mid, slope, yerr=err, fmt="o", ms=7, color=_INK, mec="white", mew=1.5, capsize=0, zorder=3,
                label=r"local slope $\Delta\log\bar Y/\Delta\log L$ $\pm1.96$ se")
    k = L >= ZHANG_LM
    f = lambda v, g, a0, a1, y1: v ** g * (a0 + a1 * v ** y1)  # noqa: E731
    q, _ = curve_fit(f, L[k], y[k], p0=[3.5, 1, -1, -0.5], sigma=(y * sl)[k], absolute_sigma=True, maxfev=200000)
    xz = np.geomspace(ZHANG_LM, 400, 200)
    eff = q[0] + q[2] * q[3] * xz ** q[3] / (q[1] + q[2] * xz ** q[3])
    inside = xz <= L[-1]
    bx.plot(xz[inside], eff[inside], lw=2, color=_AQUA,
            label=rf"Zhang ansatz, $L\geq{ZHANG_LM}$: $d_f$={q[0]:.3f}, $y_1$={q[3]:.2f}")
    bx.plot(xz[~inside], eff[~inside], lw=2, ls=(0, (4, 2)), color=_AQUA)  # extrapolation past the data
    bx.axhline(g_full, lw=2, color=_BLUE, label="log-log, all sizes")
    bx.axhline(g_win, lw=2, color=_ORANGE, label=rf"log-log, $L\geq{lmin_window:.0f}$")
    bx.axhline(LITERATURE, lw=1.5, ls="--", color=_MUTED, label=f"literature {LITERATURE:.4f}")
    bx.axvline(L[-1], lw=1, color=_GRIDLINE)
    bx.text(L[-1] * 1.06, 3.46, "largest simulated $L$;\ndashed = extrapolated", fontsize=8, color=_MUTED, va="bottom")
    bx.set(xscale="log", xlabel="$L$", ylabel="slope", ylim=(3.45, 4.2), title="(b) local slopes never settle")
    bx.set_xticks([4, 8, 16, 32, 60, 120, 240], ["4", "8", "16", "32", "60", "120", "240"])
    bx.minorticks_off()
    bx.legend(frameon=False, fontsize=8.5, loc="upper right")
    for a_ in (ax, bx):
        a_.grid(color=_GRIDLINE, lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main(data: Path, out: Path | None) -> None:
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

    if out is not None:
        plot(out, L, y, sl, float(r["window"]["lmin"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="protocol_arm.py run directory (data/zhang2020_d5)")
    parser.add_argument("--plot", type=Path, default=None, help="also write the two-panel log-log figure here")
    args = parser.parse_args()
    main(args.data, args.plot)
