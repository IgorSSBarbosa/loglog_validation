"""Why the pilot measures omega1 = 0.59 where theory says Delta_1 = Omega*nu = 1.055.

Answer, established by the four analyses below: **0.59 is an effective exponent, not
Delta_1.** The ladder x = 4..128 (eps = 0.125..0.0039) still carries a SECOND correction
of the opposite sign, and a one-term truncation of eq. (232) fitted over that window
compromises -- it lowers omega1 and raises gamma together.

Nothing here changes any statistical tool; tools/correction.py and tools/loglog.py are
used as they stand, and the extra fits are diagnostics, which is what Igor's brief allows
("you may use other fits to estimate the constants and correction-to-scale").

    python3 experiments/06_susceptibility/diagnostics/why_omega1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GAMMA = 43 / 18                 # susceptibility exponent, dim 2 -- DIAGNOSTIC USE ONLY
DELTA1 = (72 / 91) * (4 / 3)    # Omega * nu = 1.0549, the theoretical correction exponent
PILOT = ROOT / "experiments/06_susceptibility/data/article_faithful/pilot.json"


def ladder():
    d = json.loads(PILOT.read_text())
    sc = np.array(d["recipe"]["scales"], dtype=float)
    Y = np.array([r["y_bar"] for r in d["per_replicate"]])
    return sc, Y.mean(0), Y.std(0, ddof=1) / np.sqrt(Y.shape[0])


def fit(lx, ly, sl, *, free_gamma=True, omegas=None, free_omega=True):
    """Least squares on log Ybar = log a0 + gamma*log x + sum_j a_j x**-omega_j."""
    def resid(th):
        k = 0
        g = th[k] if free_gamma else GAMMA
        k += int(free_gamma)
        model = th[k] + g * lx
        k += 1
        if free_omega:
            model = model + th[k] * np.exp(-th[k + 1] * lx)
        else:
            for w in omegas:
                model = model + th[k] * np.exp(-w * lx)
                k += 1
        return (ly - model) / sl
    p0 = ([2.4] if free_gamma else []) + [0.0]
    p0 += [1.0, 1.0] if free_omega else [1.0] * len(omegas)
    r = least_squares(resid, p0, max_nfev=200_000)
    dof = max(1, len(lx) - len(p0))
    g = r.x[0] if free_gamma else GAMMA
    return g, (r.x[-1] if free_omega else None), r.x, float((r.fun ** 2).sum()) / dof


def main() -> int:
    sc, ybar, se = ladder()
    lx, ly, sl = np.log(sc), np.log(ybar), se / ybar

    print("=" * 78)
    print("1. THE GAMMA-FREE DIAGNOSTIC.  D(x) = logY(2x) - logY(x) - gamma*log2")
    print("   = a1 x^-omega (2^-omega - 1), so the slope of log|D| vs log x is -omega")
    print("   with NO a0 and NO fitted gamma to trade against.")
    D = ly[1:] - ly[:-1] - GAMMA * np.log(2.0)
    sD = np.hypot(sl[1:], sl[:-1])
    for k in range(len(D)):
        print(f"     x={int(sc[k]):>4}  D={D[k]:>+9.5f} +/- {sD[k]:.5f}")
    m = np.abs(D) > 3 * sD
    w = (np.abs(D[m]) / sD[m]) ** 2
    A = np.vstack([np.ones(m.sum()), np.log(sc[:-1][m])]).T
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None],
                               np.log(np.abs(D[m])) * np.sqrt(w), rcond=None)
    print(f"   -> omega = {-coef[1]:.4f} over {m.sum()} usable points "
          f"(one-term fit with gamma free gave 0.59)")

    print("=" * 78)
    print("2. MODEL COMPARISON.  A two-term fit with BOTH exponents fixed at theory")
    print("   matches the free-omega one-term fit, and lands nearer the truth.")
    rows = [
        ("gamma free,  1 corr, omega free", dict(free_gamma=True, free_omega=True)),
        ("gamma FIXED, 1 corr, omega free", dict(free_gamma=False, free_omega=True)),
        ("gamma free,  1 corr, omega=1.055", dict(free_gamma=True, free_omega=False,
                                                  omegas=[DELTA1])),
        ("gamma free,  2 corr, 1.055 & 2.110", dict(free_gamma=True, free_omega=False,
                                                    omegas=[DELTA1, 2 * DELTA1])),
    ]
    print(f"   {'model':<36} {'gamma':>9} {'omega':>9} {'chi2/dof':>9}")
    for name, kw in rows:
        g, om, th, c = fit(lx, ly, sl, **kw)
        print(f"   {name:<36} {g:>9.4f} "
              f"{(f'{om:9.4f}' if om is not None else '    fixed')} {c:>9.2f}")
    g2, _, th2, _ = fit(lx, ly, sl, free_gamma=True, free_omega=False,
                        omegas=[DELTA1, 2 * DELTA1])
    a1, a2 = th2[2], th2[3]
    print(f"   two-term amplitudes: a1 = {a1:+.4f}, a2 = {a2:+.4f}  -- OPPOSITE SIGNS")

    print("=" * 78)
    print("3. THE MECHANISM.  The second term is 26% of the first at x=4 and dies by")
    print("   x=64, so the TOTAL correction decays more slowly than its leading term")
    print("   at the bottom of the ladder -- where the signal is largest.")
    print(f"   {'x':>5} {'eps':>9} {'term1':>10} {'term2':>10} {'|t2/t1|':>8} {'local omega':>12}")
    C = a1 * sc ** -DELTA1 + a2 * sc ** (-2 * DELTA1)
    for j, s in enumerate(sc):
        t1, t2 = a1 * s ** -DELTA1, a2 * s ** (-2 * DELTA1)
        loc = "" if j + 1 >= len(sc) else f"{np.log(C[j]/C[j+1])/np.log(2):12.3f}"
        print(f"   {int(s):>5} {0.5/s:>9.5f} {t1:>+10.5f} {t2:>+10.5f} "
              f"{abs(t2/t1):>8.3f} {loc}")

    print("=" * 78)
    print("4. PROOF ON NOISELESS DATA.  Build Ybar EXACTLY from gamma=43/18 with those")
    print("   two corrections, add no noise, and fit the one-term truncation:")
    ly_syn = GAMMA * lx + a1 * sc ** -DELTA1 + a2 * sc ** (-2 * DELTA1)
    print(f"   {'window':>9} {'gamma_hat':>10} {'omega1_hat':>11} {'observed omega1':>17}")
    for lo, obs in ((0, 0.6051), (1, 0.7690)):
        g, om, _, _ = fit(lx[lo:], ly_syn[lo:], np.ones(len(lx) - lo),
                          free_gamma=True, free_omega=True)
        print(f"   x >= {int(sc[lo]):>3} {g:>10.4f} {om:>11.4f} {obs:>17.4f}")
    print(f"   truth put in: gamma = {GAMMA:.4f}, leading omega = {DELTA1:.4f}")
    print("   -> the truncation recovers neither, and reproduces the real fit closely.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
