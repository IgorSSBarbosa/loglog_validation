"""Tail fits of the long runs' ladders 4..2^23 (local_slope.json): how sure is gamma?

On each p's ln Ybar, from a lower cut i_min up to 2^23, eq. (pilot-fit)

    ln Ybar_i = A + gamma ln i + a1 i^-omega

is fitted by weighted least squares (cells are independent draws, se = se_log_y), with
gamma free and with gamma = 1/2 imposed, for a range of i_min. Then the profile chi2 in
gamma (A, a1, omega minimised at each gamma) gives the gamma values the tail cannot
reject, 95% by Delta chi2 < 3.84; and a joint fit of both p with one gamma (A, a1,
omega per p) asks whether a common limit fits. The fit is the one in predict.py.

    python3 experiments/50_rwre_lazy_long_omega/tail_fits.py
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
from predict import fit  # noqa: E402

STUDIES = {"0.04": "long_p0.04_24h", "0.3": "long_p0.3_24h"}
CUTS = {"0.04": [2 ** j for j in (8, 10, 12, 14, 16)], "0.3": [2 ** j for j in (12, 13, 14, 15, 16)]}
GAMMAS = np.round(np.arange(0.40, 0.5201, 0.0025), 4)


def ladder(p: str, i_min: int):
    a = json.loads((EXP / "data" / STUDIES[p] / "local_slope.json").read_text())["analysis"]
    sc = np.asarray(a["scales"], float)
    t = sc >= i_min
    return sc[t], np.asarray(a["log_y"])[t], np.asarray(a["se_log_y"])[t]


def profile(data, gammas):
    """chi2 at each fixed gamma, minimised over A, a1, omega."""
    return np.array([fit(*data, gamma=g)[3] for g in gammas])


def joint(datas, gamma=None):
    """Both p with one gamma (fitted if None); A, a1, omega per p. Returns (gamma, chi2, dof)."""
    def resid(th):
        g = th[0] if gamma is None else gamma
        rest = th[1:] if gamma is None else th
        out = []
        for q, (sc, ly, se) in enumerate(datas):
            A, a1, w = rest[3 * q:3 * q + 3]
            out.append((A + g * np.log(sc) + a1 * sc ** -w - ly) / se)
        return np.concatenate(out)

    best = None
    for w0 in (0.05, 0.1, 0.2, 0.4):
        x0 = ([0.5] if gamma is None else [])
        lb = ([-np.inf] if gamma is None else [])
        for sc, ly, _ in datas:
            x0 += [ly[0] - 0.5 * math.log(sc[0]), -1.0, w0]
            lb += [-np.inf, -np.inf, 1e-3]
        r = optimize.least_squares(resid, x0, bounds=(lb, np.inf))
        if best is None or r.cost < best.cost:
            best = r
    n = sum(len(d[0]) for d in datas)
    return (best.x[0] if gamma is None else gamma), 2 * best.cost, n - len(best.x)


def main() -> int:
    for p in STUDIES:
        print(f"\np = {p}: ln Ybar = A + gamma ln i + a1 i^-omega on i_min..2^23")
        print("  i_min   pts   free: gamma    omega    chi2/dof  p     |  gamma=1/2: omega   chi2/dof  p     |  95% profile gamma")
        for c in CUTS[p]:
            d = ladder(p, c)
            names, x, cov, ch, dof, _ = fit(*d)
            v = dict(zip(names, x)); se = dict(zip(names, np.sqrt(np.diag(cov))))
            nh, xh, _, chh, dofh, _ = fit(*d, gamma=0.5)
            prof = profile(d, GAMMAS)
            ok = GAMMAS[prof - prof.min() < 3.84]
            edge = "<" if ok[0] == GAMMAS[0] else ""
            print(f"  2^{int(math.log2(c)):<4d} {len(d[0]):4d}   {v['gamma']:.4f}±{se['gamma']:.4f} "
                  f"{v['omega']:.3f}±{se['omega']:.3f}  {ch:5.1f}/{dof:<2d} {chi2.sf(ch, dof):.2f}  |  "
                  f"{dict(zip(nh, xh))['omega']:.3f}   {chh:5.1f}/{dofh:<2d} {chi2.sf(chh, dofh):.2f}  |  "
                  f"[{edge}{ok[0]:.4f}, {ok[-1]:.4f}]")

    print("\njoint, one gamma for both p (A, a1, omega per p)")
    for c04, c03 in [(2 ** 10, 2 ** 13), (2 ** 12, 2 ** 14), (2 ** 14, 2 ** 15)]:
        ds = [ladder("0.04", c04), ladder("0.3", c03)]
        g, ch, dof = joint(ds)
        _, chh, dofh = joint(ds, gamma=0.5)
        print(f"  cuts 2^{int(math.log2(c04))} / 2^{int(math.log2(c03))}: gamma = {g:.4f}, "
              f"chi2 {ch:.1f}/{dof} (p {chi2.sf(ch, dof):.2f});  gamma = 1/2: chi2 {chh:.1f}/{dofh} "
              f"(p {chi2.sf(chh, dofh):.2f}), Delta chi2 = {chh - ch:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
