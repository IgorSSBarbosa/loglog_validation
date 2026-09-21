"""Noiseless stability of the two ladder designs against their design constants.

Truth: S(eps, L) = eps**-gamma * F(L/xi(eps)), xi = xi0 * eps**-nu, F(u) = 1 - exp(-u).
No physical correction to scaling, so any gamma_hat != gamma comes from the DESIGN.
xi0 is set so the nominal design (eps0 = 0.09, c = 4, nu_box = 0.69) has G = 0.16 at
x = 16, the value rounding_zigzag.py measured. The estimator is the OLS slope of log S
on log(1/eps), which both designs use.

    current   p = p_c - eps0/2**k,               L = ceil(c * (eps0/eps)**nu_box)
    exact     L = 8, 16, ..., 128,               p = p_c - eps0 * (c/L)**(1/nu_box)

The exact design IS the current one with rho = 2**(1/nu_box) instead of 2 and an integer
c, which is what makes every L an integer with nothing rounded
(plans/exact_box_ladder.md §1). Measured 2026-09-21, nominal row, units of 1e-4
(gamma_hat - gamma | worst local slope - gamma):

    current, c: -17.7 | 85.1    current, 2c: +0.6 | 1.6    exact, c: +8.8 | 8.9    exact, 2c: +0.9 | 0.9

The F is a model: it drops G ~10x per doubling of c, where d = 2 measured 4-7x.

    python3 experiments/09_percolation_susceptibility_gpu/diagnostics/ladder_stability.py
"""

from __future__ import annotations

import math

import numpy as np

GAMMA, NU = 1.430, 0.6845          # d = 4 literature values -- DIAGNOSTIC USE ONLY
EPS0, C, NU_BOX = 0.09, 4.0, 0.69


def elasticity(t: float) -> float:
    """dlogF/dlogu at u = t for F(u) = 1 - exp(-u)."""
    return t * math.exp(-t) / (1 - math.exp(-t))


def _t_nominal() -> float:
    lo, hi = 0.5, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if elasticity(mid) > 0.16 else (lo, mid)
    return lo


T_NOM = _t_nominal()
XI0 = (C * 16 ** NU_BOX) / T_NOM / (EPS0 / 16) ** (-NU)


def S(eps: float, L: float) -> float:
    return eps ** (-GAMMA) * (1 - math.exp(-L / (XI0 * eps ** (-NU))))


def current(eps0: float, c: float, nu_box: float) -> tuple[list[float], list[int]]:
    eps = [eps0 / 2 ** k for k in range(1, 8)]
    return eps, [math.ceil(c * (eps0 / e) ** nu_box) for e in eps]


def exact(eps0: float, c: float, nu_box: float) -> tuple[list[float], list[int]]:
    Ls = [8, 16, 32, 64, 128]
    return [eps0 * (c / L) ** (1 / nu_box) for L in Ls], Ls


def score(eps, L) -> tuple[float, float, np.ndarray]:
    lx = np.log(1 / np.array(eps))
    ly = np.log([S(e, l) for e, l in zip(eps, L)])
    local = np.diff(ly) / np.diff(lx)
    return np.polyfit(lx, ly, 1)[0] - GAMMA, float(np.abs(local - GAMMA).max()), local


def main() -> None:
    cases = [("nominal (eps0=0.09, c=4, nu_box=0.69)", {}), ("eps0 x 1/2", {"eps0": EPS0 / 2}),
             ("eps0 x 2", {"eps0": EPS0 * 2}), ("c = 3", {"c": 3.0}), ("c = 5", {"c": 5.0}),
             ("nu_box = nu exactly", {"nu_box": NU}), ("nu_box = 0.66 (below nu)", {"nu_box": 0.66}),
             ("nu_box = 0.71 (above nu)", {"nu_box": 0.71})]
    print(f"G at nominal, x = 16: {elasticity(T_NOM):.3f}\n")
    print("gamma_hat - gamma | worst local slope - gamma   (units of 1e-4)")
    print(f"{'perturbation':<40}" + "".join(f"{h:>20}" for h in
          ("current, c", "current, 2c", "exact, c", "exact, 2c")))
    for name, kw in cases:
        e0, c, nb = kw.get("eps0", EPS0), kw.get("c", C), kw.get("nu_box", NU_BOX)
        row = []
        for design, k in ((current, 1), (current, 2), (exact, 1), (exact, 2)):
            b, w, _ = score(*design(e0, c * k, nb))
            row.append(f"{b * 1e4:+7.1f} | {w * 1e4:6.1f}")
        print(f"{name:<40}" + "".join(f"{r:>20}" for r in row))


if __name__ == "__main__":
    main()
