"""Is the zig-zag in the d = 4 local slopes the rounded box side?

The d = 4 susceptibility pilots draw at L = ceil(box_factor * x**nu_box), so each rung's
box factor c_x = L/x**nu_box sits above the declared 4 by a different amount (+8.5% at
x = 2, +0.25% at x = 256). If S(p, L) is still growing with L at c = 4, every rung is
shifted by G * log(c_x/4), G = dlogS/dlogL, and that shift is not monotone in x.

Pools the three runs that share the design exactly (box_factor 4, nu_box 0.69,
eps0 0.09, torus, moment 1) -- they are independent draws, so rungs pool by inverse
variance -- and fits eq. (232) with one correction, with and without the known
regressor log(c_x/4). Measured 2026-09-21 (plans/exact_box_ladder.md §1):

    one correction            chi2 = 47.0 / 4 dof   (rejected)
    + G * log(c_x/4)          chi2 =  3.7 / 3 dof   G = 0.162 +/- 0.023, omega1 = 1.49 +/- 0.17

Diagnostic only: nothing here feeds a tool or a recipe.

    python3 experiments/09_percolation_susceptibility_gpu/diagnostics/rounding_zigzag.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = {"low": "pilot_gsusc_d4_low/pilot.json", "pilot": "pilot_gsusc_d4/pilot.json",
        "final": "autopilot_gsusc_d4_210m/final.json"}
BOX_FACTOR, NU_BOX = 4.0, 0.69


def pooled_rungs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log Ybar per rung, pooled over runs; se from each replicate's own n (sigma_log)."""
    per: dict[int, list[tuple[str, float, float]]] = {}
    for tag, rel in RUNS.items():
        d = json.loads((DATA / rel).read_text())
        scales = d.get("scales") or d["recipe"]["scales"]
        reps = d["per_replicate"]
        ly = np.array([[math.log(v) for v in r["y_bar"]] for r in reps])
        sl = np.array([r["sigma_log"] for r in reps])
        for j, x in enumerate(scales):
            se = math.sqrt(float((sl[:, j] ** 2).sum())) / len(reps)
            per.setdefault(int(x), []).append((tag, float(ly[:, j].mean()), se))
    xs, ys, es = [], [], []
    print(f"{'x':>4} {'L':>4} {'c_x/4-1':>8}   per-run log Ybar (se)")
    for x in sorted(per):
        w = np.array([1 / s ** 2 for _, _, s in per[x]])
        m = np.array([mu for _, mu, _ in per[x]])
        L = math.ceil(BOX_FACTOR * x ** NU_BOX)
        print(f"{x:>4} {L:>4} {100 * (L / (BOX_FACTOR * x ** NU_BOX) - 1):7.2f}%   " +
              "  ".join(f"{t}:{mu:.5f}({s:.5f})" for t, mu, s in per[x]))
        xs.append(x); ys.append(float((w * m).sum() / w.sum())); es.append(1 / math.sqrt(w.sum()))
    return np.array(xs, float), np.array(ys), np.array(es)


def fit(label: str, model, p0, names, x, y, e, lc) -> None:
    res = least_squares(lambda p: (model(p, x, lc) - y) / e, p0, max_nfev=20000)
    cov = np.linalg.pinv(res.jac.T @ res.jac)
    chi2, dof = float((res.fun ** 2).sum()), len(x) - len(p0)
    print(f"{label}\n   chi2 = {chi2:.1f} / {dof} dof; " + ", ".join(
        f"{n}={v:.4f}+/-{math.sqrt(cov[i, i]):.4f}" for i, (n, v) in enumerate(zip(names, res.x))))


def main() -> None:
    x, y, e = pooled_rungs()
    lc = np.log(np.ceil(BOX_FACTOR * x ** NU_BOX) / (BOX_FACTOR * x ** NU_BOX))
    smooth = lambda p, x, l: p[0] + p[1] * np.log(x) + p[2] * x ** (-p[3])
    rounded = lambda p, x, l: smooth(p, x, l) + p[4] * l
    for lo in (2, 4):
        m = x >= lo
        print(f"\n--- rungs x >= {lo} ({int(m.sum())} rungs)")
        fit("  eq. (232), one correction", smooth, [2.0, 1.43, 0.1, 1.0],
            ["b", "gamma", "a1", "omega1"], x[m], y[m], e[m], lc[m])
        fit("  eq. (232) + G*log(c_x/4)", rounded, [2.0, 1.43, 0.1, 1.0, 0.15],
            ["b", "gamma", "a1", "omega1", "G"], x[m], y[m], e[m], lc[m])


if __name__ == "__main__":
    main()
