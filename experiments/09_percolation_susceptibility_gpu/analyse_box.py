"""Exact box ladder, step T2.2: read calibrate_box.py's artifact and score it.

For each rung x (a fixed p) it fits log S against log(c/4), c = L / x**nu_box, by
weighted least squares with a quadratic,

    log S = a + G * u + b * u**2,     u = log(c / 4),     sigma_i = se_i / S_i,

so G is the local elasticity d log S / d log L AT c = 4 and the same fit gives it at
any other c (at c = 8: G + 2 b log 2). sigma_i is the delta-method se of log S_hat.

The acceptance criteria of plans/exact_box_ladder.md T2.2 are scored, not fed to any
fit (they are printed as targets, ground rule 4):

  1. THE GATE. G at c = 4, pooled over x, agrees with 0.162 +- 0.023 (the value the
     pooled omega_1 pilots imply, plan section 1) within 2 sigma combined. If not,
     the zig-zag has another cause and the plan is wrong: STOP.
  2. Finite-size scaling: G_x(4) agrees across x within 2 sigma of the pooled value.
  3. Compute neutrality: cv**2 * L**dim is flat across c within +-25% of its mean
     at each x (d = 2 measured 5.7, 6.7, 7.3 e4). It is the cost of a fixed
     relative precision, so a failure is a finding about what a bigger box costs,
     not a stop.
  4. Parity: L = 43 and 45 beside 44 at x = 32 are as smooth as the neighbours,
     by (a) the second difference log S(43) + log S(45) - 2 log S(44) against the
     small value the fitted G predicts for it, and (b) each one's residual from the
     quadratic fit through the sweep cells at that x. If this fails, the sqrt(2)
     pilot grid must use even L only.

It also records the number T2.3 needs for the box factor already chosen: the drift
bound G(c) * (nu_box - nu_low) on gamma_hat if nu_box is above nu, with nu_low the
low end of the literature's 2 sigma, next to a quarter of the target se.

The quadratic above was written before any data existed, and the first real run
showed S(p, L) SATURATING in c: a quadratic in log c cannot follow that (one rung's
fit is rejected, and G(c = 8) comes out negative). So a POST-HOC block follows the
pre-specified verdicts, unchanged: the secant elasticities between neighbouring c,
which need no functional form, pooled over x with their spread. T2.3 is read from the
top secant, not from the quadratic's G(c = 8).

    python3 experiments/09_percolation_susceptibility_gpu/analyse_box.py --tag calibration_d4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import read_artifact  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- targets, printed for scoring only ---------------------------------------
G4_TARGET = (0.162, 0.023)          # plan section 1: pooled pilots, chi2 3.7 / 3
Z_MAX = 2.0
NEUTRALITY_BAND = 0.25
NU_LITERATURE = (0.6845, 0.0023)    # d = 4 site percolation, as used by the plan
TARGET_SE = {"the literature's": 0.006, "one more decimal": 0.0006}


def wls(u: np.ndarray, y: np.ndarray, sig: np.ndarray, deg: int):
    """Weighted polynomial fit: beta, cov(beta), chi2, dof."""
    X = np.vander(u, deg + 1, increasing=True)
    w = 1.0 / sig ** 2
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    chi2 = float(((y - X @ beta) ** 2 * w).sum())
    return beta, cov, chi2, len(y) - deg - 1


def elasticity(beta, cov, u0: float) -> tuple[float, float]:
    """G(u0) = b1 + 2 b2 u0 and its se from the fit's covariance."""
    g = np.array([0.0, 1.0, 2.0 * u0])
    return float(g @ beta), float(np.sqrt(g @ cov @ g))


def pool(vals, ses) -> tuple[float, float]:
    w = 1.0 / np.asarray(ses) ** 2
    return float((w * np.asarray(vals)).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="calibration_d4")
    ap.add_argument("--data-root", type=Path, default=HERE / "data")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score a run that has not finished (the artifact is "
                         "rewritten after every cell)")
    a = ap.parse_args(argv)

    art = read_artifact(a.data_root / a.tag, "box_calibration")
    if not art["complete"] and not a.allow_partial:
        raise SystemExit(f"{len(art['cells'])} of {art['cells_planned']} cells drawn; "
                         f"pass --allow-partial to score a run in progress")
    cfg, cells = art["config"], art["cells"]
    dim, nu_box = cfg["dim"], cfg["nu_box"]
    sweep = [c for c in cells if not c.get("parity")]
    xs = sorted({c["x"] for c in sweep})
    print(f"{art['artifact']}: {len(cells)} cells, seed {cfg['seed']}, "
          f"dim {dim}, nu_box {nu_box}, eps0 {cfg['eps0']}\n")

    fits = {}
    G4, G8 = [], []
    for x in xs:
        cs = sorted((c for c in sweep if c["x"] == x), key=lambda c: c["c"])
        u = np.log([c["c"] / 4.0 for c in cs])
        y = np.log([c["mean"] for c in cs])
        sig = np.array([c["se"] / c["mean"] for c in cs])
        beta, cov, chi2, dof = wls(u, y, sig, 2)
        g4, s4 = elasticity(beta, cov, 0.0)
        g8, s8 = elasticity(beta, cov, np.log(2.0))
        fits[x] = (beta, cov)
        G4.append((g4, s4))
        G8.append((g8, s8))

        print(f"x = {x}   p = {cs[0]['p']:.6f}")
        print(f"  {'L':>4} {'c':>7} {'n':>7} {'S_hat':>11} {'se/S':>7} {'cv':>6} "
              f"{'cv^2 L^d':>10} {'local G':>16}")
        for i, c in enumerate(cs):
            loc = ""
            if i:
                d = np.log(c["c"] / cs[i - 1]["c"])
                loc = (f"{(y[i] - y[i - 1]) / d:+.3f} +- "
                       f"{np.hypot(sig[i], sig[i - 1]) / d:.3f}")
            print(f"  {c['L']:>4} {c['c']:>7.3f} {c['n']:>7} {c['mean']:>11.4f} "
                  f"{sig[i]:>7.3%} {c['cv']:>6.3f} {c['cv'] ** 2 * c['L'] ** dim:>10.3g} "
                  f"{loc:>16}")
        print(f"  quadratic fit chi2 = {chi2:.2f} / {dof} dof;  "
              f"G(c=4) = {g4:.4f} +- {s4:.4f},  G(c=8) = {g8:.4f} +- {s8:.4f}\n")

    g4, s4 = pool(*zip(*G4))
    g8, s8 = pool(*zip(*G8))
    print(f"pooled over x = {xs}:  G(c=4) = {g4:.4f} +- {s4:.4f}"
          f"   G(c=8) = {g8:.4f} +- {s8:.4f}\n")

    # 1. the gate
    t, ts = G4_TARGET
    z = (g4 - t) / np.hypot(s4, ts)
    gate = abs(z) <= Z_MAX
    print(f"1. GATE   G(c=4) = {g4:.4f} +- {s4:.4f} vs target {t} +- {ts}: "
          f"z = {z:+.2f}  {verdict(gate)}"
          + ("" if gate else "\n   STOP: the zig-zag has another cause; the plan is wrong."))

    # 2. finite-size scaling
    zs = [(g - g4) / s for g, s in G4]
    fss = all(abs(v) <= Z_MAX for v in zs)
    print(f"2. FSS    G_x(4) - pooled, in se: "
          + ", ".join(f"x={x}: {v:+.2f}" for x, v in zip(xs, zs)) + f"  {verdict(fss)}")

    # 3. compute neutrality
    worst = 0.0
    parts = []
    for x in xs:
        v = np.array([c["cv"] ** 2 * c["L"] ** dim for c in sweep if c["x"] == x])
        dev = float(np.abs(v / v.mean() - 1.0).max())
        worst = max(worst, dev)
        parts.append(f"x={x}: {dev:.0%}")
    print(f"3. COST   max |cv^2 L^d / mean - 1| across c: " + ", ".join(parts)
          + f"  (band {NEUTRALITY_BAND:.0%})  {verdict(worst <= NEUTRALITY_BAND)}")

    # 4. parity
    px = sorted({c["x"] for c in cells if c.get("parity")})
    for x in px:
        odd = {c["L"]: c for c in cells if c["x"] == x and c.get("parity")}
        mid = [c for c in sweep if c["x"] == x and c["L"] in
               {L + d for L in odd for d in (-1, 1)}]
        if len(odd) == 2 and mid:
            Ls = sorted(odd)
            c0 = mid[0]
            lo, hi = odd[Ls[0]], odd[Ls[1]]
            r = (np.log(lo["mean"]) + np.log(hi["mean"]) - 2 * np.log(c0["mean"]))
            expect = g4 * (np.log(lo["L"]) + np.log(hi["L"]) - 2 * np.log(c0["L"]))
            sr = np.sqrt((lo["se"] / lo["mean"]) ** 2 + (hi["se"] / hi["mean"]) ** 2
                         + 4 * (c0["se"] / c0["mean"]) ** 2)
            za = (r - expect) / sr
            print(f"4a. PARITY x={x}: log S({lo['L']}) + log S({hi['L']}) - 2 log S({c0['L']}) "
                  f"= {r:+.4f} +- {sr:.4f}, fitted G predicts {expect:+.4f}: "
                  f"z = {za:+.2f}  {verdict(abs(za) <= Z_MAX)}")
        beta, cov = fits[x]
        for L in sorted(odd):
            c = odd[L]
            u = np.log(c["c"] / 4.0)
            row = np.array([1.0, u, u * u])
            pred = float(row @ beta)
            var = (c["se"] / c["mean"]) ** 2 + float(row @ cov @ row)
            zb = (np.log(c["mean"]) - pred) / np.sqrt(var)
            print(f"4b. PARITY x={x}: L = {L}: residual from the sweep's quadratic fit "
                  f"{np.log(c['mean']) - pred:+.4f}, z = {zb:+.2f}  "
                  f"{verdict(abs(zb) <= Z_MAX)}")

    # POST-HOC, added after the first real run: the quadratic above is the
    # pre-specified reading, but S(p, L) SATURATES in c, which a quadratic in
    # log c cannot follow (x = 64: chi2 23.7 / 2; G(c = 8) came out negative).
    # The secants between neighbouring c need no functional form. They share
    # cells, so adjacent secants are anticorrelated; each is a plain ratio.
    print("\nPOST-HOC  secant elasticities, model-free (pre-specified criteria above "
          "are unchanged)")
    n_int = min(len([c for c in sweep if c["x"] == x]) for x in xs) - 1
    top = None
    for j in range(n_int):
        vals, ses, edge = [], [], None
        for x in xs:
            cs = sorted((c for c in sweep if c["x"] == x), key=lambda c: c["c"])
            lo, hi = cs[j], cs[j + 1]
            d = np.log(hi["c"] / lo["c"])
            vals.append(np.log(hi["mean"] / lo["mean"]) / d)
            ses.append(np.hypot(lo["se"] / lo["mean"], hi["se"] / hi["mean"]) / d)
            edge = (lo["c"], hi["c"]) if edge is None else edge
        g, s = pool(vals, ses)
        chi2 = sum(((v - g) / e) ** 2 for v, e in zip(vals, ses))
        print(f"  c ~ {edge[0]:.1f} -> {edge[1]:.1f}: "
              + "  ".join(f"x={x}: {v:+.3f}+-{e:.3f}" for x, v, e in zip(xs, vals, ses))
              + f"   pooled {g:+.3f} +- {s:.3f}  (spread across x: chi2 {chi2:.1f} / "
              f"{len(xs) - 1})")
        top = (g, s)

    # T2.3, from the top secant: the plan's rule asks for G at the c already chosen
    nu_low = NU_LITERATURE[0] - 2 * NU_LITERATURE[1]
    gt, st = top
    bound = (abs(gt) + 2 * st) * (nu_box - nu_low)
    print(f"\nT2.3      G at the top of the sweep (secant, c ~ {edge[0]:.0f} -> "
          f"{edge[1]:.0f}) = {gt:+.4f} +- {st:.4f}; drift bound "
          f"(|G| + 2 se) * (nu_box - nu_low) = ({abs(gt):.4f} + {2 * st:.4f}) * "
          f"({nu_box} - {nu_low:.4f}) = {bound:.5f}\n          against a quarter of "
          f"the target se: " + ", ".join(f"{k} {v / 4:.5f}: {verdict(bound <= v / 4)}"
                                         for k, v in TARGET_SE.items()))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
