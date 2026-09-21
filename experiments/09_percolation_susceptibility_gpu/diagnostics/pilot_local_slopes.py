"""Score an omega_1 pilot on the exact box ladder: local slopes, fit, conversion.

Reads a finished study (`pilot.json` and its constants) and prints what
plans/exact_box_ladder.md T4.1 asks of it:

  - the rung means and the LOCAL slopes between successive rungs, each with its se
    (rungs are independent draws, so the se is the quadrature of the two rungs');
  - the CHANGE of slope between successive windows, with its se, and whether the
    slope ever changes direction by more than 2 sigma on both sides. The old
    x-ladder pilot zig-zagged there at 5-31 sigma (diagnostics/rounding_zigzag.py);
  - the one-correction fit of eq. (232) (`tools.correction.fit_correction`, the
    pipeline's own), its chi2 and dof, the standardized residual on every rung, and
    one fit per replicate, so the spread the pipeline's se rests on is visible;
  - in L units gamma_L and omega_L, and the conversion the model documents,
    gamma_susc = nu_box * gamma_L and omega_x = nu_box * omega_L, made HERE and not
    in a driver (the tools never learn nu_box).

The literature values are printed to score against and are never an input
(ground rule 4). The scales are the box sides, so the local slope is per ln 2 only
on a 2^k grid; on any other grid the divisor is the actual ln of the ratio.

    python3 experiments/09_percolation_susceptibility_gpu/diagnostics/pilot_local_slopes.py \\
        --study pilot_gsusc_L_d4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import read_artifact  # noqa: E402
from tools.constants import load  # noqa: E402
from tools.correction import fit_correction  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# printed for scoring only
DELTA1 = 0.77          # plan section 1: the expected omega_x
GAMMA_LIT = (1.430, 0.006)
Z_ZIGZAG = 2.0


def chi2_of(fit: dict, scales, y_bar, sigma_log) -> tuple[float, np.ndarray]:
    i, y, s = (np.asarray(v, dtype=float) for v in (scales, y_bar, sigma_log))
    model = np.log(fit["a0"]) + fit["gamma"] * np.log(i) + fit["a1"] * i ** -fit["omega1"]
    z = (model - np.log(y)) / s
    return float((z ** 2).sum()), z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True)
    ap.add_argument("--data-root", type=Path, default=DATA)
    a = ap.parse_args(argv)

    sd = a.data_root / a.study
    pilot = read_artifact(sd, "pilot")
    consts = load(sd)
    scales = np.array(pilot["scales"], dtype=float)
    y, s = np.array(pilot["y_bar"]), np.array(pilot["sigma_log"])
    params = pilot["recipe"]["params"]
    nu_box = float(params["nu_box"])
    R = pilot["replicates"]
    print(f"{a.study}: {pilot['recipe']['model']}, scales {[int(v) for v in scales]}, "
          f"{R} replicates, seed {pilot['recipe'].get('seed')}, "
          f"box_factor {params['box_factor']}, nu_box {nu_box}\n")

    print(f"{'L':>5} {'n/rep':>8} {'Y_bar':>12} {'se(log)':>10} {'cv':>7}")
    for L, n, yb, sg, cv in zip(scales, pilot["n"], y, s, pilot["cv_per_scale"]):
        print(f"{int(L):>5} {n:>8} {yb:>12.4f} {sg:>10.2e} {cv:>7.3f}")

    dl = np.diff(np.log(scales))
    slope = np.diff(np.log(y)) / dl
    sse = np.hypot(s[:-1], s[1:]) / dl
    print("\nlocal slopes d log Y / d log L between successive rungs")
    for k, (sl, e) in enumerate(zip(slope, sse)):
        print(f"  {int(scales[k]):>4} -> {int(scales[k + 1]):<4} {sl:+.4f} +- {e:.4f}")

    # change of slope between successive windows: (lnY[k+2] - lnY[k+1]) / d2 - (lnY[k+1] - lnY[k]) / d1
    print("\nchange of slope between successive windows")
    dz = []
    for k in range(len(slope) - 1):
        ds = slope[k + 1] - slope[k]
        w = np.array([1.0 / dl[k], -(1.0 / dl[k] + 1.0 / dl[k + 1]), 1.0 / dl[k + 1]])
        e = float(np.sqrt(((w * s[k:k + 3]) ** 2).sum()))
        dz.append(ds / e)
        print(f"  window {k + 1} -> {k + 2}: {ds:+.4f} +- {e:.4f}  (z = {ds / e:+.1f})")
    reversals = [k for k in range(len(dz) - 1)
                 if dz[k] * dz[k + 1] < 0 and min(abs(dz[k]), abs(dz[k + 1])) > Z_ZIGZAG]
    monotone = all(v > 0 for v in dz) or all(v < 0 for v in dz)
    print(f"  the slope changes direction by more than {Z_ZIGZAG:.0f} sigma on both sides: "
          + ("NEVER" if not reversals else f"at windows {[k + 2 for k in reversals]}")
          + f"   (monotone: {monotone})")

    fit = fit_correction(scales, y, sigma_log=s)
    chi2, z = chi2_of(fit, scales, y, s)
    dof = len(scales) - 4
    print(f"\neq. (232), one correction, pooled means: gamma_L = {fit['gamma']:.4f}, "
          f"omega_L = {fit['omega1']:.4f}, a1 = {fit['a1']:.3f}, a0 = {fit['a0']:.4f}, "
          f"converged = {fit['converged']}")
    print(f"  chi2 = {chi2:.2f} / {dof} dof, rel_rmse = {fit['rel_rmse']:.2e}; "
          f"standardized residual per rung: " + ", ".join(f"{v:+.2f}" for v in z))

    per = [fit_correction(scales, r["y_bar"], sigma_log=r["sigma_log"])
           for r in pilot["per_replicate"]]
    om = np.array([f["omega1"] for f in per])
    ga = np.array([f["gamma"] for f in per])
    print(f"  per replicate ({len(per)} fits, converged {sum(f['converged'] for f in per)}): "
          f"omega_L " + " ".join(f"{v:.2f}" for v in om)
          + f"\n    mean {om.mean():.3f}, sd {om.std(ddof=1):.3f}, "
          f"sd/sqrt(R) {om.std(ddof=1) / np.sqrt(len(om)):.3f};  gamma_L mean "
          f"{ga.mean():.4f}, sd/sqrt(R) {ga.std(ddof=1) / np.sqrt(len(ga)):.4f}")

    w, e = consts["omega1"].value, consts["omega1"].se
    g = fit["gamma"]
    print(f"\npipeline constants: omega_L = {w:.4f} +- {e:.4f}  "
          f"(se/omega = {e / w:.1%}, criterion < 50%),  a1 = {consts['a1'].value:.3f} "
          f"+- {consts['a1'].se:.3f},  d = {consts['d'].value:.4f} +- {consts['d'].se:.4f}")
    print(f"in the tools' units (L):   gamma_L = {g:.4f},  omega_L = {w:.4f} +- {e:.4f}")
    print(f"written up (x1 nu_box = {nu_box}):  omega_x = {nu_box * w:.4f} +- {nu_box * e:.4f}"
          f"   [scoring target Delta_1 ~ {DELTA1}]")
    print(f"  gamma_susc = nu_box * gamma_L = {nu_box * g:.4f} from this pilot's fit alone "
          f"(indicative; production run measures it)   "
          f"[scoring target {GAMMA_LIT[0]:.3f}({round(GAMMA_LIT[1] * 1000)})]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
