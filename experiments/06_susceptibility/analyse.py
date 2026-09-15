"""Step 3: read the two ladder runs and report gamma_susc, 1/sigma and tau.

Both runs are the same design at two moments, drawn from independent streams:

    moment 1  ->  e1 = gamma_susc = (3 - tau)/sigma
    moment 2  ->  e2 = (4 - tau)/sigma

    1/sigma = e2 - e1        tau = 3 - e1/(e2 - e1)

Standard errors come from a nonparametric bootstrap over the n samples WITHIN each
rung (the runs have no replicates), resampling every rung independently, refitting
the exponent on each bootstrap ladder. The two moments are bootstrapped separately
and paired at random, which is right because the runs are independent -- and is why
the se on 1/sigma is not the difference of two ses.

Targets (dim = 2) are printed for scoring only; nothing here feeds them to a fit.

    python3 experiments/06_susceptibility/analyse.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.optimize import curve_fit  # noqa: E402

from tools.loglog import compare_methods, gamma_drop_leading, ols_slope  # noqa: E402
from tools.persistence import load_metadata, load_samples  # noqa: E402

HERE = Path(__file__).resolve().parent

TARGETS = {"e1": 43 / 18, "e2": 177 / 36, "inv_sigma": 91 / 36, "tau": 187 / 91,
           "d_f": 91 / 48}


def _fit(scales: np.ndarray, y_bar: np.ndarray, m0: int) -> float:
    """OLS slope on the ladder with the first m0 rungs dropped."""
    order = np.argsort(scales)
    lx = np.log(np.asarray(scales, dtype=float)[order][m0:])
    ly = np.log(np.asarray(y_bar, dtype=float)[order][m0:])
    return float(ols_slope(lx, ly)[0])


def _fit_corrected(scales: np.ndarray, y_bar: np.ndarray, m0: int,
                   omega: float, gamma0: float) -> float:
    """eq. (232) with the correction exponent DECLARED, not fitted:

        log Ybar = log a0 + gamma*log x + log1p(a1 * x**-omega)

    omega is declared because this ladder cannot identify it -- fitted free it
    runs to 20-50 with a diverging covariance, the same degeneracy the pilot
    gates in src/study/ exist to catch. omega = 1 is what theory says to expect
    here: the correction-to-scaling exponent in eps is Delta_1 = Omega*nu
    (= 0.79 * 4/3 = 1.05 in dim 2), and the analytic background of chi(p)
    contributes x^-1 as well, so both leading corrections are ~1/x.
    """
    order = np.argsort(scales)
    x = np.asarray(scales, dtype=float)[order][m0:]
    ly = np.log(np.asarray(y_bar, dtype=float)[order][m0:])

    def model(x, log_a0, gamma, a1):
        return log_a0 + gamma * np.log(x) + np.log1p(a1 * x ** (-omega))

    popt, _ = curve_fit(model, x, ly, p0=[ly[0], gamma0, 1.0], maxfev=60000)
    return float(popt[1])


def _bootstrap(samples: dict[int, np.ndarray], m0: int, reps: int,
               rng: np.random.Generator, omega: float | None = None,
               gamma0: float = 2.4) -> np.ndarray:
    scales = np.array(sorted(samples))
    out = np.empty(reps)
    for b in range(reps):
        y_bar = np.array([
            samples[int(s)][rng.integers(0, samples[int(s)].size,
                                         samples[int(s)].size)].mean()
            for s in scales])
        out[b] = (_fit(scales, y_bar, m0) if omega is None
                  else _fit_corrected(scales, y_bar, m0, omega, gamma0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default=str(HERE / "data" / "gamma_susc_m1"))
    ap.add_argument("--m2", default=str(HERE / "data" / "gamma_susc_m2"))
    ap.add_argument("--m0", type=int, default=None,
                    help="rungs to drop; default: scan and report the whole ladder")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--omega", type=float, default=1.0,
                    help="declared correction exponent for the corrected fit")
    ap.add_argument("--corr-m0", type=int, default=2,
                    help="first rung of the corrected fit (default 2 -> x >= 4)")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=str(HERE / "data" / "exponents.json"))
    args = ap.parse_args()

    runs = {}
    for name, path in (("e1", args.m1), ("e2", args.m2)):
        s = load_samples(path)
        runs[name] = {int(k): np.asarray(v, dtype=float) for k, v in s.items()}
        meta = load_metadata(Path(path))
        print(f"{name}: {path}  moment={meta['params']['moment']} "
              f"n={meta['n'] if not isinstance(meta['n'], dict) else meta['n']} "
              f"seed={meta['seed']}")

    scales = np.array(sorted(runs["e1"]))
    print(f"\n{'x':>6} {'p':>10} {'eps':>10} "
          f"{'Ybar(m1)':>14} {'cv':>7} {'se/Y':>8} "
          f"{'Ybar(m2)':>16} {'cv':>7}")
    for s in scales:
        a, b = runs["e1"][int(s)], runs["e2"][int(s)]
        pa = 0.5927460507921 - 0.5 / s
        print(f"{s:>6} {pa:>10.6f} {0.5/s:>10.6f} "
              f"{a.mean():>14.4f} {a.std(ddof=1)/a.mean():>7.3f} "
              f"{a.std(ddof=1)/np.sqrt(a.size)/a.mean():>8.4%} "
              f"{b.mean():>16.4g} {b.std(ddof=1)/b.mean():>7.3f}")

    y1 = np.array([runs["e1"][int(s)].mean() for s in scales])
    y2 = np.array([runs["e2"][int(s)].mean() for s in scales])

    print(f"\ndrop-leading ladder (OLS slope after dropping m0 rungs)")
    print(f"{'m0':>4} {'scales':>26} {'e1':>9} {'e2':>9} "
          f"{'1/sigma':>9} {'tau':>9} {'d_f=d/(tau-1)':>14}")
    for rec in gamma_drop_leading(scales, y1):
        m0 = rec["m0"]
        a, b = rec["gamma_hat"], _fit(scales, y2, m0)
        inv_sigma = b - a
        tau = 3 - a / inv_sigma if inv_sigma != 0 else float("nan")
        df = 2 / (tau - 1) if tau != 1 else float("nan")
        used = f"{rec['scales_used'][0]}..{rec['scales_used'][-1]}"
        print(f"{m0:>4} {used:>26} {a:>9.4f} {b:>9.4f} "
              f"{inv_sigma:>9.4f} {tau:>9.4f} {df:>14.4f}")

    m0 = args.m0 if args.m0 is not None else max(0, len(scales) - 4)
    rng = np.random.default_rng(args.seed)
    ba = _bootstrap(runs["e1"], m0, args.boot, rng)
    bb = _bootstrap(runs["e2"], m0, args.boot, rng)
    binv = bb - ba
    btau = 3 - ba / binv
    bdf = 2 / (btau - 1)

    point = {"e1": _fit(scales, y1, m0), "e2": _fit(scales, y2, m0)}
    point["inv_sigma"] = point["e2"] - point["e1"]
    point["tau"] = 3 - point["e1"] / point["inv_sigma"]
    point["d_f"] = 2 / (point["tau"] - 1)
    ses = {"e1": ba.std(ddof=1), "e2": bb.std(ddof=1),
           "inv_sigma": binv.std(ddof=1), "tau": btau.std(ddof=1),
           "d_f": bdf.std(ddof=1)}

    print(f"\nscored at m0 = {m0} (scales {sorted(scales)[m0:]}), "
          f"{args.boot} bootstrap resamples")
    print(f"{'quantity':>10} {'estimate':>11} {'se':>9} {'target':>11} {'z':>8}")
    for k in ("e1", "e2", "inv_sigma", "tau", "d_f"):
        z = (point[k] - TARGETS[k]) / ses[k]
        print(f"{k:>10} {point[k]:>11.4f} {ses[k]:>9.4f} "
              f"{TARGETS[k]:>11.4f} {z:>8.2f}")

    # --- the same ladder with the correction term, omega declared ---------
    print(f"\neq. (232) with the correction exponent DECLARED at omega = {args.omega} "
          f"(it is NOT identifiable on this ladder: fitted free it runs to 20-50)")
    print(f"{'quantity':>10} {'m0':>3} {'estimate':>11} {'se':>9} {'target':>11} {'z':>8}")
    corrected = {}
    for cm0 in (args.corr_m0, args.corr_m0 + 1):
        cpoint = {"e1": _fit_corrected(scales, y1, cm0, args.omega, TARGETS["e1"]),
                  "e2": _fit_corrected(scales, y2, cm0, args.omega, TARGETS["e2"])}
        ca = _bootstrap(runs["e1"], cm0, args.boot, rng, args.omega, TARGETS["e1"])
        cb = _bootstrap(runs["e2"], cm0, args.boot, rng, args.omega, TARGETS["e2"])
        cinv = cb - ca
        cpoint["inv_sigma"] = cpoint["e2"] - cpoint["e1"]
        cpoint["tau"] = 3 - cpoint["e1"] / cpoint["inv_sigma"]
        cpoint["d_f"] = 2 / (cpoint["tau"] - 1)
        ctau = 3 - ca / cinv
        cse = {"e1": ca.std(ddof=1), "e2": cb.std(ddof=1),
               "inv_sigma": cinv.std(ddof=1), "tau": ctau.std(ddof=1),
               "d_f": (2 / (ctau - 1)).std(ddof=1)}
        for k in ("e1", "e2", "inv_sigma", "tau", "d_f"):
            z = (cpoint[k] - TARGETS[k]) / cse[k]
            print(f"{k:>10} {cm0:>3} {cpoint[k]:>11.4f} {cse[k]:>9.4f} "
                  f"{TARGETS[k]:>11.4f} {z:>8.2f}")
        corrected[cm0] = {"point": cpoint, "se": cse}

    print("\nall estimators, moment 1 (tools/loglog.compare_methods):")
    n1 = int(runs["e1"][int(scales[0])].size)
    cmp1 = compare_methods(scales, y1, n1, true_gamma=TARGETS["e1"])
    for name, rec in cmp1.items():
        if isinstance(rec, dict) and "gamma_hat" in rec:
            print(f"  {name:>22}: {rec['gamma_hat']:.4f}")

    Path(args.out).write_text(json.dumps(
        {"m0": m0, "point": point, "se": ses, "targets": TARGETS,
         "omega_declared": args.omega,
         "corrected": {str(k): v for k, v in corrected.items()},
         "y_bar_moment1": dict(zip(map(int, scales), y1.tolist())),
         "y_bar_moment2": dict(zip(map(int, scales), y2.tolist()))}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
