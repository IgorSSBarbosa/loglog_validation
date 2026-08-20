"""Experiment B: estimate the correction-to-scaling exponent omega_1 from a
saved run (plans/three_experiment_ladder.md section 3).

Reads a run directory produced by src/generate.py and applies both of
tools/correction.py's estimators to the same sample means:

  1. `fit_correction`      -- fits article eq. (232)'s one-correction
                              truncation, log Y_bar = log a0 + gamma*log i
                              + a1*i^-omega1, all four parameters free.
  2. `omega1_from_bias_decay` -- fits how tools/loglog.py's
                              `gamma_drop_leading` sequence converges as the
                              most contaminated small scales are dropped.

These are different functionals of the data, so agreement between them is
evidence and disagreement is a signal that the one-correction model does not
describe this run. Neither is an article-named estimator (the article gives
the model and the gamma estimators, not an omega_1 procedure), so both are
validated against planted ground truth in tools/tests/test_correction.py.

Generation and analysis are separate scripts throughout this repo, so this
never draws samples -- point it at an existing run directory. Output goes to
<run_dir>/omega1.json, alongside that run's samples.

CLI:
    python3 estimate_omega1.py -data ../experiments/01_srw/data/omega1
    python3 estimate_omega1.py -data <run_dir> --expect-omega1 1.0 --expect-gamma 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports

from correction import fit_correction, omega1_from_bias_decay  # noqa: E402
from loglog import gamma_drop_leading  # noqa: E402
from persistence import load_metadata, load_samples  # noqa: E402

# Windows retaining fewer than this many scales are dropped before fitting the
# bias decay: a gamma-hat from 2-3 points is dominated by its own noise, and
# carries essentially no information about how the bias decays.
MIN_WINDOW_SCALES = 4


def estimate(run_dir: str | Path, *, min_window_scales: int = MIN_WINDOW_SCALES) -> dict:
    """Both omega_1 estimates for one run directory."""
    run_dir = Path(run_dir)
    samples = load_samples(run_dir)
    meta = load_metadata(run_dir)

    scales = sorted(samples)
    y_bar = np.array([float(np.mean(samples[i])) for i in scales])
    counts = np.array([len(samples[i]) for i in scales], dtype=float)
    # SE of log(Y_bar) by the delta method: sd(Y)/ (sqrt(n) * Y_bar).
    sigma_log = np.array(
        [float(np.std(samples[i], ddof=1)) for i in scales]
    ) / (np.sqrt(counts) * y_bar)

    direct = fit_correction(scales, y_bar, sigma_log=sigma_log)

    windows = [w for w in gamma_drop_leading(scales, y_bar)
               if len(w["scales_used"]) >= min_window_scales]
    if len(windows) >= 4:
        decay = omega1_from_bias_decay(
            [w["scales_used"][0] for w in windows], [w["gamma_hat"] for w in windows]
        )
    else:
        decay = {"error": f"only {len(windows)} windows with >= {min_window_scales} scales; "
                          "need 4 to fit 3 parameters"}

    return {
        "run_dir": str(run_dir),
        "model": meta.get("model"),
        "scales": [int(i) for i in scales],
        "n": [int(c) for c in counts],
        "y_bar": [float(v) for v in y_bar],
        "sigma_log": [float(v) for v in sigma_log],
        "direct_fit": direct,
        "bias_decay_fit": decay,
        "seed": meta.get("seed"),
    }


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Run directory written by generate.py (contains samples.npz/samples/ + metadata.json)",
    )
    parser.add_argument(
        "--expect-omega1", dest="expect_omega1", type=float, default=None,
        help="Known omega_1 to check against, e.g. 1.0 for srw's |S_k| "
             "(see experiments/01_srw/README.md). Reporting only -- never fed to the estimators.",
    )
    parser.add_argument(
        "--expect-gamma", dest="expect_gamma", type=float, default=None,
        help="Known gamma to check against, e.g. 0.5 for srw's |S_k|.",
    )
    parser.add_argument(
        "--tol", dest="tol", type=float, default=0.15,
        help="Relative tolerance for the PASS/FAIL check against --expect-* (default 0.15).",
    )
    args = parser.parse_args(argv)

    result = estimate(args.data)
    out_path = Path(args.data) / "omega1.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    d = result["direct_fit"]
    print(f"run      = {result['run_dir']}")
    print(f"model    = {result['model']!r}   seed = {result['seed']}")
    print(f"scales   = {result['scales']}")
    print(f"n        = {result['n']}")
    print("\ndirect fit of eq. (232), log Y_bar = log a0 + gamma*log i + a1*i^-omega1:")
    print(f"  gamma   = {d['gamma']:.4f}")
    print(f"  omega1  = {d['omega1']:.4f}")
    print(f"  a0      = {d['a0']:.4f}")
    print(f"  a1      = {d['a1']:.4f}")
    print(f"  rel_rmse={d['rel_rmse']:.3e}   converged={d['converged']}")

    b = result["bias_decay_fit"]
    print("\nbias-decay fit of gamma_hat(i) = gamma_inf + a*i^-omega1 (drop_leading windows):")
    if "error" in b:
        print(f"  unavailable: {b['error']}")
    else:
        print(f"  gamma_inf = {b['gamma_inf']:.4f}")
        print(f"  omega1    = {b['omega1']:.4f}")
        print(f"  a         = {b['a']:.4f}")
        print(f"  rel_rmse  = {b['rel_rmse']:.3e}   converged={b['converged']}")
        spread = abs(b["omega1"] - d["omega1"])
        print(f"\n  the two estimators differ by {spread:.4f} in omega1"
              + ("  -- consistent" if spread < 0.25 else
                 "  -- LARGE: the one-correction model may not describe this run"))

    checks = []
    if args.expect_omega1 is not None:
        checks.append(("omega1", d["omega1"], args.expect_omega1))
    if args.expect_gamma is not None:
        checks.append(("gamma", d["gamma"], args.expect_gamma))
    if checks:
        print(f"\nagainst known values (reporting only -- the estimators never saw these), "
              f"rel tol {args.tol}:")
        for name, got, want in checks:
            ok = abs(got - want) <= args.tol * abs(want)
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.4f}, expected {want:.4f}")

    print(f"\noutput = {out_path}")


if __name__ == "__main__":
    _main()
