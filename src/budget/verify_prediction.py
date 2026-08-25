"""Check allocation_table.py's predictions against reality, cheaply.

The table predicts two things for a given ladder: how long it will take, and
how accurate gamma-hat will be. Both are predictions from measured constants,
not measurements, so both deserve checking -- especially the wall clock, whose
only input is a throughput number calibrated on one machine at one moment.

For each m0 it runs the tuned ladder R times with independent seeds
(SeedSequence.spawn, ground rule 2), timing each run and estimating gamma with
the article's closed-form weights, then reports:

  predicted seconds  vs  measured seconds
  predicted RMSE     vs  measured RMSE

Keep the machine otherwise idle: this measures wall clock, so a competing job
invalidates the timing half of the answer (the RMSE half is unaffected).

CLI:
    python3 verify_prediction.py                       # ~4 min at the defaults
    python3 verify_prediction.py --m0 3 4 5 --replicates 6 --tag my_check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from artifacts import write_artifact  # noqa: E402
from allocation import allocation_constants, predict_error, total_cost  # noqa: E402
from loglog import gamma_closed_form  # noqa: E402
from models import get_model  # noqa: E402
from persistence import run_dir as _run_dir  # noqa: E402

from allocation_table import (  # noqa: E402
    choose_group,
    discover_groups,
    measured_a1,
    measured_cv,
    measured_throughput,
)


def run_ladder(model, params, m0, m, rho, n, seed_seq):
    """One replicate of the ladder: returns (gamma_hat, elapsed_seconds)."""
    spec = get_model(model)
    rng = np.random.default_rng(seed_seq)
    scales = [int(round(rho ** k)) for k in range(m0 + 1, m0 + m + 1)]
    t0 = time.perf_counter()
    y_bar = np.array([float(np.mean(spec.simulate(i, n, params, rng))) for i in scales])
    elapsed = time.perf_counter() - t0
    return float(gamma_closed_form(scales, y_bar, rho, m0)), elapsed


# Cells needing more than this many samples per scale are skipped rather than
# attempted. Budget here is derived FROM m0 (inverting the tuned rule), and
# n grows like rho**(2*m0 + const), so a seemingly innocuous --m0 20 asks for
# ~1e25 samples: numpy raises "maximum allowed dimension exceeded" and the run
# dies partway. Skipping with a message beats crashing after the earlier,
# useful cells have already been computed.
MAX_N = 200_000_000


def verify(model, params, m0_values, *, m, rho, d, omega1, a1, cv, throughput,
           replicates, true_gamma, seed=None, progress=False, max_n=MAX_N):
    seed_seq = np.random.SeedSequence(seed)
    streams = iter(seed_seq.spawn(len(m0_values) * replicates))
    c = allocation_constants(d, omega1, rho, m, a1, cv)

    cells = []
    for m0 in m0_values:
        B = rho ** ((m0 - c["offset"]) / c["theta2"])
        n_exact = B / (c["G"] * rho ** (d * m0))
        if n_exact < 2 or n_exact > max_n:
            if progress:
                why = "too small a budget" if n_exact < 2 else f"n={n_exact:.3g} exceeds max_n"
                print(f"  m0={m0:>2} skipped ({why})", file=sys.stderr, flush=True)
            continue
        n = int(n_exact)
        cost = total_cost(n, m0, m, rho, d)
        pred = predict_error(n, m0, d, omega1, rho, m, a1, cv)
        hats, secs = [], []
        for _ in range(replicates):
            g, e = run_ladder(model, params, m0, m, rho, n, next(streams))
            hats.append(g)
            secs.append(e)
        v = np.array(hats)
        cells.append({
            "m0": m0, "n": n, "cost_steps": cost,
            "predicted_seconds": cost / throughput,
            "measured_seconds_median": float(np.median(secs)),
            "measured_seconds_all": [float(x) for x in secs],
            "predicted_rmse": pred["rmse"],
            "measured_rmse": float(sqrt(np.mean((v - true_gamma) ** 2))),
            "measured_bias": float(v.mean() - true_gamma),
            "measured_sd": float(v.std(ddof=1)),
            "gamma_hats": [float(x) for x in v],
        })
        if progress:
            k = cells[-1]
            print(f"  m0={m0:>2} n={n:>10,}  time {k['predicted_seconds']:>8.2f}s pred / "
                  f"{k['measured_seconds_median']:>8.2f}s meas   "
                  f"rmse {k['predicted_rmse']:.3e} pred / {k['measured_rmse']:.3e} meas",
                  file=sys.stderr, flush=True)
    return {"model": model, "params": params, "m": m, "rho": rho, "d": d,
            "omega1": omega1, "a1": a1, "cv": cv, "throughput": throughput,
            "replicates": replicates, "true_gamma": true_gamma,
            "seed": seed_seq.entropy, "cells": cells,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m0", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=MAX_N,
                        help="skip any ladder needing more than this many samples per scale")
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--d", type=float, default=1.0)
    parser.add_argument("--omega1", type=float, default=1.0)
    parser.add_argument("--true-gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--tag", type=str, default="prediction_check")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "experiments" / "01_srw" / "data")
    args = parser.parse_args(argv)

    root = args.data_root
    groups = discover_groups(root)
    group = choose_group(groups, requested=args.group, interactive=False)
    runs = group["runs"] if group else []
    a1, a1_se, a1_src = measured_a1(runs)
    cv, cv_src = measured_cv(runs[0]) if runs else measured_cv(root / "omega1")
    tp_json = root / "allocation" / "result.json"
    tp, tp_src = measured_throughput(tp_json)

    print(f"constants: a1={a1:+.4f} ({a1_src}); cv={cv:.4f} ({cv_src});")
    print(f"           throughput={tp:.3g} steps/s ({tp_src})")
    if group:
        print(f"run group: {group['name']!r} ({group['replicates']} replicate(s))")
    print(f"\nrunning {len(args.m0)} ladders x {args.replicates} replicates "
          f"-- keep the machine idle, this times wall clock\n", file=sys.stderr)

    result = verify(
        "srw", {"q": 0.5}, args.m0, m=args.m, rho=args.rho, d=args.d,
        omega1=args.omega1, a1=a1, cv=cv, throughput=tp,
        replicates=args.replicates, true_gamma=args.true_gamma,
        seed=args.seed, progress=True, max_n=args.max_n,
    )

    rd = _run_dir(root, args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    write_artifact(rd, "prediction_check", result,
                   produced_by="src/budget/verify_prediction.py")

    print(f"\n{'m0':>4} {'n':>12} {'pred s':>10} {'meas s':>10} {'ratio':>7}"
          f" {'pred rmse':>11} {'meas rmse':>11} {'ratio':>7}")
    print("-" * 78)
    for k in result["cells"]:
        tr = k["measured_seconds_median"] / k["predicted_seconds"]
        rr = k["measured_rmse"] / k["predicted_rmse"]
        print(f"{k['m0']:>4} {k['n']:>12,} {k['predicted_seconds']:>10.3f} "
              f"{k['measured_seconds_median']:>10.3f} {tr:>6.2f}x "
              f"{k['predicted_rmse']:>11.3e} {k['measured_rmse']:>11.3e} {rr:>6.2f}x")

    tratios = [k["measured_seconds_median"] / k["predicted_seconds"] for k in result["cells"]]
    rratios = [k["measured_rmse"] / k["predicted_rmse"] for k in result["cells"]]
    print(f"\ntiming    : measured/predicted spans {min(tratios):.2f}x-{max(tratios):.2f}x "
          f"(median {np.median(tratios):.2f}x)")
    print(f"accuracy  : measured/predicted spans {min(rratios):.2f}x-{max(rratios):.2f}x "
          f"(median {np.median(rratios):.2f}x)")
    print(f"\nRMSE at {result['replicates']} replicates is itself noisy: its own relative "
          f"sd is ~1/sqrt(2R) = {1 / sqrt(2 * result['replicates']):.0%},")
    print("so treat the accuracy column as an order-of-magnitude check, not a calibration.")
    print(f"\noutput = {rd / 'result.json'}")


if __name__ == "__main__":
    _main()
