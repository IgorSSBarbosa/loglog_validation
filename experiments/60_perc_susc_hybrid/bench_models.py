"""Experiment 60: explore vs layers (CPU) vs hybrid, wall time per sample at each x.

explore with these params is experiment 54's model (its other params are the defaults).

Same (x, n, seed) for the three models, so each returns the same Y (checked).
The CPU models run on all numba threads; the hybrid on one fewer, since that
core drives the GPU (as in sweep_cut.py). Saves data/bench_models_d{dim}.json.
User, 2026-11-12: where does the hybrid start to beat the CPU models?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import percolation_susceptibility_explore as E  # noqa: E402
from models import percolation_susceptibility_hybrid as H  # noqa: E402
from models import percolation_susceptibility_layers as L  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--eps0", type=float, default=0.09)
    ap.add_argument("--x", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--n", type=int, nargs="+",
                    default=[10_000_000, 10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000])
    ap.add_argument("--cut", type=int, default=H.DEFAULT_CUT)
    ap.add_argument("--seed", type=int, default=2026111260)
    ap.add_argument("--models", nargs="+", default=["explore", "layers", "hybrid"],
                    choices=["explore", "layers", "hybrid"],
                    help="explore first if given: the others are checked against it")
    ap.add_argument("--max-live", type=int, nargs="+", default=None,
                    help="layers caps to time (one layers row each); default: the model's")
    ap.add_argument("--adaptive", action="store_true",
                    help="add a layers row with recycle='adaptive' (default max_live)")
    ap.add_argument("--label", default="", help="suffix of the output files")
    args = ap.parse_args()
    if len(args.n) != len(args.x):
        ap.error("--n needs one value per --x")
    threads = numba.get_num_threads()
    base = {"dim": args.dim, "eps0": args.eps0}
    models = [
        ("explore", E, {**base, "max_sites": 1_000_000_000}, threads),
        *[(f"layers" if c is None else f"layers_cap{c}", L,
           base if c is None else {**base, "max_live": c}, threads)
          for c in (args.max_live or [None])],
        *([("layers_adaptive", L, {**base, "recycle": "adaptive"}, threads)]
          if args.adaptive else []),
        ("hybrid", H, {**base, "cut": args.cut}, threads - 1),
    ]
    models = [m for m in models if m[0].split("_")[0] in args.models]
    for _, mod, params, t in models:  # warm-up: numba and NVRTC compiles
        numba.set_num_threads(t)
        mod.simulate(16, 20_000, params, np.random.default_rng(0))
    rows = []
    for x, n in zip(args.x, args.n):
        y0 = None
        for name, mod, params, t in models:
            numba.set_num_threads(t)
            t0 = time.perf_counter()
            try:
                y = mod.simulate(x, n, params, np.random.default_rng([args.seed, x]))
            except MemoryError as err:  # a cap too small is a result of this bench
                print(f"x {x:5d} n {n:9d} {name:16s} MemoryError: {err}", flush=True)
                rows.append({"model": name, "x": x, "n": n, "memory_error": str(err)})
                continue
            wall = time.perf_counter() - t0
            same = None if y0 is None else bool(np.array_equal(y, y0))
            y0 = y if y0 is None else y0
            rows.append({"model": name, "threads": t, "x": x, "n": n, "wall": wall,
                         "us_per_sample": 1e6 * wall / n, "mean": float(y.mean()),
                         "identical_to_explore": same})
            print(f"x {x:5d} n {n:9d} {name:16s} {t:2d} thr  wall {wall:8.1f} s  "
                  f"{1e6 * wall / n:9.3f} us/sample  mean {y.mean():.6g}  same {same}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"bench_models_d{args.dim}{args.label}.json"
    f.write_text(json.dumps({"produced_by": "experiments/60_perc_susc_hybrid/bench_models.py",
                             "created": datetime.now().isoformat(timespec="seconds"),
                             "config": vars(args), "rows": rows}, indent=1))
    print(f"saved {f}")


if __name__ == "__main__":
    main()
