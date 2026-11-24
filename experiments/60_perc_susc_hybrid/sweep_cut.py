"""Experiment 60: the hybrid's wall time against the hand-off cut, at full load.

For one (x, n, seed), runs the CPU layers model on all numba threads (the
baseline), then the hybrid at each cut with one thread fewer (that core
drives the GPU). Every run must return the baseline's Y exactly. Records
wall, CPU-batch time, GPU busy time, the wall after the last CPU chunk (only
the GPU working) and the number of hand-offs; saves data/sweep_d{dim}_x{x}.json.
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
from models import percolation_susceptibility_hybrid as H  # noqa: E402
from models import percolation_susceptibility_layers as L  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=int, default=4096)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--eps0", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--cuts", type=int, nargs="+",
                    default=[2 ** k for k in range(8, 23, 2)])
    ap.add_argument("--label", default="",
                    help="suffix of the output file, to keep earlier sweeps")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the CPU-only run (its Y is then not checked)")
    args = ap.parse_args()
    threads = numba.get_num_threads()
    rows = []
    # warm-up: numba compiles, NVRTC compiles
    H.explore_hybrid(64, n=2000, dim=args.dim, eps0=args.eps0, cut=16,
                     rng=np.random.default_rng(0))
    L.explore_layers(64, n=2000, dim=args.dim, eps0=args.eps0, max_layers=1,
                     rng=np.random.default_rng(0))
    y0 = None
    if not args.no_baseline:
        t = time.perf_counter()
        y0, info = L.explore_layers(args.x, n=args.n, dim=args.dim, eps0=args.eps0,
                                    max_layers=1, rng=np.random.default_rng(args.seed))
        wall = time.perf_counter() - t
        rows.append({"model": "layers (CPU)", "threads": threads, "wall": wall,
                     "max_size": int(info["size"].max())})
        print(f"CPU layers, {threads} threads: wall {wall:8.1f} s  "
              f"max|C| {info['size'].max():.3g}", flush=True)
    numba.set_num_threads(threads - 1)
    for cut in args.cuts:
        y, info = H.explore_hybrid(args.x, n=args.n, dim=args.dim, eps0=args.eps0,
                                   cut=cut, rng=np.random.default_rng(args.seed))
        same = None if y0 is None else bool(np.array_equal(y, y0))
        rows.append({"model": "hybrid", "threads": threads - 1, "cut": cut,
                     "wall": info["wall"], "cpu_s": info["cpu_s"],
                     "gpu_s": info["gpu_s"], "gpu_only_s": info["gpu_wait_s"],
                     "handoffs": int(info["on_gpu"].sum()),
                     "max_size": int(info["size"].max()), "identical": same})
        print(f"hybrid cut {cut:>8}: wall {info['wall']:8.1f} s  cpu {info['cpu_s']:8.1f}  "
              f"gpu {info['gpu_s']:8.1f}  gpu-only tail {info['gpu_wait_s']:7.1f}  "
              f"handoffs {info['on_gpu'].sum():6d}  identical {same}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"sweep_d{args.dim}_x{args.x}{args.label}.json"
    f.write_text(json.dumps({"produced_by": "experiments/60_perc_susc_hybrid/sweep_cut.py",
                             "created": datetime.now().isoformat(timespec="seconds"),
                             "config": vars(args), "rows": rows}, indent=1))
    print(f"saved {f}")


if __name__ == "__main__":
    main()
