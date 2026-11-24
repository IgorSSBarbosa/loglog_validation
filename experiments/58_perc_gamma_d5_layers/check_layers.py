"""Experiment 58, Step 1: the layered model against the explore model.

validate  same rng -> identical Y and reads, sample by sample, at dims 2..5,
          with the default cap and with a tiny max_live that forces rebuilds
bench     d = 5, eps0 = 0.07: time, reads, peak live sites, depth, rebuilds per
          scale; saves per-sample arrays and the layer profile to data/bench/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import percolation_susceptibility_explore as X  # noqa: E402
from models import percolation_susceptibility_layers as L  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "bench"


def validate(args: argparse.Namespace) -> None:
    for dim, eps0, xs in ((2, 0.15, (4, 16)), (3, 0.15, (16, 64)),
                          (4, 0.07, (32, 128)), (5, 0.07, (64, 512))):
        for x in xs:
            for max_live in (L.DEFAULT_MAX_LIVE, 20_000):
                a, ia = X.explore(x, n=args.n, dim=dim, eps0=eps0, max_sites=10 ** 9,
                                  rng=np.random.default_rng(args.seed + x))
                b, ib = L.explore_layers(x, n=args.n, dim=dim, eps0=eps0,
                                         max_live=max_live,
                                         rng=np.random.default_rng(args.seed + x))
                same = np.array_equal(a, b) and np.array_equal(ia["reads"], ib["reads"])
                ok = (ib["prof_sum"].sum() == ib["size"].sum()
                      and np.all(ib["need3"] <= ib["size"]))
                print(f"dim={dim} x={x:4d} max_live={max_live:9d}  identical={same}  "
                      f"profile sums to sum|C|={ok}  max|C|={ib['size'].max():9d}  "
                      f"max peak_live={ib['peak_live'].max():8d}  "
                      f"rebuilds={ib['rebuilds'].sum()}", flush=True)
                if not (same and ok):
                    raise SystemExit("MISMATCH")


def bench(args: argparse.Namespace) -> None:
    numba.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    L.explore_layers(8, n=100, dim=5, eps0=0.07, rng=rng)
    for x, n in zip(args.x, args.nx):
        t = time.perf_counter()
        y, i = L.explore_layers(x, n=n, dim=5, eps0=0.07, max_live=args.max_live, rng=rng)
        dt = time.perf_counter() - t
        np.savez_compressed(OUT / f"{args.tag}_x{x}.npz", seconds=dt, threads=args.threads,
                            y=y, **{k: v for k, v in i.items()})
        s, pl = i["size"], i["peak_live"]
        big = s > 10 ** 6
        print(f"x={x:5d} n={n:9d} thr={args.threads:2d} {dt:8.1f}s "
              f"{1e9 * dt * args.threads / i['reads'].sum():5.1f} ns/read(thread)  "
              f"mean Y {y.mean():9.4g}  max|C| {s.max():.3g}  max live {pl.max():.3g}  "
              f"max MiB {i['peak_bytes'].max() / 2 ** 20:7.1f}  max depth {i['depth'].max()}  "
              f"rebuilds {i['rebuilds'].sum()}  "
              f"max need3 {i['need3'].max():.3g}  "
              f"need3/|C| over |C|>1e6: median {np.median(i['need3'][big] / s[big]) if big.any() else np.nan:.3g} "
              f"max {(i['need3'][big] / s[big]).max() if big.any() else np.nan:.3g}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "bench"])
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--x", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--nx", type=int, nargs="+", default=[10 ** 6] * 4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-live", type=int, default=L.DEFAULT_MAX_LIVE)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()
    {"validate": validate, "bench": bench}[args.mode](args)


if __name__ == "__main__":
    main()
