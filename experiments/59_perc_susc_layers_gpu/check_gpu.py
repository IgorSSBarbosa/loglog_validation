"""Experiment 59: one cluster grown on the GPU against one CPU thread.

validate  same seed -> identical |C|, depth and layer sizes N_l as the CPU
          model's _one_sample, d = 2..5, with a small max_live (rebuilds) and
          with the hard load limit lowered to 1/2 (refused inserts, repeats)
scan      d = 5: |C| of n samples on all CPU threads (the CPU layers model);
          saves the seeds and sizes, to pick clusters of every size from
bench     per log2 bin of |C|, time the same clusters on one CPU thread and on
          the GPU; fits GPU time = a * depth + b * |C|
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import percolation_susceptibility_layers as L  # noqa: E402
from models import percolation_susceptibility_layers_gpu as G  # noqa: E402
from models.percolation_susceptibility import p_at  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "bench_gpu"


def _setup(x: int, dim: int, eps0: float) -> tuple[int, int, int]:
    p = p_at(x, dim, eps0)
    bits, key0 = G._geometry(dim)
    return min(int(p * 2.0 ** 64), 2 ** 64 - 1), bits, key0


_PROF = np.zeros((2, 1 << 18))


def _cpu(seed, thresh, key0, bits, dim):
    """CPU _one_sample and its layer sizes. The profile rows are reused (zeroing
    2 x 2 MiB per call would be timed as the sample's cost)."""
    r = L._one_sample(np.uint64(seed), np.uint64(thresh), np.int64(key0), bits, dim,
                      L.DEFAULT_MAX_LIVE, _PROF[0], _PROF[1])
    prof = _PROF[0, :r[5] + 1].copy()
    _PROF[:, :r[5] + 1] = 0.0
    return r, prof


def validate(args: argparse.Namespace) -> None:
    cases = ((2, 0.15, 64, 20_000), (3, 0.15, 256, 20_000), (4, 0.07, 128, 1 << 20),
             (5, 0.07, 512, 1 << 20), (5, 0.07, 512, 1000))
    all_ok = True
    for hard in (G._HARD_LOAD, 0.5 + 1e-9):
        G._HARD_LOAD = hard
        for dim, eps0, x, max_live in cases:
            thresh, bits, key0 = _setup(x, dim, eps0)
            seeds = np.random.default_rng(args.seed + 1000 * dim + x).integers(
                0, 2 ** 63, size=args.n, dtype=np.int64).astype(np.uint64)
            bad = big = reb = rep = 0
            for s in seeds:
                c, prof = _cpu(s, thresh, key0, bits, dim)
                g = G.grow_cluster(int(s), thresh, key0, bits, dim, max_live=max_live)
                ok = (c[0] == g["size"] and c[5] == g["depth"]
                      and np.array_equal(prof, g["layers"])
                      and prof.sum() == g["size"])
                bad += not ok
                big = max(big, g["size"])
                reb += g["rebuilds"]
                rep += g["repeats"]
            all_ok &= bad == 0
            print(f"hard load {hard:.2f}  dim={dim} x={x:4d} max_live={max_live:8d}  "
                  f"mismatches {bad}/{args.n}  max|C|={big:9d}  rebuilds={reb:5d}  "
                  f"repeats={rep:5d}")
    print("ALL IDENTICAL" if all_ok else "MISMATCH")


def scan(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    t = time.perf_counter()
    _, info = L.explore_layers(args.x, n=args.n, dim=args.dim, eps0=args.eps0,
                               max_layers=1, rng=rng)
    wall = time.perf_counter() - t
    # the draw explore_layers made, replayed
    seeds = np.random.default_rng(args.seed).integers(
        0, 2 ** 63, size=args.n, dtype=np.int64).astype(np.uint64)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"scan_d{args.dim}_x{args.x}.npz"
    np.savez(f, seeds=seeds, size=info["size"], depth=info["depth"], x=args.x,
             dim=args.dim, eps0=args.eps0, seed=args.seed, wall=wall)
    s = info["size"]
    print(f"{f}: n={args.n} wall={wall:.1f}s  max|C|={s.max()}  "
          f"samples with |C|>=1e6: {(s >= 10 ** 6).sum()}")


def bench(args: argparse.Namespace) -> None:
    z = np.load(OUT / f"scan_d{args.dim}_x{args.x}.npz")
    dim, x = int(z["dim"]), int(z["x"])
    thresh, bits, key0 = _setup(x, dim, float(z["eps0"]))
    size = z["size"]
    pick = []
    for b in range(0, int(np.log2(max(size.max(), 1))) + 1):
        idx = np.flatnonzero((size >= 2 ** b) & (size < 2 ** (b + 1)))
        pick += list(idx[:args.per_bin])
    # warm up both (numba compile, NVRTC compile)
    _cpu(z["seeds"][pick[0]], thresh, key0, bits, dim)
    G.grow_cluster(int(z["seeds"][pick[-1]]), thresh, key0, bits, dim)
    rows = []
    for i in pick:
        s = int(z["seeds"][i])
        t = time.perf_counter()
        g = G.grow_cluster(s, thresh, key0, bits, dim)
        t_gpu = time.perf_counter() - t
        t_cpu = np.nan
        if g["size"] <= args.cpu_max:
            t = time.perf_counter()
            c, _ = _cpu(s, thresh, key0, bits, dim)
            t_cpu = time.perf_counter() - t
            assert c[0] == g["size"]
        rows.append((g["size"], g["depth"], t_cpu, t_gpu, g["peak_bytes"]))
        print(f"|C|={g['size']:11d} depth={g['depth']:7d}  cpu={t_cpu:9.4f}s  "
              f"gpu={t_gpu:9.4f}s  cpu/gpu={t_cpu / t_gpu:7.2f}", flush=True)
    a = np.array(rows, dtype=float)
    np.save(OUT / f"bench_d{dim}_x{x}.npy", a)
    # GPU time = a * depth + b * |C|, least squares on clusters >= 1e3 sites
    m = a[:, 0] >= 1e3
    coef, *_ = np.linalg.lstsq(a[m][:, [1, 0]], a[m, 3], rcond=None)
    print(f"\nGPU fit: {coef[0] * 1e6:.2f} us per layer + {coef[1] * 1e9:.3f} ns per site")
    c = a[np.isfinite(a[:, 2]) & (a[:, 0] >= 1e3)]
    print(f"CPU: median {np.median(c[:, 2] / c[:, 0]) * 1e9:.1f} ns per site "
          f"({np.median(c[:, 2] / c[:, 0] / (1 + 2 * dim)) * 1e9:.1f} ns per read)")
    print(f"\n{'|C| bin':>10} {'n':>3} {'depth':>8} {'cpu ms':>10} {'gpu ms':>10} {'cpu/gpu':>8}")
    for b in range(0, int(np.log2(a[:, 0].max())) + 1, 2):
        r = a[(a[:, 0] >= 2 ** b) & (a[:, 0] < 2 ** (b + 2))]
        if r.size:
            med = np.median(r, axis=0)
            print(f"{2 ** b:>10d} {len(r):>3d} {med[1]:8.0f} {med[2] * 1e3:10.3f} "
                  f"{med[3] * 1e3:10.3f} {med[2] / med[3]:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--seed", type=int, default=20261112)
    for name in ("scan", "bench"):
        q = sub.add_parser(name)
        q.add_argument("--x", type=int, default=4096)
        q.add_argument("--dim", type=int, default=5)
    sc = sub.choices["scan"]
    sc.add_argument("--n", type=int, default=1_000_000)
    sc.add_argument("--eps0", type=float, default=0.07)
    sc.add_argument("--seed", type=int, default=20261112)
    be = sub.choices["bench"]
    be.add_argument("--per-bin", type=int, default=4)
    be.add_argument("--cpu-max", type=float, default=3e8,
                    help="skip the CPU timing above this |C| (1 thread, ~25 ns/read)")
    args = ap.parse_args()
    {"validate": validate, "scan": scan, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
