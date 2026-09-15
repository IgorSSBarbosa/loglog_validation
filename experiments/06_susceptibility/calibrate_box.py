"""Step 1: how big must the torus be before S(p, L) stops moving?

prompts/gamma_exponent.tex asks for "increase L until S(p,L) stabilizes". Done
per sample that is a data-dependent stopping rule (biased, and it destroys the
declared cost the allocation machinery needs), so it is done ONCE here instead,
as a calibration: at a few fixed rungs x, sweep the box_factor in
L = ceil(box_factor * x**nu_box) and report S_hat with its standard error.

The output is the box_factor the ladder then DECLARES -- models/percolation_tau.py's
DF_LOWER pattern: a design constant demonstrated safe, not asserted.

Ground rule 2: every (x, box_factor) cell draws its own independent stream.

    python3 experiments/06_susceptibility/calibrate_box.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import percolation_susceptibility as ps  # noqa: E402
from tools.rng import as_seed_sequence  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--box-factor", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--nu-box", type=float, default=4.0 / 3.0)
    ap.add_argument("--moment", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", type=str, default=str(HERE / "data" / "calibration.json"))
    args = ap.parse_args()

    seeds = as_seed_sequence(args.seed).spawn(len(args.x) * len(args.box_factor))
    rows = []
    print(f"{'x':>5} {'bf':>5} {'L':>7} {'sites/sample':>13} "
          f"{'S_hat':>12} {'se':>10} {'se/S':>8} {'cv':>7} {'sec':>7}")
    k = 0
    for x in args.x:
        for bf in args.box_factor:
            L = ps.box_side(x, bf, args.nu_box)
            t0 = time.perf_counter()
            y = ps.percolation_susceptibility(
                x, n=args.n, dim=2, box_factor=bf, nu_box=args.nu_box,
                moment=args.moment,
                rng=np.random.default_rng(seeds[k]))
            dt = time.perf_counter() - t0
            k += 1
            mean = float(y.mean())
            se = float(y.std(ddof=1) / np.sqrt(args.n))
            row = {"x": x, "box_factor": bf, "nu_box": args.nu_box, "L": L,
                   "sites_per_sample": L ** 2, "n": args.n, "mean": mean,
                   "se": se, "cv": float(y.std(ddof=1) / mean), "seconds": dt,
                   "p": ps.p_at(x, 2, 0.5), "eps": ps.epsilon(x, 0.5)}
            rows.append(row)
            print(f"{x:>5} {bf:>5.1f} {L:>7} {L**2:>13,} "
                  f"{mean:>12.4f} {se:>10.4f} {se/mean:>8.4%} "
                  f"{row['cv']:>7.3f} {dt:>7.1f}")
        # saturation summary for this x: compare each to the largest box
        ref = [r for r in rows if r["x"] == x][-1]
        for r in [r for r in rows if r["x"] == x][:-1]:
            z = (r["mean"] - ref["mean"]) / np.hypot(r["se"], ref["se"])
            r["z_vs_largest"] = float(z)
            r["rel_vs_largest"] = float(r["mean"] / ref["mean"] - 1.0)
            print(f"      bf={r['box_factor']:<5.1f} vs bf={ref['box_factor']:.1f}: "
                  f"{r['rel_vs_largest']:+.3%}  (z = {z:+.2f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
