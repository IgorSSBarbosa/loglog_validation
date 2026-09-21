"""Exact box ladder, step T2: how much does S(p, L) still grow with L, in d = 4?

plans/exact_box_ladder.md section 1: on the x-ladder the ceiling gives rung x its
own c_x = L / x**nu_box, and where S(p, L) still grows with L the rung is shifted by
G * log(c_x / c), G = d log S / d log L at FIXED p. The pooled pilots put G at
0.162 +- 0.023 at c = 4. This measures it directly, with no ladder in the way.

At a few fixed rungs x (so a fixed p = p_c - eps0/x) it sweeps the integer box side
L so that c = L / x**nu_box is about 3, 4, 5, 6, 8, and reports S_hat with its
standard error per cell. Two extra sides beside the middle one (an odd/even pair)
check that parity does not matter, which tools/correction.py's srw staircase says
it might. experiments/09_percolation_susceptibility_gpu/analyse_box.py reads the
artifact and applies the acceptance criteria written in README.md.

This is a calibration, so the number of samples per cell is fixed BEFORE the draw
from the sample noise already measured on the same design, never from the cell's
own data: a stopping rule on the data is what
models/percolation_susceptibility.py exists to avoid. With L / xi fixed the
coefficient of variation is a property of c, not of x (0.50-0.56 at c = 4 over
x = 2..64 in pilot_gsusc_d4_low) and falls with c: 0.51 -> 0.18 from c = 4 to 8 in a
300-sample smoke run at x = 8 and 16, an exponent of 1.5-1.7 (Gaussian counting of
the c**4 correlation volumes would give 2). So

    n = ceil((cv4 * (4 / c)**k / se_target)**2),   k = 1.5, capped at --n-max.

That is a plan for the noise, not a claim about it: the realized se/S is recorded
and is what the analysis uses.

Ground rule 2: every cell draws its own stream, a spawned SeedSequence child.
Ground rule 5: seed, config and elapsed time are recorded in the artifact.

    python3 experiments/09_percolation_susceptibility_gpu/calibrate_box.py --dry-run
    python3 experiments/09_percolation_susceptibility_gpu/calibrate_box.py --tag calibration_d4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import percolation_susceptibility as ps  # noqa: E402
from models.percolation_susceptibility_gpu import (  # noqa: E402
    _BYTES_PER_SITE,
    _check_design,
    _susceptibility_gpu_at,
)
from tools.artifacts import write_artifact  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402

HERE = Path(__file__).resolve().parent
DIM = 4
PRODUCED_BY = "experiments/09_percolation_susceptibility_gpu/calibrate_box.py"


def plan_cells(xs, cs, nu_box, parity_x, parity_L) -> list[dict]:
    cells = [{"x": x, "L": int(round(c * x ** nu_box)), "c_target": c}
             for x in xs for c in cs]
    cells += [{"x": parity_x, "L": L, "c_target": None, "parity": True}
              for L in parity_L]
    for cell in cells:
        cell["c"] = cell["L"] / cell["x"] ** nu_box
    return cells


def samples_for(c: float, cv4: float, k: float, se_target: float, n_max: int) -> int:
    return min(n_max, math.ceil((cv4 * (4.0 / c) ** k / se_target) ** 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="calibration_d4",
                    help="output directory name under --data-root")
    ap.add_argument("--data-root", type=Path, default=HERE / "data")
    ap.add_argument("--x", type=int, nargs="+", default=[16, 32, 64],
                    help="rungs of the x-ladder; each fixes p = p_c - eps0/x")
    ap.add_argument("--c", type=float, nargs="+", default=[3, 4, 5, 6, 8],
                    help="target c = L / x**nu_box; L is the nearest integer")
    ap.add_argument("--parity-x", type=int, default=32)
    ap.add_argument("--parity-L", type=int, nargs="*", default=[43, 45],
                    help="an odd/even pair beside the sweep (c = 3.93, 4.12 at x = 32)")
    ap.add_argument("--eps0", type=float, default=0.09)
    ap.add_argument("--nu-box", type=float, default=0.69)
    ap.add_argument("--moment", type=int, default=1)
    ap.add_argument("--se-target", type=float, default=0.005, help="se/S per cell")
    ap.add_argument("--cv4", type=float, default=0.51,
                    help="cv of one sample at c = 4, measured in pilot_gsusc_d4_low")
    ap.add_argument("--cv-exponent", type=float, default=1.5,
                    help="cv falls as c**-k; 1.5 from the smoke run, see the docstring")
    ap.add_argument("--n-max", type=int, default=60_000)
    ap.add_argument("--throughput", type=float, default=8.4e8,
                    help="sites/s, for the time estimate only (pilot_gsusc_d4_low)")
    ap.add_argument("--max-gib", type=float, default=24.0,
                    help="refuse a cell whose one sample needs more device memory")
    ap.add_argument("--seed", type=int, default=2026092201)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cells, samples, memory and time; draw nothing")
    a = ap.parse_args(argv)

    _check_design(DIM, a.moment, "torus")
    cells = plan_cells(a.x, a.c, a.nu_box, a.parity_x, a.parity_L)
    seeds = spawn(a.seed, len(cells))
    total_s = 0.0
    for cell in cells:
        cell["n"] = samples_for(cell["c"], a.cv4, a.cv_exponent, a.se_target, a.n_max)
        cell["p"] = ps.p_at(cell["x"], DIM, a.eps0)
        cell["sites_per_sample"] = (cell["L"] + 1) * cell["L"] ** (DIM - 1)
        cell["gib_per_sample"] = _BYTES_PER_SITE * cell["sites_per_sample"] / 2 ** 30
        cell["predicted_seconds"] = cell["n"] * cell["sites_per_sample"] / a.throughput
        if cell["gib_per_sample"] > a.max_gib:
            raise SystemExit(f"x = {cell['x']}, L = {cell['L']}: one sample needs "
                             f"{cell['gib_per_sample']:.1f} GiB > --max-gib {a.max_gib}")
        total_s += cell["predicted_seconds"]

    head = (f"{'x':>4} {'L':>4} {'c':>7} {'p':>9} {'n':>7} {'GiB/smp':>8} "
            f"{'pred s':>8}")
    print(head)
    for cell in cells:
        print(f"{cell['x']:>4} {cell['L']:>4} {cell['c']:>7.3f} {cell['p']:>9.6f} "
              f"{cell['n']:>7} {cell['gib_per_sample']:>8.2f} "
              f"{cell['predicted_seconds']:>8.0f}"
              + ("   parity" if cell.get("parity") else ""))
    print(f"predicted total: {total_s / 60:.0f} min at {a.throughput:.3g} sites/s")
    if a.dry_run:
        return 0

    run_dir = a.data_root / a.tag
    config = {"dim": DIM, "eps0": a.eps0, "nu_box": a.nu_box, "moment": a.moment,
              "geometry": "torus", "se_target": a.se_target, "cv4": a.cv4,
              "cv_exponent": a.cv_exponent, "n_max": a.n_max, "seed": a.seed,
              "throughput_assumed": a.throughput}
    rows: list[dict] = []
    print(f"\n{'x':>4} {'L':>4} {'c':>7} {'n':>7} {'S_hat':>12} {'se/S':>8} "
          f"{'cv':>7} {'sec':>7}", flush=True)
    for cell, ss in zip(cells, seeds):
        t0 = time.perf_counter()
        y = _susceptibility_gpu_at(cell["L"], cell["p"], cell["n"], DIM, a.moment,
                                   "torus", np.random.default_rng(ss),
                                   name="calibrate_box", cpu=None)
        dt = time.perf_counter() - t0
        mean = float(y.mean())
        se = float(y.std(ddof=1) / np.sqrt(cell["n"]))
        rows.append({**cell, "mean": mean, "se": se,
                     "cv": float(y.std(ddof=1) / mean), "seconds": dt,
                     "seed": seed_record(ss)})
        print(f"{cell['x']:>4} {cell['L']:>4} {cell['c']:>7.3f} {cell['n']:>7} "
              f"{mean:>12.4f} {se / mean:>8.4%} {rows[-1]['cv']:>7.3f} {dt:>7.1f}",
              flush=True)
        write_artifact(run_dir, "box_calibration",
                       {"config": config, "cells": rows, "complete": False,
                        "cells_planned": len(cells)}, produced_by=PRODUCED_BY)
    path = write_artifact(run_dir, "box_calibration",
                          {"config": config, "cells": rows, "complete": True,
                           "cells_planned": len(cells)}, produced_by=PRODUCED_BY)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
