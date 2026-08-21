"""Budget-planning table for gamma-hat: how long must I run, for what precision?

Builds the table from THIS repo's measured results rather than from assumed
constants, so it stays honest as the measurements improve:

  a1          <- Experiment B's fit          (experiments/*/data/<tag>/omega1.json)
  cv          <- the same run's samples      (sd(Y_i)/E(Y_i), checked scale-free)
  throughput  <- Experiment C's wall clock   (total simulated steps / elapsed)

Each row is one integer m0. For that m0 the script reports the budget at which
it is the optimal choice, the per-scale n it buys, the wall-clock cost, and the
predicted precision of gamma-hat -- i.e. read down the "time" column to what you
can afford, and read across for the ladder to run and the accuracy to expect.

The allocation is `tools/allocation.py`'s `tuned_allocation`: Proposition
prop:opt with its dropped multiplicative constant restored (see that module and
experiments/01_srw/README.md's Experiment C). Using prop:opt's uncalibrated m0
instead costs a factor ~2.2-2.4 in RMSE at these budgets, which the --compare
flag shows column by column.

CLI:
    python3 allocation_table.py                       # defaults: srw, d=1, omega1=1, rho=2, m=6
    python3 allocation_table.py --compare             # add prop:opt's untuned numbers
    python3 allocation_table.py --max-m0 20 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports

from allocation import (  # noqa: E402
    allocation_constants,
    optimal_allocation,
    predict_error,
    total_cost,
    tuned_allocation,
)
from persistence import load_samples  # noqa: E402

# Fallbacks, used only when the corresponding measured run is absent. Each is
# the value this repo actually measured, so the table degrades to "last known
# good" rather than to a guess.
FALLBACK_A1 = -0.2748          # Experiment B, 5 replicates (truth -1/4)
FALLBACK_CV = sqrt(3.14159265358979 / 2 - 1)   # half-normal limit for |S_k|
FALLBACK_THROUGHPUT = 1.53e8   # steps/second, Experiment C's sweep


def measured_a1(run_dirs) -> tuple[float, str]:
    """Correction amplitude a_1 from Experiment B, averaged over replicates."""
    vals = []
    for rd in run_dirs:
        p = Path(rd) / "omega1.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["direct_fit"]["a1"])
    if not vals:
        return FALLBACK_A1, "fallback (no omega1.json found)"
    return float(np.mean(vals)), f"measured, {len(vals)} replicate(s)"


def measured_cv(run_dir) -> tuple[float, str]:
    """Coefficient of variation sd(Y_i)/E(Y_i), averaged over scales.

    Also reports the spread across scales: the allocation math assumes cv is
    scale-free, and for |S_k| it is, but that is an assumption worth seeing.
    """
    rd = Path(run_dir)
    if not (rd / "samples.npz").exists() and not (rd / "samples").is_dir():
        return FALLBACK_CV, "fallback (no samples found)"
    s = load_samples(rd)
    per_scale = [float(np.std(v, ddof=1) / np.mean(v)) for _, v in sorted(s.items())]
    return float(np.mean(per_scale)), (
        f"measured over {len(per_scale)} scales, spread "
        f"{min(per_scale):.4f}-{max(per_scale):.4f}"
    )


def measured_throughput(result_json) -> tuple[float, str]:
    """Simulated steps per second, from Experiment C's own wall clock."""
    p = Path(result_json)
    if not p.exists():
        return FALLBACK_THROUGHPUT, "fallback (no allocation result.json found)"
    r = json.loads(p.read_text())
    steps = sum(c["cost"] * r["replicates"] for c in r["cells"] if not c["skipped"])
    return steps / r["elapsed_seconds"], (
        f"measured, {steps:.3g} steps in {r['elapsed_seconds']:.0f}s"
    )


def human_time(seconds: float) -> str:
    """Seconds -> the largest unit that keeps the number readable."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    for scale, unit in ((1.0, "s"), (60.0, "min"), (3600.0, "h"),
                        (86400.0, "d"), (31557600.0, "yr")):
        v = seconds / scale
        if v < 1000 or unit == "yr":
            if unit == "yr" and v >= 1e5:
                return f"{v:.2g} yr"
            return f"{v:,.1f} {unit}"
    return f"{seconds:.3g} s"  # pragma: no cover


def budget_for_m0(m0: float, d: float, omega1: float, rho: float, m: int,
                  a1: float, cv: float) -> float:
    """Invert m0_tuned(B) = theta2*(log_rho B + log_rho kappa) for B."""
    c = allocation_constants(d, omega1, rho, m, a1, cv)
    return rho ** ((m0 - c["offset"]) / c["theta2"])


def build_rows(m0_values, *, d, omega1, rho, m, a1, cv, throughput, compare=False):
    rows = []
    for m0 in m0_values:
        B = budget_for_m0(m0, d, omega1, rho, m, a1, cv)
        n = int(B / (allocation_constants(d, omega1, rho, m, a1, cv)["G"] * rho ** (d * m0)))
        if n < 1:
            continue
        cost = total_cost(n, m0, m, rho, d)
        err = predict_error(n, m0, d, omega1, rho, m, a1, cv)
        row = {
            "m0": m0,
            "n": n,
            "scales": f"{int(rho ** (m0 + 1))}..{int(rho ** (m0 + m))}",
            "budget_steps": cost,
            "seconds": cost / throughput,
            "time": human_time(cost / throughput),
            "rmse": err["rmse"],
            "bias": err["bias"],
            "sd": err["sd"],
        }
        if compare:
            # What prop:opt's uncalibrated m0 would have chosen at this budget.
            po = optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)
            if po["integer_feasible"]:
                pe = predict_error(po["n"], po["m0"], d, omega1, rho, m, a1, cv)
                row["prop_opt_m0"] = po["m0"]
                row["prop_opt_rmse"] = pe["rmse"]
                row["penalty"] = pe["rmse"] / err["rmse"]
        rows.append(row)
    return rows


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d", type=float, default=1.0, help="cost exponent (Experiment A)")
    parser.add_argument("--omega1", type=float, default=1.0, help="correction exponent (Experiment B)")
    parser.add_argument("--rho", type=float, default=2.0, help="scale ratio")
    parser.add_argument("--m", type=int, default=6, help="number of scales in the window")
    parser.add_argument("--min-m0", type=int, default=0)
    parser.add_argument("--max-m0", type=int, default=24)
    parser.add_argument("--a1", type=float, default=None, help="override the measured a1")
    parser.add_argument("--cv", type=float, default=None, help="override the measured cv")
    parser.add_argument("--throughput", type=float, default=None,
                        help="override the measured steps/second")
    parser.add_argument("--compare", action="store_true",
                        help="also show what prop:opt's uncalibrated m0 would cost")
    parser.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    parser.add_argument("--data-root", type=Path,
                        default=HERE.parent / "experiments" / "01_srw" / "data",
                        help="where to look for the measured runs")
    args = parser.parse_args(argv)

    root = args.data_root
    a1, a1_src = (args.a1, "override") if args.a1 is not None else measured_a1(
        sorted(root.glob("omega1_rep*")) or [root / "omega1"])
    cv, cv_src = (args.cv, "override") if args.cv is not None else measured_cv(root / "omega1")
    tp, tp_src = (args.throughput, "override") if args.throughput is not None else \
        measured_throughput(root / "allocation" / "result.json")

    c = allocation_constants(args.d, args.omega1, args.rho, args.m, a1, cv)

    print("inputs")
    print(f"  a1         = {a1:+.4f}      ({a1_src})")
    print(f"  cv         = {cv:.4f}       ({cv_src})")
    print(f"  throughput = {tp:.3g} steps/s  ({tp_src})")
    print(f"  d = {args.d}   omega1 = {args.omega1}   rho = {args.rho}   m = {args.m}")
    print("\nderived constants (tools/allocation.py: allocation_constants)")
    print(f"  Cb = {c['Cb']:.6f}   |bias| = Cb * rho^(-m0*omega1)")
    print(f"  Cs = {c['Cs']:.6f}   sd     = Cs * n^(-1/2)")
    print(f"  G  = {c['G']:.1f}       B      = n * rho^(d*m0) * G")
    print(f"  m0_tuned = theta2*log_rho(B) {c['offset']:+.3f}"
          f"   <- prop:opt omits this offset")
    print(f"  at the optimum, |bias|/sd = sqrt(d/(2*omega1)) = "
          f"{sqrt(args.d / (2 * args.omega1)):.4f}\n")

    rows = build_rows(range(args.min_m0, args.max_m0 + 1), d=args.d, omega1=args.omega1,
                      rho=args.rho, m=args.m, a1=a1, cv=cv, throughput=tp,
                      compare=args.compare)

    head = f"{'m0':>4} {'n':>16} {'budget (time)':>16} {'RMSE(gamma)':>13}"
    if args.compare:
        head += f" {'prop:opt m0':>12} {'its RMSE':>11} {'penalty':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (f"{r['m0']:>4} {r['n']:>16,} {r['time']:>16} {r['rmse']:>13.3e}")
        if args.compare and "penalty" in r:
            line += (f" {r['prop_opt_m0']:>12} {r['prop_opt_rmse']:>11.3e} "
                     f"{r['penalty']:>7.2f}x")
        print(line)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            fields = list(rows[0]) if rows else []
            wtr = csv.DictWriter(fh, fieldnames=fields)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\ncsv = {args.csv}")

    print("\nEach row is the budget at which that m0 is the optimal choice. Scales for"
          f" row m0 are rho^(m0+1)..rho^(m0+{args.m}).")
    print("RMSE is predicted, not measured -- Experiment C found it within ~10% of"
          " simulation at B = 1e7..1e9.")


if __name__ == "__main__":
    _main()
