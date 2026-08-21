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


def find_omega1_runs(data_root: Path) -> list[Path]:
    """Run directories holding an Experiment B fit, under OR at `data_root`.

    Accepting `data_root` itself as a run directory matters: pointing this
    script at `.../data/Huge_test/` (a run) rather than `.../data/` (the
    parent of runs) is the natural mistake, and silently falling back to
    hardcoded constants while printing a confident table is the worst
    possible response to it.
    """
    root = Path(data_root)
    if (root / "omega1.json").exists():
        return [root]
    reps = sorted(root.glob("omega1_rep*"))
    hits = [d for d in reps if (d / "omega1.json").exists()]
    if hits:
        return hits
    return [d for d in sorted(root.glob("*")) if (d / "omega1.json").exists()]


def measured_a1(run_dirs) -> tuple[float, float | None, str]:
    """Correction amplitude a_1 from Experiment B: (value, stderr, provenance).

    The stderr is across replicates, so it exists only with >= 2 of them. It
    is what `offset_uncertainty` turns into an m0 range -- reporting a1
    without it invites exactly the "which table is right?" confusion, since
    two runs of the same experiment legitimately differ by more than the gap
    between the tables they produce.
    """
    vals = []
    for rd in run_dirs:
        p = Path(rd) / "omega1.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["direct_fit"]["a1"])
    if not vals:
        return FALLBACK_A1, None, "FALLBACK -- no omega1.json found"
    if len(vals) == 1:
        return float(vals[0]), None, "measured, 1 replicate (no stderr available)"
    se = float(np.std(vals, ddof=1) / sqrt(len(vals)))
    return float(np.mean(vals)), se, f"measured, {len(vals)} replicates"


def measured_cv(run_dir) -> tuple[float, str]:
    """Coefficient of variation sd(Y_i)/E(Y_i), averaged over scales.

    Also reports the spread across scales: the allocation math assumes cv is
    scale-free, and for |S_k| it is, but that is an assumption worth seeing.
    """
    rd = Path(run_dir)
    if not (rd / "samples.npz").exists() and not (rd / "samples").is_dir():
        return FALLBACK_CV, "FALLBACK -- no samples found"
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
        return FALLBACK_THROUGHPUT, "FALLBACK -- no allocation result.json found"
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


def offset_uncertainty(d, omega1, rho, m, a1, cv, a1_se) -> dict | None:
    """How far the tuned m0 moves if a1 is off by +/- one standard error.

    kappa is proportional to Cb^2, and Cb is proportional to |a1|, so the
    offset moves by 2*theta2*log_rho(|a1'|/|a1|) -- logarithmically, which is
    why even a 25% error in a1 barely moves m0. Near the optimum RMSE is
    quadratic in the m0 error, so a shift of delta costs only
    sqrt((rho^(-2*delta*omega1) + (2*omega1/d)*rho^(d*delta)) / (1 + 2*omega1/d)).
    """
    if a1_se is None or a1_se <= 0:
        return None
    base = allocation_constants(d, omega1, rho, m, a1, cv)["offset"]
    lo = allocation_constants(d, omega1, rho, m, abs(a1) - a1_se, cv)["offset"] \
        if abs(a1) - a1_se > 0 else None
    hi = allocation_constants(d, omega1, rho, m, abs(a1) + a1_se, cv)["offset"]
    spread = max(abs(hi - base), abs(lo - base) if lo is not None else 0.0)
    r = 2 * omega1 / d
    penalty = sqrt((rho ** (-2 * spread * omega1) + r * rho ** (d * spread)) / (1 + r))
    return {"offset": base, "lo": lo, "hi": hi, "spread": spread, "penalty": penalty}


def build_budget_rows(budgets, *, d, omega1, rho, m, a1, cv, throughput):
    """Rows indexed by BUDGET rather than by m0.

    This is the view to compare across two runs of this script. Indexing by m0
    asks "at what budget is this m0 optimal", so any change in the constants
    slides every row along the budget axis and two honest tables look wildly
    different. Indexed by budget the question is "given this much time, what
    should I run", and the answer is nearly invariant -- the constants only
    enter through a rounding decision on m0.
    """
    rows = []
    for B in budgets:
        t = tuned_allocation(B, d, omega1, rho, m, a1=a1, cv=cv)
        if not t["integer_feasible"]:
            continue
        rows.append({
            "budget_steps": t["cost"],
            "seconds": t["cost"] / throughput,
            "time": human_time(t["cost"] / throughput),
            "m0": t["m0"],
            "n": t["n"],
            "scales": f"{int(rho ** (t['m0'] + 1))}..{int(rho ** (t['m0'] + m))}",
            "rmse": t["rmse"],
            "bias": t["bias"],
            "sd": t["sd"],
        })
    return rows


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


def _write_csv(path, rows) -> None:
    if not path or not rows:
        return
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\ncsv = {path}")


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
    parser.add_argument("--by-budget", action="store_true",
                        help="index rows by budget instead of by m0 -- the view that stays "
                             "put when the measured constants change")
    parser.add_argument("--min-log10-budget", type=int, default=6)
    parser.add_argument("--max-log10-budget", type=int, default=20)
    parser.add_argument("--compare", action="store_true",
                        help="also show what prop:opt's uncalibrated m0 would cost")
    parser.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    parser.add_argument("--data-root", type=Path,
                        default=HERE.parent / "experiments" / "01_srw" / "data",
                        help="where to look for the measured runs")
    args = parser.parse_args(argv)

    root = args.data_root
    runs = find_omega1_runs(root)
    if args.a1 is not None:
        a1, a1_se, a1_src = args.a1, None, "override"
    else:
        a1, a1_se, a1_src = measured_a1(runs)
    cv_run = runs[0] if runs else root / "omega1"
    cv, cv_src = (args.cv, "override") if args.cv is not None else measured_cv(cv_run)
    tp_json = root / "allocation" / "result.json"
    if not tp_json.exists() and (root.parent / "allocation" / "result.json").exists():
        tp_json = root.parent / "allocation" / "result.json"
    tp, tp_src = (args.throughput, "override") if args.throughput is not None else \
        measured_throughput(tp_json)

    c = allocation_constants(args.d, args.omega1, args.rho, args.m, a1, cv)

    fellback = [n for n, src in (("a1", a1_src), ("cv", cv_src), ("throughput", tp_src))
                if src.startswith("FALLBACK")]
    if fellback:
        print("!" * 78)
        print(f"WARNING: no measured data found for {', '.join(fellback)} under")
        print(f"  {root}")
        print("Using this repo's last-known-good constants instead. If you meant to point")
        print("at a run directory, pass the directory that CONTAINS omega1.json; if at the")
        print("parent of several runs, pass the one containing omega1/ or omega1_rep*/.")
        print("The table below is still self-consistent, but it does NOT describe that data.")
        print("!" * 78 + "\n")

    print("inputs")
    print(f"  a1         = {a1:+.4f}"
          + (f" +/- {a1_se:.4f}" if a1_se else " " * 11)
          + f"  ({a1_src})")
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
          f"{sqrt(args.d / (2 * args.omega1)):.4f}")

    unc = offset_uncertainty(args.d, args.omega1, args.rho, args.m, a1, cv, a1_se)
    if unc:
        print(f"\nsensitivity to a1 (+/- 1 se): offset moves by at most "
              f"{unc['spread']:.3f} in m0,")
        print(f"  which costs at most {unc['penalty']:.4f}x in RMSE -- the optimum is "
              f"quadratic, so it is flat.")
    else:
        print("\nsensitivity: no stderr for a1 (needs >= 2 replicates), so the offset's")
        print("  uncertainty is not quantified here. Run replicates to get it.")
    print()

    if args.by_budget:
        budgets = [10.0 ** e for e in range(args.min_log10_budget, args.max_log10_budget + 1)]
        rows = build_budget_rows(budgets, d=args.d, omega1=args.omega1, rho=args.rho,
                                 m=args.m, a1=a1, cv=cv, throughput=tp)
        head = f"{'budget (time)':>16} {'m0':>4} {'n':>16} {'RMSE(gamma)':>13}"
        print(head)
        print("-" * len(head))
        for r in rows:
            print(f"{r['time']:>16} {r['m0']:>4} {r['n']:>16,} {r['rmse']:>13.3e}")
        _write_csv(args.csv, rows)
        print("\nIndexed by budget: this is the view to compare across runs of this script.")
        print("It answers 'given this much time, what should I run', which barely moves when")
        print("the measured constants do -- unlike the m0-indexed table, whose rows slide")
        print("along the budget axis. Scales for row m0 are rho^(m0+1)..rho^(m0+%d)." % args.m)
        return

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

    _write_csv(args.csv, rows)

    print("\nEach row is the budget at which that m0 is the optimal choice. Scales for"
          f" row m0 are rho^(m0+1)..rho^(m0+{args.m}).")
    print("RMSE is predicted, not measured -- Experiment C found it within ~10% of"
          " simulation at B = 1e7..1e9.")


if __name__ == "__main__":
    _main()
