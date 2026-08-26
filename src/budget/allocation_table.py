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
    python3 src/budget/allocation_table.py                       # defaults: srw, d=1, omega1=1, rho=2, m=6
    python3 src/budget/allocation_table.py --compare             # add prop:opt's untuned numbers
    python3 src/budget/allocation_table.py --max-m0 20 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from artifacts import artifact_path, find_artifacts  # noqa: E402
from allocation import (  # noqa: E402
    allocation_constants,
    optimal_allocation,
    predict_error,
    total_cost,
    tuned_allocation,
)
from constants import (  # noqa: E402
    Constant,
    format_table,
    measured,
    override,
    require,
)
from correction import fit_correction  # noqa: E402
from persistence import load_samples  # noqa: E402

# There are no fallback constants any more, deliberately -- see tools/constants.py
# for the full argument. Briefly: the old FALLBACK_D = 1.0 and
# FALLBACK_CV = sqrt(pi/2 - 1) were srw's EXACT truths, so on srw the table
# printed the right answer whether or not anything had been measured; and
# --d / --omega1 defaulted to 1.0 with no provenance marker, so every table
# this repo published silently used omega1 = 1 while Experiment B's own runs
# measured 0.907, 0.986, 1.198, 0.486. A missing constant is now an error that
# names the flag and the command that would produce it.


def _fit_summary(d: Path) -> dict | None:
    f = artifact_path(d, "omega1")
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (ValueError, OSError):
        return None


def discover_groups(data_root: Path) -> list[dict]:
    """All groups of same-configuration Experiment B runs under `data_root`.

    A "group" is a set of runs that may legitimately be pooled: same model,
    same scale grid, same per-scale n. Runs of DIFFERENT configurations are
    not replicates of one another, and combining them would be meaningless
    rather than merely imprecise -- data/ routinely holds several unrelated
    runs at once.

    Two layouts are recognised, and both collapse to the same thing here:
      <root>/<group>/rep0, rep1, ...   (preferred -- one folder per config)
      <root>/omega1_rep0, omega1_rep1, ...  (flat, older runs)

    Returns one dict per group, sorted with the largest (most replicates)
    first, each carrying `name`, `runs`, `scales`, `n` and `replicates`.
    """
    root = Path(data_root)
    candidates: list[Path] = []
    if artifact_path(root, "omega1").exists():
        candidates = [root]
    else:
        for f in find_artifacts(root, "omega1"):
            candidates.append(f.parent)

    groups: dict[tuple, dict] = {}
    for d in candidates:
        r = _fit_summary(d)
        if r is None:
            continue
        scales, n = tuple(r.get("scales", ())), tuple(r.get("n", ()))
        # Without a recorded scale grid there is no way to confirm two runs
        # share a configuration, so each becomes its own group rather than
        # being pooled on faith (older files predate y_bar/scales being saved).
        key = ((r.get("model"), scales, n) if scales else ("__unknown__", str(d)))
        g = groups.setdefault(key, {"runs": [], "scales": list(scales),
                                    "n": list(n), "model": r.get("model")})
        g["runs"].append(d)

    out = []
    for g in groups.values():
        runs = sorted(g["runs"])
        # Name a group by the folder that holds its replicates: for the nested
        # <group>/rep0, rep1, ... layout that is the group folder, even when
        # only one replicate exists so far. Runs sitting directly in the data
        # root are named by their own directory.
        parents = {p.parent for p in runs}
        if len(parents) == 1:
            parent = next(iter(parents))
            name = runs[0].name if parent == root else parent.name
        else:
            name = os.path.commonprefix([p.name for p in runs]).rstrip("_-") or runs[0].name
        out.append({**g, "runs": runs, "name": name, "replicates": len(runs)})
    return sorted(out, key=lambda g: (-g["replicates"], g["name"]))


def format_groups(groups) -> str:
    lines = [f"{'#':>3}  {'group':<26} {'reps':>4}  {'scales':<22} {'n per scale'}"]
    lines.append("-" * 88)
    for j, g in enumerate(groups, start=1):
        sc = g["scales"]
        scales = f"{sc[0]}..{sc[-1]} ({len(sc)})" if sc else "?"
        n = g["n"]
        nn = f"{min(n):,}..{max(n):,}" if n else "?"
        lines.append(f"{j:>3}  {g['name']:<26} {g['replicates']:>4}  {scales:<22} {nn}")
    return "\n".join(lines)


def choose_group(groups, *, requested: str | None, interactive: bool):
    """Pick one group: by name if asked, else the only/largest one.

    Prompts ONLY when stdin is a terminal, the choice is genuinely ambiguous,
    and no name was given -- never in a pipeline, where blocking on input
    would hang a batch job rather than help anyone.
    """
    if not groups:
        return None
    if requested is not None:
        for g in groups:
            if g["name"] == requested:
                return g
        raise SystemExit(
            f"no run group named {requested!r}. Available:\n{format_groups(groups)}"
        )
    if len(groups) == 1:
        return groups[0]
    if interactive and sys.stdin.isatty():
        print("Several Experiment B run groups are available:\n")
        print(format_groups(groups))
        print()
        while True:
            raw = input(f"select 1-{len(groups)} (or blank for 1): ").strip()
            if raw == "":
                return groups[0]
            if raw.isdigit() and 1 <= int(raw) <= len(groups):
                return groups[int(raw) - 1]
            print("  not a valid choice")
    return groups[0]


def find_omega1_runs(data_root: Path) -> list[Path]:
    """Runs of the single best-supported configuration under `data_root`.

    Thin wrapper over `discover_groups` kept for callers that just want the
    default choice; use `discover_groups` + `choose_group` to see or pick
    among several.
    """
    groups = discover_groups(data_root)
    return groups[0]["runs"] if groups else []


def measured_a1(run_dirs) -> tuple[float, float | None, str]:
    """Correction amplitude a_1 from Experiment B: (value, stderr, provenance).

    Replicates are POOLED and refitted once, NOT averaged fit-by-fit. The
    distinction matters and is not cosmetic: fit_correction is a nonlinear
    function of the data, so E[a1_hat] != a1, and averaging R separate fits
    converges to E[a1_hat] rather than to a1 -- a bias that does not shrink
    no matter how many replicates are added. Measured on this experiment's
    own configuration (250 trials, truth a1 = -0.25):

        R      mean-of-fits bias    pooled-fit bias    RMSE (mean / pooled)
        1          -0.0046             -0.0046          0.0787 / 0.0787
        5          -0.0094             +0.0027          0.0383 / 0.0348
        20         -0.0114             +0.0038          0.0212 / 0.0171

    Mean-of-fits is heading for -0.262, not -0.25, and by R=20 its bias is
    already comparable to its own spread. Pooling averages the sample means
    (weighted by n) BEFORE the nonlinear step, so the fit sees data with
    sqrt(R) less noise and the nonlinear bias shrinks with it. omega_1 is
    barely affected either way (0.0393 vs 0.0395 at R=20); a1 is what the
    allocation offset depends on.

    Pooling needs each run's y_bar/n/sigma_log, which estimate_omega1.py
    records in omega1.json -- so it works from the stored summaries and does
    not require keeping the samples. Falls back to averaging fits for older
    files that lack them.

    The stderr is still taken from the SPREAD of the per-replicate fits,
    divided by sqrt(R): that spread estimates the sd of a single fit, and the
    pooled estimator's sd is close to it over sqrt(R) (measured 0.0166 against
    0.0785/sqrt(20) = 0.0176). It needs >= 2 replicates to exist at all.
    """
    return measured_correction(run_dirs)["a1_triple"]


def measured_correction(run_dirs) -> dict:
    """Experiment B's fit, pooled across replicates: a1 AND omega1, with stderrs.

    Same pooling as described in `measured_a1` -- this is where it actually
    happens; `measured_a1` is the a1-only view. omega_1 is returned too because
    it is what sets the predicted error-decay exponent
    -omega1/(d + 2*omega1), which src/report/plot_allocation.py draws as the
    "expected" reference against Experiment C's measured slope.

    Returns {'a1', 'a1_se', 'omega1', 'omega1_se', 'replicates', 'provenance',
    'a1_triple'} -- the last being (a1, a1_se, provenance) for callers that
    want the older shape.
    """
    loaded = []
    for rd in run_dirs:
        p = artifact_path(Path(rd), "omega1")
        if p.exists():
            loaded.append(json.loads(p.read_text()))

    def _out(a1, a1_se, om, om_se, prov, reps):
        return {"a1": a1, "a1_se": a1_se, "omega1": om, "omega1_se": om_se,
                "replicates": reps, "provenance": prov,
                "a1_triple": (a1, a1_se, prov)}

    if not loaded:
        return _out(None, None, None, None, "no omega1.json found", 0)

    a1s = [r["direct_fit"]["a1"] for r in loaded]
    oms = [r["direct_fit"]["omega1"] for r in loaded]
    R = len(loaded)
    a1_se = float(np.std(a1s, ddof=1) / sqrt(R)) if R > 1 else None
    om_se = float(np.std(oms, ddof=1) / sqrt(R)) if R > 1 else None

    if R == 1:
        return _out(float(a1s[0]), None, float(oms[0]), None,
                    "measured, 1 replicate (no stderr available)", 1)

    poolable = [r for r in loaded
                if all(k in r for k in ("scales", "y_bar", "n", "sigma_log"))]
    scale_sets = {tuple(r["scales"]) for r in poolable}
    if len(poolable) == R and len(scale_sets) == 1:
        scales = np.array(poolable[0]["scales"], dtype=float)
        y = np.array([r["y_bar"] for r in poolable], dtype=float)
        n = np.array([r["n"] for r in poolable], dtype=float)
        sig = np.array([r["sigma_log"] for r in poolable], dtype=float)
        y_pool = (y * n).sum(axis=0) / n.sum(axis=0)          # weighted by sample count
        sig_pool = 1.0 / np.sqrt((1.0 / sig**2).sum(axis=0))  # inverse-variance
        fit = fit_correction(scales, y_pool, sigma_log=sig_pool)
        return _out(float(fit["a1"]), a1_se, float(fit["omega1"]), om_se,
                    f"measured, {R} replicates POOLED then refitted once", R)

    why = ("scale grids differ" if len(scale_sets) > 1 else "some runs lack y_bar/n")
    return _out(float(np.mean(a1s)), a1_se, float(np.mean(oms)), om_se,
                f"measured, {R} replicates, mean of fits ({why}; pooling unavailable "
                f"-- this retains a nonlinear bias, see measured_a1's docstring)", R)


def measured_cv(run_dir) -> tuple[float | None, str]:
    """Coefficient of variation sd(Y_i)/E(Y_i), averaged over scales.

    Also reports the spread across scales: the allocation math assumes cv is
    scale-free, and for |S_k| it is, but that is an assumption worth seeing.
    """
    rd = Path(run_dir)
    if not (rd / "samples.npz").exists() and not (rd / "samples").is_dir():
        return None, "no samples found"
    s = load_samples(rd)
    per_scale = [float(np.std(v, ddof=1) / np.mean(v)) for _, v in sorted(s.items())]
    return float(np.mean(per_scale)), (
        f"measured over {len(per_scale)} scales, spread "
        f"{min(per_scale):.4f}-{max(per_scale):.4f}"
    )


def measured_cost_exponent(data_root) -> tuple[float, float | None, str]:
    """Cost exponent d from measure_cost.py runs: (value, stderr, provenance).

    Takes the AFFINE fit's d, not the pure power law's: a single
    simulate(k, n=1, ...) call pays a fixed per-call overhead that does not
    scale with k, and the pure fit folds that into d (measured 0.77 against a
    true 1.0 on this grid -- see experiments/01_srw/README.md).

    Replicates are averaged directly here, unlike a1: d comes from a fit to
    *timings*, and what varies between runs is machine jitter rather than a
    sampling distribution, so there is no pooled-refit equivalent to prefer.
    """
    root = Path(data_root)
    ds = []
    for f in find_artifacts(root, "cost_probe"):
        try:
            r = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        aff = r.get("affine")
        if isinstance(aff, dict) and "d" in aff and aff.get("converged", True):
            ds.append(float(aff["d"]))
    if not ds:
        return None, None, "no cost_probe.json found"
    if len(ds) == 1:
        return ds[0], None, "measured, 1 cost probe (no stderr available)"
    se = float(np.std(ds, ddof=1) / sqrt(len(ds)))
    return float(np.mean(ds)), se, f"measured, {len(ds)} cost probes"


def predicted_rate(omega1: float, d: float, *, omega1_se=None, d_se=None) -> dict:
    """The article's error-decay exponent -omega1/(d + 2*omega1), with its
    uncertainty propagated from the two MEASURED inputs.

        dtheta/domega1 = -d / (d + 2 omega1)^2
        dtheta/dd      =  omega1 / (d + 2 omega1)^2

    Both partials carry (d + 2 omega1)^-2, so the relative weight of the two
    inputs is just d : omega1 -- comparable in magnitude here, which means the
    error budget is decided by which input is measured worse, not by the
    algebra. In practice omega_1 dominates by ~20x (see Experiment C's
    write-up): it is intrinsically hard to measure, while d is easy.
    """
    denom = d + 2 * omega1
    theta = -omega1 / denom
    var = 0.0
    parts = {}
    if omega1_se:
        c = d / denom**2
        parts["omega1"] = c * omega1_se
        var += parts["omega1"] ** 2
    if d_se:
        c = omega1 / denom**2
        parts["d"] = c * d_se
        var += parts["d"] ** 2
    return {"theta": theta, "se": sqrt(var) if var else None,
            "contributions": parts, "omega1": omega1, "d": d}


def measured_throughput(result_json) -> tuple[float | None, None, str]:
    """Simulated steps per second, from Experiment C's own wall clock.

    Returns (value, se, provenance) for symmetry with the other measured_*
    helpers; there is no se -- it is one wall-clock ratio, not a spread.
    """
    p = Path(result_json)
    if not p.exists():
        return None, None, "no allocation_sweep.json found"
    r = json.loads(p.read_text())
    steps = sum(c["cost"] * r["replicates"] for c in r["cells"] if not c["skipped"])
    return steps / r["elapsed_seconds"], None, (
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


def input_sensitivity(name: str, se: float, *, d, omega1, rho, m, a1, cv) -> dict:
    """How far ONE input's own standard error moves the plan, and what it costs.

    Generalizes `offset_uncertainty`, which only ever varied a1. That was the
    wrong one to single out: a1 enters the offset only through
    log_rho(|a1'|/|a1|), so even a 25% error barely moves m0, whereas omega1
    sits in theta2 = 1/(d + 2*omega1) AND in the exponent rho^(-m0*omega1), and
    is by far the hardest of the three to measure. Reporting the a1 line alone
    made the plan look better determined than it is.

    Returns the shift in the tuned m0 and the RMSE penalty of being that far
    off. The penalty is the same quadratic-flatness formula as
    `offset_uncertainty`'s -- near the optimum a shift of delta costs
    sqrt((rho^(-2*delta*omega1) + (2*omega1/d)*rho^(d*delta)) / (1 + 2*omega1/d)),
    which is why a whole step of m0 is worth only about 1.1x.
    """
    kw = dict(rho=rho, m=m, a1=a1, cv=cv)
    base = allocation_constants(d, omega1, **kw)["offset"]

    def offset_at(**bump):
        args = {"d": d, "omega1": omega1, **kw, **bump}
        return allocation_constants(args["d"], args["omega1"], args["rho"],
                                    args["m"], args["a1"], args["cv"])["offset"]

    shifts = []
    for sign in (+1, -1):
        try:
            if name == "a1":
                v = abs(a1) + sign * se
                if v <= 0:
                    continue
                shifts.append(offset_at(a1=-v if a1 < 0 else v))
            elif name == "omega1":
                v = omega1 + sign * se
                if v <= 0:
                    continue
                shifts.append(offset_at(omega1=v))
            elif name == "d":
                v = d + sign * se
                if v <= 0:
                    continue
                shifts.append(offset_at(d=v))
            else:
                return {"delta_m0": 0.0, "penalty": 1.0, "offset": base}
        except (ValueError, ZeroDivisionError):
            continue

    delta = max((abs(o - base) for o in shifts), default=0.0)
    r = 2 * omega1 / d
    penalty = sqrt((rho ** (-2 * delta * omega1) + r * rho ** (d * delta)) / (1 + r))
    signed = max(shifts, key=lambda o: abs(o - base), default=base) - base
    return {"delta_m0": signed, "penalty": penalty, "offset": base}


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
    parser.add_argument("--d", type=float, default=None,
                        help="cost exponent; measured from cost_probe.json when absent")
    parser.add_argument("--omega1", type=float, default=None,
                        help="correction exponent; measured from omega1.json when absent")
    parser.add_argument("--rho", type=float, default=2.0, help="scale ratio")
    parser.add_argument("--m", type=int, default=6, help="number of scales in the window")
    parser.add_argument("--min-m0", type=int, default=0)
    parser.add_argument("--max-m0", type=int, default=24)
    parser.add_argument("--a1", type=float, default=None, help="override the measured a1")
    parser.add_argument("--cv", type=float, default=None, help="override the measured cv")
    parser.add_argument("--throughput", type=float, default=None,
                        help="override the measured steps/second")
    parser.add_argument("--list", dest="list_groups", action="store_true",
                        help="list the Experiment B run groups found under --data-root, then exit")
    parser.add_argument("--group", type=str, default=None,
                        help="name of the run group to use (see --list). Without it, the group "
                             "with the most replicates is used; if several tie and stdin is a "
                             "terminal, you are asked which")
    parser.add_argument("--no-prompt", action="store_true",
                        help="never ask interactively, even on a terminal -- for scripts")
    parser.add_argument("--by-budget", action="store_true",
                        help="index rows by budget instead of by m0 -- the view that stays "
                             "put when the measured constants change")
    parser.add_argument("--min-log10-budget", type=int, default=6)
    parser.add_argument("--max-log10-budget", type=int, default=20)
    parser.add_argument("--compare", action="store_true",
                        help="also show what prop:opt's uncalibrated m0 would cost")
    parser.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "experiments" / "01_srw" / "data",
                        help="where to look for the measured runs")
    args = parser.parse_args(argv)

    root = args.data_root
    groups = discover_groups(root)

    if args.list_groups:
        if not groups:
            print(f"no Experiment B runs (omega1.json) found under {root}")
        else:
            print(f"Experiment B run groups under {root}:\n")
            print(format_groups(groups))
            print("\nPass --group <name> to pick one. Runs are grouped by (model, scales, n):")
            print("only runs sharing all three are replicates of one configuration and may be")
            print("pooled. The default is the group with the most replicates.")
        return

    group = choose_group(groups, requested=args.group,
                         interactive=not args.no_prompt)
    runs = group["runs"] if group else []
    if group is not None and len(groups) > 1:
        print(f"using run group {group['name']!r} "
              f"({group['replicates']} replicate(s)); "
              f"{len(groups) - 1} other group(s) available -- see --list\n")

    # Every constant is assembled the same way: an explicit override wins,
    # otherwise the measurement, otherwise nothing at all. `require` turns the
    # last case into an error naming the flag and the command that produces it
    # -- see tools/constants.py for why silence was worse than a crash here.
    cv_run = runs[0] if runs else root / "omega1"
    corr = measured_correction(runs)
    tp_json = artifact_path(root / "allocation", "allocation_sweep")
    if not tp_json.exists() and artifact_path(root.parent / "allocation",
                                              "allocation_sweep").exists():
        tp_json = artifact_path(root.parent / "allocation", "allocation_sweep")
    d_val, d_se, d_src = measured_cost_exponent(root)
    cv_val, cv_src = measured_cv(cv_run)
    tp_val, tp_se, tp_src = measured_throughput(tp_json)

    found = {}
    for name, val, se, src in (
            ("d", d_val, d_se, d_src),
            ("omega1", corr["omega1"], corr["omega1_se"], corr["provenance"]),
            ("a1", corr["a1"], corr["a1_se"], corr["provenance"]),
            ("cv", cv_val, None, cv_src),
            ("throughput", tp_val, tp_se, tp_src)):
        given = getattr(args, name, None)
        if given is not None:
            found[name] = override(given, name)
        elif val is not None:
            found[name] = measured(val, se, src)

    consts = {n: require(found, n) for n in ("d", "omega1", "a1", "cv", "throughput")}
    d, omega1 = consts["d"].value, consts["omega1"].value
    a1, a1_se = consts["a1"].value, consts["a1"].se
    omega1_se = consts["omega1"].se
    cv, tp = consts["cv"].value, consts["throughput"].value

    c = allocation_constants(d, omega1, args.rho, args.m, a1, cv)

    print("inputs")
    print(format_table(consts))
    print(f"  {'rho':<11}{args.rho:>10.4f}              (design choice)")
    print(f"  {'m':<11}{args.m:>10d}              (design choice)")
    if any(k.is_override for k in consts.values()):
        print("\n  Lines marked NOT MEASURED were supplied by hand. They are as good as\n"
              "  your knowledge of them; nothing here checked them against data.")

    print("\nderived constants (tools/allocation.py: allocation_constants)")
    print(f"  Cb = {c['Cb']:.6f}   |bias| = Cb * rho^(-m0*omega1)")
    print(f"  Cs = {c['Cs']:.6f}   sd     = Cs * n^(-1/2)")
    print(f"  G  = {c['G']:.1f}       B      = n * rho^(d*m0) * G")
    print(f"  m0_tuned = theta2*log_rho(B) {c['offset']:+.3f}"
          f"   <- prop:opt omits this offset")
    print(f"  at the optimum, |bias|/sd = sqrt(d/(2*omega1)) = "
          f"{sqrt(d / (2 * omega1)):.4f}")

    print("\nerror budget -- how far each input's own uncertainty moves the plan")
    print(f"  {'input':<11}{'+/- 1 se':>12}{'moves m0 by':>14}{'worst-case RMSE':>18}")
    for name in ("omega1", "a1", "d"):
        k = consts[name]
        if k.se is None:
            print(f"  {name:<11}{'--':>12}{'not quantified (needs >= 2 replicates)':>32}")
            continue
        sens = input_sensitivity(name, k.se, d=d, omega1=omega1, rho=args.rho,
                                 m=args.m, a1=a1, cv=cv)
        print(f"  {name:<11}{k.se:>12.4f}{sens['delta_m0']:>+14.2f}"
              f"{sens['penalty']:>17.4f}x")
    print("  (the optimum is quadratic in m0, so it is flat: a whole step of m0"
          " costs ~1.1x)")
    if all(consts[n].se is None for n in ("omega1", "a1", "d")):
        print("  Nothing here has a standard error: every constant came from a single\n"
              "  run or was supplied by hand. Run replicates to quantify this.")

    print()

    if args.by_budget:
        budgets = [10.0 ** e for e in range(args.min_log10_budget, args.max_log10_budget + 1)]
        rows = build_budget_rows(budgets, d=d, omega1=omega1, rho=args.rho,
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

    rows = build_rows(range(args.min_m0, args.max_m0 + 1), d=d, omega1=omega1,
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
