"""Experiment C: does Proposition prop:opt's budget allocation actually deliver
its promised accuracy for gamma-hat? (plans/three_experiment_ladder.md section 4)

Proposition prop:opt (eq. 945-946) says: given a budget B, spend it on m scales
rho^k for k = m0+1..m0+m with a uniform n per scale, choosing

    n  = kappa * B**theta1,   m0 = theta2 * log_rho(B),
    theta1 = 2*omega1/(d + 2*omega1),   theta2 = 1/(d + 2*omega1),

which is claimed to give error decay |gamma_hat - gamma| ~ B**(-omega1/(d+2*omega1)).

That claim has two separable halves, and this experiment tests them separately
because they can (and do) come apart:

  RATE  -- does the error really fall like B**(-omega1/(d+2*omega1))?
  POINT -- is the specific m0 it names the one that minimizes the error at a
           given finite B?

A rate theorem is only asymptotic-up-to-constants, so POINT can fail while RATE
holds. Measuring only the allocation the theorem names, with no comparison arm,
could not tell the difference -- which is why this script sweeps m0 rather than
evaluating a single point (the "control arm" of the original plan, generalized:
every other m0 at the same budget is a control).

Design notes:

- Estimator is `tools/loglog.py`'s `gamma_closed_form`, the article's own
  closed-form w_{k,m} weights (eq. 523-526), since prop:opt is stated for that
  estimator and this ladder is exactly the consecutive rho^k grid it requires.
  `gamma_all_points` (generic OLS) is recorded alongside as a cross-check.
- Every (budget, m0, replicate) cell draws fresh independent randomness via
  SeedSequence.spawn (ground rule 2). Samples are NOT persisted: the sweep
  draws far more than it is worth storing, and the per-cell gamma-hat values
  (which ARE persisted) are the actual result. The base seed is recorded, and
  spawning is deterministic, so any cell can be regenerated exactly.
- `true_gamma` is used ONLY to score a finished estimate. It is never passed to
  an estimator (see experiments/01_srw/README.md on why srw has no target_fn).

CLI:
    python3 src/budget/allocation_experiment.py -meta experiments/01_srw/recipes/sweep_allocation.json \\
        --tag allocation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from artifacts import artifact_path, default_out_dir, load_recipe, write_artifact  # noqa: E402
from allocation import (  # noqa: E402
    optimal_allocation,
    total_cost,
    tuned_allocation,
)
from loglog import gamma_all_points, gamma_closed_form  # noqa: E402
from persistence import run_dir as _run_dir  # noqa: E402
from rng import spawn  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generate"))   # the shared draw loop
from generate import generate  # noqa: E402


def ladder(m0: int, m: int, rho: float) -> list[int]:
    """The scales of Definition def:alloc: rho**k for k = m0+1, ..., m0+m.

    Rounds to integers, and refuses to return a grid where that rounding has
    collided. At rho = 1.5, m0 = 0, m = 6 the naive result is
    [2, 2, 3, 5, 8, 11]: two scales equal, so the ladder has m-1 distinct
    points while every downstream formula assumes m. `gamma_closed_form` would
    catch it -- its grid check rejects anything that is not exactly rho**k --
    but only after the samples had been drawn, and `total_cost` would already
    have charged for the duplicate. Raise here instead.
    """
    if m < 2:
        raise ValueError(f"m must be >= 2 for the weights to exist; got {m}")
    scales = [int(round(rho ** k)) for k in range(m0 + 1, m0 + m + 1)]
    if len(set(scales)) != len(scales):
        raise ValueError(
            f"rho={rho} with m0={m0}, m={m} rounds to a grid with repeated "
            f"scales {scales}; use an integer rho (or a larger m0) so that "
            f"rho**k are distinct integers")
    return scales


def n_for_budget(budget: float, m0: int, m: int, rho: float, d: float) -> int:
    """Largest uniform per-scale n whose ladder cost stays within `budget`."""
    return int(budget / total_cost(1.0, m0, m, rho, d))


def run_cell(
    model: str,
    params: dict,
    m0: int,
    m: int,
    rho: float,
    n: int,
    seed_seq: np.random.SeedSequence,
) -> dict:
    """One replicate: draw the ladder, return both gamma-hat estimates.

    The draw itself is `src/generate/generate.py`'s, not a second copy of it --
    `reduce=np.mean` because only the per-scale mean is ever used, which retains
    one scale's array rather than the whole ladder's (381 MB -> 63 MB at this
    sweep's largest cell). `seed_seq` is a
    SPAWNED child (ground rule 2) and is handed over as a SeedSequence, never
    as `seed_seq.entropy`: a child carries its parent's entropy, so the int
    spelling would give every replicate the same stream (tools/rng.py).
    """
    scales = ladder(m0, m, rho)
    means = generate(model, scales, n, params, seed=seed_seq, reduce=np.mean)
    y_bar = np.array([float(means[i]) for i in scales])
    return {
        "closed_form": float(gamma_closed_form(scales, y_bar, rho, m0)),
        "all_points": float(gamma_all_points(scales, y_bar)),
    }


def sweep(
    model: str,
    params: dict,
    budgets,
    m0_values,
    *,
    m: int,
    rho: float,
    d: float,
    omega1: float,
    replicates: int,
    true_gamma: float,
    seed: int | None = None,
    progress: bool = False,
    a1: float | None = None,
    cv: float | None = None,
) -> dict:
    """RMSE of gamma-hat over a (budget x m0) grid, `replicates` draws per cell.

    PAIRED ARMS. Two named allocations are scored on the same grid, against the
    same empirical argmin:

      prop:opt  -- `optimal_allocation`, the theorem's m0 = theta2 log_rho(B),
                   correct up to the multiplicative constant it discards.
      tuned     -- `tuned_allocation`, the same rule with that constant
                   restored from measured a1 and cv (see tools/allocation.py's
                   `allocation_constants`).

    Running them as a pair, rather than replacing one with the other, is the
    point: the untuned arm is the claim under test and the tuned arm is the
    proposed fix, and only scoring both on identical draws shows what the
    constant is worth. Pass `a1` and `cv` to enable the tuned arm; without them
    only prop:opt is marked.
    """
    seed_seq = np.random.SeedSequence(seed)
    # One independent stream per (budget, m0, replicate) cell -- ground rule 2.
    streams = iter(spawn(seed_seq, len(budgets) * len(m0_values) * replicates))

    have_tuned = a1 is not None and cv is not None
    cells = []
    t0 = time.perf_counter()
    for B in budgets:
        opt = optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)
        tun = (tuned_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m, a1=a1, cv=cv)
               if have_tuned else None)
        for m0 in m0_values:
            n = n_for_budget(B, m0, m, rho, d)
            if n < 2:
                # Skip rather than fabricate: a ladder this deep cannot be
                # afforded at this budget, which is information, not an error.
                cells.append({"budget": B, "m0": m0, "n": n, "skipped": True})
                for _ in range(replicates):
                    next(streams)
                continue
            hats = [run_cell(model, params, m0, m, rho, n, next(streams))
                    for _ in range(replicates)]
            cell = {"budget": B, "m0": m0, "n": n, "skipped": False,
                    "scales": ladder(m0, m, rho),
                    "cost": total_cost(n, m0, m, rho, d),
                    "is_prop_opt_m0": m0 == opt["m0"],
                    "is_tuned_m0": bool(tun and m0 == tun["m0"])}
            for key in ("closed_form", "all_points"):
                v = np.array([h[key] for h in hats])
                cell[key] = {
                    "mean": float(v.mean()),
                    "bias": float(v.mean() - true_gamma),
                    "sd": float(v.std(ddof=1)),
                    "rmse": float(sqrt(np.mean((v - true_gamma) ** 2))),
                    "values": [float(x) for x in v],
                }
            cells.append(cell)
            if progress:
                print(f"\r  B={B:.0e} m0={m0:>3} n={n:>10,} "
                      f"rmse={cell['closed_form']['rmse']:.3e} "
                      f"({time.perf_counter() - t0:.0f}s)" + " " * 8,
                      end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)

    return {
        "model": model,
        "params": params,
        "m": m,
        "rho": rho,
        "d": d,
        "omega1": omega1,
        "replicates": replicates,
        "true_gamma": true_gamma,
        "seed": seed_seq.entropy,
        "budgets": list(budgets),
        "m0_values": list(m0_values),
        "cells": cells,
        "a1": a1,
        "cv": cv,
        "prop_opt": {
            str(B): optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)
            for B in budgets
        },
        "tuned": ({
            str(B): tuned_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m,
                                     a1=a1, cv=cv)
            for B in budgets
        } if have_tuned else None),
        "elapsed_seconds": time.perf_counter() - t0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def summarize(result: dict, estimator: str = "closed_form") -> dict:
    """Per-budget: each arm's m0, the empirically best m0, and the price of each gap.

    `penalty_factor` (prop:opt) and `penalty_tuned` are RMSE relative to the
    best m0 available on the grid -- the honest yardstick, since the best cell
    is measured on the same draws rather than predicted.
    """
    out = {}
    tuned_all = result.get("tuned")
    for B in result["budgets"]:
        rows = [c for c in result["cells"]
                if c["budget"] == B and not c["skipped"]]
        if not rows:
            continue
        best = min(rows, key=lambda c: c[estimator]["rmse"])
        predicted = result["prop_opt"][str(B)]["m0"]
        at_pred = next((c for c in rows if c["m0"] == predicted), None)
        entry = {
            "prop_opt_m0": predicted,
            "best_m0": best["m0"],
            "best_rmse": best[estimator]["rmse"],
            "rmse_at_prop_opt_m0": at_pred[estimator]["rmse"] if at_pred else None,
            "penalty_factor": (at_pred[estimator]["rmse"] / best[estimator]["rmse"])
            if at_pred else None,
            "tuned_m0": None,
            "rmse_at_tuned_m0": None,
            "penalty_tuned": None,
        }
        if tuned_all and tuned_all.get(str(B)):
            tm0 = tuned_all[str(B)]["m0"]
            at_tuned = next((c for c in rows if c["m0"] == tm0), None)
            entry["tuned_m0"] = tm0
            if at_tuned:
                entry["rmse_at_tuned_m0"] = at_tuned[estimator]["rmse"]
                entry["penalty_tuned"] = (at_tuned[estimator]["rmse"]
                                          / best[estimator]["rmse"])
        out[str(B)] = entry
    return out


def rate_exponent(budgets, rmses) -> float:
    """Slope of log(rmse) vs log(budget) -- the measured error-decay exponent."""
    x, y = np.log(np.asarray(budgets, float)), np.log(np.asarray(rmses, float))
    return float(np.polyfit(x, y, 1)[0])


def rate_exponent_se(budgets, replicates: int) -> float:
    """Standard error of `rate_exponent`, from the noise in each RMSE.

    An RMSE estimated from R replicates is itself a random quantity: it is
    sqrt of a mean of squares, so its RELATIVE standard deviation is about
    1/sqrt(2R), and hence sd(log RMSE) ~ 1/sqrt(2R) too. The slope of an OLS
    line through points with that much scatter has

        se(slope) = sd(log rmse) / sqrt(Sxx),   Sxx = sum (log B - mean log B)^2

    Derived from the known noise rather than from the fit residuals on
    purpose: these sweeps use 3-4 budgets, so a residual-based estimate would
    have 1-2 degrees of freedom and be nearly useless. Without this the
    measured exponent looks exact, and comparing it to a predicted one with
    error bars silently overstates any disagreement.
    """
    x = np.log(np.asarray(budgets, float))
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx <= 0 or replicates < 1:
        return float("nan")
    return float((1.0 / sqrt(2 * replicates)) / sqrt(sxx))


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-meta", "--meta", dest="meta", required=True, type=Path)
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--tag", dest="tag", type=str, default="allocation")
    args = parser.parse_args(argv)

    cfg = load_recipe(args.meta, "sweep")
    result = sweep(
        cfg["model"], cfg["params"], cfg["budgets"], cfg["m0_values"],
        m=cfg["m"], rho=cfg["rho"], d=cfg["d"], omega1=cfg["omega1"],
        replicates=cfg["replicates"], true_gamma=cfg["true_gamma"],
        seed=cfg.get("seed"), progress=True,
        a1=cfg.get("a1"), cv=cfg.get("cv"),
    )

    out_dir = args.out_dir or default_out_dir(args.meta)
    rd = _run_dir(out_dir, args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    write_artifact(rd, "allocation_sweep", result,
                   produced_by="src/budget/allocation_experiment.py",
                   recipe=args.meta)

    summary = summarize(result)
    theory_rate = -cfg["omega1"] / (cfg["d"] + 2 * cfg["omega1"])

    print(f"\nmodel={result['model']!r}  m={result['m']}  rho={result['rho']}  "
          f"d={result['d']}  omega1={result['omega1']}  replicates={result['replicates']}")
    print(f"seed={result['seed']}  elapsed={result['elapsed_seconds']:.0f}s\n")

    for B in result["budgets"]:
        rows = [c for c in result["cells"] if c["budget"] == B and not c["skipped"]]
        print(f"budget B = {B:.0e}")
        print(f"  {'m0':>4} {'n':>12} {'bias':>11} {'sd':>11} {'rmse':>11}")
        for c in rows:
            s = c["closed_form"]
            mark = ("  <-- prop:opt" if c["is_prop_opt_m0"] else "") + \
                   ("  <-- tuned" if c.get("is_tuned_m0") else "")
            print(f"  {c['m0']:>4} {c['n']:>12,} {s['bias']:>11.3e} {s['sd']:>11.3e} "
                  f"{s['rmse']:>11.3e}{mark}")
        s = summary[str(B)]
        line = (f"  best m0={s['best_m0']}   |   prop:opt m0={s['prop_opt_m0']}"
                + (f" ({s['penalty_factor']:.2f}x)" if s["penalty_factor"] else ""))
        if s["tuned_m0"] is not None:
            line += (f"   |   tuned m0={s['tuned_m0']}"
                     + (f" ({s['penalty_tuned']:.2f}x)" if s["penalty_tuned"] else ""))
        print(line + "\n")

    # PAIRED ARMS side by side: the constant's worth, in one table.
    if result.get("tuned"):
        print("paired allocation arms (RMSE relative to the best m0 on the grid):")
        print(f"  {'B':>9} {'best m0':>8} | {'prop:opt m0':>11} {'penalty':>8} "
              f"| {'tuned m0':>9} {'penalty':>8}")
        for B in result["budgets"]:
            s = summary.get(str(B))
            if not s:
                continue
            pf = f"{s['penalty_factor']:.2f}x" if s["penalty_factor"] else "--"
            pt = f"{s['penalty_tuned']:.2f}x" if s["penalty_tuned"] else "--"
            tm = s["tuned_m0"] if s["tuned_m0"] is not None else "--"
            print(f"  {B:>9.0e} {s['best_m0']:>8} | {s['prop_opt_m0']:>11} {pf:>8} "
                  f"| {str(tm):>9} {pt:>8}")
        print()

    # RATE: does the error fall like B**(-omega1/(d+2*omega1))?
    print("error-decay rate, log(rmse) vs log(B):")
    arms = [("at prop:opt's m0", "prop_opt_m0"), ("at the best m0", "best_m0")]
    if result.get("tuned"):
        arms.insert(1, ("at the tuned m0", "tuned_m0"))
    for label, pick in arms:
        bs, rs = [], []
        for B in result["budgets"]:
            s = summary.get(str(B))
            if not s:
                continue
            r = {"prop_opt_m0": s["rmse_at_prop_opt_m0"],
                 "tuned_m0": s.get("rmse_at_tuned_m0"),
                 "best_m0": s["best_rmse"]}[pick]
            if r:
                bs.append(B)
                rs.append(r)
        if len(bs) >= 3:
            got = rate_exponent(bs, rs)
            print(f"  {label:<18}: measured {got:+.4f}   theory {theory_rate:+.4f}   "
                  f"{'OK' if abs(got - theory_rate) < 0.08 else 'MISMATCH'}")
        else:
            print(f"  {label:<18}: need >=3 budgets, have {len(bs)}")

    print(f"\noutput = {artifact_path(rd, 'allocation_sweep')}")


if __name__ == "__main__":
    _main()
