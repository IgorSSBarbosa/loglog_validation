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
    python3 allocation_experiment.py -meta ../experiments/01_srw/allocation_config.json \\
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
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports

from allocation import optimal_allocation, total_cost  # noqa: E402
from loglog import gamma_all_points, gamma_closed_form  # noqa: E402
from models import get_model  # noqa: E402
from persistence import run_dir as _run_dir  # noqa: E402


def ladder(m0: int, m: int, rho: float) -> list[int]:
    """The scales of Definition def:alloc: rho**k for k = m0+1, ..., m0+m."""
    return [int(round(rho ** k)) for k in range(m0 + 1, m0 + m + 1)]


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
    """One replicate: draw the ladder, return both gamma-hat estimates."""
    spec = get_model(model)
    rng = np.random.default_rng(seed_seq)
    scales = ladder(m0, m, rho)
    y_bar = np.empty(len(scales))
    for j, i in enumerate(scales):
        y_bar[j] = float(np.mean(spec.simulate(i, n, params, rng)))
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
) -> dict:
    """RMSE of gamma-hat over a (budget x m0) grid, `replicates` draws per cell."""
    seed_seq = np.random.SeedSequence(seed)
    # One independent stream per (budget, m0, replicate) cell -- ground rule 2.
    streams = iter(seed_seq.spawn(len(budgets) * len(m0_values) * replicates))

    cells = []
    t0 = time.perf_counter()
    for B in budgets:
        opt = optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)
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
                    "is_prop_opt_m0": m0 == opt["m0"]}
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
        "prop_opt": {
            str(B): optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)
            for B in budgets
        },
        "elapsed_seconds": time.perf_counter() - t0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def summarize(result: dict, estimator: str = "closed_form") -> dict:
    """Per-budget: prop:opt's m0, the empirically best m0, and the price of the gap."""
    out = {}
    for B in result["budgets"]:
        rows = [c for c in result["cells"]
                if c["budget"] == B and not c["skipped"]]
        if not rows:
            continue
        best = min(rows, key=lambda c: c[estimator]["rmse"])
        predicted = result["prop_opt"][str(B)]["m0"]
        at_pred = next((c for c in rows if c["m0"] == predicted), None)
        out[str(B)] = {
            "prop_opt_m0": predicted,
            "best_m0": best["m0"],
            "best_rmse": best[estimator]["rmse"],
            "rmse_at_prop_opt_m0": at_pred[estimator]["rmse"] if at_pred else None,
            "penalty_factor": (at_pred[estimator]["rmse"] / best[estimator]["rmse"])
            if at_pred else None,
        }
    return out


def rate_exponent(budgets, rmses) -> float:
    """Slope of log(rmse) vs log(budget) -- the measured error-decay exponent."""
    x, y = np.log(np.asarray(budgets, float)), np.log(np.asarray(rmses, float))
    return float(np.polyfit(x, y, 1)[0])


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-meta", "--meta", dest="meta", required=True, type=Path)
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--tag", dest="tag", type=str, default="allocation")
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    result = sweep(
        cfg["model"], cfg["params"], cfg["budgets"], cfg["m0_values"],
        m=cfg["m"], rho=cfg["rho"], d=cfg["d"], omega1=cfg["omega1"],
        replicates=cfg["replicates"], true_gamma=cfg["true_gamma"],
        seed=cfg.get("seed"), progress=True,
    )

    out_dir = args.out_dir or (args.meta.resolve().parent / "data")
    rd = _run_dir(out_dir, args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))

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
            mark = "  <-- prop:opt" if c["is_prop_opt_m0"] else ""
            print(f"  {c['m0']:>4} {c['n']:>12,} {s['bias']:>11.3e} {s['sd']:>11.3e} "
                  f"{s['rmse']:>11.3e}{mark}")
        s = summary[str(B)]
        print(f"  prop:opt m0={s['prop_opt_m0']}, empirically best m0={s['best_m0']}"
              + (f", cost of the gap: {s['penalty_factor']:.2f}x RMSE"
                 if s["penalty_factor"] else "")
              + "\n")

    # RATE: does the error fall like B**(-omega1/(d+2*omega1))?
    print("error-decay rate, log(rmse) vs log(B):")
    for label, pick in (("at prop:opt's m0", "prop_opt_m0"), ("at the best m0", "best_m0")):
        bs, rs = [], []
        for B in result["budgets"]:
            s = summary.get(str(B))
            if not s:
                continue
            r = s["rmse_at_prop_opt_m0"] if pick == "prop_opt_m0" else s["best_rmse"]
            if r:
                bs.append(B)
                rs.append(r)
        if len(bs) >= 3:
            got = rate_exponent(bs, rs)
            print(f"  {label:<18}: measured {got:+.4f}   theory {theory_rate:+.4f}   "
                  f"{'OK' if abs(got - theory_rate) < 0.08 else 'MISMATCH'}")
        else:
            print(f"  {label:<18}: need >=3 budgets, have {len(bs)}")

    print(f"\noutput = {rd / 'result.json'}")


if __name__ == "__main__":
    _main()
