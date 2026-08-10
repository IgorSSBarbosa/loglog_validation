"""Checkpoint 0.1 acceptance script — "generator matches its own formula".

See ../../PLAN.md, "Phase 0" table: passes when, at each tested scale i, the
empirical mean of `generate()` draws is within 3 Monte-Carlo standard errors
of the exact target E[Y_i] = a0 * i**gamma * exp(a1 * i**(-omega1)).

Unlike a real model, this generator's E[Y_i] equals the formula exactly by
construction for every i (not just asymptotically), so "large i" carries no
special asymptotic meaning here — testing across several widely separated
scales is a stronger, uniformly valid check of the same fidelity, not a
different one. We use i = 2**10, 2**15, 2**20.

The seed is left unset (fresh entropy each run, per PLAN.md's independence
rule) — pass/fail is a genuine Monte-Carlo draw each time, not a fixed
golden output. The generator's own metadata JSON (out_dir=..., recording the
seed entropy actually used) makes any given run fully reproducible after the
fact. Results are also summarized to a fixed-path result.json (overwritten
on rerun, not timestamped).

Run: python3 run_checkpoint_0_1.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from generator import SyntheticParams, generate, mean_Y

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "checkpoint_0_1"

# Planted constants agreed for Phase 0's first pass (see conversation / PLAN.md).
PARAMS = SyntheticParams(gamma=0.5, a0=1.0, a1=1.0, omega1=1.0, sigma_inf2=0.04, family="lognormal")
SCALES = [2**10, 2**15, 2**20]
N = 5000
N_SE = 3.0  # tolerance, in Monte-Carlo standard errors


def run() -> bool:
    samples = generate(SCALES, N, PARAMS, seed=None, out_dir=OUT_DIR, tag="checkpoint_0_1")

    rows = []
    all_passed = True
    for i in SCALES:
        y = samples[i]
        target = float(mean_Y(i, PARAMS))
        mean_hat = float(y.mean())
        # Exact SE of the sample mean under the planted lognormal xi_i:
        # Var(xi_i) = sigma_inf2 exactly, so Var(Y_i) = target^2 * sigma_inf2.
        se = np.sqrt(PARAMS.sigma_inf2) * target / np.sqrt(len(y))
        diff = mean_hat - target
        passed = bool(abs(diff) <= N_SE * se)
        all_passed = all_passed and passed
        rows.append(
            dict(i=i, target=target, mean_hat=mean_hat, se=se, diff=diff, tol=N_SE * se, passed=passed)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "checkpoint": "0.1",
                "params": asdict(PARAMS),
                "n": N,
                "n_se_tolerance": N_SE,
                "rows": rows,
                "passed": all_passed,
            },
            indent=2,
            sort_keys=True,
        )
    )

    header = f"{'i':>10} {'target':>14} {'mean_hat':>14} {'diff':>12} {'tol(3SE)':>12}  status"
    print(header)
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['i']:>10} {r['target']:>14.4f} {r['mean_hat']:>14.4f} "
            f"{r['diff']:>12.4f} {r['tol']:>12.4f}  {status}"
        )
    print()
    print("Checkpoint 0.1:", "PASS" if all_passed else "FAIL")
    print(f"Result written to {result_path}")
    print(f"Generator metadata written to {OUT_DIR / 'checkpoint_0_1.json'}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
