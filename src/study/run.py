"""Step 3 of a study: execute the plan that was accepted.

Reads plan.json and draws it. Takes no scientific arguments at all -- m0, n,
the scale ladder and the replicate count all come from the plan, which came
from the constants, which came from the pilot. That chain is the point: the
numbers a run is sized by are never retyped, so they cannot drift between the
plan you approved and the run you got.

Samples are summarized to (y_bar, sigma_log) per scale and NOT kept by
default. The plan can easily ask for 10^7 samples on each of 6 scales, which
is half a gigabyte per replicate, and nothing downstream needs the draws
themselves -- report.py works from the summaries, exactly as
estimate_omega1.py does. Pass --keep-samples when you want them on disk.

CLI:
    python3 src/study/run.py --study mystudy --data-root experiments/01_srw/data
    python3 src/study/run.py --study mystudy --data-root ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "generate"))
sys.path.insert(0, str(ROOT / "src" / "budget"))

from artifacts import artifact_path, write_artifact  # noqa: E402
from models import get_model  # noqa: E402
from rng import seed_record, spawn  # noqa: E402

from allocation_table import human_time  # noqa: E402
from generate import generate  # noqa: E402


def summarize(s):
    """y_bar, sigma_log and cv for one scale, in one pass -- see pilot.py."""
    mean = float(np.mean(s))
    sd = float(np.std(s, ddof=1))
    return mean, sd / (np.sqrt(len(s)) * mean), sd / mean


def execute(plan: dict, recipe_params: dict, model: str, sd: Path, *,
            seed=None, keep_samples=False) -> dict:
    """Draw the plan's replicates; return the per-replicate summaries."""
    scales, n, R = plan["scales"], plan["n"], plan.get("replicates", 1)
    reps, seeds = [], []
    t0 = time.perf_counter()
    for k, ss in enumerate(spawn(seed, R)):
        print(f"  replicate {k + 1}/{R}  (n={n:,} x {len(scales)} scales) ...",
              end="", flush=True, file=sys.stderr)
        t = time.perf_counter()
        if keep_samples:
            out = generate(model, scales, n, recipe_params, seed=ss,
                           out_dir=sd, tag=f"samples/rep{k}")
            stats = {i: summarize(out[i]) for i in scales}
        else:
            stats = generate(model, scales, n, recipe_params, seed=ss,
                             reduce=summarize)
        reps.append({"y_bar": [stats[i][0] for i in scales],
                     "sigma_log": [stats[i][1] for i in scales],
                     "cv": [stats[i][2] for i in scales]})
        seeds.append(seed_record(ss))
        print(f" {time.perf_counter() - t:.1f}s", file=sys.stderr)
    return {"replicates": R, "scales": scales, "n": n, "m0": plan["m0"],
            "rho": plan["rho"], "m": plan["m"], "model": model,
            "params": recipe_params, "per_replicate": reps, "seeds": seeds,
            "elapsed_seconds": time.perf_counter() - t0,
            "samples_kept": bool(keep_samples)}


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--keep-samples", action="store_true",
                   help="persist the raw draws as well as the summaries. Off by "
                        "default: a planned run is often hundreds of MB per replicate "
                        "and nothing downstream reads them")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be drawn and stop")
    a = p.parse_args(argv)

    sd = Path(a.data_root) / a.study
    pj = artifact_path(sd, "plan")
    if not pj.exists():
        raise SystemExit(
            f"no plan.json in {sd}\n"
            f"  Accept a plan first:\n"
            f"    python3 src/study/plan.py --study {a.study} "
            f"--data-root {a.data_root} --time <duration> --accept")
    plan = json.loads(pj.read_text())

    pj_pilot = artifact_path(sd, "pilot")
    if not pj_pilot.exists():
        raise SystemExit(f"no pilot.json in {sd}; the plan does not know what model to draw")
    pilot = json.loads(pj_pilot.read_text())
    if "recipe" not in pilot:
        raise SystemExit(
            f"{pj_pilot} has no 'recipe' -- it was not written by src/study/pilot.py.\n"
            f"  run.py needs the model and params the pilot used.")
    model = pilot["recipe"]["model"]
    params = pilot["recipe"].get("params", {})

    print(f"study      = {sd}")
    print(f"model      = {model}  params={params}")
    print(f"plan       = m0={plan['m0']}  scales={plan['scales']}")
    print(f"             n={plan['n']:,} per scale x {plan.get('replicates', 1)}")
    print(f"expected   = {human_time(plan.get('total_seconds', plan['seconds']))}, "
          f"se(gamma) ~ {plan['rmse']:.4g} per replicate")
    if a.dry_run:
        print("\n--dry-run: nothing drawn.")
        return

    print(file=sys.stderr)
    result = execute(plan, params, model, sd, seed=a.seed,
                     keep_samples=a.keep_samples)
    result["plan"] = plan
    write_artifact(sd, "final", result, produced_by="src/study/run.py")

    got, want = result["elapsed_seconds"], plan.get("total_seconds")
    print(f"\ndrew {result['replicates']} replicate(s) in {human_time(got)}"
          + (f"  (predicted {human_time(want)}, ratio {got / want:.2f}x)"
             if want else ""))
    print(f"output = {artifact_path(sd, 'final')}")
    print(f"\nnext: python3 src/study/report.py --study {a.study} "
          f"--data-root {a.data_root}")


if __name__ == "__main__":
    _main()
