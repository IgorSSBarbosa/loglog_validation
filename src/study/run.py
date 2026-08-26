"""Step 3 of a study: execute the plan that was accepted.

Reads the recipe plan.py generated and draws it. Takes no scientific arguments
at all -- m0, n, the scale ladder and the replicate count all come from the
plan, which came from the constants, which came from the pilot. That chain is
the point: the numbers a run is sized by are never retyped, so they cannot
drift between the plan you approved and the run you got.

Two files, two jobs. `plan.json` is the DECISION -- the RMSE the allocation
should buy, the constants it was made from, the wall clock it should take --
and is what the run is reported against. `recipes/samples_<study>_final.json`
is the same allocation as an ordinary samples recipe, and is what is actually
drawn; `python3 src/generate/generate.py -meta <that file>` draws it too, with
no study machinery involved. What this driver adds over the bare generator is
replicates, on-the-fly summarizing, and the refusal to run at all until a plan
has been accepted.

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "generate"))
sys.path.insert(0, str(ROOT / "src" / "budget"))

from artifacts import artifact_path, load_recipe, write_artifact  # noqa: E402
from rng import seed_record, spawn  # noqa: E402
from summary import replicate_summary, summarize_scale  # noqa: E402

from allocation_table import human_time  # noqa: E402
from generate import generate  # noqa: E402


def execute(plan: dict, recipe: dict, sd: Path, *,
            seed=None, keep_samples=False) -> dict:
    """Draw the plan's replicates; return the per-replicate summaries.

    The recipe says WHAT to draw (model, params) and the plan says HOW MUCH
    (scales, n, replicates) -- they agree by construction, since plan.py wrote
    the recipe from the plan. Each replicate gets its own spawned seed, and
    the seed is recorded, so any one of them can be redrawn alone.
    """
    model, params = recipe["model"], recipe.get("params", {})
    scales, n, R = plan["scales"], plan["n"], plan.get("replicates", 1)
    reps, seeds = [], []
    t0 = time.perf_counter()
    for k, ss in enumerate(spawn(seed, R)):
        print(f"  replicate {k + 1}/{R}  (n={n:,} x {len(scales)} scales) ...",
              end="", flush=True, file=sys.stderr)
        t = time.perf_counter()
        if keep_samples:
            out = generate(model, scales, n, params, seed=ss,
                           out_dir=sd, tag=f"samples/rep{k}")
            stats = {i: summarize_scale(out[i]) for i in scales}
        else:
            # reduce= collapses each scale inside generate() and frees the
            # draws immediately: a planned run is routinely hundreds of MB per
            # replicate, and nothing downstream reads the samples themselves.
            stats = generate(model, scales, n, params, seed=ss,
                             reduce=summarize_scale)
        reps.append(replicate_summary(stats, scales))
        seeds.append(seed_record(ss))
        print(f" {time.perf_counter() - t:.1f}s", file=sys.stderr)
    return {"replicates": R, "scales": scales, "n": n, "m0": plan["m0"],
            "rho": plan["rho"], "m": plan["m"], "model": model,
            "params": params, "per_replicate": reps, "seeds": seeds,
            "elapsed_seconds": time.perf_counter() - t0,
            "samples_kept": bool(keep_samples)}


def _load_plan_recipe(plan: dict, sd: Path, study: str, data_root) -> dict:
    """The recipe the plan was accepted with.

    Since 2026-08-26 plan.py writes one and records its path. Studies planned
    before that have only pilot.json's copy of the pilot recipe, whose scales
    and n are the PILOT's -- wrong for a final run -- so only model and params
    are taken from it, exactly as this script used to do.
    """
    path = plan.get("recipe_path")
    if path and Path(path).exists():
        return load_recipe(path, "samples")

    pilot_path = artifact_path(sd, "pilot")
    if not pilot_path.exists():
        raise SystemExit(
            f"the plan names no recipe and there is no pilot.json in {sd}, so\n"
            f"  nothing says what to draw. Re-accept the plan:\n"
            f"    python3 src/study/plan.py --study {study} "
            f"--data-root {data_root} --time <duration> --accept")
    pilot = json.loads(pilot_path.read_text())
    if "recipe" not in pilot:
        raise SystemExit(
            f"{pilot_path} has no 'recipe' -- it was not written by "
            f"src/study/pilot.py.\n  run.py needs the model and params the "
            f"pilot used.")
    if path:
        print(f"note: {path} is gone; falling back to the pilot's model/params",
              file=sys.stderr)
    return pilot["recipe"]


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

    recipe = _load_plan_recipe(plan, sd, a.study, a.data_root)

    print(f"study      = {sd}")
    print(f"recipe     = {plan.get('recipe_path', '(from pilot.json)')}")
    print(f"model      = {recipe['model']}  params={recipe.get('params', {})}")
    print(f"plan       = m0={plan['m0']}  scales={plan['scales']}")
    print(f"             n={plan['n']:,} per scale x {plan.get('replicates', 1)}")
    print(f"expected   = {human_time(plan.get('total_seconds', plan['seconds']))}, "
          f"se(gamma) ~ {plan['rmse']:.4g} per replicate")
    if a.dry_run:
        print("\n--dry-run: nothing drawn.")
        return

    print(file=sys.stderr)
    result = execute(plan, recipe, sd, seed=a.seed,
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
