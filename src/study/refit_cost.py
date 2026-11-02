"""Step 2b of a study: re-fit the cost on a short run over the PLANNED window.

    pilot.py   measure d, omega1, a1, cv from a cheap run
    plan.py    choose m0 and n for a budget
    refit_cost.py  <- you are here: time the planned scales, re-fit eq. (cost)
    plan.py    again, on the re-fitted cost
    run.py     draw it

The article, Step 2, "Re-fitting the cost": when m0* + m exceeds the largest
pilot scale, eq. (n-of-m0) rests on an extrapolation of cost-hat, and memory,
cache hierarchy and fixed start-up costs can make the cost at large scales
differ from the pilot fit. It recommends re-fitting eq. (cost) on a short run
over the scales of the planned window. Made a standard step (user,
2026-10-31) after experiment 49's 9 h plans: d_f in d = 2 planned on 8..1024
drew 512..16384 at 3.3x its predicted time, d = 3 at 1.6x.

WHAT IT DOES. Reads plan.json and draws `n_short` samples at each scale of the
planned window in one `simulate` call per scale, timing each call. `n_short`
is REFIT_SHARE of the plan's total draws per scale (all replicates), so the
short run costs about that share of the remaining budget at the OLD cost
model; for a latency-bound model it is raised to a full device
(`device_fill`), or to one replicate's n when the model does not know its
device, because below saturation a call costs one sample's latency and the
per-sample time would be overstated. If that makes the short run cost more
than REFIT_MAX_SHARE of the budget left, it is skipped and says so.

Per-sample time t_i = elapsed_i / n_short is fitted by eq. (cost),
t(i) = kappa0 + kappa1 i^d (tools/cost_model.estimate_cost_affine), and
  - constants.json's d is replaced by the window's d, with provenance;
  - throughput = sum_i cost_hint(i) / sum_i t-hat(i) over the window, the same
    formula pilot.py uses for a batched model, goes into cost_refit.json,
    which plan.py prefers over pilot.json's throughput.
Every round is appended to cost_refit.json's `rounds`, with what the old cost
model predicted for the same short run, so the size of the correction is on
record.

The draws are timed and thrown away, like a cost probe's: they come from their
own SeedSequence (autopilot passes a spawned child) and never reach a summary.

CLI (step by step; autopilot.py runs this by default):
    python3 src/study/refit_cost.py --study mystudy --data-root experiments/01_srw/data
    python3 src/study/plan.py --study mystudy --data-root ... --time 9h --accept
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools.constants import load, measured, save  # noqa: E402
from tools.cost_model import estimate_cost_affine  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools.rng import as_seed_sequence, seed_record  # noqa: E402

from src.budget.allocation_table import human_time  # noqa: E402

#: Share of the planned draws (per scale, all replicates) the short run takes:
#: about 1% of the remaining budget at the old cost model.
REFIT_SHARE = 0.01

#: Above this share of the budget left, the short run is not worth its cost
#: and is skipped (a latency-bound model whose device size is unknown needs one
#: whole replicate's n per scale, 1/R of the run).
REFIT_MAX_SHARE = 0.10

#: Fixed stream for the warm-up call, as pilot.py's COST_PROBE_SEED: it pays
#: one-time device setup and touches no other stream.
_WARMUP_SEED = 0


def short_run_size(spec, plan: dict) -> int:
    """n_short for this plan and model (see the module docstring)."""
    n_plan, R = int(plan["n"]), int(plan.get("replicates", 1))
    n = max(1, int(n_plan * R * REFIT_SHARE))
    if spec.latency_bound:
        floor = spec.device_fill() if spec.device_fill is not None else n_plan
        n = max(n, min(floor, n_plan))
    return n


def refit_cost(sd: Path, *, seed, seconds_left: float | None = None,
               log=print) -> dict:
    """One re-fit round on the accepted plan's window. Returns the round's record.

    `seconds_left` is the budget still unspent, for the REFIT_MAX_SHARE check;
    None skips the check (step-by-step use, where the user decides).
    """
    ss = as_seed_sequence(seed)    # once, so a fresh-entropy seed is the one recorded
    plan = read_artifact(sd, "plan")
    pilot = read_artifact(sd, "pilot")
    recipe = pilot["recipe"]
    model, params = recipe["model"], recipe.get("params", {})
    spec = get_model(model)
    scales = [int(i) for i in plan["scales"]]
    n_short = short_run_size(spec, plan)
    prior = (read_artifact(sd, "cost_refit")
             if artifact_path(sd, "cost_refit").exists() else None)
    if prior and prior.get("pilot_created") != pilot.get("created"):
        prior = None              # left over from an earlier pilot in this directory
    tp_before = prior["throughput"] if prior else plan["throughput"]
    consts = load(sd)
    d_before = consts["d"].value

    def work(i, d):
        return spec.cost_hint(i, params) if spec.cost_hint else float(i) ** d

    predicted = n_short * sum(work(i, d_before) for i in scales) / tp_before
    rec = {"scales": scales, "n_short": n_short, "n_plan": int(plan["n"]),
           "replicates": int(plan.get("replicates", 1)),
           "predicted_seconds_before": predicted,
           "d_before": d_before, "throughput_before": tp_before,
           "seed": seed_record(ss)}
    log(f"\ncost re-fit: {n_short:,} sample(s) at each of {scales[0]}..{scales[-1]} "
        f"(the planned window), predicted {human_time(predicted)} at the current "
        f"cost model")
    if seconds_left is not None and predicted > REFIT_MAX_SHARE * seconds_left:
        rec.update(skipped=True,
                   why=(f"the short run would take {human_time(predicted)}, more "
                        f"than {REFIT_MAX_SHARE:g} of the {human_time(seconds_left)} "
                        f"left"))
        log(f"  skipped: {rec['why']}. The plan keeps the pilot's cost model.")
        return _append(sd, prior, rec, pilot)

    rng = np.random.default_rng(ss)
    spec.simulate(scales[0], 1, params, np.random.default_rng(_WARMUP_SEED))
    elapsed = []
    t_start = time.perf_counter()
    for i in scales:
        t0 = time.perf_counter()
        spec.simulate(i, n_short, params, rng)
        elapsed.append(time.perf_counter() - t0)
        log(f"  i = {i}: {elapsed[-1]:.3f} s, {elapsed[-1] / n_short:.4g} s per sample")
    spent = time.perf_counter() - t_start
    per_sample = [t / n_short for t in elapsed]
    aff = estimate_cost_affine(scales, per_sample)
    d_new = float(aff["d"])
    tp = (sum(work(i, d_new) for i in scales)
          / sum(aff["a"] + aff["b"] * float(i) ** d_new for i in scales))
    rec.update(skipped=False, elapsed=elapsed, per_sample=per_sample, affine=aff,
               d=d_new, throughput=tp, seconds=spent,
               ratio_to_predicted=spent / predicted)
    log(f"  drew in {human_time(spent)} ({spent / predicted:.2f}x the prediction); "
        f"eq. (cost) on the window: d = {d_new:.4f}"
        + (f" +/- {aff['d_se']:.4f}" if aff.get("d_se") is not None else "")
        + f", kappa0 = {aff['a']:.3g} s; throughput {tp_before:.4g} -> {tp:.4g} /s")

    consts["d"] = measured(d_new, aff.get("d_se"),
                           f"cost re-fit, {n_short:,} samples on the planned "
                           f"window {scales[0]}..{scales[-1]}")
    save(sd, consts)
    return _append(sd, prior, rec, pilot)


def _append(sd: Path, prior: dict | None, rec: dict, pilot: dict) -> dict:
    """Write cost_refit.json: every round, and the throughput plan.py should use.

    `pilot_created` ties it to the pilot it re-fits: plan.py ignores a re-fit
    whose pilot has since been redrawn in the same directory."""
    rounds = (prior or {}).get("rounds", []) + [rec]
    last = next((r for r in reversed(rounds) if not r["skipped"]), None)
    write_artifact(sd, "cost_refit", {
        "pilot_created": pilot.get("created"),
        "rounds": rounds,
        "throughput": last["throughput"] if last else rec["throughput_before"],
        "scales": last["scales"] if last else None,
    }, produced_by="src/study/refit_cost.py")
    return rec


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True,
                   help="experiment data dir holding <study>/ with an accepted plan.json")
    p.add_argument("--seed", default=None,
                   help="seed of the short run's draws (default: fresh entropy, recorded)")
    a = p.parse_args(argv)
    sd = Path(a.data_root) / a.study
    if not artifact_path(sd, "plan").exists():
        raise SystemExit(f"no plan.json in {sd}: accept a plan first "
                         f"(src/study/plan.py ... --accept)")
    seed = int(a.seed) if a.seed is not None else None
    refit_cost(sd, seed=seed)
    print(f"\nnext: python3 src/study/plan.py --study {a.study} --data-root "
          f"{a.data_root} --time <the budget left> --accept")


if __name__ == "__main__":
    _main()
