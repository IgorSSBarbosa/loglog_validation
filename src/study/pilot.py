"""Step 1 of a study: a short run that measures the constants a plan needs.

The workflow this belongs to (see src/study/README.md):

    pilot.py   <- you are here: measure d, omega1, a1, cv from a cheap run
    plan.py       what a longer run would cost, and whether the pilot is good
                  enough to plan on
    run.py        execute the accepted plan
    report.py     gamma-hat with its error, the log-log plot, the details

A study is one directory that accumulates state, so no constant is ever
copied by hand between steps. Before this existed you had to read omega1 out
of omega1.json, d out of cost_probe.json, and type both into
allocation_table.py -- and if you forgot, it silently used 1.0 for each.

Replicates matter here and the default is 1 on purpose. One replicate gives
every constant a value and NO standard error, so `plan` cannot tell you how
well determined the plan is. omega1 is the one that hurts: measured across
single replicates of the same configuration it ranged 0.49 to 1.25. Use
--replicates 3+ when you intend to act on the answer; --more adds replicates
to an existing pilot without redrawing the ones you have.

CLI:
    python3 src/study/pilot.py -meta experiments/01_srw/recipes/samples_pilot.json \\
        --study mystudy
    python3 src/study/pilot.py --study mystudy --more 4     # tighten an existing pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "generate"))
sys.path.insert(0, str(ROOT / "src" / "estimate"))

from artifacts import artifact_path, default_out_dir, load_recipe, write_artifact  # noqa: E402
from constants import format_table, measured, save  # noqa: E402
from correction import fit_correction  # noqa: E402
from cost_model import estimate_cost_affine  # noqa: E402
from models import get_model  # noqa: E402
from persistence import load_samples  # noqa: E402
from rng import spawn  # noqa: E402

from generate import generate, resolve_n  # noqa: E402

#: Repeats per scale when timing the cost. Small: the affine fit needs the
#: shape of cost(i), not a precise absolute time.
COST_REPEATS = 5

#: A single simulate(k, n=1, ...) call pays a fixed ~10-25 us of Python/NumPy
#: dispatch that does not scale with k at all. Below this, that overhead IS the
#: measurement: timing the pilot's own ladder (8..256 for srw) returned
#: d = 8.0 +/- 280. The probe therefore climbs away from the sample scales
#: until one call is slow enough for the work to dominate.
COST_TARGET_SECONDS = 2e-3

#: Ceiling on the climb, so a model with a steep cost cannot hang the pilot.
COST_MAX_DOUBLINGS = 24
COST_TIME_BUDGET = 20.0


def study_dir(root: Path, name: str) -> Path:
    """Where a study lives: alongside every other run of that experiment."""
    return Path(root) / name


def _pilot_replicate(model, params, scales, n, seed_seq) -> dict:
    """One replicate: y_bar, its sigma_log, and the cv, from a single draw.

    Returns the same three summaries estimate_omega1.py records, so a pilot
    replicate and an Experiment B replicate are interchangeable downstream.
    """
    def summarize(s):
        mean = float(np.mean(s))
        sd = float(np.std(s, ddof=1))
        return mean, sd / (np.sqrt(len(s)) * mean), sd / mean   # y_bar, sigma_log, cv

    stats = generate(model, scales, n, params, seed=seed_seq, reduce=summarize)
    return {"y_bar": [stats[i][0] for i in scales],
            "sigma_log": [stats[i][1] for i in scales],
            "cv": [stats[i][2] for i in scales]}


def _time_one(spec, i, params, rng, repeats) -> float:
    spec.simulate(i, 1, params, rng)                       # warm the code path
    t0 = time.perf_counter()
    for _ in range(repeats):
        spec.simulate(i, 1, params, rng)
    return (time.perf_counter() - t0) / repeats


def measure_cost_exponent(model, params, scales, repeats=COST_REPEATS) -> dict:
    """Time simulate() per scale and fit cost(i) = a + b*i^d.

    Two things make this harder than it looks, and both bit on the first run.

    The affine form, not the pure power law: a single simulate(k, n=1, ...)
    call pays a fixed per-call overhead that does not scale with k, and the
    pure fit folds it into d (0.78 against a true 1.0).

    And the probe must not use the pilot's own scale ladder. The ladder is
    chosen so the CORRECTION term is visible, which means small scales -- and
    at srw's 8..256 a single call is almost entirely dispatch, giving
    d = 8.0 +/- 280. So the probe climbs geometrically from the largest sample
    scale until one call takes COST_TARGET_SECONDS, bounded by a doubling count
    and a wall-clock budget so a steep model cannot hang the pilot.

    A model that declares `cost_hint` gets that reported too: it is exact where
    the clock is not, and disagreement means either the hint is wrong or the
    machine has stopped being compute-bound.
    """
    spec = get_model(model)
    rng = np.random.default_rng(0)

    probe, elapsed = [], []
    i = int(max(scales))
    t_start = time.perf_counter()
    for _ in range(COST_MAX_DOUBLINGS):
        t = _time_one(spec, i, params, rng, repeats)
        probe.append(i)
        elapsed.append(t)
        if t >= COST_TARGET_SECONDS and len(probe) >= 4:
            break
        if time.perf_counter() - t_start > COST_TIME_BUDGET:
            break
        i *= 2

    out = {"scales": probe, "elapsed": elapsed, "repeats": repeats,
           "reached_target": bool(elapsed and elapsed[-1] >= COST_TARGET_SECONDS)}
    try:
        aff = estimate_cost_affine(probe, elapsed)
        out["affine"] = aff
        if elapsed and aff.get("a") is not None:
            out["overhead_share"] = float(aff["a"] / elapsed[0]) if elapsed[0] else None
    except ValueError as exc:
        out["affine"] = {"error": str(exc)}
    if spec.cost_hint is not None:
        from cost_model import declared_exponent
        out["declared_d"] = declared_exponent(probe, spec.cost_hint, params)
    return out


def pilot(recipe: dict, sd: Path, replicates: int, seed=None,
          existing: list | None = None) -> dict:
    """Draw `replicates` replicates, fit the constants, write them to `sd`."""
    model, params = recipe["model"], recipe.get("params", {})
    scales = [int(x) for x in recipe["scales"]]
    # `n` may be a scalar, a list, or an ALLOCATION RULE -- generate.py's own
    # resolver handles all three, so a pilot recipe can say
    # {"rule": "snr", "budget": 1e8, ...} exactly like any other recipe. That
    # matters here: a flat n starves the large scales and leaves omega1
    # unidentified (0.06 +/- 0.18 measured), while snr at the same wall clock
    # gives 0.98 +/- 0.28.
    n = resolve_n(recipe)

    reps = list(existing or [])
    base = seed if seed is not None else recipe.get("seed")
    have, want = len(reps), len(reps) + replicates
    spec = get_model(model)
    drawn_seconds, drawn_steps = 0.0, 0.0
    counts_now = n if isinstance(n, (list, tuple)) else [n] * len(scales)
    for k, ss in enumerate(spawn(base, replicates)):
        print(f"  replicate {have + k + 1}/{want} ...",
              end="", flush=True, file=sys.stderr)
        t0 = time.perf_counter()
        reps.append(_pilot_replicate(model, params, scales, n, ss))
        dt = time.perf_counter() - t0
        drawn_seconds += dt
        # Steps, not seconds, is the budget unit (see CATALOG / the cost-model
        # discussion): the model's declared cost_hint is exact where a clock is
        # not. Throughput converts one to the other for THIS machine.
        drawn_steps += sum(c * (spec.cost_hint(i, params) if spec.cost_hint
                                else 1.0)
                           for i, c in zip(scales, counts_now))
        print(f" {dt:.1f}s", file=sys.stderr)

    R = len(reps)
    y = np.array([r["y_bar"] for r in reps], float)
    sig = np.array([r["sigma_log"] for r in reps], float)
    counts = np.array(n if isinstance(n, (list, tuple)) else [n] * len(scales), float)

    # Pooled then refitted once, never averaged fit-by-fit: fit_correction is
    # nonlinear, so averaging R fits converges to E[a1_hat], not a1 -- a bias
    # that no number of replicates removes (see allocation_table.measured_a1).
    nn = np.broadcast_to(counts, y.shape)
    y_pool = (y * nn).sum(axis=0) / nn.sum(axis=0)
    sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))
    fit = fit_correction(scales, y_pool, sigma_log=sig_pool)

    # The stated errors come from the SPREAD of the per-replicate fits, which
    # needs R >= 2. With one replicate every se is None and plan.py says so.
    per = [fit_correction(scales, np.array(r["y_bar"]),
                          sigma_log=np.array(r["sigma_log"])) for r in reps] \
        if R > 1 else []
    def spread(key):
        return float(np.std([p[key] for p in per], ddof=1) / np.sqrt(R)) if R > 1 else None

    cost = measure_cost_exponent(model, params, scales)
    d_val = cost.get("affine", {}).get("d")
    d_se = cost.get("affine", {}).get("d_se")

    cv_per_scale = np.array([r["cv"] for r in reps], float).mean(axis=0)
    throughput = (drawn_steps / drawn_seconds) if drawn_seconds > 0 else None
    prov = (f"pilot, {R} replicate{'s' if R > 1 else ''}, pooled then refitted once"
            if R > 1 else "pilot, 1 replicate (no stderr available)")

    consts = {
        "omega1": measured(fit["omega1"], spread("omega1"), prov),
        "a1":     measured(fit["a1"], spread("a1"), prov),
        "cv":     measured(float(cv_per_scale.mean()), None,
                           f"pilot, mean over {len(scales)} scales, spread "
                           f"{cv_per_scale.min():.4f}-{cv_per_scale.max():.4f}"),
    }
    declared = cost.get("declared_d")
    if declared is not None:
        # The model states its own cost, which is exact where a clock is not.
        # The timing still runs, as a cross-check reported by plan.py.
        gap = abs(d_val - declared) if d_val is not None else None
        note = (f"declared by the model's cost_hint (clock agrees: "
                f"{d_val:.4f}, gap {gap:.4f})" if gap is not None and gap < 0.1
                else f"declared by the model's cost_hint"
                      + (f" -- CLOCK DISAGREES: {d_val:.4f}" if gap is not None else ""))
        consts["d"] = measured(declared, None, note)
    elif d_val is not None:
        consts["d"] = measured(d_val, d_se,
                               f"pilot cost probe, affine fit over "
                               f"{len(cost['scales'])} scales "
                               f"({cost['scales'][0]}..{cost['scales'][-1]})")

    sd.mkdir(parents=True, exist_ok=True)
    save(sd, consts)
    write_artifact(sd, "pilot", {
        "recipe": recipe, "replicates": R, "scales": scales,
        "n": [int(x) for x in counts],
        "y_bar": y_pool.tolist(), "sigma_log": sig_pool.tolist(),
        "cv_per_scale": cv_per_scale.tolist(),
        "direct_fit": fit, "per_replicate": reps, "cost": cost,
        "gamma_pilot": fit["gamma"], "a0_pilot": fit["a0"],
        "throughput": throughput, "drawn_seconds": drawn_seconds,
        "drawn_steps": drawn_steps,
    }, produced_by="src/study/pilot.py")
    return {"constants": consts, "fit": fit, "cost": cost, "replicates": R,
            "reps": reps, "cv_per_scale": cv_per_scale, "throughput": throughput}


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-meta", "--meta", dest="meta", type=Path, default=None,
                   help="samples recipe for the pilot draw. Omit only with --more, "
                        "which reuses the recipe already recorded in the study.")
    p.add_argument("--study", required=True,
                   help="study name; the directory is <experiment>/data/<name>/")
    p.add_argument("--data-root", type=Path, default=None,
                   help="experiment data dir; defaults to the recipe's own")
    p.add_argument("--replicates", type=int, default=1,
                   help="replicates to draw. 1 gives no standard errors, so plan.py "
                        "cannot say how well determined the plan is -- use 3+ when "
                        "you intend to act on the answer")
    p.add_argument("--more", type=int, default=0,
                   help="add this many replicates to an existing pilot, keeping the "
                        "ones already drawn")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)

    existing = None
    if a.more:
        root = a.data_root or (Path(a.meta).resolve().parent.parent / "data"
                               if a.meta else None)
        if root is None:
            raise SystemExit("--more needs --data-root or -meta to locate the study")
        sd = study_dir(root, a.study)
        prev = artifact_path(sd, "pilot")
        if not prev.exists():
            raise SystemExit(f"no pilot in {sd} to add to; run without --more first")
        old = json.loads(prev.read_text())
        recipe, existing, reps = old["recipe"], old["per_replicate"], a.more
        print(f"adding {a.more} replicate(s) to the {old['replicates']} already in {sd}")
    else:
        if a.meta is None:
            raise SystemExit("-meta is required (or use --more on an existing pilot)")
        recipe = load_recipe(a.meta, "samples")
        root = a.data_root or default_out_dir(a.meta)
        sd = study_dir(root, a.study)
        reps = a.replicates

    r = pilot(recipe, sd, reps, seed=a.seed, existing=existing)

    print(f"\nstudy   = {sd}")
    print(f"model   = {recipe['model']}  scales = {r['fit'] and recipe['scales']}")
    print(f"\nconstants measured ({r['replicates']} replicate(s))")
    print(format_table(r["constants"]))
    if r["replicates"] < 2:
        print("\n  No standard errors: one replicate has no spread to measure. "
              "omega1\n  ranged 0.49-1.25 across single replicates of one configuration "
              "in this\n  repo's own runs, so treat it as indicative. "
              f"Add more:\n    python3 src/study/pilot.py --study {a.study} --more 3 "
              f"--data-root {root}")
    if r["throughput"]:
        print(f"\nthroughput  = {r['throughput']:.3g} steps/s "
              f"(this machine, from the pilot's own clock)")
    print(f"\ngamma from the pilot itself: {r['fit']['gamma']:.4f}  "
          f"(indicative -- the plan exists to measure it properly)")
    print(f"\nnext: python3 src/study/plan.py --study {a.study} --data-root {root}")


if __name__ == "__main__":
    _main()
