"""Two observables, one budget: which one's gamma-hat is closer to the truth?

The question this exists for (Igor, 2026-09-04). Everything else in this repo
asks whether an estimator is correct GIVEN an observable. This asks the other
question: given a fixed computational budget and a fixed estimator, does a
different choice of Y_i converge faster? The concrete case is percolation --
the number of open sites connected to a full SIDE of the box (PLAN.md ground
rule 7) versus the cluster of the ORIGIN (what
`presentation18-05-2026/coding` measured). Both have the same gamma = d_f =
91/48, so the comparison is entirely about bias and noise. See
experiments/03_percolation_zd/README.md, Experiment P2.

Nothing here is percolation-specific: an "arm" is just a samples recipe, so
this compares any two (model, params, scales, budget) choices that are meant
to estimate the same gamma.

What it does, and the two things it is careful about
---------------------------------------------------
For each arm and each replicate r = 1..R it draws a COMPLETE, independent
experiment (every scale, that arm's own n) and computes one gamma-hat from
it. R gamma-hats per arm give a bias, a spread and an RMSE against a stated
truth -- which is the only honest way to say one observable converges faster
than another. A single run of each would compare two draws from two
distributions by their means alone.

  1. Independence (ground rule 2). Every (arm, replicate) gets its own
     SeedSequence spawned from one root, so no randomness is shared between
     replicates, between arms, or between scales. The arms are NOT run on
     common random numbers: a paired design would reduce the variance of the
     DIFFERENCE, but the quantity being reported is each arm's own RMSE, and
     sharing draws across two different observables of the same lattice would
     make those two RMSEs dependent in a way nothing downstream accounts for.

  2. The truth is never handed to an estimator. `--truth` enters only in the
     final scoring arithmetic (bias, RMSE), exactly like
     src/estimate/estimate_omega1.py's `--expect-gamma`. It is a CLI argument
     rather than a model attribute for the reason models/percolation2d.py
     gives: MODELS entries deliberately carry no true_gamma_key here.

Two gamma estimators are reported per replicate, both from tools/loglog.py:
`gamma_all_points` (OLS over every scale) and `gamma_drop_leading` at
--m0 (the article's own finite-size remedy, eq. 523-531's m_0). An observable
can win on one and not the other -- a heavily contaminated observable is
helped more by dropping scales -- and that difference is itself the finding,
so neither is chosen for the reader.

Samples are not persisted (each replicate is reduced to per-scale
(mean, se, n) as it is drawn), for the same reason
src/budget/allocation_experiment.py does not persist its sweep: R x 2 full
ladders is a lot of disk for data whose only use is one number each, and the
spawning is deterministic, so any replicate can be regenerated exactly.

CLI:
    python3 src/estimate/compare_observables.py \
        --arm south=experiments/03_percolation_zd/recipes/samples_df_south.json \
        --arm origin=experiments/03_percolation_zd/recipes/samples_df_origin.json \
        --replicates 12 --truth 1.8958333333333333 --tag compare_anchors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from src.generate.generate import generate, resolve_n  # noqa: E402
from tools.artifacts import artifact_path, default_out_dir, load_recipe, write_artifact  # noqa: E402
from tools.cost_model import declared_exponent  # noqa: E402
from tools.loglog import gamma_all_points, gamma_drop_leading  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools.persistence import normalize_scales_n, run_dir as _run_dir  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402


def _summarize(y: np.ndarray) -> tuple[float, float, int]:
    """One scale's samples -> (mean, se of the mean, n).

    Passed to `generate(reduce=...)`, so the raw array is dropped the moment
    the scale finishes: a replicate never holds more than one scale's draws.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), n


def _gamma_at_m0(scales, y_bar, m0: int) -> float:
    """gamma-hat with the m0 smallest scales dropped (article eq. 523-531's m_0).

    tools/loglog.py's `gamma_drop_leading` returns the whole sequence of
    windows; this picks the one that dropped exactly m0, and raises rather
    than silently returning the nearest window, because a comparison in which
    the two arms used different windows would not be a comparison.
    """
    for w in gamma_drop_leading(scales, y_bar):
        if len(scales) - len(w["scales_used"]) == m0:
            return float(w["gamma_hat"])
    raise ValueError(
        f"no drop-leading window drops exactly {m0} of {len(scales)} scales; "
        f"--m0 must leave at least 2 scales")


def _arm_plan(recipe_path) -> dict:
    """Resolve one arm's recipe into the concrete (scales, n) it will draw.

    Done once per arm, before any replicate: `resolve_n` prints the allocation
    it chose, and doing it per replicate would print R identical tables and
    re-derive an answer that cannot change.
    """
    cfg = load_recipe(recipe_path, "samples")
    scales, n = normalize_scales_n(cfg["scales"], resolve_n(cfg))
    spec = get_model(cfg["model"])
    d = (declared_exponent(scales, spec.cost_hint, cfg.get("params", {}))
         if spec.cost_hint is not None else None)
    cost = (sum(n_i * spec.cost_hint(i, cfg.get("params", {}))
                for i, n_i in zip(scales, n)) if spec.cost_hint is not None else None)
    return {
        "recipe": str(recipe_path),
        "model": cfg["model"],
        "params": cfg.get("params", {}),
        "scales": scales,
        "n": n,
        "declared_d": d,
        "cost_units": cost,
        "allocation": cfg.get("_allocation_resolved"),
    }


def compare(arms: dict, *, replicates: int, m0: int, truth: float | None,
            seed=None, progress: bool = False) -> dict:
    """R independent replicates of every arm; gamma-hats and their moments.

    `arms` maps a short name to a samples-recipe path. Each arm's replicates
    are drawn from streams spawned off one root SeedSequence, arm by arm, so
    the whole comparison is reproducible from `seed` alone.
    """
    plans = {name: _arm_plan(path) for name, path in arms.items()}
    sys.stdout.flush()      # resolve_n printed each arm's allocation; keep it
                            # above the per-replicate progress on stderr

    root = np.random.SeedSequence(seed)
    streams = iter(spawn(root, len(plans) * replicates))

    out: dict = {}
    for name, plan in plans.items():
        gammas_all, gammas_m0, cvs, ybars, seconds = [], [], [], [], []
        for r in range(replicates):
            stream = next(streams)
            t0 = time.perf_counter()
            summarized = generate(plan["model"], plan["scales"], plan["n"],
                                  plan["params"], seed=stream, reduce=_summarize)
            seconds.append(time.perf_counter() - t0)

            scales = sorted(summarized)
            y_bar = np.array([summarized[i][0] for i in scales])
            se = np.array([summarized[i][1] for i in scales])
            counts = np.array([summarized[i][2] for i in scales], dtype=float)
            # cv of the observable itself, per scale: se * sqrt(n) / mean. This
            # is the sigma_k of Assumption 6 (eq. 332); whether it is flat in k
            # is the assumption, and is half the point of the comparison.
            cvs.append(list(se * np.sqrt(counts) / y_bar))
            ybars.append(list(y_bar))

            gammas_all.append(gamma_all_points(scales, y_bar))
            gammas_m0.append(_gamma_at_m0(scales, y_bar, m0))
            if progress:
                print(f"  [{name} {r + 1}/{replicates}] gamma_all="
                      f"{gammas_all[-1]:.4f}  ({seconds[-1]:.1f}s)", file=sys.stderr)

        out[name] = {
            **plan,
            "seconds_per_replicate": seconds,
            "gamma_all_points": gammas_all,
            f"gamma_m0_{m0}": gammas_m0,
            "cv_per_scale_mean": list(np.mean(np.array(cvs), axis=0)),
            # The raw log-log points themselves, averaged over replicates.
            # Kept because "which arm has the larger correction-to-scaling
            # amplitude" is answered by Y_bar_i / i**gamma and by nothing
            # else in this file -- without it the artifact records the
            # verdict but not the evidence for it.
            "y_bar_per_scale_mean": list(np.mean(np.array(ybars), axis=0)),
            "scores": {
                key: _score(values, truth)
                for key, values in (("gamma_all_points", gammas_all),
                                    (f"gamma_m0_{m0}", gammas_m0))
            },
        }
    return {"arms": out, "replicates": replicates, "m0": m0, "truth": truth,
            "seed": seed_record(root)}


def _score(values, truth: float | None) -> dict:
    """Mean, spread and (given a truth) bias and RMSE of one arm's gamma-hats.

    RMSE is reported rather than only the standard deviation because the two
    arms are allowed to differ in BIAS -- a more contaminated observable has a
    larger finite-size bias at the same scales (article Prop. 820), and an
    arm can be quieter and still worse. `se_of_mean` says how well determined
    the reported bias itself is, so a bias smaller than it is not a finding.
    """
    v = np.asarray(values, dtype=np.float64)
    out = {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "se_of_mean": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"),
    }
    if truth is not None:
        out["bias"] = float(v.mean() - truth)
        out["rmse"] = float(np.sqrt(np.mean((v - truth) ** 2)))
        # RMSE**2 = bias**2 + sd**2 (up to the 1/R in the ddof), so which term
        # is larger decides whether MORE BUDGET would help this arm at all.
        # Reported as a flag rather than left to the reader because the
        # head-to-head budget ratio below is only meaningful when both arms
        # are variance-dominated.
        out["bias_dominated"] = bool(abs(out["bias"]) > out["sd"])
    return out


def format_report(res: dict) -> str:
    """The table, arm by arm, plus the head-to-head that answers the question."""
    m0 = res["m0"]
    keys = ["gamma_all_points", f"gamma_m0_{m0}"]
    lines = [f"replicates = {res['replicates']} per arm   m0 = {m0}   "
             f"truth = {res['truth']}   seed = {res['seed']}", ""]

    for name, arm in res["arms"].items():
        lines.append(f"arm {name!r}: model={arm['model']} params={arm['params']}")
        lines.append(f"  scales = {arm['scales']}")
        lines.append(f"  n      = {arm['n']}")
        if arm["cost_units"] is not None:
            lines.append(f"  cost   = {arm['cost_units']:.4g} work units/replicate "
                         f"({np.mean(arm['seconds_per_replicate']):.1f}s measured)")
        lines.append("  cv(Y_i) per scale = "
                     + " ".join(f"{c:.3f}" for c in arm["cv_per_scale_mean"]))
        for key in keys:
            s = arm["scores"][key]
            row = (f"  {key:<18} mean={s['mean']:.4f}  sd={s['sd']:.4f}"
                   f"  se(mean)={s['se_of_mean']:.4f}")
            if "bias" in s:
                row += f"  bias={s['bias']:+.4f}  rmse={s['rmse']:.4f}"
                row += "  <-- BIAS-DOMINATED" if s["bias_dominated"] else ""
            lines.append(row)
        lines.append("")

    names = list(res["arms"])
    if len(names) == 2 and res["truth"] is not None:
        a, b = names
        lines.append(f"head to head ({a} vs {b}), same budget:")
        for key in keys:
            ra = res["arms"][a]["scores"][key]["rmse"]
            rb = res["arms"][b]["scores"][key]["rmse"]
            better, worse, ratio = ((a, b, rb / ra) if ra < rb else (b, a, ra / rb))
            note = ("  (>= : "
                    + worse + " is bias-dominated, so no budget closes the gap)"
                    if res["arms"][worse]["scores"][key]["bias_dominated"] else "")
            lines.append(f"  {key:<18} rmse {a}={ra:.4f}  {b}={rb:.4f}   "
                         f"-> {better} wins by {ratio:.2f}x in RMSE, "
                         f"{ratio ** 2:.1f}x in budget{note}")
        lines.append("")
        lines.append("  (RMSE falls as budget**-theta, so a k-fold RMSE ratio is a k**2-fold")
        lines.append("   budget ratio only while both arms are VARIANCE-dominated. Compare each")
        lines.append("   arm's bias against its se(mean) above first: a bias-dominated arm")
        lines.append("   cannot buy its way level at any budget, and the budget figure then")
        lines.append("   understates how much worse it is.)")
    return "\n".join(lines)


def _parse_arm(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"--arm needs NAME=PATH, got {text!r} (e.g. south=recipes/samples_df_south.json)")
    name, path = text.split("=", 1)
    return name, Path(path)


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", action="append", required=True, type=_parse_arm, metavar="NAME=RECIPE",
                   help="a samples recipe to compare, named. Repeat for each arm (2 for a "
                        "head-to-head).")
    p.add_argument("--replicates", type=int, default=10,
                   help="independent full experiments per arm. Each contributes one gamma-hat; "
                        "the spread of those is what the comparison is made of, so 1 is useless "
                        "and fewer than ~8 gives an RMSE with a wide error of its own")
    p.add_argument("--m0", type=int, default=2,
                   help="how many of the smallest scales the second estimator drops "
                        "(article eq. 523-531's m_0)")
    p.add_argument("--truth", type=float, default=None,
                   help="known gamma, for bias/RMSE. REPORTING ONLY -- never reaches an "
                        "estimator. Omit to report mean and spread with no scoring")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--out-dir", type=Path, default=None,
                   help="output directory for <tag>/. Defaults to the first arm's\n"
                        "experiment data/")
    p.add_argument("--tag", type=str, default="observable_comparison")
    args = p.parse_args(argv)

    arms = dict(args.arm)
    res = compare(arms, replicates=args.replicates, m0=args.m0, truth=args.truth,
                  seed=args.seed, progress=True)
    print(format_report(res))

    out_dir = args.out_dir or default_out_dir(list(arms.values())[0])
    rd = _run_dir(out_dir, args.tag)
    write_artifact(rd, "observable_comparison", res,
                   produced_by="src/estimate/compare_observables.py")
    print(f"output = {artifact_path(rd, 'observable_comparison')}")


if __name__ == "__main__":
    _main()
