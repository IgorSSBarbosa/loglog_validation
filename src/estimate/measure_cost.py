"""Single, shared cost-model probe for every model (tools/models.py): measures
the computational-cost exponent d, article Assumption cost_is_power_law
(cost(i) = i**d), by timing MODELS[model].simulate(i, n=1, ...) at a small
grid of scales -- instead of each experiment keeping its own copy of this
driver.

Only meaningful for models whose per-call cost genuinely grows with scale
(e.g. "srw": generating k i.i.d. +-1 steps and summing them is Theta(k),
so d should recover close to 1 -- the verification step for
tools/cost_model.py's estimator). Pointed at "synthetic" -- drawing from a
closed-form formula, cost ~constant in i -- this will just (correctly)
measure d~=0; that's an expected, uninteresting result, not a bug.

Repeated timing measurements at a fixed scale all target the same true
deterministic quantity (unlike Y_i, which is genuinely stochastic) -- noise
here is OS/interpreter jitter, which only ever adds delay. So aggregation is
robust rather than a sample mean -- the one place this departs from the
sample-mean framing used elsewhere in this codebase for Y_i. The aggregator
is selectable (tools/cost_model.py's AGGREGATORS: min/median/mean/q95/iqmean,
settable per-recipe via an "aggregator" key) and defaults to `median`, which
resists the one-sided jitter like `min` does but, unlike `min`, carries a
distribution-free confidence interval so the cost curve gets honest error
bars. Empirically the choice barely moves d_hat (see cost_model.py).

Two fits of those timings are reported, and which one to believe depends on
the overhead: cost(i) = c*i^d (Assumption cost_is_power_law itself) and
cost(i) = a + b*i^d. This probe times simulate(k, n=1, ...), and a single
call carries a fixed ~20us of Python/NumPy dispatch that does not scale with
k at all -- at the smallest scales that overhead IS the measurement, dragging
the pure-power d_hat far below the truth. The affine fit separates it out and
recovers d correctly, so acceptance is checked against the affine d whenever
it is available.

This module only measures and saves -- it never plots (mirrors
generate.py/plot_loglog.py's split). See plot_cost.py for the log-log plot.

CLI:
    python3 src/estimate/measure_cost.py -meta experiments/01_srw/recipes/cost_probe.json --tag my_run
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
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from artifacts import artifact_path, default_out_dir, load_recipe  # noqa: E402
from cost_model import (  # noqa: E402
    DEFAULT_AGGREGATOR,
    compare_cost_models,
    fit_cost_probe,
    format_cost_comparison,
    median_ci,
    time_over_scales,
)
from loglog import gamma_drop_leading  # noqa: E402
from models import get_model  # noqa: E402
from persistence import content_id, run_dir as _run_dir  # noqa: E402

ACCEPTANCE_RANGE = (0.8, 1.2)  # around the known ground truth d = 1, for models where that applies


def measure(
    model: str,
    scales,
    repeats: int,
    params: dict,
    *,
    seed: int | None = None,
    aggregator: str = DEFAULT_AGGREGATOR,
) -> dict:
    """Time MODELS[model].simulate(k, n=1, params, rng) `repeats` times at
    each k in `scales`; estimate d from the per-scale aggregated elapsed time.

    Timing and fitting are tools/cost_model.py's `time_over_scales` and
    `fit_cost_probe` -- the same two the pilot's probe uses, which differs
    only in choosing its own scales instead of taking a ladder (see
    src/study/pilot.py's measure_cost_exponent). What this driver adds is the
    ladder as the EXPERIMENT: a named grid, its per-scale confidence
    intervals, and the acceptance check below.

    Reports two fits of the same timings:

    - `d_hat`: the pure power law cost(i) = c * i**d of Assumption
      cost_is_power_law (eq. 353).
    - `affine`: cost(i) = a + b * i**d, which additionally models the fixed
      per-call overhead. Prefer this one whenever `affine["a"]` is not small
      relative to the timings at the smallest scales -- there the pure fit is
      measuring overhead rather than simulation cost, and is badly biased
      downward (on `time_measure`, d_hat=0.10 against a true d=1, which the
      affine fit recovers as 0.95 with a=23.7us).
    """
    spec = get_model(model)
    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    probe = time_over_scales(spec, scales, params, rng, repeats, aggregator)
    fit_cost_probe(probe, spec.cost_hint, params)
    times = {k: probe["elapsed_all"][str(k)] for k in probe["scales"]}

    return {
        "model": model,
        "params": params,
        "scales": probe["scales"],
        "repeats": repeats,
        "seed": seed_seq.entropy,
        "aggregator": aggregator,
        "elapsed": probe["elapsed"],
        "elapsed_ci": [list(median_ci(times[k])) for k in probe["scales"]],
        "elapsed_min": [min(times[k]) for k in probe["scales"]],
        "elapsed_all": {str(k): times[k] for k in probe["scales"]},
        "d_hat": probe["d_hat"],
        "affine": probe["affine"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-meta", "--meta", dest="meta", required=True, type=Path,
        help='Recipe JSON (read-only): {"model": "srw", "params": {...}, "scales": [...], '
        '"repeats": ..., "seed": null}',
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", type=Path, default=None,
        help="Output directory for <tag>/. Defaults to the experiment's data/ directory (the recipe's grandparent when it sits in recipes/).",
    )
    parser.add_argument("--tag", dest="tag", type=str, default="cost_probe")
    args = parser.parse_args(argv)

    cfg = load_recipe(args.meta, "cost_probe")
    model = cfg["model"]
    params = cfg["params"]
    scales = cfg["scales"]
    repeats = cfg["repeats"]
    seed = cfg.get("seed")
    aggregator = cfg.get("aggregator", DEFAULT_AGGREGATOR)

    result = measure(model, scales, repeats, params, seed=seed, aggregator=aggregator)

    out_dir = args.out_dir or default_out_dir(args.meta)
    rd = _run_dir(out_dir, args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    out_path = artifact_path(rd, "cost_probe")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    local_slopes = gamma_drop_leading(result["scales"], result["elapsed"])

    print(f"model={model!r}  params={params}  repeats={repeats}  seed={result['seed']}  "
          f"aggregator={result['aggregator']!r}")
    print(f"{'k':>10} {'cost_ms':>12} {'ci_lo_ms':>12} {'ci_hi_ms':>12}")
    for k, t, (lo_ci, hi_ci) in zip(result["scales"], result["elapsed"], result["elapsed_ci"]):
        print(f"{k:>10} {t * 1e3:>12.4f} {lo_ci * 1e3:>12.4f} {hi_ci * 1e3:>12.4f}")

    print(f"\npure power law  cost(i) = c*i^d      : d_hat = {result['d_hat']:.4f}")
    aff = result["affine"]
    if "error" in aff:
        print(f"affine fit unavailable: {aff['error']}")
    else:
        print(f"affine          cost(i) = a + b*i^d  : d_hat = {aff['d']:.4f}   "
              f"overhead a = {aff['a'] * 1e6:.2f} us   rel_rmse = {aff['rel_rmse']:.4f}")
        smallest = result["elapsed"][0]
        share = aff["a"] / smallest if smallest > 0 else float("nan")
        print(f"  overhead is {share:.0%} of the measured cost at the smallest scale "
              f"k={result['scales'][0]}"
              + ("  -- pure-power d_hat is unreliable here, prefer the affine one"
                 if share > 0.2 else ""))

    spec = get_model(model)
    if spec.cost_hint is not None:
        cmp = compare_cost_models(result["scales"], result["elapsed"],
                                  spec.cost_hint, params)
        result["declared_vs_measured"] = cmp
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        print("\ndeclared cost vs the wall clock:")
        print(format_cost_comparison(cmp))
    else:
        print(f"\nmodel {model!r} declares no cost_hint -- nothing to cross-check "
              f"the measured d against (see tools/models.py's ModelSpec).")

    print("\nd_hat dropping leading m0 (finite-overhead check at small k):")
    for est in local_slopes:
        print(f"  m0={est['m0']}: scales={est['scales_used']}  d_hat={est['gamma_hat']:.4f}")

    lo, hi = ACCEPTANCE_RANGE
    d_for_acceptance = aff["d"] if "error" not in aff else result["d_hat"]
    passed = lo <= d_for_acceptance <= hi
    print(f"\n{'PASS' if passed else 'FAIL'} vs [{lo}, {hi}] on d_hat={d_for_acceptance:.4f} "
          f"(ground truth d=1 only applies to "
          f"genuinely scale-costly models, e.g. srw -- see module docstring)")
    print(f"\noutput = {out_path}")
    print(f"plot   = python3 src/report/plot_cost.py -data {rd}")


if __name__ == "__main__":
    _main()
