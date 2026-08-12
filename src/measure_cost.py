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
here is OS/interpreter jitter, which only ever adds delay. So aggregation
uses min, not mean, across repeats (standard microbenchmark practice, e.g.
Python's own timeit) -- the one place this departs from the sample-mean
framing used elsewhere in this codebase for Y_i.

This module only measures and saves -- it never plots (mirrors
generate.py/plot_loglog.py's split). See plot_cost.py for the log-log plot.

CLI:
    python3 measure_cost.py -meta ../experiments/01_srw/cost_probe_config.json --tag my_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports

from cost_model import estimate_cost_exponent  # noqa: E402
from loglog import gamma_drop_leading  # noqa: E402
from models import get_model  # noqa: E402
from persistence import content_id, run_dir as _run_dir  # noqa: E402

ACCEPTANCE_RANGE = (0.8, 1.2)  # around the known ground truth d = 1, for models where that applies


def measure(model: str, scales, repeats: int, params: dict, *, seed: int | None = None) -> dict:
    """Time MODELS[model].simulate(k, n=1, params, rng) `repeats` times at
    each k in `scales`; estimate d from the per-scale minimum elapsed time."""
    spec = get_model(model)
    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    elapsed_all: dict[int, list[float]] = {}
    elapsed_min: list[float] = []
    for k in scales:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            spec.simulate(k, 1, params, rng)
            times.append(time.perf_counter() - t0)
        elapsed_all[k] = times
        elapsed_min.append(min(times))

    d_hat = estimate_cost_exponent(scales, elapsed_min)

    return {
        "model": model,
        "params": params,
        "scales": list(scales),
        "repeats": repeats,
        "seed": seed_seq.entropy,
        "elapsed_min": elapsed_min,
        "elapsed_all": {str(k): v for k, v in elapsed_all.items()},
        "d_hat": d_hat,
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
        help="Output directory for <tag>/. Defaults to a 'data' directory next to the recipe file.",
    )
    parser.add_argument("--tag", dest="tag", type=str, default="cost_probe")
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    model = cfg["model"]
    params = cfg["params"]
    scales = cfg["scales"]
    repeats = cfg["repeats"]
    seed = cfg.get("seed")

    result = measure(model, scales, repeats, params, seed=seed)

    out_dir = args.out_dir or (args.meta.resolve().parent / "data")
    rd = _run_dir(out_dir, args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    out_path = rd / "result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    local_slopes = gamma_drop_leading(result["scales"], result["elapsed_min"])

    print(f"model={model!r}  params={params}  repeats={repeats}  seed={result['seed']}")
    print(f"{'k':>10} {'min_ms':>12}")
    for k, t in zip(result["scales"], result["elapsed_min"]):
        print(f"{k:>10} {t * 1e3:>12.4f}")
    print(f"\nd_hat (all scales)      = {result['d_hat']:.4f}")
    print("d_hat dropping leading m0 (finite-overhead check at small k):")
    for est in local_slopes:
        print(f"  m0={est['m0']}: scales={est['scales_used']}  d_hat={est['gamma_hat']:.4f}")

    lo, hi = ACCEPTANCE_RANGE
    passed = lo <= result["d_hat"] <= hi
    print(f"\n{'PASS' if passed else 'FAIL'} vs [{lo}, {hi}] (ground truth d=1 only applies to "
          f"genuinely scale-costly models, e.g. srw -- see module docstring)")
    print(f"\noutput = {out_path}")
    print(f"plot   = python3 plot_cost.py -data {rd}")


if __name__ == "__main__":
    _main()
