"""Cost-model probe (see README's "Cost-model probe" section): measures the
computational-cost exponent d, article Assumption cost_is_power_law
(cost(i) = i**d), by timing srw() at a small grid of scales.

Ground truth for this specific simulator: generating k i.i.d. +-1 steps and
summing them is Theta(k), so d should recover close to 1. This is the
verification step for tools/cost_model.py's estimator -- not a claim about
percolation or any other future, more expensive simulator's cost exponent.

Repeated timing measurements at a fixed scale all target the same true
deterministic quantity (unlike Y_i, which is genuinely stochastic) -- noise
here is OS/interpreter jitter, which only ever adds delay. So aggregation
uses min, not mean, across repeats (standard microbenchmark practice, e.g.
Python's own timeit) -- the one place this experiment departs from the
sample-mean framing used elsewhere in this codebase for Y_i.

CLI:
    python3 measure_cost.py                       # default scales/repeats
    python3 measure_cost.py -meta cost_config.json --tag my_run
    python3 measure_cost.py --plot                # also save images/<tag>.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for tools/
sys.path.insert(0, str(HERE))  # this dir, for srw

from tools.cost_model import estimate_cost_exponent  # noqa: E402
from tools.loglog import gamma_drop_leading  # noqa: E402

from srw import srw  # noqa: E402

DEFAULT_SCALES = [256, 1024, 4096, 16384, 65536, 262144, 1048576]  # 2**8 .. 2**20, step 2
DEFAULT_REPEATS = 20
ACCEPTANCE_RANGE = (0.8, 1.2)  # around the known ground truth d = 1


def measure(scales, repeats: int, *, seed: int | None = None, q: float = 0.5) -> dict:
    """Time srw(k) `repeats` times at each k in `scales`; estimate d from the
    per-scale minimum elapsed time."""
    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    elapsed_all: dict[int, list[float]] = {}
    elapsed_min: list[float] = []
    for k in scales:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            srw(k, q=q, rng=rng)
            times.append(time.perf_counter() - t0)
        elapsed_all[k] = times
        elapsed_min.append(min(times))

    d_hat = estimate_cost_exponent(scales, elapsed_min)

    return {
        "scales": list(scales),
        "repeats": repeats,
        "seed": seed_seq.entropy,
        "q": q,
        "elapsed_min": elapsed_min,
        "elapsed_all": {str(k): v for k, v in elapsed_all.items()},
        "d_hat": d_hat,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-meta", "--meta", dest="meta", type=Path, default=None,
        help='Optional recipe JSON: {"scales": [...], "repeats": ..., "seed": null, "q": 0.5}. '
        "Defaults to the module’s built-in scale grid/repeat count if omitted.",
    )
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--tag", dest="tag", type=str, default="cost_probe")
    parser.add_argument(
        "--plot", dest="plot", action="store_true",
        help="Also save a supplementary log-log plot of elapsed time vs scale to "
        "images/<tag>.png (ground rule 1: supplements the numeric check above, never "
        "replaces it). Off by default so routine reruns don't overwrite the committed "
        "evidence figure.",
    )
    args = parser.parse_args(argv)

    if args.meta is not None:
        cfg = json.loads(args.meta.read_text())
        scales = cfg.get("scales", DEFAULT_SCALES)
        repeats = cfg.get("repeats", DEFAULT_REPEATS)
        seed = cfg.get("seed")
        q = cfg.get("q", 0.5)
    else:
        scales, repeats, seed, q = DEFAULT_SCALES, DEFAULT_REPEATS, None, 0.5

    result = measure(scales, repeats, seed=seed, q=q)

    out_dir = args.out_dir or (HERE / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    local_slopes = gamma_drop_leading(result["scales"], result["elapsed_min"])

    print(f"repeats={repeats}  seed={result['seed']}  q={q}")
    print(f"{'k':>10} {'min_ms':>12}")
    for k, t in zip(result["scales"], result["elapsed_min"]):
        print(f"{k:>10} {t * 1e3:>12.4f}")
    print(f"\nd_hat (all scales)      = {result['d_hat']:.4f}")
    print("d_hat dropping leading m0 (finite-overhead check at small k):")
    for est in local_slopes:
        print(f"  m0={est['m0']}: scales={est['scales_used']}  d_hat={est['gamma_hat']:.4f}")

    lo, hi = ACCEPTANCE_RANGE
    passed = lo <= result["d_hat"] <= hi
    print(f"\n{'PASS' if passed else 'FAIL'}: d_hat={result['d_hat']:.4f} vs acceptance range [{lo}, {hi}]")
    print(f"\noutput = {out_path}")

    if args.plot:
        image_path = _save_plot(result, out_dir=HERE / "images", tag=args.tag)
        print(f"image  = {image_path}")


def _save_plot(result: dict, *, out_dir: Path, tag: str) -> Path:
    """Supplementary log-log plot of elapsed time vs scale (ground rule 1:
    a figure supplements the numeric check above, never replaces it). Deferred
    matplotlib import/backend selection: `measure()` itself (used by
    test_cost_probe.py) has no matplotlib dependency."""
    import matplotlib

    matplotlib.use("Agg")
    from tools.loglog_plot import loglog_plot

    samples = {int(k): np.asarray(v) for k, v in result["elapsed_all"].items()}
    ax = loglog_plot(samples, label="elapsed time (all repeats)")
    ax.set_xlabel("k")
    ax.set_ylabel("elapsed time (s)")
    ax.set_title(f"srw() cost probe: d_hat={result['d_hat']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.png"
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


if __name__ == "__main__":
    _main()
