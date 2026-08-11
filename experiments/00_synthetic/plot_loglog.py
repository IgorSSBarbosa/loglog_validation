"""Log-log plot of a synthetic experiment's persisted data, plus a comparison
of a few gamma-hat estimators over the same data.

Thin glue: loads the samples generator.py already saved, and hands them to
the generic, experiment-agnostic `tools/loglog_plot.py` and `tools/loglog.py`
(neither imports from `experiments/`, so both work the same way for SRW,
percolation, etc. once those exist). Since the synthetic model's E[Y_i] is
known exactly, the true curve/gamma is overlaid as a reference when available.
This script only reads data -- it never generates any, so run generator.py
first.

Takes a **data path** (the .npz), not a JSON: a single recipe can produce many
different runs (different tags/seeds), so pointing this at a JSON would be
ambiguous about which run's data you mean. Metadata (for the reference-curve
overlay and true_gamma) is read from the same stem's .json if present, but
isn't required -- missing metadata just means no overlay/true_gamma, not a
failure to plot or estimate.

Writes three outputs, all named after the data file's stem: the raw-data plot
to `images/<stem>.png`, a comparison of the four gamma-hat estimators (see
tools/loglog.py's compare_methods) to `images/<stem>_estimates.png`, and the
underlying numbers for the latter to `data/<stem>_results.json`.

Run (after generating some data, e.g. `generator.py -meta example_config.json
--tag demo_run`):

    python3 plot_loglog.py -data data/demo_run.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for tools/
sys.path.insert(0, str(HERE))  # this dir, for generator

from tools.loglog import compare_methods  # noqa: E402
from tools.loglog_plot import estimates_plot, loglog_plot, loglog_points  # noqa: E402

from generator import load_metadata, load_samples, mean_Y, params_from_json  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Y_bar_i vs i (log-log) from generator.py's saved output.")
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Data .npz written by generator.py (e.g. data/demo_run.npz).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<data-stem>.png",
    )
    args = parser.parse_args(argv)

    try:
        samples = load_samples(args.data)
    except FileNotFoundError as e:
        parser.error(str(e))

    meta = load_metadata(args.data)
    target_fn, label, true_gamma = None, None, None
    if meta is not None:
        params = params_from_json(meta["params"])
        target_fn = lambda i: mean_Y(i, params)  # noqa: E731
        label = params.family
        true_gamma = params.gamma
    else:
        print(f"note: no metadata at {args.data.with_suffix('.json')} -- plotting data only, no reference curve")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, target_fn=target_fn, label=label)
    ax.set_title(f"Synthetic log-log plot ({args.data.stem})")

    out = args.out or (HERE / "images" / f"{args.data.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    scales, y_bar, _se, n = loglog_points(samples)
    results = compare_methods(scales, y_bar, n, true_gamma=true_gamma)
    results["source_npz"] = str(args.data)
    results_path = args.data.parent / f"{args.data.stem}_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    estimates_plot(results, ax=ax2)
    ax2.set_title(f"$\\hat\\gamma$ estimator comparison ({args.data.stem})")
    out2 = out.parent / f"{args.data.stem}_estimates.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    m = results["methods"]
    print(f"\ngamma estimates (true_gamma={true_gamma}):")
    print(f"  all_points        : {m['all_points']['gamma_hat']:.4f}")
    two_pt = [e["gamma_hat"] for e in m["two_point"]["estimates"]]
    print(f"  two_point         : {['%.4f' % g for g in two_pt]}")
    drop = [e["gamma_hat"] for e in m["drop_leading"]["estimates"]]
    print(f"  drop_leading      : {['%.4f' % g for g in drop]}")
    mle = m["mle"]
    flag = "" if mle["trustworthy"] else "  ** NOT TRUSTWORTHY, see diagnostics in results.json **"
    print(f"  mle               : {mle['gamma_hat']:.4f}{flag}")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
