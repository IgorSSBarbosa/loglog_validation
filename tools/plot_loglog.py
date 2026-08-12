"""Single, shared log-log plotter for every model's persisted data -- reads
the run's own metadata to dispatch, instead of each experiment keeping its
own copy of this driver.

Thin glue: loads the samples tools/generate.py already saved, and hands
them to the generic tools/loglog_plot.py / tools/loglog.py (neither of which
knows about any specific model). When the run's model (tools/models.py) has
a known closed form (`target_fn`/`true_gamma_key` -- currently only
"synthetic"), the reference curve is overlaid and tools/loglog.py's four
gamma-hat estimators are compared against it. When it doesn't (currently
"srw" -- no article-sanctioned closed form yet, see
experiments/01_srw/README.md), this script **deliberately** only plots the
raw data: computing a gamma_hat with nothing to validate it against would be
an unvalidated number, easy to mistake for a checked result.

Takes a **run directory** (`<out_dir>/<tag>/`, as tools/generate.py writes),
not a recipe -- a single recipe can produce many different runs (different
tags/seeds), so pointing this at a recipe would be ambiguous about which
run's data you mean. Metadata is read from `<run_dir>/metadata.json` if
present, but isn't required -- missing metadata just means no overlay/model
dispatch, not a failure to plot.

Writes the raw-data plot to `images/<tag>.png` next to the run's experiment
folder (i.e. sibling to `data/`), and, only when a target_fn is known, also
the four-estimator comparison to `images/<tag>_estimates.png` plus the
underlying numbers to `<run_dir>/results.json`.

Run (after generating some data, e.g. `generate.py -meta
../experiments/00_synthetic/example_config.json --tag demo_run`):

    python3 plot_loglog.py -data ../experiments/00_synthetic/data/demo_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this dir, for bare imports below

from loglog import compare_methods  # noqa: E402
from loglog_plot import estimates_plot, loglog_plot, loglog_points  # noqa: E402
from models import get_model  # noqa: E402
from persistence import load_metadata, load_samples  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Y_bar_i vs i (log-log) from a run directory tools/generate.py wrote."
    )
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Run directory written by tools/generate.py (e.g. ../experiments/00_synthetic/data/demo_run).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<tag>.png next to the run's experiment folder.",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        samples = load_samples(args.data)
    except FileNotFoundError as e:
        parser.error(str(e))

    tag = args.data.name
    meta = load_metadata(args.data)
    target_fn, label, true_gamma = None, None, None
    if meta is not None:
        model, params = meta["model"], meta["params"]
        spec = get_model(model)
        label = model
        if spec.target_fn is not None:
            target_fn = lambda i, _spec=spec, _params=params: _spec.target_fn(i, _params)  # noqa: E731
        if spec.true_gamma_key is not None:
            true_gamma = params.get(spec.true_gamma_key)
    else:
        print(f"note: no metadata at {args.data / 'metadata.json'} -- plotting data only, no reference curve")

    out = args.out or (args.data.resolve().parents[1] / "images" / f"{tag}.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, target_fn=target_fn, label=label)
    ax.set_title(f"Log-log plot ({tag})" + ("" if target_fn is not None else " -- exploratory, no known target"))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    if target_fn is None:
        print("no target_fn for this model -- skipping gamma-hat estimator comparison "
              "(would be an unvalidated number; see this script's module docstring)")
        return

    scales, y_bar, _se, n = loglog_points(samples)
    results = compare_methods(scales, y_bar, n, true_gamma=true_gamma)
    results["source"] = str(args.data)
    results_path = args.data / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    estimates_plot(results, ax=ax2)
    ax2.set_title(f"$\\hat\\gamma$ estimator comparison ({tag})")
    out2 = out.parent / f"{tag}_estimates.png"
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
