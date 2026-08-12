"""Single, shared log-log plotter for every model's persisted data -- reads
the run's own metadata to dispatch, instead of each experiment keeping its
own copy of this driver.

Thin glue: loads the samples src/generate.py already saved, and hands them
to the generic tools/loglog_plot.py / tools/loglog.py (neither of which
knows about any specific model). The raw-data plot always overlays the
all-points OLS fit (solid line, gamma_hat in the legend) -- this needs no
known ground truth, it's computed from the data itself. When the run's
model (tools/models.py) has a known closed form (`target_fn` -- currently
only "synthetic"), the known E[Y_i] curve is *also* overlaid (dashed), for
comparison against the fit.

results.json (tools/loglog.py's compare_methods -- all four gamma-hat
estimators, not just all-points) is always written, run for any model,
regardless of whether a known target_fn exists -- comparing estimators
against each other doesn't require a known ground truth, only comparing
against one does. When true_gamma is unknown (e.g. "srw" -- no
article-sanctioned closed form yet, see experiments/01_srw/README.md), a
note makes clear the numbers are exploratory, not a validated checkpoint
result. The four-estimator comparison chart (estimates.png) is opt-in via
--estimates, since unlike results.json it's a supplementary figure, not the
numeric result itself (ground rule 1).

Takes a **run directory** (`<out_dir>/<tag>/`, as src/generate.py writes),
not a recipe -- a single recipe can produce many different runs (different
tags/seeds), so pointing this at a recipe would be ambiguous about which
run's data you mean. Metadata is read from `<run_dir>/metadata.json` if
present, but isn't required -- missing metadata just means no model
dispatch (no target_fn overlay, no true_gamma), not a failure to plot or
fit.

Writes `<run_dir>/plot.png` and `<run_dir>/results.json` -- the same folder
as samples.npz/metadata.json -- and, with --estimates,
`<run_dir>/estimates.png`. Everything about one run lives in one place.
`data/` is gitignored, so none of this is committed by default; copy a
specific plot into the experiment's `images/` folder when you want to keep
it as evidence (ground rule 1/6 -- committed deliberately, one at a time,
not auto-populated by every run).

Run (after generating some data, e.g. `generate.py -meta
../experiments/00_synthetic/example_config.json --tag demo_run`):

    python3 plot_loglog.py -data ../experiments/00_synthetic/data/demo_run
    python3 plot_loglog.py -data ../experiments/00_synthetic/data/demo_run --estimates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports

from loglog import compare_methods  # noqa: E402
from loglog_plot import estimates_plot, loglog_plot, loglog_points  # noqa: E402
from models import get_model  # noqa: E402
from persistence import load_metadata, load_samples  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Y_bar_i vs i (log-log) from a run directory src/generate.py wrote."
    )
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Run directory written by src/generate.py (e.g. ../experiments/00_synthetic/data/demo_run).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to <run_dir>/plot.png (same folder as samples.npz).",
    )
    parser.add_argument(
        "--estimates", dest="estimates", action="store_true",
        help="Also save the four-estimator comparison chart to <run_dir>/estimates.png "
        "(tools/loglog_plot.py's estimates_plot). Off by default; results.json (the "
        "underlying numbers, all four estimators) is written either way.",
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

    scales, y_bar, _se, n = loglog_points(samples)
    results = compare_methods(scales, y_bar, n, true_gamma=true_gamma)
    results["source"] = str(args.data)
    results_path = args.data / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    all_points = results["methods"]["all_points"]
    gamma_hat, a0_hat = all_points["gamma_hat"], all_points["a0_hat"]
    fit_fn = lambda i, _g=gamma_hat, _a=a0_hat: _a * np.asarray(i, dtype=np.float64) ** _g  # noqa: E731

    out = args.out or (args.data / "plot.png")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(
        samples, ax=ax, target_fn=target_fn, label=label,
        fit_fn=fit_fn, fit_label=rf"OLS fit ($\hat\gamma$={gamma_hat:.4f})",
    )
    ax.set_title(f"Log-log plot ({tag})")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Results written to {results_path}")

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

    if true_gamma is None:
        print("\nnote: no known true_gamma for this model -- gamma-hat values above are "
              "exploratory, not checked against a validated ground truth")

    if not args.estimates:
        return

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    estimates_plot(results, ax=ax2)
    ax2.set_title(f"$\\hat\\gamma$ estimator comparison ({tag})")
    out2 = args.data / "estimates.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
