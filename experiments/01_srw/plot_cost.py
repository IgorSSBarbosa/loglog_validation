"""Log-log plot of a cost-model probe's persisted timing data (measure_cost.py).

Thin glue, mirroring this repo's generate.py/plot_loglog.py split (and
00_synthetic's generator.py/plot_loglog.py before it): measure_cost.py only
measures and saves, this script only loads and plots -- previously
measure_cost.py bundled both behind a --plot flag, inconsistent with how
every other experiment separates generation from plotting.

Takes a **data path** (the JSON measure_cost.py writes -- not a .npz+.json
pair like generate.py/generator.py, since a cost probe's own output is
already one self-contained file; "elapsed_all" inside it is already
{scale: array of repeat times}, tools/loglog_plot.py's exact required
shape, so no reformatting is needed).

Run (after measuring, e.g. `measure_cost.py --tag demo_run`):

    python3 plot_cost.py -data data/demo_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for tools/

from tools.loglog_plot import loglog_plot  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot elapsed time vs scale (log-log) from measure_cost.py's saved output."
    )
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Data JSON written by measure_cost.py (e.g. data/cost_probe.json).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<data-stem>.png",
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        parser.error(
            f"no data at {args.data}.\n"
            f"Measure some first: python3 measure_cost.py --tag <name>\n"
            f"then pass the printed 'output' path (data/<name>.json) here."
        )
    result = json.loads(args.data.read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = {int(k): np.asarray(v) for k, v in result["elapsed_all"].items()}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, label="elapsed time (all repeats)")
    ax.set_xlabel("k")
    ax.set_ylabel("elapsed time (s)")
    ax.set_title(f"srw() cost probe ({args.data.stem}): d_hat={result['d_hat']:.3f}")

    out = args.out or (HERE / "images" / f"{args.data.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
