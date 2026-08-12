"""Single, shared log-log plot of a cost-model probe's persisted timing data
(tools/measure_cost.py) -- one script, reused across every model, rather
than a per-experiment copy.

Thin glue, mirroring generate.py/plot_loglog.py's split: measure_cost.py
only measures and saves, this script only loads and plots.

Takes a **run directory** (`<out_dir>/<tag>/`, as tools/measure_cost.py
writes -- holding `result.json`, not the samples.npz+metadata.json pair
generate.py's runs use, since a cost probe's own output is already one
self-contained file). "elapsed_all" inside it is already {scale: array of
repeat times}, tools/loglog_plot.py's exact required shape, so no
reformatting is needed.

Run (after measuring, e.g. `measure_cost.py -meta cost_probe_config.json
--tag demo_run`):

    python3 plot_cost.py -data ../experiments/01_srw/data/demo_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this dir, for bare imports below

from loglog_plot import loglog_plot  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot elapsed time vs scale (log-log) from a run directory tools/measure_cost.py wrote."
    )
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Run directory written by tools/measure_cost.py (e.g. ../experiments/01_srw/data/cost_probe).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<tag>.png next to the run's experiment folder.",
    )
    args = parser.parse_args(argv)

    result_path = args.data / "result.json"
    if not result_path.exists():
        parser.error(
            f"no data at {result_path}.\n"
            f"Measure some first: python3 measure_cost.py -meta <recipe.json> --tag <name>\n"
            f"then pass the printed run directory here."
        )
    result = json.loads(result_path.read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = args.data.name
    samples = {int(k): np.asarray(v) for k, v in result["elapsed_all"].items()}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, label="elapsed time (all repeats)")
    ax.set_xlabel("k")
    ax.set_ylabel("elapsed time (s)")
    ax.set_title(f"{result['model']}() cost probe ({tag}): d_hat={result['d_hat']:.3f}")

    out = args.out or (args.data.resolve().parents[1] / "images" / f"{tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
