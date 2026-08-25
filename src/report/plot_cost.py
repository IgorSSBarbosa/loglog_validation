"""Single, shared log-log plot of a cost-model probe's persisted timing data
(src/measure_cost.py) -- one script, reused across every model, rather
than a per-experiment copy.

Thin glue, mirroring generate.py/plot_loglog.py's split: measure_cost.py
only measures and saves, this script only loads and plots.

Takes a **run directory** (`<out_dir>/<tag>/`, as src/measure_cost.py
writes -- holding `result.json`, not the samples.npz+samples_meta.json pair
generate.py's runs use, since a cost probe's own output is already one
self-contained file). "elapsed_all" inside it is already {scale: array of
repeat times}, tools/loglog_plot.py's exact required shape, so no
reformatting is needed.

Writes `<run_dir>/plot.png` -- the same folder as `result.json` -- matching
plot_loglog.py's convention: everything about one run lives in one place,
and since `data/` is gitignored, nothing here is auto-committed. Copy a
specific plot into the experiment's `images/` folder when you want to keep
it as evidence (ground rule 1/6 -- committed deliberately, one at a time).

Run (after measuring, e.g. `measure_cost.py -meta ../../experiments/01_srw/recipes/cost_probe.json
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
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from artifacts import artifact_path  # noqa: E402
from loglog_plot import loglog_plot  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot elapsed time vs scale (log-log) from a run directory src/measure_cost.py wrote."
    )
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Run directory written by src/measure_cost.py (e.g. ../experiments/01_srw/data/cost_probe).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to <run_dir>/plot.png (same folder as result.json).",
    )
    args = parser.parse_args(argv)

    result_path = artifact_path(args.data, "cost_probe")
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

    out = args.out or (args.data / "plot.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
