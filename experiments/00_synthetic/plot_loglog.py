"""Log-log plot of a synthetic experiment's persisted data.

Thin glue: loads the samples generator.py already saved (the .npz paired with
a metadata JSON, same stem -- see generator.py's docstring), and hands them to
the generic, experiment-agnostic `tools/loglog_plot.py` (which never imports
from `experiments/`, so it works the same way for SRW, percolation, etc. once
those exist). Since the synthetic model's E[Y_i] is known exactly, the true
curve is overlaid as a reference. This script only reads data -- it never
generates any, so run generator.py first.

Run (after generating some data, e.g. `generator.py -meta example_config.json
--tag demo_run`):

    python3 plot_loglog.py -meta data/demo_run.json
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

from tools.loglog_plot import loglog_plot  # noqa: E402

from generator import load, mean_Y, params_from_json  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Y_bar_i vs i (log-log) from generator.py's saved output.")
    parser.add_argument(
        "-meta", "--meta", dest="meta", required=True, type=Path,
        help="Metadata JSON written by generator.py; its paired <stem>.npz is loaded.",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<meta-stem>.png",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    params = params_from_json(cfg["params"])
    samples = load(args.meta)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, target_fn=lambda i: mean_Y(i, params), label=params.family)
    ax.set_title(f"Synthetic log-log plot ({args.meta.stem})")

    out = args.out or (HERE / "images" / f"{args.meta.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
