"""Log-log plot of a synthetic experiment's data.

Thin glue: reads a JSON config in generator.py's shape (params, scales, n,
seed -- see example_config.json), draws the samples via `generator.generate`,
and hands them to the generic, experiment-agnostic `tools/loglog_plot.py`
(which never imports from `experiments/`, so it works the same way for SRW,
percolation, etc. once those exist). Since the synthetic model's E[Y_i] is
known exactly, the true curve is overlaid as a reference.

Run: python3 plot_loglog.py -meta example_config.json
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

from generator import generate, mean_Y, params_from_json  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Y_bar_i vs i (log-log) for a synthetic config.")
    parser.add_argument("-meta", "--meta", dest="meta", required=True, type=Path)
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<meta-stem>.png",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    params = params_from_json(cfg["params"])
    samples = generate(cfg["scales"], cfg["n"], params, seed=cfg.get("seed"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, target_fn=lambda i: mean_Y(i, params), label=params.family)
    ax.set_title(f"Synthetic log-log plot ({args.meta.name})")

    out = args.out or (HERE / "images" / f"{args.meta.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
