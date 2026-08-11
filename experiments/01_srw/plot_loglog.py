"""Log-log plot of an SRW experiment's persisted data.

Thin glue: loads the samples generate.py already saved, and hands them to
the generic, experiment-agnostic tools/loglog_plot.py -- same shape as
experiments/00_synthetic/plot_loglog.py, minus the reference-curve overlay
and gamma-hat estimator comparison. Deliberately so: there is still no
article-sanctioned closed-form E[Y_i]/gamma/omega1 for SRW (see README's
"Gamma-estimation ladder" section, still blocked) to overlay or validate a
gamma_hat against -- running tools/loglog.py's estimators on this data would
produce *a* number, but not a checked one, and could be mistaken for
progress on the still-blocked Phase 1 ladder. This script only plots the
raw data.

Takes a **data path** (the .npz), not a JSON, for the same reason
00_synthetic's plot_loglog.py does: a recipe can produce many runs.

Run (after generating some data, e.g. `generate.py -meta example_config.json
--tag demo_run`):

    python3 plot_loglog.py -data data/demo_run.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for tools/
sys.path.insert(0, str(HERE))  # this dir, for generate

from tools.loglog_plot import loglog_plot  # noqa: E402

from generate import load_metadata, load_samples  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot |S_k| mean vs k (log-log) from generate.py's saved output.")
    parser.add_argument(
        "-data", "--data", dest="data", required=True, type=Path,
        help="Data .npz written by generate.py (e.g. data/demo_run.npz).",
    )
    parser.add_argument(
        "-o", "--out", dest="out", type=Path, default=None,
        help="Output PNG path; defaults to images/<data-stem>.png",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        samples = load_samples(args.data)
    except FileNotFoundError as e:
        parser.error(str(e))

    meta = load_metadata(args.data)
    q = meta["params"]["q"] if meta is not None else None
    if meta is None:
        print(f"note: no metadata at {args.data.with_suffix('.json')} -- plotting data only")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    loglog_plot(samples, ax=ax, label=f"q={q}" if q is not None else None)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\overline{|S_k|}$")
    ax.set_title(f"SRW log-log plot ({args.data.stem}) -- exploratory, no validated gamma target")

    out = args.out or (HERE / "images" / f"{args.data.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
