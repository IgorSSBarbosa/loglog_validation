"""The introduction's site-percolation picture (article Figure 1, `fig:criticalwindow`).

A 3x3 grid: rows are p below, at and above p_c, columns are box sizes i (the
article's system size). Closed sites are black; open clusters are coloured by
their size on a log scale, so the largest clusters stand out.

The figure is illustrative only -- no estimate is read off it -- so it does not
go through generate.py. Columns are labelled "box size = i sites" because the
article reserves n for the number of samples.

Run:

    python3 src/report/plot_percolation_cut.py --out ../article_writting/images/percolation_cut.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from rng import seed_record, spawn  # noqa: E402

PROBS = (0.580, 0.593, 0.600)
ROW_LABELS = ("Subcritical\n($p < p_c$)", "Critical\n($p \\approx p_c$)", "Supercritical\n($p > p_c$)")
SIZES = (100, 500, 1000)


def cluster_size_map(open_sites: np.ndarray) -> np.ndarray:
    """Size of the open cluster each site belongs to (0 on closed sites), 4-neighbour."""
    from scipy import ndimage

    labels, _ = ndimage.label(open_sites)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return sizes[labels]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output image path (.png or .pdf)")
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    children = spawn(args.seed, len(PROBS) * len(SIZES))
    cmap = plt.get_cmap("gist_ncar").copy()
    cmap.set_bad("black")      # closed sites

    fig, axes = plt.subplots(len(PROBS), len(SIZES), figsize=(10, 9.5))
    norm = None
    for r, p in enumerate(PROBS):
        for c, i in enumerate(SIZES):
            rng = np.random.default_rng(children[r * len(SIZES) + c])
            sizes = cluster_size_map(rng.random((i, i)) < p)
            # One colour scale per panel: cluster sizes grow with the box, and a
            # shared scale would leave the small boxes almost uniformly dark.
            norm = LogNorm(vmin=1, vmax=max(int(sizes.max()), 2))
            ax = axes[r, c]
            ax.imshow(np.ma.masked_equal(sizes, 0), cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"$p = {p:.3f}$", fontsize=10)
            if r == 0:
                ax.annotate(f"box size = {i} sites", xy=(0.5, 1.18), xycoords="axes fraction",
                            ha="center", fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=12, fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(["small", "big"])
    cbar.set_label("cluster size (log scale, per panel)")
    fig.suptitle("Site percolation", fontsize=15, fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    # The seed is printed rather than written beside the image, so the article's
    # images/ folder only receives the figure itself.
    print(f"wrote {args.out} (seed {json.dumps(seed_record(args.seed))})")


if __name__ == "__main__":
    main()
