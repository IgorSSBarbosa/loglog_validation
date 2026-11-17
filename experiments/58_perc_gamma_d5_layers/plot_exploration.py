"""The article's percolation illustration: the cluster of the origin grown layer by layer.

Replaces experiment 51's stack exploration (user, 2026-11-17: the figure should show
the layer algorithm). Drawn in d = 2 below p_c so that it can be seen. As in
models/percolation_susceptibility_layers.py, the cluster grows breadth first by
chemical distance l: layer l+1 is the set of unseen open neighbours of layer l.
Neighbouring open sites differ in l by at most 1, so while layer l is expanded only
layers l-1, l, l+1 can be looked up, and every layer <= l-2 is dropped from memory.

The first three panels show the state after layer l has been expanded: the dropped
layers, the live layers l-1 and l, the new layer l+1, and the closed sites read so
far. The last panel colours the whole cluster by l.

An illustration, not a measurement: site states are drawn lazily from a numpy
Generator instead of the model's hash, and the seed is the first one, counting up
from --seed, whose cluster has between --min and --max sites, so that the picture
is legible. That selection conditions on the size: nothing is estimated from it.

CLI:
    python3 experiments/58_perc_gamma_d5_layers/plot_exploration.py [--out DIR] [--seed S] [--p P]

Writes perc_exploration.pdf/.png into --out (default experiments/58_perc_gamma_d5_layers/images).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

EXP = Path(__file__).resolve().parent

DROPPED = "#c9daf5"
LIVE = "#2a6fdb"
NEW = "#e8622c"
CLOSED = "#9a9a9a"

Site = tuple[int, int]


def explore(p: float, rng: np.random.Generator) -> tuple[list[list[Site]], list[set[Site]]]:
    """Grow C(0) on Z^2 breadth first. Returns (layers, closed sites read while
    expanding each layer)."""
    state: dict[Site, bool] = {(0, 0): True}   # the origin is conditioned open
    layers = [[(0, 0)]]
    closed_by_layer = []
    while layers[-1]:
        nxt, closed = [], set()
        for site in layers[-1]:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (site[0] + dx, site[1] + dy)
                if nb in state:
                    continue
                state[nb] = bool(rng.random() < p)
                if state[nb]:
                    nxt.append(nb)
                else:
                    closed.add(nb)
        layers.append(nxt)
        closed_by_layer.append(closed)
    layers.pop()
    return layers, closed_by_layer


def square(ax: plt.Axes, site: Site, **kw) -> None:
    ax.add_patch(Rectangle((site[0] - 0.45, site[1] - 0.45), 0.9, 0.9, **kw))


def frame(ax: plt.Axes, box: tuple[int, int, int, int]) -> None:
    x0, x1, y0, y1 = box
    ax.plot([0], [0], marker="+", ms=8, color="black", mew=1.2)
    ax.set_xlim(x0 - 1.5, x1 + 1.5)
    ax.set_ylim(y0 - 1.5, y1 + 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


def draw_stage(ax: plt.Axes, layers: list[list[Site]], closed_by_layer: list[set[Site]],
               l: int, box: tuple[int, int, int, int]) -> int:
    """State after expanding layer l. Returns the number of live (stored) sites."""
    for k in range(l + 1):
        for site in closed_by_layer[k]:
            square(ax, site, facecolor=CLOSED, edgecolor="none")
    for k in range(l + 2):
        colour = DROPPED if k <= l - 2 else LIVE if k <= l else NEW
        for site in layers[k]:
            square(ax, site, facecolor=colour, edgecolor="none")
    frame(ax, box)
    return sum(len(layers[k]) for k in range(max(0, l - 1), l + 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=EXP / "images")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--p", type=float, default=0.55, help="below p_c = 0.5927 on Z^2")
    ap.add_argument("--min", type=int, default=150)
    ap.add_argument("--max", type=int, default=300)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    while True:
        layers, closed_by_layer = explore(args.p, np.random.default_rng(seed))
        size = sum(len(layer) for layer in layers)
        if args.min <= size <= args.max:
            break
        seed += 1
    sites = [s for layer in layers for s in layer]
    sites += [s for c in closed_by_layer for s in c]
    xs = [s[0] for s in sites]
    ys = [s[1] for s in sites]
    box = (min(xs), max(xs), min(ys), max(ys))
    depth = len(layers) - 1

    stages = (depth // 4, depth // 2, 3 * depth // 4)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, l in zip(axes[:3], stages):
        live = draw_stage(ax, layers, closed_by_layer, l, box)
        ax.set_title(f"layer $\\ell = {l}$ expanded: {live} stored", fontsize=9, loc="left")
    cmap = plt.get_cmap("viridis")
    for k, layer in enumerate(layers):
        for site in layer:
            square(axes[3], site, facecolor=cmap(k / depth), edgecolor="none")
    frame(axes[3], box)
    axes[3].set_title(f"done: $|C(0)| = {size}$, depth {depth}", fontsize=9, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, depth))
    cb = fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.03)
    cb.set_label("chemical distance $\\ell$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = [Rectangle((0, 0), 1, 1, facecolor=DROPPED, label="layers $\\leq \\ell-2$, dropped"),
               Rectangle((0, 0), 1, 1, facecolor=LIVE, label="layers $\\ell-1, \\ell$, stored"),
               Rectangle((0, 0), 1, 1, facecolor=NEW, label="layer $\\ell+1$, stored"),
               Rectangle((0, 0), 1, 1, facecolor=CLOSED, label="closed, read"),
               plt.Line2D([], [], marker="+", ls="", ms=8, color="black", mew=1.2,
                          label="origin")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"perc_exploration.{ext}", dpi=200)
    n_closed = sum(len(c) for c in closed_by_layer)
    print(f"wrote {args.out}/perc_exploration.pdf/.png  (p {args.p}, seed {seed}, "
          f"|C(0)| {size}, depth {depth}, stages {stages}, {n_closed} closed sites read)")


if __name__ == "__main__":
    main()
