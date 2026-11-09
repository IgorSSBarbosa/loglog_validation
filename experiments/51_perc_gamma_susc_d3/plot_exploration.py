"""The article's percolation illustration: the cluster of the origin grown by exploration.

Four snapshots of one exploration below p_c, drawn in d = 2 so that it can be seen
(the runs are in d = 3; the algorithm is the same). As in
models/percolation_susceptibility_explore.py, the search pops a site from a stack,
reads the state of each neighbour the first time it is touched, and pushes the open
ones it has not stored yet; only the open sites of the cluster are stored. The
panels show the open sites found, the closed sites read, the sites still on the
stack and the site being expanded; unread sites are blank.

An illustration, not a measurement: site states are drawn lazily from a numpy
Generator instead of the model's hash, and the seed is the first one, counting up
from --seed, whose cluster has between --min and --max sites, so that the picture
is legible. That selection conditions on the size: nothing is estimated from it.

CLI:
    python3 experiments/51_perc_gamma_susc_d3/plot_exploration.py [--out DIR] [--seed S] [--p P]

Writes perc_exploration.pdf/.png into --out (default experiments/51_perc_gamma_susc_d3/images).
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

OPEN = "#2a6fdb"
CLOSED = "#9a9a9a"
STACK = "#e8622c"
P_C_SQUARE_SITE = 0.592746


def explore(p: float, rng: np.random.Generator) -> tuple[dict, list]:
    """Grow C(0) on Z^2. Returns (state of every site read, event log).

    The log holds one entry per pop: (site popped, list of (neighbour, open?, pushed?)).
    """
    state = {(0, 0): True}                     # the origin is conditioned open
    stack = [(0, 0)]
    log = []
    while stack:
        site = stack.pop()
        reads = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (site[0] + dx, site[1] + dy)
            if nb in state:
                reads.append((nb, state[nb], False))
                continue
            state[nb] = bool(rng.random() < p)
            if state[nb]:
                stack.append(nb)
            reads.append((nb, state[nb], state[nb]))
        log.append((site, reads))
    return state, log


def snapshot(log: list, upto: int) -> tuple[set, set, list, tuple]:
    """(open found, closed read, stack, site being expanded) after `upto` pops."""
    found, closed, stack = {(0, 0)}, set(), [(0, 0)]
    current = (0, 0)
    for site, reads in log[:upto]:
        stack.pop()
        current = site
        for nb, is_open, pushed in reads:
            if is_open:
                found.add(nb)
                if pushed:
                    stack.append(nb)
            else:
                closed.add(nb)
    return found, closed, stack, current


def draw(ax: plt.Axes, found: set, closed: set, stack: list, current: tuple | None,
         box: tuple[int, int, int, int]) -> None:
    x0, x1, y0, y1 = box
    for (x, y) in closed:
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor=CLOSED,
                               edgecolor="none"))
    for (x, y) in found:
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor=OPEN,
                               edgecolor="none"))
    for (x, y) in stack:
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, facecolor="none",
                               edgecolor=STACK, lw=1.4))
    if current is not None:
        ax.plot([current[0]], [current[1]], marker="o", ms=4, color="white",
                markeredgecolor="black", markeredgewidth=0.6)
    ax.plot([0], [0], marker="+", ms=8, color="black", mew=1.2)
    ax.set_xlim(x0 - 1, x1 + 1)
    ax.set_ylim(y0 - 1, y1 + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


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
        state, log = explore(args.p, np.random.default_rng(seed))
        size = sum(state.values())
        if args.min <= size <= args.max:
            break
        seed += 1
    xs = [s[0] for s in state]
    ys = [s[1] for s in state]
    box = (min(xs), max(xs), min(ys), max(ys))

    pops = len(log)
    stages = (max(1, pops // 10), pops // 3, 2 * pops // 3)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, k in zip(axes[:3], stages):
        found, closed, stack, current = snapshot(log, k)
        draw(ax, found, closed, stack, current, box)
        ax.set_title(f"after {k} of {pops} pops", fontsize=9, loc="left")
    found, closed, _, _ = snapshot(log, pops)
    draw(axes[3], found, closed, [], None, box)
    axes[3].set_title(f"done: $|C(0)| = {len(found)}$", fontsize=9, loc="left")
    handles = [Rectangle((0, 0), 1, 1, facecolor=OPEN, label="open, stored"),
               Rectangle((0, 0), 1, 1, facecolor=CLOSED, label="closed, read"),
               Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=STACK, lw=1.4,
                         label="on the stack"),
               plt.Line2D([], [], marker="o", ls="", ms=5, color="white",
                          markeredgecolor="black", label="being expanded"),
               plt.Line2D([], [], marker="+", ls="", ms=8, color="black", mew=1.2,
                          label="origin")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"perc_exploration.{ext}", dpi=200)
    print(f"wrote {args.out}/perc_exploration.pdf/.png  (p {args.p}, seed {seed}, "
          f"|C(0)| {len(found)}, {len(closed)} closed sites read, {pops} pops)")


if __name__ == "__main__":
    main()
