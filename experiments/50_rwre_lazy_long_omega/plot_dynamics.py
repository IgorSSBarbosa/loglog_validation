"""The article's RWRE illustration: one time step, and a space-time diagram.

(a) A schematic of one step of the dynamics: the walker reads eta_t(X_t) and steps
    left with probability p on a particle, 1-p on an empty site; then the
    environment makes 4 brick-wall sweeps, each swapping the two sites of every bond
    {x, x+1} with x = b (mod 2) independently with probability 1/2.
(b) A space-time diagram of a short walk drawn by exactly those rules, with the
    environment revealed lazily as in the article: a site is drawn Bernoulli(1/2)
    only the first time the walker reads it, and the walker reads it only when its
    uniform u falls between min(p, 1-p) and max(p, 1-p). Occupied, empty and
    still-unrevealed sites are shaded differently, and the walker's path is drawn on top.

An illustration, not a measurement: drawn on the CPU on a window wide enough that
nothing revealed reaches its edge, not by the GPU sampler, and nothing is estimated.

CLI:
    python3 experiments/50_rwre_lazy_long_omega/plot_dynamics.py [--out DIR] [--seed S] [--p P]

Writes rwre_dynamics.pdf/.png into --out (default experiments/50_rwre_lazy_long_omega/images).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

EXP = Path(__file__).resolve().parent

WALKER = "#e8622c"
PARTICLE = "#2a6fdb"
EMPTY = "#ffffff"
HIDDEN = "#dcdcdc"
SWEEPS = 4
UNREVEALED, HOLE, PART = -1, 0, 1


def simulate(p: float, steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(environment rows, walker positions) for t = 0..steps, on a centred window.

    Rows hold -1 (unrevealed), 0 (empty), 1 (particle). The window has 2*W+1 sites
    with W = 4*steps: the walker moves at most `steps`, and a revealed state moves at
    most one site per sweep, so nothing revealed reaches the edge.
    """
    w = 4 * steps
    eta = np.full(2 * w + 1, UNREVEALED, dtype=np.int8)
    x = w
    lo, hi = min(p, 1 - p), max(p, 1 - p)
    rows, pos = [eta.copy()], [x]
    for _ in range(steps):
        u = rng.random()
        if u < lo:
            left = True
        elif u >= hi:
            left = False
        else:                                  # only now does the site matter
            if eta[x] == UNREVEALED:
                eta[x] = PART if rng.random() < 0.5 else HOLE
            left = u < (p if eta[x] == PART else 1 - p)
        x += -1 if left else 1
        for _ in range(SWEEPS):
            b = int(rng.integers(2))
            left_sites = np.arange(b, eta.size - 1, 2)
            swap = left_sites[rng.random(left_sites.size) < 0.5]
            eta[swap], eta[swap + 1] = eta[swap + 1].copy(), eta[swap].copy()
        rows.append(eta.copy())
        pos.append(x)
    return np.array(rows), np.array(pos) - w


def schematic(ax: plt.Axes) -> None:
    """One step: the walker's read and jump, then one sweep's swaps."""
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-3.3, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    occ = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    xw = 4
    for row_y, label in ((0.0, "time $t$"), (-2.4, "after one sweep ($b = 0$)")):
        ax.text(-0.6, row_y + 0.4, label, fontsize=8, ha="left", va="bottom")
    # top row: environment at time t
    for k, o in enumerate(occ):
        ax.add_patch(Circle((k, 0), 0.28, facecolor=PARTICLE if o else EMPTY,
                            edgecolor="black", lw=0.8))
    ax.plot([xw], [0.75], marker="v", ms=11, color=WALKER)
    ax.text(xw, 1.75, r"walker at $X_t$ reads $\eta_t(X_t)$", fontsize=8, ha="center")
    for dx in (-1, 1):
        ax.add_patch(FancyArrowPatch((xw, 0.75), (xw + dx, 0.75), arrowstyle="-|>",
                                     mutation_scale=9, color=WALKER, lw=1.1,
                                     connectionstyle=f"arc3,rad={-0.5 * dx}"))
    ax.text(xw - 1.15, 1.05, r"left: $p$ if $\eta=1$," "\n" r"$1-p$ if $\eta=0$",
            fontsize=7.5, ha="right", va="center")
    ax.text(xw + 1.15, 1.05, "right: otherwise", fontsize=7.5, ha="left", va="center")
    # one sweep, parity b = 0: bonds {0,1}, {2,3}, {4,5}, {6,7}; each swaps w.p. 1/2
    swapped = [True, False, True, False]
    new = occ.copy()
    for j, s in enumerate(swapped):
        a = 2 * j
        ax.add_patch(FancyArrowPatch((a, -0.45), (a + 1, -0.45), arrowstyle="<|-|>",
                                     mutation_scale=7, lw=1.0,
                                     color="black" if s else "#aaaaaa",
                                     connectionstyle="arc3,rad=0.6"))
        if s:
            new[a], new[a + 1] = new[a + 1], new[a]
    ax.text(4, -1.25, r"each of $4$ sweeps: draw a parity $b$, swap every bond "
            r"$\{x, x{+}1\}$, $x \equiv b$, w.p. $1/2$", fontsize=7.5, ha="center")
    for k, o in enumerate(new):
        ax.add_patch(Circle((k, -2.4), 0.28, facecolor=PARTICLE if o else EMPTY,
                            edgecolor="black", lw=0.8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=EXP / "images")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--half-width", type=int, default=32, help="sites shown each side")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, pos = simulate(args.p, args.steps, np.random.default_rng(args.seed))
    w = (rows.shape[1] - 1) // 2
    h = args.half_width
    view = rows[:, w - h:w + h + 1]
    assert np.all(rows[:, :w - h] == UNREVEALED) and np.all(rows[:, w + h + 1:] == UNREVEALED), \
        "revealed sites outside the shown window: raise --half-width"

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    schematic(a)
    a.set_title("(a) one time step", fontsize=10, loc="left")
    cmap = ListedColormap([HIDDEN, EMPTY, PARTICLE])
    b.imshow(view + 1, cmap=cmap, vmin=0, vmax=2, origin="lower", aspect="auto",
             interpolation="nearest", extent=(-h - 0.5, h + 0.5, -0.5, args.steps + 0.5))
    b.plot(pos, np.arange(args.steps + 1), color=WALKER, lw=1.6)
    b.set_xlabel("site $x$")
    b.set_ylabel("time $t$")
    b.set_title(f"(b) the walk at $p = {args.p:g}$, environment revealed lazily",
                fontsize=10, loc="left")
    handles = [plt.Line2D([], [], color=WALKER, lw=1.6, label="walker $X_t$")] + [
        plt.Line2D([], [], marker="s", ls="", ms=8, markerfacecolor=c,
                   markeredgecolor="#888888", label=lab)
        for c, lab in ((PARTICLE, "particle"), (EMPTY, "empty"), (HIDDEN, "unrevealed"))]
    b.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"rwre_dynamics.{ext}", dpi=200)
    print(f"wrote {args.out}/rwre_dynamics.pdf/.png  (seed {args.seed}, p {args.p}, "
          f"{args.steps} steps; walker ends at {pos[-1]})")


if __name__ == "__main__":
    main()
