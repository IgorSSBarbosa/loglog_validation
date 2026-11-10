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

    One row per SWEEP, not per walker step (user, 2026-11-10): with 4 sweeps a step,
    rows per step showed revealed states jumping up to 4 sites and seemingly
    appearing away from the walker. Per sweep, a revealed state moves at most one
    site between rows, and every reveal is marked where the walker read it. The
    walker's path is a staircase: read and jump at integer t, then the sweeps.
    `simulate` asserts both rules (a new revealed cell appears only at a read, at
    the walker's site; a revealed state moves at most one site per sweep).

An illustration, not a measurement: drawn on the CPU on a window wide enough that
nothing revealed reaches its edge, not by the GPU sampler, and nothing is estimated.
`--check N` runs N independent walks of the figure's rules and compares the law of
X_steps with MODELS["rwre"] (full environment, models/rwre.py) by a two-sample KS
test, so the walk drawn is the RWRE's.

CLI:
    python3 experiments/50_rwre_lazy_long_omega/plot_dynamics.py [--out DIR] [--seed S] [--p P] [--check N]

Writes rwre_dynamics.pdf/.png into --out (default experiments/50_rwre_lazy_long_omega/images).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]

WALKER = "#e8622c"
PARTICLE = "#2a6fdb"
EMPTY = "#ffffff"
HIDDEN = "#ececec"
TRACK_EMPTY = "#555555"
SWEEPS = 4
UNREVEALED, HOLE, PART = -1, 0, 1


def simulate(p: float, steps: int, rng: np.random.Generator
             ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    """(environment rows, walker positions, reveals, tracks), on a centred window.

    rows[SWEEPS * t + s] is the environment after s sweeps of step t (s = 0: what
    the walker reads at time t); the last row is time `steps`. Rows hold -1
    (unrevealed), 0 (empty), 1 (particle). positions[t] = X_t, t = 0..steps.
    reveals lists (t, X_t) for each read of an unrevealed site; tracks[j] is the
    site of the j-th revealed state in every row from its reveal on. The window has
    2*W+1 sites with W = 4*steps: the walker moves at most `steps`, and a revealed
    state moves at most one site per sweep, so nothing revealed reaches the edge.
    """
    w = 4 * steps
    eta = np.full(2 * w + 1, UNREVEALED, dtype=np.int8)
    # Each revealed state carries an id through the swaps, to check how it moves.
    ident = np.full(eta.size, -1, dtype=np.int64)
    x = w
    lo, hi = min(p, 1 - p), max(p, 1 - p)
    rows, pos, reveals, where = [], [x], [], []
    for t in range(steps):
        rows.append(eta.copy())
        u = rng.random()
        if u < lo:
            left = True
        elif u >= hi:
            left = False
        else:                                  # only now does the site matter
            if eta[x] == UNREVEALED:
                eta[x] = PART if rng.random() < 0.5 else HOLE
                ident[x] = len(reveals)
                reveals.append((t, x - w))
                rows[-1][x] = eta[x]           # revealed at the read, drawn there
            left = u < (p if eta[x] == PART else 1 - p)
        where.append(_where(ident, len(reveals)) - w)
        x += -1 if left else 1
        for s in range(SWEEPS):
            where_before = _where(ident, len(reveals))
            b = int(rng.integers(2))
            left_sites = np.arange(b, eta.size - 1, 2)
            swap = left_sites[rng.random(left_sites.size) < 0.5]
            eta[swap], eta[swap + 1] = eta[swap + 1].copy(), eta[swap].copy()
            ident[swap], ident[swap + 1] = ident[swap + 1].copy(), ident[swap].copy()
            assert np.all(np.abs(_where(ident, len(reveals)) - where_before) <= 1), \
                "a revealed state moved more than one site in a sweep"
            if s < SWEEPS - 1:
                rows.append(eta.copy())
                where.append(_where(ident, len(reveals)) - w)
        pos.append(x)
    rows.append(eta.copy())
    where.append(_where(ident, len(reveals)) - w)
    rows = np.array(rows)
    _check_reveals(rows, reveals, w)
    tracks = [np.array([wr[j] for wr in where[SWEEPS * t:]])
              for j, (t, _) in enumerate(reveals)]
    return rows, np.array(pos) - w, reveals, tracks


def _where(ident: np.ndarray, count: int) -> np.ndarray:
    """Site of each revealed state, by id 0..count-1."""
    where = np.empty(count, dtype=np.int64)
    known = ident >= 0
    where[ident[known]] = np.flatnonzero(known)
    return where


def _check_reveals(rows: np.ndarray, reveals: list[tuple[int, int]], w: int) -> None:
    """The number of revealed cells grows only at a read, by one, at the walker."""
    known = (rows != UNREVEALED).sum(axis=1)
    growth = np.diff(np.concatenate([[0], known]))
    at = {SWEEPS * t: x for t, x in reveals}
    for r, g in enumerate(growth):
        assert g == (1 if r in at else 0), f"row {r}: revealed count changed by {g}"
    for r, x in at.items():
        assert rows[r, x + w] != UNREVEALED


def check_law(p: float, steps: int, n: int, seed: int) -> None:
    """KS of X_steps: these rules (lazy) against MODELS["rwre"] (full environment)."""
    from scipy.stats import ks_2samp
    sys.path.insert(0, str(ROOT / "models"))
    import rwre
    lazy_ss, full_ss = np.random.SeedSequence(seed).spawn(2)
    lazy_rng = np.random.default_rng(lazy_ss)
    lazy = np.array([simulate(p, steps, lazy_rng)[1][-1] for _ in range(n)])
    full = rwre.walk(steps, n, {"p": p}, np.random.default_rng(full_ss))
    res = ks_2samp(lazy, full)
    print(f"check: X_{steps} at p {p}, {n} walks each; mean lazy {lazy.mean():+.3f} "
          f"full {full.mean():+.3f}; sd lazy {lazy.std():.3f} full {full.std():.3f}; "
          f"KS D {res.statistic:.4f}, p-value {res.pvalue:.3f}")


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
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--half-width", type=int, default=24, help="sites shown each side")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="also KS-test the law of X_steps against MODELS['rwre'], N walks each")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check:
        check_law(args.p, args.steps, args.check, args.seed + 1)

    rows, pos, reveals, tracks = simulate(args.p, args.steps, np.random.default_rng(args.seed))
    w = (rows.shape[1] - 1) // 2
    h = args.half_width
    assert np.all(rows[:, :w - h] == UNREVEALED) and np.all(rows[:, w + h + 1:] == UNREVEALED), \
        "revealed sites outside the shown window: raise --half-width"

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    schematic(a)
    a.set_title("(a) one time step", fontsize=10, loc="left")
    # Each revealed state is drawn as its own track, one segment per sweep, from
    # the walker's read on; every site off the tracks is unrevealed.
    b.set_facecolor(HIDDEN)
    for (t0, _), tr in zip(reveals, tracks):
        y = t0 + np.arange(tr.size + 1) / SWEEPS
        y[-1] = args.steps                     # the last row is time `steps` itself
        value = rows[SWEEPS * t0, tr[0] + w]
        b.plot(np.repeat(tr, 2), np.repeat(y, 2)[1:-1], lw=0.9,
               color=PARTICLE if value == PART else TRACK_EMPTY)
    # staircase: read and jump at integer t, then the sweeps with the walker still
    t = np.arange(args.steps + 1)
    sx = np.repeat(pos, 2)[1:]
    sy = np.repeat(t, 2)[:-1]
    b.plot(sx, sy, color=WALKER, lw=1.4, zorder=2)
    rt, rx = np.array(reveals).T
    b.plot(rx, rt, ls="", marker="o", ms=3.2, mfc="white", mec="black", mew=0.7,
           zorder=3)
    b.set_xlim(-h - 0.5, h + 0.5)
    b.set_ylim(0, args.steps)
    b.set_xlabel("site $x$")
    b.set_ylabel("time $t$")
    b.set_title(f"(b) the walk at $p = {args.p:g}$, environment revealed lazily",
                fontsize=10, loc="left")
    handles = [plt.Line2D([], [], color=WALKER, lw=1.2, label="walker $X_t$"),
               plt.Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec="black",
                          label="site revealed")] + [
        plt.Line2D([], [], marker="s", ls="", ms=8, markerfacecolor=c,
                   markeredgecolor="#888888", label=lab)
        for c, lab in ((HIDDEN, "unrevealed"),)] + [
        plt.Line2D([], [], color=c, lw=1.2, label=lab)
        for c, lab in ((PARTICLE, "revealed particle"), (TRACK_EMPTY, "revealed empty"))]
    b.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"rwre_dynamics.{ext}", dpi=200)
    print(f"wrote {args.out}/rwre_dynamics.pdf/.png  (seed {args.seed}, p {args.p}, "
          f"{args.steps} steps, {SWEEPS} rows per step; {len(reveals)} reveals; "
          f"walker ends at {pos[-1]})")


if __name__ == "__main__":
    main()
