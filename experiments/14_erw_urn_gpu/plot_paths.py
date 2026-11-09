"""The article's ERW illustration: sample paths of S_i in the three regimes.

One panel per regime -- diffusive (p = 0.6, the worked example), critical (p = 3/4),
superdiffusive (p = 0.9) -- with a few independent walks, the envelope
+-2 sqrt(E S_i^2) from the exact second-moment recursion
E S_{k+1}^2 = 1 + (1 + 2(2p-1)/k) E S_k^2 (Bercu 2018), and, on one walk, the
observable Y_i = |S_i| at the last step.

An illustration, not a measurement: the walks are drawn on the CPU by the
definition in the article (step k uniform on 1..n, repeated with probability p),
not by the GPU urn sampler, and nothing here is estimated.

CLI:
    python3 experiments/14_erw_urn_gpu/plot_paths.py [--out DIR] [--seed S]

Writes erw_paths.pdf/.png into --out (default experiments/14_erw_urn_gpu/images).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parent

PATH = "#2a6fdb"      # walks
HIGHLIGHT = "#e8622c" # the walk whose |S_i| is marked
REF = "#7a7a7a"       # envelope
THIN = 10             # draw every 10th step: 1000 points a walk is below what a panel resolves

REGIMES = ((0.6, "diffusive, $p = 0.6$", r"$\gamma = 1/2$"),
           (0.75, "critical, $p = 3/4$", r"$\sqrt{i\log i}$: no $\gamma$"),
           (0.9, "superdiffusive, $p = 0.9$", r"$\gamma = 2p-1 = 0.8$"))


def walk(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """S_1..S_n of one ERW, by the definition: X_{k+1} = +-X_u, u uniform on 1..k."""
    x = np.empty(n, dtype=np.int8)
    x[0] = 1 if rng.random() < 0.5 else -1
    u = rng.integers(0, np.arange(1, n), endpoint=False)   # u_k uniform on 0..k-1
    keep = rng.random(n - 1) < p
    for k in range(1, n):
        x[k] = x[u[k - 1]] if keep[k - 1] else -x[u[k - 1]]
    return np.cumsum(x, dtype=np.int64)


def second_moment(p: float, n: int) -> np.ndarray:
    """E S_k^2, k = 1..n, from the exact recursion with E S_1^2 = 1."""
    m = np.empty(n)
    m[0] = 1.0
    for k in range(1, n):
        m[k] = 1.0 + (1.0 + 2.0 * (2.0 * p - 1.0) / k) * m[k - 1]
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=EXP / "images")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--n", type=int, default=10_000, help="steps per walk")
    ap.add_argument("--walks", type=int, default=6, help="walks per regime")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    children = np.random.SeedSequence(args.seed).spawn(len(REGIMES))
    i = np.arange(1, args.n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
    for ax, (p, title, note), ss in zip(axes, REGIMES, children):
        rng = np.random.default_rng(ss)
        paths = [walk(p, args.n, rng) for _ in range(args.walks)]
        env = 2.0 * np.sqrt(second_moment(p, args.n))
        ax.fill_between(i[::THIN], -env[::THIN], env[::THIN], color=REF, alpha=0.12, linewidth=0)
        ax.plot(i[::THIN], env[::THIN], color=REF, lw=1, ls="--")
        ax.plot(i[::THIN], -env[::THIN], color=REF, lw=1, ls="--")
        for s in paths[1:]:
            ax.plot(i[::THIN], s[::THIN], color=PATH, lw=0.8, alpha=0.55)
        s0 = paths[0]
        ax.plot(i[::THIN], s0[::THIN], color=HIGHLIGHT, lw=1.2)
        # the observable: |S_n| on the highlighted walk, drawn from the axis to S_n
        ax.annotate("", xy=(args.n, s0[-1]), xytext=(args.n, 0),
                    arrowprops=dict(arrowstyle="<->", color=HIGHLIGHT, lw=1.2,
                                    shrinkA=0, shrinkB=0))
        ax.text(0.97 * args.n, s0[-1] / 2, r"$Y_i = |S_i|$", color="black",
                ha="right", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(0.03, 0.96, note, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("step $i$")
        lim = 1.15 * max(env[-1], max(np.abs(s).max() for s in paths))
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, args.n * 1.02)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("position $S_i$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"erw_paths.{ext}", dpi=300)
    print(f"wrote {args.out}/erw_paths.pdf/.png  (seed {args.seed}, n {args.n}, "
          f"{args.walks} walks per regime)")


if __name__ == "__main__":
    main()
