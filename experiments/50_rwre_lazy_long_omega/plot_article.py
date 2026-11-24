"""Figures of the article's RWRE appendix (user, 2026-11-04).

Nothing is drawn; everything is read from data already on disk.

  - rwre_sweep.pdf: (a) log Ybar_i against log i for 19's `n4e6` sweep (p = 0, 0.05,
    ..., 0.5; n = 800,000 x 5 at every scale), (b) its local slopes
    s_k = log2(Ybar_{2i}/Ybar_i) (± CLT se), with the local slopes of this experiment's
    two long runs (p = 0.04, 0.3, 4..2^23, `local_slope.json`) on top, and a band over
    the pairs between Avena & Thomann's times 2^13..2^16. Slopes are taken
    within a draw, never across draws (ground rule 2).
  - rwre_p<p>_pipeline_{pilot,longrun,checks}.pdf: copies of `46/plot_pipeline.py`'s
    figures of the two long runs.

    python3 experiments/50_rwre_lazy_long_omega/plot_article.py --out ../article_writting/images
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "experiments/19_rwre_lazy_omega/data"
LONG = ROOT / "experiments/50_rwre_lazy_long_omega/data"
SWEEP_PS = ("0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5")
LONG_PS = ("0.04", "0.3")
PIPELINE = ("pilot", "longrun", "checks")


def sweep_ladder(p: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scales, log Ybar and its se (over all n x R samples) of 19's `n4e6` study at p."""
    pilot = json.loads((SWEEP / f"omega_p{p}_n4e6" / "pilot.json").read_text())
    return (np.asarray(pilot["scales"], float), np.log(pilot["y_bar"]),
            np.asarray(pilot["sigma_log"], float))


def long_slopes(p: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower scale, s_k and its CLT se of the long run at p, from `local_slope.json`."""
    a = json.loads((LONG / f"long_p{p}_24h" / "local_slope.json").read_text())["analysis"]
    return np.asarray(a["scales"][:-1], float), np.asarray(a["s"]), np.asarray(a["se_s_clt"])


def plot_sweep(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    rc = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
          "legend.frameon": False, "axes.grid": True, "grid.color": "#e6e6e6"}
    with plt.rc_context(rc):
        fig, (ax, axs) = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
        for j, p in enumerate(SWEEP_PS):
            c = cmap(j / (len(SWEEP_PS) - 1) * 0.92)
            i, ly, sl = sweep_ladder(p)
            ax.plot(i, np.exp(ly), "o-", color=c, ms=2.5, lw=0.8, label=f"$p={p}$")
            s = np.diff(ly) / np.log(2)
            se = np.hypot(sl[:-1], sl[1:]) / np.log(2)
            axs.errorbar(i[:-1], s, yerr=se, fmt="o-", color=c, ms=2.5, lw=0.8,
                         elinewidth=0.8)
        ii = np.array([4, 2**20])
        ax.plot(ii, 1.6 * np.sqrt(ii / 4), color="#7a7a7a", lw=1, ls="--",
                label=r"$\propto i^{1/2}$")
        for p, mk in zip(LONG_PS, ("s", "D")):
            k, s, se = long_slopes(p)
            axs.errorbar(k, s, yerr=se, fmt=mk, color="black", mfc="white", ms=3.5,
                         elinewidth=0.8, capsize=0, lw=0, label=f"long run, $p={p}$")
        axs.axhline(0.5, color="#7a7a7a", lw=1, ls="--")
        # Avena & Thomann (2012) simulated n = 2^13..2^16. Slopes sit at the lower scale
        # of their pair, so the pairs between those times are at 2^13..2^15.
        axs.axvspan(2**12.75, 2**15.25, color="#e8e4d8", zorder=0, lw=0,
                    label="Avena–Thomann's times")
        ax.set(xscale="log", yscale="log", xlabel="scale $i$", ylabel=r"$\overline{Y}_i$")
        ax.set_xscale("log", base=2)
        axs.set_xscale("log", base=2)
        axs.set(xlabel="lower scale $i$ of the pair", ylabel="$s_k$")
        ax.set_title("(a) log-log plot", fontsize=9, loc="left")
        axs.set_title(r"(b) local slopes $s_k$", fontsize=9, loc="left")
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")
        axs.legend(fontsize=7, loc="upper right")
        fig.savefig(out / "rwre_sweep.pdf")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="directory to write the PDFs to")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_sweep(args.out)
    for p in LONG_PS:
        for name in PIPELINE:
            shutil.copyfile(LONG / f"long_p{p}_24h" / f"pipeline_{name}.pdf",
                            args.out / f"rwre_p{p}_pipeline_{name}.pdf")
    print(f"wrote rwre_sweep.pdf and {len(LONG_PS) * len(PIPELINE)} pipeline copies to {args.out}")


if __name__ == "__main__":
    main()
