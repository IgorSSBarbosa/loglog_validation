"""Log-log figures for experiments 44 (Avena & Thomann RWRE) and 45 (ERW practice).

protocol_arm.py keeps only per-size means and variances, so src/report/plot_loglog.py
(which needs samples.npz) cannot draw these. Each replicate draws the same n per
time, so pooling the 500 replicates' means is the mean of every walk drawn:

    .venv/bin/python experiments/plot_lit_walks.py [--root <checkout holding data/>]

Writes experiments/44_lit_avena_thomann2012_rwre/images/rwre_loglog.png and
experiments/45_lit_practice_erw/images/erw_loglog.png. Exact exponents come from
each artifact's true_gamma, used only for the reference lines (ground rule 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loglog_plot import _AQUA, _BLUE, _CRITICAL, _GRIDLINE, _MUTED, _ORANGE  # noqa: E402

#: (experiment, figure name, ylabel, [(artifact tag, label, colour)]). ERW colours follow experiments/plot_walk_section.py so a p keeps its colour.
#: When every series has an exact exponent, (b) plots slope minus it: ERW's p = 0.9
#: (0.8) would otherwise squash the three gamma = 1/2 series into one line.
FIGURES = [
    ("44_lit_avena_thomann2012_rwre", "rwre_loglog.png", r"$\bar S_n$ (SD of the position)",
     [("avena_thomann_p0.5", "$p=1/2$", _BLUE),
      ("avena_thomann_p0.333", "$p=1/3$", _ORANGE),
      ("avena_thomann_p0", "$p=0$", _AQUA)]),
    ("45_lit_practice_erw", "erw_loglog.png", r"$\overline{|X_n|}$",
     [("practice_p0.25", "$p=0.25$", _AQUA),
      ("practice_p0.6", "$p=0.6$", _BLUE),
      ("practice_p0.75", "$p=0.75$", _ORANGE),
      ("practice_p0.9", "$p=0.9$", _CRITICAL)]),
]


def pooled(art: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Times, pooled mean, se of its log, and the mean all-times regression estimate."""
    reps = art["per_replicate"]
    n = np.array(art["n"], float)
    y = np.array([r["y_bar"] for r in reps])
    var = np.array([r["var"] for r in reps])
    y_pool = y.mean(axis=0)
    sl = np.sqrt(var.mean(axis=0) / (len(reps) * n)) / y_pool
    return np.array(art["scales"], float), y_pool, sl, float(np.mean([r["full"]["gamma"] for r in reps]))


def figure(data_root: Path, exp: str, name: str, ylabel: str, series: list) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2))
    arts = [json.loads((data_root / exp / "data" / tag / "protocol_arm.json").read_text()) for tag, _, _ in series]
    centred = all(a.get("true_gamma") is not None for a in arts)
    truth_drawn = False
    for art, (_, label, colour) in zip(arts, series):
        shift = art["true_gamma"] if centred else 0.0
        L, y, sl, g_mean = pooled(art)
        R = len(art["per_replicate"])
        g, c = np.polyfit(np.log(L), np.log(y), 1)
        ax.plot(L, y, "o", ms=7, color=colour, mec="white", mew=1.5, zorder=3)
        ax.plot(L, np.exp(c) * L ** g, lw=2, color=colour, label=rf"{label}: slope {g:.4f}")

        mid = np.sqrt(L[1:] * L[:-1])
        slope = np.diff(np.log(y)) / np.diff(np.log(L))
        err = 1.96 * np.hypot(sl[1:], sl[:-1]) / np.diff(np.log(L))
        bx.errorbar(mid, slope - shift, yerr=err, fmt="o-", ms=7, lw=1, color=colour, mec="white", mew=1.5,
                    capsize=0, zorder=3, label=rf"{label}: mean $\hat\gamma$ over {R} = {g_mean:.4f}")
        bx.hlines(g_mean - shift, L[0], L[-1], lw=2, color=colour, alpha=0.45)
        if art.get("true_gamma") is not None:
            bx.hlines(art["true_gamma"] - shift, L[0], L[-1], lw=1.5, ls="--", color=_MUTED,
                      label=None if truth_drawn else "exact exponent")
            truth_drawn = True

    ax.set(xscale="log", yscale="log", xlabel="time $n$", ylabel=ylabel,
           title=f"(a) the log-log plot, means pooled over {R} replicates")
    bx.set(xscale="log", xlabel="time $n$", ylabel="slope minus exact exponent" if centred else "slope",
           title="(b) local slopes; shaded bar = mean regression estimate")
    for a_ in (ax, bx):
        a_.set_xticks(L, [f"$2^{{{int(np.log2(v))}}}$" for v in L])
        a_.minorticks_off()
        a_.grid(color=_GRIDLINE, lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)
        a_.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    out = ROOT / "experiments" / exp / "images" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="checkout whose experiments/*/data/ holds the runs (data/ is gitignored)")
    args = parser.parse_args()
    for exp, name, ylabel, series in FIGURES:
        print(f"wrote {figure(args.root / 'experiments', exp, name, ylabel, series)}")


if __name__ == "__main__":
    main()
