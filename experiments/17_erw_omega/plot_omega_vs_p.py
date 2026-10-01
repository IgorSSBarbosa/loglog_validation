"""omega against p for a finished `omega_sweep.py` run: one figure, every estimate with its se.

Reporting only. It reads what the sweep wrote and draws the README's two hypotheses
next to it. Nothing here feeds a fit, which is why they live in this file and not in
`omega_sweep.py` (ground rule 4).

    python3 experiments/17_erw_omega/plot_omega_vs_p.py --tag n8m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
# tools/loglog_plot.py's palette (the dataviz skill's validated default, light mode).
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#e1e0d9"


def read(tag: str, data: Path) -> dict[str, np.ndarray]:
    run = json.loads((data / f"omega_{tag}.json").read_text())
    rows = []
    for lab, r in run["results"].items():
        om = json.loads((data / f"omega_p{lab}_{tag}" / "constants.json").read_text())["omega1"]
        tail = r["fits"]["P_tail"]
        w = [v for v in r["exact"]["w"] if v is not None]
        rows.append((float(lab), om["value"], om["se"], tail["omega"], tail["se_omega"], w[-1]))
    rows.sort()
    cols = np.array(rows).T
    return dict(zip(["p", "pilot", "pilot_se", "tail", "tail_se", "exact"], cols)) | {
        "tail_k": run["args"]["tail"], "exact_max": run["args"]["exact_max"]}


def plot(d: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    pp = np.linspace(0.08, 0.97, 600)
    m = np.minimum(1, np.abs(3 - 4 * pp))
    c = np.minimum(1, np.abs(3 - 4 * pp) / 2)
    # Labels sit where neither curve nor data points are: M above its plateau, C under its descent.
    for curve, ls, txt, xy, off, ha in ((m, "-", "M: min(1, |3−4p|)", (0.18, 1.0), (0, 6), "center"),
                                        (c, "--", "C: min(1, |3−4p|/2)", (0.58, 0.15), (0, -12), "left")):
        crit = np.abs(pp - 0.75) < 1e-3
        ax.plot(np.where(crit, np.nan, pp), curve, ls, color=MUTED, lw=1.4, zorder=1)
        ax.annotate(txt, xy, xytext=off, textcoords="offset points", color=INK, fontsize=8, ha=ha)
    ax.axvline(0.75, color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.text(0.752, 1.12, "p = 3/4: log correction,\nno ω to read", color=INK, fontsize=8, va="top")

    dx = 0.006
    ax.errorbar(d["p"] - dx, d["pilot"], yerr=d["pilot_se"], fmt="o", ms=6, color=BLUE, capsize=3,
                mec="white", mew=1, label="pilot.py eq. (232) fit, 4..$2^{20}$ (± replicate se; bars smaller than the dot)", zorder=3)
    ax.errorbar(d["p"] + dx, d["tail"], yerr=d["tail_se"], fmt="s", ms=6, color=ORANGE, capsize=3,
                mec="white", mew=1, label=f"(P) γ + c k^−ω on drawn s_k, k ≥ {d['tail_k']} (± se)", zorder=3)
    ax.plot(d["p"], d["exact"], "D", ms=6, color=AQUA, mec="white", mew=1, zorder=4,
            label=f"exact w_k at the top of the DP ladder (k ≤ {d['exact_max']}; no noise)")

    ax.set_xlabel("memory parameter p")
    ax.set_ylabel("correction exponent ω")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 1.6)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("erw_urn_gpu: correction exponent of E|S_k| against p", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()
    d = read(a.tag, a.data_root)
    out = a.data_root / f"omega_vs_p_{a.tag}.png"
    plot(d, out)
    print("p      pilot ω1 ± se      (P) tail ω ± se     exact w_top")
    for i in range(len(d["p"])):
        print(f"{d['p'][i]:<6g} {d['pilot'][i]:.3f} ± {d['pilot_se'][i]:.3f}      "
              f"{d['tail'][i]:.3f} ± {d['tail_se'][i]:.3f}       {d['exact'][i]:.3f}")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
