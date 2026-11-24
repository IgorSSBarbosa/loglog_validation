"""The article's Step 5 figure: the ranges eq. (aic) weighs, and the average over them.

Recomputes the joint refit of `src/study/report.py` (`joint_refit`, user 2026-11-11:
the pilot and the long run in one set of log-log points, every range of
>= JOINT_MIN_POINTS consecutive scales, gamma free, scored by AIC = chi2 + 2*4 +
2*N_cut) from the study's pilot.json and final.json, checks that it picks the
range and omega1 stored in answer.json, and draws:

(a) the correction log Y_bar_i - gamma_r log i (gamma_r the best range's own slope)
    with its error bars, pilot and long-run points marked apart, the fit on the
    best-AIC range (solid on it, dashed beyond) and the fit on all scales (dotted),
    with their chi2/dof, and the AIC-weighted omega1 the interval uses;
(b) every candidate range as a bar over the scales it keeps, sorted by AIC, shaded
    by its weight exp(-AIC/2) normalised over the ranges, with its AIC, cut count
    and omega1; the best-AIC range is outlined, and ranges left out of the
    average (omega1 on a bound, or an open profile interval) are hatched.

Reads a finished study directory only; truths appear nowhere (ground rule 4).

CLI:
    python3 experiments/46_pipeline_erwurn_gpu/plot_aic.py --study DIR [--out DIR]

Writes pipeline_aic.pdf/.png into --out (default the study directory).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src" / "study"))   # report.py sets up tools/ itself

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import report  # noqa: E402

DATA = "#2a6fdb"
FIT = "#e8622c"
REF = "#7a7a7a"
SHOWN = 25


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.study
    out.mkdir(parents=True, exist_ok=True)

    pilot = json.loads((args.study / "pilot.json").read_text())
    answer = json.loads((args.study / "answer.json").read_text())
    scales, logy, sig, src = report.joint_points(pilot, answer["scales"], answer["y_bar"],
                                                 answer["sigma_log"])
    rf = report.omega_ranges(scales, logy, sig, np.e,
                             min_points=report.JOINT_MIN_POINTS, gamma_pin=None)
    best = rf["best"]
    stored = answer["joint_refit"]
    # the figure must show the refit the interval used, not a re-derivation of it
    assert (best["i_min"], best["i_max"]) == (stored["i_min"], stored["i_max"]), (best, stored)
    assert math.isclose(rf["omega"], stored["omega1"], rel_tol=1e-9), (rf["omega"], stored)

    fits = rf["fits"]
    full = [f for f in fits if f["i_min"] == scales[0] and f["i_max"] == scales[-1]]
    N = len(scales)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                               gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) the correction. omega_ranges counts k from each range's own first scale,
    # natural-log base: log y = a + b log(i/i_min) + c (i/i_min)^-omega, b = gamma pinned.
    slope = best["b"]
    r = logy - slope * np.log(scales)
    keep = (scales >= best["i_min"]) & (scales <= best["i_max"])
    for name, mk, sel in (("pilot", "o", src != "run"), ("long run", "s", src == "run")):
        for on_, col, mfc, tag in ((keep, DATA, None, "kept"), (~keep, REF, "white", "cut")):
            j = sel & on_
            if j.any():
                a.errorbar(scales[j], r[j], yerr=sig[j], fmt=mk, ms=5, color=col, mfc=mfc,
                           capsize=2, lw=1, label=f"{name}, {tag}")
    grid = np.geomspace(scales[0], scales[-1], 300)

    def model(f: dict, i: np.ndarray) -> np.ndarray:
        x = i / f["i_min"]
        return f["a"] + f["b"] * np.log(x) + f["c"] * x ** (-f["omega"]) - slope * np.log(i)

    on = (grid >= best["i_min"]) & (grid <= best["i_max"])
    a.plot(grid[on], model(best, grid[on]), color=FIT, lw=2,
           label=(rf"best AIC ({rf['best_weight']:.2f}): $\hat\omega_1 = {best['omega']:.2f}$, "
                  rf"$\chi^2 = {best['chi2']:.1f}/{best['dof']}$"))
    a.plot(grid[~on], model(best, grid[~on]), color=FIT, lw=1.2, ls="--")
    if full:
        f = full[0]
        a.plot(grid, model(f, grid), color=REF, lw=1.2, ls=":",
               label=(rf"all scales: $\hat\omega_1 = {f['omega']:.2f}$, "
                      rf"$\chi^2 = {f['chi2']:.1f}/{f['dof']}$"))
    a.text(0.03, 0.97, rf"AIC-weighted $\hat\omega_1 = {rf['omega']:.2f} \pm {rf['total']:.2f}$",
           transform=a.transAxes, fontsize=8.5, va="top")
    a.set_xscale("log", base=2)
    a.set_xlabel("scale $i$")
    a.set_ylabel(r"$\log\overline{Y}_i - \gamma_r\log i$")
    a.set_title("(a) the correction and the refit", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower right")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # (b) the candidate ranges, best AIC on top. The union of pilot and long-run scales
    # gives ~80 ranges, unreadable as rows, so only the SHOWN lowest-AIC ones are drawn.
    order = sorted(fits, key=lambda f: f["aic"])[:SHOWN]
    wmax = max(f["weight"] for f in fits)
    lk = np.log2(scales)
    for row, f in enumerate(order):
        y = len(order) - 1 - row
        x0, x1 = math.log2(f["i_min"]) - 0.4, math.log2(f["i_max"]) + 0.4
        eligible = not (f["interval_open"] or f["omega_on_bound"])
        shade = 0.12 + 0.88 * f["weight"] / wmax
        b.add_patch(Rectangle((x0, y - 0.38), x1 - x0, 0.76,
                              facecolor=DATA if eligible else "white",
                              alpha=shade if eligible else 1.0,
                              hatch=None if eligible else "////",
                              edgecolor=REF if not eligible else "none", lw=0.5))
        if f is best or (f["i_min"], f["i_max"]) == (best["i_min"], best["i_max"]):
            b.add_patch(Rectangle((x0, y - 0.38), x1 - x0, 0.76, facecolor="none",
                                  edgecolor=FIT, lw=2))
        b.text(lk[-1] + 0.9, y, f"{f['aic']:5.1f} {N - f['points']:3d} {f['omega']:5.2f}",
               fontsize=6.5, va="center", family="monospace")
    b.text(lk[-1] + 0.9, len(order) + 0.2, "  AIC cut   ω₁", fontsize=7, va="bottom",
           family="monospace")
    b.set_xlim(lk[0] - 0.6, lk[-1] + 5.0)
    b.set_ylim(-0.8, len(order) + 1.2)
    b.set_xticks(lk)
    b.set_xticklabels([f"$2^{{{int(round(k))}}}$" for k in lk], fontsize=6)
    b.set_yticks([])
    b.set_xlabel("scales kept")
    b.set_title(rf"(b) the {len(order)} best of {len(fits)} ranges, by AIC $= \chi^2 + 2\cdot4 + 2\,(N-K)$",
                fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=DATA, label=r"weight $\propto e^{-\mathrm{AIC}/2}$"),
               Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=FIT, lw=2, label="best AIC"),
]
    if any(f["interval_open"] or f["omega_on_bound"] for f in order):
        handles.append(Rectangle((0, 0), 1, 1, facecolor="white", hatch="////", edgecolor=REF,
                                 lw=0.5, label=r"$\hat\omega_1$ not determined"))
    b.legend(handles=handles, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.13),
             ncol=3, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pipeline_aic.{ext}", dpi=200)
    print(f"wrote {out}/pipeline_aic.pdf/.png  ({len(fits)} ranges; best AIC "
          f"{best['i_min']}..{best['i_max']} ({rf['best_weight']:.3f} of the kept weight, "
          f"omega1 {best['omega']:.4f}); AIC-weighted omega1 {rf['omega']:.4f} "
          f"+- {rf['total']:.4f}")


if __name__ == "__main__":
    main()
