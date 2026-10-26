"""Figure G5 of plans/percolation_experiments.md: error decay under the article's
allocation, d = 2 (experiment 22) beside d = 3 (experiment 23).

Per panel: RMSE of gamma-hat against the budget B for every m0 on the grid (thin), the
empirical best m0 at each B, the tuned allocation's m0, prop:opt's m0 (eq. 946) where it
was affordable, and a line of the predicted slope -omega1/(d + 2 omega1) (eq. 941) through
the best arm's first point. Reads the two finished `allocation_sweep`s; draws nothing.

CLI:
    python3 experiments/23_perc_df_d3_error_decay/plot_allocation_arms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import read_artifact  # noqa: E402

SWEEPS = {
    "d = 2, south anchor (22)": ROOT / "experiments/22_perc_df_d2_error_decay/data/error_decay",
    "d = 3, face_far (23)": ROOT / "experiments/23_perc_df_d3_error_decay/data/error_decay",
}


def arms(result: dict) -> dict:
    """RMSE per (B, m0), and which m0 each arm picks at each B."""
    cells = [c for c in result["cells"] if not c["skipped"]]
    budgets = sorted({c["budget"] for c in cells})
    rmse = {(c["budget"], c["m0"]): c["closed_form"]["rmse"] for c in cells}
    best = {B: min((m0 for (b, m0) in rmse if b == B), key=lambda m0: rmse[(B, m0)]) for B in budgets}
    tuned = {B: result["tuned"][str(B)]["m0"] for B in budgets}
    prop = {B: result["prop_opt"][str(B)]["m0"] for B in budgets}
    return {"budgets": budgets, "rmse": rmse, "best": best, "tuned": tuned, "prop": prop,
            "m0s": sorted({m0 for (_, m0) in rmse})}


def main() -> None:
    from matplotlib.figure import Figure

    from tools.loglog_plot import _AQUA, _BLUE, _GRIDLINE, _INK, _MUTED, _ORANGE

    fig = Figure(figsize=(11, 4.4), constrained_layout=True)
    axes = fig.subplots(1, 2)
    for ax, (label, path) in zip(axes, SWEEPS.items()):
        res = read_artifact(path, "allocation_sweep")
        a = arms(res)
        B = np.array(a["budgets"])
        for m0 in a["m0s"]:
            pts = [(b, a["rmse"][(b, m0)]) for b in B if (b, m0) in a["rmse"] and np.isfinite(a["rmse"][(b, m0)])]
            if pts:
                x, y = zip(*pts)
                ax.loglog(x, y, "-", color=_GRIDLINE, lw=1.2, zorder=1)
                ax.annotate(f"m₀={m0}", (x[-1], y[-1]), fontsize=7, color=_MUTED, xytext=(3, 0), textcoords="offset points")
        best = [a["rmse"][(b, a["best"][b])] for b in B]
        tuned = [a["rmse"].get((b, a["tuned"][b]), np.nan) for b in B]
        prop = [a["rmse"].get((b, a["prop"][b]), np.nan) if a["prop"][b] is not None else np.nan for b in B]
        ax.loglog(B, best, "o-", color=_AQUA, lw=2, label="best m₀ on the grid", zorder=3)
        ax.loglog(B, tuned, "s--", color=_BLUE, lw=1.5, label="tuned allocation", zorder=4)
        if np.isfinite(prop).any():
            ax.loglog(B, prop, "^-", color=_ORANGE, lw=1.5, label="prop:opt, eq. (946)", zorder=4)
        else:
            ax.text(0.03, 0.04, "prop:opt m₀ = 10–12: not affordable at any B", transform=ax.transAxes,
                    fontsize=8, color=_ORANGE)
        slope = -res["omega1"] / (res["d"] + 2 * res["omega1"])
        ax.loglog(B, best[0] * (B / B[0]) ** slope, ":", color=_INK, lw=1.2,
                  label=f"predicted slope {slope:+.3f}, eq. (941)")
        ax.set(xlabel="budget B (cost units)", ylabel="RMSE of γ̂ (R = 20 per cell)", title=label)
        ax.grid(True, which="both", color=_GRIDLINE, lw=0.5)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    out = HERE / "data" / "allocation_arms_d2_d3.png"
    fig.savefig(out, dpi=130)
    print(f"figure = {out}")


if __name__ == "__main__":
    main()
