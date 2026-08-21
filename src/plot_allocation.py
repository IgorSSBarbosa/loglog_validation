"""Plot Experiment C: the bias-variance tradeoff in m0, and the error-decay rate.

Reads an allocation_experiment.py run directory and writes <run_dir>/plot.png
(same one-run-one-folder convention as plot_loglog.py / plot_cost.py --
generation and plotting are always separate scripts here).

Two panels, answering the experiment's two questions:

  LEFT  -- RMSE(gamma_hat) against m0, one curve per budget. The U-shape IS the
           bias-variance tradeoff prop:opt reasons about: small m0 keeps
           correction-contaminated scales (bias), large m0 buys fewer samples
           (variance). prop:opt's chosen m0 and the empirical argmin are marked
           on every curve, so the systematic gap between them is visible rather
           than tabulated.

  RIGHT -- RMSE against budget on log-log, the error-decay law. Fitted slopes
           are compared with the exponent -omega1/(d + 2*omega1) PREDICTED FROM
           EXPERIMENT B's measured omega_1, drawn as a band from that
           measurement's own standard error. This is the cross-experiment
           check: B's omega_1 predicts C's decay rate.

Colors are tools/loglog_plot.py's palette unchanged -- the dataviz skill's
validated categorical slots 1-3, plus muted ink for references. Reusing an
already-validated palette keeps every chart in this repo consistent.

CLI:
    python3 plot_allocation.py -data ../experiments/01_srw/data/allocation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # helper modules live there, as bare imports
sys.path.insert(0, str(HERE))

from allocation_experiment import rate_exponent, summarize  # noqa: E402
from allocation_table import discover_groups, measured_correction  # noqa: E402

# dataviz palette, identical to tools/loglog_plot.py's (categorical slots 1-3,
# fixed order, never cycled; muted ink for reference marks).
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_SERIES = (_BLUE, _ORANGE, _AQUA)
_INK, _MUTED, _GRIDLINE = "#0b0b0b", "#898781", "#e1e0d9"


def _budget_series(result, summary, estimator="closed_form", warn=True):
    """(budgets, rmse at prop:opt's m0, rmse at the empirical argmin).

    A budget is dropped when prop:opt's own m0 was not among the swept
    `m0_values` -- there is then nothing to compare against at that budget.
    That is easy to cause by accident (sweeping m0 = 2..4 while prop:opt wants
    5) and would otherwise render an empty right-hand panel with no
    explanation, so the dropped budgets are named.
    """
    bs, at_opt, at_best, missing = [], [], [], []
    for B in result["budgets"]:
        s = summary.get(str(B))
        if not s:
            continue
        if s["rmse_at_prop_opt_m0"] is None:
            missing.append((B, s["prop_opt_m0"]))
            continue
        bs.append(B)
        at_opt.append(s["rmse_at_prop_opt_m0"])
        at_best.append(s["best_rmse"])
    if warn and missing:
        which = ", ".join(f"B={B:.0e} wants m0={m0}" for B, m0 in missing)
        print(f"[plot_allocation] {len(missing)} budget(s) omitted from the decay-rate "
              f"panel: prop:opt's m0 was not swept ({which}); "
              f"m0_values={result['m0_values']}", file=sys.stderr)
    return np.array(bs), np.array(at_opt), np.array(at_best)


def plot_allocation(result: dict, expected: dict | None = None,
                    estimator: str = "closed_form", ax=None):
    """Draw both panels. `expected` is measured_correction()'s dict, or None."""
    summary = summarize(result, estimator)
    d, omega1_recipe = result["d"], result["omega1"]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    else:
        fig, axes = ax[0].figure, ax

    # ---- LEFT: RMSE vs m0, one curve per budget -------------------------
    axL = axes[0]
    for j, B in enumerate(result["budgets"]):
        rows = sorted((c for c in result["cells"]
                       if c["budget"] == B and not c["skipped"]),
                      key=lambda c: c["m0"])
        if not rows:
            continue
        colour = _SERIES[j % len(_SERIES)]
        m0s = [c["m0"] for c in rows]
        rmse = [c[estimator]["rmse"] for c in rows]
        axL.plot(m0s, rmse, "-", color=colour, linewidth=2, zorder=2,
                 label=f"$B=10^{{{int(round(np.log10(B)))}}}$")

        s = summary[str(B)]
        best = next(c for c in rows if c["m0"] == s["best_m0"])
        axL.plot([best["m0"]], [best[estimator]["rmse"]], marker="o", markersize=9,
                 color=colour, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        at_opt = next((c for c in rows if c["m0"] == s["prop_opt_m0"]), None)
        if at_opt is not None:
            axL.plot([at_opt["m0"]], [at_opt[estimator]["rmse"]], marker="X", markersize=11,
                     color=colour, markeredgecolor="white", markeredgewidth=1.5, zorder=4)

    axL.set_yscale("log")
    axL.set_xlabel(r"scale offset $m_0$")
    axL.set_ylabel(r"RMSE of $\hat\gamma$")
    axL.set_title("Error vs. where the scale window starts", loc="left", fontsize=11)
    axL.grid(True, which="both", ls="-", lw=0.6, color=_GRIDLINE, alpha=0.8)
    axL.set_axisbelow(True)
    # Marker meanings go in the legend, never color alone.
    handles, labels = axL.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], marker="o", linestyle="none", color=_MUTED,
                   markeredgecolor="white", markersize=9),
        plt.Line2D([], [], marker="X", linestyle="none", color=_MUTED,
                   markeredgecolor="white", markersize=11),
    ]
    labels += ["best $m_0$ (measured)", r"$m_0$ from prop:opt"]
    axL.legend(handles, labels, fontsize=9, frameon=False)

    # ---- RIGHT: error-decay rate ----------------------------------------
    axR = axes[1]
    bs, at_opt, at_best = _budget_series(result, summary, estimator)
    fitted = {}
    if len(bs) >= 2:
        for label, ys, colour in (("at prop:opt's $m_0$", at_opt, _ORANGE),
                                  ("at the best $m_0$", at_best, _BLUE)):
            slope = rate_exponent(bs, ys) if len(bs) >= 3 else float("nan")
            fitted[label] = slope
            axR.plot(bs, ys, marker="o", markersize=8, linewidth=2, color=colour,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=3,
                     label=f"{label} — fitted {slope:+.3f}")

        # Expected exponent, from Experiment B's omega_1 (not from this run).
        om = expected["omega1"] if expected else omega1_recipe
        om_se = (expected or {}).get("omega1_se")
        theory = -om / (d + 2 * om)
        anchor = at_best[0] * 1.9
        ref = anchor * (bs / bs[0]) ** theory
        src = (f"Exp B: $\\omega_1={om:.3f}$" + (f"$\\pm{om_se:.3f}$" if om_se else "")
               if expected else f"recipe: $\\omega_1={om:.3f}$")
        axR.plot(bs, ref, "--", color=_MUTED, linewidth=1.8, zorder=2,
                 label=f"expected {theory:+.3f} ({src})")
        if om_se:
            lo = -(om - om_se) / (d + 2 * (om - om_se))
            hi = -(om + om_se) / (d + 2 * (om + om_se))
            axR.fill_between(bs, anchor * (bs / bs[0]) ** min(lo, hi),
                             anchor * (bs / bs[0]) ** max(lo, hi),
                             color=_MUTED, alpha=0.15, zorder=1, linewidth=0)

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("budget $B$ (simulated steps)")
    axR.set_ylabel(r"RMSE of $\hat\gamma$")
    axR.set_title("Error-decay rate: measured vs. predicted from Experiment B",
                  loc="left", fontsize=11)
    axR.grid(True, which="both", ls="-", lw=0.6, color=_GRIDLINE, alpha=0.8)
    axR.set_axisbelow(True)
    axR.legend(fontsize=9, frameon=False, loc="upper right")

    fig.tight_layout()
    return fig, axes, fitted


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-data", "--data", dest="data", required=True, type=Path,
                        help="allocation_experiment.py run directory (holds result.json)")
    parser.add_argument("--estimator", default="closed_form",
                        choices=("closed_form", "all_points"))
    parser.add_argument("--group", default=None,
                        help="Experiment B run group supplying the expected omega_1")
    parser.add_argument("--no-expected", action="store_true",
                        help="use the recipe's omega_1 instead of Experiment B's measurement")
    parser.add_argument("-o", "--out", dest="out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = json.loads((args.data / "result.json").read_text())

    expected = None
    if not args.no_expected:
        groups = discover_groups(args.data.parent)
        groups = [g for g in groups if args.group is None or g["name"] == args.group]
        if groups:
            expected = measured_correction(groups[0]["runs"])
            print(f"expected omega_1 from run group {groups[0]['name']!r}: "
                  f"{expected['omega1']:.4f}"
                  + (f" +/- {expected['omega1_se']:.4f}" if expected["omega1_se"] else "")
                  + f"  ({expected['provenance']})")

    fig, _, fitted = plot_allocation(result, expected, args.estimator)
    out = args.out or (args.data / "plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    om = expected["omega1"] if expected else result["omega1"]
    theory = -om / (result["d"] + 2 * om)
    print(f"\nerror-decay exponent, measured vs expected ({theory:+.4f}):")
    for label, slope in fitted.items():
        print(f"  {label:<22} {slope:+.4f}   delta {slope - theory:+.4f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    _main()
