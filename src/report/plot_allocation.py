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
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports
sys.path.insert(0, str(ROOT / "src" / "budget"))   # allocation_experiment lives in the budget layer

from artifacts import read_artifact  # noqa: E402
from coverage import combine_se, consistency_threshold  # noqa: E402
from allocation_experiment import rate_exponent, rate_exponent_se, summarize  # noqa: E402
from allocation_table import (  # noqa: E402
    discover_groups,
    measured_correction,
    measured_cost_exponent,
    predicted_rate,
)

# dataviz palette, identical to tools/loglog_plot.py's (categorical slots 1-3,
# fixed order, never cycled; muted ink for reference marks). The right panel
# has exactly two series, so it uses these directly.
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_SERIES = (_BLUE, _ORANGE, _AQUA)
_INK, _MUTED, _GRIDLINE = "#0b0b0b", "#898781", "#e1e0d9"


def _budget_colors(n: int):
    """Colors for the left panel's budget curves.

    Budget is an ORDERED variable, so it takes a sequential ramp rather than
    categorical slots. That matters once a sweep has more than three budgets:
    cycling a 3-color categorical palette gave B=1e4 and B=1e7 the same blue,
    which is not a readability nitpick but an outright misreading -- the eye
    groups them as one series. A perceptually-uniform ramp also encodes the
    ordering itself, so "darker = more budget" is legible without the legend.
    """
    if n <= len(_SERIES):
        return list(_SERIES[:n])
    import matplotlib.cm as cm

    # Trim the extremes: the lightest end is invisible on white, the darkest
    # is hard to tell from the ink used for reference marks.
    return [cm.viridis(x) for x in np.linspace(0.88, 0.12, n)]


def _resolve_rate(expected, result) -> dict:
    """The predicted decay exponent, however much the caller supplied.

    Accepts a fully-built `predicted_rate` dict, or just omega_1 (and
    optionally d) to compute one from, or nothing at all -- in which case the
    recipe's own values are used. Computing it here rather than demanding it
    pre-built keeps `plot_allocation` callable with a bare
    {"omega1": ...} dict, which is what every caller that is not the CLI
    actually has.
    """
    if expected and "predicted_rate" in expected:
        return expected["predicted_rate"]
    if expected and "omega1" in expected:
        return predicted_rate(
            expected["omega1"], expected.get("d", result["d"]),
            omega1_se=expected.get("omega1_se"), d_se=expected.get("d_se"))
    return predicted_rate(result["omega1"], result["d"])


def _budget_series(result, summary, estimator="closed_form", warn=True):
    """(budgets, rmse at prop:opt's m0, rmse at the empirical argmin).

    A budget is dropped when prop:opt's own m0 was not among the swept
    `m0_values` -- there is then nothing to compare against at that budget.
    That is easy to cause by accident (sweeping m0 = 2..4 while prop:opt wants
    5) and would otherwise render an empty right-hand panel with no
    explanation, so the dropped budgets are named.
    """
    bs, at_opt, at_best, at_tuned, missing = [], [], [], [], []
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
        at_tuned.append(s.get("rmse_at_tuned_m0"))
    if warn and missing:
        which = ", ".join(f"B={B:.0e} wants m0={m0}" for B, m0 in missing)
        print(f"[plot_allocation] {len(missing)} budget(s) omitted from the decay-rate "
              f"panel: prop:opt's m0 was not swept ({which}); "
              f"m0_values={result['m0_values']}", file=sys.stderr)
    # The tuned arm is all-or-nothing: a partial series would fit a slope over
    # a different budget span than the other two, which is not comparable.
    tuned = (np.array(at_tuned, dtype=float)
             if at_tuned and all(v is not None for v in at_tuned) else None)
    return np.array(bs), np.array(at_opt), np.array(at_best), tuned


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
    palette = _budget_colors(len(result["budgets"]))
    for j, B in enumerate(result["budgets"]):
        rows = sorted((c for c in result["cells"]
                       if c["budget"] == B and not c["skipped"]),
                      key=lambda c: c["m0"])
        if not rows:
            continue
        colour = palette[j]
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
        # The paired arm: same rule with its dropped constant restored. Drawn
        # open so it reads as a prediction sitting on the measured curve rather
        # than as a third measurement.
        if s.get("tuned_m0") is not None:
            at_tuned = next((c for c in rows if c["m0"] == s["tuned_m0"]), None)
            if at_tuned is not None:
                axL.plot([at_tuned["m0"]], [at_tuned[estimator]["rmse"]],
                         marker="s", markersize=10, markerfacecolor="none",
                         color=colour, markeredgewidth=2.0, zorder=5)

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
    if any(summary.get(str(B), {}).get("tuned_m0") is not None
           for B in result["budgets"]):
        handles.append(plt.Line2D([], [], marker="s", linestyle="none",
                                  markerfacecolor="none", color=_MUTED,
                                  markeredgewidth=2.0, markersize=10))
        labels.append(r"$m_0$ from prop:opt + tuned constant")
    axL.legend(handles, labels, fontsize=9, frameon=False)

    # ---- RIGHT: error-decay rate ----------------------------------------
    axR = axes[1]
    bs, at_opt, at_best, at_tuned = _budget_series(result, summary, estimator)
    fitted = {}
    if len(bs) >= 2:
        # The fitted slope has its own uncertainty, from the noise in each
        # RMSE. Without it the measured exponent looks exact and any gap to
        # the prediction is overstated -- which is exactly how a consistent
        # result first read as a 3-sigma discrepancy here.
        slope_se = rate_exponent_se(bs, result["replicates"]) if len(bs) >= 3 else None
        series = [("at prop:opt's $m_0$", at_opt, _ORANGE),
                  ("at the best $m_0$", at_best, _BLUE)]
        if at_tuned is not None:
            series.insert(1, ("at the tuned $m_0$", at_tuned, _AQUA))
        for label, ys, colour in series:
            slope = rate_exponent(bs, ys) if len(bs) >= 3 else float("nan")
            fitted[label] = slope
            axR.plot(bs, ys, marker="o", markersize=8, linewidth=2, color=colour,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=3,
                     label=f"{label} — fitted {slope:+.3f}" + (f"$\\pm{slope_se:.3f}$" if slope_se else ""))

        # Expected exponent, from Experiment B's omega_1 (not from this run).
        pr = _resolve_rate(expected, result)
        theory, th_se = pr["theta"], pr["se"]
        anchor = at_best[0] * 1.9
        ref = anchor * (bs / bs[0]) ** theory
        src = (f"$\\omega_1={pr['omega1']:.3f}$, $d={pr['d']:.3f}$")
        axR.plot(bs, ref, "--", color=_MUTED, linewidth=1.8, zorder=2,
                 label=f"predicted {theory:+.3f}"
                       + (f"$\\pm{th_se:.3f}$" if th_se else "") + f"  ({src})")
        if th_se:
            axR.fill_between(bs, anchor * (bs / bs[0]) ** (theory - th_se),
                             anchor * (bs / bs[0]) ** (theory + th_se),
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

    result = read_artifact(args.data, "allocation_sweep")

    expected = None
    if not args.no_expected:
        groups = discover_groups(args.data.parent)
        groups = [g for g in groups if args.group is None or g["name"] == args.group]
        if groups:
            expected = measured_correction(groups[0]["runs"])
            d_hat, d_se, d_src = measured_cost_exponent(args.data.parent)
            expected["d"], expected["d_se"] = d_hat, d_se
            expected["predicted_rate"] = predicted_rate(
                expected["omega1"], d_hat,
                omega1_se=expected["omega1_se"], d_se=d_se)
            print(f"omega_1 from run group {groups[0]['name']!r}: "
                  f"{expected['omega1']:.4f}"
                  + (f" +/- {expected['omega1_se']:.4f}" if expected["omega1_se"] else "")
                  + f"  ({expected['provenance']})")
            print(f"d       : {d_hat:.4f}"
                  + (f" +/- {d_se:.4f}" if d_se else "") + f"  ({d_src})")

    fig, _, fitted = plot_allocation(result, expected, args.estimator)
    out = args.out or (args.data / "plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pr = _resolve_rate(expected, result)
    expected_reps = (expected or {}).get("replicates")
    theory, th_se = pr["theta"], pr["se"]
    contrib = (", ".join(f"{k}: {v:.5f}" for k, v in pr["contributions"].items())
               or "no stderrs available -- run replicates")

    slope_se = rate_exponent_se(result["budgets"], result["replicates"])
    print(f"\npredicted -omega1/(d+2*omega1) = {theory:+.4f}"
          + (f" +/- {th_se:.4f}" if th_se else ""))
    print(f"  error budget of the prediction ({contrib})")
    print(f"measured slope se = {slope_se:.4f}  "
          f"(RMSE over R={result['replicates']} draws carries ~1/sqrt(2R) "
          f"relative sd, across {len(result['budgets'])} budgets)\n")
    # The prediction's error comes from omega_1 (and d), each measured from a
    # handful of replicates, so the combined statistic is not standard normal
    # and the familiar |z| < 2 is the wrong cut-off. Welch-Satterthwaite gives
    # the effective dof; at R = 5 replicates a "2.5 sigma discrepancy" is in
    # fact consistent. Measured in src/check_coverage.py -- normal-quantile
    # intervals at R = 5 cover 88%, not 95%.
    comb, dof_eff = combine_se([(slope_se, None),
                                (th_se, (expected_reps - 1) if expected_reps else None)])
    crit = consistency_threshold(dof_eff)
    print(f"  consistency cut-off |z| < {crit:.3f} "
          f"(t at {dof_eff:.1f} effective dof, not the normal's 1.960)")
    for label, slope in fitted.items():
        delta = slope - theory
        z = delta / comb if comb else float("nan")
        print(f"  {label:<22} {slope:+.4f} +/- {slope_se:.4f}   "
              f"delta {delta:+.4f} +/- {comb:.4f}   z = {z:+.2f}"
              + ("   consistent" if abs(z) < crit else "   DISCREPANT"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    _main()
