"""The three new figures of `experiments/percolation_section.md`.

1. `27_perc_df_d3_facefar/images/estimator_walkthrough.png`: how gamma-hat is formed
   on a real run. (a) two-point local slopes of the pilot (4..512) and of the final run
   (32..1024), with the local slope implied by the pilot's eq. (232) fit and the final
   window shaded with gamma-hat +/- the eq. (720) half-width; (b) the eq. (526) weights
   that turn the six log-means of that window into gamma-hat.
2. `25_perc_tau_d2_coverage/images/box_rule.png`: tau-hat's bias on the window
   256..8192 against the box rule's exponent, and the per-scale cv of each pilot
   (Assumption 6).
3. `images/coverage_forest.png`: eq. (720) coverage with Wilson CIs for every scored
   arm (20-23, 25, 26), and the planned runs 27, 29, 30 in units of their own bound.

Reads finished artifacts only; draws nothing. Literature values appear here for
reporting only, copied from each experiment's README (ground rule 4).

CLI:
    python3 experiments/plot_percolation_section.py [--root DIR]

`--root` is the checkout whose (gitignored) `experiments/*/data/` holds the artifacts;
it defaults to this one. Figures are always written next to this script.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coverage import wilson_score_interval  # noqa: E402
from tools.loglog import closed_form_weights, ols_slope  # noqa: E402
from tools.loglog_plot import _AQUA, _BLUE, _CRITICAL, _GRIDLINE, _INK, _MUTED, _ORANGE  # noqa: E402

# Reporting only (README "Known values" of each experiment).
DF3_LIT = 2.52293
TAU2_GAMMA = 1 - 187 / 91


def _json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def local_slopes(scales, y_bar, sigma_log):
    """Two-point slopes log(Y_2i / Y_i)/log 2 at the geometric midpoint, with 1 SE."""
    s = np.asarray(scales, dtype=float)
    ly = np.log(np.asarray(y_bar, dtype=float))
    sl = np.asarray(sigma_log, dtype=float)
    dx = np.log(s[1:] / s[:-1])
    return (np.sqrt(s[1:] * s[:-1]), (ly[1:] - ly[:-1]) / dx,
            np.sqrt(sl[1:] ** 2 + sl[:-1] ** 2) / dx)


def fig_walkthrough(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    pilot = _json(data / "27_perc_df_d3_facefar/data/pilot/pilot.json")
    ans = _json(data / "27_perc_df_d3_facefar/data/pilot/answer.json")
    fit = pilot["direct_fit"]

    fig = Figure(figsize=(11, 4.3), constrained_layout=True)
    ax, axw = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})

    lo, hi = ans["scales"][0], ans["scales"][-1]
    g, h = ans["gamma"], ans["wilson"]["half_width"]
    ax.axvspan(lo, hi, color=_GRIDLINE, alpha=0.6, lw=0, zorder=0)
    ax.fill_between([lo, hi], g - h, g + h, color=_BLUE, alpha=0.15, lw=0,
                    label=f"run: γ̂ = {g:.4f} ± {h:.4f}, eq. (720)")
    ax.plot([lo, hi], [g, g], "-", color=_BLUE, lw=1.5)

    x, y, e = local_slopes(pilot["scales"], pilot["y_bar"], pilot["sigma_log"])
    ax.errorbar(x, y, e, fmt="o", color=_ORANGE, ms=5, capsize=2, label="pilot, local slope ± 1 SE")
    xf, yf, ef = local_slopes(ans["scales"], ans["y_bar"], ans["sigma_log"])
    ax.errorbar(xf * 1.04, yf, ef, fmt="s", color=_BLUE, ms=5, capsize=2, label="run, local slope ± 1 SE")

    # Local slope between i and 2i implied by the pilot's eq. (232) fit.
    i = np.geomspace(pilot["scales"][0], ans["scales"][-1] / 2, 200)
    om = fit["omega1"]
    curve = fit["gamma"] + fit["a1"] * i ** (-om) * (2 ** (-om) - 1) / math.log(2)
    ax.plot(i * math.sqrt(2), curve, "-", color=_ORANGE, lw=1.2,
            label=f"pilot fit, eq. (232): ω₁ = {om:.2f}, a₁ = {fit['a1']:.2f}")
    ax.axhline(DF3_LIT, color=_MUTED, ls=":", lw=1.2, label=f"literature {DF3_LIT} (not used)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(2.42, 2.60)
    ax.set(xlabel="scale i (midpoint of i, 2i)", ylabel="local slope  Δlog Ȳ / Δlog i",
           title="(a) d_f, d = 3: pilot → window → γ̂")
    ax.grid(True, which="both", color=_GRIDLINE, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.text(lo * 1.05, 2.592, f"window k = {ans['m0'] + 1}..{ans['m0'] + ans['m']}  (m₀ = {ans['m0']}, m = {ans['m']})",
            fontsize=8, color=_INK)

    m, m0 = ans["m"], ans["m0"]
    w = closed_form_weights(m, m0)
    k = np.arange(m0 + 1, m0 + m + 1)
    axw.bar(k, w, color=_BLUE, width=0.6)
    axw.axhline(0, color=_INK, lw=0.6)
    for kk, ww in zip(k, w):
        axw.text(kk, ww + (0.008 if ww >= 0 else -0.008), f"{ww:+.3f}", ha="center",
                 va="bottom" if ww >= 0 else "top", fontsize=7.5)
    axw.set_xticks(k, [f"{kk}\n{2 ** kk}" for kk in k])
    axw.set(xlabel="k   (scale ρᵏ below)", ylabel="w_{k,m}, eq. (526)",
            title="(b) γ̂ = Σ w_{k,m} log Ȳ_{ρᵏ} / log ρ")
    axw.set_ylim(-0.2, 0.2)
    axw.grid(True, axis="y", color=_GRIDLINE, lw=0.5)
    fig.savefig(out, dpi=130)


def window_gamma(pilot: dict, lo: int, hi: int) -> np.ndarray:
    """Per-replicate OLS slope on scales lo..hi of a pilot (each replicate is independent)."""
    s = np.asarray(pilot["scales"])
    keep = (s >= lo) & (s <= hi)
    return np.array([ols_slope(np.log(s[keep]), np.log(np.asarray(r["y_bar"])[keep]))[0]
                     for r in pilot["per_replicate"]])


def fig_box_rule(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    base = data / "25_perc_tau_d2_coverage/data"
    arms = [(0.5, "pilot_deep", _ORANGE), (48 / 91, "pilot_box_df", _AQUA), (0.6, "pilot_cutoff", _BLUE)]
    fig = Figure(figsize=(10, 3.9), constrained_layout=True)
    ax, axc = fig.subplots(1, 2)
    for be, name, color in arms:
        p = _json(base / name / "pilot.json")
        g = window_gamma(p, 256, 8192)
        se = g.std(ddof=1) / math.sqrt(g.size)
        ax.errorbar([be], [g.mean() - TAU2_GAMMA], [1.96 * se], fmt="o", color=color, ms=7, capsize=3)
        ax.annotate(f"{g.mean() - TAU2_GAMMA:+.4f}", (be, g.mean() - TAU2_GAMMA),
                    xytext=(8, 0), textcoords="offset points", fontsize=8, va="center")
        label = {0.5: "s^(1/2)", 0.6: "s^0.6"}.get(be, "s^(48/91) = s^(1/d_f)")
        axc.plot(p["scales"], p["cv_per_scale"], "o-", color=color, label=f"box side ∝ {label}")
    ax.axhline(0, color=_MUTED, ls=":", lw=1.2)
    ax.axvline(48 / 91, color=_GRIDLINE, lw=6, zorder=0)
    ax.set(xlabel="box exponent  (box side = 16·s^exponent)",
           ylabel="mean γ̂ − (1 − τ),  window s = 256..8192",
           title="(a) bias of the window's γ̂ (= 1 − τ̂), 5 replicates, ±1.96 SE")
    ax.set_xlim(0.47, 0.63)
    ax.grid(True, color=_GRIDLINE, lw=0.5)
    axc.set_xscale("log", base=2)
    axc.set(xlabel="cluster size s", ylabel="cv of Y_s", title="(b) Assumption 6: cv per scale")
    axc.grid(True, which="both", color=_GRIDLINE, lw=0.5)
    axc.legend(fontsize=8, frameon=False)
    fig.savefig(out, dpi=130)


def pooled_coverage(path: Path, m0s=None) -> dict:
    ts = _json(path)
    cells = [c for c in ts["cells"] if m0s is None or c["m0"] in m0s]
    hits = sum(c["coverage"]["hits"] for c in cells)
    reps = sum(c["coverage"]["replicates"] for c in cells)
    # A cell holding a gamma-hat = -inf replicate (a zero top-rung mean, experiment 23)
    # has infinite RMSE and a bound/RMSE of 0; it says nothing about the bound's width.
    ratios = [c["coverage"]["bound_over_rmse"] for c in cells
              if math.isfinite(c["coverage"]["rmse"])]
    return {"hits": hits, "reps": reps, "ratio": (min(ratios), max(ratios))}


def fig_coverage_forest(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    e = data
    rows = [
        ("20  d_f, d=2 (CLT run)", e / "20_perc_df_d2_clt/data/clt/theorem_scores.json"),
        ("21  d_f, d=3 (CLT run)", e / "21_perc_df_d3_clt/data/clt/theorem_scores.json"),
        ("22  d_f, d=2 (30 cells)", e / "22_perc_df_d2_error_decay/data/error_decay/theorem_scores.json"),
        ("23  d_f, d=3 (24 cells)", e / "23_perc_df_d3_error_decay/data/error_decay/theorem_scores.json"),
        ("25  τ, d=2, box ∝ s^(1/2)", e / "25_perc_tau_d2_coverage/data/coverage/theorem_scores.json"),
        ("25  τ, d=2, box ∝ s^(48/91)", e / "25_perc_tau_d2_coverage/data/coverage_box_df/theorem_scores.json"),
        ("26  γ/ν, d=2, origin", e / "26_perc_gammanu_d2_coverage/data/coverage/theorem_scores.json"),
    ]
    fig = Figure(figsize=(11.5, 4.2), constrained_layout=True)
    ax, axf = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    for y, (label, path) in enumerate(reversed(rows)):
        c = pooled_coverage(path)
        lo, hi = wilson_score_interval(c["hits"], c["reps"])
        p = c["hits"] / c["reps"]
        color = _CRITICAL if hi < 0.95 else _BLUE
        ax.plot([lo, hi], [y, y], "-", color=color, lw=2)
        ax.plot([p], [y], "o", color=color, ms=6)
        r0, r1 = c["ratio"]
        ratio = f"{r0:.1f}" if abs(r1 - r0) < 0.05 else f"{r0:.1f}–{r1:.1f}"
        ax.text(1.02, y, f"{c['hits']}/{c['reps']}", va="center", fontsize=8)
        ax.text(1.16, y, f"×{ratio}", va="center", fontsize=8, color=_MUTED)
    ax.axvline(0.95, color=_MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.set_xlim(0.3, 1.3)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    ax.set(xlabel="fraction with |γ̂ − γ| ≤ eq. (720) half-width  (Wilson 95% CI)",
           title="(a) coverage: hits / replicates,  grey: bound / RMSE")
    ax.grid(True, axis="x", color=_GRIDLINE, lw=0.5)

    runs = [
        ("27  d_f, d=3", "27_perc_df_d3_facefar", DF3_LIT),
        ("29  d_f, d=4", "29_perc_df_d4_facefar", 3.0446),
        ("30  d_f, d=5", "30_perc_df_d5_facefar", 3.5260),
    ]
    for y, (label, exp, lit) in enumerate(reversed(runs)):
        a = _json(e / exp / "data/pilot/answer.json")
        g, h = a["gamma"], a["wilson"]["half_width"]
        tlo, thi = ((c - g) / h for c in a["ci"])
        axf.plot([-1, 1], [y, y], "-", color=_BLUE, lw=6, alpha=0.35, solid_capstyle="butt",
                 label="eq. (720) bound" if y == 0 else None)
        axf.plot([tlo, thi], [y, y], "-", color=_INK, lw=2.5, solid_capstyle="butt",
                 label="t(4) replicate interval" if y == 0 else None)
        axf.plot([(lit - g) / h], [y], "D", color=_ORANGE, ms=7,
                 label="literature (not used)" if y == 0 else None)
        axf.text(1.08, y, f"{g:.4f} ± {h:.4f}\nlit. {lit}", va="center", fontsize=7.5)
    axf.axvline(0, color=_GRIDLINE, lw=1)
    axf.set_yticks(range(len(runs)), [r[0] for r in reversed(runs)], fontsize=8.5)
    axf.set_xlim(-1.3, 1.85)
    axf.set_ylim(-0.7, len(runs) - 0.3)
    axf.set(xlabel="(value − γ̂) / eq. (720) half-width",
            title="(b) planned runs, in units of their own bound")
    axf.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3)
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="checkout holding experiments/*/data (default: this one)")
    args = ap.parse_args()
    data = args.root / "experiments"
    jobs = [
        (fig_walkthrough, HERE / "27_perc_df_d3_facefar/images/estimator_walkthrough.png"),
        (fig_box_rule, HERE / "25_perc_tau_d2_coverage/images/box_rule.png"),
        (fig_coverage_forest, HERE / "images/coverage_forest.png"),
    ]
    for fn, out in jobs:
        out.parent.mkdir(parents=True, exist_ok=True)
        fn(data, out)
        print(f"figure = {out}")


if __name__ == "__main__":
    main()
