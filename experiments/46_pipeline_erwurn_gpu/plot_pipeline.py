"""The figures of the article's Section "Simulations" (experiment 46), one per pipeline step.

1. `pipeline_pilot`: (a) the pilot's eq. (pilot-fit) fit to ln Y_bar against k, with the
   correction left after the asymptote is subtracted; (b) the cost probe with its affine
   fit cost(i) = kappa0 + kappa1 i^d, eq. (cost).
   Only the per-scale median over the pilot's cost probes is drawn, with one affine fit to it.
2. `pipeline_allocation`: the estimated finite-size bias and estimated standard error of
   gamma-hat against m0, eqs. (bFS) and (bSE) with the pilot estimates plugged in and
   n(m0) from the plan's budget, with the m0 the plan chose. These are estimates, not
   measurements: only their crossing is used.
   `pipeline_extrapolation`: the pilot correction extended over the planned window.
3. `pipeline_longrun`: the long run's log-log plot with the fitted slope gamma-hat.
4. `pipeline_checks`: the Step 4 checks on the long-run data, one panel each for
   the local slopes, (A4), (A5), (A6), (A8) and (A1)+(A3). The local slopes come from
   `src/study/local_slope.py` (local_slope.json), which brings every scale from the
   pilot's smallest to the window's top to the final's n. (A4)-(A6) are drawn as
   points, not lines: they are estimates at separate scales. (A8) compares the long run's time
   with the planner's cost model; (A1)+(A3) is the log-log plot of pilot and long
   run together, with the pilot's fit extrapolated across the window and the pilot
   refitted with gamma pinned to gamma-hat (the omega1, a1 of B_fs), with its chi2.

Reads a finished study directory only (pilot.json, plan.json, final.json, answer.json).
Truths appear nowhere (ground rule 4).

CLI:
    python3 experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study DIR [--out DIR]

Writes a .pdf (for the article) and a .png (for the README) of each figure into
`--out`, by default the study directory.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.allocation import allocation_constants, max_m0_for_scale  # noqa: E402
from tools.cost_model import fit_cost_probe  # noqa: E402

DATA = "#2a6fdb"      # measured points
FIT = "#e8622c"       # fitted curves
REF = "#7a7a7a"       # asymptotes, references
WINDOW = "#ecebe4"    # shaded planned window

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "lines.markersize": 4.5})


def load(sd: Path) -> dict:
    d = {name: json.loads((sd / f"{name}.json").read_text())
         for name in ("pilot", "plan", "final", "answer", "local_slope")}
    # Only a study planned on a re-fitted cost has one (src/study/refit_cost.py).
    rf = sd / "cost_refit.json"
    d["cost_refit"] = json.loads(rf.read_text()) if rf.exists() else None
    return d


def thin(n: int) -> str:
    """Digit groups split by mathtext thin spaces: a bare ',' in $...$ reads as a list."""
    return f"{int(n):,}".replace(",", r"\,")


def pooled(reps: list[dict], field: str) -> np.ndarray:
    """Mean over replicates of one per-scale summary field."""
    return np.mean([r[field] for r in reps], axis=0)


def pilot_curve(fit: dict, rho: float):
    """eq. (pilot-fit) in natural logs: f(k) = gamma ln(rho) k + ln a0 + a1 rho^(-omega1 k).

    `direct_fit` stores the multiplicative a0 and the natural-log a1 of
    eq. (J-order expansion), so only a0 needs its log.
    """
    g, a0, a1, w = fit["gamma"], fit["a0"], fit["a1"], fit["omega1"]
    full = lambda k: g * math.log(rho) * k + math.log(a0) + a1 * rho ** (-w * k)
    asym = lambda k: g * math.log(rho) * k + math.log(a0)
    return full, asym


def fig_pilot(d: dict, rho: float):
    pilot = d["pilot"]
    k = np.log(pilot["scales"]) / math.log(rho)
    y = np.log(pilot["y_bar"])
    se = np.asarray(pilot["sigma_log"])
    full, asym = pilot_curve(pilot["direct_fit"], rho)
    kk = np.linspace(k[0] - 0.3, k[-1] + 0.3, 200)

    fig = plt.figure(figsize=(7.2, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.05, 1])
    ax, axr, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    ax.sharex(axr)

    ax.errorbar(k, y, yerr=se, fmt="o", color=DATA, label=r"$\log\overline{Y}_{\rho^k}$")
    ax.plot(kk, full(kk), color=FIT, label=r"fit $f(k)$")
    ax.plot(kk, asym(kk), "--", color=REF, lw=1.1, label=r"asymptote $\hat\alpha k+\hat\beta$")
    ax.set_ylabel(r"$\log\overline{Y}$")
    ax.legend(loc="upper left")
    ax.tick_params(labelbottom=False)
    fit = pilot["direct_fit"]
    ax.set_title(rf"(a) scaling fit: $\hat\gamma_{{\rm pilot}}={fit['gamma']:.4f}$, "
                 rf"$\hat\omega_1={fit['omega1']:.2f}$, $\hat a_1={fit['a1']:.2f}$",
                 fontsize=9, loc="left")

    axr.errorbar(k, y - asym(k), yerr=se, fmt="o", color=DATA)
    axr.plot(kk, full(kk) - asym(kk), color=FIT, label=r"$\hat a_1\rho^{-\hat\omega_1 k}$")
    axr.axhline(0, color=REF, lw=0.8, ls="--")
    axr.set_xlabel(r"$k=\log_\rho i$")
    axr.set_ylabel("residual")
    axr.legend(loc="lower right")

    c = pilot["cost"]
    i = np.asarray(c["probes"][0]["scales"], float)
    ii = np.geomspace(i[0], i[-1], 200)
    # One curve, the article's cost-hat: eq. (cost) fitted by the repo's own fitter
    # to the per-scale median over probes, which a single disturbed probe cannot move.
    med = np.median([pr["elapsed"] for pr in c["probes"]], axis=0)
    aff = fit_cost_probe({"scales": list(i), "elapsed": list(med)})["affine"]
    axc.loglog(i, med, "o", color=DATA, label=f"median over {len(c['probes'])} probes")
    axc.loglog(ii, aff["a"] + aff["b"] * ii ** aff["d"], color=FIT,
               label=rf"$\widehat{{\rm cost}}=\hat\kappa_0+\hat\kappa_1 i^{{\hat d}}$, $\hat d={aff['d']:.2f}$")
    axc.set_xlabel("scale $i$")
    axc.set_ylabel("seconds per sample")
    axc.legend(loc="upper left")
    axc.set_title("(b) cost per sample", fontsize=9, loc="left")
    return fig


def tradeoff(d: dict) -> dict:
    """The article's plug-in b_FS-hat and b_SE-hat (eqs. bFS, bSE) for every m0, one replicate.

    n(m0) is plan.py's own (eq. n-of-m0). The curves are the article's formulas with
    the pilot estimates plugged in, not plan.py's internal constants
    (tools/allocation.py, which keep the exact weight sums and a 1/log(rho)): those
    differ by constant factors that move neither curve's shape nor, as asserted
    below, the crossing.
    """
    plan = d["plan"]
    cst = {k: v["value"] for k, v in plan["constants_at_plan_time"].items()}
    rho, m, w, a1 = plan["rho"], plan["m"], cst["omega1"], cst["a1"]
    c = allocation_constants(cst["d"], w, rho, m, a1, cst["cv"])
    top = max_m0_for_scale(plan["max_scale"], m, rho) if plan.get("max_scale") else plan["m0"] + 6
    m0 = np.arange(0, top + 1)
    n = np.floor(plan["budget"] / (c["G"] * rho ** (cst["d"] * m0)))
    # The figure is only worth printing if it reads the plan's own computation.
    j = int(plan["m0"])
    assert n[j] == plan["n"], (n[j], plan["n"])
    assert math.isclose(c["Cb"] * rho ** (-j * w), plan["bias"], rel_tol=1e-9)
    assert math.isclose(c["Cs"] / math.sqrt(n[j]), plan["sd"], rel_tol=1e-9)
    fs = 6 / (m * (m + 1)) * abs(a1) * rho ** (-w * m0) / (rho ** w - 1)
    se = np.sqrt(12 * cst["cv"] ** 2 / (n * m ** 3))
    assert int(m0[np.argmin(np.maximum(fs, se))]) == j, "eq. (m0-star) disagrees with the plan"
    return {"m0": m0, "n": n, "fs": fs, "se": se, "chosen": j, "cst": cst, "m": m}


def fig_allocation(d: dict, rho: float):
    t = tradeoff(d)
    cst, j = t["cst"], t["chosen"]
    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    ax.semilogy(t["m0"], t["fs"], "o-", color=FIT,
                label=r"$\hat b_{\rm FS}(m_0)=\dfrac{6}{m(m+1)}\,"
                      r"\dfrac{|\hat a_1|\,\rho^{-\hat\omega_1 m_0}}{\rho^{\hat\omega_1}-1}$")
    ax.semilogy(t["m0"], t["se"], "s-", color=DATA,
                label=r"$\hat b_{\rm SE}(m_0)=\sqrt{\dfrac{12\,\hat\sigma^2_\infty}{n(m_0)\,m^3}}$")
    # Stops at the crossing, so it never runs behind the formulas in the legend.
    lo = min(t["fs"].min(), t["se"].min()) / 2
    ax.set_ylim(bottom=lo)
    ax.vlines(j, lo, max(t["fs"][j], t["se"][j]), color=REF, ls="--", lw=1)
    ax.annotate(rf"$m_0^\star={j}$" "\n" rf"$n(m_0^\star)={thin(t['n'][j])}$",
                (j, t["se"][0]), xytext=(-5, 4), textcoords="offset points",
                ha="right", fontsize=8)
    ax.set_xlabel(r"smallest scale $m_0$")
    ax.set_ylabel(r"estimated error of $\hat\gamma$")
    ax.legend(loc="upper right", fontsize=8.5, labelspacing=1.2, handlelength=1.6)
    ax.set_title("estimated bias against estimated standard error\n"
                 rf"pilot plug-ins: $m={t['m']}$, $\rho={rho:g}$, $\hat a_1={cst['a1']:.3f}$, "
                 rf"$\hat\omega_1={cst['omega1']:.3f}$, $\hat\sigma^2_\infty={cst['cv'] ** 2:.3f}$, "
                 rf"$\hat d={cst['d']:.3f}$", fontsize=8.5, loc="left")
    return fig


def fig_extrapolation(d: dict, rho: float):
    pilot, plan = d["pilot"], d["plan"]
    k = np.log(pilot["scales"]) / math.log(rho)
    kw = np.log(plan["scales"]) / math.log(rho)
    full, asym = pilot_curve(pilot["direct_fit"], rho)
    kk = np.linspace(k[0] - 0.3, kw[-1] + 0.5, 300)
    y = np.log(pilot["y_bar"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.axvspan(kw[0] - 0.5, kw[-1] + 0.5, color=WINDOW, label="planned window")
    ax.plot(k, y - asym(k), "o", color=DATA, label="pilot points")
    ax.plot(kk, full(kk) - asym(kk), color=FIT, label=r"$\hat a_1\rho^{-\hat\omega_1 k}$")
    ax.axhline(0, color=REF, lw=0.8, ls="--")
    ax.set_xlabel(r"$k=\log_\rho i$")
    ax.set_ylabel(r"$\log\overline{Y}-(\hat\alpha k+\hat\beta)$")
    ax.legend(loc="lower right")
    ax.set_title("pilot correction, extrapolated", fontsize=9, loc="left")
    return fig


def fig_longrun(d: dict, rho: float):
    final, ans = d["final"], d["answer"]
    k = np.log(final["scales"]) / math.log(rho)
    y = np.log(pooled(final["per_replicate"], "y_bar"))
    R = len(final["per_replicate"])
    se = pooled(final["per_replicate"], "sigma_log") / math.sqrt(R)
    slope = ans["gamma"] * math.log(rho)
    icpt = float(np.mean(y - slope * k))   # the OLS intercept for this slope
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.2, 3.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True)
    ax.errorbar(k, y, yerr=se, fmt="o", color=DATA, label=r"$\log\overline{Y}_{\rho^k}$")
    ax.plot(k, slope * k + icpt, color=FIT,
            label=rf"slope $\hat\gamma\log\rho$, $\hat\gamma={ans['gamma']:.5f}$")
    ax.set_ylabel(r"$\log\overline{Y}$")
    ax.legend(loc="upper left")
    ax.set_title(rf"long run: $m_0^\star={final['m0']}$, $m={final['m']}$, "
                 rf"$n={thin(final['n'])}$ for {R} replicates", fontsize=8, loc="left")
    axr.errorbar(k, (y - slope * k - icpt) * 1e4, yerr=se * 1e4, fmt="o", color=DATA)
    axr.axhline(0, color=REF, lw=0.8, ls="--")
    axr.set_ylabel(r"residual $\times10^{4}$")
    axr.set_xlabel(r"$k=\log_\rho i$")
    return fig


def fig_checks(d: dict, rho: float):
    pilot, plan, final = d["pilot"], d["plan"], d["final"]
    reps = final["per_replicate"]
    i = np.asarray(final["scales"], float)
    k = np.log(i) / math.log(rho)
    R, n = len(reps), final["n"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)

    def per_rep(ax, values, label):
        """Each replicate a light point, their mean a full one; scales are not joined."""
        for v in values:
            ax.plot(k, v, "o", color=DATA, alpha=0.35, ms=3.5)
        mean = np.mean(values, axis=0)
        ax.plot(k, mean, "o", color=DATA, ms=5.5, label=label)
        return mean

    ax = axs[0, 0]
    ls = d["local_slope"]["analysis"]
    # s_k is the slope between rho^k and rho^(k+1), so it sits between them.
    ks = np.log(ls["scales"][:-1]) / math.log(rho) + 0.5
    ax.axvspan(k[0], k[-1], color=WINDOW, label="long-run window")
    ax.errorbar(ks, ls["s"], yerr=ls["se_s"], fmt="o", color=DATA, ms=3.5,
                label=r"$s_k \pm$ se")
    ax.axhline(d["answer"]["gamma"], color=REF, ls="--", lw=1,
               label=rf"$\hat\gamma={d['answer']['gamma']:.5f}$")
    ax.set_ylabel(r"$s_k$")
    ax.set_title(r"(A1) local slopes $s_k$", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7)

    per_rep(axs[0, 1], [r["abs_moment"] for r in reps], r"$\overline{|\xi_i|^{3}}$")
    axs[0, 1].set_title(r"(A4) $\mathbb{E}|\xi_i|^{2+\delta}$, $\delta=1$", fontsize=9, loc="left")
    axs[0, 1].ticklabel_format(axis="y", useOffset=False)

    # report.py's Lambda: the largest over scales of the replicate mean.
    L = per_rep(axs[0, 2], [r["log_moment"] for r in reps],
                r"$\overline{|\log\xi_i|^{3}}$").max()
    assert math.isclose(L, d["answer"]["Lambda"], rel_tol=1e-12), (L, d["answer"]["Lambda"])
    axs[0, 2].axhline(L, color=REF, ls="--", lw=1, label=rf"$\hat L={L:.2f}$")
    axs[0, 2].set_title(r"(A5) $\mathbb{E}|\log\xi_i|^{2+\delta}$", fontsize=9, loc="left")

    ax = axs[1, 0]
    s2 = per_rep(ax, [np.square(r["cv"]) for r in reps], r"$\hat\sigma_i^2$")
    ax.axhline(s2.max(), color=REF, ls="--", lw=1, label=rf"$\hat\sigma^2_{{\max}}={s2.max():.4f}$")
    ax.set_title(r"(A6) $\hat\sigma_i^2=\widehat{\mathbb{V}}\,\xi_i$", fontsize=9, loc="left")

    ax = axs[1, 1]
    c = pilot["cost"]
    ps = np.asarray(c["probes"][0]["scales"], float)
    med = np.median([pr["elapsed"] for pr in c["probes"]], axis=0)
    # The planner's own cost model (plan.py's plan_for_budget): i^d, converted to
    # the model's work unit by cost_ratio, at the pilot's throughput.
    dd, tp = plan["constants_at_plan_time"]["d"]["value"], plan["throughput"]
    plan_cost = lambda s: np.asarray(s, float) ** dd * plan["cost_ratio"] / tp
    assert math.isclose(n * float(np.sum(plan_cost(i))), plan["seconds"], rel_tol=1e-3)
    ii = np.geomspace(ps[0], i[-1], 200)
    ax.loglog(ps, med, "o", color=DATA, ms=3.5, label="pilot probe (median)")
    for r in [r for r in ((d["cost_refit"] or {}).get("rounds") or []) if not r["skipped"]]:
        ax.loglog(r["scales"], r["per_sample"], "^", color=REF, ms=4.5, zorder=3,
                  label=f"cost re-fit, {r['n_short']:,} per scale")
    ax.loglog(ii, plan_cost(ii), color=FIT, label="planner's model")
    # The long run times whole replicates, not single scales: its measured total is
    # shared across the window in the proportions the plan predicts.
    ratio = final["elapsed_seconds"] / (R * plan["seconds"])
    ax.loglog(i, plan_cost(i) * ratio, "s", color=DATA, mfc="white", ms=4,
              label=rf"long run ($\times{ratio:.2f}$)")
    ax.set_xlabel("scale $i$")
    ax.set_ylabel("s per sample")
    ax.set_title("(A8) cost", fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = axs[1, 2]
    full, _ = pilot_curve(pilot["direct_fit"], rho)
    kp = np.log(pilot["scales"]) / math.log(rho)
    kin = np.linspace(kp[0] - 0.3, kp[-1], 100)
    kout = np.linspace(kp[-1], k[-1] + 0.3, 100)
    ax.plot(kp, np.log(pilot["y_bar"]), "o", color=DATA, ms=3.5, label="pilot")
    ax.plot(k, np.log(pooled(reps, "y_bar")), "s", color=DATA, mfc="white", ms=4, label="long run")
    ax.plot(kin, full(kin), color=FIT, label="pilot fit $f(k)$")
    ax.plot(kout, full(kout), "--", color=FIT, label="...extrapolated")
    # The refit with gamma pinned to gamma-hat is what B_fs is built from (report.py,
    # user 2026-11-02); its chi2 p-value is the warning that one correction is not enough.
    rf = d["answer"]["pilot_refit"]
    pinned, _ = pilot_curve(rf, rho)
    ax.plot(np.concatenate([kin, kout]), pinned(np.concatenate([kin, kout])), ":", color=REF,
            label=r"pilot refit, $\gamma=\hat\gamma$")
    ax.text(0.97, 0.04, rf"refit: $\chi^2/\mathrm{{dof}}={rf['chi2']:.1f}/{rf['dof']}$" "\n"
            rf"$p={rf['p_value']:.1g}$" + (", warning" if rf["warn"] else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.set_ylabel(r"$\log\overline{Y}$")
    ax.set_title("(A1)+(A3) extrapolation", fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=7)

    for ax in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 2]):
        ax.set_xlabel(r"$k=\log_\rho i$")
    for ax, loc in ((axs[0, 1], "upper right"), (axs[0, 2], "lower right"),
                    (axs[1, 0], "upper right")):
        ax.legend(loc=loc, fontsize=7)
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", type=Path, required=True, help="finished study directory")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: --study)")
    a = ap.parse_args(argv)
    out = a.out or a.study
    out.mkdir(parents=True, exist_ok=True)
    d = load(a.study)
    rho = float(d["plan"]["rho"])
    for name, make in (("pipeline_pilot", fig_pilot), ("pipeline_allocation", fig_allocation),
                       ("pipeline_extrapolation", fig_extrapolation),
                       ("pipeline_longrun", fig_longrun), ("pipeline_checks", fig_checks)):
        fig = make(d, rho)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        print(f"wrote {out / name}.pdf/.png")


if __name__ == "__main__":
    main()
