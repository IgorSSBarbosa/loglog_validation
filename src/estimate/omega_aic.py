"""The AIC fit: omega_1 and gamma from the log-log points on an AIC-chosen range
(user, 2026-11-05).

On a study's log-log points L_k = mean log Ybar at i = rho^k, the model
L = a + b k + c rho^(-omega k) (eq. (232) truncated at J = 1) is fitted by weighted
least squares on every run of >= 6 consecutive points: the triple (i_min, i_max,
omega). The left cut drops scales where omega_2, ... still act, the right cut scales
the model does not describe. Each range is scored by AIC with data cuts (Jay & Neil,
Phys. Rev. D 103, 114502 (2021)), AIC = chi2 + 2*4 + 2*N_cut, weight ~ exp(-AIC/2), so
dropping a point must buy a chi2 drop > 2. Reported: omega and gamma = b / log rho
averaged over the closed ranges with those weights (Jay & Neil eq. 15; user,
2026-11-09), each +- stat (weighted rms of the ranges' profile intervals / linearized
se) +- range systematic (weighted sd over closed ranges), their eq. 18. The figure
draws the highest-weight range. See `tools/local_slope.omega_ranges`.

Points are weighted by the CLT se, sqrt(sum_r cv_r^2 / n_r) / R, from each cell's own cv
and n (user, 2026-11-05): the sd of R = 3 replicate means rests on 2 dof.

Model-agnostic: it reads only <study>/local_slope.json (src/study/local_slope.py, or
autopilot's last step), so any experiment that runs the pipeline gets it. It never
sees a known gamma or omega_1 (ground rule 4). Nothing is drawn.

Standard in the pipeline: src/study/local_slope.py runs it on every completed run.
Run by hand on a study whose local_slope.json predates that, or on several at once.
Writes <study>/omega_aic.json and omega_aic_fit.png.

CLI:
    python3 src/estimate/omega_aic.py --data-root <D> --study <s> [<s> ...] [--min-points 6]
    python3 src/estimate/omega_aic.py --data-root <D> --all
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import ARTIFACTS, artifact_path, read_artifact, write_artifact  # noqa: E402
from tools import local_slope as LS  # noqa: E402

PRODUCED_BY = "src/estimate/omega_aic.py"

# Same palette as tools/loglog_plot.py.
_BLUE, _ORANGE, _MUTED = "#2a78d6", "#eb6834", "#898781"


def log_points(sd: Path) -> dict:
    """The study's log-log points and their CLT se, from local_slope.json."""
    ls = read_artifact(sd, "local_slope")
    a = ls["analysis"]
    scales, rho = a["scales"], a["rho"]
    by_scale = {i: [c for c in ls["cells"] if c["scale"] == i] for i in scales}
    # CLT se from every replicate's cv and n, not the sd of R = 3 replicate means
    # (user, 2026-11-05); the replicate se stays alongside as a cross-check.
    se = LS.clt_se_log([[c["cv"] for c in by_scale[i]] for i in scales],
                       [[c["n"] for c in by_scale[i]] for i in scales])
    return {"study": sd.name, "model": ls.get("model"), "params": ls.get("params"),
            "rho": rho, "scales": scales, "log_y": a["log_y"],
            "source": "local_slope.json: analysis log_y; cells cv, n (one n at every scale)",
            "se_log_y": {"used": "clt", "clt": se.tolist(), "replicates": a["se_log_y"]}}


def analyse(sd: Path, min_points: int = 6) -> dict:
    pts = log_points(sd)
    return {**pts, "range_fit": LS.omega_ranges(pts["scales"], pts["log_y"],
                                                np.asarray(pts["se_log_y"]["clt"]), pts["rho"],
                                                min_points=min_points)}


def plot_aic_fit(res: dict, out: Path) -> None:
    """The AIC fit: the best range's a + b x + c rho^(-omega x), x = k - k_min, on the
    log-log points. (a) the data and the curve; (b) the data minus the fitted line
    a + b x against c rho^(-omega x), the only part that carries omega. Points outside
    the chosen range are drawn hollow and grey; they did not enter the fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rf, rho = res["range_fit"], res["rho"]
    b = rf["best"]
    lr = math.log(rho)
    k = np.log(np.asarray(res["scales"], float)) / lr
    L, se = np.asarray(res["log_y"]), np.asarray(res["se_log_y"]["clt"])
    k1, k2 = math.log(b["i_min"]) / lr, math.log(b["i_max"]) / lr
    inside = (k >= k1 - 1e-9) & (k <= k2 + 1e-9)
    x = k - k1
    xs = np.linspace(k1, k2, 400) - k1
    line = lambda xx: b["a"] + b["b"] * xx  # noqa: E731
    corr = lambda xx: b["c"] * rho ** (-b["omega"] * xx)  # noqa: E731

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18, 5))
    for ax, y, curve in ((a0, L, lambda xx: line(xx) + corr(xx)),
                         (a1, L - line(x), corr)):
        ax.errorbar(k[inside], y[inside], yerr=se[inside], color=_BLUE, marker="o", ms=5,
                    lw=0, elinewidth=1, label="in the chosen range")
        ax.errorbar(k[~inside], y[~inside], yerr=se[~inside], color=_MUTED, marker="o", ms=5,
                    mfc="white", lw=0, elinewidth=1, label="cut (not fitted)")
        ax.plot(xs + k1, curve(xs), color=_ORANGE, lw=2, label="AIC fit")
        ax.axvspan(k1, k2, color=_ORANGE, alpha=0.06, lw=0)
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_xlabel(f"k = log_{rho:g} i")
    a0.set(ylabel="mean log Ȳ", title="(a) log-log points and the AIC fit")
    a1.axhline(0, color=_MUTED, lw=0.8)
    a1.set(ylabel="mean log Ȳ − (a + b·(k − k_min))",
           title="(b) data minus the fitted line, vs c·ρ^(−ω(k − k_min))\n(cut points lie off this axis)")
    # the correction decays to ~0: scale to the in-range points, not the cut left edge
    yin = (L - line(x))[inside]
    sp = np.ptp(np.concatenate([yin, corr(xs)]))
    a1.set_ylim(min(yin.min(), corr(xs).min()) - 0.15 * sp - 2 * se[inside].min(),
                max(yin.max(), corr(xs).max()) + 0.15 * sp + 2 * se[inside].min())
    # (c) the same on a log scale, where the model is a line of slope -omega log rho
    sgn = 1.0 if b["c"] > 0 else -1.0
    yc = sgn * (L - line(x))
    floor = 0.3 * min(se[inside].min(), abs(corr(xs[-1])), yc[yc > 0].min())
    for m_, col, face in ((inside, _BLUE, _BLUE), (~inside, _MUTED, "white")):
        pos = m_ & (yc > 0)
        a2.errorbar(k[pos], yc[pos], yerr=[np.clip(yc[pos] - floor, 0, se[pos]), se[pos]],
                    color=col, marker="o", ms=5, mfc=face, lw=0, elinewidth=1)
        neg = m_ & (yc <= 0)
        a2.plot(k[neg], np.full(neg.sum(), floor * 1.3), "v", color=col, mfc=face, ms=6, ls="none")
    a2.plot(xs + k1, abs(corr(xs)), color=_ORANGE, lw=2)
    a2.axvspan(k1, k2, color=_ORANGE, alpha=0.06, lw=0)
    a2.set_yscale("log")
    a2.set_ylim(floor, None)
    a2.grid(True, color="#e1e0d9", lw=0.6, which="both")
    a2.set(xlabel=f"k = log_{rho:g} i", ylabel=f"{'' if sgn > 0 else '−'}(mean log Ȳ − line), log scale",
           title=f"(c) the same, log scale: slope −ω̂·log ρ  (▽: ≤ 0, no log)")
    a0.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{res['study']}: highest-weight range i = {b['i_min']}..{b['i_max']} "
                 f"({b['points']} points, weight {rf['best_weight']:.2f})"
                 f"  —  AIC-weighted ω̂ = {rf['omega']:.3f} ± {rf['stat']:.3f} (stat) ± {rf['systematic']:.3f}"
                 f" (range),  γ̂ = b / log ρ = {rf['gamma']:.6f} ± {rf['gamma_stat']:.6f} ± "
                 f"{rf['gamma_systematic']:.6f},  χ² = {b['chi2']:.1f} / {b['dof']}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def omega_aic(sd: Path, min_points: int = 6) -> Path | None:
    """The whole tool on the study in `sd`: writes omega_aic.json and omega_aic_fit.png,
    prints the top ranges. src/study/local_slope.py calls it on every completed run.
    A ladder shorter than `min_points` has no range to fit: says so, writes nothing,
    returns None."""
    m = len(read_artifact(sd, "local_slope")["analysis"]["scales"])
    if m < min_points:
        print(f"[omega_aic] {sd.name}: {m} log-log points < {min_points}; no range fit, "
              "nothing written")
        return None
    res = analyse(sd, min_points)
    path = write_artifact(sd, "omega_aic", res, produced_by=PRODUCED_BY)
    rf = res["range_fit"]
    print(f"[omega_aic] {sd.name}  (rho = {res['rho']:g}, {len(res['scales'])} points); "
          f"{rf['criterion']}")
    print(f"{'i_min':>8} {'i_max':>8} {'pts':>4} {'omega':>7} {'68% interval':>17} "
          f"{'chi2/dof':>10} {'gamma':>9} {'weight':>7}")
    for f in sorted(rf["fits"], key=lambda f: -f["weight"])[:8]:
        bad = "*" if f["interval_open"] or f["omega_on_bound"] else " "
        print(f"{f['i_min']:>8} {f['i_max']:>8} {f['points']:>4} {f['omega']:7.3f} "
              f"[{f['omega_lo']:6.3f}, {f['omega_hi']:6.3f}]{bad}"
              f" {f['chi2']:5.1f}/{f['dof']:<3} {f['gamma']:9.5f} {f['weight']:7.3f}")
    print("(* = interval open or omega on a bound: left out of the average; weight "
          f"{rf['weight_left_out']:.3f})")
    if rf["best"] is None:
        print(f"no estimate: {rf['reason']}")
    else:
        print(f"omega = {rf['omega']:.3f} +- {rf['stat']:.3f} (stat) +- {rf['systematic']:.3f}"
              f" (range), AIC-weighted over the kept ranges; the highest-weight range "
              f"{rf['best']['i_min']}..{rf['best']['i_max']} carries {rf['best_weight']:.2f}")
        print(f"gamma = {rf['gamma']:.6f} +- {rf['gamma_stat']:.6f} (stat) +- "
              f"{rf['gamma_systematic']:.6f} (range)")
        plot_aic_fit(res, artifact_path(sd, "omega_aic_fit_figure"))
    print(f"[omega_aic] -> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, required=True, help="where the studies live")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--study", nargs="+", help="one or more study directory names")
    g.add_argument("--all", action="store_true",
                   help="every study under --data-root that has local_slope.json")
    ap.add_argument("--min-points", type=int, default=6,
                    help="shortest range fitted (>= 5; 6 keeps dof >= 2)")
    args = ap.parse_args()
    if args.all:
        sds = sorted(p.parent for p in args.data_root.glob(f"*/{ARTIFACTS['local_slope']}"))
        if not sds:
            raise SystemExit(f"no study under {args.data_root} has {ARTIFACTS['local_slope']}")
    else:
        sds = [args.data_root / s for s in args.study]
    for sd in sds:
        omega_aic(sd, args.min_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
