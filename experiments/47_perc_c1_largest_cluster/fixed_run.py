"""A long run on a PUBLISHED allocation, then the pipeline's Step 4 and estimates on it.

User, 2026-11-05: Zhang et al.'s 5D allocation (arXiv:2004.11289 Sec. II: L = 4, 6, 8,
10, 12, 14, 16, 20, 24, 32, 40, 48, 60; 1e6 samples for L <= 20, 1e5 above) is the LONG
RUN of a study on the largest cluster C1, and the study gets the same checks as every
pipeline run. The ladder is not rho^k, so the pipeline's own drivers (run.py,
report.py, local_slope.py, plot_pipeline.py) refuse it; this driver does the same steps
on any increasing ladder, with `tools/local_slope.py`'s pair-form functions.

Steps, on a study directory that already holds a pilot (src/study/pilot.py):

  draw    the allocation, R replicates of the recipe's per-scale n (so the pooled run
          is R x n), one spawned seed per replicate, summarized per scale exactly as
          run.py does. Writes final.json (n is a per-scale list, rho and m0 are null).
  report  * gamma-hat: the OLS slope of log Ybar on log i -- the article's eq. (526)
            IS the OLS slope on a rho^k window, so this is its generalization -- on
            every window [i_min, top] with >= 3 scales. The headline is the top m = 6
            scales, the pipeline's m. Its se is the replicate spread (t(R-1)), and
            the closed-form delta-method sd sum_k w_k^2 cv_k^2 / (n_k R) beside it.
            No eq. (wilson) interval: the theorem is proved for rho^k ladders only.
          * eq. (232) fitted on the long-run means with a cutoff i >= i_min (as
            experiment 43 did), gamma free: omega1, a1, chi2 per cutoff.
          * the pilot refitted with gamma pinned to the headline gamma-hat
            (report.pinned_refit, the B_fs refit of every pipeline study).
          * local slopes in pair form (tools/local_slope.analyse_pairs), with the
            CLT se and the n-mismatch term, which is not 0 here: n changes at L = 24.
          Writes fixed_run_report.json.
  plot    fixed_longrun.{pdf,png} and fixed_checks.{pdf,png}: plot_pipeline.py's
          Step 3 and Step 4 figures, with log2 i on the x axis (i is not rho^k).

Truths appear nowhere (ground rule 4); the experiment README scores the answer.

    python3 experiments/47_perc_c1_largest_cluster/fixed_run.py draw --study DIR \\
        --recipe experiments/47_perc_c1_largest_cluster/recipes/samples_zhang_c1_d5.json \\
        --replicates 5 --seed 2026110551
    python3 experiments/47_perc_c1_largest_cluster/fixed_run.py report --study DIR
    python3 experiments/47_perc_c1_largest_cluster/fixed_run.py plot --study DIR
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import local_slope as LS  # noqa: E402
from tools import progress as P  # noqa: E402
from tools.artifacts import load_recipe, read_artifact, write_artifact  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402
from tools.summary import replicate_summary, summarize_scale  # noqa: E402

from src.generate.generate import generate  # noqa: E402
from src.study.report import pinned_refit  # noqa: E402
from src.study.run import preflight  # noqa: E402

PRODUCED_BY = "experiments/47_perc_c1_largest_cluster/fixed_run.py"
M_HEADLINE = 6      # the pipeline's m: the headline gamma-hat uses the top 6 scales
LEVEL = 0.95


# --- draw ---------------------------------------------------------------------------

def draw(sd: Path, recipe_path: Path, R: int, seed: int) -> None:
    pilot = read_artifact(sd, "pilot")
    recipe = load_recipe(recipe_path, "samples")
    model, params = recipe["model"], recipe.get("params", {})
    if (pilot["recipe"]["model"], pilot["recipe"].get("params", {})) != (model, params):
        raise SystemExit(f"the pilot in {sd} is {pilot['recipe']['model']} "
                         f"{pilot['recipe'].get('params')}, the recipe {model} {params}")
    if seed_record(seed) == pilot["recipe"].get("seed"):
        raise SystemExit(f"--seed {seed} is the pilot's; pass another")
    scales, n = [int(i) for i in recipe["scales"]], recipe["n"]
    if not isinstance(n, list) or len(n) != len(scales):
        raise SystemExit("the recipe needs one n per scale (a published allocation)")
    preflight(model, scales, params)
    reps, seeds, per_scale_seconds = [], [], []
    t0 = time.perf_counter()

    def result() -> dict:
        return {"replicates": len(reps), "scales": scales, "n": n, "m0": None, "rho": None,
                "m": len(scales), "model": model, "params": params,
                "per_replicate": reps, "seeds": seeds,
                "per_scale_seconds": per_scale_seconds,
                "elapsed_seconds": time.perf_counter() - t0, "samples_kept": False,
                "allocation": {"recipe": str(recipe_path), "comment": recipe.get("_comment")}}

    for k, ss in enumerate(spawn(seed, R)):
        P.say(f"  replicate {k + 1}/{R}  ({len(scales)} scales, {sum(n):,} samples) ...")
        secs = {}
        stat = generate(model, scales, n, params, seed=ss, reduce=summarize_scale,
                        on_scale=lambda i, n_i, s: secs.__setitem__(int(i), float(s)))
        reps.append(replicate_summary(stat, scales))
        seeds.append(seed_record(ss))
        per_scale_seconds.append([secs[i] for i in scales])
        write_artifact(sd, "final_partial", {**result(), "replicates_planned": R},
                       produced_by=PRODUCED_BY)
        P.say(f"    replicate {k + 1}/{R} drawn in {sum(secs.values()):.1f}s")
    out = write_artifact(sd, "final", result(), produced_by=PRODUCED_BY)
    print(f"output = {out}")


# --- report -------------------------------------------------------------------------

def ols_weights(i: np.ndarray) -> np.ndarray:
    """w with w @ log Ybar = the OLS slope on log i; eq. (526)'s weights on a rho^k window."""
    x = np.log(i)
    return (x - x.mean()) / np.sum((x - x.mean()) ** 2)


def gamma_window(i, y, y_pool, cv, n, R) -> dict:
    w = ols_weights(i)
    g = float(w @ np.log(y_pool))
    per_rep = np.log(y) @ w
    se = float(per_rep.std(ddof=1) / math.sqrt(R))
    q = float(stats.t.ppf(0.5 + LEVEL / 2, R - 1))
    return {"scales": i.astype(int).tolist(), "gamma": g, "se_replicates": se,
            "interval_t": [g - q * se, g + q * se], "dof": R - 1,
            "sd_closed_form": float(np.sqrt(np.sum(w ** 2 * cv ** 2 / (n * R)))),
            "per_replicate": per_rep.tolist()}


def fit_232(i, y, sig) -> dict:
    """eq. (232), J = 1, gamma free, weighted by sigma_log; restarts as tools/correction."""
    li, ly, wt = np.log(i), np.log(y), 1 / sig
    res = lambda t: wt * (t[0] + t[1] * li + t[2] * i ** (-t[3]) - ly)  # noqa: E731
    g0, a00 = np.polyfit(li, ly, 1)
    best = None
    for om in (0.25, 0.5, 1.0, 2.0, 4.0):
        for a1 in (-1.0, 1.0):
            f = optimize.least_squares(res, [a00, g0, a1, om],
                                       bounds=([-np.inf] * 3 + [1e-3], [np.inf] * 3 + [20]))
            if best is None or f.cost < best.cost:
                best = f
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    chi2, dof = float(2 * best.cost), int(i.size - 4)
    return {"scales": i.astype(int).tolist(), "gamma": float(best.x[1]),
            "se_gamma": float(np.sqrt(cov[1, 1])), "a1": float(best.x[2]),
            "omega1": float(best.x[3]), "se_omega1": float(np.sqrt(cov[3, 3])),
            "omega1_at_bound": bool(np.isclose(best.x[3], [1e-3, 20]).any()),
            "chi2": chi2, "dof": dof,
            "p_value": float(stats.chi2.sf(chi2, dof)) if dof > 0 else None}


def report(sd: Path) -> None:
    pilot, final = read_artifact(sd, "pilot"), read_artifact(sd, "final")
    reps = final["per_replicate"]
    R = len(reps)
    i = np.asarray(final["scales"], float)
    n = np.asarray(final["n"], float)
    y = np.array([r["y_bar"] for r in reps], float)
    sig = np.array([r["sigma_log"] for r in reps], float)
    cv = np.array([r["cv"] for r in reps], float).mean(axis=0)
    y_pool = y.mean(axis=0)
    sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))

    windows = [gamma_window(i[k:], y[:, k:], y_pool[k:], cv[k:], n[k:], R)
               for k in range(i.size - 2)]
    head = gamma_window(i[-M_HEADLINE:], y[:, -M_HEADLINE:], y_pool[-M_HEADLINE:],
                        cv[-M_HEADLINE:], n[-M_HEADLINE:], R)
    cut = [fit_232(i[k:], y_pool[k:], sig_pool[k:]) for k in range(i.size - 4)]
    refit = pinned_refit(pilot, head["gamma"], head["se_replicates"])
    slopes = LS.analyse_pairs(i, [y[:, k] for k in range(i.size)], cv=cv, n=n)
    out = write_artifact(sd, "fixed_run_report", {
        "headline": head, "windows": windows, "fit_232_cutoff": cut,
        "pilot_refit": refit, "local_slopes": slopes,
        "pooled": {"scales": i.astype(int).tolist(), "y_bar": y_pool.tolist(),
                   "sigma_log": sig_pool.tolist(), "cv": cv.tolist(), "n_total": (n * R).tolist()},
        "note": "no eq. (wilson) interval: the theorem is proved for rho^k ladders only"},
        produced_by=PRODUCED_BY)

    print(f"long run: {R} replicates, scales {final['scales']}, n per replicate {final['n']}")
    print(f"\nheadline gamma-hat, OLS on the top {M_HEADLINE} scales {head['scales']}:")
    print(f"  {head['gamma']:.5f} +- {head['se_replicates']:.5f} (replicates), "
          f"t({R - 1}) 95% [{head['interval_t'][0]:.5f}, {head['interval_t'][1]:.5f}]; "
          f"closed-form sd {head['sd_closed_form']:.5f}")
    print("\nOLS window scan (gamma-hat on i >= i_min):")
    for w in windows:
        print(f"  i_min = {w['scales'][0]:>3}  ({len(w['scales']):>2} scales)  "
              f"{w['gamma']:.5f} +- {w['se_replicates']:.5f}")
    print("\neq. (232) on the long-run means, cutoff i >= i_min:")
    for f in cut:
        p = "n/a" if f["p_value"] is None else f"{f['p_value']:.2g}"
        print(f"  i_min = {f['scales'][0]:>3}  omega1 = {f['omega1']:.3f} +- {f['se_omega1']:.3f}  "
              f"a1 = {f['a1']:+.3f}  gamma = {f['gamma']:.4f} +- {f['se_gamma']:.4f}  "
              f"chi2 = {f['chi2']:.1f}/{f['dof']}  p = {p}"
              + ("  [omega1 ON BOUND]" if f["omega1_at_bound"] else ""))
    print(f"\npilot refit, gamma pinned to {head['gamma']:.5f}: omega1 = {refit['omega1']:.3f} "
          f"+- {refit['omega1_se']:.3f}, a1 = {refit['a1']:+.3f}, chi2 = {refit['chi2']:.2f}/"
          f"{refit['dof']}, p = {refit['p_value']:.2g}" + ("  WARNING" if refit["warn"] else ""))
    fp, fl = slopes["fits"]["power"], slopes["fits"]["log"]
    print("\nlocal slopes (pair form):")
    for k in range(len(slopes["s"])):
        print(f"  {int(i[k]):>3} -> {int(i[k + 1]):>3}  s = {slopes['s'][k]:.4f} +- "
              f"{slopes['se_s'][k]:.4f}  (CLT {slopes['se_s_clt'][k]:.4f}; "
              f"n-mismatch {slopes['n_mismatch'][k]:+.1e})")
    if fp is not None:
        print(f"  (P) gamma = {fp['gamma']:.4f}, omega = {fp['omega']:.3f}"
              + (f" +- {fp['se_omega']:.3f}" if "se_omega" in fp else "")
              + f", chi2 = {fp['chi2']:.1f}/{fp['dof']}")
    if fl is not None:
        print(f"  (L) gamma = {fl['gamma']:.4f} +- {fl['se_gamma']:.4f}, chi2 = "
              f"{fl['chi2']:.1f}/{fl['dof']}")
    print(f"\noutput = {out}")


# --- plot ---------------------------------------------------------------------------

def plot(sd: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # plot_pipeline.py's palette and style, so the figures sit beside its own.
    DATA, FIT, REF, WINDOW = "#2a6fdb", "#e8622c", "#7a7a7a", "#ecebe4"
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.6, "lines.markersize": 4.5})
    pilot, final = read_artifact(sd, "pilot"), read_artifact(sd, "final")
    rep = read_artifact(sd, "fixed_run_report")
    reps, head = final["per_replicate"], rep["headline"]
    i = np.asarray(final["scales"], float)
    k = np.log2(i)
    pool = rep["pooled"]
    ly, se = np.log(pool["y_bar"]), np.asarray(pool["sigma_log"])

    def curve(f: dict, x2):
        """eq. (232) in natural logs at i = 2^x2."""
        ii = 2.0 ** np.asarray(x2)
        return (f["gamma"] * np.log(ii) + math.log(f["a0"]) + f["a1"] * ii ** -f["omega1"],
                f["gamma"] * np.log(ii) + math.log(f["a0"]))

    # Step 3: the long run's log-log plot with the headline OLS line over its window.
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.2, 4.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]}, constrained_layout=True)
    hw = np.log2(head["scales"])
    w = ols_weights(np.asarray(head["scales"], float))
    lh = ly[-len(hw):]
    icpt = float(np.mean(lh) - head["gamma"] * np.mean(np.log(head["scales"])))
    ax.axvspan(hw[0], hw[-1], color=WINDOW, label=f"top {len(hw)} scales")
    ax.errorbar(k, ly, yerr=se, fmt="o", color=DATA, label=r"$\log\overline{Y}_i$")
    ax.plot(k, head["gamma"] * np.log(i) + icpt, color=FIT,
            label=rf"OLS slope $\hat\gamma={head['gamma']:.4f}$")
    ax.set_ylabel(r"$\log\overline{Y}_i$")
    ax.legend(fontsize=7, loc="upper left")
    axr.axhline(0, color=REF, lw=0.8)
    axr.errorbar(k, ly - (head["gamma"] * np.log(i) + icpt), yerr=se, fmt="o", color=DATA, ms=3.5)
    axr.set_ylabel("residual")
    axr.set_xlabel(r"$\log_2 i$")
    assert np.isclose(w @ lh, head["gamma"])
    for ext in ("pdf", "png"):
        fig.savefig(sd / f"fixed_longrun.{ext}", dpi=200)
    plt.close(fig)

    # Step 4: plot_pipeline.py's panels on the long run.
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    ax_ = fig.subplot_mosaic([["A1", "A1", "A4", "A4", "A5", "A5"],
                              ["A6", "A6", "A6", "X", "X", "X"]])

    def per_rep(ax, values, label):
        for v in values:
            ax.plot(k, v, "o", color=DATA, alpha=0.35, ms=3.5)
        mean = np.mean(values, axis=0)
        ax.plot(k, mean, "o", color=DATA, ms=5.5, label=label)
        return mean

    ax = ax_["A1"]
    ls = rep["local_slopes"]
    mid = 0.5 * (k[:-1] + k[1:])      # s_k is the slope between i_k and i_{k+1}
    ax.axvspan(hw[0], hw[-1], color=WINDOW, label="headline window")
    ax.errorbar(mid, ls["s"], yerr=ls["se_s"], fmt="o", color=DATA, ms=3.5, label=r"$s_k\pm$se")
    ax.axhline(head["gamma"], color=REF, ls="--", lw=1, label=rf"$\hat\gamma={head['gamma']:.4f}$")
    ax.set_ylabel(r"$s_k$")
    ax.set_title(r"(A1) local slopes $s_k$", fontsize=9, loc="left")
    ax.legend(loc="best", fontsize=7)

    per_rep(ax_["A4"], [r["abs_moment"] for r in reps], r"$\overline{|\xi_i|^{3}}$")
    ax_["A4"].set_title(r"(A4) $\mathbb{E}|\xi_i|^{2+\delta}$, $\delta=1$", fontsize=9, loc="left")
    ax = ax_["A5"]
    s2 = per_rep(ax, [np.square(r["cv"]) for r in reps], r"$\hat\sigma_i^2$")
    ax.axhline(s2.max(), color=REF, ls="--", lw=1, label=rf"$\hat\sigma^2_{{\max}}={s2.max():.4f}$")
    ax.set_title(r"(A5) $\hat\sigma_i^2=\widehat{\mathbb{V}}\,\xi_i$", fontsize=9, loc="left")

    ax = ax_["A6"]
    c = pilot["cost"]
    ps = np.asarray(c["probes"][0]["scales"], float)
    med = np.median([pr["elapsed"] for pr in c["probes"]], axis=0)
    per_sample = np.median(np.asarray(final["per_scale_seconds"]), axis=0) / np.asarray(final["n"])
    ax.loglog(ps, med, "o", color=DATA, ms=3.5, label="pilot probe (median)")
    ax.loglog(i, per_sample, "s", color=DATA, mfc="white", ms=4, label="long run (median)")
    ax.set_xlabel("scale $i$")
    ax.set_ylabel("s per sample")
    ax.set_title("(A6) cost", fontsize=9, loc="left")
    ax.legend(loc="lower right", fontsize=7)

    ax = ax_["X"]
    kp = np.log2(pilot["scales"])
    kin = np.linspace(kp[0] - 0.3, kp[-1], 100)
    kout = np.linspace(kp[-1], max(kp[-1], k[-1]) + 0.3, 100)
    full, _ = curve(pilot["direct_fit"], np.concatenate([kin, kout]))
    rf = rep["pilot_refit"]
    pinned, _ = curve(rf, np.concatenate([kin, kout]))
    kk = np.concatenate([kin, kout])
    ax.plot(kp, np.log(pilot["y_bar"]), "o", color=DATA, ms=3.5, label="pilot")
    ax.plot(k, ly, "s", color=DATA, mfc="white", ms=4, label="long run")
    ax.plot(kk, full, color=FIT, label="pilot fit")
    ax.plot(kk, pinned, ":", color=REF, label=r"pilot refit, $\gamma=\hat\gamma$")
    ax.text(0.97, 0.04, rf"refit: $\chi^2/\mathrm{{dof}}={rf['chi2']:.1f}/{rf['dof']}$" "\n"
            rf"$p={rf['p_value']:.1g}$" + (", warning" if rf["warn"] else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.set_ylabel(r"$\log\overline{Y}$")
    ax.set_title("(A1)+(A3) extrapolation", fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=7)
    for name in ("A1", "A4", "A5", "X"):
        ax_[name].set_xlabel(r"$\log_2 i$")
    for name in ("A4", "A5"):
        ax_[name].legend(loc="upper right", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(sd / f"fixed_checks.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {sd}/fixed_longrun.pdf/.png and fixed_checks.pdf/.png")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="draw the allocation as the long run")
    d.add_argument("--study", type=Path, required=True)
    d.add_argument("--recipe", type=Path, required=True)
    d.add_argument("--replicates", type=int, default=5)
    d.add_argument("--seed", type=int, required=True)
    for name in ("report", "plot"):
        sub.add_parser(name).add_argument("--study", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "draw":
        draw(a.study, a.recipe, a.replicates, a.seed)
    elif a.cmd == "report":
        report(a.study)
    else:
        plot(a.study)


if __name__ == "__main__":
    main()
