"""Figures of a finished `sweep_p.py` sweep: local slopes, log-log fits, gamma_hat and
omega1 against p.

Asked by the user (2026-10-16), after the `2h` sweep, in the style of
experiments/19_rwre_lazy_omega's `slopes_*` and `pilot_fit_*` figures. Nothing is drawn;
everything is read from each study's pilot.json, final.json, answer.json and
constants.json.

Each study has two independent draws: the pilot on 4..4096 and the final run on its own
6 higher scales, with different n and seeds. Local slopes are taken within a draw and
never across the two (ground rule 2), so each p's slope curve has two segments, and a
gap wherever the final ladder doesn't start at 8192.

  - slopes_<tag>.png: s_k (± CLT se) from `tools/local_slope.analyse` at rho = 2.
  - loglog_<tag>.png: log2 Ybar_i (± se) of both draws, with the pilot's eq. (232) fit
    (solid on the pilot's scales, dashed where it is extrapolated over the final's).
  - gamma_vs_p_<tag>.png: the final gamma_hat with its Wilson 95% interval, against
    gamma = 1/2 (p <= 3/4) and 2p - 1 (p > 3/4).
  - omega_vs_p_<tag>.png: the pilot's omega1 ± se (the one that sized each plan; the
    final fit's omega1 is noise, see sweep_p.py's summary), against the user's
    conjecture omega = 1 for p < 1/2 and |4p - 3| above (user, 2026-10-16).
  - omega_loglog_vs_p_<tag>.png (with --omega-loglog): each study's omega_loglog.json
    (`src/estimate/omega_loglog.py`, the loglog-log plot: pilot gamma_k against the
    final's gamma-hat) ± se, next to the pilot's eq. (232) omega1 and the conjecture.
    Run that driver on every study first; a missing file raises.
  - with --article DIR: erw_gamma_vs_p.pdf and erw_omega_vs_p.pdf for the article's ERW
    appendix (user, 2026-11-02), without titles. The omega figure draws only the pilot
    refitted with gamma pinned to gamma-hat (answer.json's pilot_refit), open where its chi2
    warns; the other two estimates are in the article's table. The gamma figure adds a panel of
    gamma_hat - gamma with its Wilson interval, which panel (a)'s scale hides. Needs
    --omega-loglog's inputs.

  - with --aic (user, 2026-11-05): omega_aic_vs_p_<tag>.png, the AIC fit's omega
    (src/estimate/omega_aic.py's range_fit: a + bk + c rho^(-omega k) on the AIC-chosen
    range of the local-slope ladder) against the conjecture, error bars the fit's stat
    error only; and gamma_aic_vs_p_<tag>.png, |gamma_aic - gamma| against p with the fit's
    stat se, beside the standard method's |gamma_hat - gamma| (answer.json: the log-log
    OLS on the final run's last m scales) with its replicate se. The range systematic is
    left out of both figures (user, 2026-11-05): it double-counts each range's noise. A
    study without omega_aic.json, or whose fit gave no estimate, is left out and named.

    python3 experiments/14_erw_urn_gpu/plot_sweep.py --tag 2h [--omega-loglog] [--article DIR] [--aic]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import local_slope as LS  # noqa: E402

RHO = 2.0


def label(p: float) -> str:
    return f"{p:.4g}"


def gamma_true(p: float) -> float:
    return 0.5 if p <= 0.75 else 2 * p - 1


def omega_conjecture(p: np.ndarray) -> np.ndarray:
    return np.where(p < 0.5, 1.0, np.abs(4 * p - 3))


def slopes(scales: list, reps: list, n: list) -> tuple[np.ndarray, np.ndarray]:
    """(s_k, CLT se) of one draw, placed at the lower scale of each pair."""
    y = [[r["y_bar"][j] for r in reps] for j in range(len(scales))]
    cv = [float(np.mean([r["cv"][j] for r in reps])) for j in range(len(scales))]
    res = LS.analyse(scales, y, RHO, cv=cv, n=n)
    return np.asarray(res["s"]), np.asarray(res["se_s_clt"])


def load(sd: Path) -> dict:
    pilot = json.loads((sd / "pilot.json").read_text())
    final = json.loads((sd / "final.json").read_text())
    answer = json.loads((sd / "answer.json").read_text())
    om = json.loads((sd / "constants.json").read_text())["omega1"]
    assert pilot["direct_fit"]["omega1"] == om["value"], f"{sd}: omega1 not from direct_fit"
    assert final["scales"] == answer["scales"], f"{sd}: final.json and answer.json disagree"
    s_p, se_p = slopes(pilot["scales"], pilot["per_replicate"], pilot["n"])
    s_f, se_f = slopes(final["scales"], final["per_replicate"], [final["n"]] * len(final["scales"]))
    return {"pilot": pilot, "answer": answer, "omega1": om,
            "s_pilot": (s_p, se_p), "s_final": (s_f, se_f)}


def plot_slopes(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    ps = sorted(res)
    for t, p in enumerate(ps):
        r, c = res[p], cmap(t / max(len(ps) - 1, 1))
        for key, scales, mk in (("s_pilot", r["pilot"]["scales"], "o"),
                                ("s_final", r["answer"]["scales"], "s")):
            s, se = r[key]
            ax.errorbar(scales[:-1], s, yerr=se, color=c, marker=mk, ms=4, lw=1,
                        label=f"p={label(p)}" if key == "s_pilot" else None)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="k (lower scale of the pair)", ylabel="s_k",
           title="drawn local slopes s_k (± se): pilot ● and final ■")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_loglog(res: dict, out: Path, plt) -> None:
    # Wide, legend outside the axes: with 21 p's inside it covered the final's points
    # (user, 2026-11-02).
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("viridis")
    ps = sorted(res)
    for t, p in enumerate(ps):
        r, c = res[p], cmap(t / max(len(ps) - 1, 1))
        pilot, ans, f = r["pilot"], r["answer"], r["pilot"]["direct_fit"]
        # sigma_log is the se of ln Ybar; the plot is in log2.
        for d, mk in ((pilot, "o"), (ans, "s")):
            ax.errorbar(np.log2(d["scales"]), np.log2(d["y_bar"]),
                        yerr=np.asarray(d["sigma_log"]) / math.log(2),
                        color=c, marker=mk, ms=4, lw=0)
        ip, ifin = np.log2(pilot["scales"]), np.log2(ans["scales"])

        def f_i(i):
            # fit_correction's a0 is multiplicative and a1 is in ln units (loglog_plot.fit_plot).
            return (f["gamma"] * i + math.log2(f["a0"])
                    + f["a1"] / math.log(2) * 2.0 ** (-f["omega1"] * i))

        ii = np.linspace(ip[0] - 0.3, ip[-1], 200)
        ax.plot(ii, f_i(ii), color=c, lw=0.8,
                label=f"p={label(p)}: γ̂={ans['gamma']:.4f}, ω₁={f['omega1']:.3f}±{r['omega1']['se']:.3f}")
        jj = np.linspace(ip[-1], ifin[-1] + 0.3, 200)
        ax.plot(jj, f_i(jj), color=c, lw=0.8, ls="--")
    ax.set(xlabel=r"$i=\log_2(k)$", ylabel=r"$\log_2\overline{Y}_i$",
           title=r"$\log_2\overline{Y}_i$ (± se), pilot ● final ■, and the pilot's "
                 r"$f(i)=\gamma i+a_0+a_1 2^{-\omega_1 i}$")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_gamma(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    g = np.array([res[p]["answer"]["gamma"] for p in ps])
    lo = np.array([res[p]["answer"]["wilson"]["interval"][0] for p in ps])
    hi = np.array([res[p]["answer"]["wilson"]["interval"][1] for p in ps])
    ax.plot([0, 0.75], [0.5, 0.5], color="#888780", lw=1.2, label="γ = 1/2 (diffusive, p ≤ 3/4)")
    ax.plot([0.75, 1], [0.5, 1], color="#888780", lw=1.2, ls="--",
            label="γ = 2p − 1 (superdiffusive, p > 3/4)")
    ax.errorbar(ps, g, yerr=[g - lo, hi - g], color="#1f6f8b", marker="o", ms=5, lw=0,
                elinewidth=1.2, capsize=3, label="γ̂ (final run), Wilson 95%")
    ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
    ax.set(xlabel="p", ylabel="γ", xlim=(0, 1), title="γ̂ against p")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_omega(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    om = np.array([res[p]["omega1"]["value"] for p in ps])
    se = np.array([res[p]["omega1"]["se"] for p in ps])
    pp = np.linspace(0, 1, 801)
    ax.plot(pp, omega_conjecture(pp), color="#888780", lw=1.2,
            label="conjecture: ω = 1 (p < 1/2), |4p − 3| (p ≥ 1/2)")
    ax.errorbar(ps, om, yerr=se, color="#b5452b", marker="o", ms=5, lw=0,
                elinewidth=1.2, capsize=3, label="ω₁ pilot fit, eq. (232), on 4..4096 (± se)")
    ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
    ax.set(xlabel="p", ylabel="ω", xlim=(0, 1), ylim=(0, None), title="ω₁ against p")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_omega_loglog(res: dict, data_root: Path, tag: str, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    ll = [json.loads((data_root / f"sweep_p{label(p)}_{tag}" / "omega_loglog.json").read_text())
          for p in ps]
    ok = np.array([r["omega"] is not None for r in ll])
    om = np.array([r["omega"] if r["omega"] is not None else np.nan for r in ll])
    se = np.array([r["se"] if r["se"] is not None else np.nan for r in ll])
    bad = np.array([r["p_value"] is not None and r["p_value"] < 0.01 for r in ll])
    pp = np.linspace(0, 1, 801)
    ax.plot(pp, omega_conjecture(pp), color="#888780", lw=1.2,
            label="conjecture: ω = 1 (p < 1/2), |4p − 3| (p ≥ 1/2)")
    ax.errorbar(ps, [res[p]["omega1"]["value"] for p in ps],
                yerr=[res[p]["omega1"]["se"] for p in ps], color="#b5452b", alpha=0.35,
                marker="o", ms=4, lw=0, elinewidth=1, capsize=2,
                label="ω₁ pilot fit, eq. (232), on 4..4096 (± se)")
    for sel, face, lab in ((ok & ~bad, "#1f6f8b", "ω loglog-log: pilot γ_k vs final γ̂ (± se)"),
                           (ok & bad, "none", "ω loglog-log, line rejected (χ² p < 0.01)")):
        if sel.any():
            ax.errorbar(ps[sel], om[sel], yerr=se[sel], color="#1f6f8b", mfc=face, marker="s",
                        ms=6, lw=0, elinewidth=1.2, capsize=3, label=lab)
    ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
    ax.set(xlabel="p", ylabel="ω", xlim=(0, 1), ylim=(0, None), title="ω from the loglog-log plot against p")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_article(res: dict, data_root: Path, tag: str, out_dir: Path, plt) -> None:
    ps = np.array(sorted(res))
    g = np.array([res[p]["answer"]["gamma"] for p in ps])
    lo = np.array([res[p]["answer"]["wilson"]["interval"][0] for p in ps])
    hi = np.array([res[p]["answer"]["wilson"]["interval"][1] for p in ps])
    gt = np.array([gamma_true(p) for p in ps])
    hit = (lo <= gt) & (gt <= hi)
    blue, grey = "#2a6fdb", "#7a7a7a"
    rc = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
          "legend.frameon": False, "axes.grid": True, "grid.color": "#e6e6e6"}
    with plt.rc_context(rc):
        fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        ax.plot([0, 0.75], [0.5, 0.5], color=grey, lw=1.2, label=r"$\gamma=1/2$")
        ax.plot([0.75, 1], [0.5, 1], color=grey, lw=1.2, ls="--", label=r"$\gamma=2p-1$")
        ax.plot(ps, g, "o", color=blue, ms=4, label=r"$\hat\gamma$")
        ax.set(xlabel="$p$", ylabel=r"$\gamma$", xlim=(0, 1))
        ax.legend(loc="upper left")
        ax.set_title(r"(a) $\hat\gamma$ against $p$", fontsize=9, loc="left")
        d = g - gt
        for sel, face, lab in ((hit, blue, "interval contains $\\gamma$"),
                               (~hit, "white", "interval misses $\\gamma$")):
            axd.errorbar(ps[sel], d[sel], yerr=[g[sel] - lo[sel], hi[sel] - g[sel]], fmt="o",
                         color=blue, mfc=face, ms=4, capsize=2, lw=0, elinewidth=1, label=lab)
        axd.axhline(0, color=grey, lw=0.8)
        axd.set_yscale("symlog", linthresh=1e-4)
        axd.set_ylim(top=0.08)
        axd.set(xlabel="$p$", ylabel=r"$\hat\gamma-\gamma$", xlim=(0, 1))
        axd.legend(loc="upper left")
        axd.set_title(r"(b) error, with its 95% interval", fontsize=9, loc="left")
        for a in (ax, axd):
            a.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
        fig.savefig(out_dir / "erw_gamma_vs_p.pdf")
        plt.close(fig)

        # Only the refit with gamma pinned to the final's gamma-hat: the omega1 of B_fs
        # (report.py). The pilot fit and the loglog-log stay in the article's table (user,
        # 2026-11-02). Open where its chi2 warns.
        fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
        pp = np.linspace(0, 1, 801)
        ax.plot(pp, omega_conjecture(pp), color=grey, lw=1.2,
                label=r"conjecture $\omega=\min\{1,|4p-3|\}$")
        rf = [res[p]["answer"]["pilot_refit"] for p in ps]
        om = np.array([r["omega1"] for r in rf])
        se = np.array([r["omega1_se"] for r in rf])
        warn = np.array([r["warn"] for r in rf])
        for sel, face, lab in ((~warn, blue, r"pilot refit $\hat\omega_1$, $\gamma=\hat\gamma$"),
                               (warn, "white", r"pilot refit, $\chi^2$ warning")):
            ax.errorbar(ps[sel], om[sel], yerr=se[sel], fmt="o", color=blue, mfc=face, ms=4.5,
                        capsize=2, elinewidth=1, label=lab)
        ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
        ax.set(xlabel="$p$", ylabel=r"$\omega$", xlim=(0, 1), ylim=(0, None))
        ax.legend(loc="lower left", fontsize=8)
        fig.savefig(out_dir / "erw_omega_vs_p.pdf")
        plt.close(fig)


def plot_aic(res: dict, data_root: Path, tag: str, outs: dict, plt) -> None:
    blue, grey, orange = "#2a78d6", "#888780", "#eb6834"
    rows = []
    for p in sorted(res):
        f = data_root / f"sweep_p{label(p)}_{tag}" / "omega_aic.json"
        rf = json.loads(f.read_text())["range_fit"] if f.exists() else None
        if rf is None or rf["best"] is None:
            print(f"[plot] sweep_p{label(p)}_{tag}: no AIC estimate (omega_aic.json "
                  f"{'gives none' if rf else 'missing'}), left out of the AIC figures", flush=True)
            continue
        rows.append((p, rf))
    if not rows:
        raise SystemExit("no study has omega_aic.json: run src/estimate/omega_aic.py first")
    ps = np.array([p for p, _ in rows])
    rf = [r for _, r in rows]
    om = np.array([r["omega"] for r in rf])
    st = np.array([r["stat"] for r in rf])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    pp = np.linspace(0, 1, 801)
    ax.plot(pp, omega_conjecture(pp), color=grey, lw=1.2,
            label="conjecture: ω = 1 (p < 1/2), |4p − 3| (p ≥ 1/2)")
    ax.errorbar(ps, om, yerr=st, fmt="o", color=blue, ms=6, elinewidth=1.5, capsize=3,
                label="ω̂ AIC fit ± stat (68% profile interval)")
    for p, r in rows:
        b = r["best"]
        ax.text(p, 0.02, f"{b['i_min']}..{b['i_max']}", rotation=90, ha="center", va="bottom",
                fontsize=6.5, color=grey)
    ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
    ax.set(xlabel="p", ylabel="ω", xlim=(0, 1), ylim=(0, 1.25),
           title="ω from the AIC fit a + bk + cρ^(−ωk) against p (grey: chosen range i_min..i_max)")
    ax.title.set_fontsize(10)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outs["omega_aic_vs_p"], dpi=130)
    plt.close(fig)

    g = np.array([r["gamma"] for r in rf])
    gst = np.array([r["gamma_stat"] for r in rf])
    truth = np.array([gamma_true(p) for p in ps])
    err = np.abs(g - truth)
    # the standard method: answer.json's gamma-hat, the log-log OLS on the final (long)
    # run's last m scales, and its replicate se
    std = [res[p]["answer"] for p in ps]
    err_std = np.abs(np.array([a["gamma"] for a in std]) - truth)
    se_std = np.array([a["se"] for a in std])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dx = 0.006
    ax.plot(ps - dx, err, "o", color=blue, ms=7, label="|γ̂_AIC − γ|", zorder=3)
    ax.plot(ps - dx, gst, "_", color=blue, ms=14, mew=2, label="AIC fit: estimated se (stat)")
    ax.plot(ps + dx, err_std, "s", color=orange, ms=6, zorder=3,
            label="|γ̂ − γ|, standard: log-log OLS on the final run's last m scales")
    ax.plot(ps + dx, se_std, "_", color=orange, ms=14, mew=2,
            label="standard: estimated se (replicates)")
    ax.axvline(0.75, color="#c0bfb8", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set(xlabel="p", ylabel="|γ̂ − γ|  and  se", xlim=(0, 1),
           title="|γ̂ − γ| against p: AIC fit vs the standard final-run fit, each with its estimated se\n(γ = 1/2 for p ≤ 3/4, 2p − 1 above; reporting only)")
    ax.title.set_fontsize(10)
    ax.grid(True, color="#e1e0d9", lw=0.6, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outs["gamma_aic_vs_p"], dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="the sweep's tag: studies sweep_p<p>_<tag>")
    ap.add_argument("--data-root", type=Path, default=EXP / "data",
                    help="where the studies live (default: this experiment's data/)")
    ap.add_argument("--omega-loglog", action="store_true",
                    help="also draw omega_loglog_vs_p_<tag>.png from each study's omega_loglog.json")
    ap.add_argument("--article", type=Path, default=None, metavar="DIR",
                    help="also write the article's erw_gamma_vs_p.pdf and erw_omega_vs_p.pdf into DIR")
    ap.add_argument("--aic", action="store_true",
                    help="also draw omega_aic_vs_p_<tag>.png and gamma_aic_vs_p_<tag>.png from "
                         "each study's omega_aic.json (the AIC fit)")
    args = ap.parse_args()

    pat = re.compile(rf"sweep_p([0-9.]+)_{re.escape(args.tag)}")
    res = {}
    for sd in sorted(args.data_root.glob(f"sweep_p*_{args.tag}")):
        m = pat.fullmatch(sd.name)
        if not (sd.is_dir() and m):
            continue
        if not (sd / "answer.json").exists():
            print(f"[plot] {sd.name}: no answer.json (final not drawn), left out", flush=True)
            continue
        res[float(m.group(1))] = load(sd)
    if not res:
        raise SystemExit(f"no finished study sweep_p*_{args.tag} under {args.data_root}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outs = {name: args.data_root / f"{name}_{args.tag}.png"
            for name in ("slopes", "loglog", "gamma_vs_p", "omega_vs_p")}
    plot_slopes(res, outs["slopes"], plt)
    plot_loglog(res, outs["loglog"], plt)
    plot_gamma(res, outs["gamma_vs_p"], plt)
    plot_omega(res, outs["omega_vs_p"], plt)
    if args.omega_loglog:
        outs["omega_loglog_vs_p"] = args.data_root / f"omega_loglog_vs_p_{args.tag}.png"
        plot_omega_loglog(res, args.data_root, args.tag, outs["omega_loglog_vs_p"], plt)
    if args.aic:
        for name in ("omega_aic_vs_p", "gamma_aic_vs_p"):
            outs[name] = args.data_root / f"{name}_{args.tag}.png"
        plot_aic(res, args.data_root, args.tag, outs, plt)
    if args.article:
        args.article.mkdir(parents=True, exist_ok=True)
        plot_article(res, args.data_root, args.tag, args.article, plt)
        print(f"[plot] -> {args.article}/erw_gamma_vs_p.pdf, erw_omega_vs_p.pdf", flush=True)
    print(f"[plot] p = {sorted(res)}", flush=True)
    for o in outs.values():
        print(f"[plot] -> {o}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
