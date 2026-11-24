"""Figures of the RWRE sieve against p, in the style of experiments/14_erw_urn_gpu/plot_sweep.py.

Asked by the user (2026-11-02): the ERW sieve's gamma and omega figures, for this
sieve. Nothing is drawn; everything is read from each study's answer.json,
constants.json and omega_loglog.json.

  - slopes_3h.png: the local slopes s_k (± se) of local_slope.json (src/study/
    local_slope.py: every scale 4..top at the long run's n, so one curve per p, no
    pilot/final split as in plot_sweep.py's), placed at the pair's lower scale; (a)
    the whole ladder, (b) the top from 2^12, with gamma = 1/2 as a reference.
  - loglog_3h.png: log2 Ybar_i (± se) of the pilot (4..1024) and the long run, with
    the pilot's eq. (pilot-fit) fit, solid on its scales and dashed where it is
    extrapolated over the long run's (plot_sweep.py's `loglog`).
  - gamma_vs_p_3h.png: the long run's gamma_hat with its eq. (wilson) 95% interval.
    No truth line: none is proven for p != 1/2 (ground rule 4); gamma = 1/2 is drawn
    as a reference only.
  - omega_vs_p_3h.png: the pilot's omega1 +/- se (constants.json, eq. (pilot-fit) on
    4..1024), the one that sized each plan and entered the Wilson bound.
  - omega_loglog_vs_p_3h.png: each study's omega_loglog.json (`src/estimate/
    omega_loglog.py`, the loglog-log plot: pilot gamma_k against the long run's
    gamma-hat) +/- se, the long run's omega, next to the pilot's (faded). Run that
    driver on every study first; a missing file raises.

Each p's study is `data/sieve_p<p>_3h[_cap<c>]`; with a cap fallback, the largest cap
that finished (run_p.sh).

    python3 experiments/48_rwre_lazy_psmall_sieve/plot_sieve.py
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

#: plot_sweep.py's colours, so the two sieves read alike.
GAMMA, PILOT, LOGLOG, REF = "#1f6f8b", "#b5452b", "#1f6f8b", "#888780"


def studies(data: Path) -> dict[float, Path]:
    out: dict[float, tuple[int, Path]] = {}
    for sd in data.glob("sieve_p*_3h*"):
        mt = re.fullmatch(r"sieve_p([0-9.]+)_3h(?:_cap(\d+))?", sd.name)
        if not mt or not (sd / "answer.json").exists():
            continue
        p, cap = float(mt.group(1)), int(mt.group(2) or 0)
        if p not in out or cap > out[p][0]:
            out[p] = (cap, sd)
    return {p: sd for p, (_, sd) in sorted(out.items())}


def load(sd: Path) -> dict:
    return {"pilot": json.loads((sd / "pilot.json").read_text()),
            "answer": json.loads((sd / "answer.json").read_text()),
            "omega1": json.loads((sd / "constants.json").read_text())["omega1"],
            "loglog": json.loads((sd / "omega_loglog.json").read_text()),
            "slopes": json.loads((sd / "local_slope.json").read_text())["analysis"]}


def _finish(fig, ax, out: Path, ps: np.ndarray, **kw) -> None:
    ax.set(xlabel="p", xticks=ps, **kw)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_slopes(res: dict, out: Path, plt) -> None:
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    cmap = plt.get_cmap("viridis")
    ps = sorted(res)
    for ax, lo, title in zip(axes, (0, 12), ("(a) the whole ladder", "(b) from 2¹²")):
        for t, p in enumerate(ps):
            a, c = res[p]["slopes"], cmap(t / max(len(ps) - 1, 1))
            k = np.log2(a["scales"][:-1])
            sel = k >= lo
            ax.errorbar(k[sel], np.asarray(a["s"])[sel], yerr=np.asarray(a["se_s"])[sel],
                        color=c, marker="o", ms=4, lw=0.8, elinewidth=1, capsize=2,
                        label=f"p={p:.4g}")
        ax.axhline(0.5, color=REF, lw=1.2, ls="--", label="γ = 1/2 (reference)")
        ax.set(xlabel=r"$\log_2 k$ (lower scale of the pair)", ylabel=r"$s_k$", title=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))   # log2 k is an integer (user, 2026-11-03)
        ax.grid(True, color="#e1e0d9", lw=0.6)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.suptitle(r"RWRE: local slopes $s_k=\log_2(\overline{Y}_{2k}/\overline{Y}_k)$ "
                 r"(± se), local_slope.py at the long run's n")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_loglog(res: dict, out: Path, plt) -> None:
    # Wide, legend outside the axes, as plot_sweep.py's (user, 2026-11-02).
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
                label=f"p={p:.4g}: γ̂={ans['gamma']:.4f}, "
                      f"ω₁={f['omega1']:.3f}±{r['omega1']['se']:.3f}")
        jj = np.linspace(ip[-1], ifin[-1] + 0.3, 200)
        ax.plot(jj, f_i(jj), color=c, lw=0.8, ls="--")
    ax.set(xlabel=r"$i=\log_2(k)$", ylabel=r"$\log_2\overline{Y}_i$",
           title=r"RWRE: $\log_2\overline{Y}_i$ (± se), pilot ● long run ■, and the "
                 r"pilot's $f(i)=\gamma i+a_0+a_1 2^{-\omega_1 i}$")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_gamma(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    g = np.array([res[p]["answer"]["gamma"] for p in ps])
    lo, hi = np.array([res[p]["answer"]["wilson"]["interval"] for p in ps]).T
    ax.axhline(0.5, color=REF, lw=1.2, ls="--", label="γ = 1/2 (reference, not a known value)")
    ax.errorbar(ps, g, yerr=[g - lo, hi - g], color=GAMMA, marker="o", ms=5, lw=0,
                elinewidth=1.2, capsize=3, label="γ̂ (long run), Wilson 95%")
    _finish(fig, ax, out, ps, ylabel="γ", title="RWRE: γ̂ against p")


def plot_omega(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    ax.errorbar(ps, [res[p]["omega1"]["value"] for p in ps],
                yerr=[res[p]["omega1"]["se"] for p in ps], color=PILOT, marker="o", ms=5,
                lw=0, elinewidth=1.2, capsize=3,
                label="ω₁ pilot fit, eq. (pilot-fit), on 4..1024 (± se)")
    _finish(fig, ax, out, ps, ylabel="ω", ylim=(0, None), title="RWRE: ω₁ (pilot) against p")


def plot_omega_loglog(res: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    ll = [res[p]["loglog"] for p in ps]
    ok = np.array([r["omega"] is not None for r in ll])
    om = np.array([r["omega"] if r["omega"] is not None else np.nan for r in ll])
    se = np.array([r["se"] if r["se"] is not None else np.nan for r in ll])
    bad = np.array([r["p_value"] is not None and r["p_value"] < 0.01 for r in ll])
    ax.errorbar(ps, [res[p]["omega1"]["value"] for p in ps],
                yerr=[res[p]["omega1"]["se"] for p in ps], color=PILOT, alpha=0.35,
                marker="o", ms=4, lw=0, elinewidth=1, capsize=2,
                label="ω₁ pilot fit, eq. (pilot-fit), on 4..1024 (± se)")
    for sel, face, lab in ((ok & ~bad, LOGLOG, "ω loglog-log: pilot γ_k vs long-run γ̂ (± se)"),
                           (ok & bad, "none", "ω loglog-log, line rejected (χ² p < 0.01)")):
        if sel.any():
            ax.errorbar(ps[sel], om[sel], yerr=se[sel], color=LOGLOG, mfc=face, marker="s",
                        ms=6, lw=0, elinewidth=1.2, capsize=3, label=lab)
    _finish(fig, ax, out, ps, ylabel="ω", ylim=(0, None),
            title="RWRE: ω from the loglog-log plot (long run) against p")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=EXP / "data",
                    help="where the studies live, and where the figures go "
                         "(copy to images/ by hand, ground rule 6)")
    a = ap.parse_args()
    res = {p: load(sd) for p, sd in studies(a.data_root).items()}
    if not res:
        raise SystemExit(f"no finished study sieve_p*_3h* under {a.data_root}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outs = {name: a.data_root / f"{name}_3h.png"
            for name in ("slopes", "loglog", "gamma_vs_p", "omega_vs_p", "omega_loglog_vs_p")}
    plot_slopes(res, outs["slopes"], plt)
    plot_loglog(res, outs["loglog"], plt)
    plot_gamma(res, outs["gamma_vs_p"], plt)
    plot_omega(res, outs["omega_vs_p"], plt)
    plot_omega_loglog(res, outs["omega_loglog_vs_p"], plt)
    print(f"[plot] p = {sorted(res)}", flush=True)
    for o in outs.values():
        print(f"[plot] -> {o}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
