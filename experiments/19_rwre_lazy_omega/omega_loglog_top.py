"""omega from the loglog-log plot of exp 19's pilot-only sweeps, and omega against p.

The method is `tools/loglog_log.py`'s (user, 2026-10-16). This experiment's studies are
`omega_sweep.py`'s pilots on 4..K with no final run, so the general driver
(`src/estimate/omega_loglog.py`, gamma-hat from answer.json) doesn't apply. The user
made it a separate case (2026-10-16):

  - gamma-hat = the slope of the last two points, gamma_{K-1} = (log Ybar_K -
    log Ybar_{K/2}) / log 2. It is the same draw as the gamma_k, so D_k = gamma_k -
    gamma-hat is one linear map of the scale means and its covariance is exact
    (`loglog_log.differences` with gamma_weights). The top pair is gamma-hat itself.
  - The window is the tail after the peak: pairs from the peak up to the one below the
    top. 16 and 19 show a hump in gamma_k at p = 0.2-0.45; before its peak the decay
    isn't eq. (232)'s. The peak is the argmax of gamma_k after its LAST resolved rise,
    gamma_j - gamma_{j-1} > LL.RESOLVED_Z se (user, 2026-10-16). With no resolved rise
    (a monotone gamma_k, or a hump the data don't resolve) the window is the whole
    ladder. Two rules were tried first and dropped: the global argmax is the first pair
    at p = 0.2-0.25; "the last gamma_k above both neighbours by > 2 se" can't see a
    hump's flat top (z = 0.0 between 2^10 and 2^11 at p = 0.35) and fired on a noise
    blip at p = 1/2, 2^17.
  - Inside the window, resolved points with the majority sign, as in the general case.

Per study, writes omega_loglog_top.{json,png} into it. Over the sweep, writes
data/omega_loglog_vs_p_<tag>.{png,md}: omega ± se against p, next to the pilot's eq.
(232) omega1 where that fit converged.

    python3 experiments/19_rwre_lazy_omega/omega_loglog_top.py --tag n4e6
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools import loglog_log as LL  # noqa: E402
from src.estimate.omega_loglog import pilot_means  # noqa: E402

PRODUCED_BY = "experiments/19_rwre_lazy_omega/omega_loglog_top.py"
_BLUE, _RED, _MUTED = "#2a78d6", "#b5452b", "#898781"


def label(p: float) -> str:
    return f"{p:.4g}"


def peak_after_last_rise(s, cov) -> int:
    """Argmax of gamma_k (top pair excluded) from its last rise resolved at RESOLVED_Z se."""
    start = 0
    for j in range(1, len(s) - 1):
        se = math.sqrt(cov[j, j] + cov[j - 1, j - 1] - 2 * cov[j, j - 1])
        if s[j] - s[j - 1] > LL.RESOLVED_Z * se:
            start = j
    return start + int(np.argmax(s[start:-1]))


def analyse(sd: Path) -> dict:
    pilot = read_artifact(sd, "pilot")
    scales, L, var = pilot_means(pilot)
    rho = LL.ladder_ratio(scales)
    g = np.zeros(len(scales))
    g[-2], g[-1] = -1 / math.log(rho), 1 / math.log(rho)
    J = LL.slope_matrix(len(scales), rho)
    peak = peak_after_last_rise(J @ L, J @ np.diag(var) @ J.T)
    window = np.zeros(len(scales) - 1, bool)
    window[peak:-1] = True
    res = LL.omega_loglog(scales, L, var, gamma_weights=g, window=window)
    return {**res, "study": sd.name, "peak_lower_scale": int(scales[peak]),
            "window": window.tolist(),
            "gamma_hat_source": f"pilot.json: slope of the top pair {scales[-2]}..{scales[-1]}",
            "gamma_k_source": "pilot.json per_replicate: consecutive slopes",
            "pilot_scales": scales, "pilot_n": pilot["n"],
            "pilot_replicates": len(pilot["per_replicate"]),
            "omega1_pilot_fit": pilot["direct_fit"]["omega1"],
            "omega1_pilot_fit_converged": pilot["direct_fit"]["converged"]}


def fit_label(res: dict) -> str | None:
    if res["omega"] is None:
        return None
    return f"ω = {res['omega']:.3f} ± {res['se']:.3f}, χ²/dof {res['chi2']:.1f}/{res['dof']}"


def plot_study(res: dict, pilot: dict, out: Path, plt) -> None:
    rho = res["rho"]
    lr = math.log(rho)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.log(pilot["scales"]) / lr
    a.errorbar(x, np.log(pilot["y_bar"]) / lr, yerr=np.asarray(pilot["sigma_log"]) / lr,
               color=_BLUE, marker="o", ms=4, lw=0, elinewidth=1, label="pilot")
    xx = np.linspace(x[0], x[-1], 50)
    y_top = np.log(pilot["y_bar"][-1]) / lr
    a.plot(xx, y_top + res["gamma_hat"] * (xx - x[-1]), color=_MUTED, lw=1, ls="--",
           label=f"top-pair slope γ̂ = {res['gamma_hat']:.4f}")
    a.set(xlabel=f"log_{rho:g} i", ylabel=f"log_{rho:g} Ȳ_i", title="(a) log-log plot")
    a.legend(fontsize=8)
    lab = fit_label(res)
    LL.plot_panel(b, res, _BLUE, lab)
    b.axvline(math.log(res["peak_lower_scale"]) / lr, color=_MUTED, lw=0.6, ls=":")
    b.set(xlabel=f"log_{rho:g} k (lower scale of the pair)", ylabel=f"log_{rho:g} |γ_k − γ̂|",
          title="(b) loglog-log plot: filled = fitted, dotted = window start"
                + ("" if lab else f"\nno fit: {res['why_none']}"))
    if lab:
        b.legend(fontsize=8)
    for ax in (a, b):
        ax.grid(True, color="#e1e0d9", lw=0.6)
    fig.suptitle(res["study"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_sweep(res: dict, tag: str, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    conv = np.array([res[p]["omega1_pilot_fit_converged"] for p in ps])
    om1 = np.array([res[p]["omega1_pilot_fit"] for p in ps])
    if conv.any():
        ax.plot(ps[conv], om1[conv], "o", color=_RED, alpha=0.4, ms=5,
                label="ω₁ pilot fit, eq. (232), whole ladder (converged only)")
    ok = np.array([res[p]["omega"] is not None for p in ps])
    bad = np.array([res[p]["p_value"] is not None and res[p]["p_value"] < 0.01 for p in ps])
    om = np.array([res[p]["omega"] if res[p]["omega"] is not None else np.nan for p in ps])
    se = np.array([res[p]["se"] if res[p]["se"] is not None else np.nan for p in ps])
    for sel, face, lab in ((ok & ~bad, "#1f6f8b", "ω loglog-log: γ_k vs top-pair γ̂, tail after peak (± se)"),
                           (ok & bad, "none", "ω loglog-log, line rejected (χ² p < 0.01)")):
        if sel.any():
            ax.errorbar(ps[sel], om[sel], yerr=se[sel], color="#1f6f8b", mfc=face, marker="s",
                        ms=6, lw=0, elinewidth=1.2, capsize=3, label=lab)
    for p in ps[~ok]:
        ax.annotate("no fit", (p, 0.02), ha="center", fontsize=7, color=_MUTED)
    ax.set(xlabel="p", ylabel="ω", xlim=(-0.02, 0.52), ylim=(0, None),
           title=f"ω from the loglog-log plot against p (tag {tag})")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def table(res: dict, tag: str) -> str:
    lines = [f"# omega from the loglog-log plot, exp 19, tag `{tag}`", "",
             "gamma-hat = slope of the top pair; fit on resolved points (|D| > "
             f"{LL.RESOLVED_Z:g} sd) with the majority sign, from the argmax of gamma_k after its last rise resolved at that z, up "
             "(`omega_loglog_top.py`). GLS on the delta-method covariance.", "",
             "| p | ladder | gamma-hat ± se | peak of γ_k | points | ω ± se | χ²/dof | χ² p | eq. (232) ω₁ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(res):
        r = res[p]
        om = (f"{r['omega']:.3f} ± {r['se']:.3f}", f"{r['chi2']:.1f}/{r['dof']}", f"{r['p_value']:.2g}") \
            if r["omega"] is not None else (f"none ({r['why_none']})", "–", "–")
        o1 = f"{r['omega1_pilot_fit']:.3f}" + ("" if r["omega1_pilot_fit_converged"] else " (not conv.)")
        lines.append(f"| {label(p)} | {r['pilot_scales'][0]}..{r['pilot_scales'][-1]} | "
                     f"{r['gamma_hat']:.4f} ± {r['gamma_hat_se']:.4f} | {r['peak_lower_scale']} | "
                     f"{r['n_points']} | {om[0]} | {om[1]} | {om[2]} | {o1} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="the sweep's tag: studies omega_p<p>_<tag>")
    ap.add_argument("--data-root", type=Path, default=EXP / "data",
                    help="where the studies live (default: this experiment's data/)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pat = re.compile(rf"omega_p([0-9.]+)_{re.escape(args.tag)}")
    res = {}
    for sd in sorted(args.data_root.glob(f"omega_p*_{args.tag}")):
        m = pat.fullmatch(sd.name)
        if not (sd.is_dir() and m):
            continue
        r = analyse(sd)
        write_artifact(sd, "omega_loglog_top", r, produced_by=PRODUCED_BY)
        plot_study(r, read_artifact(sd, "pilot"), artifact_path(sd, "omega_loglog_top_figure"), plt)
        res[float(m.group(1))] = r
    if not res:
        raise SystemExit(f"no study omega_p*_{args.tag} under {args.data_root}")
    fig_out = args.data_root / f"omega_loglog_vs_p_{args.tag}.png"
    md_out = args.data_root / f"omega_loglog_vs_p_{args.tag}.md"
    plot_sweep(res, args.tag, fig_out, plt)
    text = table(res, args.tag)
    md_out.write_text(text)
    print(text)
    print(f"[omega_loglog_top] -> {fig_out}, {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
