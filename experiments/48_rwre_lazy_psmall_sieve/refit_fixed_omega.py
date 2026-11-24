"""The long run refitted by eq. (pilot-fit) with omega1 held at the pilot's value.

Asked by the user (2026-11-02), after the sieve's log-log plot showed the long run
falling below the pilot's extrapolated fit: what gamma does the long run give when
the correction's exponent is the pilot's and only gamma, a0 and a1 are fitted to the
long run's own points?

With omega1 fixed, eq. (pilot-fit) in natural logs,

    ln Ybar_i = ln a0 + gamma ln i + a1 i**-omega1,

is linear in (ln a0, gamma, a1), so it is weighted least squares with weights
1/sigma_log**2, in closed form: no starts, no convergence question. Fitted on the
long run's 6 scales (final.json):
  - gamma, a0, a1 on the pooled Ybar (the replicate means averaged, as
    report.analyse does) with its chi2;
  - its error: the same fit on each replicate, sd/sqrt(R) -- independent of whether
    the model is right, unlike the WLS covariance, which is also printed.

Writes, per study, `refit_fixed_omega.png`: (a) log2 Ybar of the pilot and the long
run, the pilot's own fit (dashed) and the refit on the long run (solid), with the
refit's asymptote gamma*i + log2 a0; (b) the long run's residuals from the refit, in
se units. And `gamma_vs_p_refit_3h.png` in the data root: the refitted gamma against
p, next to the long run's gamma-hat with its Wilson 95% interval (faded). Nothing is
drawn (no samples); no truth is plotted, gamma = 1/2 is a reference only.

    python3 experiments/48_rwre_lazy_psmall_sieve/refit_fixed_omega.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_sieve import GAMMA, PILOT, REF, studies  # noqa: E402

REFIT = "#2a6fdb"


def wls(scales, y_bar, sigma_log, omega1: float) -> dict:
    """ln a0 + gamma ln i + a1 i**-omega1 by WLS on ln y_bar. Closed form."""
    i = np.asarray(scales, float)
    X = np.column_stack([np.ones_like(i), np.log(i), i ** -omega1])
    y = np.log(np.asarray(y_bar, float))
    w = 1.0 / np.asarray(sigma_log, float) ** 2
    A = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    cov = np.linalg.inv(A)
    r = (y - X @ beta) / np.asarray(sigma_log, float)
    return {"ln_a0": float(beta[0]), "gamma": float(beta[1]), "a1": float(beta[2]),
            "se_wls": float(math.sqrt(cov[1, 1])), "chi2": float(r @ r),
            "dof": len(i) - 3, "resid": r.tolist()}


def refit(sd: Path) -> dict:
    final = json.loads((sd / "final.json").read_text())
    answer = json.loads((sd / "answer.json").read_text())
    om = json.loads((sd / "constants.json").read_text())["omega1"]
    scales, reps = final["scales"], final["per_replicate"]
    w1 = om["value"]
    pooled = wls(scales, answer["y_bar"], answer["sigma_log"], w1)
    per_rep = [wls(scales, r["y_bar"], r["sigma_log"], w1)["gamma"] for r in reps]
    from scipy.stats import chi2
    pooled.update(omega1=w1, omega1_se=om["se"], R=len(reps), per_rep=per_rep,
                  se_rep=float(np.std(per_rep, ddof=1) / math.sqrt(len(reps))),
                  p_value=float(chi2.sf(pooled["chi2"], pooled["dof"])))
    return pooled


def plot_study(sd: Path, p: float, fit: dict, plt) -> Path:
    pilot = json.loads((sd / "pilot.json").read_text())
    ans = json.loads((sd / "answer.json").read_text())
    pf = pilot["direct_fit"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for d, mk, c, lab in ((pilot, "o", PILOT, "pilot"), (ans, "s", REFIT, "long run")):
        ax.errorbar(np.log2(d["scales"]), np.log2(d["y_bar"]),
                    yerr=np.asarray(d["sigma_log"]) / math.log(2),
                    color=c, marker=mk, ms=4, lw=0, elinewidth=1, label=f"{lab} (± se)")
    ip, ifin = np.log2(pilot["scales"]), np.log2(ans["scales"])
    ii = np.linspace(ip[0] - 0.3, ifin[-1] + 0.3, 400)
    ln2 = math.log(2)

    def pilot_f(k):
        return pf["gamma"] * k + math.log2(pf["a0"]) + pf["a1"] / ln2 * 2.0 ** (-pf["omega1"] * k)

    def refit_f(k):
        return fit["gamma"] * k + fit["ln_a0"] / ln2 + fit["a1"] / ln2 * 2.0 ** (-fit["omega1"] * k)

    ax.plot(ii, pilot_f(ii), color=PILOT, lw=1, ls="--",
            label=f"pilot fit: γ={pf['gamma']:.4f}, ω₁={pf['omega1']:.3f}")
    jj = np.linspace(ifin[0] - 0.3, ifin[-1] + 0.3, 200)
    ax.plot(jj, refit_f(jj), color=REFIT, lw=1.6,
            label=f"long-run refit, ω₁ fixed at the pilot's: γ={fit['gamma']:.4f} "
                  f"± {fit['se_rep']:.4f}")
    ax.plot(ii, fit["gamma"] * ii + fit["ln_a0"] / ln2, color=REF, lw=0.8, ls=":",
            label="refit asymptote γ i + log₂ a₀")
    ax.set(ylabel=r"$\log_2\overline{Y}_i$",
           title=f"RWRE p = {p:.4g}: long run refitted with ω₁ = {fit['omega1']:.3f} "
                 f"(pilot)  —  χ² = {fit['chi2']:.1f}/{fit['dof']}, p = {fit['p_value']:.2g}")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, loc="upper left")

    axr.axhline(0, color=REF, lw=0.8)
    axr.plot(ifin, fit["resid"], "s", color=REFIT, ms=5)
    axr.set(xlabel=r"$i=\log_2(k)$", ylabel="residual / se")
    axr.grid(True, color="#e1e0d9", lw=0.6)
    fig.tight_layout()
    out = sd / "refit_fixed_omega.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_gamma(res: dict, answers: dict, out: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ps = np.array(sorted(res))
    g = np.array([answers[p]["gamma"] for p in ps])
    lo, hi = np.array([answers[p]["wilson"]["interval"] for p in ps]).T
    ax.axhline(0.5, color=REF, lw=1.2, ls="--", label="γ = 1/2 (reference, not a known value)")
    ax.errorbar(ps, g, yerr=[g - lo, hi - g], color=GAMMA, alpha=0.35, marker="o", ms=4,
                lw=0, elinewidth=1, capsize=2, label="γ̂ (long run, m-point), Wilson 95%")
    ax.errorbar(ps + 0.002, [res[p]["gamma"] for p in ps], yerr=[res[p]["se_rep"] for p in ps],
                color=REFIT, marker="s", ms=5, lw=0, elinewidth=1.2, capsize=3,
                label="γ, eq. (pilot-fit) on the long run, ω₁ fixed at the pilot's "
                      "(± replicate sd/√R)")
    ax.set(xlabel="p", ylabel="γ", xticks=ps,
           title="RWRE: γ from the long run refitted with the pilot's ω₁")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()
    sds = studies(a.data_root)
    if not sds:
        raise SystemExit(f"no finished study sieve_p*_3h* under {a.data_root}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, answers = {}, {}
    for p, sd in sds.items():
        res[p] = fit = refit(sd)
        answers[p] = json.loads((sd / "answer.json").read_text())
        out = plot_study(sd, p, fit, plt)
        print(f"p={p:<5} omega1={fit['omega1']:.4f}  gamma={fit['gamma']:.5f} "
              f"± {fit['se_rep']:.5f} (rep) ± {fit['se_wls']:.5f} (wls)  a1={fit['a1']:+.4g}  "
              f"chi2={fit['chi2']:.1f}/{fit['dof']} p={fit['p_value']:.2g}  "
              f"[gamma-hat {answers[p]['gamma']:.5f}]  -> {out}", flush=True)
    out = a.data_root / "gamma_vs_p_refit_3h.png"
    plot_gamma(res, answers, out, plt)
    print(f"[plot] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
