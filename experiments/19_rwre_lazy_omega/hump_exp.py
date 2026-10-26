"""Exponential decay of s_k after a hand-picked hump, n4e6 sweep (exp 19, user 2026-10-15).

    python3 experiments/19_rwre_lazy_omega/hump_exp.py experiments/19_rwre_lazy_omega/images/hump_exp_n4e6.png
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from scipy import optimize, stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data" / "omega_n4e6.json"
OUT = Path(sys.argv[1])
# first k of the fit: one rung past the peak read off the plot (peak itself excluded);
# p <= 0.1 and 0.5 have no hump, so their start is just past the steepest rungs.
START = {"0": 16, "0.05": 16, "0.1": 16, "0.15": 128, "0.2": 256, "0.25": 256,
         "0.3": 512, "0.35": 4096, "0.4": 8192, "0.45": 262144, "0.5": 8}

def cov_s(r: dict) -> np.ndarray:
    # s_j = l_{j+1} - l_j, l = log2 Ybar, independent per scale: tridiagonal covariance
    sig2 = (np.asarray(r["cv"]) ** 2 / (r["n"] * r["R"])) / math.log(2) ** 2
    S = len(r["scales"]); D = np.zeros((S - 1, S))
    D[np.arange(S - 1), np.arange(1, S)], D[np.arange(S - 1), np.arange(S - 1)] = 1, -1
    return D @ np.diag(sig2) @ D.T

def gls(k, s, C, basis, x0_grid):
    """s = g + c*basis(k, theta); profile theta on a grid, then refine; full GLS."""
    L = np.linalg.cholesky(np.linalg.inv(C))
    def prof(th):
        A = np.vstack([np.ones_like(k), basis(k, th)]).T
        Aw, yw = L.T @ A, L.T @ s
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        r = yw - Aw @ coef
        return float(r @ r), coef
    chi = [prof(t)[0] for t in x0_grid]; j = int(np.argmin(chi))
    lo, hi = x0_grid[max(j - 1, 0)], x0_grid[min(j + 1, len(x0_grid) - 1)]
    th = optimize.minimize_scalar(lambda t: prof(t)[0], bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-12}).x
    chi2, (g, c) = prof(th)
    # covariance from the numerical Jacobian at the optimum
    f = lambda p: p[0] + p[1] * basis(k, p[2])
    p = np.array([g, c, th]); J = np.zeros((len(k), 3))
    for i in range(3):
        h = 1e-6 * max(abs(p[i]), 1e-8); dp = np.zeros(3); dp[i] = h
        J[:, i] = (f(p + dp) - f(p - dp)) / (2 * h)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(J.T @ np.linalg.inv(C) @ J)))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    dof = len(k) - 3
    return {"gamma": g, "c": c, "theta": th, "se": se, "chi2": chi2, "dof": dof,
            "p": float(stats.chi2.sf(chi2, dof)) if dof > 0 else None,
            "bound": bool(np.isclose(th, x0_grid[0], rtol=1e-3) or np.isclose(th, x0_grid[-1], rtol=1e-3))}

# (Ek) exponential in k:   basis exp(-k/kappa), theta = log2 kappa
Ek = lambda k, t: np.exp(-k / 2.0 ** t)
# (Ej) exponential in j = log2 k (the plotted axis) = power law k^-omega, theta = lambda
Ej = lambda k, t: np.exp(-t * np.log2(k))

res = json.loads(DATA.read_text())["results"]
rows, fits = [], {}
for p, r in res.items():
    k_all = np.asarray(r["scales"][:-1], float); s_all = np.asarray(r["s"])
    m = k_all >= START[p]; k, s = k_all[m], s_all[m]
    C = cov_s(r)[np.ix_(m, m)]
    if m.sum() < 4:
        rows.append(f"| {p} | {int(k[0])}..{int(k[-1])} ({m.sum()}) | too few points | |"); continue
    a = gls(k, s, C, Ek, np.linspace(0, 22, 441))
    b = gls(k, s, C, Ej, np.geomspace(1e-3, 5, 400))
    fits[p] = (m, a, b)
    fmt = lambda f, th: (f"γ={f['gamma']:.4f}±{f['se'][0]:.4f}, {th}, "
                         f"χ²={f['chi2']:.1f}/{f['dof']}, p={f['p']:.2g}" + (" [bound]" if f["bound"] else ""))
    ta = f"κ=2^{a['theta']:.2f}"
    tb = f"λ={b['theta']:.3f}±{b['se'][2]:.3f} (ω={b['theta'] / math.log(2):.3f})"
    rows.append(f"| {p} | {int(k[0])}..{int(k[-1])} ({m.sum()}) | {fmt(a, ta)} | {fmt(b, tb)} |")
print("| p | fit window (pts) | (Ek) γ + c e^{-k/κ} | (Ej) γ + c e^{-λ log2 k} = γ + c k^{-ω} |")
print("|---|---|---|---|"); print("\n".join(rows))

ps = [p for p in res if p in fits]
fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True); axes = axes.ravel()
for ax, p in zip(axes, res):
    r = res[p]; k = np.asarray(r["scales"][:-1], float)
    ax.errorbar(k, r["s"], yerr=r["se"], fmt="o", ms=3, color="#888", lw=0.8)
    ax.axvline(START[p], color="#ccc", ls=":")
    if p in fits:
        m, a, b = fits[p]; kk = np.geomspace(k[m][0], k[m][-1], 200)
        ax.plot(kk, a["gamma"] + a["c"] * Ek(kk, a["theta"]), "-", color="#d1495b",
                label=f"e^(-k/κ): χ²/dof {a['chi2']:.0f}/{a['dof']}")
        ax.plot(kk, b["gamma"] + b["c"] * Ej(kk, b["theta"]), "--", color="#00798c",
                label=f"e^(-λ log2 k): χ²/dof {b['chi2']:.0f}/{b['dof']}")
        ax.legend(fontsize=7)
    ax.set_xscale("log", base=2); ax.set_title(f"p={p}", fontsize=9); ax.grid(True, color="#e1e0d9", lw=0.6)
axes[-1].axis("off")
fig.suptitle("s_k after the hand-picked hump (dotted: first k fitted); full-covariance GLS")
fig.tight_layout(); fig.savefig(OUT, dpi=110)

ps = [p for p in res if p in fits]
pv = np.array([float(p) for p in ps])
om = np.array([fits[p][2]["theta"] / math.log(2) for p in ps])
ose = np.array([fits[p][2]["se"][2] / math.log(2) for p in ps])
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(pv, om, yerr=ose, fmt="o-", color="#00798c", ms=5, capsize=3)
for x, y, p in zip(pv, om, ps):
    f = fits[p][2]
    ax.annotate(f"χ²/dof {f['chi2']:.0f}/{f['dof']}", (x, y), textcoords="offset points",
                xytext=(5, 6), fontsize=7, color="#555")
ax.set(xlabel="p", ylabel="ω  (λ / ln 2)", title="decay rate after the hump: s_k = γ + c·e^{-λ log2 k} = γ + c·k^{-ω}")
ax.grid(True, color="#e1e0d9", lw=0.6); fig.tight_layout()
fig.savefig(OUT.with_name(OUT.stem + "_omega_vs_p.png"), dpi=130)

fig, ax = plt.subplots(figsize=(9, 6)); cmap = plt.get_cmap("viridis")
allp = list(res)
for t_, p in enumerate(allp):
    r = res[p]; k = np.asarray(r["scales"][:-1], float); c = cmap(t_ / (len(allp) - 1))
    ax.errorbar(k, r["s"], yerr=r["se"], fmt="o", ms=3.5, color=c, alpha=0.55, label=f"p={p}")
    if p in fits:
        m, _, b = fits[p]; kk = np.geomspace(k[m][0], k[m][-1], 200)
        ax.plot(kk, b["gamma"] + b["c"] * Ej(kk, b["theta"]), "-", color=c, lw=1.8)
ax.set_xscale("log", base=2)
ax.set(xlabel="k (lower scale of the pair)", ylabel="s_k",
       title="drawn s_k (± se) and the post-hump fit γ + c·e^{-λ log2 k} (lines)")
ax.grid(True, color="#e1e0d9", lw=0.6); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
fig.savefig(OUT.with_name(OUT.stem + "_fitlines.png"), dpi=130)
