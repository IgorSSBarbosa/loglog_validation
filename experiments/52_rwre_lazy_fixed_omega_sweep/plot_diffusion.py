"""The effective diffusion coefficient D(k) = E X_k^2 / k of rwre_gpu_lazy across p, from data already drawn.

User's conjecture (2026-11-07): the walk is diffusive at every p, with D(0) > 1 = D(1/2), and at
0 < p < 1/2 the walker reads the environment only at rate eps = 1 - 2p, so D takes a time that
diverges as p -> 1/2 to reach its value; that crossover is 16's and 19's hump. The user added
that p = 0 also takes a long time to settle. Checked here as:

  - D(k) per p, and a tail fit D_inf - A k^-theta (theta from the p = 0 tail, then fixed);
  - the collapse D(k, p) = D(k eps^nu, 0): the p-walk as the p = 0 walk on a slower clock,
    with nu fitted per p and the mismatch at nu = 4 (first order in eps^2: D - 1 ~ eps^2 sqrt k);
  - the crossover time tau(p) from the peak of R = E|X| / sqrt(E X^2) and of d ln D / d ln k.

Nothing here is an article estimator; D and R are moments. E X^2 is rebuilt from each replicate's
y_bar and cv (E Y^2 = y_bar^2 (1 + cv^2)); se is over replicates (R = 5: 4 dof, so rough).

Sources, pooled by inverse variance where scales overlap:
  19 omega_p<p>_n4e6 (4..2^18, or 2^20 at p >= 0.35; n = 800,000 x 5),
  52 fixed_p<p>_3h (2^18..2^23), 50 long_p0.3_24h (2^18..2^23, p = 0.3 only).

    python3 experiments/52_rwre_lazy_fixed_omega_sweep/plot_diffusion.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit, minimize_scalar  # noqa: E402
from scipy.stats import chi2 as chi2_dist  # noqa: E402

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
P_GRID = ("0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5")


def sources(p: str) -> list[Path]:
    out = [ROOT / f"experiments/19_rwre_lazy_omega/data/omega_p{p}_n4e6/pilot.json",
           EXP / f"data/fixed_p{p}_3h/final.json"]
    if p == "0.3":
        out.append(ROOT / "experiments/50_rwre_lazy_long_omega/data/long_p0.3_24h/final.json")
    return out


def series(p: str) -> dict:
    """Per scale: D = E X^2/k and R = E|X|/sqrt(E X^2), each with its se over replicates."""
    per_k: dict[float, list] = {}
    for f in sources(p):
        d = json.loads(f.read_text())
        m1 = np.array([r["y_bar"] for r in d["per_replicate"]])
        cv = np.array([r["cv"] for r in d["per_replicate"]])
        m2 = m1**2 * (1 + cv**2)
        for j, k in enumerate(d["scales"]):
            D = m2[:, j] / k
            R = m1[:, j] / np.sqrt(m2[:, j])
            per_k.setdefault(float(k), []).append(
                (D.mean(), D.var(ddof=1) / len(D), R.mean(), R.var(ddof=1) / len(R)))
    k = np.array(sorted(per_k))
    rows = []
    for kk in k:
        a = np.array(per_k[kk])
        wD, wR = 1 / a[:, 1], 1 / a[:, 3]
        rows.append(((a[:, 0] * wD).sum() / wD.sum(), wD.sum() ** -0.5,
                     (a[:, 2] * wR).sum() / wR.sum(), wR.sum() ** -0.5))
    r = np.array(rows)
    return dict(k=k, D=r[:, 0], Dse=r[:, 1], R=r[:, 2], Rse=r[:, 3], eps=1 - 2 * float(p))


def vertex(x: np.ndarray, y: np.ndarray) -> float:
    """Quadratic vertex in x around the argmax of y; nan at the ladder's edges."""
    i = int(np.argmax(y))
    if not 2 <= i <= len(x) - 3:
        return float("nan")
    c = np.polyfit(x[i - 2:i + 3], y[i - 2:i + 3], 2)
    return -c[1] / (2 * c[0])


def tail_fit(s: dict, theta: float | None, kmin: float) -> tuple:
    """D_inf - A k^-theta on k >= kmin. Returns (D_inf, se inflated by sqrt(chi2/dof) if > 1, chi2, dof, theta)."""
    m = s["k"] >= kmin
    lk, y, e = np.log(s["k"][m]), s["D"][m], s["Dse"][m]
    if theta is None:
        f = lambda lk, Di, A, th: Di - A * np.exp(-th * lk)  # noqa: E731
        q, c = curve_fit(f, lk, y, p0=(6, 6, 0.13), sigma=e, absolute_sigma=True, maxfev=40000)
        th = q[2]
    else:
        f = lambda lk, Di, A: Di - A * np.exp(-theta * lk)  # noqa: E731
        q, c = curve_fit(f, lk, y, p0=(6, 6), sigma=e, absolute_sigma=True)
        th = theta
    chi = float((((y - f(lk, *q)) / e) ** 2).sum())
    dof = int(m.sum()) - len(q)
    return q[0], np.sqrt(c[0, 0]) * max(1.0, np.sqrt(chi / dof)), chi, dof, th


def collapse(s: dict, ref: dict, nu: float, kmin: float = 2**6) -> tuple:
    """D(k, p) against the p = 0 curve read at k eps^nu (interpolated in log2 k)."""
    x0 = np.log2(ref["k"])
    x = np.log2(s["k"]) + nu * np.log2(s["eps"])
    m = (x >= x0[0]) & (s["k"] >= kmin)
    pred = np.interp(x[m], x0, ref["D"])
    se = np.hypot(s["Dse"][m], np.interp(x[m], x0, ref["Dse"]))
    return s["D"][m], pred, se


def running_nu(s: dict, ref: dict) -> tuple:
    """nu(k) solving D(k, p) = D(k eps^nu, 0) exactly, per scale (user, 2026-11-07).

    The p = 0 curve is inverted by linear interpolation in log2 k (it is increasing on the
    whole ladder), so log2 k0(D) - log2 k = nu log2 eps. Scales whose D is below the p = 0
    curve's first point (k = 4) cannot be placed and are dropped. se: D's se at p and the
    p = 0 curve's se at k0, divided by the p = 0 slope dD/dlog2 k (a secant over one
    doubling each side, because single doublings near the top are noisy) and by |log2 eps|.
    """
    x0, D0, s0 = np.log2(ref["k"]), ref["D"], ref["Dse"]
    m = (s["D"] >= D0[0]) & (s["D"] <= D0[-1])
    x0_of_D = np.interp(s["D"][m], D0, x0)
    slope = np.interp(x0_of_D, x0[1:-1], (D0[2:] - D0[:-2]) / (x0[2:] - x0[:-2]))
    le = np.log2(s["eps"])
    nu = (x0_of_D - np.log2(s["k"][m])) / le
    se = np.hypot(s["Dse"][m], np.interp(x0_of_D, x0, s0)) / (slope * abs(le))
    return s["k"][m], nu, se


def plot_running_nu(S: dict, out: Path) -> None:
    ref = S["0"]
    ps = [p for p in P_GRID if p not in ("0", "0.5")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    print("\nrunning nu(k): D(k, p) = D(k eps^nu, 0)")
    for ax, p in zip(axes.flat, ps):
        k, nu, se = running_nu(S[p], ref)
        ax.axhline(4, color="0.6", lw=1, ls="--", label=r"$\nu=4$")
        ax.errorbar(np.log2(k), nu, se, fmt="-o", ms=4, lw=1.5, capsize=2, color="#1f4e8c")
        ax.set_title(rf"$p={p}$, $\varepsilon={S[p]['eps']:.2f}$", fontsize=10)
        lo = max(2.0, np.nanmin(nu - se) - 0.3) if len(nu) else 2.0
        ax.set_ylim(min(lo, 3.5), min(7.0, max(5.0, np.nanmax(nu + se) + 0.3)))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        print(f"  p={p:5}: " + "  ".join(f"2^{int(np.log2(kk))}:{v:.2f}±{e:.2f}"
                                         for kk, v, e in zip(k, nu, se) if np.log2(kk) % 3 == 0 or kk == k[-1]))
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log_2 k$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\nu(k)$")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(r"Running collapse exponent: the $\nu(k)$ that puts $D(k,p)$ exactly on the $p=0$ curve, "
                 r"$D(k,p)=D(k\varepsilon^{\nu},0)$, $\varepsilon=1-2p$." "\n"
                 r"Exact collapse would give a flat line. Scales with $D(k,p)<D(4,0)$ have no $p=0$ partner and are absent.",
                 fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def plot_running_dinf(S: dict, theta: float, out: Path, span: int = 6) -> None:
    """D_inf refitted as the ladder grows (user, 2026-11-07), against log2(k_max eps^4).

    At each k_max: D_inf - A k^-theta (theta fixed) on k_max / 2^span .. k_max, se inflated
    by sqrt(chi2/dof) when > 1. The measured D(k_max) is drawn alongside. A converged
    estimate would go flat; the eps^4 axis puts every p on the p = 0 clock.
    """
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    print(f"\nrunning D_inf (theta = {theta:.4f}, window k_max/2^{span}..k_max), at the top k_max:")
    for p in P_GRID[:-1]:
        s = S[p]
        x = np.log2(s["k"])
        xs = x + 4 * np.log2(s["eps"])
        c = cmap(0.35 + 0.6 * float(p) / 0.5)
        pts = []
        for j in range(len(x)):
            if x[j] - span < x[0]:
                continue
            sub = {key: s[key][(x >= x[j] - span) & (x <= x[j])] for key in ("k", "D", "Dse")}
            Di, se, *_ = tail_fit(sub, theta, sub["k"][0])
            pts.append((xs[j], Di, se))
        if not pts:
            continue
        a = np.array(pts)
        for axis in ax:
            axis.plot(xs, s["D"], ":", lw=1.2, color=c)
        ax[0].plot(a[:, 0], a[:, 1], "-o", ms=3, lw=2 if p != "0" else 4, color=c if p != "0" else "0.6",
                   label=f"p={p}", zorder=3 if p == "0" else 2)
        ax[1].errorbar(a[:, 0], a[:, 1], a[:, 2], fmt="-o", ms=3, lw=1.5, capsize=2,
                       color=c if p != "0" else "0.6", label=f"p={p}")
        print(f"  p={p:5}: log2(k_max eps^4) = {a[-1, 0]:5.1f}  D_inf = {a[-1, 1]:.3f} ± {a[-1, 2]:.3f}")
    for axis, t in zip(ax, ("(a) running $D_\\infty$ (dotted: measured $D(k)$)", "(b) the same, with se")):
        axis.set(xlabel=r"$\log_2(k_{\max}\,\varepsilon^4)$,  $\varepsilon=1-2p$", title=t)
        axis.grid(alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel(r"$D_\infty$ fitted on $k\in[k_{\max}/2^{%d},\,k_{\max}]$" % span)
    ax[1].set_ylim(1, 8)
    ax[0].legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    fig.suptitle(rf"How the $D_\infty$ estimate moves as each $p$'s ladder grows: $D_\infty-Ak^{{-{theta:.4f}}}$, "
                 rf"$\theta$ fixed from the $p=0$ tail. A converged estimate goes flat.", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=EXP / "data/diffusion_coefficient_vs_p.png")
    ap.add_argument("--out-running", type=Path, default=EXP / "data/running_nu.png")
    ap.add_argument("--out-dinf", type=Path, default=EXP / "data/running_dinf.png")
    args = ap.parse_args()

    S = {p: series(p) for p in P_GRID}
    ref = S["0"]
    plot_running_nu(S, args.out_running)

    th0 = tail_fit(ref, None, 2**10)[4]
    plot_running_dinf(S, th0, args.out_dinf)
    print(f"p = 0 tail, k >= 2^10, theta free: theta = {th0:.4f}")
    lk, m = np.log(ref["k"]), ref["k"] >= 2**10
    for name, f, p0 in (("a + c ln k", lambda lk, a, c: a + c * lk, (1, 0.3)),
                        ("A k^g", lambda lk, A, g: A * np.exp(g * lk), (2, 0.05))):
        q, _ = curve_fit(f, lk[m], ref["D"][m], p0=p0, sigma=ref["Dse"][m], absolute_sigma=True)
        chi = (((ref["D"][m] - f(lk[m], *q)) / ref["Dse"][m]) ** 2).sum()
        print(f"  alternative {name}: chi2 {chi:.0f}/{m.sum() - 2}")

    print(f"\n p     D(2^23)          D_inf (theta={th0:.4f}, k>=2^12)  chi2/dof   "
          "log2 tau_R  log2 tau_b  collapse nu=4: max|dev|, best nu")
    rows = []
    for p in P_GRID:
        s = S[p]
        if p == "0.5":
            w = s["Dse"] ** -2
            Di, se = (s["D"] * w).sum() / w.sum(), w.sum() ** -0.5
            chi, dof = float((((s["D"] - Di) / s["Dse"]) ** 2).sum()), len(s["D"]) - 1
        else:
            Di, se, chi, dof, _ = tail_fit(s, th0, 2**12)
        x = np.log2(s["k"])
        b = np.diff(np.log(s["D"])) / np.diff(np.log(s["k"]))
        tR = vertex(x, s["R"]) if p != "0.5" else float("nan")
        tb = vertex((x[1:] + x[:-1]) / 2, b) if p != "0.5" else float("nan")
        col = ""
        if p not in ("0", "0.5"):
            obs, pred, _ = collapse(s, ref, 4.0)
            nu = minimize_scalar(lambda n: (((lambda a: (a[0] - a[1]) / a[2])(collapse(s, ref, n))) ** 2).sum(),
                                 bounds=(2, 7), method="bounded").x
            col = f"{np.abs(obs / pred - 1).max() * 100:4.1f}%, {nu:.2f}"
        rows.append((float(p), s["D"][-1], s["Dse"][-1], Di, se, chi2_dist.sf(chi, dof), tR, tb))
        print(f"{p:5} {s['D'][-1]:.3f}±{s['Dse'][-1]:.3f}   {Di:.3f}±{se:.3f}"
              f"                   {chi:6.0f}/{dof:<3}  {tR:6.2f}      {tb:6.2f}      {col}")
    R = np.array(rows)
    for j, name in ((6, "tau_R"), (7, "tau_b")):
        m = np.isfinite(R[:, j]) & (R[:, 0] >= 0.2) & (R[:, 0] < 0.5)
        c, cov = np.polyfit(np.log2(1 - 2 * R[m, 0]), R[m, j], 1, cov=True)
        print(f"nu from {name}, 0.2 <= p < 0.5: {-c[0]:.2f} ± {np.sqrt(cov[0, 0]):.2f}")

    cmap = plt.get_cmap("Blues")
    col = {p: cmap(0.35 + 0.6 * float(p) / 0.5) for p in P_GRID}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for p in P_GRID:
        ax[0].loglog(S[p]["k"], S[p]["D"] * S[p]["k"], "-o", ms=3, lw=2, color=col[p], label=f"p={p}")
    kk = np.array([4, 2**23])
    ax[0].loglog(kk, kk, "k:", lw=1, label=r"$k$ (SRW, $D=1$)")
    ax[0].set_xscale("log", base=2)
    ax[0].set(xlabel="$k$", ylabel=r"$\mathbb{E}X_k^2$", title=r"(a) squared distance $\mathbb{E}X_k^2$")
    ax[0].legend(fontsize=8, ncol=2, frameon=False)

    ok = R[:, 5] > 1e-3
    ax[1].errorbar(R[:, 0], R[:, 1], R[:, 2], fmt="s", ms=7, color="#7aa6d6", capsize=3,
                   label=r"measured $D(k)=\mathbb{E}X_k^2/k$ at $k=2^{23}$")
    ax[1].errorbar(R[ok, 0], R[ok, 3], R[ok, 4], fmt="o", ms=8, color="#1f4e8c", capsize=3,
                   label=rf"fitted $D_\infty$: $D_\infty-Ak^{{-{th0:.4f}}}$, $k\geq2^{{12}}$, fit accepted")
    ax[1].errorbar(R[~ok, 0], R[~ok, 3], R[~ok, 4], fmt="o", ms=8, mfc="none", color="#1f4e8c", capsize=3,
                   label=r"same fit, rejected ($\chi^2$ p < $10^{-3}$)")
    ax[1].axhline(1, color="gray", lw=1, ls=":")
    ax[1].set(xlabel="$p$", ylabel="diffusion coefficient $D$", title=r"(b) diffusion coefficient vs $p$", ylim=(0, 7))
    ax[1].legend(fontsize=8, frameon=False, loc="lower left")

    for p in P_GRID[:-1]:
        x = np.log2(S[p]["k"]) + 4 * np.log2(S[p]["eps"])
        if p == "0":
            ax[2].plot(x, S[p]["D"], "-", lw=5, color="0.75", zorder=0, label="p=0 (reference)")
        else:
            ax[2].plot(x, S[p]["D"], "-o", ms=3, lw=1.5, color=col[p], label=f"p={p}")
    ax[2].axhline(1, color="gray", lw=1, ls=":")
    ax[2].set(xlabel=r"$\log_2(k\,\varepsilon^4)$,  $\varepsilon=1-2p$", ylabel=r"$D(k)=\mathbb{E}X_k^2/k$",
              title=r"(c) collapse: $D(k,p)\approx D(k\varepsilon^4,0)$")
    ax[2].legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"rwre_gpu_lazy, $\alpha=1/2$: exp 19 (n4e6) + 52 (3h) + 50 ($p=0.3$), pooled. "
                 rf"$\theta={th0:.4f}$ from the $p=0$ tail, fixed for every $p$", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
