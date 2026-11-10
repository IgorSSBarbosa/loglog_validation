"""Is the aic_explore_d4 pilot's flat log-log a lucky ladder? Tables and figures of the
ladder sweep (user, 2026-11-07).

The pilot draws S(p) = E|C(0)| at p = p_c - eps, eps = 0.09 / x, x = 4..512, and its
straight-line fit has flat residuals. The sweep redraws S on seven other ladders
eps = eps0 * x**(-1/k) (`ladder_exponent` k in models/percolation_susceptibility_explore.py),
same x, one independent replicate each (recipes/samples_ladder_<name>.json, data/ladder/<name>).
On a ladder with exponent k, log Ybar has slope gamma/k in log x, so every slope below is
reported times k. Nothing is drawn here; everything is read from the runs and the pilot.

Printed: per ladder, the weighted straight-line gamma, its chi2 and p, and the AIC omega
range fit (`tools/local_slope.omega_ranges`); then the homogeneity chi2 of those gammas and
the pooled eps-space fits

    log S = log A_c - gamma log eps [+ a eps^Delta],   one A_c per ladder,

with Delta FIXED (the data cannot fit it). Delta defaults to 0.7, inside the literature's
omega nu = 0.64..0.84 (README, "Known values"): a diagnostic, not an estimator input.

Figures, written to data/<tag>/ (copy the ones to keep into images/, ground rule 6):
  - ladder_local_slopes.png: each ladder's local slopes (x k, CLT se) against the rung's
    mean eps, one panel per ladder, with the pooled fit's prediction gamma - a Delta eps^Delta.
  - ladder_gamma_vs_window.png: each ladder's straight-line gamma against the geometric
    mean eps of its window, beside the straight-line gamma the pooled correction predicts
    for that same ladder.

    python3 experiments/38_perc_gamma_d4/ladder_sweep.py [--tag ladder] [--delta 0.7]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

from local_slope import omega_ranges  # noqa: E402
from persistence import load_metadata, load_samples  # noqa: E402

#: run order of run_ladder; the pilot is drawn first in every table and figure
LADDERS = ("base", "half", "ugly", "large", "k1.2", "k1.1_e0.15", "k0.9")
PILOT = "pilot"
LIT = (1.430, 0.006)  # README "Known values": an acceptance band, drawn, never fitted


def load_ladder(run: Path) -> dict:
    """One replicate: log Ybar per scale with CLT se cv / sqrt(n)."""
    params = load_metadata(run)["params"]
    smp = load_samples(run)
    x = np.array(sorted(smp), float)
    y = [np.asarray(smp[int(i)]) for i in x]
    ybar = np.array([v.mean() for v in y])
    cv = np.array([v.std(ddof=1) for v in y]) / ybar
    n = np.array([v.size for v in y], float)
    k = float(params.get("ladder_exponent", 1.0))
    return dict(x=x, eps=params["eps0"] * x ** (-1 / k), L=np.log(ybar),
                se=cv / np.sqrt(n), k=k, eps0=float(params["eps0"]))


def load_pilot(path: Path) -> dict:
    """The 5-replicate pilot: log of the pooled mean, se from every replicate's cv."""
    d = json.loads(path.read_text())
    x = np.array(d["scales"], float)
    cv = np.array([r["cv"] for r in d["per_replicate"]])
    se = np.sqrt((cv ** 2 / np.array(d["n"], float)).sum(0)) / len(d["per_replicate"])
    eps0 = float(d["recipe"]["params"]["eps0"])
    return dict(x=x, eps=eps0 / x, L=np.log(np.array(d["y_bar"])), se=se, k=1.0, eps0=eps0)


def line_fit(c: dict) -> dict:
    lx = np.log(c["x"])
    A = np.vstack([np.ones_like(lx), lx]).T / c["se"][:, None]
    coef, *_ = np.linalg.lstsq(A, c["L"] / c["se"], rcond=None)
    cov = np.linalg.inv(A.T @ A)
    r = A @ coef - c["L"] / c["se"]
    dof = len(r) - 2
    return dict(gamma=c["k"] * coef[1], se=c["k"] * float(np.sqrt(cov[1, 1])),
                chi2=float(r @ r), dof=dof, p=float(stats.chi2.sf(r @ r, dof)))


def local_slopes(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dlx = np.diff(np.log(c["x"]))
    s = c["k"] * np.diff(c["L"]) / dlx
    se = c["k"] * np.hypot(c["se"][1:], c["se"][:-1]) / dlx
    return np.sqrt(c["eps"][1:] * c["eps"][:-1]), s, se


def pooled_fit(ladders: dict, delta: float | None) -> dict:
    """Common gamma (and a, unless delta is None) across ladders, one A per ladder."""
    names = list(ladders)
    le = np.concatenate([np.log(ladders[n]["eps"]) for n in names])
    L = np.concatenate([ladders[n]["L"] for n in names])
    se = np.concatenate([ladders[n]["se"] for n in names])
    idx = np.concatenate([[j] * len(ladders[n]["L"]) for j, n in enumerate(names)])
    cols = [(idx == j).astype(float) for j in range(len(names))] + [-le]
    if delta is not None:
        cols.append(np.exp(delta * le))
    A = np.vstack(cols).T / se[:, None]
    coef, *_ = np.linalg.lstsq(A, L / se, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    r = A @ coef - L / se
    g = len(names)
    out = dict(gamma=coef[g], se_gamma=float(np.sqrt(cov[g, g])), chi2=float(r @ r),
               dof=len(L) - A.shape[1], delta=delta)
    if delta is not None:
        out.update(a=coef[g + 1], se_a=float(np.sqrt(cov[g + 1, g + 1])))
    out["p"] = float(stats.chi2.sf(out["chi2"], out["dof"]))
    return out


def predicted_line_gamma(c: dict, fit: dict) -> float:
    """Straight-line gamma the pooled correction implies on ladder c, same weights."""
    lx = np.log(c["x"])
    model = -fit["gamma"] * np.log(c["eps"]) + fit["a"] * c["eps"] ** fit["delta"]
    A = np.vstack([np.ones_like(lx), lx]).T / c["se"][:, None]
    coef, *_ = np.linalg.lstsq(A, model / c["se"], rcond=None)
    return c["k"] * coef[1]


def report(ladders: dict, delta: float) -> tuple[dict, dict]:
    print(f"{'ladder':11} {'eps0':>7} {'k':>4} {'eps window':>19} {'gamma':>8} {'se':>7} "
          f"{'chi2/dof':>8} {'p':>5}  AIC omega range")
    fits = {}
    for name, c in ladders.items():
        f = fits[name] = line_fit(c)
        o = omega_ranges(c["x"], c["L"], c["se"], 2.0)
        aic = ("none (open or on a bound)" if o["best"] is None else
               f"omega {o['omega']:.2f} [{o['best']['omega_lo']:.2f}, {o['best']['omega_hi']:.2f}]")
        print(f"{name:11} {c['eps0']:7.4f} {c['k']:4.2f} {c['eps'].min():9.2e}-{c['eps'].max():8.2e} "
              f"{f['gamma']:8.4f} {f['se']:7.4f} {f['chi2']:5.1f}/{f['dof']} {f['p']:5.2f}  {aic}")
    g = np.array([f["gamma"] for f in fits.values()])
    w = 1 / np.array([f["se"] for f in fits.values()]) ** 2
    mean = float(w @ g / w.sum())
    chi = float(w @ (g - mean) ** 2)
    print(f"\nhomogeneity of the {len(g)} gammas: weighted mean {mean:.4f}, "
          f"chi2 {chi:.1f}/{len(g) - 1}, p = {stats.chi2.sf(chi, len(g) - 1):.4f}")
    lew = np.array([np.mean(np.log(c["eps"])) for c in ladders.values()])
    X = np.vstack([np.ones_like(lew), lew]).T * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X, g * np.sqrt(w), rcond=None)
    se_b = float(np.sqrt(np.linalg.inv(X.T @ X)[1, 1]))
    print(f"gamma against the window's mean log eps: slope {coef[1]:.4f} +- {se_b:.4f} "
          f"({coef[1] / se_b:.1f} sigma)")
    print("\npooled eps-space fits, one A per ladder:")
    p0 = pooled_fit(ladders, None)
    print(f"  no correction     : gamma {p0['gamma']:.4f} +- {p0['se_gamma']:.4f}, "
          f"chi2 {p0['chi2']:.1f}/{p0['dof']}, p = {p0['p']:.1e}")
    for d in sorted({0.5, 0.7, 0.85, 1.0, delta}):
        f = pooled_fit(ladders, d)
        print(f"  a eps^{d:<4} (fixed): gamma {f['gamma']:.4f} +- {f['se_gamma']:.4f}, "
              f"a {f['a']:+.3f} +- {f['se_a']:.3f}, chi2 {f['chi2']:.1f}/{f['dof']}, p = {f['p']:.2f}")
    pc = pooled_fit(ladders, delta)
    b = predicted_line_gamma(ladders[PILOT], pc) - pc["gamma"]
    print(f"\nstraight-line bias the Delta = {delta} fit implies on the pilot's ladder: {b:+.4f}")
    return fits, pc


def figures(ladders: dict, fits: dict, pc: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, ink, muted, band = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3dc"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": muted,
                         "xtick.color": muted, "ytick.color": muted})
    written = []

    # local slopes, one panel per ladder, shared axes
    lo = min(c["eps"].min() for c in ladders.values())
    hi = max(c["eps"].max() for c in ladders.values())
    grid = np.geomspace(lo, hi, 200)
    curve = pc["gamma"] - pc["a"] * pc["delta"] * grid ** pc["delta"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, (name, c) in zip(axes.flat, ladders.items()):
        ax.axhspan(LIT[0] - LIT[1], LIT[0] + LIT[1], color=band, lw=0, zorder=0)
        ax.plot(grid, curve, color=ink, lw=1.2, zorder=2)
        e, s, se = local_slopes(c)
        ax.errorbar(e, s, se, fmt="o", ms=4, color=data, ecolor=data, elinewidth=1,
                    capsize=0, zorder=3)
        f = fits[name]
        ax.set_title(f"{name}: eps0 = {c['eps0']:.4g}, k = {c['k']:g}\n"
                     f"line gamma {f['gamma']:.4f}({1e4 * f['se']:.0f}), "
                     f"chi2 {f['chi2']:.1f}/{f['dof']}", fontsize=8, color=ink)
        ax.set_xscale("log")
        ax.grid(axis="y", color=band, lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel(r"$\varepsilon = p_c - p$ (rung midpoint)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"local slope $\times\,k$")
    fig.suptitle(
        f"d = 4 susceptibility: local slopes on 8 ladders.  Line: pooled fit "
        rf"$\gamma - a\Delta\varepsilon^{{\Delta}}$, $\gamma$ = {pc['gamma']:.4f}, "
        rf"$a$ = {pc['a']:.2f}, $\Delta$ = {pc['delta']:g} (fixed).  Band: literature 1.430(6)",
        fontsize=9, color=ink)
    fig.tight_layout()
    p = out / "ladder_local_slopes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # straight-line gamma against the window, observed vs pooled-fit prediction
    # no literature band here: at this zoom 1.430(6) covers the whole axis
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(pc["gamma"], color=ink, lw=1, ls="--", zorder=1,
               label=rf"pooled fit $\gamma$ = {pc['gamma']:.4f}({1e4 * pc['se_gamma']:.0f})")
    for j, (name, c) in enumerate(ladders.items()):
        wm = float(np.exp(np.mean(np.log(c["eps"]))))
        f = fits[name]
        ax.errorbar(wm, f["gamma"], f["se"], fmt="o", ms=6, color=data, elinewidth=1,
                    capsize=0, zorder=3, label="straight-line fit on the ladder" if j == 0 else None)
        ax.plot(wm, predicted_line_gamma(c, pc), "o", ms=7, mfc="none", mec=ink, mew=1.2,
                zorder=4, label="what the pooled correction predicts for it" if j == 0 else None)
        # pilot and base share a window: put their names on either side
        side = {PILOT: (-6, "right"), "base": (6, "left")}.get(name, (0, "center"))
        ax.annotate(name, (wm, f["gamma"] + f["se"]), xytext=(side[0], 4),
                    textcoords="offset points", ha=side[1], fontsize=7.5, color=muted)
    ax.set_xscale("log")
    ax.set_xlabel(r"geometric mean $\varepsilon$ of the ladder's window")
    ax.set_ylabel(r"$\hat\gamma$ = k $\times$ slope")
    ax.set_title(rf"Straight-line $\hat\gamma$ moves with the window ($\Delta$ = {pc['delta']:g})",
                 fontsize=9, color=ink)
    ax.grid(axis="y", color=band, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    p = out / "ladder_gamma_vs_window.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="ladder", help="data/<tag>/<ladder> runs (default ladder)")
    ap.add_argument("--pilot", default="aic_explore_d4",
                    help="study whose pilot.json is the reference ladder (default aic_explore_d4)")
    ap.add_argument("--delta", type=float, default=0.7,
                    help="fixed correction exponent of the pooled fit (default 0.7)")
    args = ap.parse_args(argv)

    data = EXP / "data"
    ladders = {PILOT: load_pilot(data / args.pilot / "pilot.json")}
    for name in LADDERS:
        ladders[name] = load_ladder(data / args.tag / name)
    fits, pc = report(ladders, args.delta)
    for p in figures(ladders, fits, pc, data / args.tag):
        print(f"figure: {p}")


if __name__ == "__main__":
    main()
