"""The figures of `experiments/Walk_section.md` (experiments 44 and 45).

1. `images/walk_models.png`: one seeded walk of each model, for illustration only (never
   scored). `rwre`: a space-time picture of the exclusion environment with the walker's
   path, at three p, stirred by `models/rwre.py`'s own `_stir`. ERW: trajectories at the
   four p of experiment 45, from a sequential transcription of `models/erw.py`'s rule.
2. `images/walk_estimators.png`: (a) the three estimators on one replicate, drawn as the
   lines they are in the log-log plane; (b) the single-point bias against its closed form
   log c / log n.
3. `images/walk_distributions.png`: gamma-hat over the 500 replicates of each point with a
   known exponent, per estimator.
4. `images/walk_coverage.png`: coverage (Wilson 95% CI) and RMSE per point and estimator.
5. `images/walk_local_slopes.png`: local slopes of the replicate-pooled means on the
   published grid, every point, with the known exponents.

Reads finished `protocol_arm.json` artifacts only. Exponents and amplitudes appear here
for reporting only (ground rule 4).

CLI:
    python3 experiments/plot_walk_section.py [--root DIR]

`--root` is the checkout whose (gitignored) `experiments/*/data/` holds the artifacts;
it defaults to this one. Figures are always written next to this script.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coverage import wilson_score_interval  # noqa: E402
from tools.loglog_plot import _AQUA, _BLUE, _CRITICAL, _GRIDLINE, _INK, _MUTED, _ORANGE  # noqa: E402

#: Seed of the illustrations in figure 1 (ground rule 5); never scored.
ILLUSTRATION_SEED = 2026102660

#: (label, artifact dir under experiments/, truth or None, amplitude c or None).
#: c is the closed-form limit of E|X_n| / n^gamma where one is known:
#: simple random walk sqrt(2/pi); ERW below p = 3/4, sqrt(2 / (pi (3 - 4p))).
POINTS = [
    ("RWRE p=1/2", "44_lit_avena_thomann2012_rwre/data/avena_thomann_p0.5", 0.5, math.sqrt(2 / math.pi)),
    ("RWRE p=1/3", "44_lit_avena_thomann2012_rwre/data/avena_thomann_p0.333", None, None),
    ("RWRE p=0", "44_lit_avena_thomann2012_rwre/data/avena_thomann_p0", None, None),
    ("ERW p=0.25", "45_lit_practice_erw/data/practice_p0.25", 0.5, math.sqrt(2 / (math.pi * 2.0))),
    ("ERW p=0.6", "45_lit_practice_erw/data/practice_p0.6", 0.5, math.sqrt(2 / (math.pi * 0.6))),
    ("ERW p=0.9", "45_lit_practice_erw/data/practice_p0.9", 0.8, None),
    ("ERW p=3/4", "45_lit_practice_erw/data/practice_p0.75", 0.5, None),
]
ESTIMATORS = [("full", "regression, all times", _BLUE),
              ("window", "regression, χ² window", _AQUA),
              ("single_point", "single point (Avena–Thomann)", _ORANGE)]


def _load(data: Path, rel: str) -> dict:
    with open(data / rel / "protocol_arm.json") as fh:
        return json.load(fh)


def _gammas(art: dict, w: str) -> tuple[np.ndarray, np.ndarray | None]:
    rows = [r[w] for r in art["per_replicate"] if r.get(w, {}).get("gamma") is not None]
    g = np.array([r["gamma"] for r in rows])
    se = np.array([r["se"] for r in rows]) if rows and rows[0]["se"] is not None else None
    return g, se


# ---------------------------------------------------------------- figure 1

def rwre_spacetime(p: float, steps: int, rng: np.random.Generator):
    """One annealed walk with its environment recorded at every step (illustration)."""
    from models import rwre  # noqa: E402  (models/ is importable from the repo root)

    q = rwre._check({"p": p})
    w = 2 * steps + 2                        # wide enough that the walk never wraps
    half = w // 2
    col_parity = (np.arange(w) % 2).astype(np.int8)
    env = rng.random((1, w), dtype=np.float32) < q["alpha"]
    x = 0
    frames, path = [env[0].copy()], [0]
    for _ in range(steps):
        on_particle = env[0, (x + half) % w]
        p_left = q["p"] if on_particle else 1.0 - q["p"]
        x += -1 if rng.random() < p_left else 1
        for _ in range(q["env_sweeps"]):
            env = rwre._stir(env, col_parity, q["swap_prob"], rng)
        frames.append(env[0].copy())
        path.append(x)
    return np.array(frames), np.array(path), half


def erw_path(p: float, steps: int, rng: np.random.Generator) -> np.ndarray:
    """S_0..S_steps of one ERW, step by step: step t+1 repeats a uniformly chosen past
    step with probability p and reverses it otherwise; the first step is fair. The same
    rule as models/erw.py, which resolves it backwards (illustration only)."""
    s = np.empty(steps, dtype=np.int8)
    s[0] = 1 if rng.random() < 0.5 else -1
    for t in range(1, steps):
        past = s[rng.integers(t)]
        s[t] = past if rng.random() < p else -past
    return np.concatenate([[0], np.cumsum(s)])


def fig_models(out: Path) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    rng = np.random.default_rng(ILLUSTRATION_SEED)
    fig = Figure(figsize=(12, 7.2), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
    axes = top.subplots(1, 3, sharey=True)
    steps = 400
    for ax, p in zip(axes, (0.5, 1 / 3, 0.0)):
        frames, path, half = rwre_spacetime(p, steps, rng)
        lo, hi = half - 60, half + 60
        ax.imshow(frames[:, lo:hi], aspect="auto", cmap=ListedColormap(["#f4f3ee", "#c9c7bf"]),
                  interpolation="nearest", origin="lower",
                  extent=(-60.5, 59.5, -0.5, steps + 0.5))
        ax.plot(path, np.arange(steps + 1), color=_BLUE, lw=1.3)
        ax.set_title(f"RWRE, p = {p:.3g}" + ("  (simple random walk)" if p == 0.5 else ""), fontsize=9)
        ax.set_xlabel("site")
    axes[0].set_ylabel("time step")
    top.suptitle("rwre: walker (blue) on the exclusion process (grey = particle), density 1/2, γ_eff = 1",
                 fontsize=9)

    axes = bottom.subplots(1, 4, sharex=True)
    t = 4096
    for ax, (p, colour) in zip(axes, ((0.25, _AQUA), (0.6, _BLUE), (0.75, _ORANGE), (0.9, _CRITICAL))):
        for _ in range(6):
            ax.plot(erw_path(p, t, rng), color=colour, lw=0.7, alpha=0.8)
        gamma = 0.5 if p <= 0.75 else 2 * p - 1
        env = np.sqrt(np.arange(t + 1) * (np.log(np.arange(t + 1) + 1) if p == 0.75 else 1)) \
            if p <= 0.75 else np.arange(t + 1) ** gamma
        ax.plot(env, color=_MUTED, ls=":", lw=1)
        ax.plot(-env, color=_MUTED, ls=":", lw=1)
        regime = {0.25: "diffusive", 0.6: "diffusive", 0.75: "critical, √(t log t)", 0.9: "superdiffusive"}[p]
        ax.set_title(f"ERW, p = {p}: {regime}", fontsize=9)
        ax.set_xlabel("time step")
    axes[0].set_ylabel("position S_t")
    bottom.suptitle("ERW: six walks per p; dotted: ±t^γ (±√(t ln t) at p = 3/4)", fontsize=9)
    fig.suptitle(f"Illustration only, seed {ILLUSTRATION_SEED}", fontsize=8, color=_MUTED)
    fig.savefig(out, dpi=130)


# ---------------------------------------------------------------- figure 2

def fig_estimators(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.5, 4.4), constrained_layout=True)
    ax, axb = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})

    art = _load(data, POINTS[3][1])                 # ERW p = 0.25, c far from 1
    r = art["per_replicate"][0]
    n = np.array(art["scales"], dtype=float)
    x, y = np.log2(n), np.log2(r["y_bar"])
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.linspace(0, 17, 50)
    ax.plot(xs, icpt + slope * xs, color=_BLUE, lw=1.5,
            label=f"regression (intercept free): slope {slope:.4f}")
    sp = y[-1] / x[-1]
    ax.plot(xs, sp * xs, color=_ORANGE, lw=1.5, ls="--",
            label=f"single point (line through the origin): slope {sp:.4f}")
    ax.plot(xs, 0.5 * xs, color=_MUTED, ls=":", lw=1.2, label="true slope 1/2")
    ax.plot(x, y, "o", color=_INK, ms=6, zorder=5, label="log₂ Ȳ at n = 2¹³..2¹⁶")
    ax.annotate(f"intercept = log₂ c = {icpt:.2f}", (0, icpt), xytext=(4.5, -1.1),
                arrowprops={"arrowstyle": "->", "color": _MUTED}, fontsize=8)
    ax.set(xlabel="log₂ n", ylabel="log₂ Ȳ_n", xlim=(0, 17), ylim=(-1.5, 9),
           title="(a) one replicate, ERW p = 0.25: the two estimators are two lines")
    ax.grid(True, color=_GRIDLINE, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    pred, meas, labels = [], [], []
    for label, rel, truth, c in POINTS:
        if c is None:
            continue
        g, _ = _gammas(_load(data, rel), "single_point")
        pred.append(math.log(c) / math.log(max(_load(data, rel)["scales"])))
        meas.append(g.mean() - truth)
        labels.append(label)
    lim = (-0.06, 0.01)
    axb.plot(lim, lim, color=_MUTED, ls=":", lw=1)
    axb.plot(pred, meas, "o", color=_ORANGE, ms=8)
    for px, my, lab in zip(pred, meas, labels):
        axb.annotate(f"{lab}\npred {px:+.4f}, meas {my:+.4f}", (px, my), xytext=(8, -4),
                     textcoords="offset points", fontsize=7.5)
    axb.set(xlim=lim, ylim=lim, xlabel="predicted bias  log c / log n_max",
            ylabel="measured mean bias (500 replicates)",
            title="(b) single-point bias = its amplitude term")
    axb.grid(True, color=_GRIDLINE, lw=0.5)
    fig.savefig(out, dpi=130)


# ---------------------------------------------------------------- figures 3, 4

TRUTHED = [pt for pt in POINTS if pt[2] is not None]


def fig_distributions(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 3.6), constrained_layout=True)
    axes = fig.subplots(1, len(TRUTHED))
    for ax, (label, rel, truth, _) in zip(axes, TRUTHED):
        art = _load(data, rel)
        allg = np.concatenate([_gammas(art, w)[0] for w, _, _ in ESTIMATORS])
        bins = np.linspace(min(allg.min(), truth) - 0.005, max(allg.max(), truth) + 0.005, 60)
        for w, name, colour in ESTIMATORS:
            ax.hist(_gammas(art, w)[0], bins=bins, color=colour, alpha=0.55, label=name)
        ax.axvline(truth, color=_INK, lw=1.2, ls="--")
        ax.set_title(f"{label}, γ = {truth}" + (" (+ log)" if "3/4" in label else ""), fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("γ̂")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(out, dpi=130)


def fig_coverage(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.5, 4.0), constrained_layout=True)
    axc, axr = fig.subplots(1, 2)
    xs = np.arange(len(TRUTHED))
    for k, (w, name, colour) in enumerate(ESTIMATORS):
        cov, lo, hi, rmse = [], [], [], []
        for _, rel, truth, _ in TRUTHED:
            s = _load(data, rel)["scores"][w]
            rmse.append(s["rmse"])
            if s["hits"] is not None:
                a, b = wilson_score_interval(s["hits"], s["replicates"])
                cov.append(s["hits"] / s["replicates"])
                lo.append(a)
                hi.append(b)
        off = (k - 1) * 0.25
        if cov:
            cov = np.array(cov)
            axc.errorbar(xs + off, cov, yerr=[cov - np.array(lo), np.array(hi) - cov],
                         fmt="o", color=colour, capsize=3, label=name)
        axr.bar(xs + off, rmse, width=0.25, color=colour, label=name)
    axc.axhline(0.95, color=_MUTED, ls="--", lw=1)
    axc.set(ylim=(0, 1.02), ylabel="coverage of γ̂ ± 1.96 se", title="(a) coverage, Wilson 95% CI")
    axr.set(yscale="log", ylabel="RMSE of γ̂", title="(b) RMSE")
    for ax in (axc, axr):
        ax.set_xticks(xs, [t[0] for t in TRUTHED], fontsize=8)
        ax.grid(True, axis="y", which="both", color=_GRIDLINE, lw=0.5)
    axc.legend(fontsize=7.5, frameon=False, loc="lower left")
    axr.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(out, dpi=130)


# ---------------------------------------------------------------- figure 5

def fig_local_slopes(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.5, 4.2), constrained_layout=True)
    axr, axe = fig.subplots(1, 2)
    colours = [_INK, _BLUE, _ORANGE, _AQUA, _BLUE, _CRITICAL, _ORANGE]
    for (label, rel, truth, _), colour in zip(POINTS, colours):
        art = _load(data, rel)
        n = np.array(art["scales"], dtype=float)
        ys = np.array([r["y_bar"] for r in art["per_replicate"]])
        # local slope per replicate, then mean and its standard error over replicates
        s = np.diff(np.log(ys), axis=1) / np.diff(np.log(n))
        mid = np.sqrt(n[1:] * n[:-1])
        ax = axr if label.startswith("RWRE") else axe
        ax.errorbar(mid, s.mean(0), s.std(0, ddof=1) / math.sqrt(s.shape[0]), fmt="o-",
                    color=colour, ms=5, capsize=2, label=label)
        g, _ = _gammas(art, "full")
        ax.plot([n[0], n[-1]], [g.mean()] * 2, color=colour, lw=6, alpha=0.18)
        if truth is not None and "3/4" not in label:
            ax.axhline(truth, color=colour, ls=":", lw=1)
    xx = np.geomspace(2**13, 2**16, 50)
    axe.plot(xx, 0.5 + 1 / (2 * np.log(xx)), color=_ORANGE, ls="--", lw=1,
             label="p = 3/4: ½ + 1/(2 ln n), from √(n ln n)")
    axr.plot([], [], color=_MUTED, lw=6, alpha=0.3, label="band: mean of the regression γ̂")
    for ax, title in ((axr, "(a) rwre: Avena–Thomann allocation"),
                      (axe, "(b) ERW: practice allocation")):
        ax.set_xscale("log", base=2)
        ax.set(xlabel="n (midpoint of n, 2n)", ylabel="mean local slope ± 1 SE", title=title)
        ax.grid(True, which="both", color=_GRIDLINE, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="checkout holding experiments/*/data (default: this one)")
    args = ap.parse_args()
    data = args.root / "experiments"
    (HERE / "images").mkdir(exist_ok=True)
    jobs = [
        (lambda o: fig_models(o), "walk_models.png"),
        (lambda o: fig_estimators(data, o), "walk_estimators.png"),
        (lambda o: fig_distributions(data, o), "walk_distributions.png"),
        (lambda o: fig_coverage(data, o), "walk_coverage.png"),
        (lambda o: fig_local_slopes(data, o), "walk_local_slopes.png"),
    ]
    for fn, name in jobs:
        fn(HERE / "images" / name)
        print(f"figure = {HERE / 'images' / name}")


if __name__ == "__main__":
    main()
