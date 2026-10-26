"""Overview figures and the results table of `experiments/percolation_section.md`.

1. `images/exponents_map.png`: which exponent was studied in which dimension, by which
   experiment, and with what outcome.
2. `images/observables_snapshot.png`: one critical 2-D lattice, coloured by what each
   observable counts (illustration only: a single seeded draw, never scored).
3. `images/results_errors.png`: for every experiment with an estimate, its actual error
   against the literature next to its eq. (720) half-width, on a log axis.
4. The results table, printed to stdout as Markdown (`--table`), built from the same rows
   as figure 3 so that no number in it is copied by hand.

Reads finished artifacts only. Literature values appear here for reporting only, copied
from each experiment's README (ground rule 4).

CLI:
    python3 experiments/plot_percolation_results.py [--root DIR] [--table]

`--root` is the checkout whose (gitignored) `experiments/*/data/` holds the artifacts;
it defaults to this one. Figures are always written next to this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loglog_plot import _AQUA, _BLUE, _CRITICAL, _GRIDLINE, _INK, _MUTED, _ORANGE  # noqa: E402

#: Seed of the illustration in figure 2 (ground rule 5); the draw is never scored.
SNAPSHOT_SEED = 2026102601
P_C_2D = 0.59274605079210


@dataclass
class Row:
    """One experiment's estimate, as it enters the table and figure 3.

    `kind` says where the numbers come from: "cell" is one (budget, m0) cell of an
    `allocation_sweep` (theorem checks), "final" a planned run (`answer.json`), "pilot"
    the pilot's eq. (232) fit. `tau` rows report tau = 1 - gamma-hat.
    """
    exp: str
    exponent: str
    dim: int
    kind: str
    path: str
    lit: float
    lit_text: str
    tau: bool = False
    pick: str = "only"          # for "cell": "only", "tuned_top", or "m0=<k>"
    pilot: str = "pilot"        # pilot study dir, for the pilot's sampling time


ROWS = [
    Row("20", "d_f", 2, "cell", "20_perc_df_d2_clt/data/clt", 91 / 48, "91/48"),
    Row("22", "d_f", 2, "cell", "22_perc_df_d2_error_decay/data/error_decay", 91 / 48, "91/48", pick="tuned_top"),
    Row("21", "d_f", 3, "cell", "21_perc_df_d3_clt/data/clt", 2.52293, "2.52293(10)"),
    Row("23", "d_f", 3, "cell", "23_perc_df_d3_error_decay/data/error_decay", 2.52293, "2.52293(10)", pick="tuned_top"),
    Row("27", "d_f", 3, "final", "27_perc_df_d3_facefar/data/pilot", 2.52293, "2.52293(10)"),
    Row("28", "d_f", 3, "pilot", "28_perc_df_d3_axis_deep/data/pilot", 2.52293, "2.52293(10)"),
    Row("29", "d_f", 4, "final", "29_perc_df_d4_facefar/data/pilot", 3.0446, "3.0446(7)"),
    Row("30", "d_f", 5, "final", "30_perc_df_d5_facefar/data/pilot", 3.5260, "3.5260(14)"),
    Row("31", "d_f", 6, "pilot", "31_perc_df_d6_facefar/data/pilot", 4.0, "4 (+ logs)"),
    Row("25a", "τ", 2, "cell", "25_perc_tau_d2_coverage/data/coverage", 187 / 91, "187/91", tau=True),
    Row("25b", "τ", 2, "cell", "25_perc_tau_d2_coverage/data/coverage_box_df", 187 / 91, "187/91", tau=True,
        pick="tuned_top", pilot="pilot_box_df"),
    Row("32", "τ", 3, "pilot", "32_perc_tau_d3/data/pilot", 2.18909, "2.18909(5)", tau=True),
    Row("33", "τ", 4, "pilot", "33_perc_tau_d4/data/pilot", 2.3142, "2.3142(5)", tau=True),
    Row("34", "τ", 5, "pilot", "34_perc_tau_d5/data/pilot", 2.4175, "2.4175(2)", tau=True),
    Row("35", "τ", 6, "pilot", "35_perc_tau_d6/data/pilot", 2.5, "5/2 (+ logs)", tau=True),
    Row("36", "τ", 7, "pilot", "36_perc_tau_d7/data/pilot", 2.5, "5/2", tau=True),
    Row("37", "γ", 3, "pilot", "37_perc_gamma_d3/data/pilot", 1.793, "1.793(3)"),
    Row("38", "γ", 4, "pilot", "38_perc_gamma_d4/data/pilot", 1.430, "1.430(6)"),
    Row("39", "γ", 5, "pilot", "39_perc_gamma_d5/data/pilot", 1.1792, "1.1792(7)"),
    Row("26", "γ/ν", 2, "cell", "26_perc_gammanu_d2_coverage/data/coverage", 43 / 24, "43/24"),
    Row("40", "γ/ν", 3, "pilot", "40_perc_gammanu_d3/data/pilot", 2.0431, "2.0431(14)"),
]


def _json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _scales(s) -> str:
    return f"{s[0]}..{s[-1]} ({len(s)})"


def _pick_cell(sweep: dict, pick: str) -> dict:
    cells = [c for c in sweep["cells"] if not c["skipped"]]
    if pick == "only":
        (cell,) = cells
        return cell
    if pick == "tuned_top":
        top = max(c["budget"] for c in cells)
        (cell,) = [c for c in cells if c["budget"] == top and c["is_tuned_m0"]]
        return cell
    raise ValueError(f"unknown pick {pick!r}")


def measure(row: Row, data: Path) -> dict:
    """Value, half-width, error measure, cost and ladder of one row."""
    d = data / row.path
    if row.kind == "cell":
        sweep = _json(d / "allocation_sweep.json")
        cell = _pick_cell(sweep, row.pick)
        scored = {(c["budget"], c["m0"]): c for c in _json(d / "theorem_scores.json")["cells"]}
        cov = scored[(cell["budget"], cell["m0"])]["coverage"]
        live = [c for c in sweep["cells"] if not c["skipped"]]
        pilot_s = _json(d.parent / row.pilot / "pilot.json")["drawn_seconds"]
        out = {
            "gamma": cell["closed_form"]["mean"], "half": cov["half_width"],
            # per-replicate scale of the error, comparable with a per-replicate bound
            "err_mag": cov["rmse"], "err_label": "RMSE",
            "covered": f"{cov['hits']}/{cov['replicates']}",
            "samples": sum(c["n"] * len(c["scales"]) for c in live) * sweep["replicates"],
            "seconds": sweep["elapsed_seconds"], "pilot_seconds": pilot_s,
            "ladder": f"{_scales(cell['scales'])}, m₀={cell['m0']}",
            "n_text": f"{cell['n']:,} × R={sweep['replicates']}"
                      + (f", {len(live)} cells" if len(live) > 1 else ""),
            "B": cell["budget"],
        }
    elif row.kind == "final":
        ans = _json(d / "answer.json")
        fin = _json(d / "final.json")
        out = {
            "gamma": ans["gamma"], "half": ans["wilson"]["half_width"],
            "err_mag": None, "err_label": "error",
            "covered": None,
            "samples": fin["n"] * len(fin["scales"]) * fin["replicates"],
            "seconds": fin["elapsed_seconds"],
            "pilot_seconds": _json(d / "pilot.json")["drawn_seconds"],
            "ladder": f"{_scales(fin['scales'])}, m₀={fin['m0']}",
            "n_text": f"{fin['n']:,} × R={fin['replicates']}",
        }
    else:
        p = _json(d / "pilot.json")
        out = {
            "gamma": p["gamma_pilot"], "half": None,
            "err_mag": None, "err_label": "error", "covered": None,
            "samples": int(sum(p["n"]) * p["replicates"]),
            "seconds": None, "pilot_seconds": p["drawn_seconds"],
            "ladder": _scales(p["scales"]),
            "n_text": f"{max(p['n']):,}..{min(p['n']):,} × R={p['replicates']}",
        }
    value = 1 - out["gamma"] if row.tau else out["gamma"]
    out["value"] = value
    out["error"] = value - row.lit
    if out["err_mag"] is None:
        out["err_mag"] = abs(out["error"])
    if out["covered"] is None and out["half"] is not None:
        out["covered"] = "yes" if abs(out["error"]) <= out["half"] else "no"
    return out


def _fmt_time(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def results_table(data: Path) -> str:
    head = ("| exp | exponent | d | estimate | ± eq. (720) | literature | error | covered | "
            "window (scales), m₀ | n per scale × R | samples | pilot | run |")
    lines = [head, "|" + "---|" * 13]
    for r in ROWS:
        m = measure(r, data)
        half = f"{m['half']:.4f}" if m["half"] is not None else "—"
        value = {"cell": f"{m['value']:.4f}", "final": f"**{m['value']:.4f}**",
                 "pilot": f"{m['value']:.4f} *(pilot)*"}[r.kind]
        lines.append(
            f"| {r.exp} | {r.exponent} | {r.dim} | {value} | {half} | {r.lit_text} | "
            f"{m['error']:+.4f} | {m['covered'] or '—'} | {m['ladder']} | {m['n_text']} | "
            f"{m['samples']:.2e} | {_fmt_time(m['pilot_seconds'])} | {_fmt_time(m['seconds'])} |")
    return "\n".join(lines)


def fig_results(data: Path, out: Path) -> None:
    from matplotlib.figure import Figure

    rows = list(reversed(ROWS))
    fig = Figure(figsize=(11, 7.6), constrained_layout=True)
    ax = fig.subplots()
    for y, r in enumerate(rows):
        m = measure(r, data)
        if m["half"] is not None:
            ok = m["err_mag"] <= m["half"]
            ax.plot([1e-5, m["half"]], [y, y], "-", color=_BLUE, lw=7, alpha=0.25, solid_capstyle="butt")
            ax.plot([m["half"]], [y], "|", color=_BLUE, ms=14, mew=2)
            ax.plot([m["err_mag"]], [y], "o", color=_AQUA if ok else _CRITICAL, ms=7,
                    mfc=_AQUA if ok else _CRITICAL)
        else:
            wrong = r.exp == "31"
            ax.plot([m["err_mag"]], [y], "o", ms=7, mfc="none",
                    color=_CRITICAL if wrong else _MUTED, mew=1.5)
        t = m["seconds"] or 0
        ax.text(1.6, y, f"{m['value']:.4f}  vs  {r.lit_text}    {_fmt_time(m['pilot_seconds'])}"
                + (f" + {_fmt_time(t)}" if t else "") + f"    {m['samples']:.1e} samples",
                va="center", fontsize=7.5, family="monospace")
    labels = [f"{r.exp}  {r.exponent}, d={r.dim}" + ("  (pilot)" if r.kind == "pilot" else "")
              for r in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for edge in np.cumsum([len([r for r in rows if r.exponent == e]) for e in ("γ/ν", "γ", "τ")]):
        ax.axhline(edge - 0.5, color=_GRIDLINE, lw=1)
    ax.grid(True, axis="x", which="both", color=_GRIDLINE, lw=0.5)
    ax.set_xlabel("|error| against the literature value  (theorem checks: RMSE over replicates)")
    ax.set_title("filled dot inside the blue bar = covered by eq. (720);  red = not covered;  "
                 "hollow = pilot only, no bound", fontsize=9, loc="left")
    ax.text(1.6, len(rows) - 0.55, "estimate vs literature   pilot + run time   samples drawn",
            fontsize=7.5, family="monospace", color=_MUTED, va="bottom")
    fig.savefig(out, dpi=130)


# Outcome categories of figure 1, in legend order.
OUTCOMES = {
    "theorem": ("theorem checks pass (CLT, rate, eq. 720)", _AQUA),
    "run": ("planned run: eq. (720) bound contains literature", _BLUE),
    "fixed": ("eq. (720) failed, diagnosed and fixed", _ORANGE),
    "declined": ("pilot only: protocol declined to plan", "#c9c7bf"),
    "wrong": ("pilot confidently wrong (Assumption 1 fails)", _CRITICAL),
}

#: (exponent row, dim) -> (experiments, outcome, note)
MAP = {
    ("d_f", 2): ("20, 22", "theorem", "CLT, rate"),
    ("d_f", 3): ("21, 23, 27\n(28 pilot)", "theorem", "2.5222 ± 0.0133"),
    ("d_f", 4): ("29", "run", "3.094 ± 0.090"),
    ("d_f", 5): ("30", "run", "3.79 ± 0.27"),
    ("d_f", 6): ("31", "wrong", "4.36 vs 4"),
    ("τ", 2): ("25", "fixed", "box ∝ s^(1/d_f)"),
    ("τ", 3): ("32", "declined", ""), ("τ", 4): ("33", "declined", ""),
    ("τ", 5): ("34", "declined", ""), ("τ", 6): ("35", "declined", ""),
    ("τ", 7): ("36", "declined", ""),
    ("γ", 3): ("37", "declined", ""), ("γ", 4): ("38", "declined", ""),
    ("γ", 5): ("39", "declined", ""),
    ("γ/ν", 2): ("26", "theorem", "Assumption 6 fails"),
    ("γ/ν", 3): ("40", "declined", ""),
}

EXPONENT_ROWS = [
    ("d_f", "fractal dimension d_f\nY_i = sites connected to a face\nE Y_i ∝ i^(d_f)"),
    ("τ", "Fisher exponent τ\nY_s = clusters of size ≥ s per site\nE Y_s ∝ s^(1−τ)"),
    ("γ", "susceptibility γ\nY_x = mean cluster size at p_c − ε₀/x\nE Y_x ∝ x^γ"),
    ("γ/ν", "γ/ν = 2 − η\nY_i = size of the centre site's cluster\nE Y_i ∝ i^(γ/ν)"),
]


def fig_exponents_map(out: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    dims = list(range(2, 8))
    fig = Figure(figsize=(11, 5.0), constrained_layout=True)
    ax = fig.subplots()
    for yi, (key, _) in enumerate(EXPONENT_ROWS):
        y = len(EXPONENT_ROWS) - 1 - yi
        for xi, dim in enumerate(dims):
            cell = MAP.get((key, dim))
            color = OUTCOMES[cell[1]][1] if cell else "white"
            ax.add_patch(Rectangle((xi, y), 0.94, 0.9, facecolor=color, edgecolor=_GRIDLINE,
                                   alpha=0.85 if cell else 1))
            if cell:
                exps, outcome, note = cell
                ink = "white" if outcome in ("run", "wrong", "theorem") else _INK
                ax.text(xi + 0.47, y + 0.55, exps, ha="center", va="center", fontsize=9,
                        weight="bold", color=ink)
                ax.text(xi + 0.47, y + 0.18, note, ha="center", va="center", fontsize=7, color=ink)
    ax.set_xlim(0, len(dims))
    ax.set_ylim(0, len(EXPONENT_ROWS))
    ax.set_xticks([i + 0.47 for i in range(len(dims))],
                  [f"d = {d}" + ("\n(upper critical)" if d == 6 else "") for d in dims])
    ax.set_yticks([len(EXPONENT_ROWS) - 1 - i + 0.45 for i in range(len(EXPONENT_ROWS))],
                  [t for _, t in EXPONENT_ROWS], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in OUTCOMES.values()],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Critical exponents studied, by dimension (cell: experiment numbers, outcome)")
    fig.savefig(out, dpi=130)


def fig_snapshot(out: Path, L: int = 80) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from scipy import ndimage

    # The centre site is closed in ~40% of draws, which leaves the gamma/nu panel empty;
    # take the first seed from SNAPSHOT_SEED whose centre cluster is visible, and say so.
    seed = SNAPSHOT_SEED
    while True:
        open_ = np.random.default_rng(seed).random((L, L)) < P_C_2D
        lab, _ = ndimage.label(open_)       # 4-connected, plain box (illustration only)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        if sizes[lab[L // 2, L // 2]] >= 40 and lab[L // 2, L // 2]:
            break
        seed += 1
    south = np.setdiff1d(np.unique(lab[-1]), [0])
    south_conn = np.isin(lab, south)
    far = np.zeros_like(open_)
    far[: L // 2] = True
    centre = lab == lab[L // 2, L // 2] if lab[L // 2, L // 2] else np.zeros_like(open_)
    largest = int(np.argmax(sizes))

    # 0 closed, 1 open, 2 counted, 3 counted elsewhere / context
    panels = []
    a = open_.astype(int)
    a[south_conn] = 2
    panels.append((a, "d_f (d = 2, exps 20, 22):\nopen sites connected to the south side"))
    b = open_.astype(int)
    b[south_conn] = 3
    b[south_conn & far] = 2
    panels.append((b, "d_f, face_far (d ≥ 3, exps 21, 23, 27–31):\nonly the far half is counted"))
    c = open_.astype(int)
    c[centre] = 2
    panels.append((c, "γ/ν (exps 26, 40):\nthe cluster of the centre site"))
    d_ = open_.astype(int)
    d_[lab == largest] = 2
    big = np.isin(lab, np.flatnonzero(sizes >= 50)) & (lab != largest)
    d_[big] = 3
    panels.append((d_, "τ (exps 25, 32–36): Y_s counts clusters of size ≥ s\n"
                       "blue: largest cluster, orange: others of size ≥ 50"))

    cmap = ListedColormap(["#f4f3ee", "#c9c7bf", _BLUE, _ORANGE])
    fig = Figure(figsize=(12, 3.6), constrained_layout=True)
    for ax, (img, title) in zip(fig.subplots(1, 4), panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Observables on one critical square lattice (p = p_c, {L}×{L}, seed {seed}, the first "
                 f"from {SNAPSHOT_SEED} with an open centre); grey = open site", fontsize=9)
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="checkout holding experiments/*/data (default: this one)")
    ap.add_argument("--table", action="store_true", help="print the results table as Markdown")
    args = ap.parse_args()
    data = args.root / "experiments"
    if args.table:
        print(results_table(data))
        return
    (HERE / "images").mkdir(exist_ok=True)
    jobs = [
        (lambda o: fig_exponents_map(o), HERE / "images/exponents_map.png"),
        (lambda o: fig_snapshot(o), HERE / "images/observables_snapshot.png"),
        (lambda o: fig_results(data, o), HERE / "images/results_errors.png"),
    ]
    for fn, out in jobs:
        fn(out)
        print(f"figure = {out}")


if __name__ == "__main__":
    main()
