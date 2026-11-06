"""omega_1 window by window, two ways, from a study's local-slope ladder (user, 2026-11-04).

On a log-log plot with m points L_k = mean log Ybar at i = rho^k:

  (1) w_k = -log(Delta s_{k+1} / Delta s_k) / log rho, the log slope of the local
      slopes (`tools/local_slope.local_slopes`): 4 points each, m - 3 estimates;
  (2) L = a + b k + c rho^(-omega k) fitted by weighted least squares on each run of
      --width consecutive points (`tools/local_slope.omega_windows`): m - width + 1
      estimates, each with gamma = b / log rho and a1;
(2) weights each point by its CLT se, sqrt(sum_r cv_r^2 / n_r) / R, from the
cells' own cv and n (user, 2026-11-05): the sd of R = 3 replicate means rests on 2 dof
and made chi2 heavy-tailed.

Both read eq. (232) truncated at J = 1, where log E Y_i = log a0 + gamma log i +
a1 i^-omega1 holds exactly. At width 4, (2) interpolates and equals (1) identically,
so the default is 5 (user, 2026-11-04). Neither ever sees a known omega1 (ground rule 4).

Input is <study>/local_slope.json (src/study/local_slope.py, or autopilot's last step):
the same n at every scale, so the -cv^2/(2n) offset of log Ybar cancels in second
differences. Nothing is drawn. Writes <study>/omega_windows.json and .png. The range
fit with AIC cuts lives in src/estimate/omega_aic.py; when <study>/omega_aic.json
exists, its best range and omega are drawn over the windows for comparison.

Standard in the pipeline (user, 2026-11-05): src/study/local_slope.py runs it at
width 5, after omega_aic, whenever it completes, so autopilot's last step produces it
too. Run this driver by hand for another --width or on a study whose local_slope.json predates that.

CLI:
    python3 src/estimate/omega_windows.py --study <s> --data-root <D> [--width 5]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools import local_slope as LS  # noqa: E402

PRODUCED_BY = "src/estimate/omega_windows.py"

# Same palette as tools/loglog_plot.py.
_BLUE, _ORANGE, _MUTED = "#2a78d6", "#eb6834", "#898781"


def analyse(sd: Path, width: int) -> dict:
    ls = read_artifact(sd, "local_slope")
    a = ls["analysis"]
    scales, rho = a["scales"], a["rho"]
    by_scale = {i: [c for c in ls["cells"] if c["scale"] == i] for i in scales}
    # CLT se from every replicate's cv and n, not the sd of R = 3 replicate means
    # (user, 2026-11-05); the replicate se stays alongside as a cross-check.
    se = LS.clt_se_log([[c["cv"] for c in by_scale[i]] for i in scales],
                       [[c["n"] for c in by_scale[i]] for i in scales])
    return {"study": sd.name, "rho": rho, "scales": scales, "width": width,
            "log_y": a["log_y"],
            "source": "local_slope.json: analysis log_y; cells cv, n (one n at every scale)",
            "se_log_y": {"used": "clt", "clt": se.tolist(), "replicates": a["se_log_y"]},
            "slope_of_slopes": {"first_scale": scales[:len(a["w"])], "omega": a["w"],
                                "se": a["se_w"], "resolved": a["w_resolved"]},
            "window_fit": LS.omega_windows(scales, a["log_y"], se, rho, width)}


def plot(res: dict, out: Path, rf: dict | None = None) -> None:
    """omega-hat per window; `rf`, omega_aic.json's range_fit, adds its best range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lr = math.log(res["rho"])
    fig, ax = plt.subplots(figsize=(8, 5))
    m1 = res["slope_of_slopes"]
    for k0, w, se, ok in zip(m1["first_scale"], m1["omega"], m1["se"], m1["resolved"]):
        if w is None:
            continue
        ax.errorbar(math.log(k0) / lr - 0.08, w, yerr=se, color=_BLUE, marker="o", ms=5,
                    mfc=_BLUE if ok else "none", lw=0, elinewidth=1)
    for f in res["window_fit"]:
        x = math.log(f["scales"][0]) / lr + 0.08
        lo, hi = f["omega"] - f["omega_lo"], f["omega_hi"] - f["omega"]
        good = not (f["interval_open"] or f["omega_on_bound"])
        ax.errorbar(x, f["omega"], yerr=[[lo], [hi]], color=_ORANGE, marker="s", ms=5,
                    mfc=_ORANGE if good else "none", lw=0, elinewidth=1)
    ax.plot([], [], "o", color=_BLUE, label="(1) slope of local slopes, 4 points (hollow: |Δs| < 2 se)")
    ax.plot([], [], "s", color=_ORANGE,
            label=f"(2) LS fit a + bk + cρ^(−ωk), {res['width']} points (hollow: interval open)")
    ax.set(xlabel=f"log_{res['rho']:g} of the window's first scale", ylabel="ω̂",
           title=f"{res['study']}: ω̂ per window")
    if rf is not None and rf["best"] is not None:
        b = rf["best"]
        x0, x1 = math.log(b["i_min"]) / lr, math.log(b["i_max"]) / lr - (res["width"] - 1)
        x1 = max(x1, x0 + 0.5)
        ax.fill_between([x0, x1], rf["omega"] - rf["total"], rf["omega"] + rf["total"],
                        color=_MUTED, alpha=0.25, lw=0)
        ax.hlines(rf["omega"], x0, x1, color="k", lw=1.5)
        ax.plot([], [], "-", color="k",
                label=f"AIC fit (omega_aic.json), range {b['i_min']}..{b['i_max']}: ω̂ = "
                      f"{rf['omega']:.3f} ± {rf['stat']:.3f} (stat) ± {rf['systematic']:.3f} (range)")
        # centred on the range fit: windows past the noise floor reach the omega bounds
        half = max(0.75, 4 * rf["total"])
        ax.set_ylim(rf["omega"] - half, rf["omega"] + half)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fmt(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def omega_windows(sd: Path, width: int = 5) -> Path:
    """The whole tool on the study in `sd`: writes omega_windows.json and .png, prints
    the table. Draws nothing. src/study/local_slope.py calls it on every completed run
    (user, 2026-11-05: standard in the pipeline)."""
    res = analyse(sd, width)
    path = write_artifact(sd, "omega_windows", res, produced_by=PRODUCED_BY)
    aic = artifact_path(sd, "omega_aic")
    plot(res, artifact_path(sd, "omega_windows_figure"),
         read_artifact(sd, "omega_aic")["range_fit"] if aic.exists() else None)
    m1, fits = res["slope_of_slopes"], {f["scales"][0]: f for f in res["window_fit"]}
    print(f"[omega_windows] {sd.name}  (rho = {res['rho']:g}, width = {width})")
    print(f"{'i_k':>8} | {'(1) w_k':>8} {'se':>7} res | {'(2) omega':>9} {'68% interval':>17}"
          f" {'chi2':>6} {'gamma':>8} {'a1':>8}")
    for k0 in sorted(set(m1["first_scale"]) | set(fits)):
        j = m1["first_scale"].index(k0) if k0 in m1["first_scale"] else None
        w, se, ok = (m1["omega"][j], m1["se"][j], m1["resolved"][j]) if j is not None else (None,) * 3
        f = fits.get(k0)
        right = (f"{f['omega']:9.3f} [{f['omega_lo']:6.3f}, {f['omega_hi']:6.3f}]"
                 f"{'*' if f['interval_open'] else ' '} {f['chi2']:6.2f} {f['gamma']:8.5f}"
                 f" {f['a1']:8.3f}" if f else "")
        print(f"{k0:>8} | {_fmt(w):>8} {_fmt(se):>7} {'yes' if ok else ' no' if ok is not None else '   '}"
              f" | {right}")
    print("(* = interval open: chi2 never rises by 1 within the omega bounds)")
    print(f"[omega_windows] -> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True, help="the study's directory name")
    ap.add_argument("--data-root", type=Path, required=True, help="where the study lives")
    ap.add_argument("--width", type=int, default=5,
                    help="points per least-squares window (>= 4; 4 reproduces method 1)")
    args = ap.parse_args()
    omega_windows(args.data_root / args.study, args.width)
    return 0

if __name__ == "__main__":
    sys.exit(main())
