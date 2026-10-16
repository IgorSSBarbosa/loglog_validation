"""omega_1 from the loglog-log plot of a finished study (pilot + final run).

The method is `tools/loglog_log.py`'s (user, 2026-10-16). This driver fixes, for the
general case, which data play which role:

  - gamma_k: the consecutive slopes of the PILOT's log-log plot, from its replicate
    means (the autopilot's first stage, which also sized the plan);
  - gamma-hat: `answer.json`'s gamma with its replicate se, the final run's m-point
    estimator (the second stage, the long run). It never sees pilot data, so the two
    are independent draws and Cov(gamma_k - gamma-hat) is Cov(gamma_k) + se^2.

Only resolved points with the majority sign are fitted (`loglog_log.select`). Works on
any study with pilot.json and answer.json: an autopilot study, or pilot -> plan -> run
-> report. A pilot-only study has no answer.json and fails; see
`experiments/19_rwre_lazy_omega/omega_loglog_top.py` for that case.

Writes <study>/omega_loglog.json and <study>/omega_loglog.png: (a) the log-log plot,
pilot and final, with the line of slope gamma-hat through the final's points; (b) the
loglog-log plot with its fitted line. Nothing is drawn (no samples).

CLI:
    python3 src/estimate/omega_loglog.py --study <s> --data-root <D>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools import loglog_log as LL  # noqa: E402

PRODUCED_BY = "src/estimate/omega_loglog.py"

# Same palette as tools/loglog_plot.py.
_BLUE, _ORANGE, _MUTED = "#2a78d6", "#eb6834", "#898781"


def pilot_means(pilot: dict) -> tuple[list, np.ndarray, np.ndarray]:
    """(scales, mean log Ybar, its variance) of the pilot, from its replicates."""
    scales, reps = pilot["scales"], pilot["per_replicate"]
    n = pilot["n"] if isinstance(pilot["n"], list) else [pilot["n"]] * len(scales)
    L, var = LL.scale_means([r["y_bar"] for r in reps], [r["cv"] for r in reps], n, len(reps))
    return scales, L, var


def analyse(sd: Path) -> dict:
    pilot = read_artifact(sd, "pilot")
    answer = read_artifact(sd, "answer")
    scales, L, var = pilot_means(pilot)
    res = LL.omega_loglog(scales, L, var, gamma_ext=(answer["gamma"], answer["se"]))
    return {**res, "study": sd.name,
            "gamma_k_source": "pilot.json per_replicate: consecutive slopes",
            "gamma_hat_source": f"answer.json gamma (m = {answer['m']}, m0 = {answer['m0']}) "
                                "and its replicate se",
            "pilot_scales": scales, "pilot_n": pilot["n"], "pilot_replicates": len(pilot["per_replicate"]),
            "final_scales": answer["scales"], "final_replicates": answer["replicates"],
            "omega1_pilot_fit": pilot["direct_fit"]["omega1"]}


def plot(res: dict, pilot: dict, answer: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = res["rho"]
    lr = math.log(rho)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for d, mk, c, lab in ((pilot, "o", _BLUE, "pilot (γ_k)"), (answer, "s", _ORANGE, "final (γ̂)")):
        # sigma_log is the se of ln Ybar; the plot is in log_rho.
        a.errorbar(np.log(d["scales"]) / lr, np.log(d["y_bar"]) / lr,
                   yerr=np.asarray(d["sigma_log"]) / lr, color=c, marker=mk, ms=4, lw=0,
                   elinewidth=1, label=lab)
    xf = np.log(answer["scales"]) / lr
    yf = np.log(answer["y_bar"]) / lr
    xx = np.linspace(np.log(pilot["scales"][0]) / lr, xf[-1], 50)
    a.plot(xx, yf.mean() + res["gamma_hat"] * (xx - xf.mean()), color=_MUTED, lw=1, ls="--",
           label=f"slope γ̂ = {res['gamma_hat']:.5f}")
    a.set(xlabel=f"log_{rho:g} i", ylabel=f"log_{rho:g} Ȳ_i", title="(a) log-log plot")
    a.legend(fontsize=8)
    lab = (f"ω = {res['omega']:.3f} ± {res['se']:.3f}, χ²/dof {res['chi2']:.1f}/{res['dof']}"
           if res["omega"] is not None else None)
    LL.plot_panel(b, res, _BLUE, lab)
    b.set(xlabel=f"log_{rho:g} k (lower scale of the pair)",
          ylabel=f"log_{rho:g} |γ_k − γ̂|",
          title="(b) loglog-log plot: filled = fitted" + ("" if lab else f" — no fit: {res['why_none']}"))
    if lab:
        b.legend(fontsize=8)
    for ax in (a, b):
        ax.grid(True, color="#e1e0d9", lw=0.6)
    fig.suptitle(res["study"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True, help="the study's directory name")
    ap.add_argument("--data-root", type=Path, required=True, help="where the study lives")
    args = ap.parse_args()
    sd = args.data_root / args.study
    res = analyse(sd)
    path = write_artifact(sd, "omega_loglog", res, produced_by=PRODUCED_BY)
    plot(res, read_artifact(sd, "pilot"), read_artifact(sd, "answer"),
         artifact_path(sd, "omega_loglog_figure"))
    om = (f"{res['omega']:.4f} ± {res['se']:.4f} (χ² {res['chi2']:.1f}/{res['dof']}, "
          f"p = {res['p_value']:.3g})" if res["omega"] is not None else f"none: {res['why_none']}")
    print(f"[omega_loglog] {args.study}: γ̂ = {res['gamma_hat']:.6f}, {res['n_points']} points, ω = {om}")
    print(f"[omega_loglog] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
