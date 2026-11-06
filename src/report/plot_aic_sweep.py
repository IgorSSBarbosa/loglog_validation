"""The AIC fit across a sweep: omega-hat and gamma-hat against one model parameter
(2026-11-05, generalizing experiments/14_erw_urn_gpu/plot_sweep.py --aic).

Reads each study's omega_aic.json (src/estimate/omega_aic.py) and the parameter from
the study's local_slope.json `params`, so it works for any model run through the
pipeline. Error bars: the fit's stat error only; the range systematic double-counts
each range's noise and is left out (user, 2026-11-05).
No truth is drawn (ground rule 4): an experiment that wants its conjecture beside the
points keeps that in its own script, as experiment 14 does. A study without
omega_aic.json, or whose fit gave no estimate, is left out and named.

Writes <out>/omega_aic_sweep.json (the table) and omega_aic_sweep.png.

CLI:
    python3 src/report/plot_aic_sweep.py --data-root <D> --studies 'fixed_p*_3h' --param p --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402

PRODUCED_BY = "src/report/plot_aic_sweep.py"

_BLUE, _MUTED = "#2a78d6", "#898781"


def collect(sds: list[Path], param: str) -> list[dict]:
    rows = []
    for sd in sds:
        rf = (read_artifact(sd, "omega_aic", required=False) or {}).get("range_fit")
        if rf is None or rf["best"] is None:
            print(f"[aic_sweep] {sd.name}: no AIC estimate "
                  f"({'none given' if rf else 'no omega_aic.json'}), left out", flush=True)
            continue
        params = read_artifact(sd, "local_slope")["params"]
        if param not in params:
            raise SystemExit(f"{sd.name}: local_slope.json params {sorted(params)} lack {param!r}")
        b = rf["best"]
        rows.append({"study": sd.name, param: params[param], "i_min": b["i_min"],
                     "i_max": b["i_max"], "points": b["points"], "chi2": b["chi2"],
                     "dof": b["dof"], "omega": rf["omega"], "omega_stat": rf["stat"],
                     "omega_total": rf["total"], "gamma": rf["gamma"],
                     "gamma_stat": rf["gamma_stat"], "gamma_total": rf["gamma_total"]})
    return sorted(rows, key=lambda r: r[param])


def plot(rows: list[dict], param: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([r[param] for r in rows], float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, name in zip(axes, ("omega", "gamma"), ("ω̂", "γ̂")):
        y = np.array([r[key] for r in rows])
        st = np.array([r[f"{key}_stat"] for r in rows])
        ax.errorbar(x, y, yerr=st, fmt="o", color=_BLUE, ms=6, elinewidth=1.5, capsize=3,
                    label=f"{name} AIC fit ± stat")
        ax.set(xlabel=param, ylabel=name)
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.legend(fontsize=8, loc="best")
    ax = axes[0]
    for r in rows:   # the chosen range, along the bottom
        ax.annotate(f"{r['i_min']}..{r['i_max']}", (r[param], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", rotation=90, ha="center",
                    va="bottom", fontsize=6.5, color=_MUTED)
    axes[0].set_title("ω̂ from a + bk + cρ^(−ωk) on the AIC-chosen range (grey: i_min..i_max)",
                      fontsize=10)
    axes[1].set_title("γ̂ = b / log ρ from the same fit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, required=True, help="where the studies live")
    ap.add_argument("--studies", required=True, help="glob of study directory names")
    ap.add_argument("--param", required=True, help="the swept key in local_slope.json params")
    ap.add_argument("--out", type=Path, required=True, help="directory for the table and figure")
    args = ap.parse_args()
    sds = sorted(p for p in args.data_root.glob(args.studies) if p.is_dir())
    if not sds:
        raise SystemExit(f"no study matches {args.studies!r} under {args.data_root}")
    rows = collect(sds, args.param)
    if not rows:
        raise SystemExit("no study has an AIC estimate: run src/estimate/omega_aic.py first")
    args.out.mkdir(parents=True, exist_ok=True)
    path = write_artifact(args.out, "omega_aic_sweep",
                          {"param": args.param, "studies": args.studies, "rows": rows},
                          produced_by=PRODUCED_BY)
    plot(rows, args.param, artifact_path(args.out, "omega_aic_sweep_figure"))
    print(f"{args.param:>8} {'range':>17} {'omega':>7} {'stat':>6} {'total':>6} "
          f"{'gamma':>9} {'stat':>9} {'total':>9}")
    for r in rows:
        print(f"{r[args.param]:>8g} {r['i_min']:>8}..{r['i_max']:<8} {r['omega']:7.3f} "
              f"{r['omega_stat']:6.3f} {r['omega_total']:6.3f} {r['gamma']:9.5f} "
              f"{r['gamma_stat']:9.2e} {r['gamma_total']:9.2e}")
    print(f"[aic_sweep] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
