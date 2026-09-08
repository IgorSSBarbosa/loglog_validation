"""One table for a sweep of runs that differ only in the spatial dimension.

Reads the run directories `src/generate/generate.py` wrote for a set of
dimensions and reports, per dimension: the estimator's gamma-hat with a
confidence interval, the exponents it converts to, and the correction-to-
scaling constants of the direct fit of eq. (232). It measures nothing itself
-- every number comes from tools/loglog.py and tools/correction.py, the same
estimators every other experiment uses.

Where the confidence interval comes from
----------------------------------------
`gamma_drop_leading` is an OLS slope of log Ybar against log i, so it is a
FIXED linear functional of the per-scale log-means,

    gamma_hat = sum_k c_k log Ybar_k,
    c_k = (log i_k - mean log i) / sum_j (log i_j - mean log i)**2,

(identical to the article's eq. (526) weights on a consecutive rho**k grid --
tools/loglog.gamma_closed_form is the check that the two agree). The rungs are
drawn independently, ground rule 2, so

    Var(gamma_hat) = sum_k c_k**2 sigma_k**2,
    sigma_k = sd(Y_k) / (sqrt(n_k) Ybar_k),

sigma_k being the delta-method standard error of log Ybar_k. That is eq.
(583)'s CLT variance specialized to the n_k this run actually drew, rather
than to the idealized allocation. The interval is gamma_hat +/- 1.96 se.

Two things it is NOT. It is a STATISTICAL interval only: a local slope also
carries the correction-to-scaling term, so an interval that excludes the
literature value means the ladder is not asymptotic, not that the literature
is wrong -- which is the usual situation here and is why the m0 column is
printed rather than one number. And it is not the article's Wilson interval
(eq. 720), which additionally needs omega_1 and a_1 and is what
`src/study/report.py` builds once those are settled.

Truth enters here and nowhere else
----------------------------------
LITERATURE below is used for REPORTING ONLY -- printed beside the estimate,
never fed to an estimator (PLAN.md; the same rule that keeps `target_fn` out of
every real model). Pass `--no-truth` to print the table without it.

CLI:
    python3 src/report/dimension_table.py \\
        -data experiments/05_percolation_highd/data --prefix sweep_tau_d --dims 2-8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import read_artifact  # noqa: E402
from tools.correction import fit_correction  # noqa: E402
from tools.loglog import gamma_drop_leading  # noqa: E402
from tools.models import get_model  # noqa: E402

#: REPORTING ONLY. tau and d_f per spatial dimension. Nothing here reaches an
#: estimator -- see the module docstring.
#:
#: `tau` is stated OUTRIGHT rather than derived from d_f, because the relation
#: that connects them,
#:
#:     tau = 1 + dim / d_f          (hyperscaling)
#:
#: HOLDS ONLY BELOW THE UPPER CRITICAL DIMENSION d_c = 6. Above it the
#: exponents stop depending on the dimension at all: tau = 5/2 and d_f = 4 for
#: EVERY dim >= 6, and hyperscaling's failure is what defines d_c. Deriving the
#: acceptance value as 1 + dim/4 gives 2.75 at dim = 7 and 3.0 at dim = 8,
#: which is not a percolation exponent -- it is the formula being used outside
#: its range (my error, caught on the first full sweep, 2026-09-07).
#:
#: `hyperscaling` records where the relation may be used, which is what decides
#: whether a d_f can be READ OFF a measured tau at all (see `report`).
LITERATURE: dict[int, dict] = {
    2: {"tau": 187 / 91, "d_f": 91 / 48, "hyperscaling": True,
        "source": "den Nijs / Nienhuis, EXACT"},
    3: {"tau": 2.18906, "d_f": 2.5226, "hyperscaling": True,
        "source": "beta/nu = 0.4774, Xu et al. (2014)"},
    4: {"tau": 2.31383, "d_f": 3.0446, "hyperscaling": True,
        "source": "beta/nu = 0.9554, Ballesteros et al. (1997)"},
    5: {"tau": 2.41243, "d_f": 3.54, "hyperscaling": True,
        "source": "beta/nu = 1.46"},
    6: {"tau": 2.5, "d_f": 4.0, "hyperscaling": True,
        "source": "mean field, EXACT -- but dim = d_c, so log corrections"},
    7: {"tau": 2.5, "d_f": 4.0, "hyperscaling": False,
        "source": "mean field, EXACT (dim > d_c)"},
    8: {"tau": 2.5, "d_f": 4.0, "hyperscaling": False,
        "source": "mean field, EXACT (dim > d_c)"},
}


def per_scale(run_dir: Path) -> dict:
    """Ybar, its log standard error, cv and n, per scale, from one run."""
    meta = read_artifact(run_dir, "samples_meta")
    scales = [int(s) for s in meta["scales"]]
    z = np.load(run_dir / "samples.npz")
    y_bar, sigma_log, cv, n = [], [], [], []
    for s in scales:
        y = z[str(s)] if str(s) in z else z[f"scale_{s}"]
        m = float(y.mean())
        sd = float(y.std(ddof=1))
        y_bar.append(m)
        sigma_log.append(sd / math.sqrt(y.size) / m)     # delta method
        cv.append(sd / m)
        n.append(int(y.size))
    params = meta.get("params", {})
    # The real work drawn, in the model's own unit -- NOT the recipe's budget,
    # which is in allocation units and differs from it by cost_unit_ratio.
    hint = get_model(meta["model"]).cost_hint
    sites = (sum(nk * hint(s, params) for s, nk in zip(scales, n))
             if hint is not None else float("nan"))
    return {"scales": scales, "y_bar": y_bar, "sigma_log": sigma_log,
            "cv": cv, "n": n, "params": params, "sites": sites,
            "seed": meta.get("seed")}


def gamma_with_se(scales, y_bar, sigma_log, m0: int) -> tuple[float, float]:
    """OLS-on-logs slope over scales[m0:], and its standard error.

    The slope is a fixed linear functional of the log-means, so its variance is
    sum_k c_k**2 sigma_k**2 exactly -- no bootstrap, no resampling.
    """
    li = np.log(np.asarray(scales[m0:], dtype=float))
    ly = np.log(np.asarray(y_bar[m0:], dtype=float))
    sg = np.asarray(sigma_log[m0:], dtype=float)
    c = (li - li.mean()) / ((li - li.mean()) ** 2).sum()
    return float(c @ ly), float(math.sqrt(float((c ** 2) @ (sg ** 2))))


def analyse(run_dir: Path, m0: int | None) -> dict:
    d = per_scale(run_dir)
    dim = int(d["params"].get("dim", 2))
    dl = gamma_drop_leading(d["scales"], d["y_bar"])
    ladder = {r["m0"]: r["gamma_hat"] for r in dl}
    # Default m0: the largest window that still leaves >= 4 rungs, i.e. the
    # most contamination dropped while the slope is still over-determined.
    chosen = m0 if m0 is not None else max(0, len(d["scales"]) - 4)
    g, se = gamma_with_se(d["scales"], d["y_bar"], d["sigma_log"], chosen)
    out = {"dim": dim, "run": str(run_dir), **d,
           "drop_leading": ladder, "m0": chosen, "gamma": g, "se": se}
    try:
        out["direct_fit"] = fit_correction(d["scales"], d["y_bar"],
                                           sigma_log=d["sigma_log"])
    except Exception as exc:                            # noqa: BLE001
        out["direct_fit"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _fmt(x, w=8, p=4):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) and math.isfinite(x) else f"{'--':>{w}}"


def report(rows: list[dict], truth: bool = True) -> None:
    print(f"{'dim':>4} {'ladder':>14} {'sites':>10} {'m0':>3} "
          f"{'gamma_hat':>10} {'95% CI':>19} {'tau_hat':>9}"
          + (f" {'tau (lit)':>10} {'err':>8}" if truth else "")
          + f" {'d_f_hat':>9}"
          + (f" {'d_f (lit)':>10}" if truth else ""))
    for r in rows:
        dim, g, se = r["dim"], r["gamma"], r["se"]
        tau = 1.0 - g
        lit = LITERATURE.get(dim) if truth else None
        # d_f = -dim/gamma comes from hyperscaling, so it may only be read off
        # a measured tau BELOW d_c. Above it, tau carries no information about
        # d_f at all and printing a number here would invent one.
        hyper = (lit or {}).get("hyperscaling", True)
        df = (-dim / g) if (g < 0 and hyper) else float("nan")
        lad = f"{r['scales'][0]}..{r['scales'][-1]}({len(r['scales'])})"
        line = (f"{dim:>4} {lad:>14} {r.get('sites', 0):>10.2e} {r['m0']:>3} "
                f"{_fmt(g, 10)} [{g - 1.96 * se:8.4f},{g + 1.96 * se:8.4f}] "
                f"{_fmt(tau, 9)}")
        if lit:
            line += f" {lit['tau']:>10.4f} {100 * (tau - lit['tau']) / lit['tau']:>7.2f}%"
        line += f" {_fmt(df, 9)}"
        if lit:
            line += f" {lit['d_f']:>10.4f}"
            if not hyper:
                line += "   (hyperscaling fails above d_c = 6)"
        print(line)

    print(f"\n{'dim':>4} {'a0':>10} {'a1':>10} {'omega1':>8} {'rel_rmse':>10} "
          f"{'gamma (fit)':>12} {'tau (fit)':>10}   correction-to-scaling, direct fit of eq. (232)")
    for r in rows:
        f = r["direct_fit"]
        if "error" in f:
            print(f"{r['dim']:>4}   {f['error']}")
            continue
        print(f"{r['dim']:>4} {_fmt(f['a0'], 10)} {_fmt(f['a1'], 10)} "
              f"{_fmt(f['omega1'], 8, 3)} {f['rel_rmse']:>10.2e} "
              f"{_fmt(f['gamma'], 12)} {_fmt(1 - f['gamma'], 10)}"
              + ("" if f.get("converged", True) else "   NOT CONVERGED"))

    print(f"\n{'dim':>4}   gamma_hat by m0 (drop-leading ladder)")
    for r in rows:
        seq = "  ".join(f"{m}:{v:.4f}" for m, v in sorted(r["drop_leading"].items()))
        print(f"{r['dim']:>4}   {seq}")

    print(f"\n{'dim':>4}   cv per scale (Assumption 6: should be flat)")
    for r in rows:
        print(f"{r['dim']:>4}   " + " ".join(f"{c:.3f}" for c in r["cv"]))


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-data", "--data", required=True, type=Path,
                    help="directory holding the per-dimension run directories")
    ap.add_argument("--prefix", default="sweep_tau_d",
                    help="run directory name prefix; the dimension is appended")
    ap.add_argument("--dims", default="2-8", help="e.g. 2-8 or 2,3,5")
    ap.add_argument("--m0", type=int, default=None,
                    help="drop this many leading rungs (default: leave 4)")
    ap.add_argument("--no-truth", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    if "-" in a.dims:
        lo, hi = a.dims.split("-")
        dims = list(range(int(lo), int(hi) + 1))
    else:
        dims = [int(x) for x in a.dims.split(",")]

    rows = []
    for dim in dims:
        rd = a.data / f"{a.prefix}{dim}"
        if not (rd / "samples.npz").exists():
            print(f"# dim {dim}: no run at {rd}", file=sys.stderr)
            continue
        rows.append(analyse(rd, a.m0))
    if not rows:
        raise SystemExit("no runs found")
    report(rows, truth=not a.no_truth)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2, sort_keys=True, default=float))
        print(f"\njson = {a.json}")


if __name__ == "__main__":
    _main()
