"""Arm B of `plans/literature_fss_protocols.md`: a PUBLISHED allocation, this repo's estimator.

A published finite-size-scaling study fixes a list of system sizes L_j and a number of
samples n_j at each (e.g. Zhang et al. 2020: L = 4..60 in 5D, 10^6 samples per L <= 20,
10^5 above). This driver draws exactly that allocation -- the sizes and the sample counts,
taken literally from a `protocol` recipe -- and then estimates gamma with the article's
log-log regression instead of the paper's nonlinear fits (user, 2026-10-26). Three
estimates are reported per replicate:

- `full`:   the OLS slope of log Ybar on log L over every published size;
- `window`: the same slope over L >= L_min, with L_min chosen by the published papers'
            chi^2 rule (`tools/lmin_rule.py`);
- `single_point`: log Ybar_n / log n at the largest n, the estimator of Avena & Thomann
            2012 (arXiv:1201.2890, Sec. 3.3: alpha(n) = log SD_n / log n, "approximated by
            the value of alpha(n) with the biggest n"), applied to this repo's observable.
            It has no error bar in the paper and gets none here: scored on bias and RMSE only.

The two regressions come with the finite-m standard error sqrt(sum_j v_j^2 cv_j^2 / n_j)
(`tools/wilson.py`'s `sigma_se_ols`), the only error bar that applies on a non-rho^k grid
with non-uniform n. Eq. (720)'s bias terms are stated for uniform n on a rho^k grid and
are NOT applied here. The interval scored is gamma-hat +/- 1.96 se.

The budget is recorded in the cost units `autopilot.py` spends (sum_j n_j cost_hint(L_j)),
so arm A can be run at the same budget. Every replicate draws fresh randomness from its
own spawned SeedSequence (ground rule 2), through `generate()` with a `reduce` that
keeps only each scale's mean and variance. `true_gamma`, if the recipe states it, is
used only to score the finished estimates (ground rule 4). The artifact is rewritten
after every replicate, so a long run can be read while it goes.

CLI:
    python3 src/budget/protocol_arm.py -meta experiments/43_lit_zhang2020_df_d5/recipes/protocol_zhang2020_d5.json \\
        --tag zhang2020_d5 [--replicates R]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import default_out_dir, load_recipe, write_artifact  # noqa: E402
from tools.lmin_rule import select_lmin  # noqa: E402
from tools.loglog import gamma_all_points  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools.persistence import run_dir as _run_dir  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402
from tools.wilson import sigma_se_ols  # noqa: E402

from src.generate.generate import generate  # noqa: E402

Z95 = 1.959963984540054


def mean_and_var(samples) -> tuple[float, float]:
    x = np.asarray(samples, dtype=np.float64)
    return float(x.mean()), float(x.var(ddof=1))


def estimate(scales, n, y_bar, var) -> dict:
    """Both windows' gamma-hat and finite-m se from one replicate's per-scale moments."""
    scales = np.asarray(scales)
    n = np.asarray(n, dtype=float)
    y_bar = np.asarray(y_bar, dtype=float)
    var = np.asarray(var, dtype=float)
    if np.any(y_bar <= 0):
        # Assumption 2 fails on this draw: log Ybar = -inf. Kept, not repaired.
        bad = [int(s) for s in scales[y_bar <= 0]]
        return {"failed": f"non-positive mean at L = {bad}"}
    cv2 = var / y_bar**2
    sigma_log = np.sqrt(cv2 / n)
    out = {"cv2": cv2.tolist(), "sigma_log": sigma_log.tolist()}
    out["full"] = {"lmin": int(scales.min()), "gamma": gamma_all_points(scales, y_bar),
                   "se": sigma_se_ols(scales, n, cv2)}
    top = int(np.argmax(scales))
    out["single_point"] = {"lmin": int(scales[top]),
                           "gamma": float(math.log(y_bar[top]) / math.log(scales[top])),
                           "se": None}
    sel = select_lmin(scales, y_bar, sigma_log)
    out["lmin_scan"] = sel["scan"]
    if sel["lmin"] is None:
        out["window"] = {"lmin": None, "why": sel["why"]}
    else:
        k = scales >= sel["lmin"]
        out["window"] = {"lmin": sel["lmin"], "gamma": gamma_all_points(scales[k], y_bar[k]),
                         "se": sigma_se_ols(scales[k], n[k], cv2[k]), "why": sel["why"]}
    return out


def score(replicates: list[dict], true_gamma: float) -> dict:
    """Bias, sd, RMSE and coverage of gamma-hat +/- 1.96 se, per estimate.

    `single_point` has no se, so its `mean_se` and `hits` are None.
    """
    out = {}
    for w in ("full", "window", "single_point"):
        rows = [r[w] for r in replicates if w in r and r[w].get("gamma") is not None]
        if not rows:
            out[w] = {"replicates": 0}
            continue
        g = np.array([r["gamma"] for r in rows])
        err = g - true_gamma
        has_se = rows[0]["se"] is not None
        se = np.array([r["se"] for r in rows]) if has_se else None
        out[w] = {
            "replicates": len(rows),
            "mean": float(g.mean()), "bias": float(err.mean()),
            "sd": float(g.std(ddof=1)) if g.size > 1 else None,
            "rmse": float(math.sqrt(np.mean(err**2))),
            "mean_se": float(se.mean()) if has_se else None,
            "hits": int(np.sum(np.abs(err) <= Z95 * se)) if has_se else None,
            "lmins": sorted({r["lmin"] for r in rows}),
        }
    return out


def run(cfg: dict, *, replicates: int, out_rd: Path | None, progress: bool = True) -> dict:
    model, params = cfg["model"], cfg["params"]
    scales = [int(s) for s in cfg["scales"]]
    n = [int(v) for v in cfg["n"]]
    if len(scales) != len(n):
        raise ValueError(f"{len(scales)} scales but {len(n)} sample counts")
    spec = get_model(model)
    # A model may decline to declare its cost (rwre_gpu_lazy: its work grows with the
    # exponent being measured). Then there is no budget in cost units, only wall clock.
    budget = (float(sum(nj * spec.cost_hint(L, params) for L, nj in zip(scales, n)))
              if spec.cost_hint is not None else None)
    seed_seq = np.random.SeedSequence(cfg["seed"])
    streams = spawn(seed_seq, replicates)
    result = {
        "model": model, "params": params, "source": cfg.get("source"),
        "scales": scales, "n": n, "budget_per_replicate": budget,
        "replicates_planned": replicates, "seed": seed_record(seed_seq),
        "estimator": "OLS slope of log Ybar on log L (eqs. 523-531 on a general grid); "
                     "se = sqrt(sum v_j^2 cv_j^2/n_j); interval +/- 1.96 se",
        "per_replicate": [], "complete": False,
    }
    t0 = time.perf_counter()
    for r, stream in enumerate(streams):
        t = time.perf_counter()
        moments = generate(model, scales, n, params, seed=stream, reduce=mean_and_var,
                           progress=progress)
        y_bar = [moments[L][0] for L in scales]
        var = [moments[L][1] for L in scales]
        est = estimate(scales, n, y_bar, var)
        est.update({"replicate": r, "seconds": time.perf_counter() - t,
                    "y_bar": y_bar, "var": var})
        result["per_replicate"].append(est)
        result["elapsed_seconds"] = time.perf_counter() - t0
        if "true_gamma" in cfg:
            result["true_gamma"] = cfg["true_gamma"]
            result["scores"] = score(result["per_replicate"], cfg["true_gamma"])
        if out_rd is not None:
            write_artifact(out_rd, "protocol_arm", result, produced_by="src/budget/protocol_arm.py")
        if progress:
            w, f = est.get("window", {}), est.get("full", {})
            print(f"replicate {r + 1}/{replicates}: {est['seconds']:.0f}s  "
                  f"full {f.get('gamma', float('nan')):.5f} ± {f.get('se', float('nan')):.5f}  "
                  f"window(L>={w.get('lmin')}) {w.get('gamma', float('nan')):.5f} ± "
                  f"{w.get('se', float('nan')):.5f}", file=sys.stderr)
    result["complete"] = True
    if out_rd is not None:
        write_artifact(out_rd, "protocol_arm", result, produced_by="src/budget/protocol_arm.py")
    return result


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-meta", "--meta", dest="meta", required=True, type=Path)
    ap.add_argument("-o", "--out-dir", dest="out_dir", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--replicates", type=int, default=None,
                    help="override the recipe's replicate count")
    args = ap.parse_args(argv)
    cfg = load_recipe(args.meta, "protocol")
    reps = args.replicates if args.replicates is not None else int(cfg["replicates"])
    rd = _run_dir(args.out_dir or default_out_dir(args.meta), args.tag)
    rd.mkdir(parents=True, exist_ok=True)
    res = run(cfg, replicates=reps, out_rd=rd)
    B = res["budget_per_replicate"]
    print(f"budget per replicate = {f'{B:.3e} cost units' if B is not None else 'n/a (model declares no cost)'}; "
          f"elapsed = {res['elapsed_seconds']:.0f}s; artifact in {rd}")
    for w, s in res.get("scores", {}).items():
        if s.get("replicates"):
            cov = (f"mean se {s['mean_se']:.5f}  covered {s['hits']}/{s['replicates']}"
                   if s["mean_se"] is not None else f"no error bar, {s['replicates']} replicates")
            print(f"  {w:<12} mean {s['mean']:.5f}  bias {s['bias']:+.5f}  rmse {s['rmse']:.5f}  "
                  f"{cov}  L_min {s['lmins']}")


if __name__ == "__main__":
    _main()
