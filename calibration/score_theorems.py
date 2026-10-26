"""Score the article's theorems on a finished allocation sweep, drawing nothing.

Reads `src/budget/allocation_experiment.py`'s `allocation_sweep` (per-cell,
per-replicate closed-form gamma-hat values) and a pilot study directory
(`constants.json` + `pilot.json`), and checks three statements per sweep:

  CLT, eq. (583)     sqrt(n m^3) (gamma_hat - E gamma_hat) -> N(0, 12 sigma_inf^2 / log^2 rho).
                     Per cell: observed Var(gamma_hat) / predicted, with a
                     percentile bootstrap CI over replicates, and a KS test of
                     (gamma_hat - own mean) / predicted sd against N(0, 1).
  thm:wilson, (720)  |gamma_hat - gamma| <= B_fs + B_good + B_bad + Phi(alpha) sigma_se.
                     Per cell: the fraction of replicates inside the bound,
                     with a Wilson score CI, and bound / RMSE.
  prop:opt, (945-6)  per budget, best m0 vs prop:opt's m0 vs the tuned m0
                     (`allocation_experiment.summarize`), and the error-decay
                     slope of log RMSE vs log B against -omega1/(d + 2 omega1),
                     with a bootstrap CI over replicates.

Where each input comes from, and why:

- The predicted sd is `tools/wilson.py`'s `sigma_se` -- eq. (583)'s and eq.
  (720)'s sqrt(12 sigma_inf2/(n m^3))/log(rho), verbatim. sigma_inf2 is the
  pilot's cv^2 at its LARGEST scale, `moment_bounds`' and report.py's
  definition. The exact finite-m variance (`sigma_se_per_scale`) is reported
  beside it whenever the pilot measured cv at every scale of the cell; it is
  1.4% wider at m = 6 (see its docstring). Neither is silently substituted
  for the other: both ratios are recorded.
- eq. (720)'s constants follow report.py's `wilson_inputs`: sigma_max2 =
  max cv^2 over the pilot's scales, Lambda = max over scales of the
  replicate-mean log_moment, delta = `LOG_MOMENT_DELTA`. omega1 and a1 come
  from the PILOT's constants.json, never from the sweep's values.
  experiments/exponents_report.md's extension 3 is applied when a1 is
  unidentified (`plan.unidentified`) and omega1 is identified (se < omega1/2):
  B_fs uses |a1| + 2 se(a1). With omega1 unidentified there is no bound.
- The sweep's n is uniform per cell, so eq. (720) is evaluated as stated, with
  no harmonic-mean extension.
- `true_gamma` scores finished estimates only (ground rule 4); it is read from
  the sweep, where allocation_experiment recorded it from its recipe.

The bootstraps are resampling of recorded numbers, not simulation, but are
seeded through `tools/rng.py` (`--seed`) so the CIs reproduce.

CLI:
    python3 calibration/score_theorems.py \\
        --sweep experiments/21_perc_df_d3_clt/data/clt/allocation_sweep.json \\
        --pilot experiments/21_perc_df_d3_clt/data/pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                           # repo root; calibration/ -> ../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.allocation import rate_exponent, rate_exponent_se  # noqa: E402
from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools.constants import load, require  # noqa: E402
from tools.coverage import wilson_score_interval  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402
from tools.summary import LOG_MOMENT_DELTA  # noqa: E402
from tools.wilson import finite_size_bias, sigma_se, sigma_se_per_scale, wilson_interval  # noqa: E402

from src.budget.allocation_experiment import summarize  # noqa: E402
from src.study.plan import unidentified  # noqa: E402

LEVEL = 0.95
N_BOOT = 4000
KS_ALPHA = 0.05
ESTIMATOR = "closed_form"     # prop:opt and eq. (720) are stated for eq. (523)-(526)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def pilot_inputs(pilot_dir: Path) -> dict:
    """eq. (720)'s constants from a pilot study, as report.py's `wilson_inputs` takes them.

    Returns the constants plus `why`, naming anything missing. Missing pieces
    are reported, never defaulted: a bound with a term set to zero is not one.
    """
    consts = load(pilot_dir)
    pilot = read_artifact(pilot_dir, "pilot")
    omega1 = require(consts, "omega1")
    a1 = require(consts, "a1")
    cv = np.asarray(pilot["cv_per_scale"], float)
    why = []

    lm = [r.get("log_moment") for r in pilot["per_replicate"]]
    Lambda = None
    if any(x is None for x in lm):
        why.append("Lambda (this pilot predates the log_moment summary field)")
    else:
        # An expectation, so replicates average; the max is over scales.
        Lambda = float(np.mean(np.array(lm, float), axis=0).max())

    # exponents_report.md, "Three extensions", item 3.
    if omega1.se is not None and omega1.se >= omega1.value / 2:
        why.append(f"omega1 unidentified ({omega1.value:.3g} +/- {omega1.se:.3g}, "
                   f"se >= omega1/2)")
        a1_used, a1_rule = None, "n/a"
    elif unidentified(a1):
        a1_used = abs(a1.value) + 2 * a1.se
        a1_rule = (f"a1 unidentified ({a1.value:.3g} +/- {a1.se:.3g}): "
                   f"|a1| + 2 se = {a1_used:.3g} (exponents_report.md extension 3)")
    else:
        a1_used, a1_rule = a1.value, "pilot value"

    return {
        "pilot_dir": str(pilot_dir),
        "scales": [int(s) for s in pilot["scales"]],
        "cv2_by_scale": {int(s): float(c ** 2) for s, c in zip(pilot["scales"], cv)},
        "sigma_inf2": float(cv[-1] ** 2),   # largest scale -- moment_bounds' definition
        "sigma_max2": float((cv ** 2).max()),
        "Lambda": Lambda,
        "delta": LOG_MOMENT_DELTA if Lambda is not None else None,
        "omega1": omega1.value, "omega1_se": omega1.se,
        "a1": a1.value, "a1_se": a1.se, "a1_used": a1_used, "a1_rule": a1_rule,
        "why": why,
    }


# --------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------

def _percentile_ci(x: np.ndarray, level: float = LEVEL) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]


def clt_check(values: np.ndarray, sd_pred: float, rng: np.random.Generator,
              *, sd_pred_exact: float | None = None, n_boot: int = N_BOOT) -> dict:
    """eq. (583): observed / predicted variance, bootstrap CI, and KS vs N(0, 1).

    Centred at the cell's OWN mean, because eq. (583) is about gamma_hat -
    E gamma_hat: the finite-size bias is eq. (720)'s business, not the CLT's.
    Estimating that centre makes the KS p-value slightly conservative
    (Lilliefors); the scale is the PREDICTED sd, not the sample one, so a
    wrong variance shows up here as well as in the ratio.
    """
    from scipy.stats import kstest

    v = np.asarray(values, float)
    R = v.size
    var_obs = float(v.var(ddof=1))
    boot = v[rng.integers(0, R, size=(n_boot, R))].var(axis=1, ddof=1)
    ratio_ci = _percentile_ci(boot / sd_pred ** 2)
    z = (v - v.mean()) / sd_pred
    ks = kstest(z, "norm")
    out = {
        "replicates": R,
        "sd_obs": sqrt(var_obs),
        "sd_pred": sd_pred,
        "var_ratio": var_obs / sd_pred ** 2,
        "var_ratio_ci": ratio_ci,
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "sd_pred_exact": sd_pred_exact,
        "var_ratio_exact": None if sd_pred_exact is None else var_obs / sd_pred_exact ** 2,
        "var_ratio_exact_ci": (None if sd_pred_exact is None
                               else _percentile_ci(boot / sd_pred_exact ** 2)),
    }
    out["passes"] = bool(ratio_ci[0] <= 1.0 <= ratio_ci[1] and out["ks_p"] > KS_ALPHA)
    return out


def coverage_check(values: np.ndarray, true_gamma: float, half_width: float,
                   level: float = LEVEL) -> dict:
    """Fraction of replicates with |gamma_hat - gamma| <= half_width, Wilson score CI."""
    v = np.asarray(values, float)
    err = np.abs(v - true_gamma)
    hits = int((err <= half_width).sum())
    lo, hi = wilson_score_interval(hits, v.size, level=level)
    rmse = float(sqrt(np.mean(err ** 2)))
    return {
        "half_width": half_width, "hits": hits, "replicates": int(v.size),
        "coverage": hits / v.size, "coverage_ci": [lo, hi],
        "rmse": rmse, "bound_over_rmse": half_width / rmse if rmse > 0 else None,
        # A bound must cover at least `level`; "below" only when the Wilson CI
        # excludes it, so R = 50 cannot fail a bound on one unlucky replicate.
        "verdict": "below nominal" if hi < level else "consistent",
    }


def wilson_bound(n: int, m: int, m0: int, rho: float, pc: dict) -> dict | None:
    """eq. (720) for one cell, exactly as report.py calls `wilson_interval`."""
    if pc["why"]:
        return None
    w = wilson_interval(0.0, n, m, m0, rho,
                        sigma_inf2=pc["sigma_inf2"], sigma_max2=pc["sigma_max2"],
                        a1=pc["a1_used"], omega1=pc["omega1"],
                        Lambda=pc["Lambda"], delta=pc["delta"], level=LEVEL)
    # The half-width does not depend on gamma_hat; drop the placeholder centre.
    w = {k: v for k, v in w.items() if k not in ("gamma_hat", "interval")}
    # report.py's B_fs span: omega1 sits in an exponent of B_fs.
    if pc["omega1_se"]:
        lo_w = max(1e-3, pc["omega1"] - pc["omega1_se"])
        w["B_fs_span"] = sorted(finite_size_bias(m, m0, rho, pc["a1_used"], x)
                                for x in (lo_w, pc["omega1"] + pc["omega1_se"]))
    return w


def score_cell(cell: dict, sweep: dict, pc: dict, rng: np.random.Generator,
               n_boot: int = N_BOOT) -> dict:
    n, m0, m, rho = cell["n"], cell["m0"], sweep["m"], sweep["rho"]
    v = np.asarray(cell[ESTIMATOR]["values"], float)
    sd = sigma_se(n, m, rho, pc["sigma_inf2"])
    cv2 = [pc["cv2_by_scale"].get(int(s)) for s in cell["scales"]]
    sd_exact = (sigma_se_per_scale([n] * m, m, rho, cv2) if None not in cv2 else None)
    out = {
        "budget": cell["budget"], "m0": m0, "n": n, "scales": cell["scales"],
        "is_prop_opt_m0": cell.get("is_prop_opt_m0", False),
        "is_tuned_m0": cell.get("is_tuned_m0", False),
        "mean": float(v.mean()), "bias": float(v.mean() - sweep["true_gamma"]),
        "clt": clt_check(v, sd, rng, sd_pred_exact=sd_exact, n_boot=n_boot),
        "clt_exact_why": (None if sd_exact is not None else
                          "pilot has no cv at scale(s) "
                          f"{[s for s, c in zip(cell['scales'], cv2) if c is None]}"),
    }
    w = wilson_bound(n, m, m0, rho, pc)
    out["wilson"] = w
    out["coverage"] = (coverage_check(v, sweep["true_gamma"], w["half_width"])
                       if w is not None else None)
    return out


def rate_check(sweep: dict, summary: dict, rng: np.random.Generator,
               n_boot: int = N_BOOT) -> dict:
    """Slope of log RMSE vs log B per arm, with a bootstrap over replicates.

    Each cell is resampled independently (its replicates are independent
    draws, ground rule 2), and the `best` arm re-takes its argmin inside every
    resample, since "the best m0 on the grid" is itself an estimate.
    """
    theory = -sweep["omega1"] / (sweep["d"] + 2 * sweep["omega1"])
    out = {"theory": theory, "arms": {}}
    budgets = [B for B in sweep["budgets"] if str(B) in summary]
    if len(budgets) < 3:
        out["why"] = f"need >= 3 budgets with cells, have {len(budgets)}"
        return out

    tg = sweep["true_gamma"]
    boot_rmse = {}           # (B, m0) -> (n_boot,) bootstrap RMSEs
    for c in sweep["cells"]:
        if c["skipped"] or c["budget"] not in budgets:
            continue
        v = np.asarray(c[ESTIMATOR]["values"], float)
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        boot_rmse[(c["budget"], c["m0"])] = np.sqrt(np.mean((v[idx] - tg) ** 2, axis=1))

    x = np.log(np.asarray(budgets, float))
    xc = x - x.mean()
    tuned = sweep.get("tuned") or {}
    for arm in ("prop_opt", "tuned", "best"):
        pts, boots = [], []
        for B in budgets:
            s = summary[str(B)]
            if arm == "best":
                pts.append(s["best_rmse"])
                boots.append(np.min([r for (b, _), r in boot_rmse.items() if b == B], axis=0))
                continue
            m0 = s["prop_opt_m0"] if arm == "prop_opt" else (tuned.get(str(B)) or {}).get("m0")
            r = s["rmse_at_prop_opt_m0"] if arm == "prop_opt" else s["rmse_at_tuned_m0"]
            if m0 is None or r is None:
                break
            pts.append(r)
            boots.append(boot_rmse[(B, m0)])
        if len(pts) != len(budgets):
            continue           # this arm's m0 is off the grid at some budget
        y = np.log(np.array(boots))                  # (budgets, n_boot)
        slopes = (xc[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()
        got = rate_exponent(budgets, pts)
        ci = _percentile_ci(slopes)
        out["arms"][arm] = {
            "budgets": budgets, "rmse": pts, "slope": got, "slope_ci": ci,
            "slope_se_analytic": rate_exponent_se(budgets, sweep["replicates"]),
            "theory_in_ci": bool(ci[0] <= theory <= ci[1]),
        }
    return out


def score(sweep: dict, pc: dict, *, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Every check on every non-skipped cell, plus the per-budget comparison."""
    cells = [c for c in sweep["cells"] if not c["skipped"]]
    streams = spawn(seed, len(cells) + 1)
    scored = [score_cell(c, sweep, pc, np.random.default_rng(s), n_boot)
              for c, s in zip(cells, streams)]
    summary = summarize(sweep, ESTIMATOR)
    warnings = []
    for k in ("omega1", "a1"):
        sv = sweep.get(k)
        if sv is not None and not np.isclose(sv, pc[k], rtol=1e-9, atol=0):
            warnings.append(f"sweep's {k} = {sv:.6g} differs from the pilot's {pc[k]:.6g}; "
                            f"the bound uses the pilot's")
    return {
        "estimator": ESTIMATOR, "level": LEVEL, "n_boot": n_boot, "seed": seed_record(seed),
        "model": sweep["model"], "m": sweep["m"], "rho": sweep["rho"],
        "true_gamma": sweep["true_gamma"],
        "pilot": {k: v for k, v in pc.items() if k != "cv2_by_scale"},
        "cells": scored,
        "per_budget": summary,
        "rate": rate_check(sweep, summary, np.random.default_rng(streams[-1]), n_boot),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def markdown(res: dict) -> str:
    pc = res["pilot"]
    L = [f"# Theorem scores: {res['model']}, m = {res['m']}, rho = {res['rho']:g}, "
         f"estimator {res['estimator']}", ""]
    for w in res["warnings"]:
        L.append(f"> WARNING: {w}")
    L += [f"pilot `{pc['pilot_dir']}`: sigma_inf2 = {pc['sigma_inf2']:.4g}, "
          f"sigma_max2 = {pc['sigma_max2']:.4g}, Lambda = {pc['Lambda']}, "
          f"omega1 = {pc['omega1']:.4g} +/- {pc['omega1_se']}, a1 used: {pc['a1_rule']}", ""]
    if pc["why"]:
        L += [f"No eq. (720) bound: {'; '.join(pc['why'])}.", ""]

    L += ["## CLT, eq. (583), and coverage, eq. (720)", "",
          "| B | m0 | n | R | bias | sd obs | sd pred | Var ratio [95% CI] | exact-m ratio | "
          "KS p | CLT | (720) half-width | coverage [Wilson 95%] | bound/RMSE | dominant |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        k = c["clt"]
        mark = (" (prop:opt)" if c["is_prop_opt_m0"] else "") + (" (tuned)" if c["is_tuned_m0"] else "")
        ex = "--" if k["var_ratio_exact"] is None else f"{k['var_ratio_exact']:.3f}"
        if c["coverage"] is None:
            cov = "n/a | n/a | n/a | n/a"
        else:
            v = c["coverage"]
            cov = (f"{v['half_width']:.3e} | {v['coverage']:.3f} [{v['coverage_ci'][0]:.3f}, "
                   f"{v['coverage_ci'][1]:.3f}] {v['verdict']} | "
                   f"{v['bound_over_rmse']:.2f} | {c['wilson']['dominant']}")
        L.append(f"| {c['budget']:.3g} | {c['m0']}{mark} | {c['n']:,} | {k['replicates']} | "
                 f"{c['bias']:+.3e} | {k['sd_obs']:.3e} | {k['sd_pred']:.3e} | "
                 f"{k['var_ratio']:.3f} [{k['var_ratio_ci'][0]:.3f}, {k['var_ratio_ci'][1]:.3f}] | "
                 f"{ex} | {k['ks_p']:.3f} | {'pass' if k['passes'] else 'FAIL'} | {cov} |")
    L += ["", "CLT passes when the variance-ratio CI contains 1 and KS p > "
          f"{KS_ALPHA}. Coverage is 'below nominal' only when its Wilson CI excludes "
          f"{LEVEL}. Every bound omits eq. (720)'s omega2 piece (phi_plus unmeasured).", ""]

    L += ["## prop:opt, eq. (945)-(946): m0 per budget", "",
          "| B | best m0 | best RMSE | prop:opt m0 | penalty | tuned m0 | penalty |",
          "|---|---|---|---|---|---|---|"]
    for B, s in res["per_budget"].items():
        pf = f"{s['penalty_factor']:.2f}x" if s["penalty_factor"] else "--"
        pt = f"{s['penalty_tuned']:.2f}x" if s["penalty_tuned"] else "--"
        tm = s["tuned_m0"] if s["tuned_m0"] is not None else "--"
        L.append(f"| {float(B):.3g} | {s['best_m0']} | {s['best_rmse']:.3e} | "
                 f"{s['prop_opt_m0']} | {pf} | {tm} | {pt} |")

    r = res["rate"]
    L += ["", f"## Error-decay rate, theory -omega1/(d+2 omega1) = {r['theory']:+.4f}", ""]
    if "why" in r:
        L.append(f"Not fitted: {r['why']}.")
    else:
        L += ["| arm | slope | bootstrap 95% CI | analytic se | theory in CI |",
              "|---|---|---|---|---|"]
        for arm, a in r["arms"].items():
            L.append(f"| {arm} | {a['slope']:+.4f} | [{a['slope_ci'][0]:+.4f}, "
                     f"{a['slope_ci'][1]:+.4f}] | {a['slope_se_analytic']:.4f} | "
                     f"{'yes' if a['theory_in_ci'] else 'NO'} |")
    return "\n".join(L) + "\n"


def figure(res: dict, sweep: dict):
    """(a) QQ of standardized gamma-hat, (b) RMSE vs B, (c) coverage per cell."""
    from matplotlib.figure import Figure
    from scipy.stats import norm

    from tools.loglog_plot import _AQUA, _BLUE, _GRIDLINE, _MUTED, _ORANGE

    have_rate = bool(res["rate"].get("arms"))
    panels = 2 + have_rate
    fig = Figure(figsize=(4.4 * panels, 4.2), layout="constrained")
    axes = fig.subplots(1, panels)

    # (a) the cell with the most replicates (first on a tie).
    best = max(res["cells"], key=lambda c: c["clt"]["replicates"])
    raw = next(c for c in sweep["cells"] if not c["skipped"]
               and c["budget"] == best["budget"] and c["m0"] == best["m0"])
    v = np.asarray(raw[ESTIMATOR]["values"], float)
    z = np.sort((v - v.mean()) / best["clt"]["sd_pred"])
    q = norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
    ax = axes[0]
    lim = float(max(np.abs(q).max(), np.abs(z).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color=_MUTED, lw=1.5, label="N(0, 1)")
    ax.plot(q, z, "o", ms=4, color=_BLUE,
            label=rf"Var ratio {best['clt']['var_ratio']:.3f}, KS p {best['clt']['ks_p']:.2f}")
    ax.set_xlabel("normal quantile")
    ax.set_ylabel(r"$(\hat\gamma-\overline{\hat\gamma})\,/\,\mathrm{sd}_{(583)}$")
    ax.set_title(f"(a) CLT, eq. (583): B={best['budget']:.3g}, m0={best['m0']}, "
                 f"R={z.size}", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    # (b) RMSE vs B, each arm, with the predicted slope anchored at the first point.
    if have_rate:
        ax = axes[1]
        r = res["rate"]
        # Fixed categorical order (loglog_plot.py's slots 1-3); muted ink is the reference.
        colors = {"prop_opt": _BLUE, "tuned": _ORANGE, "best": _AQUA}
        for arm, a in r["arms"].items():
            ax.plot(a["budgets"], a["rmse"], "o-", color=colors[arm], lw=2, ms=6,
                    label=f"{arm}: slope {a['slope']:+.3f}")
        ref = next(iter(r["arms"].values()))
        Bs = np.asarray(ref["budgets"], float)
        ax.plot(Bs, ref["rmse"][0] * (Bs / Bs[0]) ** r["theory"], "--", color=_MUTED,
                lw=1.5, label=f"predicted slope {r['theory']:+.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("budget B")
        ax.set_ylabel(r"RMSE of $\hat\gamma$")
        ax.set_title("(b) error decay, prop:opt eq. (945)-(946)", fontsize=9)
        ax.legend(fontsize=8)

    # (c) coverage per cell.
    ax = axes[-1]
    cov = [c for c in res["cells"] if c["coverage"] is not None]
    if cov:
        p = np.array([c["coverage"]["coverage"] for c in cov])
        ci = np.array([c["coverage"]["coverage_ci"] for c in cov])
        x = np.arange(len(cov))
        # Clipped: the score interval's ends can sit a rounding error inside p.
        yerr = np.clip([p - ci[:, 0], ci[:, 1] - p], 0.0, None)
        ax.errorbar(x, p, yerr=yerr, fmt="o", ms=6, capsize=3,
                    color=_BLUE, label="coverage, Wilson 95% CI")
        ax.axhline(LEVEL, color=_MUTED, ls="--", lw=1.5, label=f"nominal {LEVEL:g}")
        ax.set_xticks(x)
        many = len(cov) > 6
        ax.set_xticklabels([f"B={c['budget']:.2g}{', ' if many else chr(10)}m0={c['m0']}"
                            for c in cov], fontsize=7, rotation=90 if many else 0)
        ax.set_ylim(min(0.5, float(ci[:, 0].min()) - 0.05), 1.02)
        ax.legend(fontsize=8, loc="lower left")
    else:
        ax.text(0.5, 0.5, "no eq. (720) bound:\n" + "\n".join(res["pilot"]["why"]),
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_ylabel(r"fraction with $|\hat\gamma-\gamma|\leq$ bound")
    ax.set_title("(c) eq. (720) coverage per cell", fontsize=9)

    for a in axes:
        a.grid(True, which="both", ls="-", lw=0.6, color=_GRIDLINE, alpha=0.8)
    fig.suptitle(f"{res['model']}: m={res['m']}, rho={res['rho']:g}", fontsize=10)
    return fig


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", required=True, type=Path,
                        help="allocation_sweep.json, or the run directory holding it")
    parser.add_argument("--pilot", required=True, type=Path,
                        help="pilot study directory (constants.json + pilot.json)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    path = artifact_path(args.sweep, "allocation_sweep") if args.sweep.is_dir() else args.sweep
    sweep = json.loads(path.read_text())
    res = score(sweep, pilot_inputs(args.pilot), seed=args.seed, n_boot=args.n_boot)
    res["sweep"] = str(path)
    print(markdown(res))

    out_dir = path.parent
    out = write_artifact(out_dir, "theorem_scores", res,
                         produced_by="calibration/score_theorems.py")
    png = artifact_path(out_dir, "theorem_scores_figure")
    figure(res, sweep).savefig(png, dpi=150)
    print(f"output = {out}\nfigure = {png}")


if __name__ == "__main__":
    _main()
