"""Step 4 of a study: the answer, with its error bar and a log-log plot.

Writes three things into the study directory:

  report.md    gamma-hat, its error, and the log-log plot -- the deliverable
  details.md   d, omega1, a1, a0, each with its own error and provenance
  answer.json  the same numbers, machine-readable
  plot.png     Y_bar_i vs i on log-log with the fitted slope

The error bar is where this file spends most of its care, because getting it
wrong is invisible. Two are reported, and they answer different questions:

  replicate interval   gamma +/- t(R-1) * sd(fits)/sqrt(R).
                       The t quantile, NOT the normal one: at R = 5,
                       P(|t_4| < 1.96) = 0.8784, so pairing a 5-point standard
                       error with 1.96 gives an interval labelled 95% that
                       covers 88%. This repo published that mistake until
                       calibration/check_coverage.py measured it.

  Wilson bound         Theorem thm:wilson (eq. 720), for gamma only. A BOUND,
                       so it overcovers, but its width comes from a closed form
                       -- sigma_se = sqrt(12*sigma_inf^2/(n*m^3)) -- rather than
                       from a 5-point spread, so it needs no t widening and it
                       carries the finite-size bias the replicate interval has
                       no term for.

Disagreement between them is informative rather than alarming: the replicate
interval measures scatter and ignores bias; the bound covers both.

CLI:
    python3 src/study/report.py --study mystudy --data-root experiments/01_srw/data
    python3 src/study/report.py --study mystudy --data-root ... --budget-analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "budget"))

from artifacts import artifact_path, write_artifact  # noqa: E402
from constants import format_table, load, measured  # noqa: E402
from correction import fit_correction  # noqa: E402
from coverage import interval  # noqa: E402
from loglog import gamma_all_points, gamma_closed_form  # noqa: E402
from loglog_plot import loglog_plot  # noqa: E402
from wilson import sigma_se  # noqa: E402

from allocation_table import human_time  # noqa: E402

LEVEL = 0.95


def analyse(final: dict, level: float = LEVEL) -> dict:
    """gamma-hat and its error from a completed run's per-replicate summaries."""
    scales = final["scales"]
    rho, m, m0, n = final["rho"], final["m"], final["m0"], final["n"]
    reps = final["per_replicate"]
    R = len(reps)

    y = np.array([r["y_bar"] for r in reps], float)
    sig = np.array([r["sigma_log"] for r in reps], float)
    cv = np.array([r["cv"] for r in reps], float).mean(axis=0)

    # Pooled then estimated once -- the same rule as everywhere else here: the
    # fit is nonlinear, so averaging R separate fits converges to E[a1_hat]
    # rather than a1, a bias no number of replicates removes.
    y_pool = y.mean(axis=0)
    sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))

    gammas = [float(gamma_closed_form(scales, yy, rho, m0)) for yy in y]
    gamma = float(gamma_closed_form(scales, y_pool, rho, m0))
    se = float(np.std(gammas, ddof=1) / sqrt(R)) if R > 1 else None

    # t(R-1), never the normal quantile: see this module's docstring.
    lo, hi, dof = None, None, None
    if se is not None:
        dof = R - 1
        lo, hi = interval(gamma, se, level=level, dof=dof)

    fit = fit_correction(scales, y_pool, sigma_log=sig_pool)

    # Wilson's fourth term, from the closed form rather than a 5-point spread.
    sigma_inf2 = float(np.mean(cv ** 2))
    w_sd = sigma_se(n, m, rho, sigma_inf2)      # (n, m, rho, sigma_inf2) -- not cv first

    sd_over_reps = float(np.std(gammas, ddof=1)) if R > 1 else None
    return {"gamma": gamma, "se": se, "level": level, "replicates": R,
            "sd_over_reps": sd_over_reps,
            "ci": [lo, hi], "dof": dof, "per_replicate_gamma": gammas,
            "gamma_all_points": float(gamma_all_points(scales, y_pool)),
            "fit": fit, "scales": scales, "n": n, "m0": m0, "m": m, "rho": rho,
            "y_bar": y_pool.tolist(), "sigma_log": sig_pool.tolist(),
            "cv": cv.tolist(), "sigma_inf2": sigma_inf2, "wilson_sd": w_sd}


def _fmt(v, se=None, digits=4):
    """Value +/- error, with enough decimals for the error to be visible.

    A planned run reaches se ~ 4e-05, which a fixed 4-decimal format renders as
    "+0.5009 +/- 0.0000" -- an error bar printed as zero is worse than none.
    """
    if se is not None and se > 0:
        digits = max(digits, int(-np.floor(np.log10(se))) + 1)
    digits = min(digits, 12)
    s = f"{v:+.{digits}f}"
    return s + (f" ± {se:.{digits}f}" if se is not None else "")


def write_report(sd: Path, res: dict, final: dict, consts: dict, plan: dict) -> Path:
    """report.md -- gamma, its error, the plot. The thing you show someone."""
    R = res["replicates"]
    lo, hi = res["ci"]
    lines = [
        f"# gamma for `{final['model']}`", "",
        f"## {_fmt(res['gamma'], res['se'])}", "",
    ]
    if lo is not None:
        lines += [f"{int(res['level'] * 100)}% interval **[{lo:.5f}, {hi:.5f}]**, "
                  f"from {R} replicates using the t({res['dof']}) quantile.", "",
                  "> The t quantile, not the normal one. At R = 5, "
                  "P(|t_4| < 1.96) = 0.8784, so a 5-point standard error paired with "
                  "1.96 gives an interval labelled 95% that covers 88%.", ""]
    else:
        lines += ["No interval: one replicate gives no spread. "
                  "Re-plan with `--replicates 3` or more.", ""]
    bias_pred = plan.get("bias")
    if bias_pred is not None and res["se"] and bias_pred > 0.5 * res["se"]:
        lines += [
            f"> **The interval above has no bias term.** It is the scatter of "
            f"{R} replicates around their own mean, and the plan predicted a "
            f"finite-size bias of {bias_pred:.3g} against a replicate se of "
            f"{res['se']:.3g} -- comparable or larger. A bias shifts every "
            f"replicate the same way, so no number of them reveals it, and the "
            f"interval can exclude the truth while looking tight. "
            f"Deepen `m0` (raise the budget, or re-plan on a better pilot) if "
            f"this matters.", ""]

    lines += [
        "![log-log](plot.png)", "",
        "## How it was measured", "",
        f"| | |", "|---|---|",
        f"| estimator | article eq. (523)-(526), closed-form weights |",
        f"| scales | {res['scales']} |",
        f"| n per scale | {res['n']:,} |",
        f"| m0, m, rho | {res['m0']}, {res['m']}, {res['rho']} |",
        f"| replicates | {R} |",
        f"| wall clock | {human_time(final['elapsed_seconds'])} |",
        "",
        f"Cross-checks: the all-points OLS slope gives "
        f"{res['gamma_all_points']:+.5f}; the eq. (232) fit gives "
        f"{res['fit']['gamma']:+.5f}.", "",
        f"The closed-form sd of a single estimate (eq. 583/720, "
        f"`sqrt(12*sigma_inf2/(n*m^3))/log(rho)`, no replicates involved) is "
        f"**{res['wilson_sd']:.3g}**"
        + (f", against **{res['sd_over_reps']:.3g}** measured as the spread of "
           f"{R} replicates." if res.get("sd_over_reps") else ".")
        + (f" A spread from {R} replicates carries ~{1 / sqrt(2 * (R - 1)):.0%} "
           f"relative sd of its own, so treat a factor-of-two disagreement as "
           f"uninformative and only a persistent one as real."
           if R > 1 else ""),
        "",
        "Full constants and their provenance: `details.md`.", "",
    ]
    p = sd / "report.md"
    p.write_text("\n".join(lines))
    return p


def write_details(sd: Path, res: dict, final: dict, consts: dict, plan: dict) -> Path:
    """details.md -- every constant, its error, and where it came from."""
    fit = res["fit"]
    lines = [
        f"# Details -- `{final['model']}`", "",
        "## Constants from the final run", "",
        "Refitted on the pooled data, so these describe the long run, not the pilot.",
        "", "| constant | value | from |", "|---|---|---|",
        f"| gamma | {_fmt(res['gamma'], res['se'])} | eq. (526) weights, "
        f"{res['replicates']} replicates |",
        f"| omega1 | {_fmt(fit['omega1'])} | eq. (232) fit, pooled |",
        f"| a1 | {_fmt(fit['a1'])} | eq. (232) fit, pooled |",
        f"| a0 | {_fmt(fit['a0'])} | eq. (232) fit, pooled |",
        f"| cv | {np.mean(res['cv']):+.4f} | mean over {len(res['cv'])} scales |",
        "",
        f"Fit quality: relative RMSE {fit['rel_rmse']:.3g}, "
        f"converged = {fit.get('converged')}.", "",
        "## Constants the plan was built on (from the pilot)", "",
        "```", format_table(consts), "```", "",
        "These sized the run; the table above is what the run then measured. "
        "Large disagreement means the pilot was not representative -- worth "
        "knowing before quoting the result.", "",
        "## Per-replicate gamma", "",
        "```",
        "\n".join(f"  rep {i}: {g:+.6f}"
                  for i, g in enumerate(res["per_replicate_gamma"])),
        "```", "",
        "## Provenance", "",
        f"- pilot: `{artifact_path(sd, 'pilot').name}`",
        f"- plan: `{artifact_path(sd, 'plan').name}`",
        f"- run: `{artifact_path(sd, 'final').name}` "
        f"({'samples kept' if final.get('samples_kept') else 'summaries only'})",
        f"- seeds: `{final.get('seeds')}`", "",
    ]
    p = sd / "details.md"
    p.write_text("\n".join(lines))
    return p


def write_budget_analysis(sd: Path, res: dict, final: dict, plan: dict) -> Path:
    """Optional: did the run cost and deliver what the plan promised?"""
    got_t = final["elapsed_seconds"]
    want_t = plan.get("total_seconds")
    want_e = plan.get("rmse")
    got_e = res["se"] * sqrt(res["replicates"]) if res["se"] else None
    lines = [
        "# Budget analysis", "",
        "| | predicted | measured | ratio |", "|---|---|---|---|",
        f"| wall clock | {human_time(want_t) if want_t else '--'} | "
        f"{human_time(got_t)} | "
        f"{f'{got_t / want_t:.2f}x' if want_t else '--'} |",
        f"| se(gamma), per replicate | {want_e:.4g} | "
        f"{f'{got_e:.4g}' if got_e else '--'} | "
        f"{f'{got_e / want_e:.2f}x' if got_e and want_e else '--'} |",
        "",
        f"An se estimated from R = {res['replicates']} replicates carries "
        f"~{1 / sqrt(2 * res['replicates']):.0%} relative sd of its own, so read the "
        f"second row as an order-of-magnitude check, not a calibration.", "",
        "## Where the budget went", "",
        "```",
        f"  m0 = {res['m0']}   scales {res['scales']}",
        f"  n  = {res['n']:,} per scale x {res['replicates']} replicates",
        f"  the deepest scale takes {res['scales'][-1] / sum(res['scales']):.0%} "
        f"of one replicate's cost",
        "```", "",
    ]
    p = sd / "budget_analysis.md"
    p.write_text("\n".join(lines))
    return p


def _plot(sd: Path, res: dict, final: dict) -> Path:
    """The log-log chart, from the run's summaries rather than its samples.

    Uses tools/loglog_plot.py's shared chart, not a private one -- the summary
    triple (y_bar, se, n) is exactly what `loglog_points` accepts. The fitted
    line is the closed-form gamma anchored at the eq. (232) fit's a0, so the
    slope drawn is the slope reported.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a0, gamma = res["fit"]["a0"], res["gamma"]
    summary = {i: (y, y * s, res["n"])
               for i, y, s in zip(res["scales"], res["y_bar"], res["sigma_log"])}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    loglog_plot(summary, ax=ax,
                fit_fn=lambda x: a0 * np.asarray(x, float) ** gamma,
                fit_label=rf"$\hat\gamma = {gamma:.5f}$")
    ax.set_title(f"{final['model']}  --  {res['replicates']} replicate(s), "
                 f"n = {res['n']:,} per scale")
    fig.tight_layout()
    out = sd / "plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--level", type=float, default=LEVEL)
    p.add_argument("--budget-analysis", action="store_true",
                   help="also write budget_analysis.md: predicted vs actual cost")
    a = p.parse_args(argv)

    sd = Path(a.data_root) / a.study
    fj = artifact_path(sd, "final")
    if not fj.exists():
        raise SystemExit(
            f"no final.json in {sd}\n"
            f"  Run the plan first:\n"
            f"    python3 src/study/run.py --study {a.study} --data-root {a.data_root}")
    final = json.loads(fj.read_text())
    plan = final.get("plan", {})
    consts = load(sd)

    res = analyse(final, level=a.level)

    fig_path = _plot(sd, res, final)

    write_artifact(sd, "answer", res, produced_by="src/study/report.py")
    rp = write_report(sd, res, final, consts, plan)
    dp = write_details(sd, res, final, consts, plan)

    print(f"gamma = {_fmt(res['gamma'], res['se'])}")
    if res["ci"][0] is not None:
        print(f"  {int(a.level * 100)}% CI [{res['ci'][0]:.5f}, {res['ci'][1]:.5f}]  "
              f"(t({res['dof']}) quantile, {res['replicates']} replicates)")
    print(f"\n  {rp}\n  {dp}\n  {fig_path}")
    if a.budget_analysis:
        print(f"  {write_budget_analysis(sd, res, final, plan)}")


if __name__ == "__main__":
    _main()
