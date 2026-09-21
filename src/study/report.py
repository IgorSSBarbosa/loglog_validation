"""Step 4 of a study: the answer, with its error bar and a log-log plot.

Writes three things into the study directory:

  report.md    gamma-hat, its error, and the log-log plot -- the deliverable
  details.md   d, omega1, a1, a0, each with its own error and provenance
  answer.json  the same numbers, machine-readable
  plot.png     Y_bar_i vs i on log-log with the fitted slope

The error bar is where this file spends most of its care, because getting it
wrong is invisible. Two are reported, and they answer different questions. The
eq. (720) bound LEADS whenever it can be assembled: the replicate interval is a
statement about R numbers scattering around their own mean, the bound is a
statement about gamma. This repo has measured a study whose t interval was
[0.50119, 0.50270] -- excluding the true 1/2 -- while the bound carrying the
same run's bias covered it. Leading with the tighter number was leading with
the one that can be confidently wrong.

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

The bound is only as good as the pilot's omega1, and not linearly so: omega1
sits in an exponent, B_fs ~ rho**(-omega1*m0). Measured here, a pilot giving
omega1 = 13.1 +/- 3.5 produced B_fs = 9.4e-07 and a bound that EXCLUDED the
truth while printing tight; the same run under omega1 = 0.885 +/- 0.125 gives
B_fs = 4.8e-03 and covers it. So B_fs is recomputed across omega1 +/- 1 se and
the span is reported whenever it exceeds `_BFS_SPAN_LIMIT`. A bound whose bias
term moves two orders of magnitude inside its own input's error bar is a
statement about the pilot, not about gamma.

CLI:
    python3 src/study/report.py --study mystudy --data-root experiments/01_srw/data
    python3 src/study/report.py --study mystudy --data-root ... --budget-analysis
    python3 src/study/report.py --study mystudy --data-root ... --partial

--partial reads `final_partial.json`, the replicates a run still in progress
has finished, and only PRINTS the answer: report.md, answer.json and plot.png
are the finished run's and are not written. It is safe while run.py or
autopilot.py is drawing -- it reads one file and touches nothing else.
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
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, write_artifact  # noqa: E402
from tools.constants import format_table, load, measured  # noqa: E402
from tools.correction import fit_correction  # noqa: E402
from tools.coverage import interval  # noqa: E402
from tools.loglog import gamma_all_points, gamma_closed_form  # noqa: E402
from tools.loglog_plot import loglog_plot  # noqa: E402
from tools.summary import LOG_MOMENT_DELTA  # noqa: E402
from tools.wilson import (  # noqa: E402
    finite_size_bias, sigma_se, wilson_interval)

from src.budget.allocation_table import human_time  # noqa: E402

LEVEL = 0.95

#: How far B_fs may swing across omega1 +/- 1 se before the bound is called
#: undetermined. 10x is generous -- a bound is allowed to be loose -- but past
#: it the number being printed is the pilot's guess about an exponent, not a
#: statement about gamma.
_BFS_SPAN_LIMIT = 10.0


def wilson_inputs(final: dict, cv: np.ndarray, reps: list) -> dict:
    """The eq. (720) constants, each from the only place it can honestly come from.

    omega1 and a1 come from the PILOT, never from a refit on the final data,
    and this is not a preference. The plan deepens m0 precisely to put the
    ladder where the correction has died -- that is what makes gamma-hat
    unbiased -- so refitting omega1 there fits noise. On this repo's own srw
    run the final refit returned omega1 = 0.0295 with converged = False, and
    feeding that to B_fs turns a half-width of 1.1e-3 into 1.5. The pilot,
    which deliberately ladders down to the small scales, measured 0.885.

    sigma_inf2 and sigma_max2 come from cv, which IS a function of the sample
    mean and so survives summarization; Lambda comes from the log_moment field
    (`tools/summary.py`), absent from any run drawn before that field existed.

    Returns the kwargs `wilson_interval` wants, plus `why` naming whatever is
    missing. A missing piece is reported, never defaulted: eq. (720) with a
    term silently set to zero is not a bound.
    """
    why = []
    c = (final.get("plan") or {}).get("constants_at_plan_time") or {}
    omega1 = (c.get("omega1") or {}).get("value")
    a1 = (c.get("a1") or {}).get("value")
    omega1_se = (c.get("omega1") or {}).get("se")
    if omega1 is None or a1 is None:
        why.append("omega1/a1 (the plan recorded no constants_at_plan_time)")

    lm = [r.get("log_moment") for r in reps]
    if any(x is None for x in lm):
        Lambda = None
        why.append("Lambda (this run predates the log_moment summary field)")
    else:
        # Lambda is max_k E|log xi_k|^(2+delta): an expectation, so replicates
        # average; the max is over scales.
        Lambda = float(np.mean(np.array(lm, float), axis=0).max())

    return {
        "sigma_inf2": float(cv[-1] ** 2),   # largest scale -- moment_bounds' definition
        "sigma_max2": float((cv ** 2).max()),
        "omega1": omega1, "a1": a1, "omega1_se": omega1_se,
        "Lambda": Lambda, "delta": LOG_MOMENT_DELTA if Lambda is not None else None,
        "why": why,
    }


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
    # sigma_inf2 is cv^2 at the LARGEST scale, which is `moment_bounds`'
    # definition and Assumption 6's honest finite-sample stand-in for the
    # limit -- not a mean over scales, which would average the limit together
    # with the scales still converging to it.
    wi = wilson_inputs(final, cv, reps)
    sigma_inf2 = wi["sigma_inf2"]
    w_sd = sigma_se(n, m, rho, sigma_inf2)      # (n, m, rho, sigma_inf2) -- not cv first

    # Theorem thm:wilson (eq. 720). A BOUND on |gamma_hat - gamma|, so it
    # overcovers; unlike the replicate interval it carries the finite-size
    # bias, which is exactly the term R replicates can never reveal.
    wilson = None
    if not wi["why"]:
        wilson = wilson_interval(gamma, n, m, m0, rho,
                                 sigma_inf2=wi["sigma_inf2"], sigma_max2=wi["sigma_max2"],
                                 a1=wi["a1"], omega1=wi["omega1"],
                                 Lambda=wi["Lambda"], delta=wi["delta"], level=level)
    wilson_why = wi["why"]

    # B_fs = |a1| * rho**(-omega1*m0) / (rho**omega1 - 1): omega1 sits in an
    # EXPONENT, so the pilot's error bar on it does not propagate linearly.
    # Measured on a deliberately bad pilot (omega1 = 13.1 +/- 3.5), B_fs ranged
    # over eleven orders of magnitude inside +/- 2 se, and the bound it produced
    # excluded the truth while looking tight. So the span is computed and
    # reported rather than left for the reader to wonder about: a bias term
    # this sensitive to its input is not a bound, whatever it prints.
    b_span = None
    if wilson is not None and wi["omega1_se"]:
        lo_w = max(1e-3, wi["omega1"] - wi["omega1_se"])
        hi_w = wi["omega1"] + wi["omega1_se"]
        b_span = sorted(finite_size_bias(m, m0, rho, wi["a1"], w)
                        for w in (lo_w, hi_w))

    sd_over_reps = float(np.std(gammas, ddof=1)) if R > 1 else None
    return {"gamma": gamma, "se": se, "level": level, "replicates": R,
            "sd_over_reps": sd_over_reps,
            "ci": [lo, hi], "dof": dof, "per_replicate_gamma": gammas,
            "gamma_all_points": float(gamma_all_points(scales, y_pool)),
            "fit": fit, "scales": scales, "n": n, "m0": m0, "m": m, "rho": rho,
            "y_bar": y_pool.tolist(), "sigma_log": sig_pool.tolist(),
            "cv": cv.tolist(), "sigma_inf2": sigma_inf2, "wilson_sd": w_sd,
            "sigma_max2": wi["sigma_max2"], "Lambda": wi["Lambda"],
            "wilson": wilson, "wilson_why": wilson_why,
            "wilson_bfs_span": b_span, "omega1_se": wi["omega1_se"]}


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
    w = res.get("wilson")
    pct = int(res["level"] * 100)

    # eq. (720) leads, when there is one. The replicate interval is a
    # statement about R numbers scattering around their own mean; the bound is
    # a statement about gamma. Measured here (src/study/README.md): a study
    # whose t interval was [0.50119, 0.50270], excluding the true 1/2, while
    # the bound that carries the same run's bias covered it. Leading with the
    # tighter number was leading with the one that can be confidently wrong.
    lines = [f"# gamma for `{final['model']}`", ""]
    if w is not None:
        wlo, whi = w["interval"]
        lines += [
            f"## {_fmt(res['gamma'], w['half_width'])}", "",
            f"{pct}% **[{wlo:.5f}, {whi:.5f}]** -- article eq. (720), Theorem "
            f"thm:wilson. A **bound** on |gamma_hat - gamma|, so it overcovers, "
            f"and unlike the replicate interval below it carries the "
            f"finite-size bias. Dominated by `{w['dominant']}`.", "",
        ]
        if lo is not None:
            lines += [
                f"For comparison, the Student-t replicate interval is "
                f"[{lo:.5f}, {hi:.5f}] ({_fmt(res['gamma'], res['se'])}, "
                f"t({res['dof']}) on {R} replicates). It is **scatter only**: "
                f"a finite-size bias shifts every replicate the same way, so no "
                f"number of them reveals it, and that interval can exclude the "
                f"truth while looking tight. This repo has measured it doing "
                f"exactly that.", ""]
    else:
        lines += [f"## {_fmt(res['gamma'], res['se'])}", ""]
        if lo is not None:
            lines += [f"{pct}% interval **[{lo:.5f}, {hi:.5f}]**, from {R} "
                      f"replicates using the t({res['dof']}) quantile.", "",
                      "> The t quantile, not the normal one. At R = 5, "
                      "P(|t_4| < 1.96) = 0.8784, so a 5-point standard error "
                      "paired with 1.96 gives an interval labelled 95% that "
                      "covers 88%.", ""]
        else:
            lines += ["No interval: one replicate gives no spread. "
                      "Re-plan with `--replicates 3` or more.", ""]
        if res.get("wilson_why"):
            lines += [f"> **This is the scatter interval, and it has no bias "
                      f"term.** The eq. (720) bound, which would carry one, "
                      f"could not be assembled: {', '.join(res['wilson_why'])}.",
                      ""]
    if w is not None:
        lines += [
            "| term | value | what it is |",
            "|---|---|---|",
            f"| B_fs | {w['B_fs']:.3g} | finite-size bias, from the pilot's "
            f"omega1/a1 |",
            f"| B_good | {w['B_good']:.3g} | Jensen bias of the log |",
            f"| B_bad | {w['B_bad']:.3g} | the stray-sample-mean event, needs "
            f"Lambda = {res['Lambda']:.3g} |",
            f"| {w['quantile']:.3g}·sigma_se | {w['se_term']:.3g} | the only "
            f"random term; sigma_se is a closed form, so no t widening |",
            "",
            f"Both intervals are correct; they answer different questions. "
            f"This one BOUNDS scatter and bias together and therefore "
            f"overcovers. The replicate interval measures SCATTER alone and "
            f"has no bias term, so it is the tighter of the two whenever the "
            f"bias is real. Disagreement between them is informative, not "
            f"alarming -- it is the size of the bias.", "",
        ]
        if not w["complete"]:
            lines += [f"> Incomplete bound: {', '.join(w['missing_terms'])}. "
                      f"A bound missing a term is not a bound -- the half-width "
                      f"above is a LOWER estimate of the true one.", ""]
        sp = res.get("wilson_bfs_span")
        if sp and sp[0] > 0 and sp[1] / sp[0] > _BFS_SPAN_LIMIT:
            lines += [
                f"> **B_fs is not determined by this pilot.** omega1 sits in an "
                f"exponent, so moving it by its own +/- 1 se "
                f"({res['omega1_se']:.3g}) swings B_fs across "
                f"[{sp[0]:.3g}, {sp[1]:.3g}] -- a factor of {sp[1] / sp[0]:.3g}. "
                f"The bound above uses the central value and can be far too "
                f"narrow; on this repo's own test of a loose pilot it excluded "
                f"the truth while printing a tight interval. Deepen the pilot "
                f"(more replicates, and a ladder reaching down to scales where "
                f"the correction is still visible) before quoting it.", ""]
    elif res.get("wilson_why"):
        lines += [f"No eq. (720) bound: {', '.join(res['wilson_why'])}.", ""]

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


def _d_check_section(sd: Path) -> list[str]:
    """The pilot's declared-vs-measured d check, reprinted in the deliverable.

    A mismatch warns on the pilot's console, which is hours and several steps
    upstream of this file. Reprinting it here is the point: d sizes every
    wall-clock prediction in the study, so a disagreement between the clock and
    a model's declared cost has to survive all the way to the document someone
    actually reads.
    """
    pj = artifact_path(sd, "pilot")
    if not pj.exists():
        return []
    chk = (json.loads(pj.read_text()).get("cost") or {}).get("d_check")
    if not chk or chk.get("declared") is None:
        return []
    verdict, z, thr = chk.get("verdict"), chk.get("z"), chk.get("threshold")
    zs = f"{z:+.2f}" if z is not None else "n/a"
    flag = "" if verdict == "pass" else "  **<-- CHECK THIS**"
    # The cutoff is printed because it is NOT 3: it is the t quantile at
    # D_MISMATCH_Z's alpha, and it moves with how many probes the se came from.
    # A reader comparing z against 3 in their head would misread the verdict.
    cut = f"+/-{thr:.2f}" if thr is not None else "n/a"
    return [
        "## Cost exponent: declared vs measured", "",
        "| | |", "|---|---|",
        f"| measured (clock) | {chk.get('measured')} +/- {chk.get('d_se')} |",
        f"| declared (model cost_hint or --assert-d) | {chk.get('declared')} |",
        f"| se from | {chk.get('se_source')} |",
        f"| z | {zs} |",
        f"| cutoff | {cut} |",
        f"| verdict | {verdict}{flag} |",
        "",
        "`d` is the MEASURED value in every case; the declaration is only ever "
        "scored against it. A mismatch means either the declared cost is wrong "
        "or the machine was not compute-bound while the probe ran -- both make "
        "the wall-clock predictions in this study unreliable, even though the "
        "gamma estimate above is unaffected.", "",
    ]


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
        *_d_check_section(sd),
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
    triple (y_bar, se, n) is exactly what `loglog_points` accepts.

    The line is the REPORTED gamma, anchored on the data's own sigma_log-
    weighted centroid. It is deliberately not `fit["a0"] * i**gamma`, which is
    what this drew until 2026-09-04 and which sat 13-15% below every point:
    eq. (232) is a0 * i**gamma * exp(a1 * i**-omega1), so a0 alone is the curve
    only once the correction has vanished, and when omega1 comes back near zero
    (as it must on a ladder chosen to have no correction left -- see
    `wilson_inputs`) the term a1*i**-omega1 is nearly constant and therefore
    degenerate with log a0. The fit then splits the true prefactor between
    them: on this repo's srw run, a0 = 0.6703 against a truth of 0.7979, with
    the missing factor 1.15 hiding in exp(a1*i**-omega1).

    Anchoring instead of fitting also keeps the drawn line honest about which
    estimator it depicts. The reported gamma comes from eq. (526)'s weights,
    which sum to zero and so annihilate the intercept exactly -- a0 is not
    something that estimator has an opinion about, and borrowing one from a
    different (nonlinear, 4-parameter) fit mixed two estimators in one line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gamma = res["gamma"]
    scales = np.asarray(res["scales"], float)
    log_y = np.log(np.asarray(res["y_bar"], float))
    w = 1.0 / np.asarray(res["sigma_log"], float) ** 2
    a0 = float(np.exp(np.sum(w * (log_y - gamma * np.log(scales))) / np.sum(w)))
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


def print_answer(res: dict, *, log=print) -> None:
    """The console answer: eq. (720) first, the scatter interval below it.

    Factored out of _main so src/study/autopilot.py prints exactly what
    report.py prints -- a driver that paraphrases its own steps is a second
    place for the wording of a result to drift.
    """
    pct = int(res["level"] * 100)
    w = res.get("wilson")
    if w is not None:
        log(f"gamma = {_fmt(res['gamma'], w['half_width'])}")
        log(f"  {pct}% [{w['interval'][0]:.5f}, {w['interval'][1]:.5f}]  "
              f"eq. (720) bound -- scatter AND bias"
              + ("" if w["complete"] else ", INCOMPLETE"))
    else:
        log(f"gamma = {_fmt(res['gamma'], res['se'])}")
    if res["ci"][0] is not None:
        log(f"  {pct}% [{res['ci'][0]:.5f}, {res['ci'][1]:.5f}]  "
              f"Student t({res['dof']}), {res['replicates']} replicates "
              f"-- scatter only, no bias term")
    if w is not None:
        sp = res.get("wilson_bfs_span")
        if sp and sp[0] > 0 and sp[1] / sp[0] > _BFS_SPAN_LIMIT:
            log(f"  !! B_fs spans [{sp[0]:.2g}, {sp[1]:.2g}] over omega1 +/- 1 se "
                  f"-- the bound's bias term is not determined by this pilot")
    elif res.get("wilson_why"):
        log(f"  no eq. (720) bound ({', '.join(res['wilson_why'])}) -- the "
              f"interval above has no bias term")


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--level", type=float, default=LEVEL)
    p.add_argument("--budget-analysis", action="store_true",
                   help="also write budget_analysis.md: predicted vs actual cost")
    p.add_argument("--partial", action="store_true",
                   help="a preliminary answer from the replicates a run in progress "
                        "has finished (final_partial.json); prints only, writes nothing")
    a = p.parse_args(argv)

    sd = Path(a.data_root) / a.study
    if a.partial:
        pj = artifact_path(sd, "final_partial")
        if not pj.exists():
            raise SystemExit(
                f"no final_partial.json in {sd}\n"
                f"  It is written when the run's first replicate finishes. A run "
                f"started\n  before checkpointing existed never writes one.")
        partial = json.loads(pj.read_text())
        done, planned = partial["replicates"], partial.get("replicates_planned", "?")
        print(f"PRELIMINARY -- {done} of {planned} replicate(s), "
              f"checkpoint written {partial.get('created', '?')}")
        if done < 2:
            print("  one replicate has no spread: the replicate interval is "
                  "unavailable until the second finishes")
        print_answer(analyse(partial, level=a.level))
        return

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

    print_answer(res)
    print(f"\n  {rp}\n  {dp}\n  {fig_path}")
    if a.budget_analysis:
        print(f"  {write_budget_analysis(sd, res, final, plan)}")


if __name__ == "__main__":
    _main()
