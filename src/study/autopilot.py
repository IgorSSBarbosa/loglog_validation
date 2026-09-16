"""One command for the whole study: pilot -> plan -> run -> report.

    python3 src/study/autopilot.py -meta <recipe> --study <name> --time 2h

The four steps still exist and are still the thing to reach for when you want
to look at a verdict before spending an afternoon. This driver is for when you
do not: it makes the decisions a human makes between them, and says which ones
it made.

WHAT IT DECIDES, AND ON WHAT

`--time` is the TOTAL. The pilot is not free, so it is budgeted rather than
assumed: it takes what it needs out of the total (capped at --pilot-cap) and
the final run gets what is left, measured rather than predicted.

The pilot doubles its DRAWS. Round 1 takes `--replicates` replicates at the
recipe's own n; if the constants are not good enough, round 2 redraws the same
number of replicates with twice the n each, round 3 with four times, and so on.
Total samples per scale go nR, 2nR, 4nR, ... Doubling means the whole search
costs at most twice what starting at the right size would have, and the SCALES
never change -- they are the recipe's, and a ladder is a modelling decision
this script has no business making.

Doubling n per replicate rather than the replicate COUNT (Igor, 2026-09-04).
Both routes shrink what the gate reads, `se = std(per-replicate fits)/sqrt(R)`:
more replicates leaves each fit's own error alone and divides by a larger
sqrt(R); more draws per replicate shrinks each fit's error with R fixed. The
draws axis turns out to be the better of the two on the statistics AND the
cheaper on compute, so nothing is traded away.

Measured on planted srw, 400 independent studies per arm, scales 8..256,
baseline 3 replicates at n_k = (2e5 ... 6.4e6):

    arm                    total draws   mean se(omega1)   mean omega1
    R=3,  n  (baseline)       3.78e+07           0.1585        1.0125
    R=6,  n  (double R)       7.56e+07           0.1207        1.0326
    R=3, 2n  (double n)       7.56e+07           0.1098        1.0031
    R=12, n  (4x R)           1.51e+08           0.0879        1.0103
    R=3, 4n  (4x n)           1.51e+08           0.0769        0.9989

At equal total draws the draws axis gives a 9-13% SMALLER se, and it is the
only one that also moves the point estimate toward the truth (1.003 and 0.999
against 1.033 and 1.010). Both follow from `fit_correction` being nonlinear:
more draws per fit shrink that fit's own dispersion and its nonlinear bias
together, while more fits at the same n average a distribution that stays
exactly as wide and as skewed as it was.

What differs in cost is the fitting. Every replicate needs its own
`fit_correction`, a non-convex four-parameter least-squares restarted from
five omega1 seeds, and the count of those grows with R while the cost of each
is flat in n. At --replicates 10, doubling R three times means
10 + 20 + 40 + 80 = 150 multi-restart fits; doubling n means 40 -- and the
sampling, which dominates either way, is identical.

Each round is a CLEAN redraw at one n, not an extension of the previous one.
Replicates drawn at different n cannot be pooled by the equal-weight rule
`pilot` uses, and the geometric argument already pays for discarding: 1 + 2 +
4 + ... + 2^k < 2 * 2^k, so throwing away every earlier round still costs less
than twice the final one. Rounds draw from independent streams (ground rule 2).

The replicate COUNT is therefore yours to choose and is never changed here.
It fixes two things this loop cannot improve: the dof of the t quantile the
final report uses, and how well `std(fits)` estimates the spread it stands for
(its own relative sd is ~1/sqrt(2(R-1)) -- 50% at R = 3, 24% at R = 10). Use
--replicates 5 or more if you intend to act on the gate's verdict.

THE GATE is the B_fs span, not the "is se small" verdict plan.py prints for a
human. eq. (720)'s finite-size term is B_fs ~ rho**(-omega1*m0), so omega1
sits in an EXPONENT and its error bar does not propagate linearly. Measured
(src/study/README.md): a pilot at omega1 = 13.1 +/- 3.5 passed plan.py's
checks, and the bound it produced was tight and EXCLUDED the truth. The span
of B_fs across omega1 +/- 1 se is the thing that caught it, so it is what
decides here. m0 comes from planning at the currently-remaining budget, which
is the m0 the answer would actually be computed at.

WHEN IT GIVES UP it prints the constants it did measure with their intervals,
the span that failed, and what to change -- and draws nothing. Spending the
rest of the budget on constants known to be undetermined produces a confident
wrong answer, which is worse than no answer.

--force overrides that last step and only that step. The diagnosis is printed
identically; what changes is that planning, drawing and reporting go ahead
afterwards, and the result carries `forced: true` in autopilot.json plus a
closing warning on the console. Use it when you want the number anyway and
have read why it is soft.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import (  # noqa: E402
    default_out_dir, load_recipe, read_artifact, write_artifact)
from tools.constants import format_table  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools import progress as P  # noqa: E402
from tools.rng import seed_record, spawn  # noqa: E402
from tools.loglog import gamma_all_points  # noqa: E402
from tools.wilson import finite_size_bias  # noqa: E402

from src.budget.allocation_table import human_time  # noqa: E402
from src.study import pilot as pilot_mod  # noqa: E402
from src.study import plan as plan_mod  # noqa: E402
from src.study import report as report_mod  # noqa: E402
from src.study import run as run_mod  # noqa: E402

#: Fraction of --time the pilot may consume before the loop gives up. The
#: pilot buys the constants the whole plan rests on, so it is worth real
#: money; past a quarter of the budget the honest reading is that the recipe,
#: not the sample count, is what needs fixing.
PILOT_CAP = 0.25

#: Rounds are geometric, so this is generous: round 6 draws 32x the recipe's
#: n per replicate, and the whole search has then cost under 64x round 1.
MAX_ROUNDS = 6

#: p-value above which a pure power law is an adequate fit to the pilot, and
#: omega1 therefore has nothing to be fitted to. Loose on purpose (0.01, not
#: 0.05): this gate refuses to spend a budget, so it should fire only when the
#: correction is genuinely invisible, not merely weak. See `ladder_check`.
LADDER_P_MAX = 0.01


def ladder_check(reps, scales) -> dict:
    """Is a pure power law an adequate fit? If it is, omega1 is unmeasurable.

    omega1 is estimated from the DEPARTURE of E Y_i from a straight line in
    log-log. If the data is consistent with no departure at all, there is
    nothing on this ladder to fit, and the four-parameter fit will wander into
    one of its two degenerate corners: omega1 -> 0 with a1 absorbing the
    intercept, or omega1 large with a1 -> +-inf. Both have been measured here.

    So this is a goodness-of-fit test on the two-parameter model:

        chi2 = sum_k (resid_k / sigma_log_k)**2,   dof = len(scales) - 2

    over ALL scales, not the residual at the smallest one. A single point's
    |resid|/sigma tops 1 about 10% of the time under the null (measured; below
    the 32% a unit-variance residual would give, because the endpoint of a
    two-parameter fit has high leverage and the fit absorbs part of its own
    residual). That is far too often for a gate that refuses to spend a
    budget, and the ranking it produced was wrong outright: a 4096..131072
    ladder scored 1.87 against 1024..32768's 1.18, calling the flatter of the
    two the healthier.

    Deliberately MODEL-FREE: it never consults the fitted a1 or omega1, which
    is the fit this check exists to second-guess. Plugging them into
    a1*i**-omega1 reports a huge correction exactly when the fit has gone
    degenerate -- the opposite of the truth, and this function's first version.

    `p` small means real curvature: the correction is visible and omega1 is
    worth fitting. `p` large means the ladder and the sample count together
    cannot see it. More samples shrink sigma_log and so do move p -- which is
    why the loop responds by doubling rather than giving up at once -- but a
    ladder starting above where the correction lives needs absurdly many:
    measured on srw, the correction is 3.65e-2 at i = 8 and 9.2e-4 at i = 512,
    a factor of 40, while a sample at i = 512 costs 64x one at i = 8.
    """
    from scipy.stats import chi2 as _chi2

    i = np.asarray(scales, float)
    y = np.asarray([r["y_bar"] for r in reps], float)
    sig = np.asarray([r["sigma_log"] for r in reps], float)
    y_pool = y.mean(axis=0)
    sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))

    log_i, log_y = np.log(i), np.log(y_pool)
    gamma = float(gamma_all_points(i, y_pool))            # pure power law, eq. (523)
    w = 1.0 / sig_pool ** 2
    intercept = float(np.sum(w * (log_y - gamma * log_i)) / np.sum(w))
    resid = log_y - (intercept + gamma * log_i)

    dof = max(1, len(i) - 2)
    chi2 = float(np.sum((resid / sig_pool) ** 2))
    k = int(np.argmin(i))
    return {"chi2": chi2, "dof": dof, "chi2_per_dof": chi2 / dof,
            "p": float(_chi2.sf(chi2, dof)),
            "scale": float(i[k]), "residual": float(abs(resid[k])),
            "noise": float(sig_pool[k]),
            "snr": float(abs(resid[k]) / sig_pool[k]) if sig_pool[k] > 0 else float("inf"),
            "gamma_power_law": gamma}


def bfs_span(consts: dict, m0: int, m: int, rho: float) -> dict | None:
    """How far B_fs moves across omega1 +/- 1 se -- the gate.

    None when omega1 has no standard error at all (one replicate), which is
    itself a failure: an undetermined input cannot be shown to be determined.
    """
    w, a1 = consts.get("omega1"), consts.get("a1")
    if w is None or a1 is None or w.se is None:
        return None
    lo_w, hi_w = max(1e-3, w.value - w.se), w.value + w.se
    lo, hi = sorted(finite_size_bias(m, m0, rho, a1.value, x) for x in (lo_w, hi_w))
    return {"lo": lo, "hi": hi, "span": (hi / lo) if lo > 0 else float("inf"),
            "at_m0": m0, "centre": finite_size_bias(m, m0, rho, a1.value, w.value)}


def _provisional_m0(consts, seconds_left, *, rho, m, replicates, throughput,
                    cost_ratio=1.0, target_se=None):
    """The m0 the plan would choose right now, for evaluating the gate at.

    The gate has to be checked at the m0 the answer will be computed at, and
    that depends on the budget still unspent -- which shrinks as the pilot
    loop runs. So it is recomputed each round rather than fixed up front.

    Asks for the m0 that FITS the remaining seconds, rather than converting
    seconds to a budget by multiplying by the throughput: that multiplication
    is only valid when the allocation's cost unit is the unit the throughput
    was measured in (tools/cost_model.cost_unit_ratio).

    Under --target-se there is no remaining budget to fit into: the budget is
    whatever the precision costs. The m0 the answer will be computed at is then
    the one `budget_for_target` picks, so the gate is evaluated there -- the
    same question asked of the other constraint.
    """
    kw = dict(d=consts["d"].value, omega1=consts["omega1"].value, rho=rho, m=m,
              a1=consts["a1"].value, cv=consts["cv"].value,
              throughput=throughput, cost_ratio=cost_ratio)
    try:
        pl = (plan_mod.budget_for_target(target_se, replicates=replicates, **kw)
              if target_se else
              plan_mod.budget_for_seconds(seconds_left / max(1, replicates), **kw))
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return pl["m0"] if pl.get("feasible") else None


def scale_draws(recipe: dict, factor: int) -> dict:
    """A copy of `recipe` asking for `factor` times as many draws per replicate.

    The three spellings of a recipe's `"n"` all have to grow the same way, and
    only the one the recipe actually uses is touched:

        {"rule": ..., "budget": B}  ->  budget * factor   (the allocation then
                                        redistributes it across scales by the
                                        same rule, so the SHAPE is preserved)
        [n_1, ..., n_k]             ->  each entry * factor
        n                           ->  n * factor

    The scales are never touched. That is the ladder, and the ladder is the
    recipe author's decision (see this module's docstring).
    """
    out = dict(recipe)
    n = recipe["n"]
    if isinstance(n, dict):
        if "budget" not in n:
            raise SystemExit(
                f"the pilot cannot grow this recipe: its \"n\" is the "
                f"{n.get('rule')!r} rule but states no \"budget\", so there is "
                f"nothing to double. Add one, or state \"n\" as a number.")
        out["n"] = {**n, "budget": float(n["budget"]) * factor}
    elif isinstance(n, (list, tuple)):
        out["n"] = [int(x) * factor for x in n]
    else:
        out["n"] = int(n) * factor
    return out


def pilot_until_determined(recipe, sd, *, seconds_budget, total_seconds,
                           replicates, seed, rho, m, throughput_guess,
                           max_rounds=MAX_ROUNDS, target_se=None,
                           reuse_cache=False,
                           span_limit=report_mod._BFS_SPAN_LIMIT, log=print):
    """The doubling loop: draw, fit, judge; double the DRAWS and repeat if not.

    Each round redraws `replicates` replicates with twice the previous round's
    n, from its own independent stream. The replicate count never changes --
    see this module's docstring for why that axis and not the other, and why
    each round is a clean redraw rather than an extension.

    TWO gates, and either alone is not enough. The B_fs span asks whether
    omega1 is pinned down; `ladder_check` asks whether there is anything on
    this ladder to pin down. A pilot can pass the span while failing the
    ladder: measured on srw over 1024..32768, the fit ran to
    omega1 = 5.52 +/- 0.14 with a1 = 3.4e14, and the span waves that through
    because a1 CANCELS in the ratio B_fs(omega1-se)/B_fs(omega1+se) -- it
    enters B_fs linearly. The goodness-of-fit test reads the same pilot as
    "a pure power law is fine" and refuses it.

    Both gates respond to more samples, so failing either is a reason to
    double rather than to stop. What separates the two failures is the
    diagnosis on the way out, which `_give_up` reads off the last round.

    Returns (consts, rounds, ok).
    """
    rounds = []
    consts, cost = None, None
    t_start = time.perf_counter()
    # One stream per round, drawn up front: a round is a fresh, independent
    # draw, never a continuation of the one it replaces (ground rule 2).
    round_seeds = spawn(seed, max_rounds)

    log(f"  pilot budget: {human_time(seconds_budget)} over at most "
        f"{max_rounds} doubling round(s)")
    for k in range(max_rounds):
        spent_before = time.perf_counter() - t_start
        if rounds and spent_before >= seconds_budget:
            log(f"  pilot: {human_time(spent_before)} spent of "
                f"{human_time(seconds_budget)} -- stopping after "
                f"{len(rounds)} round(s)")
            break
        factor = 2 ** k
        this = scale_draws(recipe, factor) if factor > 1 else recipe
        log(f"  pilot round {k + 1}: {replicates} replicate(s) at "
            f"{factor}x the recipe's draws")
        # The budget is checked BETWEEN rounds and never inside one, and round
        # 1 is not checked at all (`rounds` is empty). So a recipe whose own n
        # is large runs to completion no matter what --time said, and the
        # overrun can be arbitrary: measured on srw's samples_pilot.json
        # (budget 5e10) under --time 5m, round 1 alone ran past 609 s -- 8x
        # the pilot's 75 s slice and 2x the whole study's budget -- and was
        # still drawing when it was killed. Whether that SHOULD be bounded is
        # a modelling decision (a half-drawn round has no usable constants);
        # what is not defensible is it happening silently, so it is said here.
        if k == 0:
            log(f"    (round 1 always runs to completion -- the recipe's own "
                f"n decides how long, not the budget)")
        t = time.perf_counter()
        # The cost probe is measured once and handed to every later round.
        # d belongs to the model and the machine, not to the draws, and the
        # probe is deliberately fixed-seeded -- so re-probing each round pays
        # again for a bit-identical answer. Measured at 280 s per probe on the
        # run that prompted this, a 3-round pilot spent 14 minutes on it.
        out = pilot_mod.pilot(this, sd, replicates, seed=round_seeds[k],
                              cost=cost, reuse_cache=reuse_cache)
        cost = out["cost"]
        reps, consts = out["reps"], out["constants"]
        spent = time.perf_counter() - t_start

        lc = ladder_check(reps, [int(x) for x in recipe["scales"]])
        tp = out.get("throughput") or throughput_guess
        # The gate must be judged at the m0 the ANSWER will be computed at, so
        # this is the whole remaining budget -- not the pilot's slice of it.
        # From the RECIPE in hand, not from `out`: pilot() does not return the
        # recipe, so reading the model off `out` silently fell back to a ratio
        # of 1 -- the very assumption this is here to stop making.
        ratio_fn, _ = plan_mod._cost_ratio(
            {"recipe": this, "scales": this["scales"]},
            consts["d"].value if consts.get("d") else 1.0)
        m0 = _provisional_m0(consts, max(1e-9, total_seconds - spent),
                             rho=rho, m=m, replicates=replicates, throughput=tp,
                             cost_ratio=ratio_fn, target_se=target_se)
        span = bfs_span(consts, m0, m, rho) if m0 is not None else None
        rounds.append({"round": k + 1, "factor": factor,
                       "replicates": len(reps),
                       "seconds": time.perf_counter() - t,
                       "m0": m0, "span": span, "ladder": lc,
                       "omega1": consts["omega1"].value,
                       "omega1_se": consts["omega1"].se})

        took = time.perf_counter() - t
        cum = time.perf_counter() - t_start
        log(f"    round {k + 1} took {human_time(took)}; "
            f"{human_time(cum)} of the pilot's {human_time(seconds_budget)} used"
            + ("  <-- OVER BUDGET" if cum > seconds_budget else ""))

        flat = lc["p"] > LADDER_P_MAX
        if flat:
            log(f"    a pure power law still fits (chi2/dof = "
                f"{lc['chi2_per_dof']:.2f}, p = {lc['p']:.3g}): no correction "
                f"visible yet for omega1 to be fitted to")
        if span is None:
            log(f"    omega1 = {consts['omega1'].value:+.4f} (no se yet) "
                f"-- cannot judge, doubling the draws")
        else:
            ok = span["span"] <= span_limit
            log(f"    omega1 = {consts['omega1'].value:+.4f} "
                f"+/- {consts['omega1'].se:.4f}   B_fs spans {span['span']:.3g}x "
                f"at m0={m0}  -- {'OK' if ok else 'not yet'}")
            if ok and not flat:
                return consts, rounds, True
    return consts, rounds, False


def _bar(total_steps: int, enabled: bool):
    """A tqdm over predicted STEPS, or a no-op stand-in.

    Steps, not replicates or scales: the plan already predicts the total, `d`
    is measured, so the bar is proportional to real work and its ETA is a live
    check on the prediction rather than a decoration. A run that comes in at
    0.82x predicted shows that at thirty seconds instead of twelve minutes.

    The mechanics (TTY detection, the missing-tqdm fallback) moved to
    tools/progress.py in 2026-09 so the pilot and the probe could draw bars on
    the same terms; what stays here is the choice of UNIT, which is this
    phase's own.
    """
    return P.bar(total_steps, "drawing", unit="step", enabled=enabled)


def resume_command(sd: Path, *, replicates, target_se, seed, reuse_from) -> str:
    """The invocation that draws what --target-se just priced.

    Built here rather than in `_plan_run_report` because only the caller knows
    HOW the constants were obtained. A resume line that quietly dropped
    --reuse-pilot would send the user off to draw a fresh pilot for constants
    they already have, which is the opposite of the flag's purpose; and one
    that dropped --seed would walk into the collision guard.
    """
    parts = ["python3 src/study/autopilot.py",
             "--study", sd.name, "--data-root", str(sd.parent)]
    if reuse_from is not None:
        parts += ["--reuse-pilot", str(Path(reuse_from).name)]
    parts += ["--target-se", f"{target_se:g}", "--replicates", str(replicates)]
    if seed is not None:
        parts += ["--seed", str(seed)]
    return " ".join(parts)


def _reused_round(source: dict, consts, scales, *, rho, m, replicates,
                  seconds, target_se, ratio_fn, throughput):
    """The gate, evaluated on a pilot that was measured somewhere else.

    Reusing a pilot must not mean skipping the check on it. Both gates read
    things the source study already wrote down -- `ladder_check` the
    per-replicate summaries, the B_fs span the constants -- so they are asked
    here exactly as they are asked of a fresh round. What DOES change between
    studies is m0, because it comes from the new budget, and the span is a
    function of m0. So a pilot that passed where it was measured can fail
    here, and that is correct: the question is whether omega1 is pinned down
    well enough for the plan THIS study is about to make.
    """
    reps = source.get("per_replicate") or []
    lc = ladder_check(reps, scales)
    m0 = _provisional_m0(consts, max(1e-9, seconds), rho=rho, m=m,
                         replicates=replicates, throughput=throughput,
                         cost_ratio=ratio_fn, target_se=target_se)
    span = bfs_span(consts, m0, m, rho) if m0 is not None else None
    rnd = {"round": 0, "factor": None, "replicates": len(reps),
           "seconds": 0.0, "m0": m0, "span": span, "ladder": lc,
           "omega1": consts["omega1"].value, "omega1_se": consts["omega1"].se,
           "reused": True}
    ok = (span is not None and span["span"] <= report_mod._BFS_SPAN_LIMIT
          and lc["p"] <= LADDER_P_MAX)
    return consts, [rnd], ok


def _refuse_seed_collision(source_entropy, run_seed, source_dir) -> None:
    """A reused pilot must not come with the source study's run stream.

    autopilot spawns two children of one root: child 0 drives the pilot, child
    1 the run. Reuse the pilot and keep the root -- which is what happens by
    default, since the root falls back to the RECIPE's seed and the recipe
    comes from the study being reused -- and child 1 is the same stream, so the
    new study redraws the old one's replicates bit for bit. Two studies would
    then report the same gamma from "independent" runs, which is ground rule
    2's failure in its purest form: the one thing worse than a wasted run is a
    duplicated one presented as a second measurement.

    Cheap to detect, because every run records its seeds: compare the root
    entropy against the one in the source study's final.json.
    """
    if source_entropy is None:
        return
    rec = seed_record(run_seed)
    if int(rec.get("entropy", -1)) != int(source_entropy):
        return
    raise SystemExit(
        f"\nrefusing to reuse {source_dir} with the same root seed "
        f"({source_entropy}).\n"
        f"  The run stream is child 1 of that root in both studies, so this "
        f"run would redraw\n  the source study's replicates bit for bit and "
        f"report them as an independent\n  measurement (PLAN.md ground rule "
        f"2).\n"
        f"  Pass a different --seed, e.g. --seed {int(source_entropy) + 1}.")


def autopilot(recipe: dict, sd: Path, *, seconds: float | None = None,
              replicates: int, seed=None, rho=2.0, m=6, pilot_cap=PILOT_CAP,
              max_rounds=MAX_ROUNDS, force=False, progress=True,
              target_se: float | None = None, pilot_seconds_cap: float | None = None,
              reuse_from: Path | None = None, reuse_cache: bool = False,
              yes: bool = False, log=print) -> dict:
    """pilot -> plan -> run -> report, deciding in between. Returns the record.

    Two ways to say how big the study is, and exactly one of them is required.
    `seconds` (--time) is a budget and asks what precision it buys; `target_se`
    (--target-se) is a precision and asks what it costs. The second stops after
    planning unless `yes`, because "how long would this take" is a question,
    and answering it by starting the run would be a poor answer.

    `reuse_from` skips the pilot entirely and carries another study's constants
    in (see pilot.reuse). The gates are still evaluated, at THIS study's m0.
    """
    t0 = time.perf_counter()
    if (seconds is None) == (target_se is None):
        raise SystemExit("autopilot needs exactly one of --time or --target-se")

    # Read and validate the reused pilot, but write nothing yet: the seed
    # collision below is a refusal too, and a study directory holding the
    # constants of a run that was then refused is a trap for the next reader.
    reused = (pilot_mod.read_for_reuse(reuse_from, recipe=recipe or None)
              if reuse_from else None)
    if reused and not recipe:
        recipe = reused["recipe"]

    # Independent streams for the two draws. The pilot's samples must not
    # reappear in the run it sized -- that would be the same duplication
    # tools/rng.py's spawn(skip=) exists to prevent, one level up.
    pilot_seed, run_seed = spawn(seed if seed is not None else recipe.get("seed"), 2)
    if reused:
        _refuse_seed_collision(reused["entropy"], run_seed, reused["source_dir"])
        pilot_mod.write_reused(sd, reused, log=log)

    scales = [int(x) for x in recipe["scales"]]
    log(f"study   = {sd}")
    log(f"model   = {recipe['model']}  scales = {scales}")
    if target_se:
        log(f"target  = se(gamma) <= {target_se:g} on the mean of {replicates} "
            f"replicate(s); the budget is whatever that costs")
    else:
        log(f"budget  = {human_time(seconds)} total, at most "
            f"{human_time(seconds * pilot_cap)} of it on the pilot")
    log("")

    # The pilot's slice. Under --time it is a share of the total; under
    # --target-se there is no total to take a share of, so it is stated
    # outright (--pilot-time).
    pilot_budget = (pilot_seconds_cap if seconds is None
                    else seconds * pilot_cap)
    if reused:
        d_const = reused["constants"].get("d")
        ratio_fn, _ = plan_mod._cost_ratio(
            {"recipe": recipe, "scales": scales},
            d_const.value if d_const else 1.0)
        # The source study's own clock. Reusing a pilot means reusing the
        # throughput it measured, so the wall-clock predictions here are that
        # machine's -- which is this one, or the reuse should not have been
        # asked for.
        tp = reused["pilot"].get("throughput") or 1e8
        consts, rounds, ok = _reused_round(
            reused["pilot"], reused["constants"], scales, rho=rho, m=m,
            replicates=replicates, seconds=(seconds or 0.0), target_se=target_se,
            ratio_fn=ratio_fn, throughput=tp)
    else:
        consts, rounds, ok = pilot_until_determined(
            recipe, sd, seconds_budget=pilot_budget,
            total_seconds=seconds if seconds is not None else pilot_budget,
            replicates=replicates, seed=pilot_seed, rho=rho, m=m,
            throughput_guess=1e8, max_rounds=max_rounds, target_se=target_se,
            reuse_cache=reuse_cache, log=log)
    pilot_seconds = time.perf_counter() - t0

    if reused:
        log(f"\nconstants reused, {human_time(pilot_seconds)} spent (nothing drawn)")
    else:
        log(f"\nconstants after {rounds[-1]['replicates']} replicate(s) at "
            f"{rounds[-1]['factor']}x the recipe's draws, "
            f"{human_time(pilot_seconds)}")
    log(format_table(consts))

    # The ladder diagnostic, now that omega1/a1 are measured rather than
    # guessed. A warning, never a refusal: the scales are the recipe's.
    lc = rounds[-1]["ladder"]
    if ok and lc["p"] > LADDER_P_MAX:
        log(f"\n  !! a pure power law fits this pilot (chi2/dof = "
            f"{lc['chi2_per_dof']:.2f} on {lc['dof']} dof, p = {lc['p']:.3g}),\n"
            f"     so omega1 = {consts['omega1'].value:+.4g} is fitted to a "
            f"curvature that is not resolved.\n"
            f"     Extending the recipe's scales downward is the cheap fix: "
            f"doubling n buys precision as\n"
            f"     sqrt(n), a smaller rung buys it as a power of the scale.")

    if not ok:
        # The diagnosis is printed either way -- it is the evidence, and it is
        # the same evidence whether or not the run goes ahead. What --force
        # changes is only what happens next.
        _diagnose(consts, rounds, sd, pilot_seconds, seconds or 0.0, lc, force, log)
        if not force:
            return _record_give_up(consts, rounds, sd, pilot_seconds, lc)

    left = None if seconds is None else seconds - (time.perf_counter() - t0)
    if target_se:
        log(f"\nplanning for se(gamma) <= {target_se:g}"
            + (f" ({human_time(pilot_seconds)} went to the pilot)"
               if not reused else ""))
    else:
        log(f"\nplanning with {human_time(left)} of the budget left "
            f"({human_time(pilot_seconds)} went to the pilot)")
    rec = _plan_run_report(recipe, sd, consts, rounds, lc, seconds=left,
                           replicates=replicates, seed=run_seed, rho=rho, m=m,
                           progress=progress, pilot_seconds=pilot_seconds,
                           log=log, forced=not ok, target_se=target_se,
                           draw=yes or not target_se,
                           resume=resume_command(
                               sd, replicates=replicates, target_se=target_se,
                               seed=seed, reuse_from=reuse_from)
                           if target_se else "")
    if not ok:
        log("\n  !! --force: the constants above were NOT determined, and the "
            "answer\n     printed here inherits that. The eq. (720) bound's "
            "B_fs term is built\n     from an omega1 this pilot could not pin "
            "down; treat the interval as\n     indicative, and see "
            "`forced: true` in autopilot.json.")
    return rec


def _diagnose(consts, rounds, sd, pilot_seconds, seconds, lc, force, log) -> None:
    """Say what was measured, what failed, and why. Decides nothing.

    The failure mode this exists for is not a crash. It is a study that spends
    its whole budget and reports gamma = 0.50195 +/- 0.00076, excluding the
    truth, because omega1 was never determined. So the constants and their
    intervals are printed in full: they are the evidence for the diagnosis,
    and they are also what the next attempt starts from.

    Printed whether or not --force was given, because the evidence does not
    depend on what the caller decided to do about it. Only the last line
    differs, and the caller acts on `force`, not this function.
    """
    last = rounds[-1]
    sp = last["span"]
    head = ("PILOT DID NOT DETERMINE THE CONSTANTS -- running anyway (--force)"
            if force else
            "PILOT DID NOT DETERMINE THE CONSTANTS -- nothing was drawn")
    log(f"\n{'=' * 68}\n{head}\n")
    log("measured so far:")
    log(format_table(consts))
    if sp is not None:
        log("\nwhy it is not enough: eq. (720)'s finite-size term is "
            "B_fs ~ rho**(-omega1*m0), so omega1 sits in an exponent.")
        log(f"  at m0 = {sp['at_m0']}, moving omega1 by its own +/- 1 se "
            f"({consts['omega1'].se:.4g}) moves B_fs across")
        log(f"      [{sp['lo']:.3g}, {sp['hi']:.3g}]  -- a factor of "
            f"{sp['span']:.3g}, against a limit of "
            f"{report_mod._BFS_SPAN_LIMIT:g}")
        log("  a bound whose bias term is that undetermined is a statement "
            "about the pilot, not about gamma.")
    else:
        log("\nwhy it is not enough: omega1 has no standard error at all "
            "(one replicate has no spread),\n  so it cannot be shown to be "
            "determined.")
    if last.get("reused"):
        # A reused pilot has no rounds and no budget of its own. What failed is
        # the gate at THIS study's m0, and the fix is a better pilot, not more
        # of this one -- so say which study it came from.
        log(f"\nthese constants were reused, not drawn here: nothing in this "
            f"study can improve them.\n  Deepen the pilot where it was "
            f"measured ({last.get('replicates', '?')} replicate(s)), or drop "
            f"--reuse-pilot and draw a fresh one.")
    else:
        log(f"\nthe pilot used {human_time(pilot_seconds)} of "
            f"{human_time(seconds)} ({len(rounds)} doubling round(s), ending at "
            f"{rounds[-1]['replicates']} replicate(s) x "
            f"{rounds[-1]['factor']}x the recipe's draws).")
    if lc["p"] > LADDER_P_MAX:
        log(f"\nmost likely cause: THE LADDER. A pure power law still fits "
            f"(chi2/dof = {lc['chi2_per_dof']:.2f} on {lc['dof']} dof, "
            f"p = {lc['p']:.3g}),\n  so there is no resolved curvature for "
            f"omega1 to be fitted to. At the smallest scale i="
            f"{lc['scale']:.0f}\n  the departure is {lc['residual']:.3g} "
            f"against noise {lc['noise']:.3g}.")
        log("  Fix: extend the recipe's scales DOWNWARD. More samples buy "
            "precision as sqrt(n);\n  a smaller rung buys it as a power of "
            "the scale.")
    else:
        log(f"\nthe ladder is not the limit -- the curvature IS resolved "
            f"(chi2/dof = {lc['chi2_per_dof']:.2f}, p = {lc['p']:.3g}),\n"
            f"  omega1 is simply not pinned down yet. Fix: raise --time, or "
            f"--pilot-cap above {PILOT_CAP:g}.")
    log("\n  --force runs anyway, on constants known to be undetermined."
        if not force else
        "\n  --force was given: planning and drawing proceed on these "
        "constants.")
    log("=" * 68)


def _record_give_up(consts, rounds, sd, pilot_seconds, lc) -> dict:
    """The record written when the gate failed and nothing was drawn."""
    rec = {"ok": False, "forced": False,
           "reason": "pilot did not determine the constants",
           "rounds": rounds, "ladder": lc,
           "constants": {k: vars(v) for k, v in consts.items()},
           "pilot_seconds": pilot_seconds}
    write_artifact(sd, "autopilot", rec, produced_by="src/study/autopilot.py")
    return rec


def _plan_run_report(recipe, sd, consts, rounds, lc, *, seconds, replicates,
                     seed, rho, m, progress, pilot_seconds, log,
                     forced=False, target_se=None, draw=True,
                     resume="") -> dict:
    """Steps 2-4, with the plan accepted automatically and a bar over the run.

    `forced` means the gate failed and --force overrode it. It changes nothing
    about what is computed -- the same plan, the same draw, the same report --
    and is carried into autopilot.json so a result produced that way can never
    be mistaken later for one whose constants were determined.

    `draw=False` is --target-se without --yes: plan and stop. plan.py is then
    invoked WITHOUT --accept, so no plan.json is written -- the file means "a
    decision was taken", and the user has not taken it yet. What they get is
    the proposal, the wall clock it needs, and the command that runs it.
    """
    constraint = (["--target-se", repr(float(target_se))] if target_se
                  else ["--time", f"{seconds:.0f}s"])
    argv = (["--study", sd.name, "--data-root", str(sd.parent)] + constraint +
            ["--replicates", str(replicates), "--rho", str(rho), "--m", str(m)]
            + (["--accept"] if draw else []))
    # plan.py prints its own tables, but the bisection that precedes them
    # (plan.budget_for_seconds: up to 200 allocations) prints nothing, so on a
    # wide ladder the step opens with a silent pause.
    P.say("")
    with P.phase("planning: searching for the allocation that fits the budget"):
        plan_mod._main(argv)
    if not draw:
        log(f"\n  --target-se answers a question and stops. To draw it:"
            f"\n      {resume} --yes")
        return {"ok": True, "drawn": False, "forced": bool(forced),
                "rounds": rounds, "ladder": lc, "pilot_seconds": pilot_seconds,
                "target_se": float(target_se),
                "constants": {k: vars(v) for k, v in consts.items()}}
    plan = read_artifact(sd, "plan")
    rp = plan.get("recipe_path")
    final_recipe = load_recipe(Path(rp), "samples") if rp else recipe

    # The bar counts the model's own cost unit, cost(i) = n_i * i**d, with the
    # d the pilot MEASURED -- so on a model whose cost is not linear in i the
    # bar is still proportional to work, and its ETA stays a live check on the
    # plan's prediction rather than a second guess at it.
    d = consts["d"].value
    total_steps = plan["total_cost"]
    log(f"\ndrawing {plan['replicates']} replicate(s) "
        f"x {plan['n']:,} per scale, predicted {human_time(plan['total_seconds'])}")
    t = time.perf_counter()
    scales_plan = list(plan["scales"])
    n_scales = len(scales_plan)
    R_plan = int(plan.get("replicates", 1))
    spec_hint = get_model(final_recipe["model"]).cost_hint
    with _bar(int(total_steps), progress) as bar:
        # The bar advances once per SCALE, and a scale is a single atomic
        # `simulate` call -- so between two ticks it does not move at all, and
        # the LAST rung is most of the work. Measured on a 248 s srw study:
        # i=8192 alone was 51% of the total and i=4096 a further 25%, so a
        # completion-fed bar sat at 24% through three quarters of the run.
        # Naming the rung on the way IN is what tells the two apart --
        # "drawing the big one" against "it has stopped".
        rep = [0]
        nar = P.ScaleNarrator(scales_plan, [plan["n"]] * n_scales,
                              spec_hint, final_recipe.get("params", {}),
                              predicted_seconds=plan["total_seconds"] / R_plan)

        def begin(i, n_i, idx, _n):
            if idx == 1:
                rep[0] += 1
            bar.set_postfix_str(f"rep {rep[0]}/{R_plan}, scale {idx}/{n_scales} "
                                f"i={int(i)}")
            nar.start(i, n_i, idx)

        def tick(i, n_i, secs):
            bar.update(int(n_i * float(i) ** d))
            nar.done(i, n_i, secs)

        final = run_mod.execute(plan, final_recipe, sd, seed=seed,
                                on_scale=tick, on_scale_start=begin,
                                quiet=bool(progress))
    final["plan"] = plan
    write_artifact(sd, "final", final, produced_by="src/study/autopilot.py")
    log(f"drew {plan['replicates']} replicate(s) in "
        f"{time.perf_counter() - t:.1f} s "
        f"(predicted {plan['total_seconds']:.1f} s, "
        f"ratio {(time.perf_counter() - t) / plan['total_seconds']:.2f}x)")

    # Four steps that print nothing between them, the last of which imports
    # matplotlib on first use -- seconds of silence arriving directly after
    # the run's bar has filled and closed, which is the worst possible moment
    # for the console to look finished when it is not.
    with P.phase("analysing the run (eq. (526) refit + the eq. (720) bound)"):
        res = report_mod.analyse(final)
    with P.phase("writing report.md and details.md"):
        report_mod.write_report(sd, res, final, consts, plan)
        report_mod.write_details(sd, res, final, consts, plan)
    with P.phase("rendering plot.png (first matplotlib import is the slow part)"):
        fig = report_mod._plot(sd, res, final)
    write_artifact(sd, "answer", res, produced_by="src/study/autopilot.py")

    log("")
    report_mod.print_answer(res, log=log)
    log(f"\n  {sd / 'report.md'}\n  {sd / 'details.md'}\n  {fig}")

    rec = {"ok": True, "drawn": True, "forced": bool(forced),
           "rounds": rounds, "ladder": lc,
           "pilot_seconds": pilot_seconds, "plan": plan,
           "constants": {k: vars(v) for k, v in consts.items()},
           "gamma": res["gamma"], "wilson": res.get("wilson"),
           "ci": res["ci"], "seed": seed_record(seed)}
    if forced:
        rec["reason"] = ("pilot did not determine the constants; --force "
                         "ran the study anyway")
    write_artifact(sd, "autopilot", rec, produced_by="src/study/autopilot.py")
    return rec



def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-meta", "--meta", dest="meta", type=Path, default=None,
                   help="samples recipe: the model, its params, and the SCALES. "
                        "The ladder is never changed by this script. Optional "
                        "only with --reuse-pilot, which carries its own.")
    p.add_argument("--study", required=True, help="name for the study directory")
    p.add_argument("--data-root", type=Path, default=None,
                   help="where studies live; defaults beside the recipe")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--time",
                   help="TOTAL wall clock for everything -- pilot and run: 2h, 90m, 45s")
    g.add_argument("--target-se", type=float, dest="target_se",
                   help="the precision you need on the ANSWER (the mean of "
                        "--replicates replicates, bias included). The budget "
                        "becomes whatever that costs; plans and STOPS unless "
                        "--yes, because how long it would take is a question")
    p.add_argument("--yes", action="store_true",
                   help="with --target-se, draw the plan rather than only "
                        "printing what it would cost")
    p.add_argument("--pilot-time", default="5m", dest="pilot_time",
                   help="with --target-se, how long the pilot may take -- there "
                        "is no total for --pilot-cap to take a share of "
                        "(default 5m). Ignored under --time")
    p.add_argument("--reuse-pilot", default=None, dest="reuse_pilot",
                   help="name (or path) of a finished study whose constants to "
                        "carry in instead of drawing a pilot. Model, params and "
                        "SCALES must match. The whole budget then goes to the "
                        "run; needs a --seed distinct from that study's")
    p.add_argument("--reuse-cost", action="store_true", dest="reuse_cost",
                   help="reuse this machine's cached cost probe (d) instead of "
                        "re-timing the model. Unlike --reuse-pilot this survives "
                        "a change of ladder")
    p.add_argument("--replicates", type=int, default=3,
                   help="replicates in the first pilot round, and in the final run")
    p.add_argument("--seed", type=int, default=None,
                   help="one seed; pilot and run get independent streams of it")
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--pilot-cap", type=float, default=PILOT_CAP, dest="pilot_cap",
                   help=f"fraction of --time the pilot may spend (default {PILOT_CAP})")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, dest="max_rounds",
                   help="doubling rounds before giving up")
    p.add_argument("--force", action="store_true",
                   help="plan, draw and report even when the pilot's gate "
                        "failed. The diagnosis is printed either way; this "
                        "decides only whether the long run happens. The answer "
                        "is stamped `forced: true` in autopilot.json, because "
                        "it rests on constants known to be undetermined")
    p.add_argument("--no-progress", action="store_false", dest="progress",
                   help="no progress bar (also off automatically when not a TTY)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="per-rung probe timings and per-scale draw sizes as "
                        "well as the phase lines. The phase lines themselves "
                        "are never opt-in -- a silent run is the bug this "
                        "flag's default behaviour exists to avoid")
    a = p.parse_args(argv)
    P.set_verbose(a.verbose)

    if not 0 < a.pilot_cap < 1:
        raise SystemExit(f"--pilot-cap must be in (0, 1); got {a.pilot_cap}")
    if a.meta is None and not a.reuse_pilot:
        raise SystemExit("-meta is required (or --reuse-pilot, which carries "
                         "the recipe it was measured on)")
    if a.meta is None and a.data_root is None:
        raise SystemExit("--reuse-pilot without -meta needs --data-root to "
                         "locate both studies")
    recipe = load_recipe(a.meta, "samples") if a.meta else {}
    root = a.data_root or default_out_dir(a.meta)
    sd = pilot_mod.study_dir(root, a.study)

    reuse_from = None
    if a.reuse_pilot:
        reuse_from = Path(a.reuse_pilot)
        if not reuse_from.exists():
            reuse_from = pilot_mod.study_dir(root, a.reuse_pilot)

    rec = autopilot(recipe, sd,
                    seconds=plan_mod.parse_duration(a.time) if a.time else None,
                    target_se=a.target_se,
                    pilot_seconds_cap=plan_mod.parse_duration(a.pilot_time),
                    replicates=a.replicates, seed=a.seed, rho=a.rho, m=a.m,
                    pilot_cap=a.pilot_cap, max_rounds=a.max_rounds,
                    force=a.force, progress=a.progress, yes=a.yes,
                    reuse_from=reuse_from, reuse_cache=a.reuse_cost)
    raise SystemExit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    _main()
