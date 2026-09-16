"""Step 2 of a study: what would a longer run cost, and is the pilot good enough?

Reads the constants pilot.py measured and answers two questions the user
actually has:

  "I have 2 hours"           -> --time 2h        what precision can I get?
  "I need se(gamma) = 1e-3"  -> --target-se 1e-3 how long must I run?

It writes plan.json and STOPS. Nothing is drawn here. That is deliberate: the
plan is a proposal you look at before spending hours of compute, and the
decision about whether the pilot's constants are trustworthy enough to act on
is yours, not this script's.

What it will not do is pretend the plan is better determined than it is. The
error budget shows, per constant, how far its own standard error moves the
optimal m0 and what that costs in RMSE. omega1 is the one to watch: at a
2-second pilot its fit is genuinely unstable, and its se propagates into both
theta2 = 1/(d + 2*omega1) and the exponent rho^(-m0*omega1).

CLI:
    python3 src/study/plan.py --study mystudy --time 2h
    python3 src/study/plan.py --study mystudy --target-se 1e-3
    python3 src/study/plan.py --study mystudy --time 30m --accept   # write it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.allocation import ladder, predict_error, tuned_allocation  # noqa: E402
from tools.artifacts import (  # noqa: E402
    artifact_path,
    recipe_name,
    recipes_dir,
    write_artifact,
)
from tools.constants import format_table, load, require  # noqa: E402
from tools.cost_model import cost_unit_ratio  # noqa: E402
from tools.models import get_model  # noqa: E402

from src.budget.allocation_table import human_time, input_sensitivity  # noqa: E402

#: An se(omega1) above this moves the optimal m0 by roughly a full step. Not a
#: hard limit -- plan never refuses -- but the threshold at which the warning
#: stops being a formality. Derived from the measured flatness: a step of m0
#: costs ~1.1x in RMSE, and se(omega1) ~ 0.35 is what produces it at rho=2.
NOISY_OMEGA1_SE = 0.35

#: A constant whose standard error exceeds its own magnitude is not measured,
#: it is UNIDENTIFIED, and the distinction matters more than the size of the
#: error bar. Measured on a 2.3-second srw pilot, omega1-hat across 8 seeds was
#: [0.81, 13.0, 0.27, 2.07, 2.49, 0.05, 0.07, 0.98] against a truth of 1 -- the
#: median (0.89) is fine, so the estimator is not biased, but any single pilot
#: is a lottery. Reporting "1.66 +/- 1.10" as though it were a measurement
#: invites acting on it.
def unidentified(c) -> bool:
    """True when the error bar is as wide as the value it surrounds."""
    return c is not None and c.se is not None and c.se >= abs(c.value)

_TIME = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> float:
    """'2h', '90m', '1.5h', '3600' -> seconds."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([smhd]?)\s*", str(text).lower())
    if not m:
        raise ValueError(f"cannot read {text!r} as a duration; try 2h, 90m, 45s")
    return float(m.group(1)) * _TIME.get(m.group(2) or "s", 1)


def plan_for_budget(B: float, *, d, omega1, rho, m, a1, cv, throughput,
                    cost_ratio=1.0) -> dict:
    """The tuned allocation at budget B, with the error it should deliver.

    `cost_ratio` converts the allocation's cost unit (Assumption 7's i**d)
    into the unit `throughput` was measured in (the model's own cost_hint --
    sites, steps, seconds of burn). A float, or a callable(scales) for a model
    whose ratio depends on the ladder. It is 1.0 exactly for srw and
    percolation2d, and 256 for percolation_tau: see
    tools/cost_model.cost_unit_ratio for what leaving it out cost.
    """
    t = tuned_allocation(B, d, omega1, rho, m, a1=a1, cv=cv)
    if t.get("n") is None:
        return {"feasible": False, "budget": B,
                "why": "budget too small for even one sample per scale"}
    err = predict_error(t["n"], t["m0"], d=d, omega1=omega1, rho=rho, m=m, a1=a1, cv=cv)
    scales = ladder(t["m0"], m, rho)
    ratio = float(cost_ratio(scales) if callable(cost_ratio) else cost_ratio)
    return {"feasible": True, "budget": B, "n": t["n"], "m0": t["m0"],
            "scales": scales, "cost": t["cost"], "cost_ratio": ratio,
            "work": t["cost"] * ratio,
            "seconds": t["cost"] * ratio / throughput, **err}


def total_error(plan: dict, replicates: int) -> float:
    """The error on the ANSWER: R replicates averaged, bias and all.

    `predict_error` describes ONE replicate. Averaging R of them divides the
    scatter by sqrt(R) and does nothing whatever to the finite-size bias, which
    shifts every replicate the same way -- the fact report.py exists to state
    and the replicate interval is blind to. So the number a user who says
    "I need 1e-3" is asking for is

        sqrt(bias**2 + sd**2 / R),

    not the per-replicate rmse, and the two differ by more than sqrt(R)
    whenever the plan is bias-limited. Where bias alone already exceeds the
    target, no budget reaches it and `budget_for_target` says so rather than
    bisecting toward a wall.
    """
    R = max(1, int(replicates))
    return sqrt(plan["bias"] ** 2 + plan["sd"] ** 2 / R)


def budget_for_target(target_se: float, *, d, omega1, rho, m, a1, cv, throughput,
                      cost_ratio=1.0, replicates: int = 1,
                      lo=1e3, hi=1e18) -> dict:
    """Smallest budget whose predicted error meets `target_se`, by bisection.

    `target_se` is the error on the FINAL answer -- the mean of `replicates`
    replicates, `total_error` -- for the same reason `--time` is the total
    wall clock rather than a per-replicate one: asking for a precision and
    being handed a plan that delivers something else is the surprise this
    workflow exists to remove. At replicates=1 the two coincide, which is what
    every caller before 2026-09-14 meant.

    Bisection rather than algebra because the tuned allocation floors m0 and n
    to integers, so predicted RMSE is a staircase in B, not a smooth power law.
    """
    kw = dict(d=d, omega1=omega1, rho=rho, m=m, a1=a1, cv=cv, throughput=throughput,
              cost_ratio=cost_ratio)
    top = plan_for_budget(hi, **kw)
    if not top["feasible"] or total_error(top, replicates) > target_se:
        floor = (f"; the finite-size bias alone is {top['bias']:.3g} at the "
                 f"deepest affordable m0, and no number of replicates reduces it"
                 if top.get("feasible") and top["bias"] > target_se else "")
        return {"feasible": False,
                "why": f"target {target_se:g} unreachable below B={hi:g}{floor}"}
    for _ in range(200):
        mid = sqrt(lo * hi)
        p = plan_for_budget(mid, **kw)
        if p["feasible"] and total_error(p, replicates) <= target_se:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.001:
            break
    return plan_for_budget(hi, **kw)


def budget_for_seconds(target_seconds: float, *, d, omega1, rho, m, a1, cv,
                       throughput, cost_ratio=1.0, lo=1.0, hi=1e20) -> dict:
    """The largest plan whose PREDICTED wall clock fits `target_seconds`.

    Why a bisection rather than `B = seconds * throughput`: that identity holds
    only when the allocation's cost unit and the throughput's unit are the same
    thing, which is exactly the assumption that broke (see
    tools/cost_model.cost_unit_ratio -- it predicted 740 s for a 42-hour run).
    Bisecting on the predicted seconds, the quantity the caller actually
    constrained, is correct whatever the ratio is, including one that varies
    along the ladder.

    Below some budget no allocation is feasible at all (fewer than one sample
    per scale). Those Bs are treated as "fits", so the predicate stays
    monotone and the bisection converges on the UPPER edge of the feasible
    window; if the plan there is still infeasible, the time given cannot buy
    even one sample per scale and that is what comes back.
    """
    kw = dict(d=d, omega1=omega1, rho=rho, m=m, a1=a1, cv=cv,
              throughput=throughput, cost_ratio=cost_ratio)

    def fits(B):
        p = plan_for_budget(B, **kw)
        return (not p.get("feasible")) or p["seconds"] <= target_seconds

    if not fits(hi):
        for _ in range(200):
            mid = sqrt(lo * hi)
            if fits(mid):
                lo = mid
            else:
                hi = mid
            # Tight, so that a ratio-1 model (srw, percolation2d) lands on
            # exactly the allocation the old `B = seconds * throughput` gave:
            # the fix must be a no-op wherever the old formula was right.
            if hi / lo < 1 + 1e-12:
                break
    else:
        lo = hi
    out = plan_for_budget(lo, **kw)
    if not out.get("feasible"):
        return {"feasible": False, "budget": lo,
                "why": f"budget too small for even one sample per scale "
                       f"({target_seconds:g} s at {throughput:.3g} units/s)"}
    return out


def budget_ladder(B: float, *, replicates: int,
                  factors=(1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16),
                  **kw) -> list[dict]:
    """What the same constants buy at neighbouring budgets -- the plan in context.

    Indexed by BUDGET, not by m0. See src/budget/allocation_table.py's
    `build_budget_rows` for the argument: indexing by m0 asks "at what budget
    is this m0 optimal", so every row slides whenever a constant moves and two
    honest tables look unlike each other. Indexed by budget the question is
    "given this much time, what should I run", and the answer barely moves.

    Budgets are PER REPLICATE, the unit the plan itself is sized in, and go
    through the same `plan_for_budget` that produced the proposal -- so the
    marked row IS the proposal, rather than a second computation of it that
    could drift.
    """
    rows = []
    for f in factors:
        p = plan_for_budget(B * f, **kw)
        if not p.get("feasible"):
            continue
        p["chosen"] = f == 1
        p["total_seconds"] = p["seconds"] * replicates
        rows.append(p)
    return rows


def error_budget(consts, *, d, omega1, rho, m, a1, cv) -> list[dict]:
    """Per-constant: its se, how far that moves m0, and the RMSE cost."""
    out = []
    for name in ("omega1", "a1", "d", "cv"):
        k = consts.get(name)
        if k is None:
            continue
        if k.se is None:
            out.append({"name": name, "se": None})
            continue
        s = input_sensitivity(name, k.se, d=d, omega1=omega1, rho=rho, m=m, a1=a1, cv=cv)
        out.append({"name": name, "se": k.se, **s})
    return out


def final_recipe(source: dict, plan: dict, study: str) -> dict:
    """The accepted plan, written out as a samples recipe.

    This is the handoff. `plan.json` describes a decision -- m0, the RMSE it
    should buy, the constants it was made from -- and is not runnable; the
    recipe is the same allocation in the one format every driver in the repo
    already reads, so the planned run is not a special case:

        python3 src/generate/generate.py -meta <this file>     # draw it
        python3 src/study/run.py --study <name> ...            # ...or draw it
                                                               # with replicates
                                                               # and summaries

    Model and params come from `source`, the pilot's own recipe, so the final
    run is guaranteed to draw the same thing the constants were measured on.
    `seed` is null: a plan fixes the allocation, not the randomness, and
    ground rule 2 (PLAN.md) says every configuration draws fresh.
    """
    return {
        "kind": "samples",
        "model": source["model"],
        "params": source.get("params", {}),
        "scales": list(plan["scales"]),
        "n": int(plan["n"]),
        "seed": None,
        "replicates": int(plan.get("replicates", 1)),
        "_generated_by": "src/study/plan.py",
        "_study": study,
        "_note": [
            "GENERATED -- rewritten every time this study's plan is accepted.",
            "Edit the pilot recipe and re-plan rather than editing this file;",
            "run.py reads it back, so a hand edit here silently decouples the",
            "run from the plan.json it is reported against.",
        ],
    }


def write_final_recipe(data_root, study: str, source: dict, plan: dict) -> Path:
    """Put the final recipe beside the experiment's hand-authored ones."""
    d = recipes_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / recipe_name("samples", f"{study}_final")
    path.write_text(json.dumps(final_recipe(source, plan, study),
                               indent=2, sort_keys=True))
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True,
                   help="experiment data dir holding <study>/")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--time", help="wall clock you are willing to spend: 2h, 90m, 45s")
    g.add_argument("--target-se", type=float,
                   help="precision you need on gamma-hat; plan finds the budget")
    p.add_argument("--replicates", type=int, default=3,
                   help="independent replicates in the final run. The budget is split "
                        "between them, so --time is the TOTAL. Replicates are what give "
                        "gamma an empirical error bar; 3 is the minimum that gives a "
                        "spread at all, and t(R-1) widens fast below 5")
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--throughput", type=float, default=None,
                   help="steps/second; the pilot's measured one when absent (its "
                        "own clock, or for a batched_cost model its cost probe)")
    p.add_argument("--accept", action="store_true",
                   help="write plan.json so run.py will execute it")
    a = p.parse_args(argv)

    # report.py refits eq. (232) on the final ladder, and that fit has four
    # free parameters (tools/correction.py). Below five rungs it raises -- AFTER
    # the whole budget has been drawn, which is the one place this workflow must
    # not fail. Caught here, where the ladder width is chosen and nothing has
    # been spent. Found 2026-09-14 by running --m 3.
    if a.m < 5:
        raise SystemExit(
            f"--m {a.m} gives a {a.m}-scale ladder, and the eq. (232) refit "
            f"report.py runs on it\n  needs at least 5 (four free parameters). "
            f"The run would draw in full and then fail\n  at the report. Use "
            f"--m 5 or more; the default is 6.")

    sd = Path(a.data_root) / a.study
    consts = load(sd)
    if not consts:
        raise SystemExit(
            f"no constants.json in {sd}\n"
            f"  Run the pilot first:\n"
            f"    python3 src/study/pilot.py -meta <recipe> --study {a.study}")

    d = require(consts, "d").value
    omega1 = require(consts, "omega1").value
    a1 = require(consts, "a1").value
    cv = require(consts, "cv").value

    pj = artifact_path(sd, "pilot")
    pilot = json.loads(pj.read_text()) if pj.exists() else {}
    tp = a.throughput if a.throughput is not None else pilot.get("throughput")
    if tp is None:
        raise SystemExit(
            "no throughput measured. The pilot records one from its own draw;\n"
            "  re-run the pilot, or pass --throughput <steps/second>.")

    ratio_fn, ratio_note = _cost_ratio(pilot, d)

    print(f"study     = {sd}")
    print(f"constants ({require(consts, 'omega1').source})")
    print(format_table(consts))
    print(f"  {'rho':<11}{a.rho:>10.4f}              (design choice)")
    print(f"  {'m':<11}{a.m:>10d}              (design choice)")
    if ratio_note:
        print(ratio_note)
    print(f"  {'throughput':<11}{tp:>10.3g}              {_TP_UNIT}/s"
          + ("  (--throughput)" if a.throughput is not None else
             f"  (from {pilot['throughput_source']})"
             if pilot.get("throughput_source") else ""))

    print("\nerror budget -- how far each constant's own uncertainty moves the plan")
    print(f"  {'constant':<11}{'+/- 1 se':>12}{'moves m0 by':>14}{'worst-case RMSE':>18}")
    eb = error_budget(consts, d=d, omega1=omega1, rho=a.rho, m=a.m, a1=a1, cv=cv)
    for row in eb:
        c = consts[row["name"]]
        if row["se"] is None:
            # No spread is not one situation but two, and they need different
            # actions: a declared constant is EXACT, a single replicate has
            # simply not been measured enough times.
            why = ("exact -- declared, not fitted" if "declared" in c.source
                   else "no spread: only 1 replicate")
            print(f"  {row['name']:<11}{'--':>12}{why:>32}")
        else:
            flag = "   <-- UNIDENTIFIED" if unidentified(c) else ""
            print(f"  {row['name']:<11}{row['se']:>12.4f}"
                  f"{row['delta_m0']:>+14.2f}{row['penalty']:>17.4f}x{flag}")

    verdict, advice = _judge(consts, eb, a.study, a.data_root)
    print(f"\n{verdict}")
    if advice:
        print(advice)

    kw = dict(d=d, omega1=omega1, rho=a.rho, m=a.m, a1=a1, cv=cv, throughput=tp,
              cost_ratio=ratio_fn)
    R = max(1, a.replicates)
    # Resolved once, so the accept hint below quotes the duration actually
    # planned on. It used to interpolate `a.time` itself, which is None
    # whenever --time was omitted -- printing "--time None", a command that
    # dies in parse_duration.
    time_text = a.time or "60s"
    if a.target_se:
        pl = budget_for_target(a.target_se, replicates=R, **kw)
        head = (f"to reach se(gamma) <= {a.target_se:g} on the MEAN of {R} "
                f"replicate(s)" if R > 1 else
                f"to reach se(gamma) <= {a.target_se:g}")
    else:
        # The budget is split between replicates, so --time means the TOTAL
        # wall clock. Asking for 2h and being handed a 6h run would be the
        # kind of silent surprise this whole workflow exists to remove.
        seconds = parse_duration(time_text)
        # NOT `plan_for_budget(seconds * tp / R)`: that multiplication assumes
        # the allocation's cost unit IS the unit the throughput was measured
        # in. See budget_for_seconds and tools/cost_model.cost_unit_ratio.
        pl = budget_for_seconds(seconds / R, **kw)
        head = f"in {human_time(seconds)} total"
    if not pl.get("feasible"):
        raise SystemExit(f"\nno feasible plan {head}: {pl.get('why')}")

    pl["replicates"] = R
    pl["total_cost"] = pl["cost"] * R
    pl["total_seconds"] = pl["seconds"] * R

    print(f"\nproposed allocation, {head}")
    print(f"  m0     = {pl['m0']}          scales {pl['scales']}")
    print(f"  n      = {pl['n']:,} per scale, x{R} replicate(s)")
    print(f"  cost   = {pl['total_cost']:.4g} steps total  "
          f"({human_time(pl['total_seconds'])}; {human_time(pl['seconds'])} per replicate)")
    print(f"  gives  se(gamma) ~ {pl['rmse']:.4g} per replicate   "
          f"(|bias| {pl['bias']:.3g}, sd {pl['sd']:.3g})")
    if R > 1:
        print(f"         ~ {pl['sd'] / sqrt(R):.4g} on the mean of {R}, before the "
              f"t({R - 1}) widening")
        print(f"         ~ {total_error(pl, R):.4g} on the ANSWER -- that scatter "
              f"AND the bias, which does not average away")

    print(f"\nwhat other budgets buy, same constants "
          f"(budget is PER REPLICATE; total is x{R})")
    print(f"  {'per rep':>9} {'total':>9} {'m0':>4} {'n per scale':>14}"
          f"{'se(gamma)':>12}{'|bias|':>11}{'sd':>11}   scales")
    for r in budget_ladder(pl["budget"], replicates=R, **kw):
        print(f"  {human_time(r['seconds']):>9} {human_time(r['total_seconds']):>9} "
              f"{r['m0']:>4} {r['n']:>14,}{r['rmse']:>12.3e}{r['bias']:>11.3e}"
              f"{r['sd']:>11.3e}   {r['scales'][0]}..{r['scales'][-1]}"
              f"{'   <-- the plan above' if r['chosen'] else ''}")

    if a.accept:
        if "recipe" not in pilot:
            raise SystemExit(
                f"{pj} has no 'recipe', so the plan cannot say WHAT to draw.\n"
                f"  Re-run the pilot with -meta <recipe> to record it.")
        pl["constants_at_plan_time"] = {k: vars(v) for k, v in consts.items()}
        pl["rho"], pl["m"], pl["throughput"] = a.rho, a.m, tp
        rp = write_final_recipe(a.data_root, a.study, pilot["recipe"], pl)
        pl["recipe_path"] = str(rp)
        write_artifact(sd, "plan", pl, produced_by="src/study/plan.py")
        print(f"\naccepted -> {artifact_path(sd, 'plan')}")
        print(f"recipe   -> {rp}")
        print(f"next: python3 src/study/run.py --study {a.study} "
              f"--data-root {a.data_root}")
        print(f"  or, for a single un-replicated draw with the samples kept:")
        print(f"      python3 src/generate/generate.py -meta {rp}")
    else:
        print("\nnothing was drawn and no plan was written.")
        print(f"  To accept:  python3 src/study/plan.py --study {a.study} "
              f"--data-root {a.data_root} "
              + (f"--target-se {a.target_se:g}" if a.target_se else f"--time {time_text}")
              + " --accept")


#: What `throughput` counts. The pilot measures it with the model's own
#: cost_hint, so it is that model's unit -- sites for a lattice, steps for a
#: walk -- not the allocation's i**d.
_TP_UNIT = "cost_hint units"


def _cost_ratio(pilot: dict, d: float):
    """(callable(scales) -> ratio, note) for the pilot's model, else (1.0, "").

    The allocation charges i**d; `throughput` is measured in the model's own
    cost_hint units. `tools/cost_model.cost_unit_ratio` is the factor between
    them -- 1.0 for srw and percolation2d, 256 for percolation_tau. Computed
    per candidate ladder, so a `ceil` in a box rule is handled exactly.
    """
    recipe = pilot.get("recipe") or {}
    name, params = recipe.get("model"), recipe.get("params", {})
    if not name:
        return 1.0, ""
    try:
        spec = get_model(name)
    except ValueError:
        return 1.0, ""
    if spec.cost_hint is None:
        return 1.0, ""

    def ratio(scales):
        return cost_unit_ratio(scales, 1.0, d, spec.cost_hint, params)

    probe = ratio(pilot.get("scales") or [1])
    note = ("" if abs(probe - 1.0) < 1e-9 else
            f"  {'cost unit':<11}{probe:>10.4g}              "
            f"{name}'s cost_hint per allocation unit i**d")
    return ratio, note


def _judge(consts, eb, study, root) -> tuple[str, str]:
    """Is the pilot good enough to plan on? Says so; never decides for you.

    Three verdicts, because the actions differ. UNIDENTIFIED means the fit did
    not determine the constant and no amount of reading the number will help.
    NOISY means it is determined but loosely. Good means proceed.
    """
    more = (f"    python3 src/study/pilot.py --study {study} --more 5 "
            f"--data-root {root}")
    bad = [n for n in ("omega1", "a1") if unidentified(consts.get(n))]
    if bad:
        names = " and ".join(bad)
        return (f"VERDICT: DO NOT PLAN ON THIS PILOT -- {names} "
                f"{'are' if len(bad) > 1 else 'is'} unidentified.",
                f"  The standard error is as wide as the value itself, which means the "
                f"fit did\n  not determine it. On a 2.3-second srw pilot, omega1-hat "
                f"across 8 seeds was\n  0.05, 0.07, 0.27, 0.81, 0.98, 2.07, 2.49, 13.0 "
                f"against a truth of 1: the median\n  is right, so any ONE pilot is a "
                f"lottery. Either lengthen the pilot (raise the\n  recipe's budget) or "
                f"add replicates:\n{more}")

    om = consts.get("omega1")
    if om is None or om.se is None:
        return ("VERDICT: cannot tell -- the pilot has one replicate, so no fitted "
                "constant has\n  a standard error and the error budget above is empty.",
                f"  Add replicates before trusting the plan:\n{more}")

    worst = max((r for r in eb if r["se"] is not None),
                key=lambda r: r["penalty"], default=None)
    # The constants only ever reach the plan through m0, so the honest joint
    # test is how far they move m0 TOGETHER (user, 2026-09-06). Summed in
    # absolute value -- the pessimistic combination, since the constants are
    # fitted from one pilot and their errors are not independent -- a total
    # under one step means the plan lands on the same rung whatever the errors
    # do, which is the whole question. Individually flat inputs can still add
    # up to a step, which is why this is checked as well as `worst`.
    total_dm0 = sum(abs(r["delta_m0"]) for r in eb if r["se"] is not None)
    joint = (f"  Together the constants move m0 by {total_dm0:.2f} steps "
             f"(sum of |dm0|; " + ", ".join(
                 f"{r['name']} {r['delta_m0']:+.2f}"
                 for r in eb if r["se"] is not None) + ").")
    if om.se > NOISY_OMEGA1_SE or total_dm0 >= 1.0:
        why = ("omega1 is loosely determined "
               f"(se {om.se:.3f} > {NOISY_OMEGA1_SE})" if om.se > NOISY_OMEGA1_SE
               else f"the constants jointly move m0 by {total_dm0:.2f} steps")
        return (f"VERDICT: {why}; the plan is usable but soft.",
                f"{joint}\n  Worst single input is {worst['name']}: "
                f"{worst['delta_m0']:+.2f} steps, costing up to "
                f"{worst['penalty']:.3f}x in RMSE.\n  The optimum is quadratic in m0, "
                f"so this is survivable -- but tighten it if the\n  run is long:\n{more}")
    return (f"VERDICT: the pilot is good enough to plan on. Worst input is "
            f"{worst['name']}, costing\n  up to {worst['penalty']:.3f}x in RMSE -- the "
            f"optimum is quadratic in m0, so it is flat.", joint)


if __name__ == "__main__":
    _main()
