"""Step 1 of a study: a short run that measures the constants a plan needs.

The workflow this belongs to (see src/study/README.md):

    pilot.py   <- you are here: measure d, omega1, a1, cv from a cheap run
    plan.py       what a longer run would cost, and whether the pilot is good
                  enough to plan on
    run.py        execute the accepted plan
    report.py     gamma-hat with its error, the log-log plot, the details

A study is one directory that accumulates state, so no constant is ever
copied by hand between steps. Before this existed you had to read omega1 out
of omega1.json, d out of cost_probe.json, and type both into
allocation_table.py -- and if you forgot, it silently used 1.0 for each.

Replicates matter here and the default is 1 on purpose. One replicate gives
every constant a value and NO standard error, so `plan` cannot tell you how
well determined the plan is. omega1 is the one that hurts: measured across
single replicates of the same configuration it ranged 0.49 to 1.25. Use
--replicates 3+ when you intend to act on the answer; --more adds replicates
to an existing pilot without redrawing the ones you have.

CLI:
    python3 src/study/pilot.py -meta experiments/01_srw/recipes/samples_pilot.json \\
        --study mystudy
    python3 src/study/pilot.py --study mystudy --more 4     # tighten an existing pilot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, default_out_dir, load_recipe, write_artifact  # noqa: E402
from tools.constants import format_table, measured, override, save  # noqa: E402
from tools.correction import fit_correction  # noqa: E402
from tools.cost_model import (  # noqa: E402
    PROBE_MIN_SCALES, PROBE_REPEATS, aggregate, climb_to_target,
    estimate_cost_affine, fit_cost_probe)
from tools.models import get_model  # noqa: E402
from tools.rng import spawn  # noqa: E402
from tools.summary import replicate_summary, summarize_scale  # noqa: E402

from src.generate.generate import generate, resolve_n  # noqa: E402

#: Seed for the cost probe. Fixed, and deliberately not the pilot's own: the
#: probe measures how long simulate() TAKES, not what it returns, so it needs
#: no independence from the sample draws -- and a fixed seed keeps a re-run of
#: the pilot from moving d for reasons unrelated to the machine.
COST_PROBE_SEED = 0

#: How many INDEPENDENT cost probes the pilot runs. Not a precision knob -- a
#: calibration one. A single probe states an se for its own d_hat (the affine
#: fit's Gauss-Newton covariance, or a bootstrap over its repeat timings) that
#: measures only WITHIN-probe jitter, and the cost arm of
#: calibration/check_no_leakage.py measured both against the actual
#: probe-to-probe spread and found the covariance 3.6-18x too small and the
#: bootstrap wrong by up to 6x in either direction -- because what moves between
#: one climb and the next is cache and frequency state, the scheduler, whatever
#: else the machine is doing, and neither of them looks there. The consequence was a mismatch gate that fired on 32-34% of runs
#: whose declared cost was exact by construction. Repeating the probe measures
#: the spread that matters directly, and averaging over it shrinks the real
#: error of d by sqrt(n) as a bonus. See `_resolve_d` and D_MISMATCH_Z.
D_PROBES = 5

#: Seconds after which extra probes are abandoned (>= 2 always attempted, since
#: one probe has no spread). On srw a probe is ~0.2 s and all D_PROBES run; the
#: ceiling exists for a model whose climb runs into PROBE_TIME_BUDGET, where
#: five probes would be 100 s of a pilot's time spent on a diagnostic.
D_PROBE_TIME_BUDGET = 30.0


def study_dir(root: Path, name: str) -> Path:
    """Where a study lives: alongside every other run of that experiment."""
    return Path(root) / name


def _refuse_starved_scales(scales, n) -> None:
    """Stop before drawing anything if the allocation left a scale with n < 2.

    A single draw has no standard error, and sigma_log is what the log-log fit
    weights by -- so a starved scale does not merely add a weak point, it puts
    a NaN into the fit. Checked here rather than discovered in summarize_scale
    because by then the expensive scales have already been drawn.
    """
    counts = n if isinstance(n, (list, tuple)) else [n] * len(scales)
    starved = [int(i) for i, c in zip(scales, counts) if int(c) < 2]
    if not starved:
        return
    raise SystemExit(
        f"the allocation gives n < 2 at {len(starved)} scale(s): {starved}.\n"
        f"  A single draw has no spread, so those scales cannot be summarized and\n"
        f"  the fit they feed would be NaN. The ladder is too wide for the budget.\n"
        f"  Any of these fixes it:\n"
        f"    - raise the recipe's \"budget\"\n"
        f"    - drop the smallest scales (they cost almost nothing and get almost\n"
        f"      nothing under the snr rule, which spends where the correction is)\n"
        f'    - set "min_n": 2, accepting that it overspends the stated budget')


def _pilot_replicate(model, params, scales, n, seed_seq) -> dict:
    """One replicate, summarized: y_bar, sigma_log and cv per scale.

    `reduce=` collapses each scale's draws inside generate() and frees them
    immediately, so a pilot never holds a full sample in memory. The three
    summaries are tools/summary.py's -- the same ones run.py records, so a
    pilot replicate and a final replicate are interchangeable downstream.
    """
    stats = generate(model, scales, n, params, seed=seed_seq, reduce=summarize_scale)
    return replicate_summary(stats, scales)


def measure_cost_exponent(model, params, scales, repeats=PROBE_REPEATS,
                          probes: int = D_PROBES) -> dict:
    """Measure d = the cost exponent of Assumption cost_is_power_law (eq. 353).

    Both halves live in tools/cost_model.py and are shared with the standalone
    probe (src/estimate/measure_cost.py): `climb_to_target` picks where to
    time, `fit_cost_probe` fits cost(i) = a + b*i^d and reports the per-call
    overhead it separated out.

    The probe starts at the pilot's LARGEST scale and climbs away from there.
    It must not simply time the sample ladder: those scales are chosen so the
    correction term is visible, which means small, and at srw's 8..256 a single
    simulate() call is almost entirely Python/NumPy dispatch -- timing them
    returned d = 8.0 +/- 280. d is a property of the model, not of the window,
    so measuring it further out costs nothing.

    A model that declares `cost_hint` gets that reported too, as `declared_d`
    -- but only as something to CHECK the clock against, never as the value
    used. See `_resolve_d`: before 2026-08-29 a declaration was written
    straight into constants.json, so on srw (cost_hint(i) = i exactly) d = 1
    entered by definition and this probe never had to recover anything.

    The climb is run `probes` times from independent seeds, and the returned
    dict carries `d_across_probes` = {d, sd, se, n}: the MEAN of the per-probe
    affine exponents and the spread across them. That block is what `_resolve_d`
    prefers, and the reason is measured rather than assumed -- see D_PROBES. The
    last probe's own scales, timings and affine fit are returned alongside it so
    the record still shows one complete climb.
    """
    spec = get_model(model)
    start, fits, t0 = int(max(scales)), [], time.perf_counter()
    for j in range(max(1, probes)):
        fits.append(fit_cost_probe(
            climb_to_target(spec, params, np.random.default_rng(COST_PROBE_SEED + j),
                            start=start, repeats=repeats),
            spec.cost_hint, params))
        if len(fits) >= 2 and time.perf_counter() - t0 > D_PROBE_TIME_BUDGET:
            break
    out = fits[-1]
    ds = [float(f["affine"]["d"]) for f in fits
          if (f.get("affine") or {}).get("d") is not None]
    out["d_probes"] = ds
    out["probe_seeds"] = [COST_PROBE_SEED + j for j in range(len(fits))]
    if len(ds) >= 2:
        sd_ = float(np.std(ds, ddof=1))
        out["d_across_probes"] = {"d": float(np.mean(ds)), "sd": sd_,
                                  "se": sd_ / math.sqrt(len(ds)), "n": len(ds)}
    return out


#: |z| beyond which the clock and a declared cost are called a mismatch.
#: A WARNING, not a stop (user's call, 2026-08-29): a disagreement is evidence
#: about the machine or about the hint, and the draw that produced it is still
#: the draw you asked for -- stopping would throw away hours of sampling over a
#: diagnostic. It is recorded in pilot.json and reprinted by report.py instead,
#: so it cannot be lost by scrolling past.
#:
#: This is a NORMAL-theory cutoff: |z| > 3 is a 0.27% false-alarm rate only if
#: se(d) is KNOWN. It never is -- it is estimated from a handful of numbers, and
#: a ratio with a noisy denominator has heavier tails than the normal. So the
#: number actually compared against is `_mismatch_threshold(dof)`, the t
#: quantile at this same two-sided alpha; D_MISMATCH_Z sets the alpha, not the
#: cutoff. With D_PROBES = 5 the cutoff is 6.62, not 3.
D_MISMATCH_Z = 3.0

#: Above this share, the affine fit's `a` term is most of the cheapest
#: measurement, so the PURE power-law d_hat is measuring dispatch rather than
#: work. affine["d"] is unaffected -- separating that overhead is precisely what
#: it is for -- so this is recorded as a note, never a refusal.
OVERHEAD_SHARE_NOTE = 0.2

#: Resamples for the fallback se(d). Only used when the Gauss-Newton covariance
#: is singular, which needs len(scales) <= 3 or a degenerate Jacobian.
D_SE_BOOTSTRAP = 200


def _d_se_bootstrap(cost: dict, draws: int = D_SE_BOOTSTRAP,
                    seed: int = COST_PROBE_SEED) -> float | None:
    """se(d) by resampling the probe's own repeat times, when the fit gave none.

    `estimate_cost_affine` derives d_se from the Gauss-Newton covariance and
    returns None when that is singular (dof = len(scales) - 3 <= 0, or a
    degenerate Jacobian). Without SOME spread there is no z, and a declaration
    that cannot be checked is exactly what stage 2 exists to remove -- so fall
    back to the raw timings the probe already kept in `elapsed_all` rather than
    silently skipping the check.

    Returns None when even this is impossible (fewer than 2 repeats per scale,
    or too many refits failing), and the caller records `verdict: "unchecked"`.
    """
    raw, scales = cost.get("elapsed_all") or {}, cost.get("scales") or []
    per = [raw.get(str(k)) for k in scales]
    if len(scales) < 2 or any(not t or len(t) < 2 for t in per):
        return None
    rng = np.random.default_rng(seed)
    agg, ds = cost.get("aggregator", "median"), []
    for _ in range(draws):
        elapsed = [aggregate(list(rng.choice(t, size=len(t), replace=True)), agg)
                   for t in per]
        try:
            ds.append(float(estimate_cost_affine(scales, elapsed)["d"]))
        except Exception:                  # a resample the fit cannot solve
            continue
    if len(ds) < draws // 2:
        return None
    sd_ = float(np.std(ds, ddof=1))
    return sd_ if np.isfinite(sd_) and sd_ > 0 else None


def _mismatch_threshold(dof: int | None) -> float:
    """The |z| cutoff that actually delivers D_MISMATCH_Z's false-alarm rate.

    When se(d) comes from the spread of `dof + 1` independent probes,
    (d_hat - declared)/se is t distributed with `dof` degrees of freedom, not
    normal. Comparing it against 3 tests at a far looser alpha than 3 sounds:
    at dof = 4, P(|t| > 3) = 4%, fifteen times the normal's 0.27%.

    `dof = None` means the se is a within-probe one (fit covariance, or the
    bootstrap over repeats) and there is no honest dof to use: the cost arm
    measured those wrong by FACTORS, not by a tail (see D_PROBES), and no
    quantile fixes a scale error. The cutoff stays at D_MISMATCH_Z and the
    verdict carries the caveat instead.
    """
    alpha = 2.0 * float(stats.norm.sf(D_MISMATCH_Z))
    if dof is None or dof < 1:
        return D_MISMATCH_Z
    return float(stats.t.isf(alpha / 2.0, dof))


def _resolve_d(cost: dict, declared_override: float | None = None,
               *, trust_declared: bool = False):
    """The pilot's d. The CLOCK measures it; a declaration only CHECKS it.

    This inverts what this file used to do. Before 2026-08-29, a model that
    declared a `cost_hint` had that value written straight into constants.json
    and the timing demoted to a note -- so on srw, whose cost_hint returns
    exactly `i`, d = 1 entered BY DEFINITION and the cost probe never had to
    recover anything. That is the same class of defect tools/constants.py was
    written to close (see its docstring on FALLBACK_D), one layer up.

    Now: `d` is always the affine fit's, with its own standard error, and a
    declaration becomes an assertion scored as z = (d_hat - declared)/se(d_hat).
    |z| > D_MISMATCH_Z is reported loudly and recorded -- it does not stop the
    run (user's call; see D_MISMATCH_Z).

    `declared_override` is --assert-d, for models with no cost_hint; it takes
    the identical checked path. `trust_declared` is --trust-declared-d, the
    explicit escape hatch for a machine whose clock is not usable: it routes
    through constants.override(), so the number prints as `<-- NOT MEASURED`
    and can never pass for a measurement.

    Returns (Constant | None, check: dict, warnings: list[str]). A None
    Constant means no usable d exists at all and the caller must stop.
    """
    aff = cost.get("affine") or {}
    across = cost.get("d_across_probes") or {}
    declared = declared_override if declared_override is not None \
        else cost.get("declared_d")
    warnings = []
    if across.get("n", 0) >= 2:
        # Repeated probes: d is their mean and se is their spread, so the error
        # bar is on the quantity actually reported. This is the calibrated path.
        d_val, d_se = across["d"], across["se"]
        se_source = f"spread across {across['n']} cost probes"
        dof = int(across["n"]) - 1
    else:
        # One probe only (an older artifact, or every repeat's fit failed). The
        # se here is within-probe, and measured wrong by factors -- see D_PROBES.
        # Kept live for back-compat; any MISMATCH it raises says so.
        d_val, d_se = aff.get("d"), aff.get("d_se")
        se_source, dof = "affine fit covariance", None

    share = cost.get("overhead_share")
    if share is not None and share > OVERHEAD_SHARE_NOTE:
        # Not a refusal: this is the condition affine[] exists to survive.
        warnings.append(
            f"per-call overhead is {share:.0%} of the cheapest probe rung "
            f"(> {OVERHEAD_SHARE_NOTE:.0%}), so the pure power-law d_hat "
            f"({cost.get('d_hat')}) is measuring dispatch. The affine fit "
            f"separates it out and is what d uses.")

    check = {"declared": declared, "measured": d_val, "d_se": None,
             "z": None, "threshold": None, "verdict": None, "se_source": None,
             "n_probes": across.get("n"), "dof": dof, "overhead_share": share}

    if trust_declared:
        if declared is None:
            raise SystemExit("--trust-declared-d needs a declared d: the model "
                             "must have a cost_hint, or pass --assert-d.")
        check["verdict"] = "trusted-by-flag (NOT measured)"
        warnings.append(
            f"--trust-declared-d: d = {declared:g} was taken on trust and the "
            f"clock was NOT used. It is stamped as a user override and prints "
            f"as `<-- NOT MEASURED` everywhere it appears.")
        return override(declared, "d"), check, warnings

    if d_val is None:
        # The clock produced nothing usable. Never silently substitute the
        # declaration AS a measurement -- fall back to it stamped, or stop.
        if declared is None:
            return None, check, warnings
        check["verdict"] = "unmeasured, fell back to declared"
        warnings.append(
            f"the cost probe produced no usable d ("
            f"{aff.get('error', 'affine fit failed')}), so the declared value "
            f"{declared:g} is being used UNCHECKED, stamped as an override. "
            f"Widen the probe (PROBE_MAX_DOUBLINGS / PROBE_TARGET_SECONDS) to "
            f"get a real measurement.")
        return override(declared, "d"), check, warnings

    if d_se is None:
        d_se = _d_se_bootstrap(cost)
        se_source = "bootstrap over probe repeats" if d_se else None

    rungs = (f"{len(cost['scales'])} scales "
             f"({cost['scales'][0]}..{cost['scales'][-1]})")
    if dof is not None:
        prov = (f"pilot cost probe, mean of {across['n']} independent affine "
                f"fits over {rungs}, se from their spread")
    else:
        prov = f"pilot cost probe, affine fit over {rungs}"
        if se_source and se_source != "affine fit covariance":
            prov += f", se from {se_source}"
    c = measured(d_val, d_se, prov)
    check.update(d_se=d_se, se_source=se_source)

    if declared is None:
        check["verdict"] = "no declaration to check"
        return c, check, warnings

    if d_se is None or d_se <= 0:
        check["verdict"] = "unchecked (no se for d)"
        warnings.append(
            f"d was measured ({d_val:.4f}) and something declares {declared:g}, "
            f"but no standard error could be formed for d -- neither the fit "
            f"covariance nor a bootstrap over the probe repeats -- so the two "
            f"CANNOT be compared. Treat the agreement as unverified.")
        return c, check, warnings

    z = (d_val - declared) / d_se
    thr = _mismatch_threshold(dof)
    check["z"], check["threshold"] = float(z), thr
    if abs(z) > thr:
        check["verdict"] = "MISMATCH"
        caveat = "" if dof is not None else (
            "\n    CAVEAT: this se is a WITHIN-probe one, and on the cost arm "
            "those came out wrong by factors of 3-18 against the real "
            "probe-to-probe spread, so this alarm is weak evidence. Re-run the "
            "pilot to get the across-probe se (D_PROBES) before acting on it.")
        warnings.append(
            f"D MISMATCH: the clock measured d = {d_val:.4f} +/- {d_se:.4f} "
            f"({se_source}), but the declared cost says {declared:g} -- that is "
            f"z = {z:+.2f}, beyond +/-{thr:.2f}.\n"
            f"    Either the cost_hint is wrong, or this machine has stopped "
            f"being compute-bound (load, throttling, swap).\n"
            f"    d is the MEASURED value; every wall-clock prediction "
            f"downstream inherits this disagreement. Recorded in pilot.json "
            f"as cost.d_check." + caveat)
    else:
        check["verdict"] = "pass"
    return c, check, warnings


def pilot(recipe: dict, sd: Path, replicates: int, seed=None,
          existing: list | None = None, assert_d: float | None = None,
          trust_declared_d: bool = False) -> dict:
    """Draw `replicates` replicates, fit the constants, write them to `sd`."""
    model, params = recipe["model"], recipe.get("params", {})
    scales = [int(x) for x in recipe["scales"]]
    # `n` may be a scalar, a list, or an ALLOCATION RULE -- generate.py's own
    # resolver handles all three, so a pilot recipe can say
    # {"rule": "snr", "budget": 1e8, ...} exactly like any other recipe. That
    # matters here: a flat n starves the large scales and leaves omega1
    # unidentified (0.06 +/- 0.18 measured), while snr at the same wall clock
    # gives 0.98 +/- 0.28.
    n = resolve_n(recipe)
    _refuse_starved_scales(scales, n)

    reps = list(existing or [])
    base = seed if seed is not None else recipe.get("seed")
    have, want = len(reps), len(reps) + replicates
    spec = get_model(model)
    drawn_seconds, drawn_steps = 0.0, 0.0
    counts_now = n if isinstance(n, (list, tuple)) else [n] * len(scales)
    # skip=have: these streams EXTEND the pool, they do not restart it. Without
    # it --more redraws the replicates already on disk (see tools/rng.py:spawn).
    for k, ss in enumerate(spawn(base, replicates, skip=have)):
        print(f"  replicate {have + k + 1}/{want} ...",
              end="", flush=True, file=sys.stderr)
        t0 = time.perf_counter()
        reps.append(_pilot_replicate(model, params, scales, n, ss))
        dt = time.perf_counter() - t0
        drawn_seconds += dt
        # Steps, not seconds, is the budget unit (see CATALOG / the cost-model
        # discussion): the model's declared cost_hint is exact where a clock is
        # not. Throughput converts one to the other for THIS machine.
        drawn_steps += sum(c * (spec.cost_hint(i, params) if spec.cost_hint
                                else 1.0)
                           for i, c in zip(scales, counts_now))
        print(f" {dt:.1f}s", file=sys.stderr)

    R = len(reps)
    y = np.array([r["y_bar"] for r in reps], float)
    sig = np.array([r["sigma_log"] for r in reps], float)
    counts = np.array(n if isinstance(n, (list, tuple)) else [n] * len(scales), float)

    # Pooled then refitted once, never averaged fit-by-fit: fit_correction is
    # nonlinear, so averaging R fits converges to E[a1_hat], not a1 -- a bias
    # that no number of replicates removes (see allocation_table.measured_a1).
    nn = np.broadcast_to(counts, y.shape)
    y_pool = (y * nn).sum(axis=0) / nn.sum(axis=0)
    sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))
    fit = fit_correction(scales, y_pool, sigma_log=sig_pool)

    # The stated errors come from the SPREAD of the per-replicate fits, which
    # needs R >= 2. With one replicate every se is None and plan.py says so.
    per = [fit_correction(scales, np.array(r["y_bar"]),
                          sigma_log=np.array(r["sigma_log"])) for r in reps] \
        if R > 1 else []
    def spread(key):
        return float(np.std([p[key] for p in per], ddof=1) / np.sqrt(R)) if R > 1 else None

    cost = measure_cost_exponent(model, params, scales)

    cv_by_rep = np.array([r["cv"] for r in reps], float)     # (R, len(scales))
    cv_per_scale = cv_by_rep.mean(axis=0)
    cv_se = (float(np.std(cv_by_rep.mean(axis=1), ddof=1) / np.sqrt(R))
             if R > 1 else None)
    throughput = (drawn_steps / drawn_seconds) if drawn_seconds > 0 else None
    prov = (f"pilot, {R} replicate{'s' if R > 1 else ''}, pooled then refitted once"
            if R > 1 else "pilot, 1 replicate (no stderr available)")

    consts = {
        "omega1": measured(fit["omega1"], spread("omega1"), prov),
        "a1":     measured(fit["a1"], spread("a1"), prov),
        # cv's se is the spread of the per-REPLICATE mean cv, the same
        # construction as omega1's and a1's above -- not the spread ACROSS
        # SCALES, which is a property of the observable rather than an
        # uncertainty about it. Without it cv was the one constant in the
        # error budget with no error bar, so its influence on m0 could not be
        # weighed with the others (user, 2026-09-06).
        "cv":     measured(float(cv_per_scale.mean()), cv_se,
                           f"pilot, mean over {len(scales)} scales, spread "
                           f"{cv_per_scale.min():.4f}-{cv_per_scale.max():.4f}"),
    }
    consts["d"], d_check, d_warnings = _resolve_d(
        cost, declared_override=assert_d, trust_declared=trust_declared_d)
    if consts["d"] is None:
        raise SystemExit(
            f"\nthe cost probe produced no usable d, and nothing declares one.\n"
            f"  The affine fit cost(i) = a + b*i**d needs at least "
            f"{PROBE_MIN_SCALES} rungs and got "
            f"{len(cost.get('scales') or [])}: {cost.get('affine', {}).get('error', '')}\n"
            f"  Widen the probe (tools/cost_model.py: PROBE_MAX_DOUBLINGS, "
            f"PROBE_TARGET_SECONDS),\n"
            f"  or state it:  --assert-d <value>   (checked against the clock, "
            f"not a substitute for it)\n")
    cost["d_check"] = d_check

    sd.mkdir(parents=True, exist_ok=True)
    save(sd, consts)
    write_artifact(sd, "pilot", {
        "recipe": recipe, "replicates": R, "scales": scales,
        "n": [int(x) for x in counts],
        "y_bar": y_pool.tolist(), "sigma_log": sig_pool.tolist(),
        "cv_per_scale": cv_per_scale.tolist(),
        "direct_fit": fit, "per_replicate": reps, "cost": cost,
        "gamma_pilot": fit["gamma"], "a0_pilot": fit["a0"],
        "throughput": throughput, "drawn_seconds": drawn_seconds,
        "drawn_steps": drawn_steps, "d_warnings": d_warnings,
    }, produced_by="src/study/pilot.py")
    return {"constants": consts, "fit": fit, "cost": cost, "replicates": R,
            "reps": reps, "cv_per_scale": cv_per_scale, "throughput": throughput,
            "d_check": d_check, "d_warnings": d_warnings}


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-meta", "--meta", dest="meta", type=Path, default=None,
                   help="samples recipe for the pilot draw. Omit only with --more, "
                        "which reuses the recipe already recorded in the study.")
    p.add_argument("--study", required=True,
                   help="study name; the directory is <experiment>/data/<name>/")
    p.add_argument("--data-root", type=Path, default=None,
                   help="experiment data dir; defaults to the recipe's own")
    p.add_argument("--replicates", type=int, default=1,
                   help="replicates to draw. 1 gives no standard errors, so plan.py "
                        "cannot say how well determined the plan is -- use 3+ when "
                        "you intend to act on the answer")
    p.add_argument("--more", type=int, default=0,
                   help="add this many replicates to an existing pilot, keeping the "
                        "ones already drawn")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--assert-d", type=float, default=None, dest="assert_d",
                   help="a known cost exponent to CHECK the clock against, for "
                        "models with no cost_hint. Scored as z = (d_hat - "
                        "value)/se; a mismatch warns loudly and is recorded, "
                        "but never replaces the measurement or stops the run")
    p.add_argument("--trust-declared-d", action="store_true",
                   dest="trust_declared_d",
                   help="take the declared d on trust and skip the clock. For a "
                        "machine whose timings are unusable (shared load, "
                        "throttling). Stamped as a user override and printed as "
                        "`<-- NOT MEASURED` wherever it appears")
    a = p.parse_args(argv)

    existing = None
    if a.more:
        root = a.data_root or (Path(a.meta).resolve().parent.parent / "data"
                               if a.meta else None)
        if root is None:
            raise SystemExit("--more needs --data-root or -meta to locate the study")
        sd = study_dir(root, a.study)
        prev = artifact_path(sd, "pilot")
        if not prev.exists():
            raise SystemExit(f"no pilot in {sd} to add to; run without --more first")
        old = json.loads(prev.read_text())
        recipe, existing, reps = old["recipe"], old["per_replicate"], a.more
        print(f"adding {a.more} replicate(s) to the {old['replicates']} already in {sd}")
    else:
        if a.meta is None:
            raise SystemExit("-meta is required (or use --more on an existing pilot)")
        recipe = load_recipe(a.meta, "samples")
        root = a.data_root or default_out_dir(a.meta)
        sd = study_dir(root, a.study)
        reps = a.replicates

    r = pilot(recipe, sd, reps, seed=a.seed, existing=existing,
              assert_d=a.assert_d, trust_declared_d=a.trust_declared_d)

    print(f"\nstudy   = {sd}")
    print(f"model   = {recipe['model']}  scales = {r['fit'] and recipe['scales']}")
    print(f"\nconstants measured ({r['replicates']} replicate(s))")
    print(format_table(r["constants"]))
    if r["replicates"] < 2:
        print("\n  No standard errors: one replicate has no spread to measure. "
              "omega1\n  ranged 0.49-1.25 across single replicates of one configuration "
              "in this\n  repo's own runs, so treat it as indicative. "
              f"Add more:\n    python3 src/study/pilot.py --study {a.study} --more 3 "
              f"--data-root {root}")
    if r["throughput"]:
        print(f"\nthroughput  = {r['throughput']:.3g} steps/s "
              f"(this machine, from the pilot's own clock)")
    print(f"\ngamma from the pilot itself: {r['fit']['gamma']:.4f}  "
          f"(indicative -- the plan exists to measure it properly)")

    print(f"\nnext: python3 src/study/plan.py --study {a.study} --data-root {root}")
    # Last, and on stderr, so a mismatch is the final thing on screen rather
    # than something scrolled past above the constants table. stdout is flushed
    # first: without it the two streams interleave when the output is piped,
    # and the warning lands in the middle of the table it is meant to follow.
    sys.stdout.flush()
    for w in r["d_warnings"]:
        print(f"\n  !! {w}", file=sys.stderr)


if __name__ == "__main__":
    _main()
