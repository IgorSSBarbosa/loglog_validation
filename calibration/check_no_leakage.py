"""Move the truth and see whether the answer moves with it.

The question this repo keeps having to answer is not "are the estimates
right?" -- on srw they are, and that is exactly the problem: srw's truths
(gamma = 1/2, omega_1 = 1, a_1 = -1/4, d = 1) are simple numbers that a
hardcoded default can sit on. `tools/constants.py` documents the version of
that bug this project actually shipped, where FALLBACK_OMEGA1 = 1.0155 and a
`--omega1` defaulting to 1.0 made every published table read omega_1 = 1
whether or not anything had been measured.

Reading the code is one answer. This is the other, and the stronger one: PLANT
a truth the code cannot know, drawn from a seeded generator, appearing in no
recipe, no module constant and no default, then run the real pipeline and see
whether what comes out tracks what went in. Two criteria, and the second is
the one that catches a leak:

  1. Unbiasedness    |theta_hat - theta| / se(theta_hat) within threshold,
                     for every planted cell.
  2. Responsiveness  regress theta_hat on theta across the grid:
                     slope = 1 +/- 0.15 and R^2 >= 0.9.

A hardcoded constant passes NOTHING here: it produces a flat line, slope 0,
and criterion 2 fails loudly. Criterion 1 alone would not catch it, because a
hardcoded srw truth is CORRECT on srw. That asymmetry is the point.

Four arms, each pointed at a different estimator and a different leak channel:

  gamma        synthetic, pure power law, planted (gamma, a0). Recovered by
               the ARTICLE'S OWN estimator, tools/loglog.py's
               `gamma_closed_form` (eq. 523-526). No correction term, so that
               estimator is unbiased by construction and criterion 1 is a
               statement about sampling error alone.
  correction   synthetic with one correction term, planted
               (gamma, a0, a1, omega_1). Recovered by src/study/pilot.py --
               the real pilot, drawing through the real generator and fitting
               with tools/correction.py's `fit_correction`. This is the arm
               that covers the constants the budget machinery consumes.
  cost         synthetic with a tunable CPU burn (models/synthetic.py's
               `cost_scale`/`cost_d`), planted d. Recovered by the pilot's own
               cost path: `climb_to_target` -> `fit_cost_probe` ->
               `_resolve_d`. Without the burn this model has d = 0 exactly and
               cannot exercise the cost machinery at all.
  srw          the real lattice walk at q != 1/2, where the truth moves away
               from the values that were once hardcoded. The reference is the
               EXACT E|S_k| from the binomial law, computed here and never
               handed to the estimator; gamma_hat is compared against the same
               estimator applied to that exact mean, so the comparison
               isolates sampling error from the estimator's own bias.

Cost: about 4 minutes at the defaults. Keep the machine reasonably idle -- the
`cost` arm times a spin loop, and a competing job inflates it (the other three
arms are unaffected).

CLI:
    python3 calibration/check_no_leakage.py
    python3 calibration/check_no_leakage.py --arms correction cost --draws 8
    python3 calibration/check_no_leakage.py --plant-seed 1234 --json /tmp/nl.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                           # repo root; calibration/ -> ../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, write_artifact  # noqa: E402
from tools.correction import fit_correction  # noqa: E402
from tools.loglog import gamma_closed_form  # noqa: E402
from tools.rng import spawn  # noqa: E402

from src.study.pilot import (  # noqa: E402
    _d_se_bootstrap, _resolve_d, measure_cost_exponent, pilot, study_dir)

#: The ladder every statistical arm runs on: rho = 2, k = 1..8, so
#: `gamma_closed_form` accepts it as the consecutive grid it requires with
#: m0 = 0. Reaches down to i = 2 because that is where the correction term is
#: large enough to identify omega_1 cheaply, and out to 256 because a slope
#: needs a lever arm.
LADDER = [2 ** k for k in range(1, 9)]
RHO = 2.0
M0 = 0

#: Noise level for the synthetic arms, as Var(xi) -- so cv = sqrt(0.6) = 0.775,
#: which is srw's own measured cv (0.774) rather than a convenient small
#: number. A quieter fixture would make every arm easier in a way the real
#: model is not.
SIGMA_INF2 = 0.6

#: Planting ranges. Deliberately NOT centred on srw's truths: gamma = 1/2,
#: omega_1 = 1, a_1 = -1/4 and d = 1 all lie inside these ranges but nowhere
#: near their middles, so a constant frozen at any of them is visibly wrong at
#: most cells rather than "close enough" at all of them.
GAMMA_RANGE = (0.2, 0.8)
A0_RANGE = (0.5, 1.5)
OMEGA1_RANGE = (0.4, 2.5)
A1_RANGE = (-0.6, -0.1)

#: Cost exponents for the burn arm. Five rather than the three originally
#: planned: criterion 2 is a regression, and three points make R^2 a formality.
COST_DS = (0.5, 0.75, 1.0, 1.5, 2.0)

#: Where the cost probe starts climbing, and how long one call must burn
#: there. 1 ms against ~33 us of per-call dispatch puts the overhead term at
#: ~3% of the cheapest rung -- comfortably under the 5% the affine fit wants,
#: and the reason the burn is a spin rather than an array touch (see
#: models/synthetic.py:_burn).
COST_START_SCALE = 64
COST_BURN_START_SECONDS = 1e-3

#: q values for the srw arm. 0.5 is included as the anchor everything else in
#: this repo runs at; the rest carry the walk from diffusive (gamma = 1/2)
#: towards ballistic (gamma -> 1), which is a truth no default can be sitting on.
SRW_QS = (0.5, 0.55, 0.6, 0.7, 0.8)

#: Criterion 2, exactly as specified: slope of theta_hat on theta, and R^2.
SLOPE_TOLERANCE = 0.15
R2_MIN = 0.9

#: Criterion 1's threshold. The plan said |z| <= 3 per cell; that is the right
#: number for ONE cell and the wrong one for a run with ~25 of them, where a
#: standard error estimated from R replicates is itself a t-statistic with
#: R-1 degrees of freedom. At R = 6 and 25 cells, P(some |t_5| > 3) = 55%: a
#: green run would mean nothing and a red one would mean nothing either. So
#: the reported verdict uses a Bonferroni-corrected two-sided t quantile at
#: FAMILY_ALPHA over the whole run, and the table still flags every cell past
#: the plain 3 for a human's eye.
FAMILY_ALPHA = 0.01
PLAIN_Z = 3.0

#: The second half of criterion 1, and the reason it has two halves. `se` here
#: is the spread of the per-replicate estimates: SAMPLING error, and nothing
#: else. An estimator with any systematic floor at all -- and a nonlinear
#: least-squares fit has one -- therefore fails a pure z test as soon as n is
#: large enough, because se shrinks with n and the floor does not. Measured on
#: the correction arm: a cell with omega_1 = 2.43 (a correction that lives
#: essentially only at i = 2) recovered gamma = 0.28268 against a planted
#: 0.28201 -- z = +12 on an se of 5.6e-5, and a relative error of 0.24%.
#:
#: That is a real, small bias worth knowing about; it is not what this file is
#: looking for. A hardcoded constant misses by tens of percent, not by 0.2%.
#: So a cell satisfies criterion 1 if it agrees within its own stated error OR
#: to better than this fraction of the truth -- and the report prints both, so
#: which of the two carried it is never hidden.
REL_TOLERANCE = 0.01


def _plant(rng: np.random.Generator, lo: float, hi: float, k: int) -> list[float]:
    """`k` planted values, uniform on (lo, hi), rounded only for printing later.

    The generator is seeded and its seed is recorded in the artifact (ground
    rule 5), so a rerun reproduces the grid exactly -- but the VALUES exist
    nowhere in the repo, which is the property the whole check depends on.
    """
    return [float(x) for x in rng.uniform(lo, hi, size=k)]


def _se(values) -> float | None:
    """Standard error of the mean of `values`; None when there is no spread."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    s = float(np.std(v, ddof=1) / np.sqrt(v.size))
    return s if np.isfinite(s) and s > 0 else None


def _z(hat: float, truth: float, se: float | None) -> float | None:
    return None if not se else float((hat - truth) / se)


def _rel(row: dict) -> float:
    """|theta_hat - theta| / |theta|, or inf where the truth is zero."""
    truth = float(row["planted"])
    if truth == 0:
        return float("inf")
    return abs(float(row["recovered"]) - truth) / abs(truth)


def responsiveness(planted, recovered) -> dict:
    """Criterion 2: regress recovered on planted. Slope 0 is what a leak looks like.

    Returns {'slope', 'intercept', 'r2', 'n'}. A degenerate grid -- fewer than
    three usable points, or no spread at all in the planted values -- returns
    None for all three rather than a number that cannot mean anything.
    """
    x = np.asarray(planted, dtype=float)
    y = np.asarray(recovered, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": None, "intercept": None, "r2": None, "n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": None if r2 is None else float(r2), "n": int(x.size)}


def _run_pilot(model: str, params: dict, n: int, replicates: int, seed,
               sd: Path) -> dict:
    """One cell of a statistical arm, through the REAL pilot.

    Deliberately `src/study/pilot.py:pilot` rather than a local loop over
    `generate` + `fit_correction`: the question is what the code that runs
    does, and the pilot is what runs. It draws through the one sampler,
    summarizes with tools/summary.py, pools then refits once, and takes its
    standard errors from the spread of the per-replicate fits -- and it writes
    pilot.json, which is what this function reads back, so the numbers checked
    here are the numbers a study would act on.
    """
    recipe = {"kind": "samples", "model": model, "scales": list(LADDER),
              "n": int(n), "params": dict(params)}
    out = pilot(recipe, sd, replicates, seed=seed)
    art = json.loads(artifact_path(sd, "pilot").read_text())
    return {"pilot": out, "artifact": art}


def _per_replicate_fits(art: dict) -> list[dict]:
    """`fit_correction` on each replicate separately -- the pilot's own spread.

    The pilot keeps the standard errors of omega_1 and a_1 (they are the
    constants it exists to hand on) but not of gamma or a_0, which are not
    constants any allocation consumes. They are checked here, so their spread
    is recomputed from the per-replicate summaries the pilot recorded, by the
    same route it uses for the two it keeps.
    """
    return [fit_correction(art["scales"], np.array(r["y_bar"], dtype=float),
                           sigma_log=np.array(r["sigma_log"], dtype=float))
            for r in art["per_replicate"]]


# --------------------------------------------------------------------------
# Arm 1 -- gamma, through the article's own estimator
# --------------------------------------------------------------------------

def arm_gamma(draws: int, replicates: int, n: int, plant_rng, seed) -> dict:
    """Plant (gamma, a0) with NO correction; recover gamma by eq. (523)-(526).

    The correction is left out on purpose. `gamma_closed_form` is a weighted
    sum of log Y_bar with no correction term in it, so a planted correction
    would show up as bias -- real, expected, and nothing to do with leakage.
    Arm 2 puts the correction back and uses the estimator that models it.
    """
    gammas = _plant(plant_rng, *GAMMA_RANGE, draws)
    a0s = _plant(plant_rng, *A0_RANGE, draws)
    rows = []
    for k, (g, a0, ss) in enumerate(zip(gammas, a0s, spawn(seed, draws))):
        params = {"gamma": g, "a0": a0, "sigma_inf2": SIGMA_INF2}
        sd = _cell_dir("gamma", k)
        got = _run_pilot("synthetic", params, n, replicates, ss, sd)
        art = got["artifact"]
        # Pooled y_bar, as the pilot recorded it, through the article's estimator.
        g_hat = gamma_closed_form(art["scales"], art["y_bar"], RHO, M0)
        per = [gamma_closed_form(art["scales"], r["y_bar"], RHO, M0)
               for r in art["per_replicate"]]
        se = _se(per)
        rows.append({"cell": k, "planted": {"gamma": g, "a0": a0},
                     "gamma": {"planted": g, "recovered": float(g_hat),
                               "se": se, "z": _z(g_hat, g, se)},
                     "study": str(sd.relative_to(ROOT))})
    return {"arm": "gamma", "estimator": "tools/loglog.py:gamma_closed_form "
                                         "(article eq. 523-526)",
            "model": "synthetic", "n_per_scale": n, "replicates": replicates,
            "scales": list(LADDER), "sigma_inf2": SIGMA_INF2, "cells": rows,
            "parameters": ["gamma"]}


# --------------------------------------------------------------------------
# Arm 2 -- the correction constants, through the pilot
# --------------------------------------------------------------------------

def arm_correction(draws: int, replicates: int, n: int, plant_rng, seed) -> dict:
    """Plant (gamma, a0, a1, omega_1); recover all four through the pilot."""
    gammas = _plant(plant_rng, *GAMMA_RANGE, draws)
    a0s = _plant(plant_rng, *A0_RANGE, draws)
    omega1s = _plant(plant_rng, *OMEGA1_RANGE, draws)
    a1s = _plant(plant_rng, *A1_RANGE, draws)
    rows = []
    for k, (g, a0, om, a1, ss) in enumerate(
            zip(gammas, a0s, omega1s, a1s, spawn(seed, draws))):
        params = {"gamma": g, "a0": a0, "sigma_inf2": SIGMA_INF2,
                  "corrections": [[a1, om]]}
        sd = _cell_dir("correction", k)
        got = _run_pilot("synthetic", params, n, replicates, ss, sd)
        art, consts = got["artifact"], got["pilot"]["constants"]
        fit = art["direct_fit"]
        per = _per_replicate_fits(art)
        row = {"cell": k, "planted": {"gamma": g, "a0": a0, "omega1": om, "a1": a1},
               "study": str(sd.relative_to(ROOT)),
               "rel_rmse": fit["rel_rmse"], "converged": fit["converged"]}
        for name, truth in (("gamma", g), ("a0", a0), ("omega1", om), ("a1", a1)):
            # omega1 and a1 carry the pilot's OWN standard error (the number a
            # study would act on); gamma and a0 get the same spread computed
            # the same way, because the pilot does not keep theirs.
            se = (consts[name].se if name in consts
                  else _se([p[name] for p in per]))
            row[name] = {"planted": truth, "recovered": float(fit[name]),
                         "se": se, "z": _z(fit[name], truth, se)}
        rows.append(row)
    return {"arm": "correction",
            "estimator": "src/study/pilot.py -> tools/correction.py:fit_correction",
            "model": "synthetic", "n_per_scale": n, "replicates": replicates,
            "scales": list(LADDER), "sigma_inf2": SIGMA_INF2, "cells": rows,
            "parameters": ["gamma", "a0", "omega1", "a1"]}


# --------------------------------------------------------------------------
# Arm 3 -- the cost exponent, against a burn the clock has to find
# --------------------------------------------------------------------------

def arm_cost(replicates: int, cost_ds=COST_DS, start=COST_START_SCALE,
             burn=COST_BURN_START_SECONDS) -> dict:
    """Plant d in the spin burn; recover it with the pilot's cost path.

    `cost_scale` is chosen per cell so that one call at the probe's start scale
    burns the same `burn` seconds whatever d is -- otherwise the arm would vary
    two things at once, and a d recovered from a probe that ran for 40 ms would
    not be comparable with one that ran for 4 s.

    Replicates here are repeated PROBES, not repeated draws: the error being
    estimated is timing jitter on this machine, which is the only error a spin
    loop has.

    The honest objection to this arm, recorded rather than hidden: the burn
    targets elapsed time and the probe measures elapsed time, so it is mildly
    circular -- it shows `fit_cost_probe` recovers an exponent, not that real
    work has one. That is why the srw arm exists alongside it, where the work
    is real and d = 1 comes from the algorithm rather than from a target.
    """
    rows = []
    for d in cost_ds:
        params = {"gamma": 0.5, "a0": 1.0, "sigma_inf2": 0.0,
                  "cost_d": float(d), "cost_scale": burn / start ** float(d)}
        ests, probes, resolved = [], [], []
        for _ in range(replicates):
            probe = measure_cost_exponent("synthetic", params, [start])
            probes.append(probe)
            # The pilot's d is the mean across the probe's own repeated climbs
            # (D_PROBES), so that is what this arm must score -- not the last
            # climb's affine fit, which is one of the numbers averaged.
            across = probe.get("d_across_probes") or {}
            aff = probe.get("affine") or {}
            if across.get("d") is not None:
                ests.append(float(across["d"]))
            elif aff.get("d") is not None:
                ests.append(float(aff["d"]))
            # Every probe goes through the pilot's own decision path, not just
            # the last one: `_resolve_d` scores the model's declared cost
            # against ONE probe, and how often that verdict is wrong on a model
            # whose declaration is exact by construction is a measurement, not
            # an anecdote. See the mismatch_rate reported below.
            const, check, _ = _resolve_d(probe)
            resolved.append({"value": const.value if const else None,
                             "source": const.source if const else None,
                             "is_override": bool(const and const.is_override),
                             "verdict": check.get("verdict"), "z": check.get("z"),
                             "d_se": check.get("d_se"),
                             "se_source": check.get("se_source")})
        last = probes[-1]
        d_hat = float(np.mean(ests)) if ests else float("nan")
        se = _se(ests)
        # Four standard errors for the same d, recorded side by side because
        # they do not agree and the pilot's mismatch gate depends on which one
        # it uses: the across-CALL spread (`se`, what repeated measurement
        # actually shows), the affine fit's Gauss-Newton covariance from ONE
        # climb at dof = 1, the bootstrap over that climb's own repeat timings,
        # and -- since 2026-09-14, and what `_resolve_d` now prefers -- the
        # spread across the D_PROBES climbs inside a single call.
        se_cov = (last.get("affine") or {}).get("d_se")
        se_boot = _d_se_bootstrap(last)
        se_within_probe_set = (last.get("d_across_probes") or {}).get("se")
        misses = sum(1 for r in resolved if r["verdict"] == "MISMATCH")
        rows.append({
            "cell": len(rows), "d": {"planted": float(d), "recovered": d_hat,
                                     "se": se, "z": _z(d_hat, d, se)},
            "per_replicate_d": ests,
            "se_across_probes": se, "se_fit_covariance": se_cov,
            "se_bootstrap_over_repeats": se_boot,
            # What `_resolve_d` now uses: the spread inside ONE call's probe
            # set. It is the same KIND of quantity as `se_across_probes` above
            # -- both measure variation BETWEEN climbs -- which the two
            # single-probe ses are not, and that is the whole fix. Read them as
            # one estimate each, not as a precise ratio: an sd from 5 numbers
            # carries ~35% of itself.
            "se_within_probe_set": se_within_probe_set,
            "probe_scales": last["scales"],
            "overhead_share": last.get("overhead_share"),
            "declared_d": last.get("declared_d"),
            "resolved": resolved[-1], "resolved_all": resolved,
            "mismatches": misses, "probes": len(resolved),
        })
    probes_total = sum(r["probes"] for r in rows)
    misses_total = sum(r["mismatches"] for r in rows)
    return {"arm": "cost",
            "estimator": "src/study/pilot.py:measure_cost_exponent "
                         "(climb_to_target -> fit_cost_probe -> _resolve_d)",
            "model": "synthetic + spin burn", "replicates": replicates,
            "start_scale": start, "burn_seconds_at_start": burn,
            "cells": rows, "parameters": ["d"],
            # The FALSE-ALARM rate of `_resolve_d`'s D_MISMATCH_Z gate. Every
            # cell here declares its cost exponent exactly (the burn realizes
            # what cost_hint says), so every MISMATCH is a false alarm.
            "mismatch_rate": (misses_total / probes_total) if probes_total else None,
            "probes": probes_total, "mismatches": misses_total}


# --------------------------------------------------------------------------
# Arm 4 -- srw off its anchor: q != 1/2
# --------------------------------------------------------------------------

def exact_mean_abs_srw(k: int, q: float) -> float:
    """E|S_k| for a walk of k steps with P(+1) = q, exactly.

    S_k = 2*Bin(k, q) - k, so E|S_k| = sum_j P(Bin = j) * |2j - k|. Computed
    here in the CHECKER, never in the model or any estimator: models/srw.py
    deliberately has no target_fn, so nothing in the sampling or fitting path
    can see this number (user's decision D2, plans/three_experiment_ladder.md).
    """
    from scipy.stats import binom
    j = np.arange(k + 1)
    return float(np.dot(binom.pmf(j, k, q), np.abs(2 * j - k)))


def arm_srw(replicates: int, n: int, qs=SRW_QS, seed=None) -> dict:
    """Move srw's truth with q, and see whether gamma-hat moves with it.

    The reference is not "1/2". It is the value the article's own estimator
    returns on the EXACT mean over this ladder -- i.e. the planted truth for
    THIS functional at THIS window, which removes the estimator's correction
    bias from the comparison exactly rather than assuming it is small. What is
    left in z is sampling error, and what criterion 2 sees is whether
    gamma_hat tracks a target that runs from ~0.5 (diffusive) to ~0.9
    (ballistic) as q moves.
    """
    rows = []
    for k, (q, ss) in enumerate(zip(qs, spawn(seed, len(qs)))):
        exact = [exact_mean_abs_srw(int(i), float(q)) for i in LADDER]
        target = float(gamma_closed_form(LADDER, exact, RHO, M0))
        sd = _cell_dir("srw", k)
        got = _run_pilot("srw", {"q": float(q)}, n, replicates, ss, sd)
        art = got["artifact"]
        g_hat = float(gamma_closed_form(art["scales"], art["y_bar"], RHO, M0))
        per = [gamma_closed_form(art["scales"], r["y_bar"], RHO, M0)
               for r in art["per_replicate"]]
        se = _se(per)
        rows.append({"cell": k, "q": float(q),
                     "gamma": {"planted": target, "recovered": g_hat,
                               "se": se, "z": _z(g_hat, target, se)},
                     "exact_mean_at_ladder": exact,
                     "study": str(sd.relative_to(ROOT))})
    return {"arm": "srw", "estimator": "tools/loglog.py:gamma_closed_form "
                                       "(article eq. 523-526)",
            "model": "srw", "n_per_scale": n, "replicates": replicates,
            "scales": list(LADDER), "qs": [float(q) for q in qs],
            "cells": rows, "parameters": ["gamma"],
            "reference": "gamma_closed_form applied to the exact E|S_k|"}


# --------------------------------------------------------------------------
# The negative control: put a leak back, and check that the check fails
# --------------------------------------------------------------------------

def inject_leak(leaks: dict) -> None:
    """Hardcode a constant on purpose, so this file can be shown to FAIL.

    A check that has only ever passed is not evidence; the stage-3.0 finding
    in plans/saverepo.md was exactly that, in the other direction (an
    insensitivity test passes vacuously on an unidentified parameter). So the
    leak this file exists to catch is reproducible on demand:

        python3 calibration/check_no_leakage.py --arms correction \
                --inject-leak omega1=1.0155

    1.0155 is not an arbitrary number -- it is the literal `FALLBACK_OMEGA1`
    that `src/budget/allocation_table.py` used to carry (see
    tools/constants.py). Under it the omega_1 row goes flat: slope 0.000, no
    standard error at all (every replicate returns the same constant), and a
    relative error of 40-70% against the planted truths, while gamma, a_0 and
    a_1 stay green -- so the report localizes the leak rather than merely
    failing.

    Patched at the two module globals the arms actually call through, which is
    why this is a function here rather than a note telling you to edit the
    file: the point is to exercise the real path with one number replaced.
    """
    import src.study.pilot as _pilot

    stat = {k: v for k, v in leaks.items() if k != "d"}
    if stat:
        real_fit = _pilot.fit_correction

        def leaky_fit(*args, **kwargs):
            out = dict(real_fit(*args, **kwargs))
            out.update(stat)
            return out

        _pilot.fit_correction = leaky_fit
        globals()["fit_correction"] = leaky_fit
    if "d" in leaks:
        real_cost = globals()["measure_cost_exponent"]

        def leaky_cost(*args, **kwargs):
            probe = real_cost(*args, **kwargs)
            if isinstance(probe.get("affine"), dict):
                probe["affine"]["d"] = leaks["d"]
                probe["affine"]["d_se"] = None
            # Since 2026-09-14 `_resolve_d` PREFERS the across-probe block, so
            # the injection has to reach it too. Patching only `affine` would
            # leave this control quietly measuring the honest path and
            # reporting PASS -- a broken detector that looks like a working one,
            # which is the exact failure mode this whole file exists to catch.
            if isinstance(probe.get("d_across_probes"), dict):
                n = int(probe["d_across_probes"]["n"])
                probe["d_probes"] = [leaks["d"]] * n
                probe["d_across_probes"].update(d=leaks["d"], sd=0.0, se=0.0)
            return probe

        globals()["measure_cost_exponent"] = leaky_cost


def parse_leaks(items) -> dict:
    """`--inject-leak omega1=1.0155 d=1.0` -> {"omega1": 1.0155, "d": 1.0}."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--inject-leak wants NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in ("gamma", "a0", "omega1", "a1", "d"):
            raise SystemExit(
                f"--inject-leak: unknown constant {name!r}; one of "
                f"gamma, a0, omega1, a1, d")
        out[name] = float(value)
    return out


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

#: Where each cell's study lives. Fixed and deterministic, overwritten on
#: rerun (ground rule 6) -- and under experiments/*/data/, which is gitignored:
#: the record that gets committed is no_leakage.json, not 25 pilot draws.
DATA_ROOT = ROOT / "experiments" / "00_synthetic" / "data" / "no_leakage"


def _cell_dir(arm: str, k: int) -> Path:
    return study_dir(DATA_ROOT, f"{arm}_{k:02d}")


def _rows_for(arm: dict, name: str) -> list[dict]:
    """Every cell's {planted, recovered, se, z} for one parameter of one arm."""
    return [c[name] for c in arm["cells"] if name in c]


def bonferroni_z(n_cells: int, dof: int, alpha: float = FAMILY_ALPHA) -> float | None:
    """Two-sided t quantile at alpha/n_cells -- criterion 1's actual threshold.

    `se` here is estimated from R replicates, so (theta_hat - theta)/se is
    t-distributed with R-1 degrees of freedom, not normal, and a run asks the
    question ~25 times. Both corrections push the same way and neither is
    optional: without them a green run is luck.
    """
    if dof < 1:
        return None
    from scipy.stats import t
    return float(t.ppf(1.0 - alpha / (2.0 * max(n_cells, 1)), dof))


#: Above this ratio of median se to the spread of the planted values, a
#: parameter is not identified at the budget it was run at, and criterion 2
#: cannot say anything about leakage either way. This is stage 3.0's lesson
#: (plans/saverepo.md) turned into a printed word: an insensitivity test on an
#: unidentified parameter passes -- or here, fails -- for reasons that have
#: nothing to do with what it is looking for.
UNIDENTIFIED_SE_RATIO = 0.2


def diagnose(rows: list[dict], resp: dict) -> str:
    """Why criterion 2 came out as it did -- the column that stops a misreading.

    A FAIL means one of two very different things, and they look alike in a
    slope alone:

      flat          the recovered values barely move at all. That IS what a
                    hardcoded constant looks like from the outside.
      unidentified  they move, wildly, because the standard error is
                    comparable to the whole planted range. A budget problem,
                    not a leak -- and the run says nothing about leakage.
    """
    got = np.array([r["recovered"] for r in rows], dtype=float)
    want = np.array([r["planted"] for r in rows], dtype=float)
    ses = [r["se"] for r in rows if r["se"]]
    got_spread = float(np.std(got, ddof=1)) if got.size > 1 else 0.0
    want_spread = float(np.std(want, ddof=1)) if want.size > 1 else 0.0
    if want_spread == 0:
        return "no spread in the planted values"
    if got_spread <= 1e-9 * max(1.0, float(np.mean(np.abs(got)))):
        return "flat -- the output does not depend on the truth"
    if ses and float(np.median(ses)) > UNIDENTIFIED_SE_RATIO * want_spread:
        return "unidentified at this budget -- se is comparable to the range"
    if resp["slope"] is not None and abs(resp["slope"] - 1.0) <= SLOPE_TOLERANCE:
        return "responsive"
    return "responds, but not with slope 1"


def judge(arms: list[dict], alpha: float = FAMILY_ALPHA) -> dict:
    """Both criteria, per (arm, parameter), plus the run-wide verdict."""
    checks, total_cells = [], 0
    for arm in arms:
        for name in arm["parameters"]:
            total_cells += len(_rows_for(arm, name))
    for arm in arms:
        dof = int(arm["replicates"]) - 1
        thresh = bonferroni_z(total_cells, dof, alpha)
        for name in arm["parameters"]:
            rows = _rows_for(arm, name)
            zs = [abs(r["z"]) for r in rows if r["z"] is not None]
            rels = [_rel(r) for r in rows]
            resp = responsiveness([r["planted"] for r in rows],
                                  [r["recovered"] for r in rows])
            max_z = max(zs) if zs else None
            # Every cell must clear ONE of the two halves; a cell with no
            # standard error at all has only the relative half to clear.
            per_cell = [((r["z"] is not None and thresh is not None
                          and abs(r["z"]) <= thresh)
                         or _rel(r) <= REL_TOLERANCE) for r in rows]
            ok1 = bool(rows) and all(per_cell)
            ok2 = (resp["slope"] is not None and resp["r2"] is not None
                   and abs(resp["slope"] - 1.0) <= SLOPE_TOLERANCE
                   and resp["r2"] >= R2_MIN)
            checks.append({
                "arm": arm["arm"], "parameter": name, "cells": len(rows),
                "max_abs_z": max_z, "z_threshold": thresh, "dof": dof,
                "cells_past_plain_3": sum(1 for z in zs if z > PLAIN_Z),
                "max_rel_error": max(rels) if rels else None,
                "cells_carried_by_rel_tolerance": sum(
                    1 for r, ok in zip(rows, per_cell)
                    if ok and not (r["z"] is not None and thresh is not None
                                   and abs(r["z"]) <= thresh)),
                "unbiased": bool(ok1), "responsiveness": resp,
                "responsive": bool(ok2), "diagnosis": diagnose(rows, resp),
                "no_se": sum(1 for r in rows if not r["se"]),
            })
    return {"checks": checks, "total_cells": total_cells,
            "family_alpha": alpha, "slope_tolerance": SLOPE_TOLERANCE,
            "r2_min": R2_MIN, "plain_z": PLAIN_Z,
            "rel_tolerance": REL_TOLERANCE,
            "passed": all(c["unbiased"] and c["responsive"] for c in checks)}


def format_report(arms: list[dict], verdict: dict) -> str:
    """The tables: every cell, then the two criteria per parameter."""
    out = []
    for arm in arms:
        out.append(f"\n=== arm {arm['arm']} -- {arm['estimator']}")
        for name in arm["parameters"]:
            rows = _rows_for(arm, name)
            out.append(f"\n  {name:<8}{'planted':>10} {'recovered':>11} "
                       f"{'se':>10} {'z':>8} {'rel err':>9}")
            for r in rows:
                se = f"{r['se']:.4g}" if r["se"] else "--"
                z = f"{r['z']:+.2f}" if r["z"] is not None else "--"
                flag = "  <--" if (r["z"] is not None
                                   and abs(r["z"]) > PLAIN_Z) else ""
                out.append(f"  {'':<8}{r['planted']:>10.4f} "
                           f"{r['recovered']:>11.4f} {se:>10} {z:>8} "
                           f"{_rel(r):>8.2%}{flag}")
    for arm in arms:
        if arm["arm"] == "cost" and arm.get("mismatch_rate") is not None:
            # Every cell here declares its cost exponent exactly, so the rate
            # printed IS the false-alarm rate of _resolve_d's gate. It read
            # 32-34% while the se came from inside a single probe; taking it
            # from the spread ACROSS probes (D_PROBES, 2026-09-14) is what this
            # line exists to keep honest, in either direction.
            rate = arm["mismatch_rate"]
            out.append(
                f"\n  _resolve_d called MISMATCH on {arm['mismatches']} of "
                f"{arm['probes']} probes ({rate:.0%}) -- every one a FALSE "
                f"ALARM,\n  since the burn realizes exactly what cost_hint "
                f"declares." + ("\n  The gate is calibrated: it does not fire "
                                "on a correct declaration."
                                if rate <= 0.02 else
                                "\n  The gate is MIS-calibrated: se(d) is too "
                                "small for the cutoff it is\n  compared "
                                "against. See D_PROBES in src/study/pilot.py."))
    out.append(f"\n\n{'arm':<12}{'param':<8}{'max|z|':>8}{'thresh':>8}"
               f"{'max rel':>9}{'slope':>9}{'R^2':>8}   verdict")
    out.append("-" * 80)
    for c in verdict["checks"]:
        resp = c["responsiveness"]
        mz = f"{c['max_abs_z']:.2f}" if c["max_abs_z"] is not None else "--"
        th = f"{c['z_threshold']:.2f}" if c["z_threshold"] is not None else "--"
        mr = f"{c['max_rel_error']:.2%}" if c["max_rel_error"] is not None else "--"
        sl = f"{resp['slope']:.3f}" if resp["slope"] is not None else "--"
        r2 = f"{resp['r2']:.4f}" if resp["r2"] is not None else "--"
        why = ([] if c["unbiased"] else ["biased"]) + \
              ([] if c["responsive"] else ["not responsive"])
        v = "PASS" if not why else ("FAIL: " + " and ".join(why)
                                    + f" [{c['diagnosis']}]")
        out.append(f"{c['arm']:<12}{c['parameter']:<8}{mz:>8}{th:>8}"
                   f"{mr:>9}{sl:>9}{r2:>8}   {v}")
    carried = sum(c["cells_carried_by_rel_tolerance"] for c in verdict["checks"])
    if carried:
        out.append(f"\n{carried} cell(s) missed the z threshold but agree to "
                   f"better than {REL_TOLERANCE:.0%} of the planted value, which "
                   f"criterion 1 accepts\n(see REL_TOLERANCE): se here is "
                   f"sampling error only, and a nonlinear fit has a systematic "
                   f"floor below it.")
    return "\n".join(out)


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=["gamma", "correction", "cost", "srw"],
                   choices=["gamma", "correction", "cost", "srw"])
    p.add_argument("--draws", type=int, default=6,
                   help="planted cells per statistical arm")
    p.add_argument("--replicates", type=int, default=6,
                   help="replicates per cell; the standard error comes from "
                        "their spread, so 2 is the floor and 6 is the default")
    p.add_argument("--cost-replicates", type=int, default=8,
                   help="repeated PROBES per planted d (timing jitter only)")
    p.add_argument("--n-gamma", type=int, default=200_000,
                   help="samples per scale, gamma arm")
    p.add_argument("--n-correction", type=int, default=1_000_000,
                   help="samples per scale, correction arm. Larger because "
                        "omega_1 near the top of its range leaves a correction "
                        "visible only at the smallest scales")
    p.add_argument("--n-srw", type=int, default=20_000,
                   help="samples per scale, srw arm (each costs i steps)")
    p.add_argument("--plant-seed", type=int, default=20260904,
                   help="seeds the PLANTED TRUTHS. Recorded in the artifact; "
                        "the values themselves appear nowhere in the repo")
    p.add_argument("--seed", type=int, default=515,
                   help="seeds the sampling, independently of the planting")
    p.add_argument("--alpha", type=float, default=FAMILY_ALPHA,
                   help="family-wise level for criterion 1's threshold")
    p.add_argument("--inject-leak", nargs="+", default=None, metavar="NAME=VALUE",
                   help="hardcode a constant on purpose and confirm this check "
                        "catches it. The run is then EXPECTED to fail, and the "
                        "exit code inverts: 0 means the control worked. E.g. "
                        "--inject-leak omega1=1.0155 (the old FALLBACK_OMEGA1)")
    p.add_argument("--json", type=Path, default=None,
                   help=f"where to write the record "
                        f"(default: {artifact_path(HERE, 'no_leakage').name} "
                        f"in calibration/)")
    a = p.parse_args(argv)

    leaks = parse_leaks(a.inject_leak)
    if leaks:
        inject_leak(leaks)
        print("\n  !! NEGATIVE CONTROL: " + ", ".join(
            f"{k} hardcoded to {v:g}" for k, v in sorted(leaks.items())) +
            "\n     This run is EXPECTED to fail; a PASS would mean the check "
            "cannot see a leak.", file=sys.stderr)

    names = ["gamma", "correction", "cost", "srw"]
    # One independent stream per arm, for BOTH the planting and the sampling,
    # so adding or dropping an arm does not change what the others draw
    # (ground rule 2) -- and so `--arms correction` reruns the same grid the
    # full run planted, which a single shared generator consumed in order
    # would not.
    plant_rngs = {k: np.random.default_rng(ss)
                  for k, ss in zip(names, spawn(a.plant_seed, 4))}
    arm_seeds = dict(zip(names, spawn(a.seed, 4)))

    t0 = time.perf_counter()
    arms = []
    for name in a.arms:
        print(f"\narm {name} ...", file=sys.stderr)
        if name == "gamma":
            arms.append(arm_gamma(a.draws, a.replicates, a.n_gamma,
                                  plant_rngs["gamma"], arm_seeds["gamma"]))
        elif name == "correction":
            arms.append(arm_correction(a.draws, a.replicates, a.n_correction,
                                       plant_rngs["correction"],
                                       arm_seeds["correction"]))
        elif name == "cost":
            arms.append(arm_cost(a.cost_replicates))
        elif name == "srw":
            arms.append(arm_srw(a.replicates, a.n_srw, seed=arm_seeds["srw"]))
    elapsed = time.perf_counter() - t0

    verdict = judge(arms, a.alpha)
    print(format_report(arms, verdict))

    payload = {"arms": arms, "verdict": verdict, "elapsed_seconds": elapsed,
               "injected_leak": leaks or None,
               "plant_seed": a.plant_seed, "seed": a.seed,
               "ladder": list(LADDER), "rho": RHO, "m0": M0,
               "ranges": {"gamma": list(GAMMA_RANGE), "a0": list(A0_RANGE),
                          "omega1": list(OMEGA1_RANGE), "a1": list(A1_RANGE),
                          "cost_d": list(COST_DS), "srw_q": list(SRW_QS)}}
    # The default path is the committed RECORD, so only a complete, uninjected
    # run is allowed to overwrite it -- otherwise `--arms cost` (or, worse, an
    # --inject-leak control) would quietly replace the evidence with a fragment
    # of itself. A partial run still prints everything; it just says where to
    # put it if you want to keep it.
    complete = set(a.arms) == {"gamma", "correction", "cost", "srw"} and not leaks
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(payload, indent=2))
        written = a.json
    elif complete:
        written = write_artifact(HERE, "no_leakage", payload,
                                 produced_by="calibration/check_no_leakage.py")
    else:
        written = None
        print(f"\npartial run ({'injected leak' if leaks else 'arms: ' + ' '.join(a.arms)})"
              f" -- {artifact_path(HERE, 'no_leakage').name} left alone.\n"
              f"  Pass --json <path> to keep this one.")

    print(f"\n{verdict['total_cells']} planted cells, {elapsed / 60:.1f} min, "
          f"plant seed {a.plant_seed}")
    if written:
        print(f"output = {written}")
    if leaks:
        # The control: a PASS here is the failure. Exit 0 only if the injected
        # leak was actually caught.
        caught = not verdict["passed"]
        print(f"\nNEGATIVE CONTROL {'worked' if caught else 'DID NOT WORK'} -- "
              f"the injected leak ({', '.join(sorted(leaks))}) was "
              f"{'caught' if caught else 'MISSED'}.")
        if not caught:
            print("  A check that cannot fail is not evidence. Something in "
                  "this file\n  stopped looking at the parameter that was "
                  "hardcoded.")
        raise SystemExit(0 if caught else 1)
    if verdict["passed"]:
        print("\nPASS -- every parameter tracks the truth that was planted, and "
              "nothing\n  in the pipeline was told what that truth was.")
    else:
        print("\nFAIL -- see the verdict table above, and READ THE DIAGNOSIS in "
              "brackets.\n  `flat` is a leak: the output does not depend on the "
              "truth.\n  `unidentified at this budget` is not -- the cells are too "
              "noisy for the\n  question to have been asked at all, and the run says "
              "nothing either way.\n  Raise --n-correction / --n-srw / --replicates "
              "and run it again.")
    raise SystemExit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    _main()
