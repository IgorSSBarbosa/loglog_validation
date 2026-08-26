"""Check if the error bars of this repo are calibrated using SRW

Runs Experiment B's estimation pipeline end to end, hundreds of times, against
srw's known ground truth, and counts how often the interval it reports actually
contains the truth. See tools/coverage.py for why this is a separate question
from whether the ESTIMATES are right (they are; that was Experiment B).

The error bar under test is the one the pipeline really quotes:

    se = sd(per-replicate fits, ddof=1) / sqrt(R)      # src/budget/allocation_table.py
    interval = estimate +/- q * se

with R = 5. Both halves of that are suspect at R = 5 -- an sd from 5 points is
itself noisy, and q = 1.96 (normal) should arguably be 2.776 (t on 4 dof) --
so the script measures both quantile choices side by side rather than assuming
which is right.

Two arms, because "the error bar is wrong" and "the CLT has not kicked in" are
different diagnoses:

  planted -- y_bar drawn straight from N(mu_i, cv_i^2 mu_i^2 / n_i) using the
             EXACT mean and variance of |S_k|. Isolates the fit's error bar.
             Cheap, so it runs many trials.
  srw     -- real srw draws at reduced n. Costs orders of magnitude more per
             trial, so it runs few, and exists only to check that the planted
             arm's Gaussian assumption is not itself the thing being measured.

On ground truth: the exact E|S_k| and Var|S_k| live HERE, in a driver, and are
deliberately not registered in tools/models.py as a `target_fn`. Same rule as
allocation_experiment.py's `true_gamma` -- truth may plant data and score a
finished answer, never reach an estimator. See experiments/01_srw/README.md.

CLI:
    python3 src/estimate/check_coverage.py                      # planted arm, 500 trials
    python3 src/estimate/check_coverage.py --trials 2000
    python3 src/estimate/check_coverage.py --arm srw --trials 40
    python3 src/estimate/check_coverage.py --arm both --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from math import comb, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports
sys.path.insert(0, str(ROOT / "src" / "generate"))  # the shared draw loop

from allocation import (  # noqa: E402
    ladder,
    n_for_budget,
    rate_exponent,
    rate_exponent_se,
)
from correction import fit_correction  # noqa: E402
from wilson import (  # noqa: E402
    format_interval,
    sigma_se,
    wilson_interval,
)
from coverage import (  # noqa: E402
    coverage_multi,
    coverage_test,
    format_result,
    rescore,
    se_ratio,
)
from loglog import gamma_closed_form  # noqa: E402
from models import get_model  # noqa: E402
from generate import generate  # noqa: E402

# Experiment B's actual configuration (experiments/01_srw/recipes/samples_omega1.json,
# snr allocation at B = 5e10), replayed verbatim so the coverage measured here
# is the coverage of the interval that was really reported.
SCALES = [8, 16, 32, 64, 128, 256]
N_PER_SCALE = [166893, 667574, 2670298, 10681193, 42724772, 170899089]
REPLICATES = 5

# Ground truth for |S_k| -- see experiments/01_srw/README.md.
TRUTH = {"omega1": 1.0, "a1": -0.25, "gamma": 0.5, "a0": sqrt(2 / np.pi)}


def exact_mean(k: int) -> float:
    """E|S_k| = k * C(k-1, floor((k-1)/2)) * 2^-(k-1), exactly.

    Evaluated as one exact rational and converted once, NOT as
    `k * comb(...) * 2.0**-(k-1)`: that form overflows for k >~ 1000, where
    the binomial is a big int of thousands of bits and the power of two has
    already underflowed to 0.0. Fraction -> float rounds correctly at any k.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    return float(Fraction(k * comb(k - 1, (k - 1) // 2), 1 << (k - 1)))


def exact_sd(k: int) -> float:
    """sd|S_k| = sqrt(k - (E|S_k|)^2), since E[S_k^2] = k exactly."""
    mu = exact_mean(k)
    return sqrt(max(k - mu * mu, 0.0))


def _fit_one(y_bar, sigma_log) -> dict:
    return fit_correction(SCALES, y_bar, sigma_log=sigma_log)


def _planted_replicate(rng, n_per_scale) -> tuple[np.ndarray, np.ndarray]:
    """One replicate's y_bar and its sigma_log, drawn from the exact moments."""
    mu = np.array([exact_mean(k) for k in SCALES])
    sd = np.array([exact_sd(k) for k in SCALES])
    n = np.asarray(n_per_scale, dtype=float)
    se_mu = sd / np.sqrt(n)
    y_bar = rng.normal(mu, se_mu)
    # sigma_log as the pipeline computes it: se of the mean, delta-method'd
    # onto the log scale using the OBSERVED mean, not mu -- the estimator has
    # no access to mu.
    return y_bar, se_mu / y_bar


def _mean_and_sigma_log(s) -> tuple[float, float]:
    """Both per-scale statistics the pipeline uses, in one pass over `s`.

    sigma_log is the se of the mean delta-method'd onto the log scale using
    the OBSERVED mean, not mu -- the estimator has no access to mu.
    """
    m = float(np.mean(s))
    return m, float(np.std(s, ddof=1)) / sqrt(len(s)) / m


def _srw_replicate(rng, n_per_scale) -> tuple[np.ndarray, np.ndarray]:
    """One replicate drawn for real, consuming `rng` exactly as generate() does.

    Deliberately NOT a call to src/generate/generate.py, unlike the ladder
    draws in allocation_experiment and verify_prediction. Those own their
    stream; this one is a callback the coverage harness drives from a single
    long-lived Generator across thousands of trials, so it must take a live
    `rng` rather than a seed -- the `draw(rng, n)` contract it shares with
    `_planted_replicate`.

    The equivalence with generate() is therefore an assertion, and assertions
    in comments rot. It is pinned instead by
    tools/tests/test_check_coverage.py::test_srw_replicate_matches_generate,
    which requires the two to agree bit-for-bit on the same seed.
    """
    spec = get_model("srw")
    y_bar, sigma_log = np.empty(len(SCALES)), np.empty(len(SCALES))
    for j, k in enumerate(SCALES):
        y_bar[j], sigma_log[j] = _mean_and_sigma_log(
            spec.simulate(k, int(n_per_scale[j]), {"q": 0.5}, rng))
    return y_bar, sigma_log


def make_wilson_experiment(m0: int, m: int, rho: float, n: int, *,
                           sigma_inf2: float, sigma_max2: float,
                           a1: float, omega1: float, level: float = 0.95,
                           Lambda: float | None = None,
                           delta: float | None = None):
    """Coverage arm for the article's Wilson bound (tools/wilson.py, eq. 720).

    Differs from the other arms in what is being tested. Those ask whether an
    interval built from R replicate fits is the width it claims. This one asks
    whether a BOUND holds -- so the expected answer is not 0.95 but "at least
    0.95", and the interesting number is how far above.

    Uniform n, because the theorem assumes it, and gamma_closed_form (the
    article's own linear estimator) because the theorem is about that estimator
    and no other. One replicate per trial: the bound needs no replicates, which
    is the entire point of it.

    Returned as (estimate, se_equivalent) with se_equivalent = half_width/q so
    that `coverage_test`'s interval(est, se, dof=None) reproduces the bound
    exactly; that keeps the scoring inside the tested harness rather than
    re-implementing hit-counting here.
    """
    from scipy.stats import norm

    scales = ladder(m0, m, rho)
    mu = np.array([exact_mean(k) for k in scales])
    sd = np.array([exact_sd(k) for k in scales])
    q = float(norm.ppf(0.5 + level / 2))

    def experiment(rng):
        y_bar = rng.normal(mu, sd / sqrt(n))
        g = float(gamma_closed_form(scales, y_bar, rho, m0))
        res = wilson_interval(g, n, m, m0, rho, sigma_inf2=sigma_inf2,
                              sigma_max2=sigma_max2, a1=a1, omega1=omega1,
                              Lambda=Lambda, delta=delta, level=level)
        return g, res["half_width"] / q

    return experiment


def check_planting(n: int, trials: int, seed: int | None = None,
                   scales=None) -> list[dict]:
    """Is the planted arm's Gaussian a faithful stand-in for real srw draws?

    The planted arm asserts y_bar_i ~ N(mu_i, sigma_i^2 / n_i) with the exact
    moments of |S_k|. This tests that assertion head-on -- many independent
    real y_bar from srw, one-sample Kolmogorov-Smirnov against the planted
    normal -- instead of trying to run the whole 5-replicate pipeline on real
    draws, which at Experiment B's n costs hours per coverage point.

    Deliberately run at SMALL n, which is the conservative direction: y_bar is
    a sample mean, so its normality can only improve as n grows. Experiment B
    used n from 1.7e5 to 1.7e8; if the normal fits at n = 1e4 it certainly fits
    there. A test at the real n would cost 10^4 times more and prove less.

    Returns one row per scale with the KS statistic, its p-value, and the
    moment comparison.
    """
    from scipy import stats

    scales = list(scales or SCALES)
    rng = np.random.default_rng(seed)
    spec = get_model("srw")
    rows = []
    for k in scales:
        mu, sd = exact_mean(k), exact_sd(k)
        se = sd / sqrt(n)
        ybar = np.array([float(np.mean(spec.simulate(k, n, {"q": 0.5}, rng)))
                         for _ in range(trials)])
        ks = stats.kstest(ybar, "norm", args=(mu, se))
        rows.append({
            "scale": k, "n": n, "trials": trials,
            "exact_mean": mu, "observed_mean": float(ybar.mean()),
            "exact_se": se, "observed_sd": float(ybar.std(ddof=1)),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        })
    return rows


def make_experiment(draw, n_per_scale, replicates: int = REPLICATES,
                    params=("omega1", "a1", "gamma")):
    """An `experiment(rng) -> {name: (estimate, se)}` for coverage_multi.

    Each call replays a whole Experiment B: `replicates` independent runs of
    the ladder, combined exactly as src/budget/allocation_table.py's
    `measured_correction` combines them, and reported under BOTH combination
    rules so they can be compared on identical draws.

    The pipeline's own combination is a MISMATCHED PAIR, which is the specific
    thing worth measuring. Its point estimate comes from pooling y_bar across
    replicates and refitting once -- correct, because the per-replicate fits
    are nonlinear in the data, so averaging them keeps a bias that never
    shrinks. Its stated se comes from the SPREAD of those same per-replicate
    fits, over sqrt(R). Centre and width therefore describe two different
    estimators, and nothing guarantees the second is the right width for the
    first. `<param>/mean` is the matched-but-biased alternative, scored on the
    same trials so the cost of each is measured rather than argued.
    """
    n = np.asarray(n_per_scale, dtype=float)

    def experiment(rng):
        fits, ys, sigs = [], [], []
        for _ in range(replicates):
            y_bar, sigma_log = draw(rng, n_per_scale)
            fits.append(_fit_one(y_bar, sigma_log))
            ys.append(y_bar)
            sigs.append(sigma_log)
        y = np.asarray(ys)
        sig = np.asarray(sigs)
        nn = np.broadcast_to(n, y.shape)                        # same n each replicate
        y_pool = (y * nn).sum(axis=0) / nn.sum(axis=0)          # weighted by sample count
        sig_pool = 1.0 / np.sqrt((1.0 / sig ** 2).sum(axis=0))  # inverse-variance
        pooled = _fit_one(y_pool, sig_pool)

        out = {}
        for param in params:
            v = np.array([f[param] for f in fits], dtype=float)
            se = float(v.std(ddof=1) / sqrt(len(v)))
            out[f"{param}/pooled"] = (float(pooled[param]), se)
            out[f"{param}/mean"] = (float(v.mean()), se)
        return out

    return experiment


def make_rate_experiment(budgets, replicates: int, d: float, omega1: float,
                         rho: float, m: int):
    """Coverage of `rate_exponent_se` -- an ANALYTIC error bar, not a replicate one.

    Different failure mode from the fits above: that se is derived from the
    known noise in an RMSE (sd(log RMSE) ~ 1/sqrt(2R)) rather than measured,
    so it can be wrong by being derived under an assumption that does not hold,
    not by being noisy. Worth its own arm.
    """
    from allocation import optimal_allocation

    truth_slope = -omega1 / (d + 2 * omega1)

    def experiment(rng):
        rmses = []
        for B in budgets:
            m0 = optimal_allocation(B=B, d=d, omega1=omega1, rho=rho, m=m)["m0"]
            n = n_for_budget(B, m0, m, rho, d)
            scales = ladder(m0, m, rho)
            mu = np.array([exact_mean(k) for k in scales])
            sd = np.array([exact_sd(k) for k in scales])
            hats = []
            for _ in range(replicates):
                y_bar = rng.normal(mu, sd / sqrt(n))
                hats.append(gamma_closed_form(scales, y_bar, rho, m0))
            h = np.asarray(hats)
            rmses.append(float(sqrt(np.mean((h - TRUTH["gamma"]) ** 2))))
        return rate_exponent(budgets, rmses), rate_exponent_se(budgets, replicates)

    return experiment, truth_slope


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=("planted", "srw", "rate", "planting",
                                     "wilson", "both", "all"),
                   default="planted")
    p.add_argument("--wilson-m0", type=int, nargs="+", default=[2, 4, 6, 8, 10],
                   help="m0 values to sweep in the wilson arm")
    p.add_argument("--wilson-n", type=int, default=100_000,
                   help="uniform n per scale for the wilson arm")
    p.add_argument("--planting-n", type=int, default=10_000,
                   help="n per y_bar for the planting arm (small on purpose: "
                        "normality of a sample mean only improves with n, so a "
                        "pass here implies a pass at Experiment B's much larger n)")
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--replicates", type=int, default=REPLICATES,
                   help=f"replicates per experiment (default {REPLICATES}, "
                        "matching what Experiment B actually ran)")
    p.add_argument("--params", nargs="+", default=["omega1", "a1", "gamma"])
    p.add_argument("--centre", choices=("pooled", "mean", "both"), default="pooled",
                   help="how replicates are combined into the point estimate; "
                        "'pooled' is what the pipeline does")
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--n-scale", type=float, default=1.0,
                   help="multiply every n by this (srw arm: use <1 to keep it affordable)")
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    arms = {"both": ["planted", "planting"],
            "all": ["planting", "planted", "rate"]}.get(a.arm, [a.arm])
    out, t0 = {}, time.perf_counter()

    for arm in arms:
        if arm == "wilson":
            sig_inf2 = float(np.pi / 2 - 1)          # half-normal limit of Var(xi)
            print(f"\n{'=' * 72}\nwilson arm: article eq. (720) as a bound on gamma\n"
                  f"uniform n={a.wilson_n:,}, m={6}, rho=2, sigma_inf2={sig_inf2:.4f}, "
                  f"a1={TRUTH['a1']}, omega1={TRUTH['omega1']}\n"
                  f"a BOUND, so coverage >= {a.level:.0%} is a pass; the question is "
                  f"how conservative\n{'=' * 72}")
            print(f"  {'m0':>4} {'B_fs':>11} {'se_term':>11} {'half':>11} "
                  f"{'dominant':>9} {'coverage':>9} {'95% CI':>16}")
            for m0 in a.wilson_m0:
                scales = ladder(m0, 6, 2.0)
                cv2 = [(exact_sd(k) / exact_mean(k)) ** 2 for k in scales]
                exp = make_wilson_experiment(
                    m0, 6, 2.0, a.wilson_n, sigma_inf2=sig_inf2,
                    sigma_max2=max(cv2), a1=TRUTH["a1"], omega1=TRUTH["omega1"],
                    level=a.level)
                r = coverage_test(exp, TRUTH["gamma"], trials=a.trials,
                                  level=a.level, dof=None, seed=a.seed)
                w = wilson_interval(0.5, a.wilson_n, 6, m0, 2.0,
                                    sigma_inf2=sig_inf2, sigma_max2=max(cv2),
                                    a1=TRUTH["a1"], omega1=TRUTH["omega1"],
                                    level=a.level)
                lo, hi = r["coverage_ci"]
                print(f"  {m0:>4} {w['B_fs']:>11.3e} {w['se_term']:>11.3e} "
                      f"{w['half_width']:>11.3e} {w['dominant']:>9} "
                      f"{r['coverage']:>9.3f} {f'[{lo:.3f}, {hi:.3f}]':>16}")
                out[f"wilson/m0={m0}"] = r
            continue
        if arm == "planting":
            print(f"\n{'=' * 72}\nplanting arm: is y_bar really N(mu, sigma^2/n)?"
                  f"  n={a.planting_n:,}, {a.trials} draws per scale\n"
                  f"(small n on purpose -- conservative: normality only improves "
                  f"as n grows)\n{'=' * 72}")
            rows = check_planting(a.planting_n, a.trials, a.seed)
            print(f"  {'k':>5} {'exact mean':>12} {'observed':>12} "
                  f"{'exact se':>11} {'observed sd':>12} {'KS':>8} {'p':>8}  verdict")
            for r in rows:
                ok = "ok" if r["ks_p"] > 0.01 else "NON-NORMAL"
                print(f"  {r['scale']:>5} {r['exact_mean']:>12.6f} "
                      f"{r['observed_mean']:>12.6f} {r['exact_se']:>11.3e} "
                      f"{r['observed_sd']:>12.3e} {r['ks_stat']:>8.4f} "
                      f"{r['ks_p']:>8.4f}  {ok}")
            out["planting"] = {"rows": rows}
            continue
        if arm == "rate":
            print(f"\n{'=' * 72}\nrate_exponent_se (analytic error bar)\n{'=' * 72}")
            exp, truth = make_rate_experiment(
                [1e6, 1e7, 1e8, 1e9], 40, 1.0, 1.0, 2.0, 6)
            for dof in (None, 3):
                r = coverage_test(exp, truth, trials=max(50, a.trials // 5),
                                  level=a.level, dof=dof, seed=a.seed,
                                  progress=True)
                label = f"rate slope [q={'normal' if dof is None else f't({dof})'}]"
                print(format_result(r, label) + "\n")
                out[label] = r
            continue

        n = [max(1, int(x * a.n_scale)) for x in N_PER_SCALE]
        draw = _planted_replicate if arm == "planted" else _srw_replicate
        print(f"\n{'=' * 72}\n{arm} arm: R={a.replicates} replicates/experiment, "
              f"{a.trials} experiments, n scaled x{a.n_scale:g}\n"
              f"n = {n}\n{'=' * 72}")
        centres = ["pooled", "mean"] if a.centre == "both" else [a.centre]
        exp = make_experiment(draw, n, a.replicates, tuple(a.params))
        truths = {f"{p}/{c}": TRUTH[p] for p in a.params for c in ("pooled", "mean")}
        res = coverage_multi(exp, truths, trials=a.trials, level=a.level,
                             dofs=(None, a.replicates - 1), seed=a.seed,
                             progress=True)
        for label, r in sorted(res.items()):
            if label.split("/")[1].split(" ")[0] not in centres:
                continue
            print(format_result(r, f"{arm}/{label}") + "\n")
            out[f"{arm}/{label}"] = r

    print(f"{'=' * 72}\nsummary ({time.perf_counter() - t0:.0f}s)\n{'=' * 72}")
    print(f"  {'test':<38} {'coverage':>9} {'95% CI':>18}  verdict")
    for label, r in out.items():
        if "coverage" not in r:
            continue
        lo, hi = r["coverage_ci"]
        v = "ok" if r["calibrated"] else (
            "UNDERCOVERS" if r["coverage"] < r["nominal"] else "overcovers")
        print(f"  {label:<38} {r['coverage']:>9.3f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}  {v}")

    # The number Experiment B actually printed was "estimate +/- 1 se", which is
    # a 68.3% claim, not a 95% one. Re-scored from the same trials (free), so
    # both levels are answered on identical draws.
    if any("values" in r for r in out.values()):
        print(f"\n  the '+/- 1 se' convention, re-scored at the 68.3% level:")
        print(f"  {'test':<38} {'coverage':>9} {'95% CI':>18}  verdict")
        for label, r in out.items():
            if "values" not in r:
                continue
            rr = rescore(r, level=0.6827)
            lo, hi = rr["coverage_ci"]
            v = "ok" if rr["calibrated"] else (
                "UNDERCOVERS" if rr["coverage"] < rr["nominal"] else "overcovers")
            print(f"  {label:<38} {rr['coverage']:>9.3f} "
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}  {v}")

    if a.json:
        a.json.write_text(json.dumps(
            {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                 for kk, vv in v.items() if kk not in ("values", "ses")}
             for k, v in out.items()},
            indent=2, sort_keys=True))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    _main()
