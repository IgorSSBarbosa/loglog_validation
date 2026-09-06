"""Run every function in the repo once, on purpose, and say how it behaved.

Neither a test suite nor a calibration measurement, and it earns its place
between them. `tools/tests/` asserts closed forms on the pieces it knows are
worth asserting; the other files in `calibration/` measure statistical
properties of the pipeline. This one asks a blunter question that neither
does: **for every public function and every CLI flag in the repo, does calling
it do something sensible?** Including the paths nobody calls -- the error
branches, the flag combinations, the degenerate inputs -- which is exactly
where a defect can sit for months without a passing test noticing.

It is ordered by dependency, and that ordering is the method. `tools/` is
exercised first, leaf modules before the ones that import them, so that when
`allocation_constants` is checked its `closed_form_weights` has already been
checked separately. `models/` next, then the `src/` drivers (which call
tools/), then `calibration/` (which calls src/). A failure in a later stage
whose earlier stages passed is a failure of the composition, not of a piece.

Three outcomes, and the third is the useful one:

    PASS   the call did what its docstring says
    FAIL   it did not -- a defect, reported with what was expected and got
    NOTE   it behaved as written, and the behaviour is worth a human's eye:
           a misleading message, a flag that does nothing, dead code

NOTEs are not failures and never exit non-zero. They are the audit's actual
output -- FAILs get fixed and disappear, notes are judgements someone has to
make.

CLI:
    python3 calibration/exercise_all.py                 # everything (~7 min)
    python3 calibration/exercise_all.py --stage tools   # just the leaf layer
    python3 calibration/exercise_all.py --stage static  # dead code, layering
    python3 calibration/exercise_all.py --list          # what it would run
    python3 calibration/exercise_all.py --json out.json
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                           # repo root; calibration/ -> ../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

class Audit:
    """Collects one line per check. Deliberately tiny -- no test framework.

    pytest would want fixtures and collection; what this needs is to run a
    call, keep going when it raises, and print an ordered transcript. A
    failure here is a finding to write down, not a red build.
    """

    def __init__(self, verbose: bool = False, fail_fast: bool = False):
        self.rows: list[dict] = []
        self.section = ""
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.t0 = time.perf_counter()

    # -- reporting ---------------------------------------------------------
    def begin(self, name: str, subject: str = "") -> None:
        self.section = name
        head = f"{name}" + (f"  --  {subject}" if subject else "")
        print(f"\n{'=' * 78}\n{head}\n{'=' * 78}")

    def _record(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append({"section": self.section, "status": status,
                          "name": name, "detail": detail})
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "NOTE": " note "}[status]
        if status != "PASS" or self.verbose:
            print(f"  [{mark}] {name}" + (f"\n           {detail}" if detail else ""))
        if status == "FAIL" and self.fail_fast:
            raise SystemExit("--fail-fast: stopping at the first failure")

    def note(self, name: str, detail: str) -> None:
        """Behaved as written; a human should look at it anyway."""
        self._record("NOTE", name, detail)

    # -- the three ways to state an expectation ----------------------------
    def check(self, name: str, fn, expect=True, detail: str = "") -> object:
        """Run `fn`; PASS when its result is truthy (or equals `expect`)."""
        try:
            got = fn()
        except Exception as exc:                       # noqa: BLE001 - reporting
            self._record("FAIL", name, f"raised {type(exc).__name__}: {exc}\n"
                                       f"           {_where(exc)}")
            return None
        ok = (got is expect) if isinstance(expect, bool) else (got == expect)
        if ok or (expect is True and got):
            self._record("PASS", name, detail)
        else:
            self._record("FAIL", name, f"expected {expect!r}, got {got!r}. {detail}")
        return got

    def close(self, name: str, fn, want: float, tol: float, detail: str = "") -> None:
        """PASS when fn() is within `tol` (absolute) of `want`."""
        try:
            got = float(fn())
        except Exception as exc:                       # noqa: BLE001 - reporting
            self._record("FAIL", name, f"raised {type(exc).__name__}: {exc}\n"
                                       f"           {_where(exc)}")
            return
        if math.isfinite(got) and abs(got - want) <= tol:
            self._record("PASS", name, f"{got:.6g} vs {want:.6g}")
        else:
            self._record("FAIL", name,
                         f"got {got!r}, wanted {want:.6g} +/- {tol:.3g}. {detail}")

    def raises(self, name: str, exc_type, fn, contains: str | None = None) -> None:
        """PASS when fn() raises `exc_type`, optionally with `contains` in its text."""
        try:
            got = fn()
        except exc_type as exc:
            text = str(exc)
            if contains is not None and contains.lower() not in text.lower():
                self._record("FAIL", name,
                             f"raised {exc_type.__name__} but the message never says "
                             f"{contains!r}: {text!r}")
            else:
                self._record("PASS", name, text.splitlines()[0][:90] if text else "")
            return
        except Exception as exc:                       # noqa: BLE001 - reporting
            self._record("FAIL", name, f"raised {type(exc).__name__} "
                                       f"(wanted {exc_type.__name__}): {exc}")
            return
        self._record("FAIL", name, f"did not raise {exc_type.__name__}; returned {got!r}")

    # -- summary -----------------------------------------------------------
    def counts(self) -> dict:
        out = {"PASS": 0, "FAIL": 0, "NOTE": 0}
        for r in self.rows:
            out[r["status"]] += 1
        return out

    def report(self) -> int:
        c = self.counts()
        print(f"\n{'=' * 78}\nsummary  "
              f"({time.perf_counter() - self.t0:.0f}s)\n{'=' * 78}")
        by_section: dict[str, dict] = {}
        for r in self.rows:
            s = by_section.setdefault(r["section"], {"PASS": 0, "FAIL": 0, "NOTE": 0})
            s[r["status"]] += 1
        print(f"  {'section':<34}{'pass':>7}{'fail':>7}{'note':>7}")
        for name, s in by_section.items():
            print(f"  {name:<34}{s['PASS']:>7}{s['FAIL']:>7}{s['NOTE']:>7}")
        print(f"  {'':<34}{'-' * 21}")
        print(f"  {'total':<34}{c['PASS']:>7}{c['FAIL']:>7}{c['NOTE']:>7}")

        for status, title in (("FAIL", "failures"), ("NOTE", "notes for a human")):
            rows = [r for r in self.rows if r["status"] == status]
            if not rows:
                continue
            print(f"\n{title}:")
            for r in rows:
                print(f"  - [{r['section']}] {r['name']}")
                for line in r["detail"].splitlines():
                    print(f"      {line.strip()}")
        return 1 if c["FAIL"] else 0


def _where(exc: Exception) -> str:
    """The last repo frame in a traceback -- where it actually went wrong."""
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if str(ROOT) in f.filename]
    if not frames:
        return ""
    f = frames[-1]
    return f"at {Path(f.filename).relative_to(ROOT)}:{f.lineno} in {f.name}()"


def run_cli(args: list[str], *, expect_code: int = 0, cwd: Path = ROOT,
            timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a driver the way a human does, and keep its streams.

    Subprocess rather than calling `_main(argv)` in-process on purpose: the
    thing under test includes argparse, the sys.path bootstrap each entry
    point does for itself, and the exit code -- none of which an in-process
    call exercises.
    """
    proc = subprocess.run([sys.executable, *args], cwd=cwd, timeout=timeout,
                          capture_output=True, text=True)
    proc.args_used = args                                # for the failure message
    return proc


def cli_ok(a: Audit, name: str, args: list[str], *, expect_code: int = 0,
           stdout_has: str | None = None, stderr_has: str | None = None,
            creates: Path | None = None, timeout: int = 600
            ) -> subprocess.CompletedProcess | None:
    """One CLI invocation, with its exit code and (optionally) its output checked."""
    try:
        p = run_cli(args, timeout=timeout)
    except subprocess.TimeoutExpired:
        a._record("FAIL", name, f"timed out after {timeout}s: {' '.join(args)}")
        return None
    problems = []
    if p.returncode != expect_code:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-6:]
        problems.append(f"exit {p.returncode}, wanted {expect_code}. "
                        + " | ".join(t.strip() for t in tail))
    if stdout_has is not None and stdout_has.lower() not in p.stdout.lower():
        problems.append(f"stdout never says {stdout_has!r}")
    if stderr_has is not None and stderr_has.lower() not in (p.stderr or "").lower():
        problems.append(f"stderr never says {stderr_has!r}")
    if creates is not None and not creates.exists():
        problems.append(f"did not create {creates}")
    if problems:
        a._record("FAIL", name,
                  "; ".join(problems) + f"\n           cmd: {' '.join(args)}")
    else:
        a._record("PASS", name, " ".join(args[1:]))
    return p


def quiet(fn, *args, **kwargs):
    """Call `fn`, swallowing whatever it prints. Returns its value."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*args, **kwargs)


# ===========================================================================
# STAGE 1 -- tools/, leaf modules first (nothing here imports src/ or models/)
# ===========================================================================

def sec_rng(a: Audit) -> None:
    """tools/rng -- seeding, and recording a seed so a run can be redrawn"""
    from tools.rng import as_seed_sequence, seed_record, spawn

    a.begin("tools/rng", "seeding, and recording a seed so a run can be redrawn")

    a.check("as_seed_sequence(None) draws fresh entropy",
            lambda: isinstance(as_seed_sequence(None).entropy, int))
    a.check("as_seed_sequence(int) keeps the entropy",
            lambda: as_seed_sequence(7).entropy, expect=7)
    ss = np.random.SeedSequence(11)
    a.check("as_seed_sequence(SeedSequence) is identity",
            lambda: as_seed_sequence(ss) is ss)
    kid = ss.spawn(3)[2]
    a.check("a spawned child round-trips through seed_record",
            lambda: as_seed_sequence(seed_record(kid)).spawn_key, expect=(2,))
    a.check("seed_record of an un-spawned seed is a bare int (old metadata still loads)",
            lambda: seed_record(np.random.SeedSequence(5)), expect=5)
    a.check("seed_record of a spawned seed keeps both halves",
            lambda: sorted(seed_record(kid)), expect=["entropy", "spawn_key"])
    a.raises("a dict seed without 'entropy' says so", ValueError,
             lambda: as_seed_sequence({"spawn_key": [0]}), contains="entropy")

    a.check("spawn(seed, 0) is empty", lambda: spawn(3, 0), expect=[])
    a.raises("spawn rejects n < 0", ValueError, lambda: spawn(3, -1),
             contains="non-negative")
    a.raises("spawn rejects skip < 0", ValueError, lambda: spawn(3, 1, skip=-1),
             contains="non-negative")
    four = [s.spawn_key for s in spawn(11, 4)]
    two_then_two = ([s.spawn_key for s in spawn(11, 2)]
                    + [s.spawn_key for s in spawn(11, 2, skip=2)])
    a.check("spawn(skip=) extends a pool instead of redrawing it "
            "(the --more bug this module exists for)",
            lambda: four == two_then_two)
    a.check("without skip, a second spawn REPEATS the first -- the trap itself",
            lambda: [s.spawn_key for s in spawn(11, 2)] == four[:2])
    a.check("children of one seed all differ",
            lambda: len({tuple(s.spawn_key) for s in spawn(1, 8)}), expect=8)
    # The streams have to differ in what they DRAW, not just in their key.
    draws = {float(np.random.default_rng(s).random()) for s in spawn(99, 6)}
    a.check("and they draw six different numbers", lambda: len(draws), expect=6)


def sec_constants(a: Audit) -> None:
    """tools/constants -- no constant travels without its error and provenance"""
    from tools.constants import (CONSTANTS_FILE, Constant, format_table, format_value,
                                 load, measured, override, require, save)

    a.begin("tools/constants", "no constant travels without its error and provenance")

    m = measured(1.02, 0.03, "pilot, 3 replicates", origin="pilot.json")
    o = override(1.0, "d")
    a.check("measured() is not an override", lambda: m.is_override, expect=False)
    a.check("override() is stamped as one", lambda: o.is_override, expect=True)
    a.check("override names the flag that supplied it", lambda: "--d" in o.source)
    a.check("se=None is allowed and is not the same as absent",
            lambda: measured(1.0, None, "x").se is None)
    a.check("format_value renders an error when there is one",
            lambda: "+/-" in format_value(m))
    a.check("format_value pads when there is not",
            lambda: "+/-" not in format_value(o))
    a.check("format_table marks an override NOT MEASURED",
            lambda: "NOT MEASURED" in format_table({"d": o, "omega1": m}))
    a.check("format_table keeps the canonical order",
            lambda: format_table({"cv": m, "d": m}).splitlines()[0]
            .strip().startswith("d"))
    a.check("format_table puts unknown names last",
            lambda: format_table({"zzz": m, "d": m}).splitlines()[-1]
            .strip().startswith("zzz"))

    a.check("require() returns a present constant",
            lambda: require({"d": m}, "d").value, expect=1.02)
    a.raises("a missing constant is a hard exit naming the flag", SystemExit,
             lambda: require({}, "omega1"), contains="--omega1")
    a.raises("...and the command that would measure it", SystemExit,
             lambda: require({}, "omega1"), contains="estimate_omega1.py")

    tmp = Path(tempfile.mkdtemp(prefix="const_"))
    try:
        a.check("load() of a directory with no constants.json is {} , not an error",
                lambda: load(tmp), expect={})
        save(tmp, {"d": m, "a1": o})
        a.check("save() writes the canonical filename",
                lambda: (tmp / CONSTANTS_FILE).exists())
        back = load(tmp)
        a.check("constants round-trip through JSON",
                lambda: back["d"].value == m.value and back["d"].se == m.se
                and back["a1"].is_override)
        a.check("save() sorts, so two studies diff readably",
                lambda: list(json.loads((tmp / CONSTANTS_FILE).read_text())),
                expect=["a1", "d"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sec_summary(a: Audit) -> None:
    """tools/summary -- what a replicate IS once its draws are thrown away"""
    from tools.summary import (LOG_MOMENT_DELTA, SUMMARY_FIELDS, replicate_summary,
                               summarize_scale)

    a.begin("tools/summary", "what a replicate IS once its draws are thrown away")

    rng = np.random.default_rng(0)
    draws = rng.lognormal(0.0, 0.3, size=20000)
    y, slog, cv, lm = summarize_scale(draws)
    a.close("y_bar is the sample mean", lambda: y, float(draws.mean()), 1e-12)
    a.close("cv is sd/mean", lambda: cv,
            float(draws.std(ddof=1) / draws.mean()), 1e-12)
    a.close("sigma_log is cv/sqrt(n)", lambda: slog, cv / math.sqrt(draws.size), 1e-12)
    a.check("log_moment is finite and positive", lambda: math.isfinite(lm) and lm > 0)
    a.check("four fields, in the declared order",
            lambda: SUMMARY_FIELDS,
            expect=("y_bar", "sigma_log", "cv", "log_moment"))
    a.check("delta is pinned at 1 to match eq. (720)'s B_bad",
            lambda: LOG_MOMENT_DELTA, expect=1.0)

    a.raises("a single draw is refused, not summarized", ValueError,
             lambda: summarize_scale([3.0]), contains="at least 2 draws")
    a.raises("...and the message names the fix", ValueError,
             lambda: summarize_scale([3.0]), contains="min_n")
    a.raises("all-zero draws are refused (log of nothing)", ValueError,
             lambda: summarize_scale([0.0, 0.0, 0.0]), contains="zero")
    # |S_k| = 0 really happens; the zeros are dropped, not fatal.
    mixed = summarize_scale([0.0, 1.0, 2.0, 3.0])
    a.check("zeros among positives are dropped from log_moment, not fatal",
            lambda: math.isfinite(mixed[3]))

    stats = {8: (1.0, 0.1, 0.2, 0.3), 16: (2.0, 0.2, 0.3, 0.4)}
    a.check("replicate_summary transposes to column-major",
            lambda: replicate_summary(stats, [8, 16])["y_bar"], expect=[1.0, 2.0])
    a.check("...ordered by the ladder, not by the mapping's key order",
            lambda: replicate_summary(stats, [16, 8])["y_bar"], expect=[2.0, 1.0])


def sec_loglog(a: Audit) -> None:
    """tools/loglog -- the four gamma-hat estimators and eq. (526)'s weights"""
    from tools.loglog import (closed_form_weights, compare_methods, gamma_all_points,
                              gamma_closed_form, gamma_drop_leading, gamma_mle,
                              gamma_two_point, ols_slope)

    a.begin("tools/loglog", "the four gamma-hat estimators and eq. (526)'s weights")

    a.check("ols_slope recovers a planted line",
            lambda: np.allclose(ols_slope([0, 1, 2, 3], [1, 3, 5, 7]), (2.0, 1.0)))
    a.raises("a slope needs two points", ValueError,
             lambda: ols_slope([1.0], [2.0]), contains="at least 2")

    # Noiseless power law: every estimator must return gamma exactly.
    scales = np.array([2 ** k for k in range(1, 8)])
    gamma_true, a0 = 0.5, 3.0
    y = a0 * scales.astype(float) ** gamma_true
    a.close("gamma_all_points is exact on a noiseless power law",
            lambda: gamma_all_points(scales, y), gamma_true, 1e-12)
    a.check("gamma_all_points does not care about input order",
            lambda: math.isclose(gamma_all_points(scales[::-1], y[::-1]),
                                 gamma_all_points(scales, y)))
    tp = gamma_two_point(scales, y)
    a.check("gamma_two_point gives one estimate per adjacent pair",
            lambda: len(tp), expect=len(scales) - 1)
    a.check("...each exact",
            lambda: all(abs(e["gamma_hat"] - gamma_true) < 1e-12 for e in tp))
    dl = gamma_drop_leading(scales, y)
    a.check("gamma_drop_leading gives one per m0 = 0..len-2",
            lambda: [e["m0"] for e in dl], expect=list(range(len(scales) - 1)))
    a.check("...and its last window still has 2 scales",
            lambda: len(dl[-1]["scales_used"]), expect=2)

    w = closed_form_weights(6)
    a.close("eq. (542) identity (a): sum w = 0", lambda: float(w.sum()), 0.0, 1e-12)
    a.close("eq. (542) identity (b): sum w*k = 1",
            lambda: float(np.dot(w, np.arange(1, 7))), 1.0, 1e-12)
    a.check("weights are shift-invariant in m0 (Lemma, elementary identities)",
            lambda: np.array_equal(closed_form_weights(6, 0), closed_form_weights(6, 4)))
    a.raises("m = 1 is refused -- it used to make nan weights and a nan gamma",
             ValueError, lambda: closed_form_weights(1), contains="m must be >= 2")

    a.close("gamma_closed_form agrees with generic OLS on a consecutive grid",
            lambda: gamma_closed_form(scales, y, 2.0, 0),
            float(gamma_all_points(scales, y)), 1e-10)
    a.close("...at a shifted m0 too",
            lambda: gamma_closed_form(scales * 4,
                                      a0 * (scales * 4.0) ** gamma_true, 2.0, 2),
            gamma_true, 1e-10)
    a.raises("a non-consecutive grid is rejected by name", ValueError,
             lambda: gamma_closed_form([2, 8, 32], [1.0, 2.0, 4.0], 2.0, 0),
             contains="consecutive grid")
    a.raises("...and so is the wrong m0 for the grid given", ValueError,
             lambda: gamma_closed_form(scales, y, 2.0, 5), contains="consecutive grid")

    n = np.full(len(scales), 10_000)
    mle = gamma_mle(scales, y, n)
    a.close("gamma_mle recovers the noiseless truth", lambda: mle["gamma_hat"],
            gamma_true, 1e-3)
    a.check("gamma_mle reports all four diagnostics",
            lambda: all(k in mle for k in
                        ("converged", "region_ok", "hessian_pd", "trustworthy")))
    a.check("trustworthy is the AND of the three",
            lambda: mle["trustworthy"] ==
            (mle["converged"] and mle["region_ok"] and mle["hessian_pd"]))

    cmp = compare_methods(scales, y, n, true_gamma=gamma_true)
    a.check("compare_methods bundles all four",
            lambda: sorted(cmp["methods"]),
            expect=["all_points", "drop_leading", "mle", "two_point"])
    a.check("compare_methods is JSON-serializable",
            lambda: isinstance(json.dumps(cmp), str))
    a.check("true_gamma appears only when given",
            lambda: "true_gamma" not in compare_methods(scales, y, n))

    # Noisy: the estimators should be close to each other, not identical.
    rng = np.random.default_rng(4)
    y_noisy = y * rng.lognormal(0, 0.01, size=len(y))
    spread = abs(gamma_all_points(scales, y_noisy) - gamma_true)
    a.check("under 1% noise all_points stays within 0.01 of truth",
            lambda: spread < 0.01, detail=f"|error| = {spread:.2e}")


def sec_correction(a: Audit) -> None:
    """tools/correction -- omega_1 from eq. (232), two independent ways"""
    from tools.correction import fit_correction, omega1_from_bias_decay

    a.begin("tools/correction", "omega_1 from eq. (232), two independent ways")

    scales = np.array([8, 16, 32, 64, 128, 256, 512], dtype=float)
    a0, gamma, a1, omega1 = 0.8, 0.5, -0.25, 1.0
    y = a0 * scales ** gamma * np.exp(a1 * scales ** -omega1)

    fit = fit_correction(scales, y)
    a.close("fit_correction recovers a planted omega1",
            lambda: fit["omega1"], omega1, 1e-3)
    a.close("...a1", lambda: fit["a1"], a1, 1e-3)
    a.close("...gamma", lambda: fit["gamma"], gamma, 1e-4)
    a.close("...a0", lambda: fit["a0"], a0, 1e-3)
    a.check("...and says it converged", lambda: fit["converged"])
    a.check("rel_rmse is ~0 on exact data", lambda: fit["rel_rmse"] < 1e-6)

    sig = np.full(len(scales), 1e-3)
    a.close("uniform sigma_log weighting changes nothing",
            lambda: fit_correction(scales, y, sigma_log=sig)["omega1"], omega1, 1e-3)
    a.check("a non-uniform weighting is accepted and still identifies omega1",
            lambda: abs(fit_correction(scales, y,
                                       sigma_log=np.linspace(1e-3, 1e-2, len(scales))
                                       )["omega1"] - omega1) < 0.05)

    a.raises("mismatched lengths are named", ValueError,
             lambda: fit_correction(scales, y[:-1]), contains="differ in length")
    a.raises("4 free parameters need >= 5 scales", ValueError,
             lambda: fit_correction(scales[:4], y[:4]), contains="at least 5")
    a.raises("non-positive y_bar is refused", ValueError,
             lambda: fit_correction(scales, np.r_[y[:-1], -1.0]),
             contains="strictly positive")
    a.raises("sigma_log must match in length", ValueError,
             lambda: fit_correction(scales, y, sigma_log=sig[:-1]), contains="match")
    a.raises("sigma_log must be positive", ValueError,
             lambda: fit_correction(scales, y, sigma_log=np.zeros(len(scales))),
             contains="strictly positive")

    # The bias-decay estimator, fed the sequence its docstring names.
    from tools.loglog import gamma_drop_leading
    wins = gamma_drop_leading(scales.astype(int), y)
    dec = omega1_from_bias_decay([w["scales_used"][0] for w in wins],
                                 [w["gamma_hat"] for w in wins])
    a.check("omega1_from_bias_decay converges", lambda: dec["converged"])
    a.check("...and lands in the right ballpark (a different functional, so not exact)",
            lambda: 0.3 < dec["omega1"] < 3.0,
            detail=f"omega1 = {dec['omega1']:.4f} from bias decay vs {omega1} planted")
    a.raises("3 free parameters need >= 4 estimates", ValueError,
             lambda: omega1_from_bias_decay([8, 16, 32], [0.5, 0.5, 0.5]),
             contains="at least 4")
    a.raises("mismatched lengths are named", ValueError,
             lambda: omega1_from_bias_decay([8, 16, 32, 64], [0.5]),
             contains="differ in length")

    # The documented parity trap: a rho = sqrt(2) grid over srw's EXACT means
    # returns a nonsense omega1. Checked so the docstring's warning cannot
    # quietly stop being true.
    from fractions import Fraction
    from math import comb

    def exact_mean(k):
        return float(Fraction(k * comb(k - 1, (k - 1) // 2), 1 << (k - 1)))

    mixed = [int(round(math.sqrt(2) ** k)) for k in range(6, 17)]
    mixed_fit = fit_correction(mixed, [exact_mean(k) for k in mixed])
    a.note("the parity trap documented in tools/correction.py still reproduces",
           f"a rho=sqrt(2) grid over srw's EXACT means (zero noise) fits "
           f"omega1 = {mixed_fit['omega1']:.4g} against a truth of 1, and reports "
           f"converged={mixed_fit['converged']} with rel_rmse "
           f"{mixed_fit['rel_rmse']:.2e}. Silent, as documented: the caller must "
           f"choose the grid, nothing here can catch it.")
    even = [2 ** k for k in range(3, 10)]
    even_fit = fit_correction(even, [exact_mean(k) for k in even])
    a.close("...while a powers-of-two grid on the same exact means gives omega1 = 1",
            lambda: even_fit["omega1"], 1.0, 0.05)


def sec_coverage(a: Audit) -> None:
    """tools/coverage -- do our stated error bars cover? (checkpoint 0.4)"""
    from tools.coverage import (combine_se, consistency_threshold, coverage_multi,
                                coverage_test, format_result, interval, rescore,
                                se_ratio, wilson_score_interval)

    a.begin("tools/coverage", "do our stated error bars cover? (checkpoint 0.4)")

    lo, hi = wilson_score_interval(95, 100)
    a.check("wilson_score_interval brackets the observed proportion",
            lambda: lo < 0.95 < hi)
    a.check("...and stays inside [0, 1] at the boundary, which is the whole point",
            lambda: 0.0 <= wilson_score_interval(100, 100)[1] <= 1.0)
    a.check("...at 0 successes too",
            lambda: wilson_score_interval(0, 50)[0] >= 0.0)
    a.check("a wider level gives a wider interval",
            lambda: (wilson_score_interval(90, 100, 0.99)[1]
                     - wilson_score_interval(90, 100, 0.99)[0])
            > (hi - lo))
    a.raises("trials must be positive", ValueError,
             lambda: wilson_score_interval(0, 0), contains="positive")
    a.raises("successes must be in range", ValueError,
             lambda: wilson_score_interval(5, 3), contains="successes")
    a.raises("level must be in (0, 1)", ValueError,
             lambda: wilson_score_interval(1, 3, 1.0), contains="level")

    a.close("interval() with dof=None uses the normal quantile 1.960",
            lambda: (interval(0.0, 1.0)[1] - interval(0.0, 1.0)[0]) / 2, 1.959964, 1e-5)
    a.close("...and with dof=4 uses Student's 2.776 -- 42% wider",
            lambda: (interval(0.0, 1.0, dof=4)[1] - interval(0.0, 1.0, dof=4)[0]) / 2,
            2.776445, 1e-5)
    a.check("se = 0 gives a degenerate interval, not an error",
            lambda: interval(1.0, 0.0), expect=(1.0, 1.0))
    a.raises("a negative se is refused", ValueError,
             lambda: interval(0.0, -1.0), contains="non-negative")
    a.raises("dof < 1 is refused", ValueError,
             lambda: interval(0.0, 1.0, dof=0), contains="dof")

    # A textbook experiment whose interval is known to be right: the mean of
    # R normal draws with the t quantile covers exactly 95%. If coverage_test
    # says otherwise, the harness itself is broken.
    def gaussian_experiment(rng):
        x = rng.normal(0.0, 1.0, size=5)
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(5))

    r_t = coverage_test(gaussian_experiment, 0.0, trials=800, dof=4, seed=1)
    a.check("coverage_test measures 95% for a textbook-correct t interval",
            lambda: r_t["calibrated"],
            detail=f"measured {r_t['coverage']:.3f}, CI {r_t['coverage_ci']}")
    r_n = coverage_test(gaussian_experiment, 0.0, trials=800, dof=None, seed=1)
    a.check("...and catches the normal quantile undercovering at R = 5",
            lambda: r_n["coverage"] < r_t["coverage"],
            detail=f"normal {r_n['coverage']:.3f} vs t {r_t['coverage']:.3f} "
                   f"(theory: 0.878 vs 0.95)")
    a.close("...by about the amount theory says (P(|t_4| < 1.96) = 0.8784)",
            lambda: r_n["coverage"], 0.8784, 0.05)
    a.check("bias is reported alongside, since undercoverage has two causes",
            lambda: abs(r_t["bias"]) < 0.1)
    a.close("se_ratio is ~1 when the stated se is the right size",
            lambda: se_ratio(r_t), 1.0, 0.1)
    a.check("format_result spells out the verdict",
            lambda: "CALIBRATED" in format_result(r_t) or
                    "COVERS" in format_result(r_t).upper())
    a.check("the per-trial values are kept so rescore() is free",
            lambda: len(r_t["values"]), expect=r_t["trials"])
    a.raises("trials must be >= 1", ValueError,
             lambda: coverage_test(gaussian_experiment, 0.0, trials=0), contains="trials")

    rs = rescore(r_n, dof=4)
    a.check("rescore(dof=4) reproduces the t run exactly, on identical draws",
            lambda: rs["hits"], expect=r_t["hits"])
    a.check("rescore(level=) moves the level",
            lambda: rescore(r_t, level=0.6827)["nominal"], expect=0.6827)
    a.check("omitting dof keeps the original (the Ellipsis sentinel)",
            lambda: rescore(r_t, level=0.99)["dof"], expect=4)
    a.check("passing dof=None explicitly forces the normal quantile",
            lambda: rescore(r_t, dof=None)["dof"] is None)
    a.raises("a result without stored values cannot be rescored", KeyError,
             lambda: rescore({"nominal": 0.95}), contains="rescore")

    def multi_experiment(rng):
        x = rng.normal(0.0, 1.0, size=5)
        se = float(x.std(ddof=1) / math.sqrt(5))
        return {"mu": (float(x.mean()), se), "mu2": (float(x.mean()), se)}

    multi = coverage_multi(multi_experiment, {"mu": 0.0, "mu2": 0.0},
                           trials=200, dofs=(None, 4), seed=2)
    a.check("coverage_multi scores every quantity at every quantile from one pass",
            lambda: len(multi), expect=4)
    a.check("...on the SAME trials, so two readings of one fit are comparable",
            lambda: multi["mu [q=normal]"]["hits"] == multi["mu2 [q=normal]"]["hits"])
    a.raises("a quantity with no truth is an error, not a silent skip", KeyError,
             lambda: coverage_multi(multi_experiment, {"mu": 0.0}, trials=5, seed=2),
             contains="truth")

    se, dof = combine_se([(3.0, None), (4.0, None)])
    a.close("combine_se adds in quadrature", lambda: se, 5.0, 1e-12)
    a.check("...with infinite dof when both components are known",
            lambda: math.isinf(dof))
    se2, dof2 = combine_se([(1.0, 4), (1.0, 4)])
    a.close("Welch-Satterthwaite on two equal 4-dof components gives 8",
            lambda: dof2, 8.0, 1e-9)
    se3, dof3 = combine_se([(0.0, None), (0.0, None)])
    a.check("all-zero components give (0, inf) rather than a divide by zero",
            lambda: se3 == 0.0 and math.isinf(dof3))
    a.check("a None se is skipped",
            lambda: combine_se([(None, None), (3.0, None)])[0], expect=3.0)
    a.raises("no components at all is an error", ValueError,
             lambda: combine_se([]), contains="at least one")
    a.raises("a negative se is refused", ValueError,
             lambda: combine_se([(-1.0, None)]), contains="non-negative")

    a.close("consistency_threshold(inf) is the normal 1.960",
            lambda: consistency_threshold(float("inf")), 1.959964, 1e-5)
    a.close("consistency_threshold(4) is 2.776 -- the honest cut-off at R = 5",
            lambda: consistency_threshold(4), 2.776445, 1e-5)


def sec_wilson(a: Audit) -> None:
    """tools/wilson -- article eq. (720), the four-term bound -- gamma only"""
    from tools.wilson import (C0, bad_event_bias, finite_size_bias, format_interval,
                              good_event_bias, moment_bounds, sigma_se,
                              sigma_se_per_scale, wilson_interval)

    a.begin("tools/wilson", "article eq. (720), the four-term bound -- gamma only")

    a.close("C0 = 4 log 2 - 2", lambda: C0, 4 * math.log(2) - 2, 1e-15)
    n, m, rho, s2 = 100_000.0, 6, 2.0, 0.5708
    a.close("sigma_se is sqrt(12 sigma_inf2/(n m^3)) / log rho",
            lambda: sigma_se(n, m, rho, s2),
            math.sqrt(12 * s2 / (n * m ** 3)) / math.log(rho), 1e-15)
    a.close("for_gamma=False leaves it on the beta scale",
            lambda: sigma_se(n, m, rho, s2, for_gamma=False),
            math.sqrt(12 * s2 / (n * m ** 3)), 1e-15)
    a.check("sigma_se falls like 1/sqrt(n)",
            lambda: math.isclose(sigma_se(4 * n, m, rho, s2),
                                 sigma_se(n, m, rho, s2) / 2, rel_tol=1e-12))
    a.raises("bad inputs are named", ValueError,
             lambda: sigma_se(0, m, rho, s2), contains="bad input")
    a.raises("rho <= 1 is refused", ValueError,
             lambda: sigma_se(n, m, 1.0, s2), contains="bad input")

    # NOT equal to sigma_se at finite m, and the difference is exactly the
    # gap between the weights' true norm 12/(m(m^2-1)) and eq. (720)'s 12/m^3.
    a.close("sigma_se_per_scale is the exact sqrt(sum w_j^2 cv2_j/n_j)",
            lambda: sigma_se_per_scale([n] * m, m, rho, [s2] * m),
            math.sqrt(12 * s2 / (n * m * (m ** 2 - 1))) / math.log(rho), 1e-15)
    ratio = sigma_se_per_scale([n] * m, m, rho, [s2] * m) / sigma_se(n, m, rho, s2)
    a.close("...which exceeds eq. (720)'s sigma_se by exactly sqrt(m^2/(m^2-1))",
            lambda: ratio, math.sqrt(m ** 2 / (m ** 2 - 1)), 1e-12)
    a.check("...and the docstring says so, rather than claiming they are equal",
            lambda: "does NOT equal" in sigma_se_per_scale.__doc__
            and "verbatim" in sigma_se_per_scale.__doc__,
            detail=f"at m = {m} the article's sigma_se is "
                   f"{100 * (ratio - 1):.1f}% below the exact variance; the "
                   f"formula stays as eq. (720) writes it (Igor, 2026-09-04) "
                   f"and the gap is documented instead")
    a.check("...and is smaller when the big scales get more samples",
            lambda: sigma_se_per_scale([n, n, n, n, n, 10 * n], m, rho, [s2] * m)
            < sigma_se(n, m, rho, s2))
    a.raises("a length mismatch is named", ValueError,
             lambda: sigma_se_per_scale([n] * 5, m, rho, [s2] * m), contains="expected 6")

    b_fs = finite_size_bias(m, 2, rho, -0.25, 1.0)
    a.check("B_fs is positive", lambda: b_fs > 0)
    a.check("B_fs decays in m0 like rho^(-omega1 m0)",
            lambda: math.isclose(finite_size_bias(m, 3, rho, -0.25, 1.0) / b_fs,
                                 0.5, rel_tol=1e-12))
    a.check("B_fs is linear in |a1|",
            lambda: math.isclose(finite_size_bias(m, 2, rho, -0.5, 1.0), 2 * b_fs,
                                 rel_tol=1e-12))
    a.check("the omega2 piece is dropped when phi_plus is 0",
            lambda: finite_size_bias(m, 2, rho, -0.25, 1.0, omega2=3.0) == b_fs)
    a.check("...and included when it is not",
            lambda: finite_size_bias(m, 2, rho, -0.25, 1.0, omega2=3.0,
                                     phi_plus=0.1) > b_fs)
    a.raises("omega2 <= omega1 is refused", ValueError,
             lambda: finite_size_bias(m, 2, rho, -0.25, 1.0, omega2=0.5, phi_plus=0.1),
             contains="omega2 must exceed")
    a.raises("m0 < 0 is refused", ValueError,
             lambda: finite_size_bias(m, -1, rho, -0.25, 1.0), contains="m0")

    a.close("B_good = 6(c0+2) sigma_max2 / (n(m+1)) / log rho",
            lambda: good_event_bias(n, m, 0.6, rho),
            6 * (C0 + 2) * 0.6 / (n * (m + 1)) / math.log(rho), 1e-15)
    a.check("B_bad needs delta > 0", lambda: bad_event_bias(n, m, 0.6, 2.0, 1.0, rho) > 0)
    a.raises("delta <= 0 is refused", ValueError,
             lambda: bad_event_bias(n, m, 0.6, 2.0, 0.0, rho), contains="bad input")

    rng = np.random.default_rng(0)
    samples = {k: rng.lognormal(0, 0.3, size=20000) * k ** 0.5
               for k in (8, 16, 32, 64)}
    mb = moment_bounds(samples)
    a.check("moment_bounds reads sigma_inf2 off the LARGEST scale",
            lambda: math.isclose(mb["sigma_inf2"], mb["per_scale"][-1]["sigma2"]))
    a.check("...sigma_max2 over all of them",
            lambda: mb["sigma_max2"] >= mb["sigma_inf2"])
    a.close("...and sigma^2 matches the planted lognormal variance",
            lambda: mb["sigma_inf2"], math.expm1(0.09), 0.02)
    a.check("per-scale detail is returned, since Assumption 6 is worth seeing",
            lambda: len(mb["per_scale"]), expect=4)
    a.raises("no samples at all is an error", ValueError,
             lambda: moment_bounds({}), contains="no samples")
    a.raises("a non-positive mean is an error", ValueError,
             lambda: moment_bounds({8: np.array([-1.0, 1.0])}), contains="non-positive")

    incomplete = wilson_interval(0.5, n, m, 2, rho, sigma_inf2=s2, sigma_max2=0.6,
                                 a1=-0.25, omega1=1.0)
    a.check("a bound missing B_bad is flagged incomplete",
            lambda: incomplete["complete"], expect=False)
    a.check("...and says which terms are missing",
            lambda: len(incomplete["missing_terms"]), expect=2)
    a.check("format_interval leads with the incompleteness warning",
            lambda: format_interval(incomplete).startswith("INCOMPLETE"))
    full = wilson_interval(0.5, n, m, 2, rho, sigma_inf2=s2, sigma_max2=0.6,
                           a1=-0.25, omega1=1.0, Lambda=2.0, delta=1.0,
                           omega2=3.0, phi_plus=0.1)
    a.check("with every constant supplied the bound is complete",
            lambda: full["complete"])
    a.close("half_width is the sum of the four terms",
            lambda: full["half_width"],
            full["B_fs"] + full["B_good"] + full["B_bad"] + full["se_term"], 1e-15)
    a.check("the interval is gamma_hat +/- half_width",
            lambda: math.isclose(full["interval"][1] - full["interval"][0],
                                 2 * full["half_width"]))
    a.check("`dominant` names the largest term",
            lambda: full["dominant"],
            expect=max([("B_fs", full["B_fs"]), ("B_good", full["B_good"]),
                        ("B_bad", full["B_bad"]), ("se_term", full["se_term"])],
                       key=lambda kv: kv[1])[0])
    a.check("se_override replaces the fourth term (the non-uniform-n case)",
            lambda: math.isclose(
                wilson_interval(0.5, n, m, 2, rho, sigma_inf2=s2, sigma_max2=0.6,
                                a1=-0.25, omega1=1.0, se_override=1e-9)["sigma_se"],
                1e-9))
    a.raises("level outside (0, 1) is refused", ValueError,
             lambda: wilson_interval(0.5, n, m, 2, rho, sigma_inf2=s2, sigma_max2=0.6,
                                     a1=-0.25, omega1=1.0, level=1.5),
             contains="level")


def sec_allocation(a: Audit) -> None:
    """tools/allocation -- prop:opt (eq. 945-946), lem:budget, and the tuned constant"""
    from tools.allocation import (allocation_constants, feasible, ladder, n_for_budget,
                                  neyman_allocation, optimal_allocation, predict_error,
                                  rate_exponent, rate_exponent_se, snr_allocation,
                                  total_cost, tuned_allocation)

    a.begin("tools/allocation",
            "prop:opt (eq. 945-946), lem:budget, and the tuned constant")

    d, omega1, rho, m = 1.0, 1.0, 2.0, 6
    brute = sum(10 * (rho ** k) ** d for k in range(3, 3 + m))
    a.close("total_cost matches the brute-force sum it closes",
            lambda: total_cost(10, 2, m, rho, d), brute, 1e-9)
    a.check("total_cost accepts a continuous m0 (the Lemma's own claim)",
            lambda: total_cost(10, 2.5, m, rho, d) > total_cost(10, 2, m, rho, d))
    a.check("feasible() is true at the optimum's exponents",
            lambda: feasible(2 * omega1 / (d + 2 * omega1), 1 / (d + 2 * omega1), d))
    a.check("...and false past the budget constraint",
            lambda: feasible(1.0, 1.0, 1.0), expect=False)

    c = allocation_constants(d, omega1, rho, m, a1=-0.25, cv=0.774)
    a.check("allocation_constants returns Cb, Cs, G, kappa and the offset",
            lambda: all(c[k] is not None for k in ("Cb", "Cs", "G", "kappa", "offset")))
    a.close("G is rho^d (rho^(dm)-1)/(rho^d-1)",
            lambda: c["G"], rho ** d * (rho ** (d * m) - 1) / (rho ** d - 1), 1e-12)
    a.close("||w||^2 = 12/(m(m^2-1)), eq. (526)",
            lambda: c["w_norm_sq"], 12 / (m * (m ** 2 - 1)), 1e-15)
    a.check("Cb scales linearly with |a1|",
            lambda: math.isclose(
                allocation_constants(d, omega1, rho, m, a1=-0.5, cv=0.774)["Cb"],
                2 * c["Cb"], rel_tol=1e-12))
    zero = allocation_constants(d, omega1, rho, m, a1=0.0, cv=0.774)
    a.check("a1 = 0 means no bias to trade, so kappa/offset are None, not a crash",
            lambda: zero["kappa"] is None and zero["offset"] is None)
    a.raises("m < 2 has no weights", ValueError,
             lambda: allocation_constants(d, omega1, rho, 1, -0.25, 0.774),
             contains="m must be >= 2")
    a.raises("rho <= 1 is refused", ValueError,
             lambda: allocation_constants(d, omega1, 1.0, m, -0.25, 0.774),
             contains="rho")
    a.raises("d <= 0 is refused HERE (unlike the design-input rules)", ValueError,
             lambda: allocation_constants(0.0, omega1, rho, m, -0.25, 0.774),
             contains="d and omega1")
    a.raises("cv <= 0 is refused", ValueError,
             lambda: allocation_constants(d, omega1, rho, m, -0.25, 0.0), contains="cv")

    err = predict_error(1e6, 4, d, omega1, rho, m, -0.25, 0.774)
    a.close("predict_error's rmse is hypot(bias, sd)",
            lambda: err["rmse"], math.hypot(err["bias"], err["sd"]), 1e-15)
    a.check("bias falls with m0 and sd falls with n",
            lambda: predict_error(1e6, 5, d, omega1, rho, m, -0.25,
                                  0.774)["bias"] < err["bias"]
            and predict_error(4e6, 4, d, omega1, rho, m, -0.25, 0.774)["sd"] < err["sd"])

    opt = optimal_allocation(B=1e9, d=d, omega1=omega1, rho=rho, m=m)
    a.close("theta1 = 2 omega1/(d + 2 omega1)", lambda: opt["theta1"], 2 / 3, 1e-15)
    a.close("theta2 = 1/(d + 2 omega1)", lambda: opt["theta2"], 1 / 3, 1e-15)
    a.check("feasibility is ACTIVE at the optimum: theta1 + d theta2 == 1",
            lambda: math.isclose(opt["theta1"] + d * opt["theta2"], 1.0, rel_tol=1e-12))
    a.check("theta_feasible is a self-check that should always pass",
            lambda: opt["theta_feasible"])
    a.check("flooring n and m0 keeps the cost within budget (the module's own claim)",
            lambda: opt["cost"] <= 1e9,
            detail=f"cost {opt['cost']:.4g} vs B 1e9")
    tiny = optimal_allocation(B=1.0, d=d, omega1=omega1, rho=rho, m=m)
    a.check("when n floors below 1 the integer allocation is None, not a forced n=1",
            lambda: tiny["integer_feasible"] is False and tiny["n"] is None)
    a.check("...while the continuous quantities are still returned",
            lambda: tiny["n_exact"] > 0)
    for bad, why in ((dict(d=0.0), "d must be > 0"),
                     (dict(omega1=0.0), "omega1 must be > 0"),
                     (dict(rho=1.0), "rho must be > 1"), (dict(m=1), "m must be >= 2"),
                     (dict(B=0.5), "B must be >= 1")):
        kw = {"B": 1e9, "d": d, "omega1": omega1, "rho": rho, "m": m, **bad}
        a.raises(f"optimal_allocation refuses {list(bad)[0]} = {list(bad.values())[0]}",
                 ValueError, lambda kw=kw: optimal_allocation(**kw), contains=why)

    tun = tuned_allocation(1e9, d, omega1, rho, m, a1=-0.25, cv=0.774)
    a.check("tuned_allocation shifts m0 by the offset prop:opt drops",
            lambda: tun["m0"] != opt["m0"] or abs(tun["offset_vs_prop_opt"]) < 0.5,
            detail=f"tuned m0={tun['m0']} vs prop:opt m0={opt['m0']}, "
                   f"offset {tun['offset_vs_prop_opt']:+.3f}")
    a.check("...and the tuned cost still respects the budget",
            lambda: tun["cost"] <= 1e9)
    a.check("tuned_allocation ROUNDS m0 (flooring measurably missed the argmin)",
            lambda: tun["m0"] == max(0, int(round(tun["m0_exact"]))))
    a.check("its predicted error is attached",
            lambda: all(k in tun for k in ("bias", "sd", "rmse")))
    a.raises("a1 = 0 has no bias/variance balance to solve", ValueError,
             lambda: tuned_allocation(1e9, d, omega1, rho, m, a1=0.0, cv=0.774),
             contains="a1 == 0")
    a.raises("B < 1 is refused", ValueError,
             lambda: tuned_allocation(0.5, d, omega1, rho, m, a1=-0.25, cv=0.774),
             contains="B must be >= 1")
    a.check("at a budget too small for one sample, integer_feasible is False",
            lambda: tuned_allocation(1.0, d, omega1, rho, m, a1=-0.25,
                                     cv=0.774)["integer_feasible"], expect=False)

    scales = [8, 16, 32, 64, 128, 256]
    ney = neyman_allocation(scales, 1e9, d=1.0)
    a.check("neyman puts MORE samples on the cheap small scales (n ~ i^(-d/2))",
            lambda: ney["n"][0] > ney["n"][-1])
    a.check("...and stays within budget", lambda: ney["exhausted"] <= 1.0,
            detail=f"exhausted {ney['exhausted']:.3f}")
    snr = snr_allocation(scales, 1e9, d=1.0, omega1=1.0)
    a.check("snr at omega1=1 puts more on the LARGE scales (n ~ i^(2 omega1))",
            lambda: snr["n"][-1] > snr["n"][0])
    flat = snr_allocation(scales, 1e6, d=0.0, omega1=0.0)
    a.check("snr at the UNINFORMED omega1=0, d=0 is flat and counts samples",
            lambda: len(set(flat["n"])) == 1)
    clamp = neyman_allocation(scales, 10.0, d=1.0, min_n=2)
    a.check("min_n clamping is reported through `exhausted` > 1 rather than hidden",
            lambda: clamp["exhausted"] > 1.0,
            detail=f"exhausted {clamp['exhausted']:.1f}x")
    a.check("a per-scale sigma is accepted",
            lambda: len(snr_allocation(scales, 1e9, 1.0, 1.0,
                                       sigma=[1.0] * len(scales))["n"]), expect=6)
    for fn in (neyman_allocation, snr_allocation):
        kw = {"omega1": 1.0} if fn is snr_allocation else {}
        nm = fn.__name__
        a.raises(f"{nm} refuses empty scales", ValueError,
                 lambda fn=fn, kw=kw: fn([], 1e6, 1.0, **kw), contains="non-empty")
        a.raises(f"{nm} refuses a non-positive budget", ValueError,
                 lambda fn=fn, kw=kw: fn(scales, 0.0, 1.0, **kw), contains="budget")
        a.raises(f"{nm} refuses d < 0", ValueError,
                 lambda fn=fn, kw=kw: fn(scales, 1e6, -1.0, **kw),
                 contains="d must be >= 0")
        a.raises(f"{nm} refuses min_n < 1", ValueError,
                 lambda fn=fn, kw=kw: fn(scales, 1e6, 1.0, min_n=0, **kw),
                 contains="min_n")
        a.raises(f"{nm} refuses a mismatched sigma", ValueError,
                 lambda fn=fn, kw=kw: fn(scales, 1e6, 1.0, sigma=[1.0], **kw),
                 contains="match")
    a.raises("snr refuses omega1 < 0", ValueError,
             lambda: snr_allocation(scales, 1e6, 1.0, -1.0),
             contains="omega1 must be >= 0")

    a.check("ladder is rho^k for k = m0+1..m0+m",
            lambda: ladder(2, 4, 2.0), expect=[8, 16, 32, 64])
    a.raises("a rounding collision is refused, not silently deduplicated", ValueError,
             lambda: ladder(0, 6, 1.5), contains="repeated")
    a.raises("ladder refuses m < 2", ValueError, lambda: ladder(0, 1, 2.0),
             contains="m must be >= 2")
    a.check("n_for_budget is the largest uniform n that fits",
            lambda: total_cost(n_for_budget(1e9, 3, m, rho, d), 3, m, rho, d) <= 1e9)
    a.check("...and one more would not",
            lambda: total_cost(n_for_budget(1e9, 3, m, rho, d) + 1, 3, m, rho, d) > 1e9)

    budgets = [1e6, 1e7, 1e8, 1e9]
    rmses = [B ** -0.3333 for B in budgets]
    a.close("rate_exponent recovers a planted decay slope",
            lambda: rate_exponent(budgets, rmses), -0.3333, 1e-9)
    a.check("rate_exponent_se shrinks with more replicates",
            lambda: rate_exponent_se(budgets, 40) < rate_exponent_se(budgets, 3))
    a.check("...and is nan when there is no budget spread",
            lambda: math.isnan(rate_exponent_se([1e6, 1e6], 3)))


def sec_cost_model(a: Audit) -> None:
    """tools/cost_model -- the cost exponent d, and the two probes that measure it"""
    from tools.cost_model import (AGGREGATORS, COST_ESTIMATORS, DEFAULT_AGGREGATOR,
                                  PROBE_MIN_SCALES, aggregate, climb_to_target,
                                  compare_cost_models, declared_exponent,
                                  estimate_cost_affine, estimate_cost_exponent,
                                  fit_cost_probe, format_cost_comparison, median_ci,
                                  time_at_scale, time_over_scales)
    from tools.models import get_model

    a.begin("tools/cost_model", "the cost exponent d, and the two probes that measure it")

    scales = [2 ** k for k in range(2, 10)]
    pure = [3e-6 * k ** 1.0 for k in scales]
    a.close("estimate_cost_exponent recovers d from a pure power law",
            lambda: estimate_cost_exponent(scales, pure), 1.0, 1e-9)
    a.raises("an unknown method is named", ValueError,
             lambda: estimate_cost_exponent(scales, pure, method="nope"),
             contains="unknown method")
    a.check("COST_ESTIMATORS is the registry the docstring advertises",
            lambda: list(COST_ESTIMATORS), expect=["ols"])

    # Affine: the fit that rescued the small scales. Plant a real overhead.
    affine_truth = [2.2e-5 + 3e-8 * k ** 1.0 for k in scales]
    a.close("a pure power law fitted to a + b*i^d is badly biased -- the "
            "documented defect",
            lambda: estimate_cost_exponent(scales, affine_truth), 0.5, 0.45,
            detail="anything well below 1 reproduces the bias")
    aff = estimate_cost_affine(scales, affine_truth)
    a.close("estimate_cost_affine recovers d", lambda: aff["d"], 1.0, 0.02)
    a.close("...and the overhead a", lambda: aff["a"], 2.2e-5, 2e-6)
    a.check("...with a standard error when there are dof for one",
            lambda: aff["d_se"] is None or aff["d_se"] >= 0)
    a.raises("3 free parameters need >= 4 scales", ValueError,
             lambda: estimate_cost_affine(scales[:3], affine_truth[:3]),
             contains="at least 4")
    a.raises("non-positive timings are refused", ValueError,
             lambda: estimate_cost_affine(scales, [0.0] * len(scales)),
             contains="strictly positive")

    times = [1.0, 1.1, 1.2, 5.0]
    a.check("every advertised aggregator exists and runs",
            lambda: sorted(AGGREGATORS),
            expect=["iqmean", "mean", "median", "min", "q95"])
    a.check("min is the smallest, q95 nearly the largest",
            lambda: aggregate(times, "min") == 1.0 and aggregate(times, "q95") > 1.2)
    a.check("median resists the one-sided jitter a mean absorbs",
            lambda: aggregate(times, "median") < aggregate(times, "mean"))
    a.check("iqmean trims both tails",
            lambda: 1.0 < aggregate(times, "iqmean") < 2.0)
    a.check("the default is median (chosen for its CI, not for the estimate)",
            lambda: DEFAULT_AGGREGATOR, expect="median")
    a.raises("an unknown aggregator is named", ValueError,
             lambda: aggregate(times, "nope"), contains="unknown aggregator")
    a.raises("an empty timing sample is refused", ValueError,
             lambda: aggregate([]), contains="empty")

    lo, hi = median_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a.check("median_ci brackets the median", lambda: lo <= 5.5 <= hi)
    a.check("...falls back to (min, max) when N is small",
            lambda: median_ci([1.0, 2.0]), expect=(1.0, 2.0))
    a.check("...and to (x, x) for a single observation",
            lambda: median_ci([3.0]), expect=(3.0, 3.0))
    a.check("...and to (nan, nan) for none",
            lambda: all(math.isnan(x) for x in median_ci([])))

    srw_spec, syn_spec = get_model("srw"), get_model("synthetic")
    a.close("declared_exponent reads srw's cost_hint as exactly 1",
            lambda: declared_exponent(scales, srw_spec.cost_hint, {}), 1.0, 1e-12)
    a.close("...and the synthetic model's constant cost as exactly 0",
            lambda: declared_exponent(scales, syn_spec.cost_hint, {}), 0.0, 1e-15)
    a.raises("a non-positive cost_hint is refused", ValueError,
             lambda: declared_exponent(scales, lambda i, p: -1.0, {}),
             contains="positive")

    agree = compare_cost_models(scales, affine_truth, srw_spec.cost_hint, {})
    a.check("compare_cost_models agrees when the clock matches the declaration",
            lambda: agree["agree"], detail=f"z={agree['z']}, rel={agree['rel_gap']:.4f}")
    a.check("...and reports which fit it used",
            lambda: "affine" in agree["measured_via"])
    wrong = compare_cost_models(scales, [3e-6 * k ** 2.0 for k in scales],
                                srw_spec.cost_hint, {})
    a.check("...and disagrees when the clock says d = 2 against a declared 1",
            lambda: wrong["agree"], expect=False)
    a.check("format_cost_comparison leads with the warning when they disagree",
            lambda: format_cost_comparison(wrong).startswith("WARNING"))
    a.check("...and is quiet when they do not",
            lambda: not format_cost_comparison(agree).startswith("WARNING"))
    few = compare_cost_models(scales[:3], affine_truth[:3], srw_spec.cost_hint, {})
    a.check("with too few scales it falls back to the pure power law and says so",
            lambda: "pure power law" in few["measured_via"])
    a.check("...and says it could not compare, rather than reporting agreement",
            lambda: few["z"] is None and few["comparable"] is False
            and few["agree"] is None)
    a.check("...which the formatted block leads with",
            lambda: format_cost_comparison(few).startswith("NOT COMPARED"))

    rng = np.random.default_rng(0)
    agg, raw = time_at_scale(srw_spec, 2048, {"q": 0.5}, rng, repeats=3)
    a.check("time_at_scale returns the aggregate and every raw timing",
            lambda: len(raw) == 3 and agg > 0)
    ladder_k = [4096, 8192, 16384, 32768, 65536, 131072]
    probe = time_over_scales(srw_spec, ladder_k, {"q": 0.5}, rng, repeats=5)
    a.check("time_over_scales times the ladder it is given",
            lambda: probe["scales"], expect=ladder_k)
    a.check("...keeping the raw repeats under elapsed_all",
            lambda: len(probe["elapsed_all"]["4096"]), expect=5)
    fit_cost_probe(probe, srw_spec.cost_hint, {})
    a.close("fit_cost_probe recovers srw's d = 1 from real wall clock",
            lambda: probe["affine"]["d"], 1.0, 0.3,
            detail="wall clock on a shared machine; a wide tolerance on purpose")
    a.check("...reports the overhead share as a diagnostic",
            lambda: probe["overhead_share"] is None or probe["overhead_share"] >= 0)
    a.check("...and the declaration, for cross-checking only",
            lambda: math.isclose(probe["declared_d"], 1.0))
    climb = climb_to_target(srw_spec, {"q": 0.5}, np.random.default_rng(0),
                            start=256, repeats=3, target_seconds=1e-3,
                            time_budget=10.0)
    a.check("climb_to_target never stops before the affine fit has enough rungs",
            lambda: len(climb["scales"]) >= PROBE_MIN_SCALES)
    a.check("...doubles the scale each rung",
            lambda: climb["scales"][1] == 2 * climb["scales"][0])
    a.check("...and says whether it reached the target",
            lambda: isinstance(climb["reached_target"], bool),
            detail=f"reached_target={climb['reached_target']} at "
                   f"k={climb['scales'][-1]}")
    short = climb_to_target(srw_spec, {"q": 0.5}, np.random.default_rng(0),
                            start=8, repeats=2, target_seconds=1e9,
                            max_doublings=5)
    a.check("max_doublings caps the climb even when the target is unreachable",
            lambda: len(short["scales"]), expect=5)
    starved = climb_to_target(srw_spec, {"q": 0.5}, np.random.default_rng(0),
                              start=8, repeats=2, target_seconds=1e-9,
                              time_budget=0.0)
    a.check("...but the PROBE_MIN_SCALES floor outranks BOTH stopping rules, "
            "so the affine fit always has enough rungs",
            lambda: len(starved["scales"]) >= PROBE_MIN_SCALES,
            detail=f"a zero time budget still returned "
                   f"{len(starved['scales'])} rungs")
    a.raises("a max_doublings below that floor is a contradiction, and refused",
             ValueError,
             lambda: climb_to_target(srw_spec, {"q": 0.5},
                                     np.random.default_rng(0), start=8,
                                     max_doublings=2),
             contains="PROBE_MIN_SCALES")


def sec_artifacts(a: Audit) -> None:
    """tools/artifacts -- one place that decides what every file is called"""
    from tools.artifacts import (ARTIFACTS, LEGACY, RECIPES, artifact_path, classify,
                                 default_out_dir, find_artifacts, load_recipe, migrate,
                                 read_artifact, recipe_kind, recipe_name, recipes_dir,
                                 write_artifact)

    a.begin("tools/artifacts", "one place that decides what every file is called")

    tmp = Path(tempfile.mkdtemp(prefix="artifacts_"))
    try:
        a.check("every artifact kind maps to a distinct filename",
                lambda: len(set(ARTIFACTS.values())), expect=len(ARTIFACTS))
        a.check("artifact_path puts it in the run directory",
                lambda: artifact_path(tmp, "omega1").name, expect="omega1.json")
        a.raises("an unknown kind is refused with the list of known ones", ValueError,
                 lambda: artifact_path(tmp, "nope"), contains="unknown artifact kind")

        p = write_artifact(tmp, "cost_probe", {"d_hat": 1.0},
                           produced_by="calibration/exercise_all.py", recipe="r.json")
        body = json.loads(p.read_text())
        a.check("write_artifact stamps provenance INSIDE the file",
                lambda: body["produced_by"], expect="calibration/exercise_all.py")
        a.check("...plus the kind, the recipe and a timestamp",
                lambda: all(k in body for k in ("artifact", "created", "recipe")))
        a.check("read_artifact reads it back",
                lambda: read_artifact(tmp, "cost_probe")["d_hat"], expect=1.0)
        a.check("a missing optional artifact is None, not an error",
                lambda: read_artifact(tmp, "plan", required=False) is None)
        a.raises("a missing required one names what it looked for", FileNotFoundError,
                 lambda: read_artifact(tmp, "plan"), contains="plan.json")

        # Legacy rescue: the same file under its old name, with a warning.
        legacy_dir = tmp / "legacy"
        legacy_dir.mkdir()
        (legacy_dir / "result.json").write_text(json.dumps({"affine": {"d": 1.0}}))
        (legacy_dir / "results.json").write_text(json.dumps({"all_points": {}}))
        (legacy_dir / "metadata.json").write_text(json.dumps({"scales": [1], "n": [2]}))
        a.check("classify tells three producers' result.json apart by schema",
                lambda: classify(legacy_dir / "result.json"), expect="cost_probe")
        a.check("...and recognises a current name straight off",
                lambda: classify(tmp / "cost_probe.json"), expect="cost_probe")
        a.check("...and gives up rather than guessing on an unknown schema",
                lambda: classify(legacy_dir / "nothing.json") is None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = read_artifact(legacy_dir, "cost_probe")
        a.check("a legacy name still loads",
                lambda: got["affine"]["d"], expect=1.0)
        a.check("...but warns, so a half-migrated data/ cannot look healthy",
                lambda: any(issubclass(w.category, DeprecationWarning) for w in caught))
        a.check("find_artifacts finds by exact name, and falls back to legacy",
                lambda: len(find_artifacts(tmp, "cost_probe")) >= 1)

        dry = migrate(tmp, dry_run=True)
        a.check("migrate --dry-run renames nothing",
                lambda: (legacy_dir / "result.json").exists()
                and all(s == "would rename" for *_, s in dry))
        rows = migrate(tmp)
        a.check("migrate renames all three legacy files",
                lambda: sum(1 for *_, s in rows if s == "renamed"), expect=3)
        (legacy_dir / "result.json").write_text(json.dumps({"affine": {"d": 2.0}}))
        again = migrate(tmp)
        a.check("...and refuses to overwrite an existing target",
                lambda: any("SKIP target exists" in s for *_, s in again))

        # Recipes: the same discipline on the input side.
        a.check("recipe_kind reads the declared kind",
                lambda: recipe_kind({"kind": "sweep"}), expect="sweep")
        a.check("...and falls back to the schema when none is declared",
                lambda: recipe_kind({"model": "srw", "scales": [1], "n": 2}),
                expect="samples")
        a.check("...and returns None when nothing matches",
                lambda: recipe_kind({"hello": 1}) is None)
        a.check("recipe_name is <prefix>_<name>.json",
                lambda: recipe_name("samples", "omega1"), expect="samples_omega1.json")
        a.raises("recipe_name refuses an unknown kind", ValueError,
                 lambda: recipe_name("nope", "x"), contains="unknown recipe kind")

        good = tmp / "samples_x.json"
        good.write_text(json.dumps({"kind": "samples", "model": "srw",
                                    "scales": [8, 16], "n": 10}))
        a.check("load_recipe accepts the right kind",
                lambda: load_recipe(good, "samples")["model"], expect="srw")
        cost = tmp / "cost_x.json"
        cost.write_text(json.dumps({"kind": "cost_probe", "model": "srw",
                                    "scales": [8], "repeats": 3}))
        a.raises("the wrong recipe kind fails immediately, naming the mistake",
                 ValueError, lambda: load_recipe(cost, "samples"),
                 contains="is a cost_probe recipe")
        a.raises("...and names the tool that DOES consume it", ValueError,
                 lambda: load_recipe(cost, "samples"), contains="measure_cost.py")
        a.raises("load_recipe refuses an unknown expected kind", ValueError,
                 lambda: load_recipe(good, "nope"), contains="unknown recipe kind")
        a.check("RECIPES lists the four kinds and their consumers",
                lambda: sorted(RECIPES),
                expect=["cost_probe", "samples", "samples_shared", "sweep"])

        exp = tmp / "experiments" / "99_x"
        (exp / "recipes").mkdir(parents=True)
        a.check("default_out_dir maps recipes/ -> the EXPERIMENT's data/, "
                "not recipes/data/",
                lambda: default_out_dir(exp / "recipes" / "samples_a.json"),
                expect=exp / "data")
        a.check("...and a loose recipe still gets a sibling data/",
                lambda: default_out_dir(tmp / "samples_a.json"), expect=tmp / "data")
        a.check("recipes_dir is its exact inverse",
                lambda: recipes_dir(exp / "data"), expect=exp / "recipes")

        # artifacts.py is the one module under tools/ with a __main__.
        p_list = cli_ok(a, "tools/artifacts.py --list prints the naming table",
                        ["tools/artifacts.py", "--list"], stdout_has="filename")
        if p_list:
            a.check("...covering every registered kind",
                    lambda: all(k in p_list.stdout for k in ARTIFACTS))
        legacy2 = tmp / "legacy2"
        legacy2.mkdir()
        (legacy2 / "metadata.json").write_text(json.dumps({"scales": [1], "n": [2]}))
        cli_ok(a, "--migrate --dry-run reports without renaming",
               ["tools/artifacts.py", "--migrate", str(legacy2), "--dry-run"],
               stdout_has="would rename")
        a.check("...and really renamed nothing",
                lambda: (legacy2 / "metadata.json").exists())
        cli_ok(a, "--migrate renames in place",
               ["tools/artifacts.py", "--migrate", str(legacy2)],
               creates=legacy2 / "samples_meta.json")
        # tools/artifacts.py is the one runnable module under tools/, and the
        # two places that describe the layer have to say so.
        a.check("README.md and CATALOG.md both name artifacts.py as tools/'s one "
                "runnable exception",
                lambda: all("artifacts.py" in (ROOT / f).read_text().split(
                    "imported, n")[1][:120]
                    for f in ("README.md", "CATALOG.md")),
                detail="the layer descriptions used to say 'imported, never run'")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sec_persistence(a: Audit) -> None:
    """tools/persistence -- run directories, both sample layouts, metadata"""
    from tools.persistence import (content_id, load_metadata, load_samples,
                                   normalize_scales_n, open_scale_writer, run_dir,
                                   save_samples, write_metadata)

    a.begin("tools/persistence", "run directories, both sample layouts, metadata")

    a.check("a scalar n is broadcast over the scales",
            lambda: normalize_scales_n([8, 16, 32], 5), expect=([8, 16, 32], [5, 5, 5]))
    a.check("a per-scale list is kept",
            lambda: normalize_scales_n([8, 16], [1, 2]), expect=([8, 16], [1, 2]))
    a.check("a scalar scale is accepted too",
            lambda: normalize_scales_n(8, 3), expect=([8], [3]))
    a.raises("a mismatched n is refused", ValueError,
             lambda: normalize_scales_n([8, 16], [1, 2, 3]), contains="scalar or match")

    tmp = Path(tempfile.mkdtemp(prefix="persist_"))
    try:
        rd = run_dir(tmp, "demo")
        a.check("run_dir is <out_dir>/<tag>/", lambda: rd, expect=tmp / "demo")
        samples = {8: np.arange(10.0), 16: np.arange(20.0)}
        save_samples(rd, samples)
        a.check("save_samples writes one compressed npz",
                lambda: (rd / "samples.npz").exists())
        back = load_samples(rd)
        a.check("...which load_samples reads back exactly",
                lambda: np.array_equal(back[8], samples[8])
                and np.array_equal(back[16], samples[16]))

        chunk_rd = run_dir(tmp, "chunked")
        mm = open_scale_writer(chunk_rd, 8, 5, np.int64)
        mm[:] = np.arange(5)
        del mm
        a.check("open_scale_writer creates <run>/samples/<scale>.npy",
                lambda: (chunk_rd / "samples" / "8.npy").exists())
        chunked = load_samples(chunk_rd)
        a.check("...and load_samples reads the chunked layout too, as a memmap",
                lambda: list(chunked[8]), expect=[0, 1, 2, 3, 4])

        both = run_dir(tmp, "both")
        save_samples(both, {8: np.zeros(3)})
        mm2 = open_scale_writer(both, 8, 3, np.int64)
        del mm2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            served = load_samples(both)
        a.check("a run directory holding BOTH layouts warns rather than choosing "
                "silently", lambda: any(issubclass(w.category, RuntimeWarning)
                                        for w in caught),
                detail="the flat .npz still wins, but no longer invisibly")
        a.check("...naming which one it served",
                lambda: len(served[8]) == 3
                and "samples.npz" in str(caught[0].message))

        a.raises("a run directory with neither layout gives a clear error",
                 FileNotFoundError, lambda: load_samples(tmp / "empty"),
                 contains="generate")
        a.check("missing metadata is None, not an error (it is optional context)",
                lambda: load_metadata(rd) is None)
        write_metadata(run_dir=rd, model="srw", params={"q": 0.5}, scales=[8, 16],
                       n=[10, 20], seed=7, timing_seconds={8: 0.1, 16: 0.2})
        meta = load_metadata(rd)
        a.check("write_metadata records model, params, scales, n, seed and timings",
                lambda: all(k in meta for k in
                            ("model", "params", "scales", "n", "seed",
                             "timing_seconds", "created")))
        a.check("...with timing keys as strings, JSON's only kind",
                lambda: sorted(meta["timing_seconds"]), expect=["16", "8"])

        a.check("content_id is deterministic",
                lambda: content_id({"q": 0.5}, [8], [10], 1)
                == content_id({"q": 0.5}, [8], [10], 1))
        a.check("...and changes with any part of the content",
                lambda: content_id({"q": 0.5}, [8], [10], 1)
                != content_id({"q": 0.5}, [8], [10], 2))
        a.check("...and is short enough to be a directory name",
                lambda: len(content_id({}, [1], [1], 0)), expect=12)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sec_models_registry(a: Audit) -> None:
    """tools/models -- the registry -- a pure importer, no simulation of its own"""
    from tools.models import MODELS, ModelSpec, get_model

    a.begin("tools/models", "the registry -- a pure importer, no simulation of its own")

    a.check("all four models are registered", lambda: sorted(MODELS),
            expect=["percolation2d", "percolation_tau", "srw", "synthetic"])
    a.raises("an unknown model is refused with the list of known ones", ValueError,
             lambda: get_model("nope"), contains="unknown model")
    srw, syn = get_model("srw"), get_model("synthetic")
    a.check("ModelSpec is frozen, so a registry entry cannot be mutated in place",
            lambda: isinstance(srw, ModelSpec) and srw.__dataclass_params__.frozen)
    a.check("srw declares a cost_hint (which is what makes it plannable)",
            lambda: srw.cost_hint is not None)
    a.check("...and deliberately NO target_fn, so no driver can overlay its truth",
            lambda: srw.target_fn is None and srw.true_gamma_key is None)
    a.check("synthetic declares both, being the only model with a known closed form",
            lambda: syn.target_fn is not None and syn.true_gamma_key == "gamma")
    a.close("srw's cost_hint is exactly i", lambda: srw.cost_hint(4096, {}), 4096.0, 0)
    a.close("synthetic's is constant in i", lambda: syn.cost_hint(4096, {}), 1.0, 0)
    perc = get_model("percolation2d")
    a.check("percolation2d also declares no target_fn -- 91/48 stays out of the code",
            lambda: perc.target_fn is None and perc.true_gamma_key is None)
    a.close("...and its cost_hint is exactly i**2, the lattice's own site count",
            lambda: perc.cost_hint(512, {}), 512.0 ** 2, 0)
    tau = get_model("percolation_tau")
    a.check("percolation_tau declares no target_fn either -- tau = 187/91 and the "
            "hyperscaling relation it comes from stay out of the code",
            lambda: tau.target_fn is None and tau.true_gamma_key is None)
    a.close("...and its cost_hint is the area of the box its rung is drawn on, "
            "L(s)**2 with L(1024) = ceil(16*32) = 512",
            lambda: tau.cost_hint(1024, {}), 512.0 ** 2, 0)


def sec_loglog_plot(a: Audit) -> None:
    """tools/loglog_plot -- the shared charts"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.loglog import compare_methods
    from tools.loglog_plot import estimates_plot, loglog_plot, loglog_points

    a.begin("tools/loglog_plot", "the shared charts")

    samples = {8: np.array([1.0, 2.0, 3.0]), 16: np.array([2.0, 4.0, 6.0])}
    scales, y_bar, se, n = loglog_points(samples)
    a.check("loglog_points sorts by scale", lambda: list(scales), expect=[8, 16])
    a.close("...means the draws", lambda: y_bar[0], 2.0, 1e-12)
    a.close("...and standard-errors them", lambda: se[0], 1.0 / math.sqrt(3), 1e-12)
    a.check("...reporting n, which gamma_mle needs", lambda: list(n), expect=[3, 3])

    summarized = {8: (2.0, 0.5, 3), 16: (4.0, 1.0, 3)}
    s2, y2, se2, n2 = loglog_points(summarized)
    a.check("a (y_bar, se, n) summary triple gives the same chart, not a second one",
            lambda: list(y2) == [2.0, 4.0] and list(se2) == [0.5, 1.0]
            and list(n2) == [3, 3])

    fig, ax = plt.subplots()
    loglog_plot(samples, ax=ax, target_fn=lambda i: np.asarray(i, float) ** 0.5,
                fit_fn=lambda i: np.asarray(i, float) ** 0.5, fit_label="fit",
                label="data")
    a.check("loglog_plot draws data, a target and a fit on log-log axes",
            lambda: ax.get_xscale() == "log" and ax.get_yscale() == "log"
            and len(ax.get_lines()) >= 2)
    plt.close(fig)
    fig, ax = plt.subplots()
    loglog_plot(samples, ax=ax)
    a.check("...and works with neither overlay", lambda: ax.get_ylabel() != "")
    plt.close(fig)

    big = {2 ** k: (float(2 ** k) ** 0.5, 0.01, 1000) for k in range(3, 10)}
    res = compare_methods(*loglog_points(big)[:2], loglog_points(big)[3],
                          true_gamma=0.5)
    fig, ax = plt.subplots()
    estimates_plot(res, ax=ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    a.check("estimates_plot labels all four estimators plus the truth",
            lambda: len(labels) >= 5, detail=str(labels))
    plt.close(fig)
    res_bad = json.loads(json.dumps(res))
    res_bad["methods"]["mle"]["trustworthy"] = False
    fig, ax = plt.subplots()
    estimates_plot(res_bad, ax=ax)
    warn = [t.get_text() for t in ax.get_legend().get_texts()
            if "not trustworthy" in t.get_text()]
    a.check("an untrustworthy mle says so in words, never by colour alone",
            lambda: len(warn), expect=1)
    plt.close(fig)


# ===========================================================================
# STAGE 2 -- models/, the simulated objects themselves
# ===========================================================================

def sec_srw(a: Audit) -> None:
    """models/srw -- |S_k| for a +-1 walk -- Theta(k) on purpose"""
    from fractions import Fraction
    from math import comb

    from models import srw as srw_mod

    a.begin("models/srw", "|S_k| for a +-1 walk -- Theta(k) on purpose")

    def exact_mean(k):
        return float(Fraction(k * comb(k - 1, (k - 1) // 2), 1 << (k - 1)))

    rng = np.random.default_rng(0)
    s = srw_mod.srw(16, n=50_000, rng=rng)
    a.check("srw returns n non-negative integers", lambda: s.shape == (50_000,)
            and s.min() >= 0)
    a.check("|S_k| has the parity of k", lambda: bool(np.all(s % 2 == 16 % 2)))
    a.check("...and never exceeds k", lambda: int(s.max()) <= 16)
    a.close("the sample mean matches the exact E|S_16| to ~3 se",
            lambda: float(s.mean()), exact_mean(16),
            4 * float(s.std(ddof=1)) / math.sqrt(50_000))
    a.close("E[S_k^2] = k, so E|S_k|^2 = k too",
            lambda: float((s.astype(float) ** 2).mean()), 16.0, 0.3)

    # The block_n invariance the module docstring rests on -- it is what makes
    # generate.py's chunked path bit-identical to the in-RAM one.
    one = srw_mod.srw(7, n=50, rng=np.random.default_rng(3))
    blocked = srw_mod.srw(7, n=50, rng=np.random.default_rng(3), block_n=3)
    a.check("blocking over n is bit-identical to one unblocked call",
            lambda: np.array_equal(one, blocked))
    a.check("...at block_n = 1 as well",
            lambda: np.array_equal(one, srw_mod.srw(7, n=50,
                                                    rng=np.random.default_rng(3),
                                                    block_n=1)))
    a.check("k = 1 works and gives |S_1| = 1 always",
            lambda: set(np.unique(srw_mod.srw(1, n=100, rng=rng))), expect={1})
    a.check("n = 1 works (what the cost probe calls)",
            lambda: srw_mod.srw(64, n=1, rng=rng).shape, expect=(1,))
    biased = srw_mod.srw(64, n=20_000, q=0.9, rng=rng)
    a.close("a biased walk drifts: E|S_64| ~ 64*(2q-1) at q = 0.9",
            lambda: float(biased.mean()), 64 * 0.8, 1.0)
    a.check("an unseeded call still works (fresh entropy)",
            lambda: srw_mod.srw(8, n=3).shape, expect=(3,))
    a.check("simulate() is the registry's entry point and honours params",
            lambda: srw_mod.simulate(32, 5, {"q": 0.5}, rng).shape, expect=(5,))
    a.close("cost_hint is exactly i -- declared, not fitted",
            lambda: srw_mod.cost_hint(4096), 4096.0, 0)


def sec_percolation2d(a: Audit) -> None:
    """models/percolation2d -- the south-connected cluster at p_c, cost i**2"""
    import itertools

    from models import percolation2d as perc

    a.begin("models/percolation2d",
            "the south-connected cluster at p_c, cost i**2")

    rng = np.random.default_rng(0)
    y = perc.percolation2d(16, n=5_000, rng=rng)
    a.check("returns n counts in [0, i**2]",
            lambda: y.shape == (5_000,) and y.min() >= 0 and y.max() <= 256)
    a.check("p = 1 fills the box exactly",
            lambda: set(np.unique(perc.percolation2d(7, n=20, p=1.0, rng=rng))),
            expect={49})
    a.check("p = 0 leaves it empty",
            lambda: set(np.unique(perc.percolation2d(7, n=20, p=0.0, rng=rng))),
            expect={0})
    a.close("i = 1 is a single Bernoulli(p): mean p",
            lambda: float(perc.percolation2d(1, n=200_000, p=0.3,
                                             rng=rng).mean()), 0.3, 0.01)

    # The strongest check available here: E[Y_i] by exhaustive enumeration of
    # all 2**(i*i) lattices, against Monte Carlo. A closed form, not a second
    # simulation.
    def brute_mean(i, p, geometry="box"):
        total = 0.0
        for bits in itertools.product((False, True), repeat=i * i):
            grid = np.array(bits, dtype=bool).reshape(i, i)
            k = int(grid.sum())
            lab = perc._label_block(grid[None])
            count = int(perc._south_counts(lab, perc._roots(lab, i, geometry))[0])
            total += p ** k * (1 - p) ** (i * i - k) * count
        return total

    for geometry in ("box", "cylinder"):
        for i in (2, 3):
            n = 200_000
            draws = perc.percolation2d(i, n=n, geometry=geometry,
                                       rng=np.random.default_rng(100 + i))
            a.close(f"{geometry}: E[Y_{i}] matches exhaustive enumeration over all "
                    f"2**{i * i} lattices",
                    lambda draws=draws: float(draws.mean()),
                    brute_mean(i, perc.P_C_SQUARE_SITE, geometry),
                    4 * float(draws.std(ddof=1)) / math.sqrt(n))

    a.close("P(Y_i = 0) = (1-p)**i exactly -- row 0 entirely closed, nothing else",
            lambda: float((perc.percolation2d(4, n=200_000, p=0.5,
                                              rng=np.random.default_rng(5)) == 0).mean()),
            perc.zero_rate(4, 0.5), 0.004)

    one = perc.percolation2d(9, n=300, rng=np.random.default_rng(3))
    a.check("blocking over the sample axis is bit-identical to one unblocked call",
            lambda: np.array_equal(
                one, perc.percolation2d(9, n=300, rng=np.random.default_rng(3),
                                        block_n=7)))
    a.check("...at block_n = 1 as well",
            lambda: np.array_equal(
                one, perc.percolation2d(9, n=300, rng=np.random.default_rng(3),
                                        block_n=1)))
    stacked = perc._label_block(np.stack([
        np.ones((6, 6), dtype=bool), np.zeros((6, 6), dtype=bool),
        np.ones((6, 6), dtype=bool)]))
    a.check("the blank separator row keeps one sample's cluster out of the next",
            lambda: list(perc._south_counts(stacked, perc._roots(stacked, 6, "box"))),
            expect=[36, 0, 36])

    a.raises("an unknown anchor is refused", ValueError,
             lambda: perc.percolation2d(8, n=1, anchor="north"),
             contains="unknown anchor")
    a.raises("a box side below 1 is refused", ValueError,
             lambda: perc.percolation2d(0, n=1), contains="box side")
    a.raises("a p outside [0, 1] is refused", ValueError,
             lambda: perc.percolation2d(4, n=1, p=1.5), contains="p must be in")

    origin = perc.percolation2d(32, n=2_000, anchor="origin",
                                rng=np.random.default_rng(7))
    a.close("the origin anchor is 0 whenever the centre is closed -- Assumption 2 "
            "fails on most draws, which is why ground rule 7 rejects it",
            lambda: float((origin == 0).mean()), 1 - perc.P_C_SQUARE_SITE, 0.05)
    a.check("simulate() is the registry's entry point and honours params",
            lambda: perc.simulate(16, 5, {"anchor": "origin"}, rng).shape, expect=(5,))
    a.check("an unseeded call still works (fresh entropy)",
            lambda: perc.percolation2d(8, n=3).shape, expect=(3,))
    a.close("cost_hint is exactly i**2 -- declared, not fitted",
            lambda: perc.cost_hint(1024, {}), 1024.0 ** 2, 0)
    a.close("...and does not change with the geometry: the wrap-merge is a nearly "
            "constant factor, and only ratios across scales reach an allocation",
            lambda: perc.cost_hint(1024, {"geometry": "cylinder"}), 1024.0 ** 2, 0)

    # --- the cylinder: periodic in x ---
    a.raises("an unknown geometry is refused", ValueError,
             lambda: perc.percolation2d(8, n=1, geometry="torus"),
             contains="unknown geometry")
    box = perc.percolation2d(24, n=400, rng=np.random.default_rng(31))
    cyl = perc.percolation2d(24, n=400, geometry="cylinder",
                             rng=np.random.default_rng(31))
    a.check("the geometry does not touch the RNG, so box and cylinder see the same "
            "lattices at one seed -- and wrapping only ever adds sites",
            lambda: bool(np.all(cyl >= box) and np.any(cyl > box)))
    a.check("i = 1 and i = 2 wrap onto an edge that is already there -- a no-op",
            lambda: all(np.array_equal(
                perc.percolation2d(i, n=200, rng=np.random.default_rng(32)),
                perc.percolation2d(i, n=200, geometry="cylinder",
                                   rng=np.random.default_rng(32)))
                for i in (1, 2)))

    # Three labels chained by two wrap edges: resolving them takes two hops, so
    # a merge without the pointer-jumping step would undercount (it returns 9).
    chain = np.zeros((7, 7), dtype=bool)
    chain[0, 0] = True
    chain[1, 0:3] = True
    chain[1, 4:7] = True
    chain[2, 6] = True
    chain[3, 6] = True
    chain[3, 0:3] = True
    lab_chain = perc._label_block(chain[None])

    def chain_count(geometry):
        return int(perc._south_counts(
            lab_chain, perc._roots(lab_chain, 7, geometry))[0])

    a.check("the wrap-merge resolves a CHAIN of labels, not just a pair",
            lambda: (chain_count("box"), chain_count("cylinder")), expect=(4, 12))
    a.close("a cylinder is quieter than a box at the same scale (cv, i = 64)",
            lambda: float(np.std(perc.percolation2d(
                64, n=4000, geometry="cylinder", rng=np.random.default_rng(33)),
                ddof=1) / np.mean(perc.percolation2d(
                    64, n=4000, geometry="cylinder",
                    rng=np.random.default_rng(33)))), 0.376, 0.03)


def sec_percolation_tau(a: Audit) -> None:
    """models/percolation_tau -- the cluster-number density at p_c, scale = cluster size"""
    from models import percolation_tau as ptau

    a.begin("models/percolation_tau",
            "the cluster-number density at p_c; the ladder variable is a CLUSTER SIZE")

    rng = np.random.default_rng(0)
    L16 = ptau.box_side(16)
    y = ptau.percolation_tau(16, n=5_000, rng=rng)
    a.check("returns n per-site densities, each an integer count over L(s)**2",
            lambda: y.shape == (5_000,) and y.min() >= 0.0
            and np.allclose(y * L16 ** 2, np.round(y * L16 ** 2)))
    a.check("p = 1 is one cluster filling the torus",
            lambda: set(np.unique(ptau.percolation_tau(
                4, n=20, p=1.0, observable="tail", rng=rng))),
            expect={1.0 / ptau.box_side(4) ** 2})
    a.check("p = 0 has no clusters at all",
            lambda: set(np.unique(ptau.percolation_tau(4, n=20, p=0.0, rng=rng))),
            expect={0.0})

    # The box rule, which is the whole reason no rung has to be discarded.
    a.check("L(s) = ceil(box_factor * s**box_exponent), and the ceil tolerance "
            "does not add a row at an exact hit (16*sqrt(16) = 64)",
            lambda: [ptau.box_side(s) for s in (1, 8, 16, 64, 1024)],
            expect=[16, 46, 64, 128, 512])
    a.check("box_exponent = 0 is the fixed-box arm: every rung on one L x L box",
            lambda: [ptau.box_side(s, 40.0, 0.0) for s in (2, 200, 20_000)],
            expect=[40, 40, 40])
    a.close("cost_hint is that box's area -- so d = 2*box_exponent, exactly 1 by "
            "default, and it is a geometric fact rather than a stated formula",
            lambda: ptau.cost_hint(1024, {}), 512.0 ** 2, 0)

    # Against an independent flood fill -- the check that found the merge bug.
    def flood_sizes(grid, wrap):
        L, M = grid.shape
        seen = np.zeros_like(grid, dtype=bool)
        out = []
        for r0 in range(L):
            for c0 in range(M):
                if not grid[r0, c0] or seen[r0, c0]:
                    continue
                seen[r0, c0] = True
                stack, size = [(r0, c0)], 0
                while stack:
                    r, c = stack.pop()
                    size += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = (r + dr, c + dc)
                        if wrap:
                            rr, cc = rr % L, cc % M
                        if 0 <= rr < L and 0 <= cc < M and grid[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                out.append(size)
        return out

    grids = (np.random.default_rng(2).random((30, 32, 32))
             < ptau.P_C_SQUARE_SITE)
    for geometry, wrap in (("box", False), ("torus", True)):
        lab = ptau._label_block(grids)
        got = ptau._counts_in_range(lab, 32, ptau._roots(lab, 32, geometry), 16, 32)
        want = [sum(1 for z in flood_sizes(g, wrap) if 16 <= z < 32) for g in grids]
        a.check(f"{geometry}: the block-labelled count matches a plain flood fill "
                f"on 30 critical 32x32 lattices", lambda got=got: list(got),
                expect=want)

    one = ptau.percolation_tau(16, n=300, rng=np.random.default_rng(3))
    a.check("blocking over the sample axis is bit-identical to one unblocked call",
            lambda: np.array_equal(
                one, ptau.percolation_tau(16, n=300, rng=np.random.default_rng(3),
                                          block_n=7)))
    a.check("...at block_n = 1 as well -- the case that exposed the early-exit "
            "bug in the label merge",
            lambda: np.array_equal(
                one, ptau.percolation_tau(16, n=300, rng=np.random.default_rng(3),
                                          block_n=1)))

    def next_draw(observable, geometry):
        r = np.random.default_rng(11)
        ptau.percolation_tau(16, n=40, observable=observable, geometry=geometry, rng=r)
        return float(r.random())
    a.check("neither switch touches the RNG: all four combinations see the same "
            "lattices at one seed",
            lambda: len({next_draw(o, g) for o in ptau.OBSERVABLES
                         for g in ptau.GEOMETRIES}), expect=1)

    kw = dict(n=200, box_factor=24.0, box_exponent=0.0, geometry="torus")
    a.check("#[s, 2s) = #>=s - #>=2s exactly, sample by sample, on the same box",
            lambda: np.allclose(
                ptau.percolation_tau(9, observable="bin", bin_ratio=2.0,
                                     rng=np.random.default_rng(4), **kw),
                ptau.percolation_tau(9, observable="tail",
                                     rng=np.random.default_rng(4), **kw)
                - ptau.percolation_tau(18, observable="tail",
                                       rng=np.random.default_rng(4), **kw)))

    a.raises("an unknown observable is refused", ValueError,
             lambda: ptau.percolation_tau(8, observable="histogram"),
             contains="unknown observable")
    a.raises("an unknown geometry is refused", ValueError,
             lambda: ptau.percolation_tau(8, geometry="cylinder"),
             contains="unknown geometry")
    a.raises("a cluster size below 1 is refused", ValueError,
             lambda: ptau.percolation_tau(0), contains="cluster size")
    a.raises("a p outside [0, 1] is refused", ValueError,
             lambda: ptau.percolation_tau(4, p=1.5), contains="p must be in")
    a.raises("a bin with no room in it is refused", ValueError,
             lambda: ptau.percolation_tau(8, bin_ratio=1.0), contains="bin_ratio")
    a.raises("a box too small to hold the cluster it is looking for is refused, "
             "rather than returning an all-zero rung",
             ValueError,
             lambda: ptau.percolation_tau(100, box_factor=2.0, box_exponent=0.0),
             contains="does not fit in the box")

    a.check("simulate() is the registry's entry point and honours params",
            lambda: ptau.simulate(16, 5, {"observable": "tail",
                                          "geometry": "box"}, rng).shape,
            expect=(5,))
    a.check("an unseeded call still works (fresh entropy)",
            lambda: ptau.percolation_tau(8, n=3).shape, expect=(3,))
    # --- the shared-lattice sampler: one box, every rung ---
    windows = ptau.bin_edges([8, 16, 32])
    a.check("bin_edges gives the same windows simulate uses",
            lambda: windows, expect=[(8, 16), (16, 32), (32, 64)])
    a.check("shared_box_side inverts s_top <= cut_fraction * L**df_lower, tightly",
            lambda: (lambda L: (windows[-1][1] <= 0.05 * L ** 1.85
                                and windows[-1][1] > 0.05 * (L - 1) ** 1.85))(
                ptau.shared_box_side(windows[-1][1], 1.85, 0.05)))
    a.raises("a cut_fraction outside (0, 1] is refused", ValueError,
             lambda: ptau.shared_box_side(64, 1.85, 0.0), contains="cut_fraction")

    L_shared = ptau.shared_box_side(windows[-1][1], 1.85, 0.05)
    c = ptau.binned_counts(L_shared, 60, windows, rng=np.random.default_rng(3))
    a.check("binned_counts returns one column per window, off the same lattices",
            lambda: c.shape, expect=(60, 3))
    a.check("...and every window equals the per-rung path on those lattices -- the "
            "cheap sampler changes the BUDGET, not the observable",
            lambda: all(np.array_equal(
                c[:, j],
                np.round(ptau.percolation_tau(
                    lo, n=60, observable="bin", bin_ratio=hi / lo,
                    box_factor=float(L_shared), box_exponent=0.0,
                    rng=np.random.default_rng(3)) * L_shared ** 2).astype(np.int64))
                for j, (lo, hi) in enumerate(windows)))
    a.check("disjoint windows add up WITHIN each lattice -- the correlation, exactly",
            lambda: np.array_equal(
                *(lambda d: (d[:, 0] + d[:, 1], d[:, 2]))(
                    ptau.binned_counts(48, 100, [(8, 16), (16, 32), (8, 32)],
                                       rng=np.random.default_rng(5)))))
    samples, info = ptau.shared_sampler([8, 16, 32], 40, {}, np.random.default_rng(2))
    a.check("shared_sampler returns one array per scale plus an info dict that "
            "stamps the run as shared",
            lambda: (sorted(samples) == [8, 16, 32]
                     and all(v.shape == (40,) for v in samples.values())
                     and info["shared_lattice"] is True
                     and info["sites"] == 40 * info["L"] ** 2))
    a.raises("an unordered ladder is refused", ValueError,
             lambda: ptau.shared_sampler([16, 8], 5, {}, np.random.default_rng(0)),
             contains="strictly increasing")
    a.note("shared_sampler BREAKS ground rule 2 across scales, deliberately",
           f"its rungs come from the same lattices, so Cov(Ybar_s, Ybar_s') != 0 "
           f"(measured: a flat +0.11 pedestal, not decaying in lag). Because the "
           f"eq. (526) weights sum to zero that costs the slope's error bar 1-6%, "
           f"but every run drawn this way is stamped shared_lattice=true so it "
           f"cannot be mistaken for an independent one.")

    a.note("Y_s = 0 is ORDINARY here, unlike percolation2d's south anchor",
           f"the count of clusters at one size scale in one box is a small "
           f"near-Poisson number -- zero fraction {ptau.zero_fraction(y):.2f} at "
           f"the default box_factor = {ptau.DEFAULT_BOX_FACTOR}, i.e. a mean of "
           f"~{-np.log(max(ptau.zero_fraction(y), 1e-12)):.2f} clusters per box. "
           f"Assumption 2 is REPORTED by zero_fraction, never asserted; "
           f"box_factor is the knob that controls it.")


def sec_synthetic(a: Audit) -> None:
    """models/synthetic -- the planted eq. (232) generator"""
    from models import synthetic as syn

    a.begin("models/synthetic", "the planted eq. (232) generator")

    p = syn.SyntheticParams(gamma=0.5, a0=2.0, corrections=[(1.0, 1.0), (0.5, 2.0)])
    a.check("corrections normalize to float pairs whatever they came in as",
            lambda: p.corrections, expect=((1.0, 1.0), (0.5, 2.0)))
    a.check("omega1/a1 read off the leading correction",
            lambda: (p.omega1, p.a1), expect=(1.0, 1.0))
    a.check("...and are None when there is no correction",
            lambda: syn.SyntheticParams(gamma=0.5).omega1 is None)
    a.raises("eq. (232)'s ordering 0 < omega_1 < omega_2 is enforced", ValueError,
             lambda: syn.SyntheticParams(gamma=0.5, corrections=[(1.0, 2.0), (1.0, 1.0)]),
             contains="increasing")
    a.raises("...including a repeated omega", ValueError,
             lambda: syn.SyntheticParams(gamma=0.5, corrections=[(1.0, 1.0), (1.0, 1.0)]),
             contains="increasing")
    a.raises("omega_j <= 0 is refused", ValueError,
             lambda: syn.SyntheticParams(gamma=0.5, corrections=[(1.0, 0.0)]),
             contains="> 0")
    a.raises("an unknown noise family is refused", ValueError,
             lambda: syn.SyntheticParams(gamma=0.5, family="cauchy"),
             contains="unknown noise family")
    a.check("NOISE_FAMILIES is the registry the docstring advertises",
            lambda: list(syn.NOISE_FAMILIES), expect=["lognormal"])

    d = {"gamma": 0.5, "a0": 2.0, "corrections": [[1.0, 1.0]], "sigma_inf2": 0.04}
    a.check("params_from_dict round-trips JSON's list-of-lists",
            lambda: syn.params_from_dict(d).corrections, expect=((1.0, 1.0),))
    a.close("mean_Y is a0 i^gamma exp(sum a_j i^-omega_j), eq. (232)",
            lambda: float(syn.mean_Y(100, d)),
            2.0 * 100 ** 0.5 * math.exp(1.0 / 100), 1e-12)
    a.check("target_fn is mean_Y under the registry's name",
            lambda: float(syn.target_fn(100, d)) == float(syn.mean_Y(100, d)))

    rng = np.random.default_rng(0)
    exact = syn.simulate(64, 1000, {"gamma": 0.5, "a0": 1.0, "sigma_inf2": 0.0}, rng)
    a.check("sigma_inf2 = 0 degenerates to the noiseless power law exactly",
            lambda: float(np.ptp(exact)) == 0.0)
    a.close("...at the right value", lambda: float(exact[0]), 8.0, 1e-12)
    noisy = syn.simulate(64, 200_000, {"gamma": 0.5, "a0": 1.0, "sigma_inf2": 0.25}, rng)
    a.close("E[xi] = 1 exactly, so the sample mean lands on mean_Y",
            lambda: float(noisy.mean()) / 8.0, 1.0, 0.01)
    a.close("...and Var(xi) is the sigma_inf2 asked for",
            lambda: float(np.var(noisy / 8.0, ddof=1)), 0.25, 0.01)
    a.check("Assumption 2 (Y_i > 0) holds by construction", lambda: bool(noisy.min() > 0))
    a.close("cost_hint is constant in i -- correct, and why d = 0 here",
            lambda: syn.cost_hint(4096), 1.0, 0)

    # --- the work burn: this model's only way to have a d at all ---
    burn = {"gamma": 0.5, "a0": 1.0, "sigma_inf2": 0.6,
            "cost_d": 2.0, "cost_scale": 1e-3 / 64 ** 2}
    a.check("the burn is off by default",
            lambda: (syn.params_from_dict({"gamma": 0.5}).cost_scale,
                     syn.params_from_dict({"gamma": 0.5}).cost_d), expect=(0.0, 0.0))
    a.close("cost_hint becomes cost_scale * i**cost_d once it is on",
            lambda: syn.cost_hint(128, burn), 4e-3, 1e-12)
    a.check("...and stays ONE WORK UNIT when it is off, which is a different "
            "unit from seconds",
            lambda: syn.cost_hint(128, {"gamma": 0.5, "cost_d": 2.0}), expect=1.0)
    same = [syn.simulate(8, 40, dict(burn, cost_d=cd, cost_scale=1e-7),
                         np.random.default_rng(4242)) for cd in (0.0, 1.0, 2.0)]
    a.check("the burn consumes no randomness: three exponents, one seed, "
            "bit-identical draws",
            lambda: all(np.array_equal(same[0], y) for y in same[1:]))
    t0 = time.perf_counter()
    syn.simulate(64, 1, burn, np.random.default_rng(0))     # warm
    syn.simulate(128, 1, burn, np.random.default_rng(0))    # d = 2 -> 4 ms
    a.check("...and it really spins for the time it declares",
            lambda: 3.5e-3 < time.perf_counter() - t0 < 4e-2)
    a.raises("a burn past MAX_BURN_SECONDS is refused, not run", ValueError,
             lambda: syn.simulate(1000, 1, {"gamma": 0.5, "cost_d": 2.0,
                                            "cost_scale": 1.0},
                                  np.random.default_rng(0)),
             contains="MAX_BURN_SECONDS")
    a.raises("a negative cost_d is refused", ValueError,
             lambda: syn.params_from_dict({"gamma": 0.5, "cost_d": -1.0}),
             contains="cost_d")
    a.raises("...and a negative cost_scale", ValueError,
             lambda: syn.params_from_dict({"gamma": 0.5, "cost_scale": -1.0}),
             contains="cost_scale")


# ===========================================================================
# STAGE 3 -- src/, the drivers a human runs. Exercised through the CLI.
# ===========================================================================

#: A throwaway experiment, laid out exactly like a real one so the drivers'
#: own path resolution (recipes/ -> data/) is what gets tested, not a
#: special case. Everything here is deliberately tiny: the question is
#: whether each flag does what it says, not whether the physics is right.
TINY_RECIPES = {
    "samples_synth.json": {
        "kind": "samples", "model": "synthetic",
        "params": {"gamma": 0.5, "a0": 1.0, "corrections": [[-0.25, 1.0]],
                   "sigma_inf2": 0.04},
        "scales": [8, 16, 32, 64, 128, 256], "n": 400, "seed": 11},
    "samples_srw.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64, 128, 256], "n": 2000, "seed": 12},
    "samples_perscale.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32], "n": [300, 200, 100], "seed": 13},
    "samples_neyman.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64], "n": {"rule": "neyman", "budget": 2e6},
        "seed": 14},
    "samples_snr.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64], "n": {"rule": "snr", "budget": 2e6,
                                         "omega1": 1.0}, "seed": 15},
    "samples_blind.json": {          # states NO design constants at all
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64], "n": {"rule": "snr", "budget": 2e6},
        "seed": 16},
    "samples_badrule.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16], "n": {"rule": "nope", "budget": 1e5}, "seed": 17},
    "samples_starved.json": {        # a ladder too wide for its budget
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64, 128, 256, 512, 1024],
        "n": {"rule": "snr", "budget": 3e3, "omega1": 1.0}, "seed": 18},
    "samples_pilot.json": {
        "kind": "samples", "model": "srw", "params": {"q": 0.5},
        "scales": [8, 16, 32, 64, 128, 256],
        "n": {"rule": "snr", "budget": 2e8}, "seed": 19},
    "samples_perc_south.json": {
        "kind": "samples", "model": "percolation2d",
        "params": {"p": 0.59274605079210, "anchor": "south"},
        "scales": [8, 16, 32, 64], "n": {"rule": "neyman", "budget": 4e6},
        "seed": 30},
    "samples_perc_origin.json": {
        "kind": "samples", "model": "percolation2d",
        "params": {"p": 0.59274605079210, "anchor": "origin"},
        "scales": [8, 16, 32, 64], "n": {"rule": "neyman", "budget": 4e6},
        "seed": 31},
    "cost_perc.json": {
        "kind": "cost_probe", "model": "percolation2d",
        "params": {"p": 0.59274605079210, "anchor": "south"},
        "scales": [32, 64, 128, 256], "repeats": 5, "seed": 32},
    "cost_srw.json": {
        "kind": "cost_probe", "model": "srw", "params": {"q": 0.5},
        "scales": [4096, 8192, 16384, 32768, 65536], "repeats": 5, "seed": 20},
    "cost_synth.json": {
        "kind": "cost_probe", "model": "synthetic",
        "params": {"gamma": 0.5, "a0": 1.0, "sigma_inf2": 0.04},
        "scales": [1024, 2048, 4096, 8192], "repeats": 5, "seed": 21},
    "cost_minmax.json": {            # exercises the per-recipe aggregator key
        "kind": "cost_probe", "model": "srw", "params": {"q": 0.5},
        "scales": [4096, 8192, 16384, 32768], "repeats": 5, "seed": 22,
        "aggregator": "min"},
    "sweep_tiny.json": {
        "kind": "sweep", "model": "srw", "params": {"q": 0.5},
        "budgets": [1e5, 3e5, 1e6], "m0_values": [0, 1, 2], "m": 4, "rho": 2.0,
        "d": 1.0, "omega1": 1.0, "replicates": 3, "true_gamma": 0.5,
        "seed": 23, "a1": -0.25, "cv": 0.7741},
}


class Scratch:
    """A temporary experiment directory, torn down when the audit finishes."""

    def __init__(self, base: Path | None = None):
        self.root = Path(base or tempfile.mkdtemp(prefix="loglog_audit_"))
        self.exp = self.root / "experiments" / "99_audit"
        self.recipes = self.exp / "recipes"
        self.data = self.exp / "data"
        self.recipes.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        for name, body in TINY_RECIPES.items():
            (self.recipes / name).write_text(json.dumps(body, indent=2))

    def recipe(self, name: str) -> str:
        return str(self.recipes / name)

    def run(self, tag: str) -> Path:
        return self.data / tag

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def sec_generate(a: Audit, sc: Scratch) -> None:
    """src/generate/generate.py -- the one sampler; every other driver calls it"""
    from src.generate.generate import UNINFORMED, generate, reproduce, resolve_n

    a.begin("src/generate/generate.py", "the one sampler; every other driver calls it")

    # --- the API, where the flags cannot reach ---
    out = generate("srw", [8, 16], 50, {"q": 0.5}, seed=3)
    a.check("generate() returns {scale: draws}",
            lambda: sorted(out) == [8, 16] and out[8].shape == (50,))
    a.check("the same seed gives bit-identical draws",
            lambda: np.array_equal(generate("srw", [8], 50, {"q": 0.5}, seed=3)[8],
                                   out[8]))
    a.check("a per-scale n list is honoured",
            lambda: [len(v) for _, v in
                     sorted(generate("srw", [8, 16], [10, 20], {"q": 0.5},
                                     seed=3).items())], expect=[10, 20])
    a.check("reduce= collapses each scale as it is drawn",
            lambda: generate("srw", [8], 100, {"q": 0.5}, seed=3,
                             reduce=np.mean)[8].shape, expect=())
    a.raises("reduce= with out_dir= is refused -- it would write summaries to "
             "a file named samples", ValueError,
             lambda: generate("srw", [8], 10, {"q": 0.5}, out_dir=sc.data,
                              tag="x", reduce=np.mean),
             contains="mutually exclusive")
    a.raises("an unknown model is refused before anything is drawn", ValueError,
             lambda: generate("nope", [8], 10, {}), contains="unknown model")

    # A spawned SeedSequence must survive being handed over as an object.
    from tools.rng import spawn
    kids = spawn(7, 2)
    d0 = generate("srw", [8], 20, {"q": 0.5}, seed=kids[0])[8]
    d1 = generate("srw", [8], 20, {"q": 0.5}, seed=kids[1])[8]
    a.check("two spawned children draw different samples (ground rule 2)",
            lambda: not np.array_equal(d0, d1))

    # The chunked path: forced by a tiny max_chunk_bytes rather than a huge n.
    rd = sc.data / "chunky"
    chunked = generate("srw", [8, 16], 500, {"q": 0.5}, seed=5, out_dir=sc.data,
                       tag="chunky", max_chunk_bytes=1000)
    a.check("a run over max_chunk_bytes streams to samples/<scale>.npy",
            lambda: (rd / "samples" / "8.npy").exists()
            and not (rd / "samples.npz").exists())
    from tools.persistence import load_samples
    a.check("...and load_samples reads it back at the right length",
            lambda: len(load_samples(rd)[16]), expect=500)
    a.check("...bit-identically to the in-RAM path, which is what block_n "
            "invariance buys",
            lambda: np.array_equal(np.asarray(load_samples(rd)[8]),
                                   generate("srw", [8, 16], 500, {"q": 0.5},
                                            seed=5)[8]))
    generate("srw", [8, 16], 20, {"q": 0.5}, seed=5, out_dir=sc.data, tag="chunky")
    a.check("a rerun that crosses the chunking threshold deletes the other "
            "layout, so no stale data can shadow it",
            lambda: (rd / "samples.npz").exists()
            and not (rd / "samples").exists())

    a.check("reproduce() redraws a run directory from its own metadata",
            lambda: np.array_equal(reproduce(rd)[8], load_samples(rd)[8]))
    a.raises("...and says so when there is no metadata", FileNotFoundError,
             lambda: reproduce(sc.data / "nothing"), contains="samples_meta")

    # --- allocation rules, and where their inputs come from ---
    cfg = json.loads((sc.recipes / "samples_neyman.json").read_text())
    n_ney = quiet(resolve_n, cfg)
    a.check("the neyman rule returns one n per scale, decreasing in i",
            lambda: len(n_ney) == 4 and n_ney[0] > n_ney[-1])
    a.check("...and stamps _allocation_resolved onto the recipe, with provenance",
            lambda: cfg["_allocation_resolved"]["d_source"],
            expect="declared by srw's cost_hint")
    cfg_snr = json.loads((sc.recipes / "samples_snr.json").read_text())
    n_snr = quiet(resolve_n, cfg_snr)
    a.check("the snr rule at omega1 = 1 increases in i",
            lambda: n_snr[-1] > n_snr[0])
    a.check("...recording that omega1 came from the recipe",
            lambda: cfg_snr["_allocation_resolved"]["omega1_source"], expect="recipe")
    cfg_blind = json.loads((sc.recipes / "samples_blind.json").read_text())
    n_blind = quiet(resolve_n, cfg_blind)
    a.check("a recipe stating NO design constants still runs",
            lambda: len(n_blind), expect=4)
    a.check("...on the UNINFORMED default, stamped as such",
            lambda: cfg_blind["_allocation_resolved"]["omega1_source"],
            expect="uninformed default")
    a.check("...which is omega1 = 0, deliberately not srw's truth of 1",
            lambda: UNINFORMED["omega1"], expect=0.0)
    a.raises("an unknown rule is a clean exit naming the two that exist",
             SystemExit, lambda: quiet(resolve_n,
                                       json.loads((sc.recipes / "samples_badrule.json"
                                                   ).read_text())),
             contains="unknown allocation rule")
    a.check("a plain scalar n passes straight through",
            lambda: resolve_n({"n": 100}), expect=100)

    # --- the CLI ---
    cli_ok(a, "generate.py runs a recipe and writes both files",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_synth.json"),
            "--tag", "cli_synth"],
           stdout_has="sample_mean", creates=sc.run("cli_synth") / "samples.npz")
    a.check("...into the EXPERIMENT's data/, not the recipe's sibling",
            lambda: (sc.run("cli_synth") / "samples_meta.json").exists())
    cli_ok(a, "--seed overrides the recipe's own",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_srw.json"),
            "--tag", "cli_seeded", "--seed", "999"], stdout_has="seed     = 999")
    cli_ok(a, "--out-dir redirects the run",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_perscale.json"),
            "--out-dir", str(sc.data / "elsewhere"), "--tag", "t"],
           creates=sc.data / "elsewhere" / "t" / "samples.npz")
    cli_ok(a, "--tag may nest, which is how replicate groups are laid out",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_srw.json"),
            "--tag", "grp/rep0"], creates=sc.run("grp/rep0") / "samples.npz")
    p = cli_ok(a, "no --tag falls back to a content hash",
               ["src/generate/generate.py", "-meta",
                sc.recipe("samples_perscale.json")], stdout_has="run_dir")
    if p:
        a.check("...which is deterministic, so an identical rerun overwrites",
                lambda: run_cli(["src/generate/generate.py", "-meta",
                                 sc.recipe("samples_perscale.json")]).stdout
                .count("run_dir") == 1
                and len(list((sc.data).glob("*"))) < 50)
    cli_ok(a, "the allocation table and its provenance are printed",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_neyman.json"),
            "--tag", "cli_neyman"], stdout_has="allocation rule='neyman'")
    cli_ok(a, "a starved allocation warns on stderr and names three fixes",
           ["src/generate/generate.py", "-meta", sc.recipe("samples_starved.json"),
            "--tag", "cli_starved"], stderr_has="n < 2")
    cli_ok(a, "a cost_probe recipe handed to the sampler fails immediately",
           ["src/generate/generate.py", "-meta", sc.recipe("cost_srw.json"),
            "--tag", "wrong"], expect_code=1, stderr_has="cost_probe recipe")
    cli_ok(a, "a missing recipe is an error, not a traceback about JSON",
           ["src/generate/generate.py", "-meta", str(sc.recipes / "nope.json")],
           expect_code=1)

    import src.generate.generate as gen_mod
    a.check("the unreachable DESIGN_INPUTS branch is gone, not merely unused",
            lambda: not hasattr(gen_mod, "DESIGN_INPUTS"),
            detail="_design_input's SystemExit could never fire: both keys it is "
                   "called with are in UNINFORMED")


def sec_estimate(a: Audit, sc: Scratch) -> None:
    """src/estimate/ -- measure_cost.py (d), estimate_omega1.py (omega1, a1), compare_observables.py"""
    a.begin("src/estimate/",
            "measure_cost.py (d), estimate_omega1.py (omega1, a1), compare_observables.py")

    p = cli_ok(a, "measure_cost.py times a ladder and fits both cost models",
               ["src/estimate/measure_cost.py", "-meta", sc.recipe("cost_srw.json"),
                "--tag", "cost_cli"], stdout_has="affine",
               creates=sc.run("cost_cli") / "cost_probe.json")
    if p:
        a.check("...cross-checks the declared cost_hint against the clock",
                lambda: "declared cost vs the wall clock" in p.stdout)
        a.check("...and reports a PASS against srw's declared d = 1",
                lambda: "PASS: measured d" in p.stdout,
                detail=[l.strip() for l in p.stdout.splitlines()
                        if l.startswith(("PASS", "FAIL"))])
    cli_ok(a, "--tag and --out-dir place the probe where asked",
           ["src/estimate/measure_cost.py", "-meta", sc.recipe("cost_minmax.json"),
            "--out-dir", str(sc.data / "probes"), "--tag", "minagg"],
           creates=sc.data / "probes" / "minagg" / "cost_probe.json")
    a.check("the recipe's own aggregator key is honoured",
            lambda: json.loads((sc.data / "probes" / "minagg" /
                                "cost_probe.json").read_text())["aggregator"],
            expect="min")
    ps = cli_ok(a, "a model with constant cost is measured as d ~ 0",
                ["src/estimate/measure_cost.py", "-meta", sc.recipe("cost_synth.json"),
                 "--tag", "cost_synth"])
    if ps:
        a.check("...and PASSES, because the verdict is scored against the model's "
                "OWN declared d rather than srw's",
                lambda: "PASS" in ps.stdout and "declared 0.0000" in ps.stdout,
                detail=[l.strip() for l in ps.stdout.splitlines()
                        if l.startswith(("PASS", "FAIL"))])
    cli_ok(a, "a samples recipe handed to the cost probe fails immediately",
           ["src/estimate/measure_cost.py", "-meta", sc.recipe("samples_srw.json"),
            "--tag", "wrong"], expect_code=1, stderr_has="samples recipe")

    # estimate_omega1 needs a run on disk; make one big enough to fit 4 params.
    run_cli(["src/generate/generate.py", "-meta", sc.recipe("samples_synth.json"),
             "--tag", "for_omega1"])
    p = cli_ok(a, "estimate_omega1.py fits eq. (232) on a saved run",
               ["src/estimate/estimate_omega1.py", "-data", str(sc.run("for_omega1"))],
               stdout_has="direct fit of eq. (232)",
               creates=sc.run("for_omega1") / "omega1.json")
    if p:
        a.check("...and reports the second, independent bias-decay estimator too",
                lambda: "bias-decay fit" in p.stdout)
    p = cli_ok(a, "--expect-omega1/--expect-gamma print a PASS/FAIL against truth",
               ["src/estimate/estimate_omega1.py", "-data", str(sc.run("for_omega1")),
                "--expect-omega1", "1.0", "--expect-gamma", "0.5", "--tol", "0.5"],
               stdout_has="against known values")
    if p:
        a.check("...marked reporting-only, so nothing is fed back to the estimator",
                lambda: "never saw these" in p.stdout)
    cli_ok(a, "--tol widens the check",
           ["src/estimate/estimate_omega1.py", "-data", str(sc.run("for_omega1")),
            "--expect-gamma", "0.5", "--tol", "0.99"], stdout_has="PASS  gamma")
    cli_ok(a, "pointing it at a directory with no samples is a clear error",
           ["src/estimate/estimate_omega1.py", "-data", str(sc.data / "nothing")],
           expect_code=1, stderr_has="no data at")

    # --- compare_observables.py: two arms, one budget ---
    from src.estimate.compare_observables import _gamma_at_m0, _parse_arm, _score

    a.close("_gamma_at_m0(m0=0) is the all-points slope on a pure power law",
            lambda: _gamma_at_m0([8, 16, 32, 64, 128],
                                 np.array([float(i) ** 1.5 for i in
                                           (8, 16, 32, 64, 128)]), 0), 1.5, 1e-9)
    a.check("_gamma_at_m0(m0=2) ignores the two smallest scales entirely",
            lambda: abs(_gamma_at_m0(
                [8, 16, 32, 64, 128],
                np.array([float(i) ** 1.5 * (1.3 if i <= 16 else 1.0)
                          for i in (8, 16, 32, 64, 128)]), 2) - 1.5) < 1e-9)
    a.raises("...and refuses an m0 that would leave fewer than 2 scales", ValueError,
             lambda: _gamma_at_m0([8, 16, 32],
                                  np.array([1.0, 2.0, 3.0]), 2),
             contains="drops exactly")
    a.check("_score flags an arm whose bias exceeds its spread as BIAS-DOMINATED",
            lambda: _score([1.0, 1.0, 1.0, 1.0], 0.5)["bias_dominated"])
    a.check("...and does not flag a centred, noisy one",
            lambda: not _score([0.4, 0.6, 0.4, 0.6], 0.5)["bias_dominated"])
    a.check("_score with no truth reports spread only, no bias/rmse",
            lambda: sorted(_score([1.0, 2.0], None)),
            expect=["mean", "sd", "se_of_mean"])
    a.raises("--arm without NAME=PATH is refused", Exception,
             lambda: _parse_arm("recipes/samples_x.json"), contains="NAME=PATH")

    p = cli_ok(a, "compare_observables.py runs both percolation anchors head to head",
               ["src/estimate/compare_observables.py",
                "--arm", "south=" + sc.recipe("samples_perc_south.json"),
                "--arm", "origin=" + sc.recipe("samples_perc_origin.json"),
                "--replicates", "3", "--m0", "1",
                "--truth", "1.8958333333333333", "--seed", "5",
                "--tag", "cmp_cli"],
               stdout_has="head to head",
               creates=sc.run("cmp_cli") / "observable_comparison.json")
    if p:
        a.check("...and reports each arm's per-scale cv, which is the Assumption 6 check",
                lambda: "cv(Y_i) per scale" in p.stdout)
        a.check("...the side-connected arm wins on RMSE, as ground rule 7 predicts",
                lambda: "south wins" in p.stdout,
                detail=[l.strip() for l in p.stdout.splitlines() if "wins by" in l])
    cli_ok(a, "one arm is allowed, and then there is no head-to-head to print",
           ["src/estimate/compare_observables.py",
            "--arm", "south=" + sc.recipe("samples_perc_south.json"),
            "--replicates", "2", "--seed", "6", "--tag", "cmp_one"],
           stdout_has="arm 'south'",
           creates=sc.run("cmp_one") / "observable_comparison.json")
    cli_ok(a, "a cost-probe recipe handed to it fails immediately",
           ["src/estimate/compare_observables.py",
            "--arm", "x=" + sc.recipe("cost_perc.json"),
            "--replicates", "2", "--tag", "cmp_wrong"],
           expect_code=1, stderr_has="cost_probe recipe")


def sec_budget(a: Audit, sc: Scratch) -> None:
    """src/budget/ -- the sweep (Experiment C) and the planning table"""
    from src.budget.allocation_table import (build_budget_rows, build_rows,
                                             budget_for_m0, choose_group,
                                             discover_groups,
                                             format_groups, human_time,
                                             input_sensitivity, measured_a1,
                                             measured_correction, measured_cost_exponent,
                                             measured_cv, measured_throughput,
                                             offset_uncertainty, predicted_rate)
    from src.budget.allocation_experiment import summarize, sweep

    a.begin("src/budget/", "the sweep (Experiment C) and the planning table")

    # --- allocation_table's own helpers ---
    a.check("human_time keeps microseconds readable",
            lambda: human_time(3e-6), expect="3 us")
    a.check("...milliseconds", lambda: human_time(0.05), expect="50 ms")
    a.check("...seconds", lambda: human_time(45.0), expect="45.0 s")
    a.check("...minutes", lambda: human_time(7200.0), expect="120.0 min")
    a.check("...hours, once minutes would run past 1000",
            lambda: human_time(60000.0), expect="16.7 h")
    a.check("...and its docstring now states that rule rather than the opposite",
            lambda: "SMALLEST unit" in human_time.__doc__,
            detail="'--time 2h' still echoes back as '120.0 min'; the behaviour is "
                   "pinned by test_human_time_units and is now documented")
    a.check("...and years, capped rather than run off the page",
            lambda: "yr" in human_time(1e18))

    kw = dict(d=1.0, omega1=1.0, rho=2.0, m=6, a1=-0.25, cv=0.7741)
    a.check("budget_for_m0 inverts the tuned m0(B) rule",
            lambda: budget_for_m0(5, **kw) > budget_for_m0(4, **kw))
    from tools.allocation import tuned_allocation
    B5 = budget_for_m0(5, **kw)
    a.close("...so planning at that budget returns the m0 asked for",
            lambda: tuned_allocation(B5, 1.0, 1.0, 2.0, 6, a1=-0.25,
                                     cv=0.7741)["m0"], 5.0, 0.5)

    sens = input_sensitivity("omega1", 0.1, **kw)
    a.check("input_sensitivity reports the m0 shift and its RMSE cost",
            lambda: sens["penalty"] >= 1.0 and "delta_m0" in sens)
    a.check("a zero se moves nothing",
            lambda: input_sensitivity("omega1", 0.0, **kw)["delta_m0"], expect=0.0)
    a.check("omega1 moves the plan ~3x more than a1 at equal RELATIVE error, as "
            "the docstring claims",
            lambda: abs(input_sensitivity("omega1", 0.25, **kw)["delta_m0"])
            > 2 * abs(input_sensitivity("a1", 0.0625, **kw)["delta_m0"]),
             detail=f"omega1 +/-25%: "
                    f"{input_sensitivity('omega1', 0.25, **kw)['delta_m0']:+.3f} "
                    f"steps; a1 +/-25%: "
                    f"{input_sensitivity('a1', 0.0625, **kw)['delta_m0']:+.3f}")
    a.check("d barely moves it at all",
            lambda: abs(input_sensitivity("d", 0.1, **kw)["delta_m0"]) < 0.1)
    a.check("cv is a real branch now, not a silent no-op",
            lambda: abs(input_sensitivity("cv", 0.1, **kw)["delta_m0"]) > 0,
            detail=f"+/-0.1 on cv moves m0 by "
                   f"{input_sensitivity('cv', 0.1, **kw)['delta_m0']:+.3f} steps")
    a.raises("...and an input it does not know raises rather than answering "
             "'this does not matter'", ValueError,
             lambda: input_sensitivity("throughput", 0.1, **kw),
             contains="does not know")
    a.check("offset_uncertainty is None without an a1 stderr",
            lambda: offset_uncertainty(1.0, 1.0, 2.0, 6, -0.25, 0.7741, None) is None)
    ou = offset_uncertainty(1.0, 1.0, 2.0, 6, -0.25, 0.7741, 0.05)
    a.check("...and otherwise reports the offset, its spread and the penalty",
            lambda: ou["penalty"] >= 1.0 and ou["spread"] > 0)

    pr = predicted_rate(1.0, 1.0, omega1_se=0.1, d_se=0.01)
    a.close("predicted_rate is -omega1/(d + 2 omega1)",
            lambda: pr["theta"], -1 / 3, 1e-15)
    a.check("...with the two inputs' contributions separated",
            lambda: sorted(pr["contributions"]), expect=["d", "omega1"])
    a.check("...and no se when neither input has one",
            lambda: predicted_rate(1.0, 1.0)["se"] is None)

    rows = build_rows(range(0, 8), throughput=1e8, **kw)
    a.check("build_rows gives one row per affordable m0",
            lambda: len(rows) > 0 and all("rmse" in r for r in rows))
    a.check("...with RMSE falling as m0 (and so the budget) grows",
            lambda: rows[-1]["rmse"] < rows[0]["rmse"])
    cmp_rows = build_rows(range(0, 5), throughput=1e8, compare=True, **kw)
    a.check("--compare adds prop:opt's own m0 and what it costs",
            lambda: any("penalty" in r for r in cmp_rows))
    brows = build_budget_rows([1e6, 1e9, 1e12], throughput=1e8, **kw)
    a.check("build_budget_rows indexes by budget instead",
            lambda: len(brows), expect=3)
    a.check("...and skips a budget too small for any integer allocation",
            lambda: len(build_budget_rows([1.0], throughput=1e8, **kw)), expect=0)

    # --- discovery against the real experiment, read-only ---
    real = ROOT / "experiments" / "01_srw" / "data"
    if real.is_dir():
        groups = discover_groups(real)
        a.check("discover_groups finds this repo's own Experiment B runs",
                lambda: len(groups) > 0, detail=f"{len(groups)} group(s)")
        a.check("...sorted with the most-replicated first",
                lambda: groups[0]["replicates"] >= groups[-1]["replicates"])
        a.check("...each carrying model, scales, n and a name",
                lambda: all(k in groups[0] for k in
                            ("model", "scales", "n", "name", "runs")))
        a.check("format_groups renders a table",
                lambda: "group" in format_groups(groups).splitlines()[0])
        a.check("choose_group picks by name",
                lambda: choose_group(groups, requested=groups[0]["name"],
                                     interactive=False)["name"],
                expect=groups[0]["name"])
        a.raises("...and refuses an unknown name, listing what exists", SystemExit,
                 lambda: choose_group(groups, requested="nope", interactive=False),
                 contains="no run group named")
        a.check("choose_group never blocks on input when interactive=False",
                lambda: choose_group(groups, requested=None,
                                     interactive=False) is not None)
        a.check("the default choice is the biggest group, which is what every "
                "driver takes",
                lambda: choose_group(groups, requested=None,
                                     interactive=False)["runs"]
                == groups[0]["runs"])
        corr = measured_correction(groups[0]["runs"])
        a.check("measured_correction returns a1 and omega1 with provenance",
                lambda: corr["a1"] is not None and "measured" in corr["provenance"],
                detail=f"omega1 = {corr['omega1']}, {corr['provenance']}")
        a.check("measured_a1 is the a1-only view of the same call",
                lambda: measured_a1(groups[0]["runs"])[0] == corr["a1"])
        a.check("pooling is used when every replicate kept its y_bar",
                lambda: "POOLED" in corr["provenance"] or corr["replicates"] == 1,
                detail=corr["provenance"])
        d_val, d_se, d_src = measured_cost_exponent(real)
        a.check("measured_cost_exponent reads the AFFINE d off the cost probes",
                lambda: d_val is None or 0.5 < d_val < 1.5,
                detail=f"d = {d_val} ({d_src})")
    a.check("measured_correction on nothing returns None with a reason, not a crash",
            lambda: measured_correction([])["provenance"], expect="no omega1.json found")
    a.check("measured_cv on a directory with no samples says so",
            lambda: measured_cv(sc.data / "nothing")[1], expect="no samples found")
    a.check("measured_throughput on a missing file says so",
            lambda: measured_throughput(sc.data / "nope.json")[2],
            expect="no allocation_sweep.json found")
    a.check("measured_cost_exponent on an empty root says so",
            lambda: measured_cost_exponent(sc.data / "nothing")[2],
            expect="no cost_probe.json found")

    # --- allocation_table's CLI ---
    empty = sc.data / "empty_root"
    empty.mkdir(exist_ok=True)
    p = cli_ok(a, "with nothing measured the table refuses, naming the flag and "
                  "the command",
               ["src/budget/allocation_table.py", "--data-root", str(empty),
                "--no-prompt"], expect_code=1, stderr_has="--d")
    if p:
        a.check("...and says how to measure it rather than substituting a default",
                lambda: "measure_cost.py" in p.stderr)
    over = ["--d", "1.0", "--omega1", "1.0", "--a1", "-0.25", "--cv", "0.7741",
            "--throughput", "1e8"]
    p = cli_ok(a, "hand-supplied constants run the table",
               ["src/budget/allocation_table.py", "--data-root", str(empty),
                "--no-prompt", *over], stdout_has="derived constants")
    if p:
        a.check("...every one of them marked NOT MEASURED",
                lambda: p.stdout.count("NOT MEASURED"), expect=6)
    cli_ok(a, "--by-budget switches to the view that stays put across runs",
           ["src/budget/allocation_table.py", "--data-root", str(empty),
            "--no-prompt", "--by-budget", "--min-log10-budget", "6",
            "--max-log10-budget", "10", *over], stdout_has="Indexed by budget")
    cli_ok(a, "--compare adds prop:opt's untuned column",
           ["src/budget/allocation_table.py", "--data-root", str(empty),
            "--no-prompt", "--compare", "--max-m0", "6", *over],
           stdout_has="prop:opt m0")
    csv_path = sc.data / "table.csv"
    cli_ok(a, "--csv writes the table out",
           ["src/budget/allocation_table.py", "--data-root", str(empty),
            "--no-prompt", "--csv", str(csv_path), "--max-m0", "4", *over],
           creates=csv_path)
    cli_ok(a, "--rho and --m are honoured",
           ["src/budget/allocation_table.py", "--data-root", str(empty),
            "--no-prompt", "--rho", "3", "--m", "5", "--max-m0", "4", *over],
           stdout_has="rho            3.0000")
    cli_ok(a, "--min-m0/--max-m0 bound the rows",
           ["src/budget/allocation_table.py", "--data-root", str(empty),
            "--no-prompt", "--min-m0", "2", "--max-m0", "3", *over])
    cli_ok(a, "--list on an empty root says there is nothing, and exits 0",
           ["src/budget/allocation_table.py", "--data-root", str(empty), "--list"],
           stdout_has="no Experiment B runs")
    if real.is_dir():
        cli_ok(a, "--list on the real data root shows the run groups",
               ["src/budget/allocation_table.py", "--data-root", str(real), "--list"],
               stdout_has="run groups under")
        cli_ok(a, "...and the table builds from measured constants alone",
               ["src/budget/allocation_table.py", "--data-root", str(real),
                "--no-prompt", "--max-m0", "6"], stdout_has="inputs")
        cli_ok(a, "--group selects one of them",
               ["src/budget/allocation_table.py", "--data-root", str(real),
                "--no-prompt", "--group", discover_groups(real)[0]["name"],
                "--max-m0", "4"])
        cli_ok(a, "an unknown --group is refused with the list",
               ["src/budget/allocation_table.py", "--data-root", str(real),
                "--no-prompt", "--group", "nope"], expect_code=1,
               stderr_has="no run group named")

    # --- the sweep itself, tiny ---
    res = quiet(sweep, "srw", {"q": 0.5}, [1e5, 1e6], [0, 1], m=4, rho=2.0, d=1.0,
                omega1=1.0, replicates=3, true_gamma=0.5, seed=7, a1=-0.25, cv=0.7741)
    a.check("sweep returns one cell per (budget, m0)",
            lambda: len(res["cells"]), expect=4)
    a.check("...each with both estimators scored against the planted truth",
            lambda: all("closed_form" in c and "all_points" in c
                        for c in res["cells"] if not c["skipped"]))
    a.check("...and every cell says whether it is prop:opt's or the tuned m0",
            lambda: all(isinstance(c.get("is_prop_opt_m0"), bool)
                        for c in res["cells"] if not c["skipped"]))
    a.check("...marking exactly the cells whose m0 the rule names",
            lambda: all(c["is_prop_opt_m0"] ==
                        (c["m0"] == res["prop_opt"][str(c["budget"])]["m0"])
                        for c in res["cells"] if not c["skipped"]))
    a.check("the tuned arm runs only when a1 and cv are given",
            lambda: res["tuned"] is not None
            and quiet(sweep, "srw", {"q": 0.5}, [1e5], [0], m=4, rho=2.0, d=1.0,
                      omega1=1.0, replicates=2, true_gamma=0.5,
                      seed=7)["tuned"] is None)
    s = summarize(res)
    a.check("summarize scores each budget against the empirical argmin",
            lambda: all("best_m0" in v and "penalty_factor" in v for v in s.values()))
    a.check("...for the all_points estimator too",
            lambda: len(summarize(res, "all_points")), expect=2)

    p = cli_ok(a, "allocation_experiment.py runs a sweep recipe end to end",
               ["src/budget/allocation_experiment.py", "-meta",
                sc.recipe("sweep_tiny.json"), "--tag", "sweep_cli"],
               stdout_has="error-decay rate",
               creates=sc.run("sweep_cli") / "allocation_sweep.json", timeout=900)
    if p:
        a.check("...printing the paired prop:opt / tuned arms side by side",
                lambda: "paired allocation arms" in p.stdout)
    cli_ok(a, "a samples recipe handed to the sweep fails immediately",
           ["src/budget/allocation_experiment.py", "-meta",
            sc.recipe("samples_srw.json")], expect_code=1, stderr_has="samples recipe")


def sec_report(a: Audit, sc: Scratch) -> None:
    """src/report/ -- the three plotters"""
    a.begin("src/report/", "the three plotters")

    run_cli(["src/generate/generate.py", "-meta", sc.recipe("samples_synth.json"),
             "--tag", "plotme"])
    p = cli_ok(a, "plot_loglog.py plots a run and writes all four estimators",
               ["src/report/plot_loglog.py", "-data", str(sc.run("plotme"))],
               creates=sc.run("plotme") / "plot.png")
    if p:
        a.check("...including gamma_estimates.json, whatever the model",
                lambda: (sc.run("plotme") / "gamma_estimates.json").exists())
        a.check("...and overlays the known E Y_i for a model that has a target_fn",
                lambda: "true_gamma=0.5" in p.stdout)
    cli_ok(a, "--estimates adds the comparison chart, opt-in",
           ["src/report/plot_loglog.py", "-data", str(sc.run("plotme")),
            "--estimates"], creates=sc.run("plotme") / "estimates.png")
    cli_ok(a, "-o redirects the figure",
           ["src/report/plot_loglog.py", "-data", str(sc.run("plotme")),
            "-o", str(sc.data / "elsewhere.png")], creates=sc.data / "elsewhere.png")

    run_cli(["src/generate/generate.py", "-meta", sc.recipe("samples_srw.json"),
             "--tag", "plotsrw"])
    p = cli_ok(a, "a model with no known truth still plots and still fits",
               ["src/report/plot_loglog.py", "-data", str(sc.run("plotsrw"))],
               stdout_has="exploratory")
    nometa = sc.data / "nometa"
    nometa.mkdir(exist_ok=True)
    shutil.copy(sc.run("plotsrw") / "samples.npz", nometa / "samples.npz")
    cli_ok(a, "metadata is optional -- a run without it plots data only",
           ["src/report/plot_loglog.py", "-data", str(nometa)],
           stdout_has="no metadata")
    cli_ok(a, "a directory with no samples is refused by name",
           ["src/report/plot_loglog.py", "-data", str(sc.data / "nothing")],
           expect_code=2, stderr_has="no data at")

    run_cli(["src/estimate/measure_cost.py", "-meta", sc.recipe("cost_srw.json"),
             "--tag", "costplot"])
    cli_ok(a, "plot_cost.py plots a cost probe",
           ["src/report/plot_cost.py", "-data", str(sc.run("costplot"))],
           creates=sc.run("costplot") / "plot.png")
    cli_ok(a, "...and refuses a directory holding no probe",
           ["src/report/plot_cost.py", "-data", str(sc.run("plotme"))],
           expect_code=2, stderr_has="no data at")

    sweep_dir = sc.run("sweep_cli")
    if not (sweep_dir / "allocation_sweep.json").exists():
        run_cli(["src/budget/allocation_experiment.py", "-meta",
                 sc.recipe("sweep_tiny.json"), "--tag", "sweep_cli"], timeout=900)
    if (sweep_dir / "allocation_sweep.json").exists():
        p = cli_ok(a, "plot_allocation.py draws Experiment C's two panels",
                   ["src/report/plot_allocation.py", "-data", str(sweep_dir),
                    "--no-expected"], creates=sweep_dir / "plot.png")
        if p:
            a.check("...and prints the consistency cut-off as a t, not 1.960",
                    lambda: "consistency cut-off" in p.stdout)
        cli_ok(a, "--estimator all_points switches which gamma-hat is scored",
               ["src/report/plot_allocation.py", "-data", str(sweep_dir),
                "--estimator", "all_points", "--no-expected",
                "-o", str(sc.data / "alloc_ap.png")], creates=sc.data / "alloc_ap.png")


def sec_study(a: Audit, sc: Scratch) -> None:
    """src/study/ -- the four-step workflow: pilot -> plan -> run -> report"""
    from src.study.plan import parse_duration, unidentified
    from tools.constants import measured

    a.begin("src/study/", "the four-step workflow: pilot -> plan -> run -> report")

    a.check("parse_duration reads hours", lambda: parse_duration("2h"), expect=7200.0)
    a.check("...minutes", lambda: parse_duration("90m"), expect=5400.0)
    a.check("...seconds", lambda: parse_duration("45s"), expect=45.0)
    a.check("...days", lambda: parse_duration("1d"), expect=86400.0)
    a.check("...fractions", lambda: parse_duration("1.5h"), expect=5400.0)
    a.check("...and a bare number as seconds", lambda: parse_duration("3600"),
            expect=3600.0)
    a.raises("an unreadable duration says what to try", ValueError,
             lambda: parse_duration("soon"), contains="try 2h")
    a.check("unidentified() is true when the error bar is as wide as the value",
            lambda: unidentified(measured(1.0, 1.2, "x")))
    a.check("...false when it is not", lambda: unidentified(measured(1.0, 0.1, "x")),
            expect=False)
    a.check("...and false when there is no se to judge",
            lambda: unidentified(measured(1.0, None, "x")), expect=False)

    study = "audit_study"
    sd = sc.data / study
    p = cli_ok(a, "pilot.py measures the constants from a cheap run",
               ["src/study/pilot.py", "-meta", sc.recipe("samples_pilot.json"),
                "--study", study, "--replicates", "2", "--seed", "5"],
               stdout_has="constants measured", creates=sd / "constants.json",
               timeout=900)
    if p:
        a.check("...writing pilot.json beside them",
                lambda: (sd / "pilot.json").exists())
        a.check("...measuring d from the CLOCK, not from srw's declared cost_hint",
                lambda: "pilot cost probe" in
                json.loads((sd / "constants.json").read_text())["d"]["source"],
                detail=json.loads((sd / "constants.json").read_text())["d"]["source"])
        a.check("...and scoring the declaration against it",
                lambda: (json.loads((sd / "pilot.json").read_text())["cost"]
                         ["d_check"]["verdict"]) is not None,
                detail=str(json.loads((sd / "pilot.json").read_text())["cost"]
                           ["d_check"]))
        a.check("...recording a throughput for this machine",
                lambda: json.loads((sd / "pilot.json").read_text())["throughput"] > 0)
    p = cli_ok(a, "--more extends an existing pilot without redrawing it",
               ["src/study/pilot.py", "--study", study, "--more", "2",
                "--data-root", str(sc.data)], stdout_has="adding 2 replicate",
               timeout=900)
    if p:
        old = json.loads((sd / "pilot.json").read_text())
        a.check("...so the pool grows to 4", lambda: old["replicates"], expect=4)
        reps = old["per_replicate"]
        a.check("...with four DISTINCT replicates, not two of them copied "
                "(the sqrt(3) bug tools/rng.py:spawn(skip=) exists to stop)",
                lambda: len({tuple(r["y_bar"]) for r in reps}), expect=4)
    cli_ok(a, "--more without a pilot to add to says so",
           ["src/study/pilot.py", "--study", "nosuch", "--more", "2",
            "--data-root", str(sc.data)], expect_code=1, stderr_has="run without --more")
    cli_ok(a, "no -meta and no --more is refused",
           ["src/study/pilot.py", "--study", "x", "--data-root", str(sc.data)],
           expect_code=1, stderr_has="-meta is required")
    p = cli_ok(a, "--assert-d scores a stated d against the clock",
               ["src/study/pilot.py", "-meta", sc.recipe("samples_pilot.json"),
                "--study", "audit_assert", "--replicates", "1", "--seed", "6",
                "--assert-d", "3.0"], timeout=900)
    if p:
        a.check("...and shouts when they disagree, without stopping the run",
                lambda: "MISMATCH" in (p.stdout + p.stderr))
    p = cli_ok(a, "--trust-declared-d skips the clock and stamps the result",
               ["src/study/pilot.py", "-meta", sc.recipe("samples_pilot.json"),
                "--study", "audit_trust", "--replicates", "1", "--seed", "7",
                "--trust-declared-d"], timeout=900)
    if p:
        a.check("...printing d as NOT MEASURED so it cannot pass for a measurement",
                lambda: "NOT MEASURED" in p.stdout)
        a.check("...and recording it as a user override",
                lambda: json.loads(
                    (sc.data / "audit_trust" / "constants.json").read_text()
                )["d"]["source"].startswith("user override"))

    if (sd / "constants.json").exists():
        p = cli_ok(a, "plan.py proposes at a wall clock and draws nothing",
                   ["src/study/plan.py", "--study", study, "--data-root",
                    str(sc.data), "--time", "20s"],
                   stdout_has="proposed allocation")
        if p:
            a.check("...saying explicitly that nothing was written",
                    lambda: "nothing was drawn" in p.stdout)
            a.check("...and printing a verdict on the pilot it is planning from",
                    lambda: "VERDICT" in p.stdout)
            a.check("...with the error budget per constant",
                    lambda: "error budget" in p.stdout)
        a.check("...and really wrote no plan.json",
                lambda: not (sd / "plan.json").exists())
        cli_ok(a, "--target-se inverts the question",
               ["src/study/plan.py", "--study", study, "--data-root", str(sc.data),
                "--target-se", "1e-2"], stdout_has="to reach se(gamma)")
        cli_ok(a, "--time and --target-se are mutually exclusive",
               ["src/study/plan.py", "--study", study, "--data-root", str(sc.data),
                "--time", "20s", "--target-se", "1e-2"], expect_code=2)
        p = cli_ok(a, "--accept writes plan.json AND the final recipe",
                   ["src/study/plan.py", "--study", study, "--data-root", str(sc.data),
                    "--time", "20s", "--replicates", "3", "--accept"],
                   creates=sd / "plan.json")
        if p:
            plan = json.loads((sd / "plan.json").read_text())
            a.check("...the recipe being an ordinary samples recipe, runnable alone",
                    lambda: Path(plan["recipe_path"]).exists())
            a.check("...carrying the constants it was accepted on",
                    lambda: sorted(plan["constants_at_plan_time"]),
                    expect=["a1", "cv", "d", "omega1"])
            a.check("--time is the TOTAL: the per-replicate budget is time/R",
                    lambda: abs(plan["total_seconds"] - 3 * plan["seconds"]) < 1e-6)
    cli_ok(a, "plan.py without a pilot names the command that makes one",
           ["src/study/plan.py", "--study", "nosuch", "--data-root", str(sc.data),
            "--time", "10s"], expect_code=1, stderr_has="pilot.py")

    if (sd / "plan.json").exists():
        p = cli_ok(a, "run.py --dry-run prints the plan and draws nothing",
                   ["src/study/run.py", "--study", study, "--data-root", str(sc.data),
                    "--dry-run"], stdout_has="nothing drawn")
        a.check("...leaving no final.json", lambda: not (sd / "final.json").exists())
        cli_ok(a, "run.py executes the accepted plan",
               ["src/study/run.py", "--study", study, "--data-root", str(sc.data),
                "--seed", "3"], creates=sd / "final.json", timeout=1800)
        if (sd / "final.json").exists():
            fin = json.loads((sd / "final.json").read_text())
            a.check("...keeping only the summaries by default",
                    lambda: fin["samples_kept"], expect=False)
            a.check("...one per replicate, with all four summary fields",
                    lambda: sorted(fin["per_replicate"][0]),
                    expect=["cv", "log_moment", "sigma_log", "y_bar"])
            a.check("...and recording every replicate's seed, so any one can be redrawn",
                    lambda: len(fin["seeds"]), expect=fin["replicates"])
        cli_ok(a, "--keep-samples persists the draws as well",
               ["src/study/run.py", "--study", study, "--data-root", str(sc.data),
                "--seed", "3", "--keep-samples"],
               creates=sd / "samples" / "rep0" / "samples.npz", timeout=1800)
    cli_ok(a, "run.py without an accepted plan names the command that accepts one",
           ["src/study/run.py", "--study", "nosuch", "--data-root", str(sc.data)],
           expect_code=1, stderr_has="plan.py")

    if (sd / "final.json").exists():
        p = cli_ok(a, "report.py answers with gamma and its interval",
                   ["src/study/report.py", "--study", study, "--data-root",
                    str(sc.data)], stdout_has="gamma =", creates=sd / "report.md")
        if p:
            a.check("...writing details.md and answer.json alongside",
                    lambda: (sd / "details.md").exists()
                    and (sd / "answer.json").exists())
            a.check("...and the log-log plot", lambda: (sd / "plot.png").exists())
            ans = json.loads((sd / "answer.json").read_text())
            a.check("...leading with the eq. (720) bound when it can be assembled",
                    lambda: ans.get("wilson") is not None or ans.get("wilson_why"),
                    detail=f"wilson={'yes' if ans.get('wilson') else 'no'}, "
                           f"why={ans.get('wilson_why')}")
            a.check("...and using the t quantile for the replicate interval",
                    lambda: ans["dof"], expect=ans["replicates"] - 1)
        cli_ok(a, "--budget-analysis compares predicted against actual",
               ["src/study/report.py", "--study", study, "--data-root", str(sc.data),
                "--budget-analysis"], creates=sd / "budget_analysis.md")
        cli_ok(a, "--level changes the interval's level",
               ["src/study/report.py", "--study", study, "--data-root", str(sc.data),
                "--level", "0.99"], stdout_has="99%")
    cli_ok(a, "report.py without a run names the command that makes one",
           ["src/study/report.py", "--study", "nosuch", "--data-root", str(sc.data)],
           expect_code=1, stderr_has="run.py")

    # --- autopilot: all four, with the gates in between ---
    p = run_cli(["src/study/autopilot.py", "-meta", sc.recipe("samples_pilot.json"),
                 "--study", "audit_auto", "--data-root", str(sc.data),
                 "--time", "45s", "--replicates", "2", "--seed", "8",
                 "--no-progress"], timeout=1800)
    gave_up = p.returncode == 1
    a.check("autopilot.py runs the whole workflow and exits 0 (ran) or 1 (gave up)",
            lambda: p.returncode in (0, 1),
            detail=f"exit {p.returncode}; " +
                   ("gave up on the gates" if gave_up else "completed"))
    a.check("...writing autopilot.json either way",
            lambda: (sc.data / "audit_auto" / "autopilot.json").exists())
    a.check("...and saying which gate decided",
            lambda: "B_fs" in p.stdout or "pure power law" in p.stdout
            or "gamma =" in p.stdout)
    if gave_up:
        a.check("giving up prints the constants it did measure, as evidence",
                lambda: "measured so far" in p.stdout)
        a.check("...and draws nothing",
                lambda: not (sc.data / "audit_auto" / "final.json").exists())

    # --force, on a run rigged to fail the gates: one replicate gives omega1 no
    # standard error at all, so the B_fs span cannot even be formed, and
    # --max-rounds 1 stops the doubling loop before it can fix that.
    rigged = ["--time", "30s", "--replicates", "1", "--max-rounds", "1",
              "--seed", "9", "--no-progress"]
    p_no = run_cli(["src/study/autopilot.py", "-meta", sc.recipe("samples_pilot.json"),
                    "--study", "audit_gate", "--data-root", str(sc.data), *rigged],
                   timeout=1800)
    a.check("a one-replicate pilot fails the gate and draws nothing",
            lambda: p_no.returncode == 1
            and not (sc.data / "audit_gate" / "final.json").exists(),
            detail=f"exit {p_no.returncode}")
    p_f = run_cli(["src/study/autopilot.py", "-meta", sc.recipe("samples_pilot.json"),
                   "--study", "audit_force", "--data-root", str(sc.data),
                   "--force", *rigged], timeout=1800)
    forced = sc.data / "audit_force"
    a.check("--force runs the same study anyway, to completion",
            lambda: p_f.returncode == 0 and (forced / "final.json").exists()
            and (forced / "report.md").exists(),
            detail=f"exit {p_f.returncode}")
    a.check("...printing the SAME diagnosis, since the evidence does not depend "
            "on the decision",
            lambda: "PILOT DID NOT DETERMINE THE CONSTANTS" in p_f.stdout
            and "measured so far" in p_f.stdout)
    a.check("...and stamping the answer `forced` so it can never later pass for "
            "a determined one",
            lambda: json.loads((forced / "autopilot.json").read_text())["forced"]
            is True)
    a.check("...with a closing warning on the console, not only in the file",
            lambda: "--force:" in p_f.stdout and "NOT determined" in p_f.stdout)

    a.check("the doubling loop grows the DRAWS and leaves the replicate count "
            "alone (Igor, 2026-09-04)",
            lambda: [r["factor"] for r in
                     json.loads((forced / "autopilot.json").read_text())["rounds"]]
            == [1],
            detail="one round here; the multi-round arithmetic is pinned in "
                   "tools/tests/test_autopilot.py")
    from src.study.autopilot import scale_draws
    a.check("scale_draws grows a scalar n",
            lambda: scale_draws({"scales": [8], "n": 1000}, 4)["n"], expect=4000)
    a.check("...a per-scale list",
            lambda: scale_draws({"scales": [8, 16], "n": [10, 20]}, 2)["n"],
            expect=[20, 40])
    a.check("...and a rule's budget",
            lambda: scale_draws({"scales": [8], "n": {"rule": "snr",
                                                      "budget": 1e6}}, 8)["n"],
            expect={"rule": "snr", "budget": 8e6})
    a.check("...never the ladder",
            lambda: scale_draws({"scales": [8, 16], "n": 10}, 4)["scales"],
            expect=[8, 16])
    a.raises("a rule with no budget cannot be grown, and says so", SystemExit,
             lambda: scale_draws({"scales": [8], "n": {"rule": "snr"}}, 2),
             contains="budget")
    cli_ok(a, "--pilot-cap outside (0, 1) is refused",
           ["src/study/autopilot.py", "-meta", sc.recipe("samples_pilot.json"),
            "--study", "x", "--data-root", str(sc.data), "--time", "10s",
            "--pilot-cap", "1.5"], expect_code=1, stderr_has="pilot-cap")

    import inspect

    from src.study.autopilot import pilot_until_determined
    src_loop = inspect.getsource(pilot_until_determined)
    a.check("the gate is judged at the whole remaining budget, not a hardcoded "
            "multiple of the pilot's slice",
            lambda: "total_seconds - spent" in src_loop
            and "seconds_budget * 4" not in src_loop,
            detail="`seconds_budget * 4` reconstructed the total only at the "
                   "default --pilot-cap 0.25")


# ===========================================================================
# STAGE 4 -- calibration/, the checks on the machinery. Smoke-sized here:
# the real measurements are what these scripts are FOR, and they take hours.
# ===========================================================================

def sec_calibration(a: Audit, sc: Scratch) -> None:
    """calibration/ -- are our own error bars, and our own ETAs, honest?"""
    from calibration.check_coverage import (SCALES, TRUTH, check_planting, exact_mean,
                                            exact_sd, make_experiment,
                                            make_rate_experiment,
                                            make_wilson_experiment)

    a.begin("calibration/", "are our own error bars, and our own ETAs, honest?")

    # srw's exact moments, which live here as SCORING truth and nowhere else.
    a.close("E|S_1| = 1", lambda: exact_mean(1), 1.0, 1e-15)
    a.close("E|S_2| = 1", lambda: exact_mean(2), 1.0, 1e-15)
    a.check("E|S_{2m-1}| = E|S_2m| exactly -- the staircase correction.py warns about",
            lambda: all(math.isclose(exact_mean(2 * j - 1), exact_mean(2 * j))
                        for j in range(1, 40)))
    a.close("E|S_k| ~ sqrt(2k/pi) for large k",
            lambda: exact_mean(2048) / math.sqrt(2 * 2048 / math.pi), 1.0, 1e-3)
    a.close("...with the eq. (232) correction a1 = -1/4 visible at small k",
            lambda: (math.log(exact_mean(64) / (math.sqrt(2 / math.pi) * 8.0))
                     * 64), -0.25, 0.01)
    a.close("sd|S_k| = sqrt(k - (E|S_k|)^2)",
            lambda: exact_sd(100), math.sqrt(100 - exact_mean(100) ** 2), 1e-12)
    a.close("...so cv -> sqrt(pi/2 - 1), the half-normal limit",
            lambda: exact_sd(4096) / exact_mean(4096), math.sqrt(math.pi / 2 - 1), 1e-3)
    a.raises("k < 1 is refused", ValueError, lambda: exact_mean(0), contains="k must be")
    a.check("exact_mean survives k where the naive float form overflows",
            lambda: math.isfinite(exact_mean(4000)) and exact_mean(4000) > 0)

    a.check("TRUTH holds srw's four constants, for scoring only",
            lambda: sorted(TRUTH), expect=["a0", "a1", "gamma", "omega1"])
    a.check("SCALES is Experiment B's own ladder",
            lambda: SCALES, expect=[8, 16, 32, 64, 128, 256])

    # The arms, as callables, before the CLI runs them.
    from tools.coverage import coverage_multi, coverage_test
    from calibration.check_coverage import _planted_replicate
    exp = make_experiment(_planted_replicate, [2000] * 6, replicates=3,
                          params=("omega1", "gamma"))
    got = exp(np.random.default_rng(0))
    a.check("make_experiment reports both combination rules for every parameter",
            lambda: sorted(got),
            expect=["gamma/mean", "gamma/pooled", "omega1/mean", "omega1/pooled"])
    a.check("...each an (estimate, se) pair",
            lambda: all(len(v) == 2 for v in got.values()))

    wexp = make_wilson_experiment(2, 6, 2.0, 100_000, sigma_inf2=math.pi / 2 - 1,
                                 sigma_max2=0.6, a1=TRUTH["a1"], omega1=TRUTH["omega1"])
    est, se_eq = wexp(np.random.default_rng(0))
    a.close("the wilson arm returns a gamma-hat near 1/2", lambda: est, 0.5, 0.05)
    a.check("...and an se-equivalent that reproduces the bound through interval()",
            lambda: se_eq > 0)
    rexp, truth_slope = make_rate_experiment([1e6, 1e7, 1e8], 5, 1.0, 1.0, 2.0, 6)
    a.close("the rate arm's truth is -omega1/(d + 2 omega1)",
            lambda: truth_slope, -1 / 3, 1e-15)
    slope, slope_se = rexp(np.random.default_rng(0))
    a.check("...and it returns a measured slope with its analytic se",
            lambda: math.isfinite(slope) and slope_se > 0,
            detail=f"slope {slope:+.3f} +/- {slope_se:.3f} against {truth_slope:+.3f}")

    rows = check_planting(2000, 20, seed=1, scales=[8, 16])
    a.check("check_planting KS-tests y_bar against the planted normal, per scale",
            lambda: len(rows) == 2 and all(0 <= r["ks_p"] <= 1 for r in rows))
    a.check("...comparing the observed moments against the exact ones",
            lambda: all(abs(r["observed_mean"] - r["exact_mean"]) < 5 * r["exact_se"]
                        for r in rows))

    # --- the CLI, one invocation per arm, deliberately tiny ---
    out_json = sc.data / "coverage.json"
    p = cli_ok(a, "check_coverage.py --arm planted",
               ["calibration/check_coverage.py", "--arm", "planted", "--trials", "30",
                "--replicates", "3", "--n-scale", "0.001", "--params", "omega1",
                "--json", str(out_json)], stdout_has="coverage", creates=out_json,
               timeout=900)
    if p:
        a.check("...reporting a coverage with a Wilson CI on the coverage itself",
                lambda: "95% CI" in p.stdout or "CI [" in p.stdout)
        a.check("...and re-scoring the '+/- 1 se' convention for free",
                lambda: "68.3%" in p.stdout)
    cli_ok(a, "--centre both scores pooled and mean-of-fits on identical draws",
           ["calibration/check_coverage.py", "--arm", "planted", "--trials", "20",
            "--replicates", "3", "--n-scale", "0.001", "--params", "omega1",
            "--centre", "both"], stdout_has="/mean", timeout=900)
    p = cli_ok(a, "--arm srw draws for real instead of planting",
               ["calibration/check_coverage.py", "--arm", "srw", "--srw-trials",
                "4", "--replicates", "3", "--srw-n-scale", "0.0002",
                "--params", "gamma"], stdout_has="srw arm", timeout=900)
    if p:
        a.check("...on its OWN n and trials, since the planted arm's would be days",
                lambda: "simulated steps" in p.stdout and "x0.0002" in p.stdout)
    cli_ok(a, "--arm planting KS-tests the planted arm's own Gaussian assumption",
           ["calibration/check_coverage.py", "--arm", "planting", "--trials", "30",
            "--planting-n", "2000"], stdout_has="planting arm", timeout=900)
    cli_ok(a, "--arm rate checks an ANALYTIC error bar rather than a replicate one",
           ["calibration/check_coverage.py", "--arm", "rate", "--trials", "20"],
           stdout_has="rate_exponent_se", timeout=900)
    cli_ok(a, "--arm wilson sweeps m0 for the eq. (720) bound",
           ["calibration/check_coverage.py", "--arm", "wilson", "--trials", "40",
            "--wilson-m0", "2", "4", "--wilson-n", "20000"],
           stdout_has="wilson arm", timeout=900)
    cli_ok(a, "--level changes the nominal level under test",
           ["calibration/check_coverage.py", "--arm", "wilson", "--trials", "40",
            "--wilson-m0", "2", "--wilson-n", "20000", "--level", "0.99"],
           stdout_has="99%", timeout=900)
    from calibration.check_coverage import ALL_ARMS
    p = cli_ok(a, "--arm all runs EVERY arm, cheapest first",
               ["calibration/check_coverage.py", "--arm", "all", "--trials", "20",
                "--replicates", "3", "--params", "omega1", "--planting-n", "2000",
                "--wilson-m0", "2", "--wilson-n", "20000", "--srw-trials", "3",
                "--srw-n-scale", "0.0001"], timeout=1800)
    if p:
        markers = {"planting": "planting arm", "planted": "planted arm",
                   "srw": "srw arm", "wilson": "wilson arm",
                   "rate": "rate_exponent_se"}
        missing = [arm for arm in ALL_ARMS if markers[arm] not in p.stdout]
        a.check("...all five of them, with none silently skipped",
                lambda: not missing,
                detail=f"ALL_ARMS = {ALL_ARMS}; missing from the run: {missing}")
        a.check("...including wilson, the arm that checks the interval report.py "
                "leads with",
                lambda: "wilson arm" in p.stdout)
    p = run_cli(["calibration/check_coverage.py", "--arm", "planted", "--trials", "5",
                 "--replicates", "3", "--n-scale", "0.001", "--params", "nosuchparam"],
                timeout=600)
    a.check("an unknown --params name is caught by argparse, naming the valid ones",
            lambda: p.returncode == 2 and "invalid choice" in (p.stderr or ""),
            detail=(p.stderr or "").strip().splitlines()[-1][:90]
            if p.stderr else "")

    # verify_prediction: reads constants from a data root, then times real ladders.
    empty = sc.data / "empty_root2"
    empty.mkdir(exist_ok=True)
    cli_ok(a, "verify_prediction refuses to run on constants it cannot find",
           ["calibration/verify_prediction.py", "--data-root", str(empty),
            "--m0", "3", "--replicates", "1"], expect_code=1,
           stderr_has="no measured value")
    real = ROOT / "experiments" / "01_srw" / "data"
    if real.is_dir():
        tag = "audit_prediction_check"
        p = cli_ok(a, "...and runs the tuned ladders for real when it can",
                   ["calibration/verify_prediction.py", "--data-root", str(real),
                    "--m0", "3", "4", "--replicates", "2", "--tag", tag],
                   stdout_has="timing    :", timeout=1800)
        if p:
            a.check("...reporting predicted against measured, both halves",
                    lambda: "accuracy  :" in p.stdout)
        cli_ok(a, "--max-n skips a ladder too big to attempt, rather than crashing",
               ["calibration/verify_prediction.py", "--data-root", str(real),
                "--m0", "3", "20", "--replicates", "1", "--max-n", "1000000",
                "--tag", tag], stderr_has="skipped", timeout=1800)
        shutil.rmtree(real / tag, ignore_errors=True)
        p = run_cli(["calibration/verify_prediction.py", "--data-root", str(real),
                     "--m0", "40", "--replicates", "1", "--tag", tag], timeout=900)
        a.check("an all-skipped run ends cleanly instead of dividing an empty list",
                lambda: p.returncode == 0 and "ValueError" not in (p.stderr or ""),
                detail=(p.stderr or "").strip().splitlines()[-1][:90]
                if p.stderr else "")
        a.check("...and says what to try instead",
                lambda: "is runnable at these constants" in p.stdout)
        shutil.rmtree(real / tag, ignore_errors=True)

    # --- check_no_leakage: the falsification test, and its own falsification ---
    from calibration.check_no_leakage import (
        R2_MIN, REL_TOLERANCE, SLOPE_TOLERANCE, bonferroni_z,
        exact_mean_abs_srw, judge, parse_leaks, responsiveness)

    a.close("the srw arm's reference reproduces E|S_k| at q = 1/2",
            lambda: exact_mean_abs_srw(64, 0.5),
            64 * math.comb(63, 31) * 2.0 ** -63, 1e-12)
    a.close("...and moves to (2q-1)k as the walk goes ballistic",
            lambda: exact_mean_abs_srw(256, 0.8) / 256, 0.6, 0.01)
    a.close("responsiveness recovers an exact line",
            lambda: responsiveness([1, 2, 3, 4], [3.0, 5.0, 7.0, 9.0])["slope"],
            2.0, 1e-12)
    a.check("...and refuses a grid with no spread in the truth, rather than "
            "returning a number that cannot mean anything",
            lambda: responsiveness([1, 1, 1, 1], [1, 2, 3, 4])["slope"] is None)
    a.check("criterion 2's tolerances are the ones the plan states",
            lambda: (SLOPE_TOLERANCE, R2_MIN), expect=(0.15, 0.9))
    a.check("criterion 1's threshold widens with the number of cells and "
            "narrows with the degrees of freedom",
            lambda: bonferroni_z(40, 5) > bonferroni_z(1, 5) > 0
            and bonferroni_z(40, 5) > bonferroni_z(40, 50))
    a.check("one replicate leaves no dof, and no threshold at all",
            lambda: bonferroni_z(40, 0) is None)

    flat = {"arm": "fake", "replicates": 6, "parameters": ["omega1"],
            "cells": [{"omega1": {"planted": t, "recovered": 1.0155,
                                  "se": 0.01, "z": (1.0155 - t) / 0.01}}
                      for t in (0.4, 0.8, 1.2, 1.7, 2.2, 2.5)]}
    v = judge([flat])
    a.check("a hardcoded constant produces a FLAT line -- slope 0 -- and fails",
            lambda: abs(v["checks"][0]["responsiveness"]["slope"]) < 1e-12
            and not v["passed"])
    near = {"arm": "fake", "replicates": 6, "parameters": ["omega1"],
            "cells": [{"omega1": {"planted": t, "recovered": 1.0,
                                  "se": 0.05, "z": (1.0 - t) / 0.05}}
                      for t in (0.99, 1.0, 1.01, 0.995, 1.005, 1.002)]}
    c = judge([near])["checks"][0]
    a.check("criterion 1 alone would MISS that leak on srw, which is why 2 exists",
            lambda: c["unbiased"] and not c["responsive"])
    a.check("REL_TOLERANCE rescues a tiny bias with a tiny se, and records that "
            "it did",
            lambda: REL_TOLERANCE == 0.01)
    a.raises("--inject-leak refuses a constant it cannot inject", SystemExit,
             lambda: parse_leaks(["sigma=1.0"]), contains="unknown constant")
    a.raises("...and a value with no name", SystemExit,
             lambda: parse_leaks(["omega1"]), contains="NAME=VALUE")

    # Sized down hard, so this is a check on the machinery, not on the physics.
    # gamma / srw / cost survive that; `correction` deliberately does not, and
    # the next case is about how it says so.
    nl_json = sc.data / "no_leakage.json"
    p = cli_ok(a, "check_no_leakage.py plants a grid and reports both criteria",
               ["calibration/check_no_leakage.py", "--arms", "gamma", "srw",
                "cost", "--draws", "3", "--replicates", "3",
                "--cost-replicates", "2", "--n-gamma", "20000",
                "--n-srw", "2000", "--json", str(nl_json)],
               stdout_has="verdict", creates=nl_json, timeout=1800)
    if p:
        a.check("...naming the planted value, the recovered one and its se, "
                "cell by cell",
                lambda: "planted" in p.stdout and "recovered" in p.stdout)
        a.check("...and the seed that planted them (ground rule 5)",
                lambda: "plant seed" in p.stdout)
        a.check("...passing all three of those arms even at 1/50th the samples",
                lambda: "FAIL" not in p.stdout, detail=p.stdout[-200:])
    p = cli_ok(a, "a budget too small to identify omega_1 fails, and says it is "
                  "a BUDGET failure rather than a leak",
               ["calibration/check_no_leakage.py", "--arms", "correction",
                "--draws", "3", "--replicates", "3", "--n-correction", "50000"],
               expect_code=1, stdout_has="FAIL", timeout=1800)
    if p:
        a.check("...distinguishing `unidentified at this budget` from `flat`, "
                "which is stage 3.0's vacuous-test lesson made printable",
                lambda: "unidentified at this budget" in p.stdout,
                detail=[ln for ln in p.stdout.splitlines()
                        if "FAIL" in ln][-1][:110])
    cli_ok(a, "--arms runs one arm alone, on the same planted grid",
           ["calibration/check_no_leakage.py", "--arms", "cost",
            "--cost-replicates", "2"], stdout_has="arm cost", timeout=900)
    p = cli_ok(a, "--inject-leak hardcodes a constant and the check CATCHES it "
                  "(exit 0 means the control worked)",
               ["calibration/check_no_leakage.py", "--arms", "correction",
                "--draws", "3", "--replicates", "3", "--n-correction", "50000",
                "--inject-leak", "omega1=1.0155"],
               stdout_has="NEGATIVE CONTROL worked", timeout=1800)
    if p:
        a.check("...localizing it: the leaked parameter fails, its neighbours "
                "do not",
                lambda: "omega1" in p.stdout and "FAIL" in p.stdout)


# ===========================================================================
# STAGE 5 -- what is here that nothing uses. Static, so it needs no fixtures.
# ===========================================================================

#: Every first-party module, in dependency order. `tools/tests/` is excluded
#: (gitignored, local-only) and so is this file.
def _repo_modules() -> list[Path]:
    out = []
    for layer in ("tools", "models", "src", "calibration"):
        for f in sorted((ROOT / layer).rglob("*.py")):
            if "tests" in f.parts or f.name in ("__init__.py", "exercise_all.py"):
                continue
            out.append(f)
    return out


def _names_used(tree) -> set[str]:
    """Every identifier the module mentions, however it mentions it."""
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A name can also be reached from a string annotation or a docstring
            # reference; counting those keeps the report free of false alarms.
            used.update(w for w in node.value.replace("(", " ").replace(".", " ")
                        .replace(",", " ").split() if w.isidentifier())
    return used


def _imported_bindings(tree) -> dict[str, int]:
    """{bound name: line} for every import in the module."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue                     # `annotations` is never "used"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                out[(alias.asname or alias.name).split(".")[0]] = node.lineno
    return out


def sec_static(a: Audit) -> None:
    """static scan -- imports and functions nothing reaches"""
    a.begin("static scan", "imports and functions nothing reaches")

    modules = _repo_modules()
    a.check("every first-party module parses",
            lambda: len(modules) > 0, detail=f"{len(modules)} modules")

    trees, sources = {}, {}
    for f in modules:
        src = f.read_text()
        sources[f] = src
        trees[f] = ast.parse(src)

    # --- imports bound but never mentioned ---
    unused = []
    for f, tree in trees.items():
        used = _names_used(tree)
        for name, line in sorted(_imported_bindings(tree).items()):
            if name not in used:
                unused.append(f"{f.relative_to(ROOT)}:{line} {name}")
    # A NOTE, not a failure: an unused import costs nothing at runtime. It is
    # reported because each one is a CLAIM about what a module depends on, and
    # the layering above is checked from exactly those claims.
    if unused:
        a.note(f"{len(unused)} import(s) are bound and never used",
               "\n".join(unused) + "\n(this scan counts a name mentioned in a "
               "docstring as used, so it under-reports; every entry here is real. "
               "tools/persistence.py's `read_artifact` is the one worth a look -- "
               "the module reads metadata through `artifact_path` and json "
               "directly, so the import claims a dependency it does not have.)")
    else:
        a.check("no module imports a name it never mentions", lambda: True)

    # --- public module-level functions nothing calls ---
    entry = {"_main", "main"}
    defined: dict[str, Path] = {}
    for f, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") or node.name in entry:
                    continue
                defined[node.name] = f

    tests_dir = ROOT / "tools" / "tests"
    test_sources = {t: t.read_text() for t in sorted(tests_dir.glob("*.py"))} \
        if tests_dir.is_dir() else {}

    unreferenced, tests_only = [], []
    for name, home in sorted(defined.items()):
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        elsewhere = sum(len(pattern.findall(src)) for f, src in sources.items()
                        if f != home)
        in_tests = sum(len(pattern.findall(src)) for src in test_sources.values())
        own = len(pattern.findall(sources[home])) - 1        # minus the def itself
        if elsewhere == 0 and own == 0:
            (tests_only if in_tests else unreferenced).append(
                f"{home.relative_to(ROOT)}  {name}()"
                + (f"  [{in_tests} test reference(s)]" if in_tests else ""))

    a.check("every public function is called from somewhere in the repo",
            lambda: not unreferenced, detail="")
    if unreferenced:
        a.note(f"{len(unreferenced)} public function(s) have no caller at all",
               "\n".join(unreferenced) + "\n(not even a test. Each is either a "
               "public API kept for a caller that does not exist yet, or dead.)")
    a.check("no public function is exercised by tests alone",
            lambda: not tests_only,
            detail="\n".join(tests_only) + ("\n(a tested function with no "
            "production caller is still a maintenance cost -- either wire it up "
            "or drop it)" if tests_only else ""))

    # --- the layering rule PLAN.md states, checked rather than assumed ---
    violations = []
    for f, tree in trees.items():
        layer = f.relative_to(ROOT).parts[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                target = node.module.split(".")[0]
                if layer in ("tools", "models") and target in ("src", "calibration"):
                    violations.append(f"{f.relative_to(ROOT)}:{node.lineno} "
                                      f"imports {node.module}")
                if layer == "src" and target == "calibration":
                    violations.append(f"{f.relative_to(ROOT)}:{node.lineno} "
                                      f"imports {node.module}")
    a.check("the layering holds: tools/ and models/ import no src/, src/ imports "
            "no calibration/ (PLAN.md)", lambda: not violations,
            detail="\n".join(violations))

    # --- every module imports by full path, never by bare name (PLAN.md) ---
    bare = []
    first_party = {"tools", "models", "src", "calibration"}
    known = {f.stem for f in modules}
    for f, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                head = node.module.split(".")[0]
                if head in known and head not in first_party:
                    bare.append(f"{f.relative_to(ROOT)}:{node.lineno} "
                                f"from {node.module} import ...")
    a.check("nothing is imported by a bare module name (PLAN.md's Imports rule)",
            lambda: not bare, detail="\n".join(bare))


def sec_percolation_zd(a: Audit) -> None:
    """models/percolation_zd -- the same lattice with dim as a parameter"""
    from models import percolation2d as perc2d
    from models import percolation_zd as pzd

    a.begin("models/percolation_zd",
            "site percolation on Z^dim, dim a model parameter, cost i**dim")

    # The claim the whole generalization rests on: at dim = 2 this IS
    # models/percolation2d.py, draw for draw.
    for anchor in ("south", "origin"):
        for geometry in ("box", "cylinder"):
            a.check(f"dim=2 reproduces percolation2d bit for bit "
                    f"({anchor}, {geometry})",
                    lambda anchor=anchor, geometry=geometry: np.array_equal(
                        perc2d.percolation2d(11, n=120, anchor=anchor,
                                             geometry=geometry,
                                             rng=np.random.default_rng(4)),
                        pzd.percolation_zd(11, n=120, dim=2, anchor=anchor,
                                           geometry=geometry,
                                           rng=np.random.default_rng(4))))

    rng = np.random.default_rng(0)
    a.check("returns n counts in [0, i**dim] at dim = 3",
            lambda: (lambda y: bool(y.shape == (500,) and y.min() >= 0
                                    and y.max() <= 6 ** 3))(
                pzd.percolation_zd(6, n=500, dim=3, rng=rng)))
    a.check("p = 1 fills the box at dim = 4",
            lambda: set(np.unique(pzd.percolation_zd(4, n=10, dim=4, p=1.0,
                                                     rng=rng))),
            expect={256})
    a.check("face_far at p = 1 is the far half only",
            lambda: set(np.unique(pzd.percolation_zd(4, n=10, dim=3, p=1.0,
                                                     anchor="face_far",
                                                     rng=rng))),
            expect={2 * 16})
    a.check("block_n does not change the numbers (dim = 3 torus)",
            lambda: np.array_equal(
                pzd.percolation_zd(7, n=200, dim=3, geometry="torus",
                                   rng=np.random.default_rng(6), block_n=200),
                pzd.percolation_zd(7, n=200, dim=3, geometry="torus",
                                   rng=np.random.default_rng(6), block_n=5)))
    a.close("dim = 1 matches its closed form sum_{k<=i} p**k",
            lambda: float(pzd.percolation_zd(10, n=200_000, dim=1, p=0.7,
                                             rng=np.random.default_rng(7)).mean()),
            pzd.expected_face_count_1d(10, 0.7), 0.02)

    a.check("cost_hint(i) = i**dim, so the article's d IS the dimension",
            lambda: [pzd.cost_hint(16, {"dim": d}) for d in (2, 3, 5)],
            expect=[256.0, 4096.0, 16.0 ** 5])
    a.close("zero_rate is exactly (1-p)**(i**(dim-1))",
            lambda: float((pzd.percolation_zd(3, n=200_000, dim=2, p=0.5,
                                              rng=np.random.default_rng(8))
                           == 0).mean()),
            pzd.zero_rate(3, 2, 0.5), 0.01)

    # The error branches no test file reaches by accident.
    a.raises("an unknown anchor is refused", ValueError,
             lambda: pzd.percolation_zd(4, n=1, anchor="north"), "unknown anchor")
    a.raises("an untabulated p_c is refused rather than guessed", ValueError,
             lambda: pzd.critical_p(14), "no tabulated p_c")
    a.raises("a dimension whose 3**dim neighbourhood cannot be built is refused",
             ValueError, lambda: pzd.percolation_zd(2, n=1, dim=20, p=0.5),
             "dim must be in")
    a.raises("a lattice past int32 label space is refused up front", ValueError,
             lambda: pzd.percolation_zd(4096, n=1, dim=3), "int32 label space")

    # The criticality diagnostic: it must respond to p, in the right direction,
    # or a flat column at p_c would mean nothing.
    a.check("crossing_fraction is monotone in p",
            lambda: (lambda f: bool(f[0] < f[1] < f[2]))(
                [pzd.crossing_fraction(8, n=200, dim=3, p=p, geometry="box",
                                       rng=np.random.default_rng(9))
                 for p in (0.25, 0.3116077, 0.38)]))

    # anchor="slab": the seed set's dimension as one knob.
    a.check("slab k = 0 reproduces the origin anchor sample for sample",
            lambda: np.array_equal(
                pzd.percolation_zd(9, n=150, dim=3, anchor="origin",
                                   rng=np.random.default_rng(50)),
                pzd.percolation_zd(9, n=150, dim=3, anchor="slab", anchor_dim=0,
                                   rng=np.random.default_rng(50))))
    a.close("slab k = dim counts every open site, so gamma = dim exactly",
            lambda: float(pzd.percolation_zd(5, n=400, dim=3, p=0.4,
                                             anchor="slab", anchor_dim=3,
                                             rng=np.random.default_rng(51)).mean())
            / 125.0, 0.4, 0.01)
    a.check("slab is monotone in k (a bigger seed set contains a smaller one)",
            lambda: (lambda ms: bool(all(a_ < b_ for a_, b_ in zip(ms, ms[1:]))))(
                [pzd.percolation_zd(8, n=200, dim=3, anchor="slab", anchor_dim=k,
                                    rng=np.random.default_rng(52)).mean()
                 for k in range(4)]))
    a.raises("slab without an anchor_dim is refused, naming the field", ValueError,
             lambda: pzd.percolation_zd(4, n=1, dim=3, anchor="slab"),
             "anchor_dim")
    a.raises("an out-of-range anchor_dim is refused", ValueError,
             lambda: pzd.percolation_zd(4, n=1, dim=3, anchor="slab",
                                        anchor_dim=4), "anchor_dim must be in")

    a.note("anchor='origin' measures gamma/nu, NOT d_f -- a correction",
           "E|C(0) cap B_i| is the box-restricted SUSCEPTIBILITY, "
           "i**(dim - 2 beta/nu) = i**(gamma/nu), which is 43/24 = 1.7917 in two "
           "dimensions and not d_f = 91/48: E|C cap B_i| = P(reach i) * "
           "E[|C| | reach i] ~ i**(-beta/nu) * i**d_f. models/percolation2d.py "
           "and experiments/03_percolation_zd/README.md asserted the two anchors "
           "shared an exponent; both now carry the correction. That experiment's "
           "own origin arm measured 1.7593..1.7955, converging on 43/24, and it "
           "was recorded as an observable failing to converge. Confirmed at "
           "dim = 3: 1.995, 2.120, 1.865 against gamma/nu(3) = 2.045.")
    a.note("anchor='slab' makes the seed set's dimension a knob, and k is NOT a "
           "design constant",
           "gamma(k) = k + gamma/nu below beta/nu, d_f on the plateau "
           "beta/nu <= k <= d_f, and k above it -- so k CHANGES THE EXPONENT "
           "BEING MEASURED, unlike box_exponent or DF_LOWER. Choosing it from a "
           "literature beta/nu would assume the answer; the sweep over k, and the "
           "plateau it exposes, is the measurement. k = 0 is 'origin' identically "
           "and k = dim gives gamma = dim exactly, so both ends are pinned by a "
           "known answer. Igor's proposal, 2026-09-06; see "
           "experiments/05_percolation_highd/README.md H7.")

    a.note("PLAN.md ground rule 7's observable is a dim <= 4 statement",
           "generalizing its own 2-D derivation gives gamma_face = "
           "max(d_f, dim-1), because sum_h h**(-beta/nu) stops being dominated "
           "by h ~ i once beta/nu = dim - d_f exceeds 1, i.e. from dim = 5 on. "
           "Measured at 4e9 sites in dim = 5: anchor='face' gives a last local "
           "slope of 4.160 +/- 0.009 (heading to dim-1 = 4, 19 sigma from d_f) "
           "while anchor='face_far' gives 3.512 +/- 0.068 against d_f = 3.54. "
           "The default stays 'face' because that is what the rule says and "
           "what dim = 2 must reproduce; the choice is measured, not asserted "
           "-- experiments/05_percolation_highd/README.md, H2.")
    a.note("p_c is a LITERATURE INPUT here, unlike every other constant",
           "a wrong entry in P_C_SITE_HYPERCUBIC simulates an off-critical "
           "system, and a slightly supercritical lattice still gives a clean "
           "power law with the wrong exponent -- a bias no estimator in this "
           "repo can see. P_C_SOURCE records the reference per dimension and "
           "crossing_fraction (src/estimate/check_criticality.py) is the check; "
           "it PASSes on the BOX at dim = 3, 4, 5 and is inconclusive at "
           "dim = 6, so dim >= 6 rests on Mertens & Moore (2018) alone.")


def sec_percolation_tau_zd(a: Audit) -> None:
    """models/percolation_tau_zd -- the cluster-number density on Z^dim"""
    from models import percolation_tau as ptau
    from models import percolation_tau_zd as ptzd

    a.begin("models/percolation_tau_zd",
            "clusters at size scale s on Z^dim, gamma = 1 - tau")

    for observable in ("bin", "tail"):
        a.check(f"dim=2 reproduces percolation_tau bit for bit ({observable})",
                lambda observable=observable: np.array_equal(
                    ptau.percolation_tau(32, n=100, observable=observable,
                                         rng=np.random.default_rng(9)),
                    ptzd.percolation_tau_zd(32, n=100, dim=2,
                                            observable=observable,
                                            box_factor=16.0, box_exponent=0.5,
                                            rng=np.random.default_rng(9))))

    a.check("box_factor**dim is the sites in one budget unit, in every dim",
            lambda: [round(ptzd.default_box_factor(d) ** d, 6)
                     for d in (2, 3, 4, 6)],
            expect=[256.0] * 4)
    a.check("box_exponent defaults to 1/DF_LOWER, not 1/dim",
            lambda: [round(ptzd.default_box_exponent(d) * ptzd.DF_LOWER[d], 9)
                     for d in sorted(ptzd.DF_LOWER)],
            expect=[1.0] * len(ptzd.DF_LOWER))
    a.check("block_n does not change the numbers (dim = 3 torus)",
            lambda: np.array_equal(
                ptzd.percolation_tau_zd(8, n=150, dim=3,
                                        rng=np.random.default_rng(10),
                                        block_n=150),
                ptzd.percolation_tau_zd(8, n=150, dim=3,
                                        rng=np.random.default_rng(10),
                                        block_n=5)))
    a.check("the shared sampler reads the same observable off shared lattices",
            lambda: (lambda pair: bool(set(pair[0]) == {8, 16}
                                       and pair[1]["shared_lattice"]
                                       and pair[1]["dim"] == 3
                                       and pair[1]["sites"]
                                       == 50 * pair[1]["L"] ** 3))(
                ptzd.shared_sampler([8, 16], 50, {"dim": 3},
                                    np.random.default_rng(11))))
    a.close("Assumption 2 is reported, not asserted: the zero fraction is small "
            "at the default box_factor",
            lambda: ptzd.zero_fraction(
                ptzd.percolation_tau_zd(16, n=1000, dim=3,
                                        rng=np.random.default_rng(12))),
            0.0, 0.20)

    a.raises("an unknown observable is refused", ValueError,
             lambda: ptzd.percolation_tau_zd(8, n=1, observable="histogram"),
             "unknown observable")
    a.raises("a box too small to hold a cluster of size s is refused", ValueError,
             lambda: ptzd.percolation_tau_zd(4096, n=1, dim=2, box_factor=1.0,
                                             box_exponent=0.0),
             "does not fit in the box")
    a.raises("a dimension with no design d_f bound is refused", ValueError,
             lambda: ptzd.df_lower(11), "no design d_f bound")
    a.raises("an unordered ladder is refused by the shared sampler", ValueError,
             lambda: ptzd.shared_sampler([16, 8], 5, {"dim": 3},
                                         np.random.default_rng(13)),
             "strictly increasing")

    a.note("DF_LOWER and DEFAULT_CUT_FRACTION are DESIGN constants, and one of "
           "them was calibrated in 2-D and transferred",
           "both size boxes and neither reaches an estimator, so a wrong value "
           "changes the noise and the cost but not the fitted tau. DF_LOWER "
           "replaces models/percolation_tau.py's box_exponent = 1/2, whose "
           "cutoff drift 1 - d_f/dim is 5/96 in two dimensions and 1/3 at "
           "dim = 6. DEFAULT_CUT_FRACTION = 0.05 was measured in 2-D against "
           "the dimensionless ratio s/L**d_f -- principled to carry across, but "
           "a transfer; experiments/05_percolation_highd/README.md's H5(c) and "
           "H6(a) are the checks that it holds.")


# ===========================================================================
# Stages, and the runner
# ===========================================================================

#: Ordered by dependency: nothing in a stage may call a function from a later
#: one. That is what makes a late failure diagnosable -- see the module
#: docstring.
STAGES: dict[str, list] = {
    "tools": [sec_rng, sec_constants, sec_summary, sec_loglog, sec_correction,
              sec_coverage, sec_wilson, sec_allocation, sec_cost_model,
              sec_artifacts, sec_persistence, sec_models_registry, sec_loglog_plot],
    "models": [sec_srw, sec_percolation2d, sec_percolation_tau,
               sec_percolation_zd, sec_percolation_tau_zd, sec_synthetic],
    "src": [sec_generate, sec_estimate, sec_budget, sec_report, sec_study],
    "calibration": [sec_calibration],
    "static": [sec_static],
}

#: Sections that need the throwaway experiment directory.
NEEDS_SCRATCH = {sec_generate, sec_estimate, sec_budget, sec_report, sec_study,
                 sec_calibration}


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=(*STAGES, "all"), default="all",
                   help="which layer to exercise; they are ordered by dependency")
    p.add_argument("--list", action="store_true",
                   help="print the sections each stage would run, then exit")
    p.add_argument("--workdir", type=Path, default=None,
                   help="where the throwaway experiment goes (default: a temp dir, "
                        "removed afterwards)")
    p.add_argument("--keep", action="store_true",
                   help="do not delete the throwaway experiment -- for inspecting "
                        "what a driver actually wrote")
    p.add_argument("--verbose", action="store_true",
                   help="print every passing check, not only the failures and notes")
    p.add_argument("--fail-fast", action="store_true", dest="fail_fast")
    p.add_argument("--json", type=Path, default=None, help="write the transcript")
    a = p.parse_args(argv)

    stages = list(STAGES) if a.stage == "all" else [a.stage]
    if a.list:
        for s in stages:
            print(f"{s}:")
            for fn in STAGES[s]:
                head = (fn.__doc__ or "").strip().splitlines()
                print(f"  {fn.__name__:<24} {head[0] if head else ''}")
        return

    audit = Audit(verbose=a.verbose, fail_fast=a.fail_fast)
    scratch = Scratch(a.workdir)
    print(f"exercising: {', '.join(stages)}")
    print(f"scratch experiment: {scratch.exp}")
    try:
        for s in stages:
            for fn in STAGES[s]:
                if fn in NEEDS_SCRATCH:
                    fn(audit, scratch)
                else:
                    fn(audit)
    finally:
        code = audit.report()
        if a.json:
            a.json.write_text(json.dumps(
                {"counts": audit.counts(), "rows": audit.rows}, indent=2))
            print(f"\nwrote {a.json}")
        if a.keep:
            print(f"\nkept: {scratch.root}")
        else:
            scratch.cleanup()
    raise SystemExit(code)


if __name__ == "__main__":
    _main()
