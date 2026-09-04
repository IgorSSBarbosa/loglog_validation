# Function audit — every function, every flag, once

**Status:** run 2026-09-04 against `e677280`, tests green (420 passed).
**Resolved 2026-09-04** — every finding below is fixed; the re-run stands at
628 checks, 0 failures, 1 note (the 29 added cover models/synthetic.py's work
burn and calibration/check_no_leakage.py, both new the same day). See *What was done* at the end.
**How to reproduce:** `python3 calibration/exercise_all.py` (432 s).

The question was blunter than the test suite's: *for every public function and
every CLI flag in this repo, does calling it do something sensible?* Including
the paths nothing calls — error branches, flag combinations, degenerate inputs —
which is where a defect can sit for months without a passing test noticing.

`calibration/exercise_all.py` is that audit, ordered by dependency (`tools/` →
`models/` → `src/` → `calibration/`, then a static pass), so a failure in a late
stage whose earlier stages passed is a failure of the composition, not of a
piece. It reports three outcomes; the third is the useful one:

| | |
|---|---|
| `PASS` | the call did what its docstring says |
| `FAIL` | it did not — a defect |
| `NOTE` | it behaved as written, and the behaviour is worth a human's eye |

**Result: 586 checks — 569 PASS, 1 FAIL, 16 NOTEs, 432 s.** Nothing in the
statistical core is
wrong. Every estimator, every allocation rule, every closed form and every error
branch in `tools/` behaves as documented, including the ones with no test:
eq. (526)'s weight identities, `spawn(skip=)`'s no-duplicate guarantee,
`optimal_allocation`'s budget guarantee under flooring, srw's block-invariance,
the recipe-kind refusals, both sample layouts, and the legacy-artifact rescue.

What follows is what did not pass, and what passed but reads wrong.

---

## 1. The defect

### `--force` does nothing (`src/study/autopilot.py`)

`--force` is accepted, documented in three places, threaded through
`autopilot()` into `_give_up(consts, rounds, sd, pilot_seconds, seconds, lc,
force, log)` — and never read. `_give_up`'s body does not mention its `force`
parameter, so the study gives up identically with and without the flag.

Measured on a run rigged to fail the gate (one replicate, so ω₁ has no standard
error at all, and `--max-rounds 1`): exit 1 and no `final.json`, both with and
without `--force`. The three places that promise otherwise:

- the module docstring — *"--force overrides"*
- the give-up message itself — *"--force runs anyway, on constants known to be
  undetermined"*
- `--help` — *"run even if the constants were never determined"*

The fix is one line in `_give_up`, but the *semantics* are a decision, not a
typo: `--force` has to skip the gate and go on to `_plan_run_report`, which
means deciding what a study reports when it knows its own ω₁ is undetermined.
Leaving the flag out entirely is also a coherent answer — the give-up path
already prints everything needed to restart with a longer `--time`.

---

## 2. Dead code

| where | what |
|---|---|
| `src/generate/generate.py` | `DESIGN_INPUTS` and the `SystemExit` in `_design_input` are **unreachable**. That branch fires only for a key absent from `UNINFORMED`, and the only two keys ever passed — `"d"` and `"omega1"` — are both in it. No recipe can trigger the message. |
| `tools/wilson.py:96` | `j = np.arange(1, m + 1)` in `sigma_se_per_scale`, assigned and never used. |
| `src/study/pilot.py:353-354` | `d_val`, `d_se` assigned and never used — left over from before `_resolve_d` took the job. |
| `src/budget/allocation_table.py:631-632` | `a1_se`, `omega1_se` assigned and never used (the loop below reads `k.se`). |
| `src/report/plot_allocation.py:142` | `d`, `omega1_recipe` assigned and never used. |
| 16 imports | bound and never mentioned. `tools/artifacts.py` `sys`; `tools/persistence.py` `read_artifact`; `allocation_experiment` `rate_exponent_se`; `allocation_table` `Constant`, `log`; `measure_cost` `content_id`; `plot_allocation` `json`; `plan.py` `allocation_constants`, `ceil`, `log`, `read_artifact`; `check_coverage` `format_interval`, `sigma_se`, `se_ratio`; `verify_prediction` `get_model`, `json`. |
| `find_omega1_runs()` | the only public function in the repo with no caller outside its own tests (6 test references). Its docstring already says it is a thin wrapper "kept for callers that just want the default choice" — there are none. |

None of this costs anything at runtime. It is listed because each unused import
is a *claim* about what a module depends on, and two of them name exactly the
dependency their own docstring says was deliberately avoided.

---

## 3. Messages that mislead

**`--arm all` runs three of the five arms.** `check_coverage.py`'s map sends
`all` → `[planting, planted, rate]`, skipping `srw` and — more importantly —
`wilson`. The wilson arm checks eq. (720), the interval `report.py` now *leads
with*, so it is the easiest one to leave unrun by accident. (`both` →
`[planted, planting]` is fine; it is the pair its name suggests.)

**`measure_cost.py` prints `FAIL` for a model whose true `d` is 0.** On
`synthetic` (whose `cost_hint` is constant, so `d = 0` is *correct*) the driver
still scores against `ACCEPTANCE_RANGE = (0.8, 1.2)` and prints
`FAIL vs [0.8, 1.2] on d_hat=-0.0881`. The caveat that follows says the range
"only applies to genuinely scale-costly models", but the verdict word comes
first. The model *declares* its `d`; scoring against `declared_exponent` would
be right for every model instead of one.

**`compare_cost_models` reports `agree=True` when it could not compare.** With
fewer than 4 scales there is no standard error, the sigma test is skipped, and
`agree` keeps its initial `True` — indistinguishable in the returned dict from a
real agreement, and `format_cost_comparison` prints no warning. `measured_via`
is the only tell. Compare `pilot.py:_resolve_d`, which has a distinct
`"unchecked (no se for d)"` verdict for exactly this case.

**`--params` has no validation.** A typo reaches `TRUTH[p]` deep inside
`check_coverage._main` and exits with a bare `KeyError: 'nosuchparam'`. Every
other bad flag in this repo is caught at the boundary with a message naming the
valid values.

**`verify_prediction.py` crashes when every ladder is skipped.** With `--m0 40`
no cell survives `max_n`, and the summary lines call `min()`/`max()` on empty
lists: `ValueError: min() arg is an empty sequence`. The per-cell table is
already printed and the artifact already written, so nothing is lost — but the
exit code says failure for what is really "nothing to report".

**`human_time` keeps the smallest unit under 1000, not the largest that fits.**
So `--time 2h` is echoed back as `120.0 min`, and the switch to hours happens
only at 16.7 h. The docstring says "the largest unit that keeps the number
readable", which reads as the opposite.

**`climb_to_target` can return fewer rungs than the affine fit needs.** The
`>= PROBE_MIN_SCALES` guard applies only to the target branch; the
`max_doublings` and `time_budget` branches can exit at 3 rungs, against a
docstring that says "never before `PROBE_MIN_SCALES` rungs". Nothing breaks —
`fit_cost_probe` stores `affine: {"error": ...}` and `_resolve_d` handles a
missing `d` — but the guarantee as written does not hold.

**`input_sensitivity` silently ignores `cv`.** It dispatches on `a1`, `omega1`
and `d` and returns a no-op `{delta_m0: 0, penalty: 1}` for anything else. No
caller asks about `cv` today, so nothing is wrong — but "this input does not
move the plan" is not the same answer as "not implemented".

**`load_samples` prefers `samples.npz` over `samples/` silently.** When both
layouts exist the flat one wins and the chunked data is invisible.
`generate.py` deletes the other layout on every write precisely because of this
(and the comment there records it happening for real), but nothing in
`tools/persistence.py` itself warns: a directory assembled by hand, or by an
older `generate.py`, reads as healthy while serving stale data.

**`autopilot`'s gate is evaluated at a hardcoded 4× the pilot budget.**
`_provisional_m0` is called with `seconds_budget * 4 - spent`, where
`seconds_budget` is already `--time * --pilot-cap`. That reconstructs the total
only at the default `PILOT_CAP = 0.25`. With `--pilot-cap 0.1` the gate is
judged at 40% of the real remaining budget, hence at a smaller m₀ and a larger
B_fs span than the run will actually have — the loop doubles more than it needs
to.

**The parity trap is still silent, and slightly worse than documented.**
`tools/correction.py` warns that a ρ = √2 grid mixes the parities of srw's
staircase mean and fits the zig-zag as curvature. Re-measured on *exact* means
with zero noise: ω₁ = 18.0 against a truth of 1 (the docstring says ~17.8), but
with `converged = False`, where the docstring says "the fit converges and
reports a small residual". A powers-of-two grid on the same exact means gives
ω₁ = 1.00. The warning is right; only the "converges" half is not.

**`tools/artifacts.py` is runnable, and two layer descriptions say it is not.**
`README.md` and `CATALOG.md` both describe `tools/` as "imported, never run".
`artifacts.py` has an argparse `_main` (`--list` / `--migrate`), documented in
its own docstring, so the exception is deliberate — but it is not mentioned
where a reader would look for it.

---

## 4. One inexactness worth knowing

`sigma_se` implements eq. (720)'s fourth term, `sqrt(12 σ∞² / (n m³))`. The
*exact* variance of the same linear estimator under uniform n is
`12 σ∞² / (n m(m²−1))`, which `sigma_se_per_scale` computes. The two agree only
as m → ∞: at m = 6 the stated se is **1.42% too small**
(`sqrt(m²/(m²−1)) = 1.0142`, reproduced to 1e-12).

`sigma_se_per_scale`'s docstring says it "reduces to `sigma_se` when n and cv2
are constant", which is true only in that limit. This is small next to the bias
terms — but it is the one term eq. (720) treats as known rather than bounded,
and §5 measures what it costs.

---

## 5. The Wilson coverage experiment

> **Does the 95% confidence interval hold the truth 95% of the time?**

Run 2026-09-04, `calibration/check_coverage.py --arm wilson --trials 2000
--wilson-m0 0 2 4 6 8 10 --wilson-n 100000` (10 s), plus a term-by-term
decomposition at 20,000 trials.

The arm plants ȳ ~ N(μ_k, σ_k²/n) from srw's **exact** moments E|S_k| and
sd|S_k|, estimates γ with the article's own closed-form weights (eq. 523–526),
and counts how often |γ̂ − ½| ≤ the eq. (720) half-width. Uniform n = 100,000,
m = 6, ρ = 2, one replicate per trial — the bound needs no replicates, which is
the entire point of it.

**The planting is faithful at this n.** `--arm planting --planting-n 100000
--trials 400` KS-tests real srw ȳ against that normal: p = 0.047…0.99 across
all six scales, observed sd within 1–4% of exact. So what follows measures the
bound, not the Gaussian approximation.

### The answer: yes, and by a margin that shrinks with the ladder

| m₀ | scales | B_fs | q·σ_se | half-width | dominant | **coverage** | 95% CI |
|---|---|---|---|---|---|---|---|
| 0 | 2..64 | 5.15e-02 | 1.59e-03 | 5.32e-02 | B_fs | **1.000** | [0.998, 1.000] |
| 2 | 8..256 | 1.29e-02 | 1.59e-03 | 1.45e-02 | B_fs | **1.000** | [0.998, 1.000] |
| 4 | 32..1024 | 3.22e-03 | 1.59e-03 | 4.83e-03 | B_fs | **1.000** | [0.997, 1.000] |
| 6 | 128..4096 | 8.05e-04 | 1.59e-03 | 2.42e-03 | se_term | **0.992** | [0.987, 0.995] |
| 8 | 512..16384 | 2.01e-04 | 1.59e-03 | 1.81e-03 | se_term | **0.978** | [0.971, 0.984] |
| 10 | 2048..65536 | 5.03e-05 | 1.59e-03 | 1.66e-03 | se_term | **0.966** | [0.957, 0.973] |

It is a **bound**, so the pass condition is coverage ≥ 95%, and it passes
everywhere. It never undercovers and it never comes close to: the lowest
measured coverage in the sweep is 0.966, three standard errors above nominal.

How conservative depends entirely on where the ladder sits. At m₀ ≤ 4 the
finite-size term B_fs dominates the half-width and the bound covers *every*
trial — at m₀ = 0 it is 33× wider than the estimator's own scatter. As m₀ grows
the correction dies like ρ^(−ω₁m₀), B_fs stops mattering, and the bound
converges onto the 95% it claims from above.

### Which term is doing the work

20,000 trials per row, so each coverage carries ±0.003. Three intervals scored
on **identical draws**: the full four-term bound; the naive CLT interval
γ̂ ± 1.96·σ_se (eq. 583's term alone, no bias); and the same with the exact
weight norm from §4.

| m₀ | scales | true bias | true sd | σ_se | **bound** | CLT only | CLT, exact norm |
|---|---|---|---|---|---|---|---|
| 0 | 2..64 | 3.14e-02 | 9.53e-04 | 8.12e-04 | 1.000 | **0.000** | 0.000 |
| 2 | 8..256 | 8.07e-03 | 8.56e-04 | 8.12e-04 | 1.000 | **0.000** | 0.000 |
| 4 | 32..1024 | 2.02e-03 | 8.29e-04 | 8.12e-04 | 1.000 | 0.301 | 0.318 |
| 6 | 128..4096 | 5.10e-04 | 8.30e-04 | 8.12e-04 | 0.988 | 0.899 | 0.904 |
| 8 | 512..16384 | 1.12e-04 | 8.23e-04 | 8.12e-04 | 0.970 | 0.943 | 0.946 |
| 10 | 2048..65536 | 2.56e-05 | 8.22e-04 | 8.12e-04 | 0.955 | 0.944 | 0.948 |
| 12 | 8192..262144 | 3.61e-06 | 8.21e-04 | 8.12e-04 | 0.952 | 0.948 | 0.951 |

Three things fall out of that table.

**The bias terms are not decoration.** A 95% interval built from the CLT alone
covers **0.000** on the shallow ladders — not 0.94, not 0.5, zero. At m₀ = 2 the
finite-size bias is 8.07e-03 against a scatter of 8.56e-04: the estimator misses
by ten standard deviations, every time, in the same direction. This is the
concrete form of the warning `report.py` carries — *"a bias shifts every
replicate the same way, so no number of them reveals it"* — and it is why the
eq. (720) bound is the right thing to lead with. A replicate interval on a
shallow ladder is not merely optimistic; it is disjoint from the truth.

**Once the bias dies, the bound is nearly exact.** By m₀ = 12 the residual bias
is 3.6e-06 against a scatter of 8.2e-04, and the bound covers 0.952 against a
nominal 0.95 — 0.2 points of conservatism for a *bound*, which is as tight as a
bound can honestly be.

**The one place it is slightly optimistic is §4's 1.42%.** Look at the last two
columns at m₀ = 10, where bias/sd = 0.031 and is therefore negligible: the CLT
interval built on eq. (720)'s σ_se covers 0.944 [0.941, 0.947] — excluding 0.95 —
while the same interval on the exact weight norm covers 0.948 [0.945, 0.951],
which includes it. Theory agrees: 1.96/1.0142 = 1.933, and 2Φ(1.933) − 1 =
0.9467. So the asymptotic m³ in the fourth term costs about half a point of
coverage at m = 6.

It does not threaten the bound — the three bias terms cover that gap many times
over at every m₀ measured, which is exactly what the last column shows. But
anywhere σ_se is used *as* a standard error rather than inside the bound
(`report.py` prints it as "the closed-form sd of a single estimate", and
compares it against the replicate spread), it is 1.4% low at m = 6, and it is
cheap to be exact: `sigma_se_per_scale([n]*m, m, rho, [cv2]*m)` already
computes the right thing.

### Reading it in one line

The eq. (720) interval labelled 95% covered **96.6%–100%** across m₀ = 0…10 and
**95.2%** at m₀ = 12 — it holds, always, and is honest about being a bound
rather than an interval. The 95% interval that would *not* hold is the one
without the bias terms: on this repo's own pilot ladder (8…256, m₀ = 2) it
covers **0%**.


---

## 6. What was done

Same day, on Igor's call. The audit was re-run after each change; the harness
now *checks* each fix rather than reporting the note it replaced, so a
regression would show up as a FAIL rather than as a note nobody re-reads.

### The defect

**`--force`** now overrides the decision and only the decision. The diagnosis
prints identically either way — it is the evidence, and the evidence does not
depend on what you chose to do about it — and then planning, drawing and
reporting go ahead. The record carries `forced: true`, and the console closes
with a warning that the answer inherits an undetermined omega1.
`_give_up` split into `_diagnose` (says what happened) and `_record_give_up`
(writes the artifact when nothing was drawn), which is why the flag had been
easy to forget: one function was doing both jobs.

Worth seeing once: forced on a deliberately bad pilot, the study reported
gamma = 0.50214 with a t interval of [0.50192, 0.50235] and an eq. (720) bound
of [0.50148, 0.50279] — **both excluding the true 1/2**, because B_fs was built
from omega1 = 1.86 instead of 1. That is the failure the gate exists to
prevent, and it is exactly what the closing warning is for.

### The doubling loop, reopened

Not from the audit but from Igor's own run: the loop grew the replicate COUNT,
and every replicate costs its own non-convex four-parameter fit with five
restarts, so the fitting cost grew with the thing being doubled. It now grows
the DRAWS per replicate instead, redrawing the same R replicates at 2x, 4x, ...
n from independent streams.

Measured before changing it, 400 planted studies per arm, scales 8..256:

| arm | total draws | mean se(omega1) | mean omega1 |
|---|---|---|---|
| R=3, n (baseline) | 3.78e+07 | 0.1585 | 1.0125 |
| R=6, n (double R) | 7.56e+07 | 0.1207 | 1.0326 |
| R=3, 2n (double n) | 7.56e+07 | **0.1098** | **1.0031** |
| R=12, n (4x R) | 1.51e+08 | 0.0879 | 1.0103 |
| R=3, 4n (4x n) | 1.51e+08 | **0.0769** | **0.9989** |

At equal total draws the draws axis gives a 9–13% *smaller* se and is the only
one that also moves the point estimate toward the truth. Both follow from the
fit being nonlinear: more draws per fit shrink that fit's dispersion and its
bias together, while more fits at the same n average a distribution that stays
exactly as wide and as skewed. So the cheaper axis was also the better one —
the old docstring's argument for the other choice was wrong, and is replaced by
this table.

### Everything else

| # | resolution |
|---|---|
| 2 | `sigma_se` keeps eq. (720)'s formula verbatim. `sigma_se_per_scale`'s docstring now states the exact `sqrt(m²/(m²−1))` relationship and the measured coverage cost, instead of claiming the two are equal. |
| 3 | `--arm all` runs all five arms, cheapest first (`ALL_ARMS`). `srw` got its own `--srw-n-scale` / `--srw-trials`, because sharing the planted arm's numbers is what made it unrunnable in a group; it prints its step count and a wall-clock estimate before starting. |
| 4 | `measure_cost.py` scores the measured d against the model's **own declared** `cost_hint` (`ACCEPTANCE_REL = 0.2`), so `synthetic` — whose true d is 0 — now passes. A model declaring nothing falls back to the old window, labelled as an expectation rather than a truth. |
| 5 | `compare_cost_models` returns `comparable: False` and `agree: None` when there is no standard error, and `format_cost_comparison` leads with `NOT COMPARED`. "No test was possible" no longer shares a value with "they agree". |
| 6 | `--params` has `choices=`, so a typo is an argparse error naming the valid values. |
| 7 | `verify_prediction` reports "nothing here is runnable at these constants", names a nearer m0 and exits 0, instead of dividing an empty list. |
| 8 | `load_samples` warns (RuntimeWarning) when a run directory holds both layouts, naming the one it served. |
| 9 | The autopilot gate is judged at the whole remaining budget, not `seconds_budget * 4`, which was only the total at the default `--pilot-cap`. |
| 10 | `climb_to_target`'s `PROBE_MIN_SCALES` floor now outranks both stopping rules, and a `max_doublings` below it is refused as the contradiction it is. |
| 11 | `input_sensitivity` handles `cv` and raises on a name it does not know — validated *before* the `try` that used to swallow the raise and answer "this input does not matter". |
| 12–15 | Dead code deleted: the unreachable `DESIGN_INPUTS` branch, 16 unused imports, the dead local in `sigma_se_per_scale`, and four unused locals. |
| 16 | `find_omega1_runs` deleted; its tests keep the behaviour by calling `discover_groups` directly. `format_interval`, which the import cleanup exposed as having no caller at all, was **wired up rather than deleted** — `experiments/01_srw/README.md` already describes it as the thing that surfaces an incomplete bound, and the wilson arm now prints one full breakdown under its table. |
| 17 | Three docstrings corrected to what the code does: `human_time`, the parity-trap note in `tools/correction.py`, and the "imported, never run" description of `tools/` in both `README.md` and `CATALOG.md`. |

### Left standing, deliberately

- **The parity trap** (`tools/correction.py`). A rho = sqrt(2) grid over srw's
  exact means still fits omega1 = 18 against a truth of 1. It cannot be guarded
  against here: the parity that matters is a property of the model, which this
  module never sees. Documented, re-measured, and now honest about
  `converged = False` being the only tell.
