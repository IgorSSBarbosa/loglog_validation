# calibration

Checks whose subject is **this repo's own machinery**, not a model.

That is the line between this folder and `src/`. `src/estimate/estimate_omega1.py`
measures a property of $|S_k|$; `calibration/check_coverage.py` measures a property of
*our inference about* $|S_k|$ — whether the interval we print around $\hat\omega_1$
means what it says. Both are scripts a human runs; they answer different kinds of
question, and mixing them made `src/` harder to read.

It is also distinct from `tools/tests/`, which is gitignored, runs in seconds under
`pytest`, and checks closed forms. These are slow Monte Carlo measurements with numeric
tolerances, run deliberately and read — so they are tracked in git like any other
result-producing script.

```bash
# every public function and every CLI flag, once   (~7 min)
python3 calibration/exercise_all.py
python3 calibration/exercise_all.py --stage tools    # one layer at a time
python3 calibration/exercise_all.py --list           # what it would run

# are the error bars honest?           (~3.5 min at the defaults)
python3 calibration/check_coverage.py
python3 calibration/check_coverage.py --arm all                      # every arm
python3 calibration/check_coverage.py --arm planted --trials 2000 --centre both
python3 calibration/check_coverage.py --arm planting --trials 3000   # is the planting faithful?
python3 calibration/check_coverage.py --arm wilson --trials 1000     # eq. (720) as a bound on gamma

# does a predicted runtime predict the real one?
python3 calibration/verify_prediction.py --m0 3 4 5 6 7 --replicates 3

# is any constant secretly hardcoded?   (~75 s at the defaults)
python3 calibration/check_no_leakage.py
python3 calibration/check_no_leakage.py --arms correction --inject-leak omega1=1.0155
```

## `check_coverage.py` — do our intervals cover?

PLAN.md checkpoint 0.4. Replays Experiment B's *exact* configuration — scales
$8\dots256$ at the snr allocation's $n$, $R=5$ replicates, the same
pool-then-refit centre and `sd(fits)/\sqrt R` width the pipeline really quotes —
hundreds of times against srw's known ground truth, and counts how often the stated
interval contains it.

**It found a real defect.** Every interval this repo called 95% was covering 88%,
because a 5-point standard error was being paired with the normal quantile $1.960$
instead of $t_4=2.776$. $\Pr(|t_4|<1.96)=0.8784$; measured $0.877$–$0.882$. The script
scores both quantile choices on identical draws rather than assuming which is right,
which is the only reason the comparison is conclusive.

Five arms, answering different questions. `--arm all` runs every one of them,
cheapest first — it really does mean all, which until 2026-09-04 it did not:
the group ran three of the five and silently skipped `srw` and `wilson`, the
second being the very bound `report.py` now leads with.

| arm | question | cost |
|---|---|---|
| `planting` | is the planted arm's Gaussian assumption itself sound? | cheap |
| `wilson` | how conservative is eq. (720)'s bound? | cheap |
| `rate` | does the *analytic* error bar on the decay exponent cover? | moderate |
| `planted` | is the fit's error bar the right width? | ~200 s at 500 trials |
| `srw` | does any of it survive real draws? | its own knobs — see below |

`srw` is the only arm that simulates, and at Experiment B's real allocation it
is ~1e14 steps, i.e. days. So it takes its own `--srw-n-scale` (default 1e-3)
and `--srw-trials` (default 40) rather than sharing `--n-scale` and `--trials`,
which are sized for arms that draw nothing. Reduced n is the *conservative*
direction for the question it asks — a sample mean only gets more Gaussian as
n grows — and the arm prints the step count and a wall-clock estimate before it
starts.

`planting` is the one that validates the validation: it KS-tests real srw
$\overline Y$ against the normal `planted` assumes. Small $n$ on purpose — normality of
a sample mean only improves with $n$, so a pass there implies a pass at Experiment B's
much larger $n$. `wilson` is a different question from the rest: it scores a **bound**,
so the pass condition is coverage $\ge$ nominal and the interesting number is how much
it overcovers.

Reading a result: `--trials 500` resolves coverage to about $\pm0.04$. That is ample
for the 95%→88% failure and *not* enough for the 68.3% re-score, where a single "ok"
can be a miss the CI cannot see. Raise `--trials` before drawing a conclusion there.

Ground truth ($\mathbb E|S_k|$, $\mathrm{sd}|S_k|$) lives in this driver and is
deliberately *not* registered in `tools/models.py` as a `target_fn` — same rule as
`allocation_experiment.py`'s `true_gamma`: truth may plant data and score a finished
answer, never reach an estimator (user's decision, 2026-08-20).

## `check_no_leakage.py` — does the answer move when the truth moves?

The falsification test, and the one that answers the question this repo keeps
having to answer. On srw the estimates are *right*, and that is precisely the
problem: $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$, $d=1$ are simple numbers a
hardcoded default can sit on — `tools/constants.py` documents the version of that
bug this project actually shipped.

So: **plant** a truth the code cannot know, drawn from a seeded generator and
appearing in no recipe, module constant or default; run the real pipeline; see
whether what comes out tracks what went in.

| | criterion |
|---|---|
| 1. unbiasedness | $\lvert\hat\theta-\theta\rvert/\mathrm{se}$ within threshold at every cell, **or** agreement to better than 1% of the truth |
| 2. responsiveness | regress $\hat\theta$ on $\theta$: slope $=1\pm0.15$ and $R^2\ge0.9$ |

Criterion 2 is the one that catches a leak. A hardcoded constant gives a **flat
line** — slope 0 — while criterion 1 alone would be *fooled on srw*, because a
constant frozen at srw's truth is correct there. That asymmetry is the whole
point of the file.

Four arms, each aimed at a different estimator and a different leak channel:

| arm | plants | recovered by |
|---|---|---|
| `gamma` | $\gamma, a_0$, no correction | the article's own `gamma_closed_form`, eq. (523)–(526) |
| `correction` | $\gamma, a_0, a_1, \omega_1$ | the real `src/study/pilot.py` → `fit_correction` |
| `cost` | $d\in\{0.5,0.75,1,1.5,2\}$ via the synthetic spin burn | the pilot's `climb_to_target` → `fit_cost_probe` → `_resolve_d` |
| `srw` | $q\ne1/2$ — the real lattice walk, truth off its anchor | `gamma_closed_form`, against the **exact** $\mathbb E\lvert S_k\rvert$ |

The `srw` arm's reference is not "1/2": it is what the article's estimator returns
on the exact binomial mean over the same ladder, computed in the checker and never
handed to anything that estimates. That removes the estimator's own correction bias
from the comparison exactly, so what is left in $z$ is sampling error — and the
target runs 0.52 → 0.98 as $q$ goes 0.5 → 0.8, which is a truth no default is
sitting on.

**A check that has only ever passed is not evidence.** `--inject-leak
omega1=1.0155` hardcodes the literal old `FALLBACK_OMEGA1` into the real fit and
re-runs; the exit code inverts, so 0 means the control worked. Measured: the
$\omega_1$ row goes to slope $0.000$, no standard error at all (every replicate
returns the same constant), 39–71% relative error — while $\gamma$, $a_0$ and
$a_1$ stay green, so the report *localizes* the leak rather than merely failing.

**It also measures a false-alarm rate nothing else could.** The `cost` arm runs
on a model whose declared cost is exact *by construction*, so every `MISMATCH`
verdict `src/study/pilot.py:_resolve_d` returns there is a false alarm and can
simply be counted. Measured: **13 of 40, 32%** at the defaults; 17 of 50 at `--cost-replicates 10`. The cause is in
the same run — the standard error one probe states for its own $\hat d$ is
2.4–4.4× smaller than the probe-to-probe spread of $\hat d$, because both `se`
sources measure *within*-probe jitter while the variation that matters is
*between* probes. The point estimate is unaffected (mean $\hat d$ within 0.004 of
truth at every exponent). `plans/saverepo.md` stage 4 has the table and the
proposed fix; `mismatch_rate` in `no_leakage.json` re-measures it every run.

Criterion 1 has two halves for a reason worth knowing. `se` here is the spread of
per-replicate estimates: sampling error and nothing else. Any estimator with a
systematic floor — and a nonlinear least-squares fit has one — therefore fails a
pure $z$ test as soon as $n$ is large enough, because `se` shrinks with $n$ and the
floor does not. And the threshold is a Bonferroni-corrected $t$ quantile, not a
flat 3: at $R=6$ over 40 cells, $\Pr(\text{some }\lvert t_5\rvert>3)$ is well over
half, and a green run would mean nothing.

## `exercise_all.py` — does every function still do what it says?

The other two files here measure a *statistical* property of the pipeline. This
one asks a blunter question about the same subject: for every public function
and every CLI flag in the repo, does calling it do something sensible? Including
the paths nothing calls — error branches, flag combinations, degenerate inputs —
which is where a defect can sit for months without a passing test noticing.

Ordered by dependency (`tools/` → `models/` → `src/` → `calibration/`, then a
static pass), and that ordering is the method: `allocation_constants` is checked
only after the `closed_form_weights` it calls has been checked on its own, so a
late failure whose earlier stages passed is a failure of the composition.

Three outcomes, and the third is what it is for:

| | |
|---|---|
| `PASS` | the call did what its docstring says |
| `FAIL` | it did not — a defect, with expected vs got |
| `NOTE` | it behaved as written, and the behaviour is worth a human's eye: a flag that does nothing, a message that misleads, an unreachable branch |

NOTEs never fail the run. They are the output someone reads and decides about;
FAILs get fixed and disappear. The first run (2026-09-04, 586 checks in 432 s)
found one defect — `autopilot.py --force` is accepted, threaded through two
signatures and never read — and 16 notes. Every one of them was resolved the
same day, and the harness now *checks* each fix rather than repeating the note
it replaced, so a regression shows up as a FAIL: the run stands at **628
checks, 0 failures, 1 note** (the parity trap in `tools/correction.py`, which
is a property of the model and cannot be guarded against here).
`plans/function_audit.md` is the write-up, findings and resolutions both.

Not a replacement for `tools/tests/`, which owns the closed-form assertions and
runs in seconds under pytest. This is slower, exercises the CLIs as subprocesses
(so argparse, the `sys.path` bootstrap and the exit code are part of what is
tested), and is read rather than gated on.

## `verify_prediction.py` — does a predicted runtime predict?

The same shape one level up: `src/budget/allocation_table.py` claims a ladder will take
$t$ seconds and land within RMSE $r$; this runs the tuned ladders for real and compares.
Measured 0.94×–1.00× on timing across four orders of magnitude, 1.06× median on RMSE.

Read the accuracy column as an order-of-magnitude check, not a calibration: an RMSE over
$R$ draws carries $1/\sqrt{2R}$ relative sd itself, so at `--replicates 3` it is $\pm41\%$
before the prediction is even wrong. Ladders needing more than `--max-n` samples per
scale are skipped rather than attempted — budget is derived *from* $m_0$ here, so $n$
grows like $\rho^{2m_0}$ and a stray `--m0 20` would otherwise ask for $10^{25}$ samples.

## What does not belong here

`src/budget/allocation_experiment.py` looks similar and is not: its subject is
Proposition `prop:opt`, a claim in the **paper**. Testing the article is the whole
repo's job and stays in `src/`. The test for "does this belong in `calibration/`?" is
whether the thing it could falsify is our code or someone else's theorem.

The two measurement scripts draw through `src/generate/generate.py` rather than
repeating its loop, and `verify_prediction.py` imports the table it checks, so this
folder depends on `src/` — never the reverse. `exercise_all.py` depends on everything
by construction, which is why it is last in its own ordering.
