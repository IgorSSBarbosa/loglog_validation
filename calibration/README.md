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
python3 calibration/check_coverage.py --arm planted --trials 2000 --centre both
python3 calibration/check_coverage.py --arm planting --trials 3000   # is the planting faithful?
python3 calibration/check_coverage.py --arm wilson --trials 1000     # eq. (720) as a bound on gamma

# does a predicted runtime predict the real one?
python3 calibration/verify_prediction.py --m0 3 4 5 6 7 --replicates 3
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

Four arms, answering different questions:

| arm | question | cost |
|---|---|---|
| `planted` | is the fit's error bar the right width? | ~200 s at 500 trials |
| `planting` | is the planted arm's Gaussian assumption itself sound? | cheap |
| `rate` | does the *analytic* error bar on the decay exponent cover? | cheap |
| `wilson` | how conservative is eq. (720)'s bound? | moderate |

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
signatures and never read — and 16 notes. `plans/function_audit.md` is that
write-up.

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
