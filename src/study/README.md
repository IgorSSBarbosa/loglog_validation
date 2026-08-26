# src/study — the four-step workflow

For someone who has a model and wants $\hat\gamma$ with an honest error bar,
without reading constants out of one JSON file and typing them into another.

```bash
D=experiments/01_srw/data

# 1. cheap pilot: measures d, omega1, a1, cv, and this machine's throughput
python3 src/study/pilot.py -meta experiments/01_srw/recipes/samples_pilot.json \
        --study mystudy --replicates 3

# 2. what a longer run would cost -- proposes, draws nothing
python3 src/study/plan.py --study mystudy --data-root $D --time 90s
python3 src/study/plan.py --study mystudy --data-root $D --target-se 1e-4
python3 src/study/plan.py --study mystudy --data-root $D --time 90s --accept

# 3. execute the accepted plan
python3 src/study/run.py --study mystudy --data-root $D

# 4. the deliverable
python3 src/study/report.py --study mystudy --data-root $D --budget-analysis
```

Each step reads the study directory and writes back to it, so **no constant is
ever retyped**. Before this existed you read $\omega_1$ out of `omega1.json`,
$d$ out of `cost_probe.json`, and typed both into `allocation_table.py` — and if
you forgot, it silently used $1.0$ for each.

```
experiments/01_srw/data/mystudy/
  constants.json   every meta-constant: value, error, provenance
  pilot.json       the pilot's draws, fit, cost probe and throughput
  plan.json        the accepted allocation
  final.json       the long run's per-scale summaries and seeds
  answer.json      gamma-hat and its interval, machine-readable
  report.md        gamma +/- error + the log-log plot     <- the deliverable
  details.md       d, omega1, a1, a0, each with its error
  plot.png
  budget_analysis.md   (with --budget-analysis) predicted vs actual cost
```

## The pilot cannot determine $\omega_1$, and says so

This is the single most important thing to know about the workflow.

Measured on srw at scales $8..256$, $\hat\omega_1$ from one pilot, across
independent seeds:

| pilot | wall clock | $\hat\omega_1$ across 8 seeds | median | spread (MAD) |
|---|---|---|---|---|
| $B=10^8$, R=3 | 2.3 s | 0.05, 0.07, 0.27, 0.81, 0.98, 2.07, 2.49, **13.0** | 0.90 | 0.84 |
| $B=10^9$, R=3 | 21 s | 0.07, 0.07, 0.36, 0.97, 1.07, 1.43, 1.93, 2.32 | 1.02 | 0.79 |
| $B=5\times10^9$, R=3 | 109 s | 0.35, 0.53, 0.58, 0.79, 1.09, 1.32, 1.58, 1.88 | 0.94 | 0.40 |

Truth is $1$. **The median is right at every budget** — the estimator is not
biased, and no amount of averaging pilots would reveal a problem. What is wrong
is the spread of any *single* one.

It does converge, at about the rate theory says: between the last two rows the
MAD shrinks like $B^{-0.42}$, against the $B^{-1/2}$ of a well-behaved estimator.
(The first row is too broad to fit — its MAD is meaningless when one draw lands
at 13.) Extrapolating that rate, reaching $\mathrm{se}(\hat\omega_1)\approx0.1$
needs $B\approx10^{11}$ at $R=3$ — **tens of minutes, not a pilot**. Experiment B
in fact used $B = 5\times10^{10}$ with 6 replicates to reach $0.984 \pm 0.111$,
which is the same place by a different route.

So the practical guidance is not "run a longer pilot until $\omega_1$ is sharp".
It is: accept that a pilot bounds $\omega_1$ loosely, let `plan` tell you what
that costs, and spend the compute on the final run instead — the optimum in
$m_0$ is quadratic, so being a step off costs only ~1.1x in RMSE. A badly
*unidentified* pilot is a different matter, and that is what the strongest
verdict is for.

So `plan` refuses to pretend. It prints an error budget — how far each
constant's own standard error moves the optimal $m_0$, and what that costs in
RMSE — and one of three verdicts:

- **DO NOT PLAN ON THIS PILOT** — some constant's error bar is as wide as its
  own value. That is not a loose measurement, it is an unidentified fit, and
  reading the number is pointless.
- **loosely determined** — usable but soft; the advice is to add replicates if
  the run will be long.
- **good enough to plan on**.

It never decides for you. `--accept` is a separate invocation, on purpose.

**This is not theoretical.** The walkthrough above, run on a 2.3-second pilot
that reported `omega1 = 2.18 +/- 0.96` and `a1 = -2.72 +/- 7.43` (truths 1 and
$-1/4$), was flagged DO NOT PLAN. Accepting anyway produced
$\hat\gamma = 0.500905 \pm 0.000041$ — a tight-looking 95% interval
$[0.50073, 0.50108]$ that **excludes the true $1/2$**. The plan believed the
finite-size bias decayed like $\rho^{-5\times2.18}$ rather than $\rho^{-5}$, so
it chose $m_0$ too shallow. The warning was right.

## Why the interval can exclude the truth

The replicate interval is the scatter of $R$ estimates about *their own mean*.
A finite-size bias shifts every replicate the same way, so **no number of
replicates reveals it**. `report.md` says so explicitly whenever the plan's
predicted bias is comparable to the measured standard error.

Two errors are reported for that reason:

| | what it is | blind to |
|---|---|---|
| replicate interval | $\hat\gamma \pm t_{R-1}\,\mathrm{sd}/\sqrt R$ | bias |
| closed-form sd | $\sqrt{12\sigma_\infty^2/(nm^3)}/\log\rho$, eq. (583)/(720) | nothing — but it is a *bound* |

The $t_{R-1}$ quantile is not decoration: at $R=5$,
$\Pr(|t_4|<1.96) = 0.8784$, so pairing a 5-point standard error with $1.96$
gives an interval labelled 95% that covers 88%. This repo published exactly
that mistake until `calibration/check_coverage.py` measured it.

## The allocation rule matters more than the pilot's budget

A pilot recipe should use the `snr` rule, not a flat $n$:

```json
"n": {"rule": "snr", "budget": 1e8, "d": 1.0, "omega1": 1.0}
```

$\omega_1$ is estimated from the correction $a_1 i^{-\omega_1}$, whose size
*shrinks* with $i$. A flat $n$ over-samples the small scales, where the
correction is already resolved hundreds of times over, and starves the large
ones, where it has sunk below the noise. A flat $n=20{,}000$ pilot returned
$\hat\omega_1 = 0.06 \pm 0.18$; the `snr` rule at the same wall clock returned
$0.98 \pm 0.28$. The rule's own `omega1` is a *design* input that shapes the
sample counts, not an estimate — roughly right is enough.

## Notes

- **`d` comes from the model, not the clock, when it can.** A model with a
  `cost_hint` declares its own cost, which is exact; the pilot times it anyway
  and reports the gap. Timing alone is unreliable at the pilot's own scales —
  a single `simulate(k, n=1)` call at $k=8$ is almost entirely Python dispatch,
  which returned $d = 8.0 \pm 280$. The probe therefore climbs geometrically
  away from the sample ladder until one call is slow enough to measure.
- **`--time` is the total**, split across replicates. Asking for 2h and getting
  a 6h run is the kind of surprise this workflow exists to remove.
- **Samples are not kept** by default: a planned run is often hundreds of MB per
  replicate and nothing downstream reads the draws. `--keep-samples` if you want
  them.
- These four are thin orchestrators. Every estimator, allocation rule and chart
  is the same code the rest of `src/` uses — `generate`, `fit_correction`,
  `estimate_cost_affine`, `tuned_allocation`, `loglog_plot`.
