# Experiment 14 — `erw_urn_gpu`: the elephant random walk as an urn, on a GPU

**Question.** Does `MODELS["erw_urn_gpu"]` sample the same distribution as `MODELS["erw"]`
and `MODELS["erw_gpu"]`, closely enough that every estimator gives the same answer on
it? And does each run give the GPU enough samples to keep it busy? Speed only matters
because it buys bigger budgets and longer ladders.

The model, its regimes and the acceptance criteria for $\gamma$ are in
`experiments/11_erw/README.md` and aren't repeated here. `models/erw.py` is unchanged
and taken as correct. Nothing in `src/` or `calibration/` changed. In `tools/`, only the
registry gained the entry.

## Setup

| | |
|---|---|
| hardware | NVIDIA GeForce RTX 5090, 32 GiB (shared card), 170 SMs × 1536 threads |
| software | Python 3.10.12, numpy 2.0.2, scipy 1.13.1, **cupy-cuda12x 14.2.0**, CUDA runtime 12.9 |
| model | `models/erw_urn_gpu.py`: one CUDA thread per sample, state $(m, U_m)$ with $U_m$ the number of $+1$ steps so far, all $k$ steps in one `RawKernel` launch. Each step is two independent coins: a uniform past step is $+1$ ($\lfloor a\,m/2^{64}\rfloor<U_m$), and it's reversed ($b<\lfloor(1-p)2^{64}\rfloor$). This is eq. (2.2)'s chain, all in integers |
| RNG | one `rng.integers(0, 2**63)` per `simulate` call seeds Philox4x32-10. Sample $j$ uses subsequence $j$, so its draw doesn't depend on the launch chunk |
| reproducibility | a seed reproduces a run **on the versions above**. It never reproduces an `erw` or `erw_gpu` run |

### Why the urn (user, 2026-09-28)

The walk's past only enters through the colour of a uniformly picked step, so
$(m,U_m)$ is a sufficient state (Bercu eq. 2.2). `models/erw.py` measured this chain on
CPU and declined it (2026-09-22): a Python loop over $k$ steps. On a GPU, one thread
per sample with two registers of state is the natural shape. There's no trajectory in
memory (8 bytes per sample instead of `erw_gpu`'s ~46 per step) and no doubling passes.

### Why integers (2026-09-28)

A first draft drew one `curand_uniform_double` per step against the folded probability
$(1+(2p-1)S_m/m)/2$: 23 ps/step at a full device. The RTX 5090 runs float64 at 1/64 of
its float32 rate, so that draft was FP64-bound. The integer kernel runs at 2.35 ps/step.
It also makes $P(\text{reverse})=1-p$ exact to $2^{-64}$, with $p=0$ and $p=1$ exact.

## How many samples the GPU needs

A sample is a serial chain on one thread, so a call only uses the device if it has
enough samples. At $k=2^{16}$, $p=0.9$ (2026-09-28):

| $n$ | $2^8$–$2^{15}$ | $2^{16}$ | $2^{17}$ | $2^{18}$ | $2^{19}$ | $2^{20}$–$2^{22}$ |
|---|---|---|---|---|---|---|
| ms per call | 6.8 (flat) | 12.4 | 24.6 | 43.0 | 81.4 | 161–647 |
| ps per step | 407 → 3.17 | 2.88 | 2.86 | 2.50 | 2.37 | 2.34–2.35 |

Up to $2^{15}$ samples a call costs one sample's latency, however many there are. From
`resident_threads()` $=170\times1536=261{,}120\approx2^{18}$ up, a call is within ~7% of
the device's throughput. That's the floor this experiment keeps `n` above.

**Pilot.** `recipes/samples_super_p0.9.json` is 12's pilot template with the budget raised
from $10^9$ to $4\times10^9$. That gives 488,519 draws per scale instead of 122,129, which
is above the floor, for ~10 ms of device work per replicate.

**Final run: `sweep_p.py`'s ceiling.** `plan.py` draws one $n$ at every scale. When
$\omega_1$ is small it spends extra budget by sliding the ladder up, not by raising
$n$. 12's 1-hour sweep ended at $n=131{,}286$ at $p=0.775$, and at ~240× the speed the
same plan would drop $n$ well under the floor. So `sweep_p.py` passes autopilot a
`--max-scale`: the top of the highest ladder on which the final run can still afford
`--min-n` (default: `resident_threads()`) samples per scale per replicate. That's
computed from $(1-\texttt{pilot\_cap})\cdot T$ and the throughput *the plan will use*.
Every lower ladder affords more, so whatever $m_0$ the plan picks, $n\ge$ `--min-n`.
The summary table has a `full device` column that checks this after the fact.

### The pilot's clock underestimates this model (2026-09-28)

The plan's $n$ on a given ladder is $T\cdot\text{throughput}/\sum k$, so the ceiling
must use the plan's throughput. For a batched model that's `pilot.py`'s
$\sum k/\sum(a+bk^d)$ from its cost probe on the pilot's 4..4096. On that ladder the
urn's per-sample fixed cost ($a\approx2.7$ ns, `curand_init` and the write, about 1000
steps' worth) dominates:

| | throughput (steps/s) | $d$ |
|---|---|---|
| pilot probe on 4..4096, four re-runs | 1.80–2.15 × 10¹¹ | 1.07–1.75 (affine) |
| E1 probe on $2^{10}..2^{20}$ | ~3.5 × 10¹¹ | 1.012 ± 0.003 |
| a real final run (smoke, $p=0.9$) | 4.1 × 10¹¹ | – |

So `sweep_p.py` reruns the pilot's own probe (`pilot.resolve_cost`) on the template's
scales and divides by `THROUGHPUT_MARGIN = 1.25` for its spread. The side effect is
that plans are conservative, and a final takes about half of `--time`. $d$ from the
pilot is also poorly identified here: 1.33 ± 0.17 in one smoke pilot, against a true
value of 1. Both are the pipeline measuring $d$ on the pilot's ladder, by design
(`pilot.py`'s `_resolve_d`), and are left as they are. A higher pilot ladder would fix
both, but it's a modelling decision (open question below).

## Acceptance criteria (written 2026-09-28)

**G1. Distribution** (`tools/tests/test_erw_urn_gpu.py`, local, gitignored).
Against `erw` at $k\in\{64,1024\}$ and against `erw_gpu` at $k\in\{2^{14},2^{16}\}$,
each × $p\in\{0.3,0.75,0.9\}$, independent spawned seeds: two-sample KS $p\ge0.01$ and
$\lvert\Delta\bar Y\rvert\le3$ combined se. The tree samplers never form $U_m/m$, so
they're independent references for the urn.

**G2. Exact references** (same file, imported from `test_erw.py`):
- $\chi^2$ against the literal eq. (2.1) law at $k=7$, $n=10^6$;
- $\mathbb E\lvert S_k\rvert$ (DP) and $\mathbb E S_k^2$ (eq. A.3) at $k\in\{64,512,4096\}$,
  $n=2\times10^5$, within 4 se;
- the zero-check at $k=256$ and $2^{18}$;
- $p=1/2$ is `srw`;
- closed cases $p=0$, $p=1$, $k=1$;
- parity and range.

Also checked: the output doesn't depend on the launch chunk and is a prefix in $n$, one
integer is taken from the driver's rng, the flip threshold is exact to $2^{-64}$ at
$p\in\{0,10^{-300},\ldots,1\}$, `cost_hint`/`_check` are `erw`'s own objects, import
doesn't load cupy, and without CUDA `simulate` raises and names `erw`.

**E1. Cost.** `measure_cost.py` on `recipes/cost_probe.json` ($k=2^{10}..2^{20}$), with
the saturation walk. PASS if the affine $\hat d$ is within 20% of the declared $d=1$.

**G3. End to end.** `sweep_p.py` at $p=0.9$ must give $\hat\gamma$ within its Wilson
interval of $2p-1$, with `full device` = yes.

## G1, G2: PASS (2026-09-28)

`python3 -m pytest tools/tests/test_erw_urn_gpu.py`: **62 passed** in 5.8 s. The full
local suite: 592 passed, 12 skipped.

## E1: PASS (2026-09-28, idle card)

```
affine          cost(i) = a + b*i^d  : d_hat = 1.0119   overhead a = 0.00 us   rel_rmse = 0.0112
declared d = 1.0000   (model's cost_hint)
measured d = 1.0119 +/- 0.0031   (affine a + b*i**d)
gap = +3.88 sigma, 1.19% relative
PASS: measured d = 1.0119 against this model's declared 1.0000 (1.2% of it, tolerance 20%)
```

Marginal cost per step, saturated at $n=32768$ at every rung:
4.04, 3.35, 3.06, 2.89, 2.86, 2.79, 2.76, 2.76, 2.77, 2.86, 2.92 ps at
$k=2^{10}..2^{20}$. The small-$k$ excess is the per-sample fixed cost. Against
`erw_gpu`'s E1 (0.50–0.65 ns/step) it's ~200× faster. The +3.9σ is that fixed cost's
curvature over the first rungs: dropping them gives $\hat d$ = 1.003 from $2^{13}$.
A first E1 run while the test suite shared the GPU gave the same $\hat d$.

## G3 and the ceiling: smoke runs (2026-09-28)

`sweep_p.py --tag smoke2 --time 5m --p 0.75 0.9`. The ceiling was `--max-scale
16777216`, on a planner throughput of 2.2 × 10¹¹ / 1.25:

| p | γ true | $\hat\gamma$ ± se | final ladder | n/scale × reps | full device | min |
|---|---|---|---|---|---|---|
| 0.75 | 0.500 | 0.53236 ± 0.00016 | 524288..16777216 | 659,614 × 3 | yes (2.5×) | 2.7 |
| 0.9 | 0.800 | 0.80006 ± 0.00004 | 65536..2097152 | 5,424,026 × 3 | yes | 2.7 |

**G3 PASS at $p=0.9$**: Wilson $[0.7997, 0.8004]$ contains 0.8. At $p=0.75$ the ceiling
bound: the plan would have ended at $2^{25}$. It's conservative, since that ladder would
still have had ~330k samples. At $p=3/4$ the $\sqrt{k\log k}$ law isn't eq. (232)'s form,
so 0.532 is 11_erw's known failure, not the device's (12 read 0.5335 there). Pilot
$\omega_1$ at 0.75: $0.400\pm0.006$, against 12's $0.405\pm0.011$.

A first smoke run with the ceiling sized on the device's own marginal rate
(5.3 × 10¹¹) showed why the planner's rate has to be used: the planner read 1.8 × 10¹¹, so at the cap
$n$ could have fallen to about a third of `min_n`.

## Open

- **The sweep.** `python3 experiments/14_erw_urn_gpu/sweep_p.py --tag 1h --time 1h`
  runs 12's grid (11 × 1h). Its results go to `data/sweep_p_1h.md` and belong here once
  measured.
- **Pilot ladder (user's call).** At this speed a pilot on e.g. 64..65536 costs ~0.5 s
  and would give the planner a clean $d$ and throughput. But it changes where
  $\omega_1$ is measured, and it breaks the like-for-like with 12's pilots.
