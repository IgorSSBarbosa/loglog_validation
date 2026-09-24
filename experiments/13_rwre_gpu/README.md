# Experiment 13 — `rwre_gpu`: the walk on the SSEP, drawn on a GPU

**Question.** Does `MODELS["rwre_gpu"]` sample the same distribution as `MODELS["rwre"]`,
closely enough that every estimator gives the same answer on it? Speed only matters
because it buys bigger budgets and longer ladders. A fast model that fails this check
is a failure.

The model, its four stated decisions (environment, step order, rate, window), its exact
symmetries and what is known about $\gamma$ are in `experiments/02_rwre/README.md` and
aren't repeated here. This experiment asks whether the device changes anything, and then
spends the speed on 02's sieve of $p$. `models/rwre.py` is unchanged and taken as
correct. Nothing in `src/` or `calibration/` changed. In `tools/`, the registry gained
the entry, and `tools/cost_model.py`'s `probe_batched` gained an opt-in saturation walk
(`ModelSpec.latency_bound`, below), which no other model sets.

## Setup

| | |
|---|---|
| hardware | NVIDIA GeForce RTX 5090, 32 GiB (shared card) |
| software | Python 3.10.12, numpy 2.0.2, scipy 1.13.1, **cupy-cuda12x 14.2.0**, CUDA runtime 12.9 |
| model | `models/rwre_gpu.py`: one CUDA block per sample, its window in shared memory, all $k$ steps in one launch (a `RawKernel`). Per step, `rwre`'s order: thread 0 reads and jumps (float64 uniform), then `env_sweeps` brick-wall sweeps with the parity drawn per sample and the disjoint bonds of that parity swapped in place (float32 uniforms). `_check`, `window_width` and `cost_hint` are imported from `rwre` |
| RNG | one `rng.integers(0, 2**63)` per `simulate` call seeds cuRAND's Philox4x32-10 device generator; thread $t$ of sample $j$ uses subsequence $256j+t$. Sample $j$ is the same draw however the call is cut into launches |
| reproducibility | a seed reproduces a GPU run **on the versions above**. It never reproduces an `rwre` run |

### Why a kernel and not a CuPy port (user, 2026-09-23)

The first version ported `rwre`'s step loop to CuPy one to one, as `erw_gpu` does, and
sampled the right law (G1/G2 below passed on it). But each step is ~40 small kernels, so
a call cost ~0.55 ms per step **whatever $n$ was**:

| $k$ | $n=1$ | $n=4$ | $n=256$ | $n=4096$ | $n=32768$ |
|---|---|---|---|---|---|
| 128 | 81 ms | 77 ms | 78 ms | 91 ms | 225 ms |
| 1024 | 672 ms | 546 ms | 579 ms | 947 ms | 4195 ms |

That overhead grows with $k$, so it isn't the fixed per-call cost `batched_cost` assumes,
and `probe_batched` refused. The pilot runs that probe, so every study would have died in
its cost phase. The kernel does the same $k=1024$, $n=32768$ call in 64 ms.

### Why `latency_bound` (user, 2026-09-24)

One sample is a serial chain of $k$ steps on at most 256 threads, so it can't fill the
card. Until ~4000 samples are in flight, extra samples are free: 0.2 ms at $k=128$ for
every $n$ from 1 to 256. `probe_batched` stopped doubling $n$ once a call cleared the
fixed overhead, which happens at $n=1$, and then differenced two equal latencies. With
`latency_bound=True` it keeps doubling until the fastest call at $4n$ takes $\ge3.5\times$
the fastest at $n$, which puts the smaller call at least 7/8 of the way to a full device
and bounds the marginal cost's bias at ~5% low. Every other model's probe is unchanged:
the full local suite gives the same 530 passed, 12 skipped with and without it.

## Acceptance criteria (written 2026-09-23)

**G1. Distribution against `rwre`.** $k\in\{64,1024\}$ × five parameter sets ($p=1/3$,
$p=0$, $p=1/2$, frozen environment `env_sweeps = 0` at $p=0.2$, and $\alpha=0.3$ at
$p=0.3$), $n=4000$ per arm, independent spawned seeds: two-sample KS $p\ge0.01$, and
$\lvert\bar Y_{\rm GPU}-\bar Y_{\rm CPU}\rvert\le3$ combined se.

**G2. Exact references, GPU only.**
- $k=1$: $P(X_1=-1)=\alpha p+(1-\alpha)(1-p)$, within 4 se at $n=10^6$;
- frozen environment, $k=2$: two i.i.d. steps with that same probability, $\chi^2$
  $p>10^{-3}$;
- $p=1/2$ is `srw`: $\chi^2$ against the Binomial law of $S_{64}$ at $\alpha=0.37$,
  $p>10^{-3}$;
- the zero-check $\lvert\mathbb E X_k\rvert<4$ se at $k=256, 1024, 16384$, with parity and
  range;
- $\lvert X_k\rvert$ has the same law at $p$ and $1-p$ (KS, $k=256$, $p=0.2$ against $0.8$);
- closed cases: $\alpha=1$ with $p=0$ always right, and with $p=1$ always left;
  $\alpha=0$ with $p=0$ always left.

Also checked: the output doesn't depend on the launch chunk and is a prefix in $n$ at a
fixed seed, `cost_hint` and `_check` are `rwre`'s own objects, importing never imports
cupy, and without cupy `simulate` raises and names `rwre`.

**E1. Cost.** `measure_cost.py` on `recipes/cost_probe.json` ($k=2^7..2^{14}$), through
`probe_batched` with the saturation walk. PASS if the affine $\hat d$ is within the
driver's 20% of the declared $d=3/2$.

**G3. The control arm, end to end.** $p=1/2$ through `autopilot.py` must reproduce
`01_srw`'s exact constants, $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$, each within 2 se.

## G1, G2: PASS (2026-09-24, on the kernel)

The checks are a local script (tests stay out of git, see CLAUDE.md); 25 of 25 pass:

```
PASS  {'p': 0.3333333333333333} k=64: KS p=1.000  mean cpu=7.274 gpu=7.324  z=+0.41
PASS  {'p': 0.3333333333333333} k=1024: KS p=0.704  mean cpu=35.535 gpu=36.214  z=+1.15
PASS  {'p': 0.0} k=64: KS p=0.888  mean cpu=10.326 gpu=10.164  z=-0.97
PASS  {'p': 0.0} k=1024: KS p=0.114  mean cpu=47.459 gpu=48.635  z=+1.46
PASS  {'p': 0.5} k=64: KS p=0.241  mean cpu=6.474 gpu=6.247  z=-2.12
PASS  {'p': 0.5} k=1024: KS p=0.901  mean cpu=25.367 gpu=25.274  z=-0.22
PASS  {'p': 0.2, 'env_sweeps': 0} k=64: KS p=0.610  mean cpu=5.160 gpu=5.266  z=+1.13
PASS  {'p': 0.2, 'env_sweeps': 0} k=1024: KS p=0.962  mean cpu=14.364 gpu=14.068  z=-1.09
PASS  {'p': 0.3, 'alpha': 0.3} k=64: KS p=0.844  mean cpu=11.201 gpu=11.051  z=-0.92
PASS  {'p': 0.3, 'alpha': 0.3} k=1024: KS p=0.610  mean cpu=134.162 gpu=133.191  z=-0.91
PASS  k=1 P(X=-1)=a p+(1-a)(1-p): z=+0.93
PASS  frozen k=2: two iid steps: chi2 p=0.718
PASS  p=1/2 is srw: chi2 vs Binomial law, k=64: chi2 p=0.559
PASS  zero-check E X_k=0, k=256: z=+0.34; parity and range ok
PASS  zero-check E X_k=0, k=1024: z=-0.69; parity and range ok
PASS  zero-check E X_k=0, k=16384: z=+0.08; parity and range ok
PASS  |X| same law at p and 1-p: KS p=0.862
PASS  closed case alpha=1, p=0: always right: unique=[100]
PASS  closed case alpha=0, p=0: always left: unique=[-100]
PASS  closed case alpha=1, p=1: always left: unique=[-100]
PASS  n=0: empty
PASS  same seed: identical at any chunk, prefix of n: bit-identical
PASS  cost_hint/_check are rwre's: same objects
PASS  registry; import does not import cupy: True True False
PASS  no cupy -> RuntimeError naming rwre: RuntimeError: MODELS["rwre_gpu"] needs cupy ...
```

## E1: PASS, and why it reads low (2026-09-24)

```
affine          cost(i) = a + b*i^d  : d_hat = 1.3902   overhead a = 0.01 us
declared d = 1.4928   (model's cost_hint)
measured d = 1.3902 +/- 0.0443   (affine a + b*i**d)
gap = -2.32 sigma, 6.88% relative
PASS: measured d = 1.3902 against this model's declared 1.4928 (6.9% of it, tolerance 20%)
```

The gap is real, not noise, and it has a measured cause. At fixed $k=1024$ and
$n=16384$ (device full), varying only `window_c`:

| `window_c` | $W$ | µs/sample | ps/(step·site) |
|---|---|---|---|
| 6 | 192 | 0.91 | 4.61 |
| 12 | 384 | 1.76 | 4.46 |
| 24 | 768 | 2.91 | 3.70 |
| 48 | 1536 | 4.60 | 2.92 |
| 96 | 3072 | 8.32 | 2.64 |

Linear in $W$: $t/k = a + bW$ with $a=0.71$ ns per step (the serial part: thread 0's
read and jump, and the barriers every thread waits at) and $b=2.44$ ps per step·site. So
the device's cost is $k\,(a+b\,W(k))$, and its local exponent is
$1+\tfrac12\,bW/(a+bW)$. That's **1.28** at $k=1024$ (exactly what the $p=1/2$ pilot's
clock measured on 4..1024, $1.2813\pm0.0060$), **1.42** at $k=16384$, and 3/2 only as
$k\to\infty$. The declared $d=3/2$ counts the work, which is right. The clock reaches it
only asymptotically, and the pilot plans with the clock.

Throughput, same host, `window_c = 12`, $p=1/3$, per cost unit ($=$ one step on one
site of the window):

| $k$ | GPU | CPU (`rwre`) | speed-up |
|---|---|---|---|
| 256 | 4.58 ps | 21.3 ns | 4,600× |
| 1024 | 4.40 ps | 29.4 ns | 6,700× |
| 4096 | 3.64 ps | 39.3 ns | 10,800× |

## G3: the control arm, $p=1/2$ — PASS (2026-09-24, smoke)

`python3 experiments/13_rwre_gpu/sweep_p.py --tag smoke --time 4m --p 0.5`, i.e.
`autopilot.py` on `recipes/samples_calib_p0.5.json`, 3 replicates, 3.1 min wall clock:

| object | measured | exact | |
|---|---|---|---|
| $\omega_1$ (pilot, 4..1024) | $0.988\pm0.100$ | 1 | $0.12\sigma$ |
| $a_1$ (pilot) | $-0.256\pm0.032$ | $-1/4$ | $0.18\sigma$ |
| $\hat\gamma$, final ladder 256..8192 | $0.50013\pm0.00021$ | 1/2 | $0.6\sigma$ |
| eq. (720) 95% interval | $[0.4989, 0.5013]$ | contains 1/2 | |

The same arm in `02_rwre` (A2) took three CPU replicates on 4..1024 and returned
$0.5002\pm0.0022$. In four minutes the GPU's final ladder reaches 8192 with a replicate se
ten times smaller.

## The sieve (not yet run)

`sweep_p.py` runs `autopilot.py` once per $p$, one after another (they share one GPU),
with the same 1h wall clock and the same pilot recipe each (`samples_calib_p0.5.json`
with only $p$ changed). The grid is 02's A6 sieve plus its headline, in this order:
$p=1/3, 0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5$. $[0,1/2]$ is enough because $\lvert X_k\rvert$
has the same law at $p$ and $1-p$. The question is 02's open one: at $p=1/3$ the
effective exponent was $\ge0.579$ at $k\in[512,1024]$ and still rising, so does it settle,
keep climbing or turn over on a longer ladder?

```bash
python3 experiments/13_rwre_gpu/sweep_p.py --tag 1h          # 8 x 1h, resumable
```

Results go to `data/sweep_p_1h.md` and are recorded here once measured.
