# Experiment 12 — `erw_gpu`: the elephant random walk, drawn on a GPU

**Question.** Does `MODELS["erw_gpu"]` sample the same distribution as `MODELS["erw"]`,
closely enough that every estimator gives the same answer on it? Speed only matters
because it buys bigger budgets and longer ladders. A fast model that fails this check
is a failure.

The model, its regimes and the acceptance criteria for $\gamma$ are in
`experiments/11_erw/README.md` and aren't repeated here. This experiment asks only
whether the device changes anything. Nothing in `tools/`, `src/` or `calibration/`
changed, and `models/erw.py` is unchanged and taken as correct.

## Setup

| | |
|---|---|
| hardware | NVIDIA GeForce RTX 5090, 32 GiB (shared card), driver 595.58.03 |
| software | Python 3.10.12, numpy 2.0.2, scipy 1.13.1, **cupy-cuda12x 14.2.0**, CUDA runtime 12.9 |
| model | `models/erw_gpu.py`: `models/erw.py`'s random-recursive-tree sampler with `_resolve` ported line for line to CuPy (user, 2026-09-23). float64 uniforms, int32 ancestor table, pointer doubling by `take_along_axis` |
| RNG | one `rng.integers(0, 2**63)` per `simulate` call from the driver's generator seeds cuRAND XORWOW. The block size is a fixed function of $k$ (`block_rows`: 2 GiB / 64 B per step) |
| reproducibility | a seed reproduces a GPU run **on the versions above**. It never reproduces an `erw` run |

## Acceptance criteria (written 2026-09-23)

**G1. Distribution: `tools/tests/test_erw_gpu.py`** (local, gitignored). $k\in\{64,
1024\}\times p\in\{0.3, 0.75, 0.9\}$, $n=4000$ per arm, independent spawned seeds:
two-sample KS $p\ge0.01$, and $\lvert\bar Y_{\rm GPU}-\bar Y_{\rm CPU}\rvert\le3$
combined se.

**G2. Exact references** (same file, imported from `test_erw.py` so both models face the
same ones):
- $\chi^2$ against the literal eq. (2.1) law at $k=7$ ($p$-value $>10^{-3}$);
- $\mathbb E\lvert S_k\rvert$ (eq. 2.2 chain, by DP) and $\mathbb E S_k^2$ (eq. A.3) at
  $k\in\{64,512\}$, within 4 se;
- the zero-check $\lvert\mathbb E S_k\rvert<4$ se;
- $p=1/2$ is `srw`;
- the closed cases $p=0$, $p=1$, $k=1$;
- parity and range.

Also checked: `cost_hint` and `_check` are the CPU model's own objects, importing never
imports cupy, and without CUDA `simulate` raises and names `erw`.

**E1. Cost.** `measure_cost.py` on `recipes/cost_probe.json` ($k=2^{10}..2^{20}$),
through `probe_batched` (the registry sets `batched_cost`). PASS if the affine $\hat d$
is within the driver's 20% of the declared $d=1$.

**G3. Estimator, end to end: not run.** `samples_calib_p0.5.json` and
`samples_super_p0.9.json` are `11_erw`'s E2 and E6 arms with only `"model"` changed.
They are for comparing the pilot/plan/run/report pipeline between CPU and GPU once
`11_erw`'s $p$ grid, ladder and time per arm are decided. The criterion to use is
experiment 07's C: Welch $\lvert z\rvert\le2.5$ on $\hat\gamma$ over replicates.

## Step 1: memory and throughput (2026-09-23)

CuPy pool peak for one full block at $p=0.9$, fresh process per $k$:
**45.8, 46.0, 46.1, 46.1, 46.1 bytes/step** at $k=2^6, 2^{10}, 2^{14}, 2^{18}, 2^{20}$.
Flat in $k$, and that's where `_BYTES_PER_STEP = 64` comes from.

Throughput, $n\approx2^{26}/k$ samples on the GPU against $1/16$ of that on the CPU, same host:

| $k$ | GPU ns/step | CPU ns/step | speed-up |
|---|---|---|---|
| $2^6$ | 0.62 | 32.7 | 53× |
| $2^{10}$ | 0.51 | 36.9 | 73× |
| $2^{14}$ | 0.51 | 29.7 | 58× |
| $2^{17}$ | 0.60 | 30.9 | 52× |
| $2^{20}$ | 0.61 | 43.4 | 71× |

## G1, G2: PASS (2026-09-23)

`python3 -m pytest tools/tests/test_erw_gpu.py tools/tests/test_erw.py`: 134 passed. The
full `tools/tests/` suite: 530 passed, 12 skipped.

## E1: PASS (2026-09-23)

```
         k      cost_ms          n
      1024    0.0005139      16384
     16384     0.008217       1024
    262144       0.1697         32
   1048576       0.6763          8
affine a + b*i**d : d_hat = 1.0491 +/- 0.0137, overhead a = 0.05 us
gap = +3.58 sigma, 4.91% relative -> PASS (tolerance 20%)
drop-leading m0 = 0..9: 1.038 1.041 1.042 1.054 1.075 1.080 1.073 1.051 0.997 0.995
```

The fixed cost cancels, as it does for the other `*_gpu` models. What remains is a real
rise in the time per step: 0.50 ns at $2^{10}$ and $2^{14}$, 0.65 ns from $2^{18}$ on.
That $+3.6\sigma$ excess matches the $\log\log k$ extra doubling pass that
`models/erw.py`'s `cost_hint` docstring sets aside, and maybe a working set outgrowing
the L2 cache (not measured separately). The top three rungs double exactly with $k$ ($\hat d=0.997$). This matches the
CPU model's E1 in `11_erw`: the declared $d=1$ is right up to the $\log\log$ factor, and
the clock resolves that factor only at this ladder's precision.

## Demo (2026-09-23)

`recipes/samples_example.json` (`11_erw`'s demo, $p=0.9$, $k=64..8192$, $n=4000$, only
the model and seed changed): all-points $\hat\gamma=0.8034$, drop-leading
0.8026–0.8054, against $2p-1=0.8$ and the CPU demo's 0.8024. Indicative only: no se, and
not a criterion.
