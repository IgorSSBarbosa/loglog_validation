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

## Wilson interval with the AIC pilot refit (2026-11-07)

B_fs now takes $\omega_1, a_1$ from the pilot refitted with $\gamma$ pinned to $\hat\gamma$ on a range of
$\ge 5$ consecutive pilot scales chosen by AIC with data cuts (Jay & Neil 2021;
`local_slope.omega_ranges`, `report.pinned_refit`), not on every pilot scale as in the
2026-11-02 refit. $\hat\gamma$ is unchanged; only the eq. (720) half-width moves. The
$\omega_1$ error is the profile error $\oplus$ the spread over ranges. Old = all-scales pinned
refit (kept in `answer.json` as `pilot_refit_all_scales`), new = AIC refit (`pilot_refit`);
reports rerun 2026-11-07. A warning is $\chi^2$ $p < 0.01$ (old: or $\omega_1$ on its bound). ✓/✗: the interval contains / misses the exact $\gamma$.

| study | $\hat\gamma$ | old $\hat\omega_1$ | old $\chi^2$/dof | old radius | new $\hat\omega_1$ | range | new $\chi^2$/dof | new radius |
|---|---|---|---|---|---|---|---|---|
| `ERW_p0.75_20m` | 0.534430 | 0.308 ± 0.0031 | 91.1/8 ⚠ | 0.0225 | 0.25 ± 0.018 | 32..4096 | 1.73/5 | 0.047 |
| `sweep_p0.2_ERWGPU` | 0.500123 | 1.15 ± 0.019 | 9.59/8 | 0.000186 ✓ | 1.21 ± 0.17 | 16..4096 | 3.2/6 | 0.000171 ✓ |
| `sweep_p0.2_ERW_GPU` | 0.500089 | 1.15 ± 0.019 | 10.2/8 | 0.000187 ✓ | 1.06 ± 0.1 | 8..4096 | 2.73/7 | 0.000232 ✓ |
| `sweep_p0.3_ERW_GPU` | 0.500074 | 1.06 ± 0.024 | 3.51/8 | 0.000206 ✓ | 1.06 ± 0.12 | 4..4096 | 3.51/8 | 0.000206 ✓ |
| `sweep_p0.4_ERW_GPU` | 0.500028 | 0.875 ± 0.031 | 8.04/8 | 0.0003 ✓ | 0.779 ± 0.14 | 8..4096 | 3.95/7 | 0.00044 ✓ |
| `sweep_p0.5_ERW_GPU` | 0.500093 | 1.07 ± 0.053 | 3.87/8 | 0.000155 ✓ | 1.07 ± 0.14 | 4..4096 | 3.87/8 | 0.000155 ✓ |
| `sweep_p0.6_ERW_GPU` | 0.500192 | 0.714 ± 0.02 | 5.46/8 | 0.000506 ✓ | 0.725 ± 0.055 | 4..2048 | 2.92/7 | 0.000482 ✓ |
| `sweep_p0.725_ERW_GPU` | 0.514138 | 0.309 ± 0.0042 | 54.7/8 ⚠ | 0.0169 ✓ | 0.243 ± 0.027 | 64..4096 | 3.27/4 | 0.041 ✓ |
| `sweep_p0.75_ERWGPU` | 0.531716 | 0.289 ± 0.0031 | 44.1/8 ⚠ | 0.0208 | 0.296 ± 0.047 | 128..4096 | 1.92/3 | 0.0202 |
| `sweep_p0.75_ERW_GPU` | 0.533544 | 0.301 ± 0.0032 | 65.6/8 ⚠ | 0.0201 | 0.23 ± 0.052 | 16..256 | 1.37/2 | 0.0576 |
| `sweep_p0.775_ERW_GPU` | 0.564643 | 0.319 ± 0.0034 | 60.8/8 ⚠ | 0.0149 ✓ | 0.271 ± 0.067 | 16..256 | 1.14/2 | 0.0285 ✓ |
| `sweep_p0.7_ERW_GPU` | 0.506394 | 0.363 ± 0.0054 | 45.7/8 ⚠ | 0.00855 ✓ | 0.43 ± 0.062 | 4..128 | 1.08/3 | 0.00396 ✗ |
| `sweep_p0.8_ERW_GPU` | 0.605853 | 0.351 ± 0.0043 | 42.8/8 ⚠ | 0.00998 ✓ | 0.367 ± 0.048 | 64..2048 | 1.03/3 | 0.00869 ✓ |
| `sweep_p0.9_ERW_GPU` | 0.800261 | 0.65 ± 0.012 | 3.31/8 | 0.000497 ✓ | 0.65 ± 0.025 | 4..4096 | 3.31/8 | 0.000497 ✓ |

Coverage of the exact $\gamma$: old 11/11, new 10/11.
