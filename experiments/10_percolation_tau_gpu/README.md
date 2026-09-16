# Experiment 10 — `percolation_tau_gpu`: the cluster-number density on a GPU

**Question.** Is `MODELS["percolation_tau_gpu"]` equal in distribution to
`MODELS["percolation_tau"]`, for `simulate` per rung and for `shared_sampler` jointly
across rungs, and does experiment 03's τ pipeline (independent rungs, bin and tail, and
the shared lattice) give the same γ̂ on it? The CPU model is unchanged and taken as
correct.

Design: draw, stack-and-label and the torus merge are `percolation_zd_gpu`'s at dim = 2
(experiment 08). Declarations (box rules, windows, `cost_hint`) are imported from the CPU
model. The counting is the CPU's per-label size and label → sample map, built with
`cupy.add.at` and one scatter; `bincount` and `**` are not used (experiments 07, 09).
Setup as in `experiments/07_percolation2d_gpu/README.md` (RTX 5090, cupy-cuda12x 14.2.0,
CUDA runtime 12.9).

## Acceptance criteria (written 2026-09-16, with the test file, before its first run)

Before these were written, one smoke run (seeds 1–4, not the test seeds) checked that the
module ran and printed KS p and rung correlations. It is not evidence for anything below.

**A. Distribution — `tools/tests/test_percolation_tau_gpu.py`** (local, gitignored).
One family at 1%, n = 20 000 per arm, independent fixed seeds:
- `simulate`: observable ∈ {bin, tail} × geometry ∈ {torus, box} × s ∈ {8, 64}, defaults
  otherwise (box_factor 16, box_exponent 1/2). 8 comparisons, each KS + mean z.
- `shared_sampler`, scales 8…128 (L = 102), torus and box:
  - per rung, KS + mean z: 10 comparisons;
  - **covariance between every pair of rungs** (user, 2026-09-16): Welch z on the mean of
    the centred product (Y_s − Ȳ_s)(Y_s′ − Ȳ_s′). Rows are i.i.d., so this is a plain
    CLT with no normality assumption. 20 comparisons.
- 38 comparisons in total. Each needs KS p ≥ 0.01/38 = 2.63·10⁻⁴ and |z| ≤ 3.65.

**B. Closed forms and identities** (same file).
- E Y_s against **exact enumeration** of all 2⁹ configurations of a 3×3 lattice at p_c
  (s = 2, bin and tail × torus and box), using the CPU model's own labelling and counting.
  The GPU mean over n = 100 000 must lie within 3.29 se.
- p = 1 gives exactly 1/L² in the window that holds L², 0 elsewhere; p = 0 gives 0 (both
  samplers).
- Same seed on device: `simulate` equals `binned_counts_gpu`'s column; bin(s) =
  tail(s) − tail(2s) sample by sample; `shared_sampler` equals `binned_counts_gpu`/L², and
  its `info` (the `shared_lattice` stamp, L, windows, cut ratios) equals the CPU sampler's.
- Same seed gives the same draws, one driver integer per call, blocks spanned.
- Declarations are the CPU objects; the rejections match the CPU's. Importing leaves cupy
  unloaded, and without CUDA both samplers raise naming the CPU model.

**C. γ̂ end to end** (user, 2026-09-16: all three recipes, R = 12). Experiment 03's
`samples_tau_torus.json` (bin), `samples_tau_tail.json` and `samples_tau_shared.json`,
copied with only `"model"` changed. s = 8…2048, truth γ = 1 − 187/91 = −96/91, m₀ = 3
(03's reported window).
- bin and tail: `compare_observables.py`, CPU arm vs GPU arm, R = 12 each.
- shared: `generate_shared` has no replicate driver, so 12 streams spawned from one root
  seed per arm feed `generate_shared(...)`, and γ̂ comes from `tools/loglog.py`'s
  `gamma_all_points` and `gamma_drop_leading` exactly as `compare_observables.py` uses them.
- Welch |z| ≤ 2.5 for each recipe × estimator: six comparisons.

**D. Per-scale means, one full run per arm.**
- bin recipe via `generate.py`: |z| ≤ 3 at every one of the 9 scales; pooled Σz² reported.
- shared recipe via `generate_shared.py`: |z| ≤ 3 at every rung. No pooled χ², because
  the rungs are correlated.

**E. Cost probe (recorded, not gating).** `cost_gpu.json`, 03's probe with the model
changed.

## Step 1 — memory (2026-09-16)

Pool peak with draw, label, torus/box merge and nine windows counted, per padded site:

| L | rows | torus | box |
|---|---|---|---|
| 64 | 20 000 | 9.12 | 9.12 |
| 256 | 1 200 | 9.19 | 9.19 |
| 1024 | 80 | 9.19 | 9.19 |
| 2048 | 20 | 9.20 | 9.20 |

The counting adds nothing above the labelling's own peak: the owner scatter and the
per-window `add.at` reuse blocks the draw has already freed. `_BYTES_PER_SITE = 16`, as in
07 and 08.

## Results (2026-09-16)

**Verdict: PASS on A–D. E recorded, outside the driver's tolerance, for the n = 1 overhead
reason 07 and 08 record.** `percolation_tau_gpu` is equal in distribution to
`percolation_tau`, per rung and, for `shared_sampler`, in the covariance between rungs.
All three of 03's τ recipes give the same γ̂ on both devices.

### A — distribution, one family of 38 (PASS, 38/38)

```bash
python3 -m pytest tools/tests/test_percolation_tau_gpu.py                          # 74 passed in 27.5s
CUDA_VISIBLE_DEVICES="" python3 -m pytest tools/tests/test_percolation_tau_gpu.py  # 18 passed, 56 skipped
```

| comparisons | count | min KS p | max \|z\| |
|---|---|---|---|
| `simulate`, bin/tail × torus/box × s ∈ {8, 64} | 8 | 0.370 | 1.47 |
| `shared_sampler` per rung, s = 8…128, torus and box | 10 | 0.344 | 1.32 |
| `shared_sampler` covariance, every rung pair, torus and box | 20 | — | 1.55 |

Σz² over the 38 z = 28.1 (p = 0.88; indicative only, since the shared z are correlated).
Zero fractions agree (bin: torus 4.3–4.5%, box 0.6–0.8%, on both devices; tail: 0).

Rung correlations (Pearson r, informational), CPU → GPU: torus 0.101–0.123 → 0.103–0.126,
box 0.080–0.111 → 0.082–0.110. Both are the flat ≈ +0.1 pedestal
`models/README.md` records for the CPU sampler.

**Power of the covariance test (a diagnostic, not a criterion).** Permuting each GPU
rung's rows independently keeps every marginal and removes the correlation. Against the
torus CPU arm the covariance z then comes out at −8.3 to −12.2 on every pair, far past the
3.65 limit. So the test detects lost correlation between rungs; it is not passing only
because it has no power.

### B — closed forms and identities (PASS)

Exact enumeration (3×3, s = 2, p_c), p = 1 and p = 0 for both samplers, the same-seed
identities (simulate = `binned_counts_gpu` column; bin(s) = tail(s) − tail(2s);
`shared_sampler` = `binned_counts_gpu`/L² with the CPU's `info`), seed and stream
behaviour, block spanning, declaration identity, rejections, the no-cupy import and the
no-CUDA error: all pass, within the 74 above.

### C — γ̂ end to end, R = 12 per arm (PASS)

```bash
R=experiments/10_percolation_tau_gpu/recipes
python3 src/estimate/compare_observables.py --arm cpu=$R/samples_cpu_bin.json --arm gpu=$R/samples_gpu_bin.json \
  --replicates 12 --m0 3 --truth -1.054945054945055 --seed 20260920 --tag compare_bin
python3 src/estimate/compare_observables.py --arm cpu=$R/samples_cpu_tail.json --arm gpu=$R/samples_gpu_tail.json \
  --replicates 12 --m0 3 --truth -1.054945054945055 --seed 20260921 --tag compare_tail
```

The shared arm used a scratch replicate loop, run with root seed 20260922. It spawns 24
streams from one `SeedSequence`, CPU arm first. Each stream feeds
`src.generate.generate_shared.generate_shared(model, scales, n_lattices, params, seed=stream)`.
From each run's per-rung Ȳ it takes `tools.loglog.gamma_all_points` and
`compare_observables._gamma_at_m0(..., 3)`, the same functions `compare_observables.py` uses.

| recipe | estimator | CPU mean ± se | GPU mean ± se | Welch z |
|---|---|---|---|---|
| bin | all points | −1.0279 ± 0.0019 | −1.0279 ± 0.0013 | **+0.03** |
| bin | m₀ = 3 | −1.0454 ± 0.0042 | −1.0443 ± 0.0027 | **+0.23** |
| tail | all points | −1.0307 ± 0.0003 | −1.0304 ± 0.0002 | **+0.94** |
| tail | m₀ = 3 | −1.0425 ± 0.0007 | −1.0414 ± 0.0005 | **+1.31** |
| shared | all points | −1.0273 ± 0.0004 | −1.0279 ± 0.0004 | **−0.99** |
| shared | m₀ = 3 | −1.0437 ± 0.0009 | −1.0449 ± 0.0008 | **−0.99** |

All six |z| ≤ 1.31 ≤ 2.5. Per-scale cv agrees: bin 0.581→0.628 CPU vs 0.581→0.642 GPU;
tail 0.420→0.447 vs 0.421→0.444. Every arm sits 0.010–0.013 above γ = −96/91 at m₀ = 3.
That is 03's recorded bias on this ladder, the same on both devices. These rows compare
devices, not estimates of τ.

Wall clock per replicate: bin 5.9 s CPU → 0.30 s GPU, tail 53.3 s → 2.7 s, shared
17.0 s → 0.86 s, i.e. **≈ 20×**. The three CPU arms ran concurrently on the 32-core host.

**Recipe note.** 03's `samples_tau_torus.json` (bin) asks for a budget of 1.5625·10⁶, i.e.
4·10⁸ sites. That is 10× less than `samples_tau_tail.json` (4·10⁹), whereas
`models/README.md`'s table describes the bin run at 4·10⁹. The recipes were copied as-is,
as the criteria say.

### D — per-scale means, one full run per arm (PASS)

```bash
python3 src/generate/generate.py        -meta $R/samples_cpu_bin.json    --tag cpu_bin      #  6.6 s
python3 src/generate/generate.py        -meta $R/samples_gpu_bin.json    --tag gpu_bin      #  1.1 s
python3 src/generate/generate_shared.py -meta $R/samples_cpu_shared.json --tag cpu_shared   # 18.9 s (L = 453, 1.25e9 sites)
python3 src/generate/generate_shared.py -meta $R/samples_gpu_shared.json --tag gpu_shared   #  1.6 s
```

| s | bin: n | bin z | shared z (n = 6100) |
|---|---|---|---|
| 8 | 3790 | −0.22 | −1.06 |
| 16 | 2682 | −0.06 | −0.40 |
| 32 | 1899 | +0.08 | −0.63 |
| 64 | 1344 | +0.16 | −1.20 |
| 128 | 951 | +0.07 | +0.64 |
| 256 | 673 | +0.38 | +0.54 |
| 512 | 476 | +0.30 | +0.39 |
| 1024 | 337 | −1.14 | −0.23 |
| 2048 | 238 | −0.81 | +0.18 |

bin: max |z| = 1.14, pooled Σz² = 2.28 on 9 dof (p = 0.99). shared: max |z| = 1.20 (rel. se
0.08–1.3%); no pooled χ², because the rungs are correlated.

### E — cost probe (recorded, not gating: FAIL at the driver's 20%)

```bash
python3 src/estimate/measure_cost.py -meta $R/cost_gpu.json --tag cost_gpu
```

Run alone, with the GPU otherwise idle. Affine d̂ = **1.287 ± 0.172** against the declared
0.9993 (+1.7σ, 28.8%, outside the driver's 20%). Overhead a = 1693 µs, 101% of the
measured cost at s = 16. The pure-power d̂ is 0.044. The ladder's largest box is 1024² ≈
10⁶ sites, well under a millisecond of device work, so the n = 1 probe measures the fixed
per-call cost, as in 07 and 08. A first probe, run while the C jobs shared the GPU, gave
1.167 ± 0.220. 03's CPU probe measured 1.049 ± 0.019. What a GPU cost probe should measure
(larger n per call) is still the open follow-up.
