# Experiment 08 — `percolation_zd_gpu`: every anchor, geometry and dimension on a GPU

**Question.** Is `MODELS["percolation_zd_gpu"]` equal in distribution to
`MODELS["percolation_zd"]` across the dimensions, anchors and geometries experiment 05
uses, and does the γ̂ pipeline give the same answer on it in high dimension, where the CPU
ladder is shortest? The ported `crossing_fraction_gpu` (the H0 criticality diagnostic)
is checked here too. The CPU model is unchanged and taken as correct.

Built from `models/percolation2d_gpu.py` (experiment 07), with the same device, RNG and
blocking design. The setup table in `experiments/07_percolation2d_gpu/README.md` applies
unchanged: RTX 5090, cupy-cuda12x 14.2.0, CUDA runtime 12.9, numpy 2.0.2, scipy 1.13.1.

## Acceptance criteria (written 2026-09-16, before any run below)

**A. Distribution — `tools/tests/test_percolation_zd_gpu.py`** (local, gitignored).
Grid: dim ∈ {2, 3, 4, 6} × anchor ∈ {face, face_far, origin, slab k=1} × geometry ∈
{box, cylinder, torus}, i = 16, 8, 6, 4, plus `percolation_zd_gpu` vs `percolation2d_gpu`
at dim 2. That is 49 cells, n = 20 000 per arm, independent fixed seeds. **Family-wise 1%**
(user, 2026-09-16): per cell, KS p ≥ 0.01/49 = 2.04·10⁻⁴ and |mean z| ≤ 3.71. At 07's
per-cell 1% thresholds a correct model would fail some cell about 45% of the time.
Raising n from 4000 to 20 000 recovers the power the stricter threshold costs.

**B. Closed forms and exact identities** (same file). Zero rate for `face` at
(dim, i) = (2, 4) and (3, 2), box and torus, inside the 99.9% Wilson interval. The
1-D mean equals `expected_face_count_1d(20, 0.7)` within 3.29 se. p = 0 gives zeros and
p = 1 gives i^dim (face, origin, slab) or ⌈i/2⌉·i^(dim−1) (face_far), for dim ∈ {2, 3, 6}
× every geometry. At the same seed, `slab` k = 0 equals `origin` exactly, and `south`
equals `face` exactly. `crossing_fraction_gpu` agrees with the CPU one within a 99.9%
two-proportion interval at (3, 8) and (6, 4), and is exactly 0 at p = 0 and 1 at p = 1.
Declarations (`cost_hint`, p_c table) are the CPU objects. Import leaves cupy unloaded,
and both entry points raise without CUDA, naming `percolation_zd`.

**C. γ̂ end to end.** H3's `face_far`/cylinder recipes at dim 3 (s = 8…128) and dim 5
(s = 4…32), 4·10⁸ each, with only `"model"` changed. R = 12 replicates per arm,
`compare_observables.py`, using m₀ = 2 at dim 3 and m₀ = 1 at dim 5 (4 rungs). For each
dim × estimator: Welch **|z| ≤ 2.5**, four comparisons, ≈ 5% family-wise false alarm.
GPU containment in the CPU 95% CI is reported, not gated (07 explains why).

**D. Per-scale means, deep.** H2's 4·10⁹ runs, `face`/cylinder at dim 3 (s = 8…256) and
`face_far`/cylinder at dim 5 (s = 4…32). At each of the 10 scales,
|Ȳ_GPU − Ȳ_CPU| ≤ 3 combined se (≈ 3% family-wise). Pooled Σz² also reported.

**E. Cost probe (recorded, not gating).** `cost_gpu_d3.json`. 07 already showed that
the n = 1 probe measures GPU overhead rather than i^dim, so this only records whether
that holds in 3-D.

**Speed (recorded).** GPU vs CPU at the bottom and top rung of the dim 3 and dim 5
deep allocations.

## Step 1 — memory (2026-09-16)

CuPy pool peak per padded site, worst of box/torus × face_far/origin:

| dim | 2 | 3 | 3 | 4 | 5 | 6 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| i | 256 | 64 | 256 | 32 | 16 | 8 | 16 | 6 |
| bytes/site | 9.5 | 9.5 | 9.6 | 9.4 | 9.3 | 9.0 | 9.3 | 8.6 |

Flat in dim: `cupyx` label's buffers don't grow with the 3^dim structure.
`_BYTES_PER_SITE = 16` is kept from 07.

## Results (2026-09-16)

**Verdict: PASS on A, B, C and D.** `percolation_zd_gpu` is equal in distribution to
`percolation_zd` in every cell tested, from dim 1 to dim 6, for all four anchors and all
three geometries. γ̂ from the GPU is indistinguishable from CPU γ̂ at dim 3 and dim 5.
E disagrees with the declared d for the reason 07 found.

### A — distribution grid (PASS, 49/49)

```bash
python3 -m pytest tools/tests/test_percolation_zd_gpu.py            # 85 passed in 146.6s
CUDA_VISIBLE_DEVICES="" python3 -m pytest tools/tests/test_percolation_zd_gpu.py   # 13 passed, 72 skipped
```

Over the 48 CPU-vs-GPU cells, n = 20 000 per arm:

| dim | i | cells | min KS p | max \|mean z\| |
|---|---|---|---|---|
| 2 | 16 | 12 | 0.201 | 2.02 |
| 3 | 8 | 12 | 0.040 | 2.38 |
| 4 | 6 | 12 | 0.085 | 2.50 |
| 6 | 4 | 12 | 0.138 | 1.77 |

Limits were p ≥ 2.04·10⁻⁴ and |z| ≤ 3.71, and no cell comes within an order of magnitude
of the p limit. Pooled mean z: Σz² = 46.6 on 48 dof, **p = 0.53**. The KS p-values are
*not* uniform (KS-vs-uniform p = 7·10⁻⁵) and skew high. That is expected for integer
data with many ties, which makes the two-sample KS conservative. It is not evidence of
agreement beyond chance, which is why the pooled z is reported beside it. The dim-2
cross-model cell (`percolation_zd_gpu` vs `percolation2d_gpu`) passes too. Per-cell
numbers: `$SCRATCHPAD/ks08.json` (not kept).

### B — closed forms and exact identities (PASS)

All pass as stated: zero rates, the 1-D mean, p = 0/1 in all 36 dim × geometry × anchor cells, slab k = 0 ≡ origin and south ≡ face bit for bit at the same
seed, one rng integer per call, several blocks, and the forced merge cap raising.

`crossing_fraction_gpu` against the CPU, n = 20 000 each:

| (dim, i) | CPU | GPU | z | CPU time | GPU time |
|---|---|---|---|---|---|
| (3, 8) | 0.3073 | 0.3049 | −0.50 | 0.18 s | 0.23 s |
| (6, 4) | 0.5199 | 0.5181 | −0.35 | 9.84 s | 0.02 s |

### C — γ̂ end to end (PASS)

```bash
python3 src/estimate/compare_observables.py --arm cpu=experiments/08_percolation_zd_gpu/recipes/samples_cpu_d3_facefar.json \
  --arm gpu=experiments/08_percolation_zd_gpu/recipes/samples_gpu_d3_facefar.json \
  --replicates 12 --m0 2 --truth 2.523 --seed 20260916 --tag compare_d3_facefar
python3 src/estimate/compare_observables.py --arm cpu=experiments/08_percolation_zd_gpu/recipes/samples_cpu_d5_facefar.json \
  --arm gpu=experiments/08_percolation_zd_gpu/recipes/samples_gpu_d5_facefar.json \
  --replicates 12 --m0 1 --truth 3.54 --seed 20260917 --tag compare_d5_facefar
```

| dim | estimator | CPU mean ± se | GPU mean ± se | Welch z | GPU in CPU 95% CI |
|---|---|---|---|---|---|
| 3 | all points | 2.5346 ± 0.0060 | 2.5267 ± 0.0045 | **−1.06** | yes |
| 3 | m₀ = 2 | 2.5374 ± 0.0141 | 2.5284 ± 0.0118 | **−0.49** | yes |
| 5 | all points | 3.8396 ± 0.0185 | 3.8291 ± 0.0141 | **−0.45** | yes |
| 5 | m₀ = 1 | 3.7494 ± 0.0275 | 3.7321 ± 0.0281 | **−0.44** | yes |

Per-scale cv agrees (dim 3: CPU 0.988→0.835, GPU 0.983→0.829). Wall clock per replicate:
**dim 3, 5.9 s CPU against 0.05 s GPU; dim 5, 7.5 s against 0.1 s.** Both arms
reproduce H2's picture at this small budget: dim 5 `face_far` is bias-dominated at
4·10⁸, with only 9 samples at i = 32. Neither γ̂ is a d_f measurement at this budget,
and none was claimed. As in 07, the driver's "gpu wins by 1.50×" line ranks two RMSEs
without a significance test.

### D — per-scale means at 4·10⁹ (PASS)

```bash
python3 src/generate/generate.py -meta experiments/08_percolation_zd_gpu/recipes/samples_{cpu,gpu}_deep_d3_face.json    --tag {cpu,gpu}_deep_d3_face
python3 src/generate/generate.py -meta experiments/08_percolation_zd_gpu/recipes/samples_{cpu,gpu}_deep_d5_facefar.json --tag {cpu,gpu}_deep_d5_facefar
```

Wall clock: dim 3 **56.3 s CPU, 1.24 s GPU**; dim 5 **81.6 s CPU, 1.49 s GPU**.

| run | i | n | Ȳ CPU | Ȳ GPU | rel. se | z |
|---|---|---|---|---|---|---|
| dim 3 face | 8 | 27 954 | 83.199 | 82.802 | 0.37% | −1.28 |
| | 16 | 9 883 | 522.38 | 521.68 | 0.54% | −0.25 |
| | 32 | 3 494 | 3 169.7 | 3 185.4 | 0.83% | +0.60 |
| | 64 | 1 235 | 19 166 | 19 048 | 1.30% | −0.47 |
| | 128 | 436 | 115 123 | 112 803 | 2.09% | −0.96 |
| | 256 | 154 | 686 376 | 678 280 | 3.35% | −0.35 |
| dim 5 face_far | 4 | 17 781 | 14.123 | 14.171 | 1.12% | +0.31 |
| | 8 | 3 143 | 231.01 | 231.52 | 1.73% | +0.13 |
| | 16 | 555 | 3 123.5 | 3 305.6 | 3.14% | +1.86 |
| | 32 | 98 | 40 173 | 42 180 | 6.61% | +0.76 |

Max |z| = 1.86 ≤ 3. Pooled Σz² = 7.46 on 10 dof, **p = 0.68**.

### E — cost probe (recorded; disagrees, as 07 predicted)

Affine d̂ = **2.34 ± 0.20**, overhead a = **1576 µs**, against the declared 3 (−3.4σ). The
fixed cost is 108% of the measurement at i = 8. `measure_cost.py` times one sample per
call, and on a GPU that measures launch and sync overhead, not i^dim. Allocation uses
the declared `cost_hint`, which is unchanged.

### Speed at the deep allocations (recorded)

| dim, anchor | i | CPU Msites/s | GPU Msites/s | speed-up |
|---|---|---|---|---|
| 3, face | 8 (n = 27 954) | 45.5 | 4 187 | 92× |
| 3, face | 256 | 72.9 | 4 222 | 58× |
| 5, face_far | 4 (n = 17 781) | 17.1 | 4 118 | **240×** |
| 5, face_far | 32 | 53.7 | 3 567 | 66× |

GPU throughput is ~4·10⁹ sites/s on the cylinder at dim 3–5, flat across i. The CPU is
slowest at small boxes in high dim, so the gain is largest exactly where H8's throughput
collapsed. On the 3-D cylinder the GPU runs at about 40% of its 2-D rate (07: ~10¹⁰),
because the merge joins dim − 1 face pairs.

## What this unlocks, not yet run

At ~4·10⁹ sites/s, 4·10¹¹ sites is under two minutes. That makes the ladder extensions
plan §5 named reachable: dim 3 to i = 512–1024 (≥ 7 rungs for ω₁), dim 5 `face_far` to
i = 64, dim 6 to i = 32. With `crossing_fraction_gpu`, an H0 at dim 6 with
~500× the CPU's samples becomes possible too. Each is a new question, and needs its own
criteria and sign-off.
