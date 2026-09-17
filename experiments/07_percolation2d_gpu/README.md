# Experiment 07 — `percolation2d_gpu`: the same observable, drawn on a GPU

**Question.** Does `MODELS["percolation2d_gpu"]` sample the same distribution as
`MODELS["percolation2d"]`, closely enough that every estimator in this repo gives
the same answer on it? Speed matters only as the means to larger budgets and wider
ladders. A fast model that fails this is a failure.

This is the first of the GPU experiments planned in `plan/gpu_models.md` (local, gitignored),
and the reference implementation the other percolation GPU models will be built from.
Nothing in `tools/`, `src/` or `calibration/` changed. The CPU model is unchanged and
is taken as correct.

## Setup

| | |
|---|---|
| hardware | NVIDIA GeForce RTX 5090, 32 GiB (shared card), driver 595.58.03; host i9-13900K |
| software | Python 3.10.12, numpy 2.0.2, scipy 1.13.1, **cupy-cuda12x 14.2.0**, CUDA runtime 12.9 |
| model | `models/percolation2d_gpu.py`: CuPy port of draw → stack → `cupyx.scipy.ndimage.label` → wrap merge → count, all on device |
| RNG | one `rng.integers(0, 2**63)` per `simulate` call from the driver's generator seeds cuRAND XORWOW. The block size is a fixed function of `i` |
| reproducibility | a seed reproduces a GPU run **on the versions above**. It never reproduces a CPU run |

## Acceptance criteria (written 2026-09-16, before any run below)

**A. Distribution — `tools/tests/test_percolation2d_gpu.py`** (local, gitignored).
For anchor ∈ {south, origin} × geometry ∈ {box, cylinder} × i ∈ {8, 32}, n = 4000 per
arm, independent fixed seeds:
two-sample KS p ≥ 0.01, and |mean_GPU − mean_CPU| ≤ 3 combined se.

**B. Closed forms** (same file). Zero fraction at i = 4 inside the 99.9% Wilson interval
of `zero_rate` = (1−p_c)^4 = 0.02747, n = 20 000, both geometries. p = 0 gives all zeros
and p = 1 gives i², for every anchor × geometry. `cost_hint` is the CPU model's own
function object. Importing never imports cupy. Without CUDA, `simulate` raises
and names `percolation2d`.

**C. Estimator, end to end.** P4's cylinder recipe (south, s = 8…512, Neyman, 4·10⁸),
changed only in `"model"`. R = 12 replicates per arm through
`src/estimate/compare_observables.py`, with independent spawned seeds. For both
estimators it reports (all points, and m₀ = 2):
Welch z = (mean_GPU − mean_CPU)/√(se²_CPU + se²_GPU), **|z| ≤ 2.5**. Also reported:
whether the GPU mean lies inside the CPU arm's 95% CI for its mean.

*Why z and not only "inside the CPU CI"* (refined before running). With both arms equally
noisy, a correct GPU mean falls outside the CPU's own 95% CI about 12% of the time,
because the CI ignores the GPU arm's variance. So containment is reported, but it doesn't
decide pass or fail.

**D. Per-scale means, deep.** One 4·10⁹-unit run per model (P4's deep configuration,
`samples_*_deep.json`). At every scale, |Ȳ_GPU − Ȳ_CPU| ≤ 3 combined se. Seven scales,
so a correct model fails one by chance with probability ≈ 2%.

**E. Cost cross-check (recorded, not gating).** `measure_cost.py` on
`cost_gpu_cylinder.json` (i = 16…1024). Record the affine d̂ against the declared 2
and the overhead constant a. A GPU is expected to push the constant up and flatten
small i; a disagreement is the driver's usual warning, reported as-is.

**Gates for the labelling kernel** (plan D8, decide the kernel only). G-a: GPU ≥ 5×
CPU at the top rung. G-b: GPU ≥ 1× CPU at the bottom rung with the allocation's n.

## Step 1 — memory and gates (2026-09-16)

**Device memory.** Measured with the CuPy pool after one block, every anchor × geometry.
The peak is **9.0–10.1 bytes per padded site** for i = 8…1024. `_BYTES_PER_SITE = 16`
leaves headroom, and with the 2 GiB `_GPU_WORKING_SET_BYTES` one block holds
1.2·10⁸ sites. The CPU model's origin count uses two `bincount`s; on CuPy those
peaked at **118 bytes/site** at i = 256. The GPU model counts the origin anchor with the
same seed-set gather as the south anchor instead (clusters never span samples), which
is back at 10 bytes/site.

**Gates.** Cylinder, south, P4's own allocation. Median of 3 calls, warm kernels.

| i | n | CPU s | GPU s | CPU Msites/s | GPU Msites/s | speed-up |
|---|---|---|---|---|---|---|
| 8 | 49 212 | 0.0505 | 0.0019 | 62.4 | 1 648 | **26×** |
| 16 | 24 606 | 0.0820 | 0.0022 | 76.8 | 2 826 | 37× |
| 32 | 12 303 | 0.147 | 0.0027 | 85.7 | 4 657 | 54× |
| 64 | 6 151 | 0.290 | 0.0036 | 87.0 | 6 983 | 80× |
| 128 | 3 075 | 0.551 | 0.0059 | 91.5 | 8 505 | 93× |
| 256 | 1 537 | 1.084 | 0.0104 | 93.0 | 9 730 | 105× |
| 512 | 768 | 2.150 | 0.0199 | 93.6 | 10 123 | **108×** |
| 1024 | 100 | 1.125 | 0.0109 | 93.2 | 9 626 | 103× |
| 2048 | 25 | 1.116 | 0.0118 | 94.0 | 8 877 | 94× |

**G-a PASS (108×), G-b PASS (26×)** — `cupyx.scipy.ndimage.label` is kept. No custom
kernel is written. Small i is slower per site because each call pays fixed launch and
host-sync overhead, which E will show as the affine constant.

## Results (2026-09-16)

**Verdict: PASS on A, B, C and D.** `percolation2d_gpu` is equal in distribution
to `percolation2d` at every check that was run, and the estimator can't tell them apart.
E disagrees with the declared d, as expected, and says more about the probe than the
model (below).

### A — distribution, CPU vs GPU (PASS, 8/8)

```bash
python3 -m pytest tools/tests/test_percolation2d_gpu.py -v     # 26 passed in 1.40s
CUDA_VISIBLE_DEVICES="" python3 -m pytest tools/tests/test_percolation2d_gpu.py   # 8 passed, 18 skipped
```

n = 4000 per arm, seeds `SeedSequence([20260916, tag])` spawned into CPU/GPU.

| anchor | geometry | i | mean CPU | mean GPU | z | KS D | KS p |
|---|---|---|---|---|---|---|---|
| south | box | 8 | 25.75 | 25.63 | −0.44 | 0.0200 | 0.401 |
| origin | box | 8 | 15.43 | 15.56 | +0.38 | 0.0080 | 1.000 |
| south | cylinder | 8 | 28.61 | 29.06 | +1.61 | 0.0270 | 0.108 |
| origin | cylinder | 8 | 17.38 | 17.43 | +0.14 | 0.0140 | 0.828 |
| south | box | 32 | 339.54 | 339.34 | −0.06 | 0.0118 | 0.945 |
| origin | box | 32 | 166.04 | 165.35 | −0.16 | 0.0107 | 0.975 |
| south | cylinder | 32 | 402.56 | 396.94 | −1.61 | 0.0250 | 0.164 |
| origin | cylinder | 32 | 201.11 | 204.11 | +0.61 | 0.0132 | 0.874 |

### B — closed forms (PASS)

The zero rate at i = 4 lies inside the 99.9% Wilson interval for both geometries.
p = 0 and p = 1 are exact for all four anchor × geometry cells (i = 17, the p = 1 cylinder
being the longest wrap merge). `cost_hint` is the CPU function object. The import leaves
cupy unloaded, and `simulate` without CUDA raises with the CPU model's name. Same seed
gives the same draws. The driver's rng advances by exactly one integer per call. A
merge forced past its pass cap raises.

### C — γ̂ end to end (PASS)

```bash
python3 src/estimate/compare_observables.py \
  --arm cpu=experiments/07_percolation2d_gpu/recipes/samples_cpu_cylinder.json \
  --arm gpu=experiments/07_percolation2d_gpu/recipes/samples_gpu_cylinder.json \
  --replicates 12 --m0 2 --truth 1.8958333333333333 --seed 20260916 --tag compare_cpu_gpu
```

| estimator | CPU mean ± se | GPU mean ± se | Welch z | GPU inside CPU 95% CI |
|---|---|---|---|---|
| all points | 1.89907 ± 0.00050 | 1.89980 ± 0.00064 | **+0.90** | yes, [1.89798, 1.90016] |
| m₀ = 2 | 1.89625 ± 0.00113 | 1.89817 ± 0.00108 | **+1.23** | yes, [1.89376, 1.89874] |

Per-scale cv agrees to the third digit (CPU 0.435→0.373, GPU 0.434→0.369), and both arms
reproduce P4's cylinder picture: all points ≈ 1.900, bias-dominated. Wall clock per
replicate: **4.4 s CPU, 0.1 s GPU**. The driver's "cpu wins by 1.24× in RMSE" line
compares two RMSEs without testing the difference, and at z = 0.9 and 1.2 the two
arms are indistinguishable. It is not evidence that the GPU is worse.

### D — per-scale means at 4·10⁹ (PASS)

```bash
python3 src/generate/generate.py -meta experiments/07_percolation2d_gpu/recipes/samples_cpu_deep.json --tag cpu_deep   # 44.5 s
python3 src/generate/generate.py -meta experiments/07_percolation2d_gpu/recipes/samples_gpu_deep.json --tag gpu_deep   #  1.2 s
```

Ȳ_i / i^{91/48} (a scaled amplitude, only so the columns read like P4's table; the test is on Ȳ):

| i | n | CPU | GPU | rel. se | z |
|---|---|---|---|---|---|
| 8 | 492 125 | 0.5578 | 0.5578 | 0.09% | −0.14 |
| 16 | 246 062 | 0.5611 | 0.5613 | 0.11% | +0.38 |
| 32 | 123 031 | 0.5633 | 0.5640 | 0.16% | +0.82 |
| 64 | 61 515 | 0.5646 | 0.5662 | 0.21% | +1.31 |
| 128 | 30 757 | 0.5681 | 0.5637 | 0.30% | **−2.57** |
| 256 | 15 378 | 0.5661 | 0.5683 | 0.42% | +0.92 |
| 512 | 7 689 | 0.5688 | 0.5679 | 0.59% | −0.26 |

Max |z| = 2.57 ≤ 3. Pooled over the seven scales, Σz² = 10.07 on 7 dof, **p = 0.19**.
The one large cell is the CPU run sitting high, not the GPU run sitting low: P4's
independent CPU deep run recorded 0.5671 at i = 128, which puts the GPU at z ≈ −1.4
against it. Both columns reproduce P4's recorded cylinder row (0.5578 … 0.5698) and its
+2.1% amplitude drift.

### E — cost cross-check (recorded; disagrees, as expected)

```bash
python3 src/estimate/measure_cost.py -meta experiments/07_percolation2d_gpu/recipes/cost_gpu_cylinder.json --tag cost_gpu
```

Affine d̂ = **0.80 ± 0.23**, overhead a = **1314 µs**, against the declared 2. Pure power
d̂ = 0.05. The driver reports DISAGREE (−5.2σ). CPU for reference (P1, cylinder):
d̂ = 1.978 ± 0.039, a = 104 µs.

This measures the probe, not the model. `measure_cost.py` times `simulate(i, n=1)`, and
on the GPU one sample of 10⁶ sites (i = 1024) takes about 0.1 ms of device work against
~1.3 ms of fixed per-call cost: generator construction, kernel launches, and the host
syncs in `int(lab.max())` and `.get()`. Median time goes 1.40 ms → 1.67 ms from
i = 64 to 1024, so the i² term is barely visible. The step-1 table, which times the
allocation's own n, shows the work scaling as it should (Msites/s flat to within 10% from
i = 128 up). Allocation keeps using the declared `cost_hint` (unchanged), so nothing
downstream is affected. The consequence to carry forward: **a GPU cost probe needs n
per call large enough for device work to dominate the overhead**. That is a change to
the probe's recipe or driver, and belongs in a later checkpoint.

**Correction (2026-09-16).** "Nothing downstream is affected" holds for `generate.py`,
whose allocation takes d from `cost_hint`. It does not hold for the pilot → plan
workflow. There `pilot.py` measures d from the clock and `plan.py` converts a budget to
seconds with the pilot's throughput. Both were wrong on this model; see E′.

### E′ — a cost measurement for batched models (2026-09-16)

The follow-up above, done (user, 2026-09-16). The four `*_gpu` models are registered
with `ModelSpec.batched_cost`, and both probes then time them with
`tools/cost_model.py`'s `probe_batched`. At each scale it doubles n until one call takes
4× the per-call overhead a₀, then keeps (t(4n) − t(n)) / 3n from interleaved pairs. That
is the cost of one sample, with the fixed cost of the call cancelled. n is walked, never
sized from `cost_hint`: at a fixed work per call the overhead per sample would grow
exactly like the declaration and hand it back. Nothing was changed for any other model.

No criterion was written before these runs. The comparisons are recorded, not gated.
All runs were on an idle card (299 MiB in use, no other process).

**The standalone probe, same recipe as E.**

```bash
python3 src/estimate/measure_cost.py -meta experiments/07_percolation2d_gpu/recipes/cost_gpu_cylinder.json --tag cost_gpu_batched
```

a₀ = 1181 µs. n runs from 262 144 at i = 16 down to 64 at i = 1024, and the fixed cost is
at most 17% of the first call of a pair. Affine d̂ = **2.0202 ± 0.0043** against the
declared 2, 1.0%. Pure power d̂ = 1.9971. The driver's check: **PASS** (tolerance 20%),
where E gave 0.80 ± 0.23. The probe took 10.9 s.

**Pilot → plan → one replicate.** `recipes/samples_gpu_pilot.json` (scales 8…512, neyman,
budget 10⁹, 3 replicates), then `plan.py --time 60s`. The planned replicate was then
timed through `src.generate.generate.generate(...)`, the call `run.py` makes, once cold
and once warm.

| pilot | its probe | d | throughput (sites/s) | plan's replicate | predicted | measured |
|---|---|---|---|---|---|---|
| before | `probe_window`, single boxes 4096…32768 (~10 GiB) | 2.0615 ± 0.0030 | 3.86·10⁹, pilot clock | m₀ = 3, 16…512, n = 221 054 | 20.0 s | 7.56–7.91 s |
| + warm-up, batched probe | `probe_batched`, 8…512 | 2.0246 ± 0.0008 | 8.34·10⁹, pilot clock | m₀ = 4, 32…1024, n = 119 314 | 20.0 s | 15.68–15.95 s |
| after | `probe_batched`, 8…512 | 2.0261 ± 0.0008 | 1.05·10¹⁰, from the probe | m₀ = 4, 32…1024, n = 150 002 | 20.0 s | **19.80–20.02 s** |

What each change fixed:

- **Warm-up.** Before it, the first rung of the first replicate took 0.4 s of one-time
  CUDA setup, in a pilot that drew for 0.78 s. `pilot.py` now pays that before starting
  the clock, from the cost probe's fixed seed, so the pilot's streams are unchanged.
- **Throughput from the probe.** Even warm, 20% of a 0.36 s GPU pilot is the ~1.3 ms
  fixed cost of one call per rung, which a real run spreads over millions of samples.
  For a batched model the pilot's throughput is now Σ n_i·cost_hint(i) / Σ n_i·(a + b·iᵈ)
  over its own allocation, from the probe's affine fit. That fit predicted the second
  row's replicate at 16.0 s (measured 15.68–15.95 s) before it was used.

**Still flagged: D MISMATCH.** The five batched probes agree to ±0.0008, so the 1.3% gap
between the clock's 2.026 and the declared 2 is z = +34. The pilot's gate has no relative
floor, while `compare_cost_models` has one (5%). The error budget prices se(d) at 1.0000×
RMSE and moves m₀ by 0.00. Re-planning the same pilot with d set to exactly 2 gives an
identical plan (m₀ = 4, 32…1024, n = 150 002, se(γ) = 0.0003629). The gap is real: time
per site falls from 13.8·10⁻¹¹ s at i = 8 to 9.1·10⁻¹¹ s at i = 64, then rises about 5% to
i = 512, so the clock is not exactly i². **Left as is (user, 2026-09-16):** no relative
floor was added, for GPU or CPU models, so every GPU pilot will show this warning.

## What this unlocks, not yet run

At ~10¹⁰ sites/s a 4·10¹¹-site cylinder run is about a minute. That is the wider ladder
(i up to 2048–4096) P4 names as the prerequisite for chasing the cylinder's residual
+0.004 and for a direct ω₁. It is a new question, and needs its own criteria and
sign-off.
