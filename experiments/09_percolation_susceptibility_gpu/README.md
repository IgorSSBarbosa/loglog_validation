# Experiment 09 — `percolation_susceptibility_gpu`: the off-critical observable on a GPU

**Question.** Is `MODELS["percolation_susceptibility_gpu"]` equal in distribution to
`MODELS["percolation_susceptibility"]`, and does experiment 06's Arm A/B γ̂ pipeline give
the same answer on it? The CPU model is unchanged and taken as correct.

Design: draw, stack-and-label and merge are `percolation_zd_gpu`'s (experiment 08),
exactly as the CPU model borrows `percolation_zd`'s. What is new is the reduction: the
per-sample Σ_clusters s^(k+1) is computed as Σ_sites |C(x)|^k. Setup as in
`experiments/07_percolation2d_gpu/README.md` (RTX 5090, cupy-cuda12x 14.2.0,
CUDA runtime 12.9).

## Acceptance criteria (written 2026-09-16, with the test file, before its first run)

**A. Distribution — `tools/tests/test_percolation_susceptibility_gpu.py`** (local, gitignored).
Grid: dim ∈ {2, 3} × moment ∈ {1, 2} × geometry ∈ {torus, cylinder, box} × x ∈ {1, 4},
with eps0 = 0.5 (dim 2) and 0.15 (dim 3, under p_c/2). 24 cells, n = 20 000 per arm,
independent fixed seeds. Family-wise 1% as in 08: per cell, KS p ≥ 0.01/24 = 4.17·10⁻⁴
and |mean z| ≤ 3.53.

**B. Closed forms** (same file).
- E Y against **exact enumeration** of all 2⁹ configurations of a 3×3 lattice (p = 0.45,
  moments 1 and 2, torus and box). The GPU mean over n = 100 000 must lie within 3.29 se.
- p = 1 gives Y = L^(dim·k) exactly, and p = 10⁻¹² gives 0, for dim ∈ {2, 3} × every
  geometry × k ∈ {1, 2, 3}.
- Same seed gives the same draws, and the driver's rng advances by one integer per call.
- Declarations (`cost_hint`, `box_side`, `p_at`) are the CPU objects.
- Importing leaves cupy unloaded, and without CUDA `simulate` raises naming the CPU model.

**C. γ̂ end to end.** 06's Step-2 recipes (`box_factor` 8, ν_box = 4/3, torus,
n = 1500 uniform), cut to x = 1…32 so the CPU arm fits R = 12. Only `"model"` changes.
`compare_observables.py` runs with m₀ = 2 at moment 1 (truth 43/18) and moment 2 (truth
177/36). Welch |z| ≤ 2.5 for each moment × estimator, four comparisons.

**D. Per-scale means, full ladder.** 06's `samples_moment1.json` as-is (x = 1…128,
n = 1500; one CPU run is ≈ 4.7·10¹⁰ sites). At each of 8 scales, |Ȳ_GPU − Ȳ_CPU| ≤ 3
combined se. Pooled Σz² reported.

**E. Cost probe (recorded, not gating).** `cost_gpu.json`.

## Step 1 — memory (2026-09-16)

Measured size-histogram options on a 300 × 257 × 256 block:

| method | extra device bytes/site |
|---|---|
| `cp.bincount` | 184.3 |
| `cp.histogram` | 184.3 |
| `cupyx.scatter_add` / `cp.add.at` | 0.0 |
| float64 per-site gather + per-sample sum | 8.0 |

With `cp.add.at` and the gather, the whole block peaks at **17.2–17.8 bytes/site** at
dim 2–4, torus, moment 3. `_BYTES_PER_SITE = 24`. During development the GPU reduction
matched the CPU's `cluster_moment_sum` on identical lattices to 3·10⁻¹⁶ relative, over 27
dim × geometry × moment cells. That was a one-off check, not an acceptance test.

## Step 2 — a real defect the KS grid caught, and its fix

The first run of A **failed** one cell: dim 2, moment 1, torus, x = 1 (a 4×4 torus at
p = 0.0927), KS D = 0.0225, **p = 7.5·10⁻⁵**. The other 23 cells passed.

The cause was not the lattice. A 4×4 torus has only 2¹⁶ configurations, so the exact law
of Y was enumerated (55 support points). Both samples fit it: CPU χ² p = 0.84, GPU
χ² p = 0.36. The KS statistic sat on the atom Y = 9/(16p), and the GPU returned that one
value **in two floats one ulp apart** (`0x1.84281cc6697f8p+2` and `…7f9p+2`), where the
CPU returned only `…7f9`. KS reads a split atom as two CDF jumps.

The split came from the model, not from KS: **cupy's float `**` goes through the device
`pow()`, which is inexact on integers** (`3.0**1` → `2.9999999999999996`, `19.0**3` one ulp
low), so a few per-sample sums of s^k were not exact integers. NumPy's `**` is exact there.
Fix: s^k by repeated multiplication, exact below 2⁵³. After the fix the sums are exact
integers for k = 1, 2, 3, and the failing cell at the **same fixed seeds** gives
KS p = 0.998. The fixed seeds were not changed. A regression test
(`test_moment_sums_are_exact_integers`) pins it.

Worth carrying to 10–11: any `**` on sizes in a GPU port has the same trap.

## Results (2026-09-16)

**Verdict: PASS on A–E, after the Step-2 fix.** `percolation_susceptibility_gpu` is equal
in distribution to `percolation_susceptibility`, and both moments' γ̂ agree.

### A — distribution grid (PASS, 24/24 after the fix)

```bash
python3 -m pytest tools/tests/test_percolation_susceptibility_gpu.py            # 61 passed in 78.8s
CUDA_VISIBLE_DEVICES="" python3 -m pytest tools/tests/test_percolation_susceptibility_gpu.py   # 10 passed, 51 skipped
```

| dim | cells | min KS p | max \|mean z\| |
|---|---|---|---|
| 2 | 12 | 0.002 | 2.04 |
| 3 | 12 | 0.173 | 2.48 |

The pooled mean z gives Σz² = 31.3 on 24 dof, **p = 0.15**. The lowest cell (dim 2,
moment 2, box, x = 4, KS p = 0.002) passes its limit of 4.17·10⁻⁴. It has near-continuous
values and no representation splits. P(min of 24 ≤ 0.002) ≈ 5%. As a diagnostic *after*
the pass (it replaces nothing above), three fresh seed pairs at n = 100 000 gave
KS p = 0.56, 0.46, 0.86 and mean z = +0.23, +0.05, +0.42.

### B — closed forms (PASS)

Exact enumeration, 3×3 lattice, p = 0.45, n = 100 000:

| moment | geometry | exact E Y | GPU mean | z |
|---|---|---|---|---|
| 1 | torus | 4.28371 | 4.26797 | −1.52 |
| 1 | box | 3.48916 | 3.48159 | −0.76 |
| 2 | torus | 21.45883 | 21.53273 | +0.98 |
| 2 | box | 16.07355 | 15.99100 | −1.15 |

p = 1 and p → 0 are exact in all 18 cells. Seed and stream behaviour, declaration
identity, the no-cupy import and the no-CUDA error all pass.

### C — γ̂ end to end (PASS)

```bash
python3 src/estimate/compare_observables.py --arm cpu=experiments/09_percolation_susceptibility_gpu/recipes/samples_cpu_moment1.json \
  --arm gpu=experiments/09_percolation_susceptibility_gpu/recipes/samples_gpu_moment1.json \
  --replicates 12 --m0 2 --truth 2.388888888888889 --seed 20260918 --tag compare_moment1
python3 src/estimate/compare_observables.py --arm cpu=experiments/09_percolation_susceptibility_gpu/recipes/samples_cpu_moment2.json \
  --arm gpu=experiments/09_percolation_susceptibility_gpu/recipes/samples_gpu_moment2.json \
  --replicates 12 --m0 2 --truth 4.916666666666667 --seed 20260919 --tag compare_moment2
```

| moment | estimator | CPU mean ± se | GPU mean ± se | Welch z | GPU in CPU 95% CI |
|---|---|---|---|---|---|
| 1 | all points | 2.2369 ± 0.0011 | 2.2340 ± 0.0014 | **−1.66** | no |
| 1 | m₀ = 2 | 2.2792 ± 0.0015 | 2.2794 ± 0.0020 | **+0.07** | yes |
| 2 | all points | 4.7481 ± 0.0046 | 4.7458 ± 0.0034 | **−0.41** | yes |
| 2 | m₀ = 2 | 4.7620 ± 0.0041 | 4.7601 ± 0.0044 | **−0.31** | yes |

All |z| ≤ 1.66. One containment miss (moment 1, all points) at z = −1.66. That is the case
07's README describes, where containment ignores the GPU arm's variance, which is why
it is not the gate. Per-scale cv agrees (moment 1: CPU 0.693→0.322, GPU 0.702→0.321).
Wall clock per replicate: **15.5 s CPU, 0.9 s GPU**. Both arms sit ≈ 0.11–0.16 below 43/18
and 177/36, as expected on a ladder cut to x ≤ 32. Experiment 06 records the same bias
from its full ladder (Arm B). These rows compare devices, not estimates of γ_susc.

### D — per-scale means, full ladder (PASS)

```bash
python3 src/generate/generate.py -meta experiments/09_percolation_susceptibility_gpu/recipes/samples_cpu_deep_moment1.json --tag cpu_deep_moment1   # 10 min 28.6 s
python3 src/generate/generate.py -meta experiments/09_percolation_susceptibility_gpu/recipes/samples_gpu_deep_moment1.json --tag gpu_deep_moment1   #       35.3 s
```

| x | n | Ȳ CPU | Ȳ GPU | rel. se | z | 06 recorded (CPU) |
|---|---|---|---|---|---|---|
| 1 | 1500 | 1.4710 | 1.5015 | 2.62% | +0.79 | 1.5030 |
| 2 | 1500 | 6.9923 | 7.0399 | 1.48% | +0.46 | 6.9742 |
| 4 | 1500 | 30.742 | 30.910 | 1.25% | +0.44 | 31.133 |
| 8 | 1500 | 144.91 | 143.33 | 1.27% | −0.86 | 143.86 |
| 16 | 1500 | 700.54 | 708.70 | 1.22% | +0.95 | 706.13 |
| 32 | 1500 | 3547.1 | 3550.9 | 1.14% | +0.09 | 3541.9 |
| 64 | 1500 | 18088 | 18326 | 1.18% | +1.12 | 18259 |
| 128 | 1500 | 95150 | 94009 | 1.19% | −1.01 | 95318 |

Max |z| = 1.12 ≤ 3. Pooled Σz² = 4.93 on 8 dof, **p = 0.77**. The GPU column also sits
inside experiment 06's recorded CPU run at every rung. **17.8× faster end to end.** The top
rung (L = 5161, 2.7·10⁷ sites per sample, 3 samples per block) is where the time goes.

### E — cost probe (PASS, recorded)

```bash
python3 src/estimate/measure_cost.py -meta experiments/09_percolation_susceptibility_gpu/recipes/cost_gpu.json --tag cost_gpu
```

Affine d̂ = **2.789 ± 0.066** against the declared 2.665 (+1.9σ, 4.6%): inside the driver's
20% tolerance, overhead a = 1502 µs. Unlike 07 and 08 the n = 1 probe works here, because
the ladder's boxes reach 2.7·10⁷ sites and device work outweighs the ~1.5 ms fixed cost
from x ≈ 16 up. 06's CPU probe measured 2.685 ± 0.022.

**Re-measured 2026-09-16 with `probe_batched`**, the batched-model probe of experiment 07's
E′, same recipe, idle card: affine d̂ = **2.6905 ± 0.0372** against 2.665, 1.0%, **PASS**.
n went from 2048 at x = 4 to 1 at x ≥ 64, and the probe took 3.8 s. It no longer depends on
the ladder reaching boxes big enough to hide the fixed cost.

---

## Step 3 — the exact box ladder (2026-09-21)

Plan: `plans/exact_box_ladder.md`. Model: `MODELS["percolation_susceptibility_L_gpu"]`
(`models/README.md`). This step is about the *ladder*, not the device: the same
observable, torus and stream, with the scale changed from the distance x to the box side L.

**Question.** Does the ceiling in L = ⌈c·x^ν_box⌉ explain why the d = 4 ω₁ pilot
(`pilot_gsusc_d4_low`) returned local slopes that zig-zag at 5–31σ, and does a ladder with
nothing rounded remove it?

**The evidence, from data already on disk** (`diagnostics/rounding_zigzag.py`,
`diagnostics/ladder_stability.py`, both rerun 2026-09-21 and reproducing the plan's numbers).
The ceiling gives rung x its own c_x = L/x^ν_box: +8.47%, +5.66%, +1.22%, +3.34%, +0.66%,
+0.68%, +0.20%, +0.25% above c = 4 at x = 2…256. Where S(p, L) still grows with L, rung x
is shifted by G·log(c_x/c), G = ∂log S/∂log L at fixed p. Pooling the three runs that
share the design exactly (8 rungs, inverse variance), eq. (232) with one correction gives
χ² = 47.0 / 4 dof; adding G·log(c_x/c) gives χ² = 3.7 / 3 with G = 0.1622 ± 0.0228 (7σ from
0, physical sign). The shift against each rung's noise is 70σ at x = 2, 32σ at 4, 8σ at 16,
0.6σ at 64, < 0.1σ at 256: an ω₁ pilot lives on the small, precise rungs, which is why it
failed while the production run at x = 16…256 barely noticed.

**Decisions (Igor, 2026-09-21).**

| | |
|---|---|
| model | new, `percolation_susceptibility_L_gpu`; the recipe's scale is the integer L; GPU only |
| box factor | **c = 8**, twice the x-ladder's 4 (plan §3's noiseless table: on the exact ladder the worst local-slope error falls 3.6–10× from c to 2c, 8.9→0.9, 19.1→5.3, 50.8→6.2 in units of 10⁻⁴) |
| ν_box | 0.69, unchanged (above ν = 0.6845(23)'s 2σ upper end 0.6891, as the model's upper-bound rule requires) |
| grid | 2^k first. If the ω₁ pilot fails to converge, the √2 grid; the rounded grid feeds only fits on the actual log L, never `gamma_closed_form` (which was deliberately left as is) |
| units | the tools report γ_L, ω_L; γ_susc = ν_box·γ_L and ω_x = ν_box·ω_L are written up here, not in a driver (no `ModelSpec` field) |
| smallest rung | at c = 8, ε₀ = 0.09 in d = 4 the model refuses L = 2, 3, 4 (p ≤ 0; smallest legal L = 5), so the smallest power of 2 is L = 8 (x = 1) and the 2^k ladder is 8, 16, 32, 64, 128 (x = 1…55.6, top set by int32 labels and memory) |

**Acceptance criteria (written before the calibration and the pilot ran).**

*T2, `calibrate_box.py` → `analyse_box.py`* (d = 4, GPU, seed 2026092201; x ∈ {16, 32, 64},
c ≈ 3, 4, 5, 6, 8, plus L = 43, 45 beside 44 at x = 32; n fixed before each draw from the
noise already measured, target se/S ≈ 0.5%):
1. **Gate.** G at c = 4, pooled over x, agrees with 0.162 ± 0.023 within 2σ combined. If
   not, the zig-zag has another cause and the plan is wrong: STOP.
2. G_x(4) agrees across the three x within 2σ of the pooled value (finite-size scaling).
3. cv²·L^dim flat across c within ±25% at each x (d = 2 measured 5.7, 6.7, 7.3·10⁴). It is
   the cost of a fixed relative precision, so a failure is a finding, not a stop.
4. Parity: L = 43, 45 as smooth as their neighbour, by the second difference and by
   residuals from the sweep's quadratic fit, both within 2σ. If not, a √2 grid uses even L.

*T4.1, the ω₁ pilot* (`recipes/samples_pilot_gsusc_L_d4.json`, L = 8…128, neyman
4.35·10¹¹ sites per replicate, 8 replicates, seed 2026092202):
- local slopes change sign between successive windows nowhere by more than 2σ (the old
  pilot did at 5–31σ);
- the one-correction fit converges with χ²/dof ≲ 2 and se(ω₁) < ω₁/2;
- ω_x = ν_box·ω_L is reported beside Δ₁ ≈ 0.77 (reporting only, ground rule 4).

*T4.2, γ production* (`autopilot.py`, 2^k grid, seed 2026092203): both gates pass without
`--force`, and the report gives γ_susc = ν_box·γ_L with its eq. (720) interval, compared
with 1.430(6) only at reporting time.

*Not a test of the ladder:* the plan's "budget 1.2·10⁹ sites" is the x-ladder recipe's
budget in x^d units (d = 2.68), which is 4.35·10¹¹ actual sites per replicate. The L recipe
states its budget in L⁴ units, so 4.35·10¹¹ reproduces the old pilot's compute.
