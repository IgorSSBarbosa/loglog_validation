# Critical exponents measured so far, and what better precision would cost

Compiled 2026-09-16 from every experiment in `experiments/00`–`10`; errors switched to the eq. (720) bound on 2026-09-17.

**Where the numbers come from.**
- Runs 01–06 are transcribed from each experiment's README. Their `data/` directories
  are not on this host, so nothing there was recomputed.
- Runs 07–10 were recomputed from the `data/` on this host.
- The pilots in §4 were drawn today on **xingu** (idle RTX 5090, i9‑13900K) through the
  unmodified `src/study/pilot.py`. The planning numbers come from `src/study/plan.py`'s
  own `budget_for_target`.
- The eq. (720) bounds (2026‑09‑17) needed cv, Λ and (ω₁, a₁) for runs whose data is not
  on this host. Those came from re‑draws of the identical committed recipes on xingu,
  through `generate()`: 13 recipes, 1–12 replicates each, the GPU port where one exists.
  The re‑draws supply constants only; no estimate in Table 1 comes from them.
- Literature values are from
  [Wikipedia, *Percolation critical exponents*](https://en.wikipedia.org/wiki/Percolation_critical_exponents)
  (read 2026‑09‑16, "Exponents for standard percolation"). "Best" means the smallest
  uncertainty quoted there.

## Headline

- **Every error is now the article's eq. (720) Wilson bound** (Theorem `thm:wilson`,
  `tools/wilson.py`). It adds the finite‑size bias B_fs, the Jensen bias B_good and the
  stray‑mean term B_bad to the statistical term. The previous version printed a standard
  error with no bias term, and 49 of its 69 estimates excluded the reference value at 95%.
- **Coverage:** 56 of the 78 estimates in Table 1 get a bound, and **54 of those 56
  contain the reference value**. The two that don't are the d = 3 *axis* pilots: their
  one‑correction fit misses a slow drift, as Table 4a had already shown. Of the other 22, 18 have no
  bound because ω₁ was not identified or the fit did not converge. 4 have no recipe or
  fittable ladder: three H7 prototypes and d = 5 `face`.
- **Pinned values** are the narrowest bound for each exponent and dimension, chosen
  without looking at the reference:

| exponent | d | pinned estimate ± eq. (720) | from | reference | inside? |
|---|---|---|---|---|---|
| γ of the walk | 1 | 0.5002 ± 0.0084 | 02 A2, p = 1/2, window 64..1024 | 1/2 | yes |
| d_f | 2 | 1.8930 ± 0.0222 | 03 P2 deep, box, 64..512 | 91/48 | yes |
| d_f | 3 | 2.5591 ± 0.0201 | pilot, axis, 32..256 | 2.52293(10) | **no** |
| d_f | 4 | 3.0770 ± 0.0866 | pilot, face_far, 16..128 | 3.0446(7) | yes |
| d_f | 5 | 3.6555 ± 0.2121 | pilot, face_far, 16..64 | 3.5260(14) | yes |
| γ/ν | 2 | 1.7716 ± 0.1028 | 03 P3 deep origin, 16..512 | 43/24 | yes |
| τ | 2 | 2.0414 ± 0.0214 | 10 C GPU, tail, 64..2048 | 187/91 | yes |
| τ | 3 | 2.1582 ± 0.1336 | 05 H5(c), 16..512 | 2.18909(5) | yes |
| γ | 2 | 2.3585 ± 0.2104 | 06 Arm A, 32..1024 | 43/18 | yes |
| γ | 3 | 1.7768 ± 0.0402 | pilot, 8..256 | 1.793(3) | yes |
| γ | 5 | 1.2188 ± 0.7959 | pilot, 8..64 (a₁ bounded conservatively) | 1.1792(7) | yes |
| e₂ | 2 | 4.8611 ± 2.4188 | 06 Arm B, 16..128 (a₁ bounded conservatively) | 177/36 | yes |

  τ(4), τ(5) and γ(4) have no bound, because their pilots could not identify ω₁.
- **Time to match the literature** (Table 4b, one RTX 5090, as planned by the article's
  own allocation): γ(3) to ±0.003 in ≈ 3 days; d_f(3) to ±10⁻⁴ in 20 days to 5 years
  depending on the observable; d_f(4) in ≈ 3 000 years; d_f(5) in ≈ 2·10⁷ years.
  Most of these are lower bounds (Table 4a).

## How to read the errors

**Table 1 contains only the article's log‑log estimator**, eq. (523)–(526): the OLS slope
of log Ȳ against log i over a consecutive ρ = 2 window, including two‑rung local slopes.
Values from the nonlinear eq. (232) fit, the bias‑decay fit or the two‑correction
diagnostics are constants; they appear only in Tables 2–4.

**Every ± in Tables 1 and 3 is the eq. (720) half‑width at 95%:**

  |γ̂ − γ| ≤ B_fs + B_good + B_bad + 1.96·σ_se.

It is a *bound* on the error, bias included, not a standard deviation or standard error.
For each run, every window with ≥ 3 rungs is evaluated (or every window the README
recorded), and the run's row shows the window with the **narrowest bound**. Within an
exponent, the narrowest row overall is bold (the pinned value).

**Where the inputs come from:**
- n per scale: the run's own data, or its committed recipe (`resolve_n`).
- cv and Λ = max E|log ξ|³ per scale: the run's own data. Otherwise, data from the
  identical recipe: a local run, or a **re‑draw** on xingu through `generate()` (these
  re‑draws only supply constants; every estimate is the recorded one). For rwre at
  p = 1/2 they are the exact srw moments.
- (ω₁, a₁): this report's pilot for the observable if one exists; otherwise a re‑draw of
  the identical recipe with 3–12 replicates; otherwise the run's own recorded fit. Every
  row names its source.

**Three extensions of the article's formula**, needed because most runs are not the
article's setting:
1. **Non‑uniform n** (Neyman allocations). The variance term uses the exact per‑scale
   form `sigma_se_per_scale`. The n in B_good and B_bad is the harmonic mean of n over the
   window. This is valid because the article's constants are exactly
   max|w_k| = 6/(m(m+1)) (checked for m = 2..11) times a sum of per‑scale terms in 1/n_k.
2. **Mean of R replicates.** Only the statistical term shrinks by √R; the bias terms
   stay.
3. **a₁ not identified but ω₁ is** (se < ω₁/2). B_fs is evaluated with |a₁| + 2·se(a₁).
   B_fs is linear in |a₁|, so this is a conservative bound. Such rows are marked. When
   ω₁ itself is not identified, the bound is **n/a**, matching `plan.py`'s "do not plan"
   verdict.

**Always omitted:** the ω₂ piece of B_fs, which needs φ⁺ and was never measured, so
every bound is formally incomplete in that term (as `tools/wilson.py` flags). B_bad is
included wherever Λ is available, which is every row with a bound except 06 Arm A (as
recorded by `report.py`).

## 1. Measured exponents

"Largest scale" is the top rung of the ladder: walk length k, box side i, cluster size s
or distance ratio x = ε₀/ε. "Time" is the wall clock of the run(s) the estimate comes
from. **n/r** = not recorded; "≈" = estimated from n × cost ÷ measured throughput.
"Dominant term" is the largest of the four terms in the bound.

### 1a. Random walk in a dynamic random environment — γ of E|X_k| ~ k^γ

| d | reference | experiment / run | observable | window | γ | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/2 (exact) | 02 A2 control, p = 1/2, 3 reps, n = 20 130 | \|X_k\| (≡ srw) | 64..1024 | **0.5002** | **± 0.0084** | yes | stray mean (B_bad) | 02 A2's own fit (R = 3) | ≈ 20 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.00, 3 reps | \|X_k\| | deepest recorded | 0.5576 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.10, 3 reps | \|X_k\| | deepest recorded | 0.5612 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.20, 3 reps | \|X_k\| | deepest recorded | 0.5734 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.30, 3 reps | \|X_k\| | deepest recorded | 0.5779 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 1/3, 3 reps | \|X_k\| | deepest recorded | 0.5715 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.40, 3 reps | \|X_k\| | deepest recorded | 0.5486 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A6 sieve, p = 0.45, 3 reps | \|X_k\| | deepest recorded | 0.5217 | **n/a** | — | — | ω₁ not identified (A5: 0.019 ± 0.001, degenerate with log a₀) | ≈ 5 min | k = 1024 |
| 1 | open | 02 A4, p = 1/3, window_c = 24 | \|X_k\| | deepest recorded | 0.5726 | **n/a** | — | — | ω₁ not identified | ≈ 10 min | k = 1024 |
| 1 | open | 02 A5 forced high ladder, p = 1/3, n = 50 | \|X_k\| | deepest recorded | 0.5706 | **n/a** | — | — | ω₁ not identified; forced past the pilot's refusal | n/r | k = 32 768 |

### 1b. Fractal dimension d_f (count of a seed‑connected cluster ~ i^{d_f})

| d | reference | experiment / run | observable | window | d_f | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 91/48 = 1.895833 | 03 P2 deep, 4·10⁹ sites | south face, box | 64..512 | **1.8930** | **± 0.0222** | yes | finite size | re-draw of `samples_df_south.json` (R=3) | 2 m 18 s | i = 512 |
| 2 | 91/48 = 1.895833 | 03 P3, 12 reps × 4·10⁸ | south face, box | 32..512 | 1.8895 | ± 0.0272 | yes | finite size | re-draw of `samples_df_south.json` (R=3) | 2.7 min | i = 512 |
| 2 | 91/48 = 1.895833 | 03 P4 box arm, 12 reps × 4·10⁸ | south face, box | 32..512 | 1.8898 | ± 0.0272 | yes | finite size | re-draw of `samples_df_south.json` (R=3) | 2.7 min | i = 512 |
| 2 | 91/48 = 1.895833 | 03 P4 cylinder arm, 12 reps × 4·10⁸ | south face, cylinder | 32..512 | 1.8975 | ± 0.1119 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | ≈ 2.8 min | i = 512 |
| 2 | 91/48 = 1.895833 | 03 P4 deep, 4·10⁹ | south face, cylinder | 32..512 | 1.9002 | ± 0.1081 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | ≈ 45 s* | i = 512 |
| 2 | 91/48 = 1.895833 | 07 C, CPU, 12 reps × 4·10⁸ | south face, cylinder | 32..512 | 1.8962 | ± 0.1119 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 53 s | i = 512 |
| 2 | 91/48 = 1.895833 | 07 C, GPU, 12 reps × 4·10⁸ | south face, cylinder | 32..512 | 1.8982 | ± 0.1118 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 1.2 s | i = 512 |
| 2 | 91/48 = 1.895833 | 07 D, CPU deep 4·10⁹ | south face, cylinder | 32..512 | 1.8990 | ± 0.1081 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 44.1 s | i = 512 |
| 2 | 91/48 = 1.895833 | 07 D, GPU deep 4·10⁹ | south face, cylinder | 32..512 | 1.8983 | ± 0.1080 | yes | finite size | re-draw of `samples_gpu_deep.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 0.7 s | i = 512 |
| 3 | 2.52293(10) | 05 H2, 4·10⁹ | face_far, cylinder | 8..256 | 2.5541 | ± 0.1774 | yes | stray mean (B_bad) | pilot `pilot_df_d3_facefar_wide` | ≈ 1 min* | i = 256 |
| 3 | 2.52293(10) | 05 H2, 4·10⁹ | face, cylinder | 8..256 | 2.6044 | ± 1.2103 | yes | finite size | re-draw of `samples_gpu_deep_d3_face.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | ≈ 1 min* | i = 256 |
| 3 | 2.52293(10) | 08 D, CPU deep 4·10⁹ | face, cylinder | 8..256 | 2.6000 | ± 1.2103 | yes | finite size | re-draw of `samples_gpu_deep_d3_face.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 56.0 s | i = 256 |
| 3 | 2.52293(10) | 08 D, GPU deep 4·10⁹ | face, cylinder | 8..256 | 2.5957 | ± 1.2107 | yes | finite size | re-draw of `samples_gpu_deep_d3_face.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 0.9 s | i = 256 |
| 3 | 2.52293(10) | 05 H7 prototype, 3·10⁸ | slab k = 1 (axis), cylinder | deepest recorded | 2.5750 | **n/a** | — | — | prototype: recipe and constants not recorded (local slope 8→16) | n/r | i = 64 |
| 3 | 2.52293(10) | 08 C, CPU, 12 reps × 4·10⁸ | face_far, cylinder | 8..128 | 2.5346 | ± 0.2087 | yes | stray mean (B_bad) | pilot `pilot_df_d3_facefar_wide` | 71 s | i = 128 |
| 3 | 2.52293(10) | 08 C, GPU, 12 reps × 4·10⁸ | face_far, cylinder | 8..128 | 2.5267 | ± 0.2074 | yes | stray mean (B_bad) | pilot `pilot_df_d3_facefar_wide` | 0.6 s | i = 128 |
| 3 | 2.52293(10) | new pilot `pilot_df_d3_axis`, GPU, 5 reps | axis (slab k = 1), cylinder | 32..256 | **2.5591** | **± 0.0201** | **no** (+1.8 half‑widths) | stray mean (B_bad) | pilot `pilot_df_d3_axis`; a₁ unidentified → \|a₁\| + 2 se used (conservative) | 65 s | i = 256 |
| 3 | 2.52293(10) | new pilot `pilot_df_d3_axis_wide`, GPU, 5 reps | axis (slab k = 1), cylinder | 8..512 | 2.5579 | ± 0.0207 | **no** (+1.7 half‑widths) | stray mean (B_bad) | pilot `pilot_df_d3_axis_wide` | 5 m 22 s | i = 512 |
| 3 | 2.52293(10) | new pilot `pilot_df_d3_facefar_wide`, GPU, 5 reps | face_far, cylinder | 8..512 | 2.5270 | ± 0.0398 | yes | stray mean (B_bad) | pilot `pilot_df_d3_facefar_wide` | 3 m 30 s | i = 512 |
| 4 | 3.0446(7) | 05 H2, 4·10⁹ | face_far, cylinder | 8..64 | 3.0949 | ± 0.2454 | yes | stray mean (B_bad) | pilot `pilot_df_d4_facefar` | n/r | i = 64 |
| 4 | 3.0446(7) | 05 H2, 4·10⁹ | face, cylinder | 16..64 | 3.3200 | ± 0.3130 | yes | finite size | re-draw of `samples_df_d4_face.json` (R=3) | n/r | i = 64 |
| 4 | 3.0446(7) | new pilot `pilot_df_d4_facefar`, GPU, 5 reps | face_far, cylinder | 16..128 | **3.0770** | **± 0.0866** | yes | stray mean (B_bad) | pilot `pilot_df_d4_facefar` | 4 m 00 s | i = 128 |
| 5 | 3.5260(14) | 05 H2, 4·10⁹ | face_far, cylinder | 8..32 | 3.6520 | ± 0.5246 | yes | finite size | pilot `pilot_df_d5_facefar` | ≈ 1.4 min* | i = 32 |
| 5 | d − 1 = 4 | 05 H2, 4·10⁹ | face, cylinder (measures d − 1) | deepest recorded | 4.1604 | **n/a** | — | — | 4 rungs: eq. (232) cannot be fitted and there is no pilot for this observable | n/r | i = 32 |
| 5 | 3.5260(14) | 08 D, CPU deep 4·10⁹ | face_far, cylinder | 8..32 | 3.7211 | ± 0.5246 | yes | finite size | pilot `pilot_df_d5_facefar` | 81.4 s | i = 32 |
| 5 | 3.5260(14) | 08 D, GPU deep 4·10⁹ | face_far, cylinder | 8..32 | 3.7547 | ± 0.5240 | yes | finite size | pilot `pilot_df_d5_facefar` | 1.1 s | i = 32 |
| 5 | 3.5260(14) | 08 C, CPU, 12 reps × 4·10⁸ | face_far, cylinder | 8..32 | 3.7494 | ± 1.0830 | yes | stray mean (B_bad) | pilot `pilot_df_d5_facefar` | 90 s | i = 32 |
| 5 | 3.5260(14) | 08 C, GPU, 12 reps × 4·10⁸ | face_far, cylinder | 8..32 | 3.7321 | ± 1.0724 | yes | stray mean (B_bad) | pilot `pilot_df_d5_facefar` | 1.2 s | i = 32 |
| 5 | 3.5260(14) | 05 H7 prototype, 6·10⁸ | slab k = 2, cylinder | deepest recorded | 3.7740 | **n/a** | — | — | prototype: recipe and constants not recorded (local slope 4→8) | n/r | i = 32 |
| 5 | 3.5260(14) | new pilot `pilot_df_d5_facefar`, GPU, 5 reps | face_far, cylinder | 16..64 | **3.6555** | **± 0.2121** | yes | finite size | pilot `pilot_df_d5_facefar` | 4 m 41 s | i = 64 |

### 1c. γ/ν = 2 − η (origin anchor: box‑restricted susceptibility ~ i^{γ/ν})

| d | reference | experiment / run | observable | window | γ/ν | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 43/24 = 1.791667 | 03 P3 origin arm, 12 reps × 4·10⁸ | origin, box | 8..512 | 1.7599 | ± 0.1937 | yes | stray mean (B_bad) | re-draw of `samples_df_origin.json` (R=3) | 2.7 min | i = 512 |
| 2 | 43/24 = 1.791667 | 03 P3 deep origin, 4·10⁹ | origin, box | 16..512 | **1.7716** | **± 0.1028** | yes | finite size | re-draw of `samples_df_origin.json` (R=3) | ≈ 2.3 min | i = 512 |
| 3 | 2.0431(14) | 05 H7 prototype (slab k = 0), 3·10⁸ | origin, cylinder | deepest recorded | 1.9950 | **n/a** | — | — | prototype: recipe and constants not recorded (local slope 8→16) | n/r | i = 64 |

### 1d. Fisher exponent τ (cluster‑number density on a cluster‑size ladder; γ̂ = 1 − τ̂)

| d | reference | experiment / run | observable | window | τ | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 187/91 = 2.054945 | 10 C, CPU, 12 reps | bin, torus | 64..2048 | 2.0454 | ± 0.0799 | yes | finite size | re-draw of `samples_gpu_bin.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 71 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 C, GPU, 12 reps | bin, torus | 64..2048 | 2.0443 | ± 0.0798 | yes | finite size | re-draw of `samples_gpu_bin.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 3.6 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 C, CPU, 12 reps | tail, torus | 64..2048 | 2.0425 | ± 0.0214 | yes | finite size | re-draw of `samples_tau_tail.json` (R=3) | 10.7 min | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 C, GPU, 12 reps | tail, torus | 64..2048 | **2.0414** | **± 0.0214** | yes | finite size | re-draw of `samples_tau_tail.json` (R=3) | 32 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 C, CPU, 12 reps | shared lattice, torus | 64..2048 | 2.0437 | ± 0.0527 | yes | finite size | the run's own fit (R = 1, correlated rungs) | 3.4 min | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 C, GPU, 12 reps | shared lattice, torus | 64..2048 | 2.0449 | ± 0.0436 | yes | finite size | the run's own fit (R = 1, correlated rungs) | 10 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 D, CPU deep | bin, torus | 64..2048 | 2.0315 | ± 0.0942 | yes | finite size | re-draw of `samples_gpu_bin.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 6.2 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 D, GPU deep | bin, torus | 64..2048 | 2.0490 | ± 0.0949 | yes | finite size | re-draw of `samples_gpu_bin.json` (R=12); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 0.7 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 D, CPU deep | shared lattice, torus | 64..2048 | 2.0500 | ± 0.0590 | yes | finite size | the run's own fit (R = 1, correlated rungs) | 18.7 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 10 D, GPU deep | shared lattice, torus | 64..2048 | 2.0493 | ± 0.0498 | yes | finite size | the run's own fit (R = 1, correlated rungs) | 1.3 s | s = 2048 |
| 2 | 187/91 = 2.054945 | 05 H8, 8·10⁹ sites | bin, torus (percolation_tau_zd) | 64..512 | 2.0407 | ± 0.0774 | yes | finite size | 05 H8's own fit (R = 1, no se) | 5.2 min | s = 512 |
| 3 | 2.18909(5) | 05 H5, 4.33·10⁹ sites | bin, torus | 8..1024 | 2.1543 | ± 0.2168 | yes | finite size | pilot `pilot_tau_d3` | ≈ 4 min | s = 1024 |
| 3 | 2.18909(5) | 05 H5(c), box_exponent = 1/2 | bin, torus | 16..512 | **2.1582** | **± 0.1336** | yes | finite size | re-draw of `samples_tau_d3_boxexp_half.json` (R=1) | n/r | s = 512 |
| 3 | 2.18909(5) | 05 H8, 8·10⁹ sites | bin, torus | 64..512 | 2.1680 | ± 0.3802 | yes | finite size | pilot `pilot_tau_d3` | 7.5 min | s = 512 |
| 3 | 2.18909(5) | new pilot `pilot_tau_d3`, CPU, 5 reps | bin, torus | 8..512 | 2.1513 | ± 0.2764 | yes | finite size | pilot `pilot_tau_d3` | 3 m 14 s | s = 512 |
| 4 | 2.3142(5) | 05 H8, 8·10⁹ sites | bin, torus | deepest recorded | 2.2828 | **n/a** | — | — | (ω₁, a₁) unidentified by pilot `pilot_tau_d4` | 9.0 min | s = 512 |
| 4 | 2.3142(5) | new pilot `pilot_tau_d4`, CPU, 5 reps | bin, torus | deepest recorded | 2.2959 | **n/a** | — | — | (ω₁, a₁) unidentified by pilot `pilot_tau_d4` | 3 m 51 s | s = 512 |
| 5 | 2.4175(2) | 05 H8, 8·10⁹ sites | bin, torus | deepest recorded | 2.3610 | **n/a** | — | — | (ω₁, a₁) unidentified by pilot `pilot_tau_d5` | 12.3 min | s = 512 |
| 5 | 2.4175(2) | new pilot `pilot_tau_d5`, CPU, 5 reps | bin, torus | deepest recorded | 2.3664 | **n/a** | — | — | (ω₁, a₁) unidentified by pilot `pilot_tau_d5` | 4 m 48 s | s = 512 |
| 6 | 5/2 | 05 H5, 1.61·10¹⁰ sites | bin, torus | deepest recorded | 2.4483 | **n/a** | — | — | eq. (232) fit not converged (ω₁ = 0.032) | 43 min | s = 1024 |
| 6 | 5/2 | 05 H8, 8·10⁹ sites | bin, torus | deepest recorded | 2.4250 | **n/a** | — | — | eq. (232) fit not converged | 23.2 min | s = 512 |
| 7 | 5/2 | 05 H8, 8·10⁹ sites | bin, torus | deepest recorded | 2.4595 | **n/a** | — | — | eq. (232) fit not converged | 55.4 min | s = 512 |
| 8 | 5/2 | 05 H8, 8·10⁹ sites | bin, torus | deepest recorded | 2.4840 | **n/a** | — | — | eq. (232) fit not converged | 155.5 min | s = 512 |

### 1e. Susceptibility exponent γ (ladder in ε = ε₀/x, E|C| ~ x^γ)

| d | reference | experiment / run | observable | window | γ | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 43/18 = 2.388889 | 06 Arm A (article method), 5 reps, n = 57 | mean cluster size, bf 4 | 32..1024 | **2.3585** | **± 0.2104** | yes | finite size | eq. (720) bound as recorded by report.py | 3.0 h + pilot | x = 1024 |
| 2 | 43/18 = 2.388889 | 06 Arm B, n = 1500 | mean cluster size, bf 8 | 8..128 | 2.3436 | ± 0.2638 | yes | finite size | 06 Arm A pilot (x = 4..128, R = 5) | ≈ 10.5 min | x = 128 |
| 2 | 43/18 = 2.388889 | 09 D, CPU deep, n = 1500 | mean cluster size, bf 8 | 8..128 | 2.3408 | ± 0.2638 | yes | finite size | 06 Arm A pilot (x = 4..128, R = 5) | 10 m 28 s | x = 128 |
| 2 | 43/18 = 2.388889 | 09 D, GPU deep, n = 1500 | mean cluster size, bf 8 | 8..128 | 2.3407 | ± 0.2636 | yes | finite size | 06 Arm A pilot (x = 4..128, R = 5) | 35 s | x = 128 |
| 2 | 43/18 = 2.388889 | 09 C, CPU, 12 reps, x ≤ 32 | mean cluster size, bf 8 | 4..32 | 2.2792 | ± 0.5793 | yes | finite size | 06 Arm A pilot (x = 4..128, R = 5) | 3.1 min | x = 32 |
| 2 | 43/18 = 2.388889 | 09 C, GPU, 12 reps, x ≤ 32 | mean cluster size, bf 8 | 4..32 | 2.2794 | ± 0.5793 | yes | finite size | 06 Arm A pilot (x = 4..128, R = 5) | 11 s | x = 32 |
| 3 | 1.793(3) [also 1.819(3)] | new pilot `pilot_gsusc_d3`, GPU, 5 reps | mean cluster size | 8..256 | **1.7768** | **± 0.0402** | yes | finite size | pilot `pilot_gsusc_d3` | 7 m 34 s | x = 256 |
| 4 | 1.430(6) | new pilot `pilot_gsusc_d4`, GPU, 5 reps | mean cluster size | deepest recorded | 1.4427 | **n/a** | — | — | (ω₁, a₁) unidentified by pilot `pilot_gsusc_d4` | 11 m 46 s | x = 128 |
| 5 | 1.1792(7) | new pilot `pilot_gsusc_d5`, GPU, 5 reps | mean cluster size | 8..64 | **1.2188** | **± 0.7959** | yes | finite size | pilot `pilot_gsusc_d5`; a₁ unidentified → \|a₁\| + 2 se used (conservative) | 15 m 51 s | x = 64 |

### 1f. e₂ = (4 − τ)/σ (second moment on the same ladder)

| d | reference | experiment / run | observable | window | e₂ | eq. (720) ± | reference inside? | dominant term | (ω₁, a₁) from | time | largest scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 177/36 = 4.916667 | 06 Arm B, n = 1500 | second moment, bf 8 | 16..128 | **4.8611** | **± 2.4188** | yes | finite size | re-draw of `samples_gpu_deep_moment2.json`, fitted on x ≥ 4 (R = 3); a₁ unidentified → \|a₁\| + 2 se used (conservative) | ≈ 10.5 min | x = 128 |
| 2 | 177/36 = 4.916667 | 09 C, CPU, 12 reps, x ≤ 32 | second moment, bf 8 | 4..32 | 4.7620 | ± 5.7430 | yes | finite size | re-draw of `samples_gpu_deep_moment2.json`, fitted on x ≥ 4 (R = 3); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 3.1 min | x = 32 |
| 2 | 177/36 = 4.916667 | 09 C, GPU, 12 reps, x ≤ 32 | second moment, bf 8 | 4..32 | 4.7601 | ± 5.7412 | yes | finite size | re-draw of `samples_gpu_deep_moment2.json`, fitted on x ≥ 4 (R = 3); a₁ unidentified → \|a₁\| + 2 se used (conservative) | 11 s | x = 32 |

\* The original run did not record a wall clock. The time given is the same work
(4·10⁹ sites, same model and geometry) as re‑timed on the i9‑13900K in 07 D / 08 D.

The rwre times are estimated from n × Σ k·W(k) ÷ 3.0·10⁷ units/s. The d = 2 τ runs whose
configuration cannot be pinned down (the models/README pilot pair and the
"shared lattice, mean of 4" runs) are dropped here. Experiment 10 C re‑ran their
identical recipes with 12 replicates; see "Not included".

---

## 2. Constants: cost exponent and correction to scaling

Fits are allowed here. Each row is **one** measurement or **one** dataset. Where two
estimators or two datasets exist for the same quantity, they are separate rows. "Reference
correction exponent" is the literature value in the ladder's own variable:
- ω (length) for an i‑ladder;
- Ω = ω/d_f for an s‑ladder;
- Δ₁ = ων = Ω/σ for an ε‑ladder.

### 2a. Cost exponent d (Assumption 7): one measurement per row

| model / observable | d declared (cost_hint on the ladder) | measurement | d measured | run |
|---|---|---|---|---|
| srw | 1 | affine clock fit, 9 independent probes | 1.0069 ± 0.0023 | 01, `cost_probe_reps` |
| srw | 1 | affine clock fit (models/README) | 1.0028 ± 0.0020 | 01 |
| rwre | 1.4928 | amortized, n = 64 per call, k = 512..4096 | 1.4926 | 02 A1 |
| rwre | 1.4928 | affine clock fit, n = 1 per call | 1.1950 ± 0.0301 | 02 A1 |
| percolation2d, box | 2 | affine clock fit, CPU | 2.0285 ± 0.0175 | 03 P1 |
| percolation2d, cylinder | 2 | affine clock fit, CPU | 1.9780 ± 0.0388 | 03 P1 |
| percolation2d_gpu, cylinder | 2 | affine, n = 1 per call (measures call overhead) | 0.80 ± 0.23 | 07 E |
| percolation2d_gpu, cylinder | 2 | batched probe | 2.0202 ± 0.0043 | 07 E′ |
| percolation2d_gpu, cylinder | 2 | batched probe inside the pilot | 2.0261 ± 0.0008 | 07 E′ |
| percolation_tau, d = 2 | 0.9993 | affine clock fit, CPU | 1.049 ± 0.019 | 03 |
| percolation_tau_gpu, d = 2 | 0.9993 | affine, n = 1 per call | 1.287 ± 0.172 | 10 E |
| percolation_tau_gpu, d = 2 | 0.9993 | batched probe | 0.9985 ± 0.0023 | 10 E′ |
| percolation_zd, d = 3 | 3 | affine clock fit, CPU | 2.934 ± 0.016 | 05 H1 |
| percolation_zd_gpu, d = 3 | 3 | affine, n = 1 per call | 2.34 ± 0.20 | 08 E |
| percolation_zd_gpu, d = 3 | 3 | batched probe | 2.9943 ± 0.0153 | 08 E′ |
| percolation_zd, d = 4 | 4 | affine clock fit, CPU | 3.809 ± 0.030 | 05 H1 |
| percolation_zd, d = 5 | 5 | affine clock fit, CPU | 4.620 ± 0.042 | 05 H1 |
| percolation_zd, d = 5 | 5 | affine, window 8..16 | 4.849 ± 0.051 | TODO, 2026‑09‑14 |
| percolation_zd, d = 5 | 5 | affine, window 8..32 | 4.799 ± 0.005 | TODO, 2026‑09‑14 |
| percolation_tau_zd, d = 3 | 1.2060 | affine clock fit, CPU | 1.1748 ± 0.0100 | 05 H5 |
| percolation_susceptibility, d = 2, bf 8 | 2.6650 | affine clock fit, CPU | 2.6850 ± 0.0223 | 06 Arm B |
| percolation_susceptibility, d = 2, bf 4 | 2.6625 | affine clock fit, 5 probes, CPU | 2.7835 ± 0.0280 | 06 Arm A pilot |
| percolation_susceptibility_gpu, d = 2, bf 8 | 2.665 | affine, n = 1 per call | 2.789 ± 0.066 | 09 E |
| percolation_susceptibility_gpu, d = 2, bf 8 | 2.665 | batched probe | 2.6905 ± 0.0372 | 09 E′ |
| percolation_zd_gpu axis, d = 3 | 3 | batched probe inside the pilot (5 probes) | 3.0263 ± 0.0032 | new `pilot_df_d3_axis` |
| percolation_zd_gpu axis, d = 3 | 3 | batched probe inside the pilot (5 probes) | 3.2343 ± 0.0023 | new `pilot_df_d3_axis_wide` |
| percolation_zd_gpu face_far, d = 3 | 3 | batched probe inside the pilot (5 probes) | 3.1315 ± 0.0063 | new `pilot_df_d3_facefar_wide` |
| percolation_zd_gpu face_far, d = 4 | 4 | batched probe inside the pilot (5 probes) | 4.2141 ± 0.0060 | new `pilot_df_d4_facefar` |
| percolation_zd_gpu face_far, d = 5 | 5 | batched probe inside the pilot (5 probes) | 5.3260 ± 0.0050 | new `pilot_df_d5_facefar` |
| percolation_susceptibility_gpu, d = 3 | 2.626 | batched probe inside the pilot (5 probes) | 2.7174 ± 0.0001 | new `pilot_gsusc_d3` |
| percolation_susceptibility_gpu, d = 4 | 2.709 | batched probe inside the pilot (5 probes) | 2.8221 ± 0.0006 | new `pilot_gsusc_d4` |
| percolation_susceptibility_gpu, d = 5 | 2.839 | batched probe inside the pilot (5 probes) | 2.8926 ± 0.0025 | new `pilot_gsusc_d5` |
| percolation_tau_zd bin, d = 3 (CPU) | 1.204 | batched probe inside the pilot (5 probes) | 1.1287 ± 0.0113 | new `pilot_tau_d3` |
| percolation_tau_zd bin, d = 4 (CPU) | 1.274 | batched probe inside the pilot (5 probes) | 1.3606 ± 0.0148 | new `pilot_tau_d4` |
| percolation_tau_zd bin, d = 5 (CPU) | 1.484 | batched probe inside the pilot (5 probes) | 1.2656 ± 0.0020 | new `pilot_tau_d5` |

### 2b. Correction to scaling, eq. (232) fits: one dataset per row

*used* marks the constants that set the eq. (720) bounds in Table 1.

| model / observable | dataset (replicates) | ladder | ω₁ | a₁ | identification | a₀ | exponent from this fit | cv over the ladder | reference correction exponent | used |
|---|---|---|---|---|---|---|---|---|---|---|
| srw | 01 Exp B (5 reps) | k = 8..256 | 1.0155 ± 0.1050 | −0.2748 ± 0.0597 | identified | 0.7979 ± 0.0017 | γ = 0.5000 ± 0.0003 | 0.7771 | exact ω₁ = 1, a₁ = −1/4, a₀ = 0.797885 | |
| srw | 01 Exp B (6 reps, pooled) | k = 8..256 | 0.9836 ± 0.1113 | −0.2353 ± 0.0432 | identified | — | — | — | exact ω₁ = 1, a₁ = −1/4 | |
| rwre p = 1/2 | 02 A2 (3 reps) | k = 4..1024 | 1.096 ± 0.430 | −0.281 ± 0.165 | identified | 0.776 | — | 0.783 | srw's exact values | used |
| rwre p = 1/3 | 02 A5 pilot | k = 4..1024 | 0.019 ± 0.001 | +14.8 | **degenerate** with log a₀ | — | — | — | unknown | |
| percolation2d, south, box | 03 P2 deep (1 run) | i = 8..512 | 0.380 | 0.422 | no se (R = 1) | 0.395 | d_f = 1.9161 | 0.476 → 0.429 | ω = 3/2; boundary 1 | |
| percolation2d, south, box | 03 P2 deep, bias‑decay fit of γ̂(i) | i = 8..512 | 0.329 | −0.051 (of γ̂(i)) | no se (R = 1) | — | d_f = 1.9059 (γ∞) | — | 〃 | |
| percolation2d, origin, box | 03 P3 deep origin (1 run) | i = 8..512 | 0.262 | 1.44 | no se (R = 1) | — | γ/ν = 1.8926 | 1.006 → 1.411 (grows) | — | |
| percolation2d, origin, box | 03 P3 deep origin, bias‑decay | i = 8..512 | 0.426 | — | no se (R = 1) | — | γ/ν = 1.8104 (γ∞) | — | — | |
| percolation2d, south, cylinder | 03 P4 deep (1 run) | i = 8..512 | 2.80 | −1.13 | no se (R = 1) | — | d_f = 1.9002 | 0.435 → 0.372 | ω = 3/2 | |
| percolation_tau, d = 2, bin | models/README (1 run, 4·10⁹) | s = 8..2048 | 0.98 | −0.89 | no se (R = 1) | — | τ = 2.044 | 0.578–0.617 | Ω = 72/91 = 0.791 | |
| percolation_zd face_far, d = 3 | 05 H2 (1 run) | i = 8..256 | 2.34 | n/r | no se (R = 1) | — | d_f = 2.5295 | 0.995 → 0.742 | ω ≈ 1.61 | |
| percolation_zd face, d = 3 | 05 H2 (1 run) | i = 8..256 | 0.89 | n/r | no se (R = 1) | — | d_f = 2.5591 | 0.447 → 0.282 | ω ≈ 1.61 | |
| percolation_zd face_far, d = 4 | 05 H2 (1 run) | i = 4..64 | 1.49 | n/r | no se (R = 1) | — | d_f = 3.0501 | 1.212 → 0.680 | ω ≈ 1.13 | |
| percolation_zd face, d = 4 | 05 H2 (1 run) | i = 4..64 | 0.63 | n/r | no se (R = 1) | — | d_f = 3.2102 | 0.466 → 0.130 | ω ≈ 1.13 | |
| percolation_tau_zd, d = 3 | 05 H5, direct fit (1 run) | s = 8..1024 | 0.16 or 0.19 (README does not attribute) | n/r | no se (R = 1) | — | τ = 2.267 | 0.34–0.38 | Ω ≈ 0.64 | |
| percolation_tau_zd, d = 3 | 05 H5, bias‑decay fit | s = 8..1024 | 0.16 or 0.19 | n/r | no se (R = 1) | — | τ = 2.231 | — | Ω ≈ 0.64 | |
| percolation_tau_zd, d = 6 | 05 H5, direct fit (1 run) | s = 8..1024 | 0.032 | n/r | not converged | — | τ = 2.87 | 0.24–0.28 | mean field | |
| percolation_tau_zd, d = 2 | 05 H8 (1 run) | s = 8..512 | 0.603 | −0.692 | no se (R = 1) | 0.0175 | τ = 2.0597 | flat ± 7% | Ω = 0.791 | used |
| percolation_tau_zd, d = 3 | 05 H8 (1 run) | s = 8..512 | 0.730 | −0.671 | no se (R = 1) | 0.0307 | τ = 2.1807 | flat ± 7% | Ω ≈ 0.64 | |
| percolation_tau_zd, d = 4 | 05 H8 (1 run) | s = 8..512 | 0.205 | −1.153 | no se (R = 1) | 0.0668 | τ = 2.3647 | flat ± 7% | Ω ≈ 0.37 | |
| percolation_tau_zd, d = 5 | 05 H8 (1 run) | s = 8..512 | 0.667 | −0.485 | no se (R = 1) | 0.0278 | τ = 2.3722 | flat ± 7% | Ω ≈ 0.21 | |
| percolation_tau_zd, d = 6 | 05 H8 (1 run) | s = 8..512 | 0.037 | −12.9 | not converged | 7.6·10³ | τ = 2.8196 | flat ± 7% | mean field | |
| percolation_tau_zd, d = 7 | 05 H8 (1 run) | s = 8..512 | 13.91 | 3.8·10¹⁰ | not converged | 0.0188 | τ = 2.4454 | falling | mean field | |
| percolation_tau_zd, d = 8 | 05 H8 (1 run) | s = 8..512 | 13.67 | 2.7·10¹⁰ | not converged | 0.0161 | τ = 2.4640 | falling | mean field | |
| susceptibility, d = 2, bf 4 | 06 Arm A pilot (5 reps) | x = 4..128 | 0.5917 ± 0.1092 | 1.0114 ± 0.1925 | identified | — | — | 0.5948 ± 0.0012 (mean) | Δ₁ = ων = 2, plus analytic ε¹‡ | used |
| susceptibility, d = 2, bf 4 | 06 Arm A run, refit | x = 32..1024 | 10.3 | 8·10¹³ | not converged | — | — | — | 〃 | |
| susceptibility, d = 2, bf 8 | 06 Arm B, ω₁ ≡ 1 declared (moment 1) | x = 4..128 | 1 (declared) | n/r | declared | — | γ = 2.3916 ± 0.0076 | 0.32–0.33 | 〃 | |
| susceptibility, d = 2, bf 8 | 06 Arm B, ω₁ ≡ 1 declared (moment 2) | x = 4..128 | 1 (declared) | n/r | declared | — | e₂ = 4.9105 ± 0.0284 | — | 〃 | |
| susceptibility, d = 2, bf 8 | 06 diagnostics, γ fixed at 43/18 | x = 4..128 | 0.81 ± 0.10 | — | — | — | — | — | 〃 | |
| susceptibility, d = 2, bf 8 | 06 diagnostics, two corrections at fixed exponents 1.055 and 2.110 | x = 4..128 | fixed | a₁ = +1.72, a₂ = −1.91 | — | — | γ = 2.3963 | — | 〃 | |
| percolation2d, south, box | **re‑draw** of `samples_df_south.json` (3 reps) | i = 8..512 | 0.782 ± 0.070 | 0.400 ± 0.019 | identified | 0.4595 | d_f = 1.8978 | 0.427–0.477 | ω = 3/2; boundary 1 | used |
| percolation2d, origin, box | **re‑draw** of `samples_df_origin.json` (3 reps) | i = 8..512 | 0.691 ± 0.065 | 0.835 ± 0.053 | identified | 0.3002 | γ/ν = 1.7980 | 1.006–1.420 | — | used |
| percolation2d, south, cylinder | **re‑draw** of `samples_gpu_deep.json` (12 reps) | i = 8..512 | 0.499 ± 0.247 | −0.071 ± 0.263 | **unidentified**: a₁ | 0.5739 | d_f = 1.8944 | 0.368–0.434 | ω = 3/2 | used |
| percolation_tau, d = 2, bin | **re‑draw** of `samples_gpu_bin.json` (12 reps) | s = 8..2048 | 0.885 ± 0.219 | −0.818 ± 1.844 | **unidentified**: a₁ | 0.01597 | τ = 2.0460 | 0.577–0.621 | Ω = 0.791 | used |
| percolation_tau, d = 2, tail | **re‑draw** of `samples_tau_tail.json` (3 reps) | s = 8..2048 | 0.657 ± 0.144 | −0.478 ± 0.011 | identified | 0.03636 | τ = 2.0499 | 0.419–0.448 | Ω = 0.791 | used |
| percolation_zd face, d = 3 | **re‑draw** of `samples_gpu_deep_d3_face.json` (12 reps) | i = 8..256 | 0.434 ± 0.127 | −0.990 ± 1.356 | **unidentified**: a₁ | 0.6687 | d_f = 2.5082 | 0.299–0.446 | ω ≈ 1.61 | used |
| percolation_zd face_far, d = 3 | **re‑draw** of `samples_df_d3_facefar.json` (3 reps) | i = 8..256 | 0.316 ± 3.945 | −0.509 ± 3.6e+08 | **unidentified**: ω₁, a₁ | 0.1904 | d_f = 2.4749 | 0.851–0.989 | ω ≈ 1.61 |  |
| percolation_zd face_far, d = 4 | **re‑draw** of `samples_df_d4_facefar.json` (3 reps) | i = 4..64 | 1.520 ± 0.157 | −2.404 ± 0.392 | identified | 0.1276 | d_f = 3.0762 | 0.686–1.213 | ω ≈ 1.13 |  |
| percolation_zd face, d = 4 | **re‑draw** of `samples_df_d4_face.json` (3 reps) | i = 4..64 | 0.765 ± 0.113 | −1.395 ± 0.096 | identified | 0.4756 | d_f = 3.2477 | 0.138–0.466 | ω ≈ 1.13 | used |
| percolation_tau_zd, d = 3, box_exponent 1/2 | **re‑draw** of `samples_tau_d3_boxexp_half.json` (1 rep) | s = 8..512 | 0.491 | −0.706 | no se (R = 1) | 0.03588 | τ = 2.2038 | 0.144–0.290 | Ω ≈ 0.64 | used |
| percolation_tau, d = 2, shared lattice (CPU run) | 10 D, own fit (1 run) | s = 8..2048 | 0.566 | −0.707 | no se (R = 1) | — | — | 0.059–1.036 | Ω = 0.791 | used |
| percolation_tau, d = 2, shared lattice (GPU run) | 10 D, own fit (1 run) | s = 8..2048 | 0.605 | −0.695 | no se (R = 1) | — | — | 0.059–1.022 | Ω = 0.791 | used |
| susceptibility, d = 2, bf 8, moment 2 | **re‑draw** of `samples_gpu_deep_moment2.json` (12 reps), fitted on x ≥ 4 | x = 4..128 | 0.637 ± 0.242 | 1.269 ± 5.051 | **unidentified**: a₁ | — | — | — | 〃 | used |
| percolation_zd_gpu axis, d = 3 | **new pilot** `pilot_df_d3_axis` (5 reps) | i = 4..256 | 2.457 ± 0.579 | −1.554 ± 3.974 | **unidentified**: a₁ | — | — | 0.585–0.939 | ω ≈ 1.61 | used |
| percolation_zd_gpu axis, d = 3 | **new pilot** `pilot_df_d3_axis_wide` (5 reps) | i = 4..512 | 1.875 ± 0.079 | −0.814 ± 0.062 | identified | — | — | 0.583–0.939 | ω ≈ 1.61 | used |
| percolation_zd_gpu face_far, d = 3 | **new pilot** `pilot_df_d3_facefar_wide` (5 reps) | i = 4..512 | 1.803 ± 0.149 | −1.469 ± 0.260 | identified | — | — | 0.828–1.175 | ω ≈ 1.61 | used |
| percolation_zd_gpu face_far, d = 4 | **new pilot** `pilot_df_d4_facefar` (5 reps) | i = 4..128 | 1.385 ± 0.067 | −2.180 ± 0.120 | identified | — | — | 0.666–1.215 | ω ≈ 1.13 | used |
| percolation_zd_gpu face_far, d = 5 | **new pilot** `pilot_df_d5_facefar` (5 reps) | i = 4..64 | 1.157 ± 0.165 | −2.841 ± 0.327 | identified | — | — | 0.389–1.056 | ω ≈ 0.74 | used |
| percolation_susceptibility_gpu, d = 3 | **new pilot** `pilot_gsusc_d3` (5 reps) | x = 4..256 | 1.080 ± 0.253 | 0.438 ± 0.185 | identified | — | — | 0.686–0.712 | Δ₁ = Ω/σ ≈ 1.41 | used |
| percolation_susceptibility_gpu, d = 4 | **new pilot** `pilot_gsusc_d4` (5 reps) | x = 4..128 | 0.025 ± 1.745 | 3.205 ± 515.401 | **unidentified**: ω₁, a₁ | — | — | 0.496–0.555 | Δ₁ ≈ 0.77 |  |
| percolation_susceptibility_gpu, d = 5 | **new pilot** `pilot_gsusc_d5` (5 reps) | x = 4..64 | 0.590 ± 0.254 | −0.425 ± 0.832 | **unidentified**: a₁ | — | — | 0.244–0.305 | Δ₁ ≈ 0.43 | used |
| percolation_tau_zd bin, d = 3 (CPU) | **new pilot** `pilot_tau_d3` (5 reps) | s = 8..512 | 0.343 ± 0.086 | −0.763 ± 0.233 | identified | — | — | 0.341–0.383 | Ω ≈ 0.64 | used |
| percolation_tau_zd bin, d = 4 (CPU) | **new pilot** `pilot_tau_d4` (5 reps) | s = 8..512 | 0.060 ± 0.071 | −7.835 ± 2.914 | **unidentified**: ω₁ | — | — | 0.300–0.338 | Ω ≈ 0.37 |  |
| percolation_tau_zd bin, d = 5 (CPU) | **new pilot** `pilot_tau_d5` (5 reps) | s = 8..512 | 0.193 ± 0.235 | −1.008 ± 2.355 | **unidentified**: ω₁, a₁ | — | — | 0.262–0.333 | Ω ≈ 0.21 |  |

‡ `experiments/06_susceptibility/README.md` writes the theoretical correction as
Δ₁ = Ων ≈ 1.055. Ω is the correction exponent in cluster size, and in ε it becomes
Ω/σ = ων = (3/2)(4/3) = **2**. So the "second correction at x^{−2.11}" found there sits
where Δ₁ = 2 is expected, and the leading x^{−1} term is the nonlinear scaling field the
README already identifies. This does not change any number the README reports. It does
change which fixed exponents the two‑term diagnostic should use (1 and 2, not 1.055 and
2.110). **This is a flag and was not re‑checked here.**

---

## 3. Exponents obtained indirectly, through scaling relations

The relations are the ones listed on the Wikipedia page:
- hyperscaling (valid for d < 6): τ = 1 + d/d_f, β/ν = d − d_f, η = 2 + d − 2d_f
- δ = d_f/(d − d_f) = 1/(τ − 2)
- the {d_f, ν} relations: ν = γ/(2d_f − d), β = ν(d − d_f), α = 2 − νd, σ = 1/(νd_f)
- the moment relations: 1/σ = e₂ − e₁, τ = 3 − e₁/(e₂ − e₁)

**Inputs are the pinned values**, the narrowest eq. (720) bound for each exponent and
dimension in Table 1, whatever the observable. **Errors are propagated as bounds.** A
bound is a worst case, so the derived interval is the range of the relation over the box
of input bounds (every relation here is monotone in each input), not a quadrature sum.
"Unbounded" means a denominator's range contains 0.

| d | exponent | relation | inputs (pinned: narrowest eq. (720) bound in Table 1) | derived | worst case over the input bounds | literature | literature inside? |
|---|---|---|---|---|---|---|---|
| 2 | β/ν | d − d_f | d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **0.1070** | [0.0848, 0.1292] | 5/48 = 0.104167 | yes |
| 2 | η | 2 + d − 2d_f | d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **0.2140** | [0.1696, 0.2584] | 5/24 = 0.208333 | yes |
| 2 | τ | 1 + d/d_f | d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **2.0565** | [2.0443, 2.0691] | 187/91 = 2.054945 | yes |
| 2 | δ | d_f/(d − d_f) | d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **17.6916** | [14.4770, 22.5915] | 91/5 = 18.2 | yes |
| 2 | d_f | d/(τ − 1) | τ = 2.0414 ± 0.0214 (10 C, GPU, 12 reps, 64..2048) | **1.9205** | [1.8819, 1.9608] | 91/48 = 1.895833 | yes |
| 2 | η | 2 − γ/ν | γ/ν = 1.7716 ± 0.1028 (03 P3 deep origin, 4·10⁹, 16..512) | **0.2284** | [0.1256, 0.3312] | 5/24 = 0.208333 | yes |
| 2 | ν | γ/(2d_f − d) | γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024); d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **1.3205** | [1.1735, 1.4751] | 4/3 | yes |
| 2 | β | ν(d − d_f) | γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024); d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **0.1413** | [0.0995, 0.1906] | 5/36 = 0.138889 | yes |
| 2 | α | 2 − νd | γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024); d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **−0.6411** | [−0.9501, −0.3471] | −2/3 | yes |
| 2 | σ | 1/(νd_f) | γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024); d_f = 1.8930 ± 0.0222 (03 P2 deep, 4·10⁹ sites, 64..512) | **0.4000** | [0.3624, 0.4449] | 36/91 = 0.395604 | yes |
| 2 | 1/σ | e₂ − e₁ | e₂ = 4.8611 ± 2.4188 (06 Arm B, n = 1500, 16..128); γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024) | **2.5026** | [−0.1266, 5.1318] | 91/36 = 2.527778 | yes |
| 2 | τ | 3 − e₁/(e₂ − e₁) | e₂ = 4.8611 ± 2.4188 (06 Arm B, n = 1500, 16..128); γ = 2.3585 ± 0.2104 (06 Arm A (article method), 5 reps, n = 57, 32..1024) | **2.0576** | unbounded (a denominator's bound contains 0) | 187/91 = 2.054945 | yes (unbounded) |
| 3 | β/ν | d − d_f | d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **0.4409** | [0.4208, 0.4610] | 0.47707(10)‡ | **no** |
| 3 | η | 2 + d − 2d_f | d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **−0.1182** | [−0.1585, −0.0780] | −0.0431(14) | **no** |
| 3 | τ | 1 + d/d_f | d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **2.1723** | [2.1631, 2.1816] | 2.18909(5) | **no** |
| 3 | δ | d_f/(d − d_f) | d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **5.8046** | [5.5077, 6.1300] | 5.29(6) [also 5.16(4)] | **no** |
| 3 | d_f | d/(τ − 1) | τ = 2.1582 ± 0.1336 (05 H5(c), box_exponent = 1/2, 16..512) | **2.5902** | [2.3224, 2.9280] | 2.52293(10) | yes |
| 3 | ν | γ/(2d_f − d) | γ = 1.7768 ± 0.0402 (new pilot `pilot_gsusc_d3`, GPU, 5 reps, 8..256); d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **0.8388** | [0.8045, 0.8744] | 0.8762(7) | **no** |
| 3 | β | ν(d − d_f) | γ = 1.7768 ± 0.0402 (new pilot `pilot_gsusc_d3`, GPU, 5 reps, 8..256); d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **0.3698** | [0.3385, 0.4031] | 0.4181(8) [also 0.4053(5)] | **no** |
| 3 | α | 2 − νd | γ = 1.7768 ± 0.0402 (new pilot `pilot_gsusc_d3`, GPU, 5 reps, 8..256); d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **−0.5164** | [−0.6232, −0.4136] | −0.625(3) | **no** |
| 3 | σ | 1/(νd_f) | γ = 1.7768 ± 0.0402 (new pilot `pilot_gsusc_d3`, GPU, 5 reps, 8..256); d_f = 2.5591 ± 0.0201 (new pilot `pilot_df_d3_axis`, GPU, 5 reps, 32..256) | **0.4659** | [0.4504, 0.4819] | 0.4524(6) | yes |
| 4 | β/ν | d − d_f | d_f = 3.0770 ± 0.0866 (new pilot `pilot_df_d4_facefar`, GPU, 5 reps, 16..128) | **0.9230** | [0.8364, 1.0096] | 0.9554(7)‡ | yes |
| 4 | η | 2 + d − 2d_f | d_f = 3.0770 ± 0.0866 (new pilot `pilot_df_d4_facefar`, GPU, 5 reps, 16..128) | **−0.1540** | [−0.3271, 0.0192] | −0.0929(9) | yes |
| 4 | τ | 1 + d/d_f | d_f = 3.0770 ± 0.0866 (new pilot `pilot_df_d4_facefar`, GPU, 5 reps, 16..128) | **2.3000** | [2.2644, 2.3376] | 2.3142(5) | yes |
| 4 | δ | d_f/(d − d_f) | d_f = 3.0770 ± 0.0866 (new pilot `pilot_df_d4_facefar`, GPU, 5 reps, 16..128) | **3.3336** | [2.9619, 3.7822] | 3.175(8) | yes |
| 5 | β/ν | d − d_f | d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **1.3445** | [1.1324, 1.5566] | 1.4740(14)‡ | yes |
| 5 | η | 2 + d − 2d_f | d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **−0.3109** | [−0.7351, 0.1133] | −0.0547(10) | yes |
| 5 | τ | 1 + d/d_f | d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **2.3678** | [2.2928, 2.4521] | 2.4175(2) | yes |
| 5 | δ | d_f/(d − d_f) | d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **2.7188** | [2.2120, 3.4153] | 2.3952(12) | yes |
| 5 | ν | γ/(2d_f − d) | γ = 1.2188 ± 0.7959 (new pilot `pilot_gsusc_d5`, GPU, 5 reps, 8..64); d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **0.5274** | [0.1546, 1.0679] | 0.5739(1) | yes |
| 5 | β | ν(d − d_f) | γ = 1.2188 ± 0.7959 (new pilot `pilot_gsusc_d5`, GPU, 5 reps, 8..64); d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **0.7091** | [0.1751, 1.6623] | 0.8454(2) | yes |
| 5 | α | 2 − νd | γ = 1.2188 ± 0.7959 (new pilot `pilot_gsusc_d5`, GPU, 5 reps, 8..64); d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **−0.6371** | [−3.3394, 1.2270] | −0.870(1) | yes |
| 5 | σ | 1/(νd_f) | γ = 1.2188 ± 0.7959 (new pilot `pilot_gsusc_d5`, GPU, 5 reps, 8..64); d_f = 3.6555 ± 0.2121 (new pilot `pilot_df_d5_facefar`, GPU, 5 reps, 16..64) | **0.5187** | [0.2720, 1.6723] | 0.49396(13) | yes |

‡ β/ν = d − d_f is computed from the best d_f; Wikipedia has no separate row for it.

**What the table says.**
- **d = 2:** every derived interval contains the exact value.
- **d = 3:** the pinned d_f is the axis pilot, whose bound misses 2.52293. Every relation
  that uses it misses too, except σ. The τ route (d_f from τ) contains the literature. That
  is the pinning rule working as specified: it chooses by width, and here the narrowest
  bound is the one whose correction model is wrong.
- **d = 4 and 5:** the bounds are wide enough to contain the literature everywhere.

For d = 3, the Wikipedia table also lists β = 0.4053(5), δ = 5.16(4) and γ = 1.819(3).
With the precise ν = 0.8762(7) and d_f = 2.52293(10), the scaling relations give
β = 0.4180, δ = 5.288 and γ = 1.7926. Those three entries are inconsistent with the rest
of the table, and the comparison above uses the consistent entries.

---

## 4. Simulation time to reach the literature's precision

**How each time was computed.**

1. **Pilot.** `src/study/pilot.py --replicates 5` draws the pilot on the recipe written
   beside each experiment (`recipes/samples_pilot_*.json`). It measures d from five
   batched cost probes, ω₁ and a₁ from eq. (232) fitted once to the pooled means, cv, and
   the throughput on xingu. The GPU pilots ran one at a time on an otherwise idle card.
   The three τ pilots ran on xingu's CPU, concurrently with each other.
2. **Plan.** `plan.py`'s `budget_for_target` finds the smallest budget whose predicted
   error on the **mean of R = 5 replicates** reaches the target. That error is
   √(bias² + sd²/5), with |bias| = C_b ρ^{−m₀ω₁}, sd = C_s/√n, and m₀ from the tuned
   `prop:opt` allocation (m = 6 rungs, ρ = 2). This is exactly
   `plan.py --target-se σ --replicates 5`. The only change is that the budget ceiling was
   raised from 10¹⁸ to 10⁴⁰, so that astronomical answers print as numbers instead of
   "unreachable".
3. **Targets.** σ is the smallest uncertainty quoted on Wikipedia for that exponent and
   dimension. σ/10 is one more decimal place, σ/100 two more.
4. **Range.** In brackets, the same plan re‑made with ω₁ ± 1 se. The ω₁ error is by far
   the dominant one: a₁, cv and d move m₀ logarithmically. The range is not always
   monotone in ω₁. m₀ is an integer staircase, and the bisection stops at the first budget
   that meets the target.
5. **Wall clock** is for one xingu (one RTX 5090, or one CPU core for τ). Replicates are
   independent, so N machines divide it by N.

**✗** = the plan's top rung needs more than 2³¹ − 1 sites in **one** sample. That is the
int32 label space of both labellers (`ndimage.label` and `cupyx`), and about 20–40 GiB of
device memory. Such a plan cannot run on the current models however long it is given; it
would need the slab‑streaming sampler (`models/percolation_zd_stream.py`). The γ pilots
could shrink `box_factor` instead: in d = 2 that moved only the amplitude.

**The last column** ignores bias completely, and is the only place in this report that uses the statistical term alone. It is the pilot's own wall clock scaled by
(se/target)² at the pilot's deepest 4‑rung window. It is what the time would be if that
window were already unbiased: an optimistic floor for that ladder, not a plan. It is the
only number available where the pilot could not identify (ω₁, a₁).

**Check before trusting a row.** Table 4a puts two numbers side by side for each pilot's
deepest window: the bias predicted by its own (ω₁, a₁), and the gap that window actually
shows to the literature.
- **Under‑predicted, so the time is a lower bound.** d_f on the axis (both ladders) and
  at d = 4, 5, and γ at d = 3 predict a bias 1.5× to ≈ 360× smaller than observed. The
  one‑correction fit sees the fast curvature at the bottom of a short ladder and misses a
  slower drift that is still present at the top. This is the effective‑exponent effect
  `06_susceptibility` documented in d = 2. The drift's true exponent is smaller than the
  fitted ω₁, and precision then costs more, through B^{−ω₁/(d+2ω₁)}.
- **Consistent.** d_f with `face_far` at d = 3 (+0.0004 predicted, +0.0016 ± 0.0062
  observed), and γ at d = 5 (+0.042 vs +0.040, though a₁ is unidentified there).
- **Over‑predicted, so the time is probably too pessimistic.** τ at d = 3 (−0.045 vs
  −0.015). Its ω₁ = 0.34 describes a correction slower than the data show.
- **Unidentified.** τ at d = 4 and 5, and γ at d = 4, have (ω₁, a₁) undetermined, so
  `plan.py`'s verdict is "do not plan on this pilot".
  - τ(4) and γ(4) have no finite plan at all: an ω₁ near 0 makes the bias floor never
    decay.
  - τ(5), and γ(5) with a₁ unidentified, are printed with that warning.
  - For all of these, only the statistical‑term‑only column is meaningful.

**Other things the times leave out.**
- **Constant throughput.** Throughput is held constant per site. The clocks measured
  d̂ = 3.03, 4.21 and 5.33 against the declared 3, 4 and 5 on the d_f ladders, which is
  cache pressure growing with box size. Larger boxes will be slower per site than
  assumed.
- **p_c digits.** p_c enters as a fixed input to 7–8 significant digits (`P_C_SITE_HYPERCUBIC`). A
  shift δp moves the system off criticality by δp·L^{1/ν}. At the rungs the σ/100 plans
  ask for (L ~ 10⁴–10⁵ in d = 3) that product reaches 10⁻³–10⁻², so p_c itself would have
  to be improved first.
- **The rate.** The error falls as B^{−ω₁/(d+2ω₁)}. For d_f, d is the spatial dimension,
  so each extra decimal costs 10^{(d+2ω₁)/ω₁}: about 10^{3.7} at d = 3 and 10^{6.6} at
  d = 5. On the τ ladder d is small (≈ 1.1–1.4), but the fitted ω₁ is smaller still, so τ
  ends up no cheaper.
- **CPU τ pilots.** They ran single‑threaded, three at a time. A GPU port of
  `percolation_tau_zd` would be about 20× faster, judging by the d = 2 port in experiment
  10.
- **d = 2.** Every exponent in d = 2 is known exactly (Table 1), so there is no
  literature precision to match, and no d = 2 row appears here.

**Table 4a: pilot constants, and a check of their extrapolation.**

| exponent | observable, pilot ladder | pilot: R, wall clock | d̂ (clock) | ω₁ | a₁ | cv | throughput (sites/s) | pilot exponent at its deepest 4‑rung window ± eq. (720) | observed − literature | bias predicted by the pilot's own ω₁, a₁ | literature best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d_f (d=3) | axis (slab k=1), cyl., i=4..256 | 5, 38 s drawn | 3.026 | 2.457 ± 0.579 | −1.55 ± 4 (unidentified) | 0.714 | 8.6e+09 | 2.5591 ± 0.0201 (32..256) | +0.0362 | +0.0001 | 2.52293(10) |
| d_f (d=3) | axis, cyl., i=4..512 | 5, 300 s drawn | 3.234 | 1.875 ± 0.079 | −0.814 ± 0.062 | 0.698 | 5.1e+09 | 2.5432 ± 0.0255 (64..512) | +0.0202 | +0.0002 | 2.52293(10) |
| d_f (d=3) | face_far, cyl., i=4..512 | 5, 192 s drawn | 3.132 | 1.803 ± 0.149 | −1.47 ± 0.26 | 0.909 | 6.8e+09 | 2.5246 ± 0.0648 (64..512) | +0.0016 | +0.0004 | 2.52293(10) |
| d_f (d=4) | face_far, cyl., i=4..128 | 5, 218 s drawn | 4.214 | 1.385 ± 0.067 | −2.18 ± 0.12 | 0.835 | 5.8e+09 | 3.0770 ± 0.0866 (16..128) | +0.0324 | +0.0208 | 3.0446(7) |
| d_f (d=5) | face_far, cyl., i=4..64 | 5, 245 s drawn | 5.326 | 1.157 ± 0.165 | −2.84 ± 0.33 | 0.624 | 4.5e+09 | 3.7005 ± 0.2735 (8..64) | +0.1745 | +0.1100 | 3.5260(14) |
| τ (d=3) | bin, torus, s=8..512 (CPU) | 5, 193 s drawn | 1.129 | 0.343 ± 0.086 | −0.763 ± 0.23 | 0.355 | 5.2e+07 | 2.1745 ± 0.3817 (64..512) | −0.0146 | −0.0448 | 2.18909(5) |
| τ (d=4) | bin, torus, s=8..512 (CPU) | 5, 229 s drawn | 1.361 | 0.060 ± 0.071 **unidentified** | −7.84 ± 2.9 | 0.311 | 4.5e+07 | 2.2862 n/a (ω₁ unidentified) (64..512) | −0.0280 | −0.3436 | 2.3142(5) |
| τ (d=5) | bin, torus, s=8..512 (CPU) | 5, 286 s drawn | 1.266 | 0.193 ± 0.235 **unidentified** | −1.01 ± 2.4 (unidentified) | 0.288 | 3.3e+07 | 2.3613 n/a (ω₁ unidentified) (64..512) | −0.0562 | −0.0717 | 2.4175(2) |
| γ (d=3) | mean cluster size, ε₀=0.15, x=4..256 | 5, 416 s drawn | 2.717 | 1.080 ± 0.253 | 0.438 ± 0.18 | 0.695 | 8.7e+08 | 1.7775 ± 0.0507 (32..256) | −0.0155 / −0.0415 | −0.0044 | 1.819(3) / 1.793(3) |
| γ (d=4) | mean cluster size, ε₀=0.09, x=4..128 | 5, 671 s drawn | 2.822 | 0.025 ± 1.745 **unidentified** | 3.2 ± 5.2e+02 (unidentified) | 0.528 | 7.5e+08 | 1.4371 n/a (ω₁ unidentified) (16..128) | +0.0071 | −0.0731 | 1.430(6) |
| γ (d=5) | mean cluster size, ε₀=0.07, x=4..64 | 5, 916 s drawn | 2.893 | 0.590 ± 0.254 | −0.425 ± 0.83 (unidentified) | 0.269 | 6.9e+08 | 1.2188 ± 0.7959 (8..64) | +0.0396 | +0.0421 | 1.1792(7) |


**Table 4b: wall clock on one xingu (RTX 5090 or one CPU core), R = 5 replicates, error on the answer ≤ target.**

| exponent | observable | literature best σ | time to σ | time to σ/10 (one more decimal) | time to σ/100 (two more decimals) | statistical term only, pilot window, bias ignored: σ / σ/10 / σ/100 |
|---|---|---|---|---|---|---|
| d_f (d=3) | axis (slab k=1), cyl., i=4..256 | 2.52293(10) → σ = 0.0001 | **20 d** (ω₁+1se: 3 d; ω₁−1se: 3.9 yr); m₀=4, top rung 1024, 1.1·10^9 sites/sample ⚠ plan.py verdict: do not plan (a₁ unidentified) | **613 yr** (ω₁+1se: 19 yr; ω₁−1se: 4.6·10^4 yr); m₀=6, top rung 4096, 6.9·10^10 sites/sample ✗ ⚠ plan.py verdict: do not plan (a₁ unidentified) | **1.7·10^5 yr** (ω₁+1se: 9.8·10^4 yr; ω₁−1se: 2.9·10^7 yr); m₀=7, top rung 8192, 5.5·10^11 sites/sample ✗ ⚠ plan.py verdict: do not plan (a₁ unidentified) | 8.2 h / 34 d / 9.3 yr |
| d_f (d=3) | axis, cyl., i=4..512 | 2.52293(10) → σ = 0.0001 | **1.0 yr** (ω₁+1se: 194 d; ω₁−1se: 11 yr); m₀=5, top rung 2048, 8.6·10^9 sites/sample ✗ | **3,537 yr** (ω₁+1se: 5,327 yr; ω₁−1se: 1.8·10^4 yr); m₀=7, top rung 8192, 5.5·10^11 sites/sample ✗ | **2.8·10^7 yr** (ω₁+1se: 7.7·10^7 yr; ω₁−1se: 3.6·10^7 yr); m₀=9, top rung 32768, 3.5·10^13 sites/sample ✗ | 9 d / 2.6 yr / 258 yr |
| d_f (d=3) | face_far, cyl., i=4..512 | 2.52293(10) → σ = 0.0001 | **5.3 yr** (ω₁+1se: 17 yr; ω₁−1se: 103 yr); m₀=6, top rung 4096, 6.9·10^10 sites/sample ✗ | **4.5·10^4 yr** (ω₁+1se: 5,606 yr; ω₁−1se: 6.4·10^5 yr); m₀=8, top rung 16384, 4.4·10^12 sites/sample ✗ | **4.2·10^8 yr** (ω₁+1se: 3.0·10^7 yr; ω₁−1se: 4.0·10^9 yr); m₀=10, top rung 65536, 2.8·10^14 sites/sample ✗ | 9 d / 2.3 yr / 235 yr |
| d_f (d=4) | face_far, cyl., i=4..128 | 3.0446(7) → σ = 0.0007 | **2,919 yr** (ω₁+1se: 522 yr; ω₁−1se: 1.2·10^4 yr); m₀=6, top rung 4096, 2.8·10^14 sites/sample ✗ | **2.7·10^8 yr** (ω₁+1se: 4.4·10^7 yr; ω₁−1se: 4.2·10^8 yr); m₀=9, top rung 32768, 1.2·10^18 sites/sample ✗ | **7.5·10^12 yr** (ω₁+1se: 8.9·10^12 yr; ω₁−1se: 1.1·10^14 yr); m₀=11, top rung 131072, 3.0·10^20 sites/sample ✗ | 55 min / 4 d / 1.1 yr |
| d_f (d=5) | face_far, cyl., i=4..64 | 3.5260(14) → σ = 0.0014 | **2.3·10^7 yr** (ω₁+1se: 6.8·10^5 yr; ω₁−1se: 2.3·10^10 yr); m₀=7, top rung 8192, 3.7·10^19 sites/sample ✗ | **5.1·10^13 yr** (ω₁+1se: 1.4·10^12 yr; ω₁−1se: 4.7·10^16 yr); m₀=10, top rung 65536, 1.2·10^24 sites/sample ✗ | **1.3·10^20 yr** (ω₁+1se: 2.6·10^17 yr; ω₁−1se: unreachable); m₀=13, top rung 524288, 4.0·10^28 sites/sample ✗ | 1.9 h / 8 d / 2.1 yr |
| τ (d=3) | bin, torus, s=8..512 (CPU) | 2.18909(5) → σ = 5e-05 | **2.2·10^9 yr** (ω₁+1se: 9.5·10^6 yr; ω₁−1se: 2.8·10^13 yr); m₀=33, top rung 5.5·10^11, 4.7·10^16 sites/sample ✗ | **7.9·10^14 yr** (ω₁+1se: 9.2·10^11 yr; ω₁−1se: 1.6·10^20 yr); m₀=43, top rung 5.6·10^14, 2.1·10^20 sites/sample ✗ | **4.2·10^20 yr** (ω₁+1se: 5.3·10^16 yr; ω₁−1se: 6.0·10^26 yr); m₀=53, top rung 5.8·10^17, 9.6·10^23 sites/sample ✗ | 8 d / 2.3 yr / 231 yr |
| τ (d=4) | bin, torus, s=8..512 (CPU) | 2.3142(5) → σ = 0.0005 | not plannable: ω₁ unidentified | not plannable: ω₁ unidentified | not plannable: ω₁ unidentified | 2.3 h / 9 d / 2.6 yr |
| τ (d=5) | bin, torus, s=8..512 (CPU) | 2.4175(2) → σ = 0.0002 | **9.6·10^17 yr** (ω₁+1se: 2.0·10^6 yr; ω₁−1se: unreachable); m₀=49, top rung 3.6·10^16, 2.2·10^26 sites/sample ✗ ⚠ plan.py verdict: do not plan (ω₁, a₁ unidentified) | **2.3·10^27 yr** (ω₁+1se: 6.1·10^11 yr; ω₁−1se: unreachable); m₀=66, top rung 4.7·10^21, 5.4·10^33 sites/sample ✗ ⚠ plan.py verdict: do not plan (ω₁, a₁ unidentified) | not plannable: ω₁ and a₁ unidentified | 1 d / 103 d / 28 yr |
| γ (d=3) | mean cluster size, ε₀=0.15, x=4..256 | 1.819(3) / 1.793(3) → σ = 0.003 | **3 d** (ω₁+1se: 14.3 h; ω₁−1se: 87 d); m₀=4, top rung 1024, 5.7·10^9 sites/sample ✗ | **350 yr** (ω₁+1se: 15 yr; ω₁−1se: 3.5·10^4 yr); m₀=7, top rung 8192, 1.4·10^12 sites/sample ✗ | **3.0·10^7 yr** (ω₁+1se: 1.3·10^5 yr; ω₁−1se: 5.3·10^9 yr); m₀=11, top rung 131072, 2.1·10^15 sites/sample ✗ | 5 min / 8.4 h / 35 d |
| γ (d=4) | mean cluster size, ε₀=0.09, x=4..128 | 1.430(6) → σ = 0.006 | not plannable: ω₁ and a₁ unidentified | not plannable: ω₁ and a₁ unidentified | not plannable: ω₁ and a₁ unidentified | 2 min / 2.8 h / 11 d |
| γ (d=5) | mean cluster size, ε₀=0.07, x=4..64 | 1.1792(7) → σ = 0.0007 | **2.0·10^7 yr** (ω₁+1se: 3,598 yr; ω₁−1se: 2.4·10^14 yr); m₀=12, top rung 262144, 3.9·10^18 sites/sample ✗ ⚠ plan.py verdict: do not plan (a₁ unidentified) | **6.1·10^13 yr** (ω₁+1se: 1.0·10^9 yr; ω₁−1se: 1.1·10^25 yr); m₀=17, top rung 8.4·10^6, 8.2·10^22 sites/sample ✗ ⚠ plan.py verdict: do not plan (a₁ unidentified) | **5.5·10^20 yr** (ω₁+1se: 3.2·10^14 yr; ω₁−1se: unreachable); m₀=23, top rung 5.4·10^8, 1.3·10^28 sites/sample ✗ ⚠ plan.py verdict: do not plan (a₁ unidentified) | 10.6 h / 44 d / 12 yr |

---

## Not included, and why

- **d = 2 τ runs whose configuration cannot be pinned down.** The models/README pilot pair
  (its recipe says 4·10⁸ sites, its README 4·10⁹) and the "shared lattice, mean of 4"
  runs (no per‑run spread recorded). Experiment 10 C re‑ran their identical recipes with
  R = 12, and those rows are in Table 1.
- **Study runs with no write‑up.** The generated recipes `prof_long_df_d3/d4`,
  `fracdim_*_d3_k1`, `q_gate`, `prof_df_d2`, `prof_tau_d2`, `fractaldimautopilottest`,
  `taufixcheck` and `tautestcorellation` exist, but their `data/` is not on this host and
  no README reports their results.
- **`calibration/check_no_leakage.py` and `check_coverage.py`.** They recover *planted*
  exponents (synthetic model, 40 cells, slopes 1.000 ± 0.015). That tests the machinery,
  not a physical exponent.
- **05 H0 (criticality), H3, H4, H6.** H0 checks p_c and measures no exponent; H3, H4
  and H6 were designed but never run.
- **`00_synthetic`.** Its checkpoints are closed‑form tests with no measured exponent.

**Reproduce §4.**

```bash
python3 src/study/pilot.py -meta experiments/08_percolation_zd_gpu/recipes/samples_pilot_df_d3_facefar_wide.json \
        --study pilot_df_d3_facefar_wide --replicates 5          # likewise for every samples_pilot_*.json
python3 src/study/plan.py --study pilot_df_d3_facefar_wide \
        --data-root experiments/08_percolation_zd_gpu/data --target-se 1e-4 --replicates 5
```

Recipes: `experiments/{05_percolation_highd,08_percolation_zd_gpu,09_percolation_susceptibility_gpu}/recipes/samples_pilot_*.json`.
Pilot data: the matching `data/pilot_*` directories (gitignored).

