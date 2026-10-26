# Plan — percolation experiments, one week (2026-10-08 → 2026-10-14)

Status: **folders 20–40 created and Day-1 pilots running on niagara (2026-10-08).**

**Execution rule (user, 2026-10-08): one simulation at a time, on one idle host.** Host chosen 2026-10-08 by probing purus, xingu, niagara, amazonas, tocantins, kolyma, tagus, nile, tarim: niagara was idle (load 0.2, GPU 0 %, no compute process). The queue is `local/chain_day1_niagara.sh`, which also waits while another user's job is on the GPU (or the load is ≥ 8 before a CPU pilot). This replaces the "32 CPU cores in the background" assumption below: τ (32–36) now runs in the same serial queue as the GPU work, so its wall clock adds to the GPU schedule instead of overlapping it.

Equation numbers come from PLAN.md's paper-objects table. `../article_writting/article.tex`
is not on this host, so they were not re-checked against the paper.

## 0. Why this plan, and how it is organized

Percolation already has eight experiments (03, 05–10), consolidated in
`experiments/exponents_report.md`. Between them they give first estimates of d_f, τ and γ
in d = 2–5, γ/ν and e₂ in d = 2, each with an eq. (720) bound; 54 of 56 bounds contain the
literature value. Its Table 4b also shows that **matching literature precision is out of
reach**: d_f(3) to ±10⁻⁴ is ≈ 20 days at best, d_f(4) ≈ 3 000 years. So the week goes to:

1. **Testing the article's theorems on a real model** (Part A). Checkpoints 0.3 (CLT) and
   0.5 (error decay + coverage) have only ever been checked on srw/synthetic.
2. **Closing specific gaps in the exponent table** (Part B), with honest bounds — not
   chasing digits.

**One folder per experiment.** Percolation has several independent variables (dimension,
exponent, anchor/observable, the theorem being tested), and mixing them in one folder is
what made `05_percolation_highd/README.md` grow to 876 lines with arms H0–H8. From
experiment 20 on, each folder answers **one question about one exponent in one
dimension**, with its own README (acceptance criteria + measured result), `recipes/`,
`data/` (gitignored) and `images/`. Naming: `NN_perc_<exponent>_d<dim>[_<question>]`.
The single exception is cost (24), where the dimension sweep *is* the question.

Existing pilot recipes for these configurations sit untracked in 05/08/09; each new
experiment gets its own copy in its own `recipes/`, so the old folders are left as they are.

## 1. Experiments

### Part A — theorem checks (≈ 10 GPU-hours)

d = 2 is the test bed: every exponent is exact, and the GPU sampler draws 4·10⁹ sites in
under a second. d = 3 `face_far` repeats the d_f checks where the answer is not exact but
the pilot's bias prediction was consistent (report Table 4a: +0.0004 predicted, +0.0016 ±
0.0062 observed).

| # | Folder | Checks | Setup | Pass criterion |
|---|---|---|---|---|
| 20 | `20_perc_df_d2_clt` | CLT, eq. (583) | d_f, cylinder, fixed (n, m, m₀), **R = 1000** fresh replicates | Var(γ̂) matches 12σ∞²/(nm³ log²ρ) within a bootstrap CI; KS of standardized γ̂ vs N(0,1) passes |
| 21 | `21_perc_df_d3_clt` | same | `face_far`, R = 200 | same |
| 22 | `22_perc_df_d2_error_decay` | error decay eq. (941) under the optimal allocation eqs. (945)–(946); **and** eq. (720) coverage, scored on the same replicates | 6 budgets B = 10⁹ … 10¹² sites (×4 apart), R = 50 per budget, constants from a pilot only | slope of log RMSE vs log B ≈ −ω₁/(d+2ω₁) within its CI; optimal beats a naive allocation at equal B (`allocation_experiment`); coverage ≥ 95% with a Wilson CI on the coverage, plus bound width vs actual error |
| 23 | `23_perc_df_d3_error_decay` | same | `face_far`, same budget grid, R = 30–50 | same |
| 24 | `24_perc_cost_d2_d5` | cost, Assumption 7 (eq. 353) | GPU batched probes d = 2–5; streaming sampler at d = 3, large i | d̂ vs dim; explain or quantify the +3–7 % overshoot (5.33 at dim 5) — cache pressure is the current guess |
| 25 | `25_perc_tau_d2_coverage` | eq. (720) coverage for a second exponent | τ, tail observable, torus, R = 50 at one budget | coverage ≥ 95 % (Wilson CI) |
| 26 | `26_perc_gammanu_d2_coverage` | same, third exponent | γ/ν, origin anchor, R = 50 | same |

22 and 23 also re-test the srw finding of `plans/three_experiment_ladder.md` §4: there
`prop:opt`'s m₀ came out ≈ 3 rungs too high (RMSE 2.2–2.4× the empirical optimum). Record
the empirical-optimum m₀ at each budget next to eq. (946)'s, to see whether percolation
shows the same offset.

Reference values for the README acceptance criteria only (ground rule 4 — never in the
code path): d = 2 has d_f = 91/48, τ = 187/91, γ/ν = 43/24, leading correction ω = 3/2 in
length, so the predicted decay slope is −1.5/(2 + 3) = −0.30. In d = 3, ω ≈ 1.6
(literature) against the pilot's fitted 1.80 ± 0.15.

### Part B — exponent table (≈ 50 GPU-hours + 32 CPU cores in the background)

| # | Folder | Route | Gap it closes |
|---|---|---|---|
| 27 | `27_perc_df_d3_facefar` | GPU, ≈ 8 h, plan.py on the existing `face_far` pilot | best d = 3 bound so far is the axis one, which **misses** the literature |
| 28 | `28_perc_df_d3_axis_deep` | `percolation_zd_stream`, CPU, to i = 1024; two-correction diagnostic | axis vs face_far: the one-correction fit misses a slow drift (Table 4a under-predicts bias up to 360×), and the axis pinning pushes derived β, ν, η, δ off in d = 3 |
| 29 | `29_perc_df_d4_facefar` | GPU, ≈ 8 h | d_f(4) bound ± 0.087 |
| 30 | `30_perc_df_d5_facefar` | GPU, ≈ 8 h | d_f(5) bound ± 0.21 |
| 31 | `31_perc_df_d6_facefar` | GPU; materialized ladder stops at i = 32 (int32 labels) | **upper critical dimension**: log corrections break Assumption 1 (eq. 232). Deliberate failure case — does the pilot refuse to plan (ω₁ → 0) or give a confident wrong answer? |
| 32 | `32_perc_tau_d3` | `percolation_tau_zd`, CPU, ladder to s = 2¹³–2¹⁴, R = 5 | τ(3) bound ± 0.13; pilot ω₁ = 0.34 over-predicts bias |
| 33 | `33_perc_tau_d4` | same | ω₁ unidentified on s ≤ 512 |
| 34 | `34_perc_tau_d5` | same | ω₁ unidentified on s ≤ 512 |
| 35 | `35_perc_tau_d6` | same; **step 0: H0 criticality check at d = 6** | τ = 5/2 exact (with logs); p_c unchecked above dim 5 |
| 36 | `36_perc_tau_d7` | same; **step 0: H0 at d = 7** | τ = 5/2 exact, above d_c: the one exact high-d target in the repo |
| 37 | `37_perc_gamma_d3` | susceptibility, GPU, ≈ 6 h | γ(3) bound ± 0.040 |
| 38 | `38_perc_gamma_d4` | wider-ladder pilot first, then GPU ≈ 6 h | ω₁ unidentified (0.025 ± 1.7) |
| 39 | `39_perc_gamma_d5` | GPU ≈ 6 h | a₁ unidentified, bound ± 0.80 |
| 40 | `40_perc_gammanu_d3` | origin anchor, GPU | only an H7 prototype exists |

Derived exponents (β, ν, η, δ, σ, α through the scaling relations) are not an experiment:
`experiments/exponents_report.md` Table 3 is regenerated from the new pinned values.

**τ stays on CPU.** A GPU port of `percolation_tau_zd` would be ≈ 20× faster (d = 2
precedent, experiment 10) but costs a day of development plus bit-for-bit tests; one
process per core on 32 cores covers the gap with no risk to the sampler.

## 2. Graphics

Each figure belongs to the experiment that produces it; cross-experiment figures go in
the regenerated `exponents_report.md`.

| # | Figure | From |
|---|---|---|
| G1 | Log-log plot with local-slope inset (local slope vs i^(−ω₁), literature marked) | every Part B folder |
| G2 | Effective exponent vs dimension, d = 2…7: d_f, τ, γ approaching mean field; d = 6 drifting | report (27–39) |
| G3 | Cost: seconds per sample vs i for d = 2–5, d̂ vs dim inset | 24 |
| G4 | CLT: histogram + QQ of standardized γ̂, observed/predicted variance ratio | 20, 21 |
| G5 | **Error decay**: RMSE vs B log-log with predicted slope, optimal vs naive, chosen n(B) and m₀(B) beneath — the headline figure | 22, 23 |
| G6 | Coverage per exponent with Wilson CIs; stacked bar of B_fs, B_good, B_bad, statistical term | 22, 23, 25, 26 |
| G7 | Bias–variance U-curve: error vs m₀ at fixed B, eq. (946)'s m₀ marked | 22, 23 |
| G8 | Forest plot: every estimate ± eq. (720) bound vs literature, one panel per exponent | report |
| G9 | Scaling-relation consistency: direct vs derived | report |
| G10 | Anchor plateau: γ̂ vs seed-slab dimension k, d = 3 and 5 | 28 (+ H7 data) |
| G11 | Cost of one more decimal: wall clock vs target precision per dimension, 10^((d+2ω)/ω) law | report |
| G12 | 2-D cluster snapshot (illustration only) | 20 |

Figures are committed one at a time into each experiment's `images/` (ground rule 6).

## 3. Schedule

One RTX 5090 ≈ 160 GPU-hours in the week; the plan uses ≈ 70, leaving ≈ 30 h of buffer
for reruns and the rest idle.

| Day | GPU | CPU (32 cores) | Analysis / writing |
|---|---|---|---|
| Thu 8 | pilots for every configuration (≈ 3 h); 20 overnight | H0 at d = 6, 7 (35, 36 step 0); τ pilots d = 3–7 | run `tools/tests`; create folders 20–40, READMEs with acceptance criteria, recipes |
| Fri 9 | 22 (≈ 3 h), 23 (≈ 3 h), 25, 26 | τ final runs 32–36 (all week); 28 starts | 20 analysis |
| Sat 10 – Sun 11 | 27, 29, 30 (≈ 8 h each); 21 | τ continues; 28 | 22, 23 analysis |
| Mon 12 | 37, 38, 39, 40; 31 | τ continues | reruns where a pilot failed to identify ω₁ |
| Tue 13 | buffer | τ finishes | figures G1–G12 |
| Wed 14 | — | — | READMEs, regenerate `exponents_report.md`, commit figures |

## 4. Risks

- **ω₁ may stay unidentified** for τ(4), τ(5) and γ(4) even on wider ladders. Ground rule
  4 forbids feeding in the literature ω. Then the result is "the method correctly refuses
  to plan", reported with the statistical term only — a finding for the paper, not a
  failure.
- **p_c's precision becomes the bias at large L** (δp · L^(1/ν)), which caps how deep the
  d ≥ 3 ladders are worth pushing.
- **The streaming sampler is 0.65× the materialized speed** (`plans/streaming_percolation.md`
  Q4), so 28 is CPU-hours-bound; cut its top rung to 512 if it falls behind.
- **Bethe lattice (04) is out of scope**: its article appendix is an empty stub, and ground
  rule 3 says stop and ask rather than guess the closed form.

## 5. Decisions

| decision | proposed | status |
|---|---|---|
| D1 — one folder per (question, exponent, dimension) from 20 on | yes (user, 2026-10-08) | **decided** |
| D2 — include d = 6 as a deliberate Assumption 1 failure | yes | open |
| D3 — τ on CPU, no GPU port of `percolation_tau_zd` | yes — but with τ now serial, the port's ≈ 20× may pay for itself; revisit after the τ pilots' wall clock is known | open |
| D5 — one simulation at a time, one host | yes (user, 2026-10-08) | **decided** |
| D4 — Bethe lattice deferred | yes | open |

## 6. Progress (updated 2026-10-13)

Runs: `local/dispatch.sh` (serial, idle-host check before each job), queue `local/queue.txt`,
log `local/queue.done`. Pilot triage: `local/triage.py`. Scoring: `calibration/score_theorems.py`.

| exp | status | headline |
|---|---|---|
| 20 | done | CLT holds at d = 2 (Var ratio 1.04, R = 1000); eq. (720) 1000/1000, 5× wide |
| 21 | done | γ̂ normal; Var 1.27× the asymptotic eq. (583), 1.04× the exact finite-m form (Assumption 6 not converged); eq. (720) 200/200, 9.5× wide |
| 22 | done | RATE −0.185 vs −0.192 ✓; prop:opt m₀ = 10–12 unaffordable; tuned within 1.10× of best; eq. (720) 30/30 cells |
| 23 | done | RATE −0.304 vs −0.267 ✓; **prop:opt costs 5.6–22× RMSE**, tuned = best; eq. (720) 479/480 |
| 25 | done | at `box_exponent` 1/2 eq. (720) covered 50 % (non-decaying box bias +0.0096); **at 48/91 (user, 2026-10-13) it covers 100/100**, bias +0.001–0.002, Assumption 6 holds |
| 26 | done | eq. (720) 50/50; Var 1.6× because Assumption 6 fails for the origin anchor (cv 1.27 → 1.66) |
| 27 | done | **d̂_f(3) = 2.5222 ± 0.0133** (eq. 720; lit. 2.52293 inside, actual error 0.0007); 3× tighter than before; wall clock 1.64× the plan |
| 28 | pilot done, not run | d off by 7.9 % |
| 29 | done | d̂_f(4) = 3.0939 ± 0.0904, lit. inside; bias +0.049 = predicted 0.048; box cap 128⁴ makes it bias-limited |
| 30 | done | d̂_f(5) = 3.7851 ± 0.2677, lit. inside by 0.009; bias 1.65× the pilot's prediction |
| 31 | pilot done | d = 6: pilot is confidently wrong (d_f 4.36, ω₁ 1.665 ± 0.015) — the deliberate Assumption 1 failure |
| 32–36 | pilots done | all RED: clock d off by 7–19 %; ω₁ 0.49 (d=3), 0.23 (d=4), unidentified at d=5, fit not converged at d=6, 7. Pilot τ̂ 2.196 / 2.352 / 2.595 / 2.783 / 2.445 |
| 37, 38, 40 | pilots done, not run | ω₁ unidentified / fit not converged |
| 39 | pilot done, not run | a₁ unidentified |
| 24 | done (from pilot probes) | cost = i^dim holds mid-ladder; the +1–7 % in d̂ is the top rung only (per-site cost ≈ 2× once a sample exceeds ~10⁸ sites), which explains 27's 1.64× overrun |
