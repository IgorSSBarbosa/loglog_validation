# Percolation experiments 20–40: summary (2026-10-08 → 2026-10-13)

Plan: `plans/percolation_experiments.md`. Details, commands and seeds: each `experiments/NN_*/README.md`.
Every run was serial on niagara (RTX 5090). Literature values appear here only as acceptance
criteria (ground rule 4). Equation numbers come from PLAN.md's table and were **not
re-checked against `article.tex`**, which is not on this host.

**Status.** 15 experiments were done end to end (20–27, 29, 30) or by analysis (24). The other
11 (28, 31–40) only got a pilot, and in 10 of them triage refused to plan.

---

## A. Focus for the article

These test the theorems directly, on a model the theory wasn't tuned for. Each passed a
stated numeric check.

### A1. Error decay and the optimal allocation: 22 (d = 2), 23 (d = 3) ★ headline

*Checks: eq. (941) rate, eqs. (945)–(946) allocation, eq. (720) coverage.*

| | d = 2 (22) | d = 3, `face_far` (23) |
|---|---|---|
| budgets | 10⁹ … 10¹², 6 × R = 20 per m₀ | 10¹⁰ … 10¹², 6 × R = 20 per m₀ |
| predicted rate −ω₁/(d+2ω₁) | −0.192 | −0.267 |
| measured rate, best m₀ | −0.185 [−0.236, −0.149] ✓ | −0.304 [−0.378, −0.231] ✓ |
| eq. (946) m₀ as stated (prop:opt) | 10–12: **unaffordable** at every B | affordable, but **5.6–22× the best RMSE** |
| tuned m₀ (constant restored from a₁, cv) | ≤ 1.10× the best | = best at all 6 budgets |
| eq. (720) coverage | 600/600 (30 cells) | 479/480 |

- The rate theorem holds in both dimensions.
- **Main finding:** eq. (946) gets the scaling of m₀ right but drops a constant that matters
  in practice. On srw it was off by about 3 rungs (2.2–2.4× RMSE). On percolation the cost
  is 5.6–22×. Restoring the constant from a₁ and cv recovers the empirical optimum. This
  is worth a remark or corollary in the paper.
- The single miss in 23 (B = 10¹⁰, n = 3 per scale) is a top-rung mean of exactly 0, so γ̂ = −∞.
  That is the article's "bad event" (Assumption 2), which is what the B_bad term covers.
- Figure: `23_.../data/allocation_arms_d2_d3.png` (RMSE vs B, prop:opt vs tuned vs best, d = 2
  and d = 3). Per-experiment panels: `22_.../data/error_decay/theorem_scores.png` and `23_.../data/error_decay/theorem_scores.png`.

### A2. CLT, eq. (583): 20 (d = 2), 21 (d = 3)

| | d = 2 (20), R = 1000 | d = 3 (21), R = 200 |
|---|---|---|
| Var ratio, asymptotic eq. (583) | **1.036 [0.946, 1.128]** ✓ | 1.27 [1.05, 1.47] ✗ |
| Var ratio, exact finite-m form | 0.971 | **1.04** ✓ |
| KS vs N(0,1) | p = 0.976 | p = 0.14 |
| eq. (720) coverage, bound / RMSE | 1000/1000, 5.3× | 200/200, 9.5× |

- γ̂ is Gaussian in both dimensions.
- At d = 3 the asymptotic variance underestimates by 27% because Assumption 6 hasn't converged
  on 8..256 (cv goes from 1.17 down to 0.83). The finite-m formula is exact. **For the paper:**
  either state the finite-m variance, or say that the asymptotic form needs cv to have flattened.
- Figures: `20_.../data/clt/theorem_scores.png` and `21_.../data/clt/theorem_scores.png` (QQ plot plus variance ratio).

### A3. Coverage of eq. (720) for three exponents, including one failure that was diagnosed: 25, 26 (+ 20–23)

| exp | exponent | coverage | bound / RMSE | note |
|---|---|---|---|---|
| 20–23 | d_f | 2279/2280 | 2.7–27 | the miss is the bad event above |
| 25, `box_exponent` = 1/2 | τ (d = 2) | **25/50 ✗** | 0.99 | bias +0.0099 that doesn't decay |
| 25, `box_exponent` = 48/91 | τ (d = 2) | **100/100** ✓ | 5.7–6.7 | bias +0.001–0.002 |
| 26 | γ/ν (d = 2) | 50/50 ✓ | 11.7 | Var ratio 1.6×: Assumption 6 fails |

- **The 25 story belongs in the paper as a cautionary example.** With the box rule s^(1/2), the
  finite-box bias drifts as s^(5/96). That doesn't decay like the corrections eq. (232) models,
  so the pilot can't see it and the bound under-covers. Tying the box to the cluster size
  (1/d_f) makes the bias an ordinary correction again, and coverage goes to 100%. So the
  bound is only as good as Assumption 1 is for the observable.
- 26 shows the other assumption failing. The origin anchor's cv keeps rising (1.27 → 1.66), so
  σ∞² doesn't exist. The bound still covers, but the CLT variance is 1.6× the prediction.
- Across all of these the bound is conservative (3–27× the actual error), and B_bad
  dominates whenever m₀ is high.

### A4. A real exponent with an honest bound: 27 (d_f, d = 3), 29 (d_f, d = 4)

| | estimate ± eq. (720) | literature | actual error | t(4) interval |
|---|---|---|---|---|
| **d_f(3)** (27) | **2.5222 ± 0.0133** | 2.52293(10) | −0.0007 | contains lit. |
| **d_f(4)** (29) | 3.0939 ± 0.0904 | 3.0446(7) | +0.049 (plan predicted 0.048) | **excludes** lit. |

- 27 is the full pipeline (pilot → plan → run → report) on a non-exact model: 3× tighter than the
  previous best (±0.040), and the actual error matches the plan's prediction (0.0012).
- 29 is the clearest argument for eq. (720). The replicate interval is confidently wrong. The
  bound contains the truth, and its bias term was predicted to within 2%.
- Runtime caveat: 27 took 1.64× its planned time (see 24).
- Figures: `27_.../data/pilot/plot.png` and `fit.png`, plus the same two files for 29.

### A5. Where Assumption 1 fails and the method gives a confident wrong answer: 31 (d = 6)

- At the upper critical dimension the log corrections break eq. (232). The pilot **does not
  refuse to plan**: it gives ω₁ = 1.67 ± 0.015 (very tight) and d̂_f = 4.36, against the
  mean-field value 4. Triage was GREEN.
- This is a short but important limitation paragraph: the pipeline can't detect log corrections.
- Only the pilot exists. A planned run would just make the wrong answer more precise.

---

## B. Supporting material (appendix or a sentence each)

| exp | what | result | use |
|---|---|---|---|
| 24 | cost exponent, Assumption 7, d = 2–6 | cost = i^dim holds mid-ladder. All of the +1–7% in d̂ comes from the top rung, where per-site cost roughly doubles once a sample passes ~10⁸ sites | explains plan overruns (27: 1.64×, 29: 1.30×). Figure `24_.../data/cost_from_pilots.png` |
| 30 | d_f, d = 5 | 3.7851 ± 0.2677, lit. 3.5260 inside by 0.009. Bias +0.259 vs 0.157 predicted | the one-correction fit under-predicts bias on short ladders (i = 4..64). Worse than September's 3.6555 ± 0.2121 |
| 28 | d_f, d = 3, axis anchor | pilot only, d off by +7.9% (RED) | axis vs `face_far` drift, unresolved |

---

## C. Not for the article as results (pilots only, refused to plan)

Triage: converged fit, ω₁ se < ω₁/2, a₁ se < |a₁|, measured cost exponent within 5% of declared.

| exp | target | pilot estimate | literature | why it stopped |
|---|---|---|---|---|
| 32 | τ, d = 3 | 2.196 | 2.18909 | cost d off −7.0% |
| 33 | τ, d = 4 | 2.352 | 2.3142 | cost d off +6.9%; ω₁ = 0.23 |
| 34 | τ, d = 5 | 2.595 | 2.4175 | ω₁ unidentified; cost d off −15% |
| 35 | τ, d = 6 | 2.783 | 5/2 (+ logs) | fit not converged; p_c unchecked |
| 36 | τ, d = 7 | 2.445 | 5/2 | fit not converged; p_c unchecked |
| 37 | γ, d = 3 | 1.7986 | 1.793(3) / 1.819(3) | ω₁ unidentified (0.80 ± 2.3) |
| 38 | γ, d = 4 | 1.4278 | 1.430(6) | ω₁ unidentified (2.45 ± 1.6) |
| 39 | γ, d = 5 | 1.2050 | 1.1792(7) | a₁ unidentified (YELLOW) |
| 40 | γ/ν, d = 3 | 2.0097 | 2.0431(14) | fit not converged |

At most, these support one sentence in the paper: "when ω₁ is not identifiable on the
affordable ladder, the method refuses to plan rather than produce a bound." 31 (section A5) is
the counterexample where it doesn't refuse.

---

## Suggested order of the percolation section

1. A1 error decay + allocation (headline figure).
2. A2 CLT.
3. A3 coverage, including the τ box-rule diagnosis.
4. A4 d_f(3), d_f(4) as the end-to-end demonstration.
5. A5 d = 6 limitation, plus the one-line refusal summary from section C.
6. Appendix: 24 cost, 30 d_f(5).
