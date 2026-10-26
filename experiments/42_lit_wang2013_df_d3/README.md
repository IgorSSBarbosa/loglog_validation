# Experiment 42 — Wang et al. 2013's allocation for $d_f$, $d=3$, with the log-log estimator

Part of `plans/literature_fss_protocols.md` (arm B). **Status: dropped (user,
2026-10-26). The literal allocation costs ~260 GPU-hours per replicate and is not run;
the recipe is kept as the record of the published protocol.**

## Question

Wang, Zhou, Zhang, Garoni & Deng, *PRE* 87, 052107 (2013), arXiv:1302.0421, estimate
$y_h=d_f=2.52295(15)$ for three-dimensional percolation. Their allocation (Sec. II):
*"linear system sizes L = 8, 12, 16, 24, 32, 48, 64, 128, 256, and 512. For each system
size, we produced at least 2.5 × 10⁷ independent samples."* Equal $n$ at every size.

Arm B draws exactly that allocation on this repo's sampler and observable and estimates
$d_f$ with the article's log-log regression (user, 2026-10-26), not with the paper's
nonlinear ansatz $L^{y_h}(a_0+b_1L^{-1.2}+b_2L^{-2})$. Each replicate reports:

- `full`: the OLS slope over all 10 sizes;
- `window`: the OLS slope over $L\ge L_{\min}$, with $L_{\min}$ chosen by the paper's own
  $\chi^2$ rule made numeric (`tools/lmin_rule.py`);

each with the finite-m standard error $\sqrt{\sum_jv_j^2\text{cv}_j^2/n_j}$ and the
interval $\pm1.96$ se. Arm A (`autopilot.py`) is then run at the same budget, and the two
are compared on RMSE and on the coverage of their own intervals.

Differences from the paper, by design (D1 of the plan): site percolation with the
`face_far` count on a cylinder (`percolation_zd_gpu`), not the largest cluster $C_1$ on a
periodic box. The exponent is universal; the corrections to scaling are not.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=2.52293(10)$ (Xu, Wang, Lv, Deng 2014; see `experiments/27_perc_df_d3_facefar/README.md`).

## Cost, and the decision it needs

```text
budget per replicate  3.84e15 cost units (sum_j n_j L_j^3)
at 4.1e9 sites/s      ~260 GPU-hours per replicate
top rung (L = 512)    87% of the budget
```

That is 26× experiment 27's planned run. Options, for the user:

1. **Scale every $n_j$ by one factor** (keeps the paper's equal-$n$ shape and size list),
   e.g. to experiment 27's $1.5\cdot10^{14}$, so arm A can reuse that budget.
2. **Truncate the list at $L=256$**, which removes 87% of the cost but changes the shape.
3. Run the literal recipe once (~11 days of GPU).

## Commands (once decided)

```bash
python3 src/budget/protocol_arm.py -meta experiments/42_lit_wang2013_df_d3/recipes/protocol_wang2013_literal.json \
    --tag wang2013_literal --replicates 1
```
