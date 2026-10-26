# Experiment 43 — Zhang et al. 2020's allocations for $d_f$, $d=4$ and $5$, with the log-log estimator

Part of `plans/literature_fss_protocols.md` (arm B, and D3, the reproduction check).
**Status: recipes written (2026-10-26); not run.**

## Question

Zhang, Hou, Fang, Hu & Deng, arXiv:2004.11289 (2020), are the source of this repo's
acceptance values $d_f(4)=3.0446(7)$ and $d_f(5)=3.5260(14)$. Their allocation (Sec. II):

- 4D: *"L = 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64, 96, 128, 176, 192, and the
  numbers of independent samples are more than 10⁶ for each L ≤ 64 and at least 10⁵ for
  each L > 64"*;
- 5D: *"more than 10⁶ samples are taken for each of L = 4, 6, 8, 10, 12, 14, 16, 20, and
  at least 10⁵ samples for each of L = 24, 32, 40, 48, 60"*.

The recipes use the stated minima, $10^6$ and $10^5$. Arm B draws these allocations on
this repo's sampler and estimates $d_f$ with the article's log-log regression, exactly
as in experiment 42 (`full` and `window` slopes, finite-m standard error, $\pm1.96$ se).

Differences from the paper, by design: site percolation with `face_far` on a cylinder,
not bond percolation with the largest cluster on a periodic box. **The 4D list is
truncated at $L=128$**: $176^4$ and $192^4$ exceed the int32 label space of
`percolation_zd_gpu`, and those two sizes carry about 80% of the published 4D budget.
So only the 5D recipe is the published allocation.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f(4)=3.0446(7)$, $d_f(5)=3.5260(14)$ (this paper; bond percolation, same universality class).

## Cost

| recipe | sizes | cost units / replicate | GPU time / replicate (4.1·10⁹ sites/s) | top rung share |
|---|---|---|---|---|
| `protocol_zhang2020_d5.json` | 13, $L=4..60$ | $1.23\cdot10^{14}$ | ≈ 8.3 h | 63% |
| `protocol_zhang2020_d4_truncated.json` | 15, $L=4..128$ | $6.2\cdot10^{13}$ | ≈ 4.2 h | 44% |

## Step 1 — D3, reproduction check (user, 2026-10-26: yes)

One replicate of the 5D recipe. It checks that the published allocation, on our
sampler, gives an estimate consistent with 3.5260 before it is used as a baseline.

```bash
python3 src/budget/protocol_arm.py -meta experiments/43_lit_zhang2020_df_d4_d5/recipes/protocol_zhang2020_d5.json \
    --tag zhang2020_d5 --replicates 1
```

## Step 2 — arm B against arm A

$R=5$ replicates of each recipe, then `autopilot.py` with `--time` equal to arm B's
measured wall clock per replicate, scored on the same truth.
