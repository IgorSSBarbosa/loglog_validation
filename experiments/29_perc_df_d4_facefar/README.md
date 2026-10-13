# Experiment 29 — $\hat d_f$, $d=4$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-13). d̂_f(4) = 3.0939 ± 0.0904 (eq. 720), literature 3.0446(7) inside; bias-limited by the int32 box cap.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.087$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=3.0446(7)$. Wikipedia also lists 3.12(2), 3.05(5), 3.003, 3.0472(14), 3.046(7), 3.046(5), 3.0479, 3.0437(11).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 0.94(15), 1.2198, 1.13(10), 1.0(2). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.31(5), 0.5(1), 0.37(4), 0.4008 give $\omega\approx0.94(15)$, 1.5(3), 1.13(12), 1.220.

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/29_perc_df_d4_facefar/recipes/samples_pilot_perc_df_d4_facefar.json \
    --study pilot --replicates 5 --data-root experiments/29_perc_df_d4_facefar/data
```

Run 2026-10-08 on niagara, 222 s drawn, ladder 4..128. d = 4.223 against the declared 4.000 (+5.6%); ω₁ = 1.26 ± 0.096; a₁ = -2.01 ± 0.14; eq. (232) fit converged; pilot γ̂ = 3.0475. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +5.6%).

## Step 2 — plan and run (2026-10-13, user approved running despite the d mismatch)

```bash
python3 src/study/plan.py   --study pilot --data-root experiments/29_perc_df_d4_facefar/data --time 45m --replicates 5 --max-scale 128 --m 5 --accept
python3 src/study/run.py    --study pilot --data-root experiments/29_perc_df_d4_facefar/data
python3 src/study/report.py --study pilot --data-root experiments/29_perc_df_d4_facefar/data
```

**Why 45 min and m = 5.** `--max-scale 128` (128⁴ = 2.7·10⁸ sites; 256⁴ exceeds the int32 label
space) caps the ladder. With the default m = 6 the ladder must start at i = 4 and the plan
predicted ±0.089, all bias; m = 5 starts at 8 and predicts ±0.048, still all bias (sd 0.0016 at
4 h). More budget buys samples, not depth, so 4 h was cut to 45 min.

Plan: m₀ = 2, scales 8..128, n = 10 260 × R = 5. Ran 58.5 min (1.30× predicted).

| | value |
|---|---|
| **d̂_f** | **3.0939** |
| eq. (720) | **± 0.0904** → [3.0035, 3.1843]: B_fs 0.072, B_bad 0.012, 1.96σ_se 0.006 |
| t(4) interval | [3.0898, 3.0980] (scatter only; **excludes** the literature value) |
| γ̂ − 3.0446 | **+0.049**, against the plan's predicted bias 0.048 |

**Reading.** The plan's bias prediction is accurate (0.049 vs 0.048) and the bound contains the
literature value; the replicate interval does not, which is exactly the case eq. (720) exists
for. The bound is no tighter than September's (± 0.087): at this box cap d_f(4) is limited by
finite-size bias, and only `models/percolation_zd_stream.py` (larger boxes) can move it.
