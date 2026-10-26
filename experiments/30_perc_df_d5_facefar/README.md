# Experiment 30 — $\hat d_f$, $d=5$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-13). d̂_f(5) = 3.7851 ± 0.2677 (eq. 720), literature 3.5260(14) just inside; the pilot under-predicted the bias.**

## Acceptance criterion

eq. (720) bound narrower than $\pm0.21$.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=3.5260(14)$. Wikipedia also lists 3.54(4), 3.69(2), 3.528, 3.524(2).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 0.96(26), 0.7178. Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.27(7), 0.2034, 0.210(2) give $\omega\approx0.95(25)$, 0.717, 0.740(7).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/30_perc_df_d5_facefar/recipes/samples_pilot_perc_df_d5_facefar.json \
    --study pilot --replicates 5 --data-root experiments/30_perc_df_d5_facefar/data
```

Run 2026-10-08 on niagara, 247 s drawn, ladder 4..64. d = 5.329 against the declared 5.000 (+6.6%); ω₁ = 1.35 ± 0.16; a₁ = -3.08 ± 0.29; eq. (232) fit converged; pilot γ̂ = 3.6293. Triage (`local/triage.py`: converged, ω₁ se < ω₁/2, a₁ se < |a₁|, d within 5 %): **RED** (d off by +6.6%).

## Step 2 — plan and run (2026-10-13, user approved running despite the d mismatch)

```bash
python3 src/study/plan.py   --study pilot --data-root experiments/30_perc_df_d5_facefar/data --time 45m --replicates 5 --max-scale 64 --m 5 --accept
python3 src/study/run.py    --study pilot --data-root experiments/30_perc_df_d5_facefar/data
python3 src/study/report.py --study pilot --data-root experiments/30_perc_df_d5_facefar/data
```

Same constraint as 29: `--max-scale 64` (64⁵ = 1.1·10⁹ sites) and m = 5 force m₀ = 1, scales
4..64; the plan predicted ±0.157, all bias, so 45 min. n = 2 181 × R = 5; ran 51.9 min (1.15×).

| | value |
|---|---|
| **d̂_f** | **3.7851** |
| eq. (720) | **± 0.2677** → [3.5174, 4.0528]: B_fs 0.226, B_bad 0.032, 1.96σ_se 0.007 |
| t(4) interval | [3.7802, 3.7900] (scatter only; excludes the literature value) |
| γ̂ − 3.5260 | **+0.259**, against the plan's predicted bias 0.157 |

**Reading.** The bound contains the literature value, by 0.009. The pilot's one-correction fit
under-predicts the bias by 1.65×, as `exponents_report.md` Table 4a already found for d = 4, 5:
on a ladder starting at i = 4 the fit sees fast curvature and misses a slower drift. This
estimate is worse than September's (3.6555 ± 0.2121 on 16..64): starting at 4 costs more bias
than the extra rung saves. At d = 5 the materialized model cannot reach the scales d_f needs.
