# Experiment 27 — $\hat d_f$, $d=3$, `face_far`, planned run

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-09). d̂_f(3) = 2.5222 ± 0.0133 (eq. 720), literature 2.52293(10) inside.**

## Acceptance criterion

eq. (720) bound narrower than the 2026-09-17 pilot's $\pm0.040$; whether it contains the literature value is reported, not tuned for.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $d_f=2.52293(10)$. Wikipedia also lists 2.523(4), 2.530(4) (both derived), 2.5230(1), 2.5226(1); 2.5226(1) is about 2.3 combined errors below 2.52293(10).
- Correction (length ladder): Wikipedia's $\omega$ cell lists 1.26(23), 1.6334, 1.62(13), 1.61(5). Via $\omega=\Omega\,d_f$ its $\Omega$ values 0.64(2), 0.65(2), 0.77(3) give $\omega\approx1.61(5)$, 1.64(5), 1.94(8); the other $\Omega$ entries are 0.50(9), 0.73(8), 0.60(8), 0.64(5).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/27_perc_df_d3_facefar/recipes/samples_pilot_perc_df_d3_facefar.json \
    --study pilot --replicates 5 --data-root experiments/27_perc_df_d3_facefar/data
```

Run 2026-10-08, 192 s drawn: d = 3.136 (+4.5 %), ω₁ = 1.757 ± 0.045, a₁ = −1.36 ± 0.07, cv = 0.91. Ready to plan.

## Step 2 — plan and run (2026-10-09)

```bash
python3 src/study/plan.py   --study pilot --data-root experiments/27_perc_df_d3_facefar/data --time 6h --replicates 5 --max-scale 1024 --accept
python3 src/study/run.py    --study pilot --data-root experiments/27_perc_df_d3_facefar/data
python3 src/study/report.py --study pilot --data-root experiments/27_perc_df_d3_facefar/data
```

Plan: m₀ = 4, scales 32..1024, n = 24 399 per scale × R = 5, predicted 360 min and error
±0.0012 on the answer (bias 0.00075, sd 0.0009). `--max-scale 1024` keeps one sample under
the int32 label space. Ran on niagara 04:35–14:23.

| | value |
|---|---|
| **d̂_f** | **2.5222** |
| eq. (720) half-width | **± 0.0133** → [2.5089, 2.5355]; B_bad 0.0088, 1.96σ_se 0.0035, B_fs 0.0009, B_good 0.0001 (ω₂ piece omitted) |
| t(4) replicate interval | [2.5194, 2.5250] (scatter only) |
| γ̂ − 2.52293 | **−0.0007**, inside both intervals |
| cross-checks | all-points OLS 2.52220; eq. (232) fit 2.52354 |
| replicate sd | 0.00223 measured vs 0.0018 closed form (5 replicates: ±35 %) |
| wall clock | **588.7 min vs 360 predicted (1.64×)** |

**Reading.** The previous best `face_far` bound was ±0.040 (`exponents_report.md`, pilot
`pilot_df_d3_facefar_wide`); this run is 3× tighter and still contains the literature value,
and the actual error (0.0007) is what the plan predicted (0.0012). Two gaps against the plan:
(i) the eq. (720) bound is 11× the predicted RMSE, almost all B_bad, the stray-mean term the plan
does not model; (ii) the run took 1.64× the predicted time: the throughput was measured on
boxes ≤ 512³ and per-site cost grows with the box (the d̂ = 3.14 the pilot measured already says
so), so the clock has to extrapolate to 1024³.

Report: `data/pilot/report.md` (`plot.png`, `fit.png`).
