# Experiment 27 — $\hat d_f$, $d=3$, `face_far`, planned run

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-09). d̂_f(3) = 2.5222 ± 0.0133 (eq. 720), literature 2.52293(10) inside. Three-term bound (2026-11-04): ± 0.0043, still inside.**

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

## Wilson interval in the article's three-term form (2026-11-04)

Theorem cor:wilson (article revision of 2026-11-04) dropped $B_{bad}$ together with the
log-moment assumption: $W = 1.96\sqrt{12\sigma_\infty^2/(n\,m(m^2-1))} + B_{fs} + 6\sigma^2_{\max}/(n(m+1))$.
Re-reported from the stored data; nothing was redrawn.

**[2.5179, 2.5265]**, half-width **0.0043** ($1.96\sigma_{se}$ 0.0036, $B_{fs}$ 0.0007,
$B_{good}$ 4e-5). It contains 2.52293. The pinned refit does not warn (ω₁ = 1.85, χ² p = 0.78).
Gap (i) above is gone: the bound is now 3.6× the predicted RMSE (0.0012), not 11×, and
the standard error dominates it. $B_{fs}$ = 0.0007 rather than 0.0009, because the refit
with γ pinned has been used since 2026-11-02.

## Wilson interval with the AIC pilot refit (2026-11-07)

B_fs now takes $\omega_1, a_1$ from the pilot refitted with $\gamma$ pinned to $\hat\gamma$ on a range of
$\ge 5$ consecutive pilot scales chosen by AIC with data cuts (Jay & Neil 2021;
`local_slope.omega_ranges`, `report.pinned_refit`), not on every pilot scale as in the
2026-11-02 refit. $\hat\gamma$ is unchanged; only the eq. (720) half-width moves. The
$\omega_1$ error is the profile error $\oplus$ the spread over ranges. Old = all-scales pinned
refit (kept in `answer.json` as `pilot_refit_all_scales`), new = AIC refit (`pilot_refit`);
reports rerun 2026-11-07. A warning is $\chi^2$ $p < 0.01$ (old: or $\omega_1$ on its bound).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old $\chi^2$/dof | old radius | new $\hat\omega_1$ | range | new $\chi^2$/dof | new radius |
|---|---|---|---|---|---|---|---|---|
| `pilot` | 2.522201 | 1.85 ± 0.071 | 2.5/5 | 0.00431 | 1.85 ± 0.2 | 4..512 | 2.5/5 | 0.00431 |

## Wilson interval with the AIC-weighted refit (2026-11-10)

$\omega_1$ is now the AIC-weighted average over the kept ranges (Jay & Neil 2021, eq. 15),
not the single highest-weight range; its error is their eq. (18). $a_1$ is refitted on the
highest-weight range with $\omega_1$ fixed at that average, so B_fs takes one coherent
$(a_1, \omega_1)$ (averaging $a_1$ separately pairs values from different fits). $\hat\gamma$
is unchanged; only B_fs and the eq. (720) half-width move.
Old = the 2026-11-07 highest-weight range, new = the average; reports rerun 2026-11-10.
"best" is the highest-weight range and its share of the kept weight; its $\chi^2$ still
gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `pilot` | 2.522201 | 1.85 | 0.00431 | 1.99 ± 0.348 | 4..512 (0.40) | -1.75 | 0.000479 | 0.0041 |
