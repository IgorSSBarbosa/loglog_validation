# Experiment 29 — $\hat d_f$, $d=4$, `face_far`

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done (2026-10-13). d̂_f(4) = 3.0939 ± 0.0904 (eq. 720), literature 3.0446(7) inside; bias-limited by the int32 box cap. Since 2026-11-02 (pinned refit) the bound excludes the literature value: three-term ± 0.0316, see the end.**

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

## Wilson interval in the article's three-term form (2026-11-04)

Theorem cor:wilson (article revision of 2026-11-04) dropped $B_{bad}$ together with the
log-moment assumption: $W = 1.96\sqrt{12\sigma_\infty^2/(n\,m(m^2-1))} + B_{fs} + 6\sigma^2_{\max}/(n(m+1))$.
Re-reported from the stored data; nothing was redrawn.

**[3.0623, 3.1255]**, half-width **0.0316** ($B_{fs}$ 0.0255, $1.96\sigma_{se}$ 0.0059,
$B_{good}$ 0.0001). It **excludes** 3.0446: the lower end is 0.018 above it.

The flip is not today's change. The four-term bound of 2026-11-02 was already
[3.0498, 3.1380]: since then, $B_{fs}$ has been built from the pilot refitted with γ pinned
to γ̂ (0.0255), not from the plan's constants (0.072 in the table above). That refit returns
ω₁ = 1.88 against the pilot fit's 1.26, and it **warns** (χ² p = 3e-8). The realized bias,
+0.049, is about twice the new $B_{fs}$. The acceptance criterion (narrower than ±0.087) is
met only formally, since a warned $B_{fs}$ is not a usable bound. Reading (unchanged): this
is bias-limited by the box cap.

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
| `pilot` | 3.093896 | 1.88 ± 0.062 | 37.9/3 ⚠ | 0.0316 | 1.87 ± 0.06 | 4..64 | 23.8/2 ⚠ | 0.0321 |

## AIC-quality check on all existing data (user, 2026-11-07)

Every dataset with these exact params was pooled (duplicates dropped; the sets agree at shared scales, chi2 p = 0.63) and fitted by `tools/local_slope.omega_ranges` (CLT se). Pooled 4..128 (6 rungs): one range only, chi2 3.1/2, omega 1.28 +- 0.04, d_f 3.0479 +- 0.0034; AIC had no range to choose.

### Pilot `aic_sqrt2` (launched 2026-11-07 on xingu (RTX 5090), seed 2026110829)

Same box cap, rungs round(4 sqrt(2)^k), 11 points, so AIC has ranges to choose among. Not geometric: `pilot.py` only, no autopilot.

```bash
.venv/bin/python src/study/pilot.py -meta experiments/29_perc_df_d4_facefar/recipes/samples_df_d4_sqrt2.json \
    --study aic_sqrt2 --data-root experiments/29_perc_df_d4_facefar/data --replicates 5 --seed 2026110829
```

Result (2026-11-07, 5 x 1281 s on xingu): **the odd rungs are off the curve.** Local slopes 3.40, 3.26, 3.66, 2.74, 3.32, 2.86, 3.20, 2.95, 3.12, 2.98: every odd i (11, 23, 45, 91) sits above the trend by tens of se, and AIC rejects every range (chi2 > 1000). Not the GPU: the CPU `percolation_zd` agrees at i = 8, 11, 16, 23 (|z| <= 1.1). Cause: `face_far` counts x_0 >= i//2 (models/percolation_zd.py:676), exactly the far half for even i, one layer more and half a layer nearer the anchor for odd i -- an O(1/i) parity offset of the observable's definition, the kind of artifact the torus's ceil(L) is suspected of in 38. The earlier ladders were all powers of 2, so never saw it. Even rungs only (4, 6, 8, 16, 32, 64, 128): AIC picks 6..128, omega = 1.29 +- 0.04, d_f = 3.0522 +- 0.0024 +- 0.0011 (chi2 4.4/2); pooled with 29 + 08, omega = 1.25 +- 0.03, d_f = 3.0492 +- 0.0021 (chi2 6.1/2), 2.1 sigma above 3.0446(7).

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
| `pilot` | 3.093896 | 1.87 | 0.0321 | 1.87 ± 0.0595 | 4..64 (1.00) ⚠ | -3.17 | 0.026 | 0.0321 |
