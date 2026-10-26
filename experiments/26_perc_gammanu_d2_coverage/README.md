# Experiment 26 — eq. (720) coverage for $\widehat{\gamma/\nu}$, $d=2$, origin anchor

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: pilot and T3 run (2026-10-08). Bound covers; CLT variance 1.6× the prediction.**

## Acceptance criterion

Coverage $\ge95\%$ (Wilson CI) over $R=50$ replicates at one planned budget.

## Known values — acceptance criteria, not inputs (ground rule 4)

- $\gamma/\nu=2-\eta=43/24$, from $\eta=5/24$ (exact; Wikipedia also gives $\gamma=43/18$, $\nu=4/3$).
- Correction (length ladder): $\omega=3/2$ (exact; $=\Omega\,d_f$ with $\Omega=72/91$).

Source: Wikipedia, *Percolation critical exponents*, read 2026-10-08.

## Step 1 — pilot

```bash
python3 src/study/pilot.py -meta experiments/26_perc_gammanu_d2_coverage/recipes/samples_pilot_perc_gammanu_d2_coverage.json \
    --study pilot --replicates 5 --data-root experiments/26_perc_gammanu_d2_coverage/data
```

Run 2026-10-08, 4 s drawn: d = 2.041 (+2.1 %), ω₁ = 0.687 ± 0.060, a₁ = +0.828 ± 0.023, cv = 1.20.

## Step 2 — T3, eq. (720) coverage (2026-10-08)

```bash
python3 src/budget/allocation_experiment.py -meta experiments/26_perc_gammanu_d2_coverage/recipes/sweep_coverage.json \
    -o experiments/26_perc_gammanu_d2_coverage/data --tag coverage
python3 calibration/score_theorems.py --sweep experiments/26_perc_gammanu_d2_coverage/data/coverage \
    --pilot experiments/26_perc_gammanu_d2_coverage/data/pilot
```

B = 2·10¹¹, tuned m₀ = 6 (ladder 128..4096), n = 6 409, R = 50, seed 2026100926, 14 min.

| quantity | value |
|---|---|
| mean γ̂ − 43/24 | −0.0031 (≈ 0.4 sd) |
| Var ratio, eq. (583) | **1.62 [1.10, 2.13]**: FAIL; KS p = 0.20 (normal) |
| eq. (720) | half-width 0.096; coverage **50/50 [0.93, 1]**; bound/RMSE = 11.7 (dominant B_bad) |

**Reading.** Normal, small bias, bound covers with a 12× margin. The variance is 1.6× the
asymptotic prediction; the exact finite-m ratio is unavailable because the ladder (128..4096)
runs past the pilot's (8..512), so per-scale cv is missing on the top rungs. The origin
anchor's cv is 1.2 and still drifting, so as in 21 this is most likely Assumption 6 not yet
converged rather than a CLT failure; a pilot on the run's own ladder would settle it.

## Step 3 — re-pilot on the run's own ladder (2026-10-09)

`recipes/samples_pilot_deep.json`, i = 128..4096, study `pilot_deep`, 9 min. γ̂ = 1.7902
(43/24 = 1.7917). cv grows steadily with scale, **1.27 → 1.66** (pilot on 8..512: 1.01 →
1.41): **Assumption 6 (σ_k → σ∞, eq. 332) fails** for the origin observable. This is the
known i^(5/48) drift of the origin anchor (`models/percolation2d.py`). It explains the
1.6× variance ratio: eq. (583)'s σ∞² does not exist here. ω₁ and a₁ are unidentified on
128..4096 (1.33 ± 1.63), as expected once the correction has died out.
