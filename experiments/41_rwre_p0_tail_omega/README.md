# Experiment 41: `rwre_gpu_lazy` at $p=0$, planned with 19's tail $\omega$

**Question (user, 2026-10-15).** 19's autopilot planned $p=0$ with the pilot's
$\omega_1=0.450\pm0.001$, which was fitted on 4..1024. 19's `n4e6` sweep shows that this is a
short-ladder value. The (P) fit $\gamma+ck^{-\omega}$ on $s_k$, $k\ge256$, gives
$\omega=0.153\pm0.024$ ($\chi^2$ p = 0.54). This experiment redoes the 2h plan and the run
with that $\omega$.

## Design (decided with the user, 2026-10-15)

- **Pilot:** 19's `samples_autopilot_p0.json` unchanged (4..1024, snr rule, budget $10^9$),
  5 replicates, seed 20261015. Only its $d$ is kept.
- **Constants from 19's `n4e6` data at $p=0$** (`tail_constants.py`, which rewrites the
  pilot's `constants.json`):
  - $\omega_1$: the (P) tail fit's $\omega$.
  - $a_1=c\ln2/(2^{-\omega}-1)$, eq. (232)'s correction carried to the local slope at the
    lower scale. Its se comes from the fit's $(c,\omega)$ covariance.
  - cv: the mean of the per-scale cv over $256..2^{18}$.
- **Plan:** `plan.py --time 2h --replicates 5`, with no `--max-scale`. At ω = 0.153 it asks for
  2²³..2²⁸ (n = 3,009 ×5 on purus's pilot). At the default `cap` of 2¹⁵ the top rungs are
  expected to overflow: mean |X| is about 900 at 2¹⁸ and grows like √k.
- **`cap` fallback (user, 2026-10-15):** on an overflow, rerun everything (pilot, constants,
  plan, run) at `cap` 2¹⁷, then 2¹⁸, then 393,216 (the bit layout's 96 KiB maximum),
  keeping the same seeds, since `cap` changes no draw. The pilot is redone because `cap`
  sets occupancy and so throughput. If 393,216 still overflows, the next step is to stagger
  the samples so they share one collective buffer instead of each having its own `cap`. That
  is done only if needed. `data/chain_niagara.sh` runs this sequence on niagara after
  14's sweep exits.

| constant | 19 (`autopilot_p0_fix`) | 41 |
|---|---|---|
| $d$ | $1.153\pm0.008$ | this pilot |
| $\omega_1$ | $0.450\pm0.001$ | $0.153\pm0.024$ |
| $a_1$ | $-0.430$ | $-0.97\pm0.11$ |
| cv | 0.732 | 0.737 |

```bash
D=experiments/41_rwre_p0_tail_omega/data   # purus, 2026-10-15
python3 src/study/pilot.py -meta experiments/41_rwre_p0_tail_omega/recipes/samples_pilot_p0.json \
    --study pilot_p0 --data-root $D --replicates 5 --seed 20261015
python3 experiments/41_rwre_p0_tail_omega/tail_constants.py --study pilot_p0 --data-root $D
python3 src/study/plan.py --study pilot_p0 --data-root $D --time 2h --replicates 5 [--accept]
# the run itself: data/chain_niagara.sh, study run_p0_cap<cap>, run seed 20261016
```

## Results (2026-10-15 to 16, niagara)

The chain started at 21:47 when 14's sweep exited, and ended at 05:31 with exit 0.

| `cap` | pilot d | throughput | plan (2h) | outcome |
|---|---|---|---|---|
| 2¹⁵ | 1.005 ± 0.003 | 8.70e8 /s | m0=22, 2²³..2²⁸, n=2,145 | the run's preflight overflowed at k=2²³ |
| 2¹⁷ | 0.999 ± 0.000 | 2.82e8 /s | m0=21, 2²²..2²⁷, n=1,557 | the preflight overflowed at k=2²⁷ |
| 2¹⁸ | 1.092 ± 0.001 | 1.28e8 /s | **m0=18, 2¹⁹..2²⁴, n=1,268 ×5** | **drawn** |

A larger `cap` costs occupancy, so throughput falls and the 2h plan gets shallower. At 2¹⁸
it stops at 2²⁴, so 393,216 and the collective buffer were never needed. The draw took
139.8 min against 119.9 predicted (1.17×). Each replicate took 1668–1688 s.

**$\hat\gamma=0.5061$**:
- Student t(4) over 5 replicates: $\pm0.0028$, giving [0.4983, 0.5140]. This interval
  covers scatter only. The plan predicted a bias of 0.0153 and a per-replicate sd of 0.0071.
  The closed-form sd is 0.00706, and the replicate spread is 0.00635.
- eq. (720) bound (superseded, see the three-term section at the end): $\pm0.311$. Nearly all of it is $B_{fs}=0.266$, because at
  $\omega=0.153$, $\rho^{\omega}-1=0.11$ inflates the bound. The bound is flagged
  incomplete, since its $\omega_2$ piece is missing.
- The all-points OLS gives 0.50615. The eq. (232) fit gives 0.278, which is not meaningful
  with $\omega_1$ fixed this low.

Against 19's `autopilot_p0_fix` ($\omega_1=0.45$, 2¹⁵..2²⁰, $\hat\gamma=0.5180\pm0.0055$),
the deeper ladder moves $\hat\gamma$ toward ½, by 0.012. The remaining 0.006 is about
2 replicate se, and smaller than the planned bias of 0.015.

## Wilson interval in the article's three-term form (2026-11-04)

Theorem cor:wilson (article revision of 2026-11-04) dropped $B_{bad}$ together with the
log-moment assumption: $W = 1.96\sqrt{12\sigma_\infty^2/(n\,m(m^2-1))} + B_{fs} + 6\sigma^2_{\max}/(n(m+1))$.
Re-reported from the stored data; nothing was redrawn.

**[0.4907, 0.5216]**, half-width **0.0155** ($1.96\sigma_{se}$ 0.0140, $B_{fs}$ 0.0009,
$B_{good}$ 0.0006). It contains ½. The ±0.311 above predates 2026-11-02: since then
$B_{fs}$ comes from the pilot refitted with γ pinned, which returns ω₁ = 0.449 (pilot fit
0.153) and **warns** (χ² p = 0). The tiny $B_{fs}$ is therefore not trustworthy. The bias the
plan predicted (0.0153) is as large as the whole half-width. The four-term bound of
2026-11-02 was ±0.0457, of which B_bad was 0.0294.

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
| `run_p0_cap262144` | 0.506147 | 0.449 ± 0.001 | 5.91e+03/6 ⚠ | 0.0155 | 0.178 ± 0.011 | 64..1024 | 1.48/2 | 0.145 |

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
| `run_p0_cap262144` | 0.506147 | 0.178 | 0.145 | 0.183 ± 0.0109 | 64..1024 (0.74) | -0.752 | 0.118 | 0.132 |

## Wilson interval with the joint refit (2026-11-11)

B_fs now takes $\omega_1, a_1$ from one AIC range fit on the pilot **and** the long run
together ($\gamma$ free, $\ge 6$ consecutive scales, a scale in both merged by inverse
variance; `report.joint_refit`, user 2026-11-11), not from the pilot refitted with $\gamma$
pinned to $\hat\gamma$, whose pin carries $\hat\gamma$'s scatter but not its bias (see
`src/study/README.md`). $\hat\gamma$ is unchanged; only B_fs and the eq. (720) half-width
move. Old = the 2026-11-10 pinned average (still in `answer.json` as `pilot_refit`), new =
`joint_refit`; reports rerun 2026-11-11. "best" is the lowest-AIC range of the union and
its share of the kept weight; its $\chi^2$ gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `run_p0_cap262144` | 0.506147 | 0.183 | 0.132 | 0.151 ± 0.0687 | 128..8388608 (0.16) | -0.919 | 0.262 | 0.276 |
