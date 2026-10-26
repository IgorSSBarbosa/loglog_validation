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
- eq. (720) bound: $\pm0.311$. Nearly all of it is $B_{fs}=0.266$, because at
  $\omega=0.153$, $\rho^{\omega}-1=0.11$ inflates the bound. The bound is flagged
  incomplete, since its $\omega_2$ piece is missing.
- The all-points OLS gives 0.50615. The eq. (232) fit gives 0.278, which is not meaningful
  with $\omega_1$ fixed this low.

Against 19's `autopilot_p0_fix` ($\omega_1=0.45$, 2¹⁵..2²⁰, $\hat\gamma=0.5180\pm0.0055$),
the deeper ladder moves $\hat\gamma$ toward ½, by 0.012. The remaining 0.006 is about
2 replicate se, and smaller than the planned bias of 0.015.
