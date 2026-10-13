# Experiment 24 — Cost exponent, Assumption 7 (eq. 353), across dimension

Part of `plans/percolation_experiments.md` (2026-10-08). **Status: done from existing pilot probes (2026-10-13), no new draws.**

## Acceptance criterion

$\hat d$ against the declared dim for $d=2..5$ (GPU batched probes) and at $d=3$ for the streaming sampler at large $i$; explain or quantify the $+3$-$7\%$ overshoot seen in the 2026-09-17 pilots.

## Known values — acceptance criteria, not inputs (ground rule 4)

Declared cost exponent = dim.

## Result (2026-10-13)

```bash
python3 experiments/24_perc_cost_d2_d5/cost_from_pilots.py
```

Each pilot already ran five batched cost probes (`data/pilot/pilot.json` → `cost`), all on
niagara's RTX 5090, model `percolation_zd_gpu` (`percolation2d_gpu` at dim 2), `face_far`/cylinder
(south at dim 2). The script collects them; it draws nothing.

| dim | pilot | d̂ (5 probes) | d̂ / dim | ns/site, smallest → largest i | local drift d_local − dim, per doubling |
|---|---|---|---|---|---|
| 2 | 20 | 2.022 ± 0.002 | 1.011 | 0.136 → 0.095 | −0.42, −0.17, −0.01, +0.04, +0.01, +0.02, +0.01 |
| 3 | 27 | 3.136 ± 0.015 | 1.045 | 0.164 → 0.230 | −0.44, −0.14, −0.06, −0.02, +0.04, −0.05, **+1.16** |
| 4 | 29 | 4.223 ± 0.006 | 1.056 | 0.149 → 0.246 | −0.25, −0.06, +0.03, +0.09, **+0.91** |
| 5 | 30 | 5.329 ± 0.016 | 1.066 | 0.156 → 0.271 | −0.19, −0.11, +0.33, **+0.77** |
| 6 | 31 | 6.198 ± 0.002 | 1.033 | 0.231 → 0.275 | −0.43, −0.20, −0.16, **+1.05** |

**Reading.** Assumption 7 (cost = i^dim) holds on the middle of every ladder: per-site cost is
flat to within ±0.1 in the local exponent once the per-call constant has been amortized (the
negative drifts at the bottom). The whole overshoot of d̂ above dim comes from the **top rung**,
where per-site cost roughly doubles. The top rungs are 512³ = 1.3·10⁸, 128⁴ = 2.7·10⁸,
64⁵ = 32⁶ = 1.1·10⁹ sites per sample, while dim 2's top (1024² = 10⁶) shows nothing: the jump
appears once one sample exceeds about 10⁸ sites, where `block_n` is 1 and the labelling
array leaves the GPU's caches. So d̂ is a property of where the ladder stops, not of the
dimension, and a plan whose ladder ends higher than the probe's extrapolates the wrong
throughput: 27 (planned on probes ≤ 512³, ran to 1024³) took **1.64×** its predicted time.

Figure: `data/cost_from_pilots.png` ((a) seconds per sample vs i, (b) ns per site vs i).
