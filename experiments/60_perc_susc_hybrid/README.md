# 60 — Hybrid: small clusters on the CPU, huge ones on the GPU

Model: `models/percolation_susceptibility_hybrid.py` (`MODELS["percolation_susceptibility_hybrid"]`).
This is part 2 of the plan in experiment 59 (user, 2026-11-12): send only the clusters above a size cut to the GPU, and choose the cut from measured timings.

## Design

- Every sample starts on a CPU thread (numba). Each thread has one table, cleared between samples, not reallocated.
- A cluster that would pass `cut` sites is handed off: its **seed** goes into a queue, and the GPU regrows it from the origin with experiment 59's `grow_cluster`.
- The lattice is a function of the seed, so the regrown cluster is the same one. No cluster state is transferred.
  - RAM→VRAM traffic per hand-off: one integer.
  - Cost per hand-off: the ≤ `cut` sites the CPU had already grown, thrown away.
- The CPU processes the samples in chunks of 4096 × threads. One Python thread feeds the queue to the GPU while the next chunk runs. The numba batch is `nogil`.

## Acceptance criteria

- Same rng → the same $Y$ and reads as the CPU layers model, sample by sample, for every cut.
- `on_gpu` is exactly $\{|C| > \text{cut}\}$.
- An error on the GPU thread is raised in the caller.

## Step 1 — validation (purus, 2026-11-12)

`tools/tests/test_percolation_susceptibility_hybrid.py`: 14 tests pass. They cover $d=2,3,5$ at cuts 1, 7, 300 and $10^7$, with chunks smaller than $n$ so that hand-offs overlap the CPU chunks that follow.
A scratch check at $n=2\cdot10^4$ ($d=5$, $x=512$ and 4096) and $n=3000$ ($d=2,3$) was identical at cuts 1, 50, $2^{16}$ and $10^7$.

**It exposed a GPU-model fix.** At $d=2$ the GPU took 17 ms per cluster, because each launch ran only ~3 layers before stopping for table room.
`percolation_susceptibility_layers_gpu` now grows the table to `_SLACK = 8` times the worst-case need whenever it rebuilds. Experiment 59's validation was rerun afterwards: identical.
$d=2$ remains GPU-hostile. Its layers are small and its clusters deep, so the cost is almost all per-layer overhead. A 2D run should keep the cut high.

## Step 2 — the cut, $d=5$, $\varepsilon_0=0.07$, $x=4096$, $n=10^6$ (purus, 30 cores, RTX 5090)

```bash
NUMBA_NUM_THREADS=30 .venv/bin/python experiments/60_perc_susc_hybrid/sweep_cut.py --x 4096 --n 1000000
NUMBA_NUM_THREADS=30 .venv/bin/python experiments/60_perc_susc_hybrid/sweep_cut.py --x 4096 --n 1000000 \
    --no-baseline --cuts 8388608 16777216 33554432 --label _high
```

- The baseline is the CPU layers model on 30 threads. The hybrid uses 29 numba threads; the 30th core drives the GPU.
- Seed 20261112, the same samples as experiment 59's scan (max $|C| = 3.1\cdot10^8$).
- Column meanings:
  - `cpu`: time inside the CPU batches.
  - `gpu`: the GPU thread's busy time.
  - `gpu-only`: wall time after the last CPU chunk, i.e. time when only the GPU was working.

| run | cut | wall s | cpu s | gpu s | gpu-only s | hand-offs | same $Y$ as baseline |
|---|---|---|---|---|---|---|---|
| layers (CPU) | — | 174.9 | — | — | — | — | — |
| hybrid | 256 | 66.4 | 0.0 | 66.3 | 66.3 | 20,087 | yes |
| hybrid | 1,024 | 64.3 | 0.1 | 64.3 | 64.3 | 11,752 | yes |
| hybrid | 4,096 | 62.1 | 0.2 | 62.1 | 61.9 | 6,662 | yes |
| hybrid | 16,384 | 59.1 | 0.4 | 59.0 | 58.7 | 3,777 | yes |
| hybrid | 65,536 | 55.0 | 0.8 | 54.9 | 54.1 | 2,113 | yes |
| hybrid | 262,144 | 50.1 | 3.4 | 49.7 | 46.7 | 1,184 | yes |
| hybrid | 1,048,576 | 44.9 | 12.5 | 43.5 | 32.3 | 614 | yes |
| hybrid | **4,194,304** | **41.4** | 34.8 | 37.5 | 6.6 | 306 | yes |
| hybrid | 8,388,608 | 57.1 | 55.5 | 33.6 | 1.6 | 222 | not checked |
| hybrid | 16,777,216 | 87.0 | 86.2 | 26.8 | 0.9 | 132 | not checked |
| hybrid | 33,554,432 | 174.9 | 174.1 | 21.1 | 0.8 | 74 | not checked |

- **Best cut on this grid: $2^{22}\approx4\cdot10^6$ sites.** Wall time is 41.4 s, 4.2× the CPU-only run, with the two sides balanced (CPU 34.8 s, GPU 37.5 s).
  The cut is the balance point between the two devices. It is not the single-thread crossover of experiment 59 ($1.6\cdot10^4$).
- **Below the optimum, the GPU queue sets the wall.** The CPU finishes in under a second while the GPU works through the hand-offs one at a time.
  GPU time hardly depends on the number of hand-offs (66 s for 20k clusters, 55 s for 2k). It is set by the large clusters.
  For those clusters the cost is mostly layers: a $10^6$-site cluster has ~5,000 layers ≈ 29 ms at 5.8 µs a layer. 614 clusters above $10^6$ took 43.5 s, ~70 ms each, as experiment 59's cost model predicts.
- **Above the optimum, the CPU sets the wall.** Its time grows with the cut much faster than the work moved to it.
  The batches are static `prange` chunks, so each chunk waits on its slowest sample. That is experiment 58's idle-thread effect, now bounded by the cut.
  At cut $2^{25}$ the hybrid is back at the baseline's wall time.

## Step 3 — two-word keys, so $x\ge 65536$ can run (purus, 2026-11-12)

- **Why.** The 10 h plan from the $n=25\cdot10^6$ pilot (`gsusc_d5_hybrid_n25m`, scales 4…512) ended at $x=65536$. There a cluster reached the one-word key's $\pm2^{11}$ edge after $4\cdot10^5$ samples, and the plan drew $3.7\cdot10^6$ per replicate.
- **Change.** The GPU model's keys are now two words: 21 bits a coordinate at $d=4,5$ ($\pm2^{20}$).
  - A site inside the old box keeps its old state; outside it a separate hash decides. Details are in the model's docstring.
  - The hybrid's CPU side keeps the one-word key and hands a cluster that reaches its edge to the GPU.
- **Validation.**
  - Experiment 59's `check_gpu.py validate`: identical sizes and layer profiles in $d=2$–5, at both hard loads.
  - Local tests: 34 pass. They include a Python reference of the far-site rule (box shrunk to $\pm4$) and edge hand-offs in the hybrid.
  - The $x=65536$ sequence that crashed now completes; its largest cluster was $1.16\cdot10^{10}$ sites, 412,722 layers, 192 MiB of VRAM.
- **Cost.** On the same 115 clusters ($x=4096$), GPU time is 1.26× the one-word kernel's.
  - The cut sweep still puts the optimum at $2^{22}$. At $x=4096$, $n=10^6$ the wall is 53.2 s (one-word: 41.4 s); at $x=16384$, $n=2\cdot10^5$ it is 29.5 s.
  - The cost probe was redone: affine $d=1.087\pm0.094$ (one-word: $0.968\pm0.094$).

### The 10 h run

- **The probe's $d$ mispredicts the plan's scales.** Measured walls per $10^6$ samples (cut $2^{22}$, 29 threads + GPU): 2048: 32 s, 4096: 48 s, 8192: 71 s, 16384: 103 s, 32768: 174 s, 65536: 292 s, 131072: 244 s.
  - That is about $x^{0.64}$, against the probe's $x^{1.09}$ on 4…4096. With the pilot's throughput, plan.py predicted 200 min for a replicate that takes ~29 min.
  - The GPU is the bottleneck at all these scales; the CPU side is busy ~40 s per $10^6$.
- **Plan.** `plan.py --time 10h --throughput 7e8` (throughput set from these timings):
  - $m_0=11$, scales 4096…131072, $n=11{,}415{,}415$ per scale × 3 replicates.
  - Predicted se on the answer 0.0030; measured-rate estimate ~9 h.
  - The plan's se uses the pilot's mean cv (19.5); the per-scale cv grows like $x^{0.41}$, so the realised se will be larger.
- **Launch.** 2026-11-12 18:51, seed 2026111262. Log: `data/gsusc_d5_hybrid_n25m_run.log`. The code is the working tree on top of `a9d5330`; the two-word key is not committed yet.

## Not done yet: two independent improvements

1. **Several clusters on the GPU at once.** A layer of a few thousand sites occupies a small fraction of the RTX 5090, and clusters are grown one after another.
   Fix: several GPU threads, each with its own CUDA stream and a smaller cooperative grid.
   Memory per cluster stays bounded (three live layers), so this keeps the "parallel within a cluster, bounded memory" principle while filling the device.
2. **Dynamic CPU scheduling.** Use many small chunks per thread instead of one static chunk, so a thread that drew a large cluster doesn't hold up the rest.

Both lower their own side's curve. The optimal cut should be measured again after either one.
