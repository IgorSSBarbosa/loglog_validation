# 59 — One cluster grown on the GPU: when does it beat a CPU thread?

Model: `models/percolation_susceptibility_layers_gpu.py` (`MODELS["percolation_susceptibility_layers_gpu"]`).
The plan (user, 2026-11-12):
1. Build GPU growth of one cluster, parallel within a chemical layer. Find where it is faster than a CPU thread.
2. Build a hybrid: small clusters on the CPU, huge ones sent to the GPU.
3. Set the size cut from measured CPU and GPU timings and RAM↔VRAM transfer costs.

This README covers part 1.

## Acceptance criteria

- **Bit-for-bit:** for the same seed, $|C|$, the depth and every layer size $N_\ell$ equal the CPU `_one_sample`'s.
  This must hold at $d=2..5$, with a small `max_live` that forces rebuilds, and with the hard load limit lowered to 1/2.
  The lowered limit forces refused inserts and repeated layers.
- **Cap:** a sample whose three live layers do not fit the table cap raises `MemoryError`.

## Step 1 — validation (purus, RTX 5090, 2026-11-12)

```bash
.venv/bin/python experiments/59_perc_susc_layers_gpu/check_gpu.py validate
```

All identical:
- 3000 sample pairs, i.e. 5 configurations × 2 load limits × 300 seeds.
- $|C|$ up to $5.2\cdot10^6$ ($d=2$).
- 8,822 rebuilds and 1,894 repeated layers.

The cap check: the one `MemoryError` seen while choosing `max_live` was legitimate. CPU `need3` was 842 against a 768-site limit.
`tools/tests/test_percolation_susceptibility_layers_gpu.py` passes: 12 tests, including `simulate` against the CPU model and the cap.

## Step 2 — timing, $d=5$, $\varepsilon_0=0.07$, $x=4096$ (purus)

```bash
NUMBA_NUM_THREADS=28 .venv/bin/python experiments/59_perc_susc_layers_gpu/check_gpu.py scan --x 4096 --n 1000000
.venv/bin/python experiments/59_perc_susc_layers_gpu/check_gpu.py bench --x 4096 --per-bin 4
```

The scan ran $10^6$ samples on the CPU layers model (176 s, max $|C|=3.1\cdot10^8$).
The bench then picked 4 clusters per $\log_2|C|$ bin. Each was timed alone on one CPU thread (`_one_sample`) and on the GPU (`grow_cluster`).
Medians per $\log_4$ bin:

| $\lvert C\rvert$ ≥ | n | depth | CPU ms | GPU ms | CPU/GPU |
|---|---|---|---|---|---|
| 1 | 8 | 0 | 0.008 | 0.134 | 0.06 |
| 64 | 8 | 26 | 0.015 | 0.521 | 0.03 |
| 1024 | 8 | 122 | 0.171 | 2.0 | 0.08 |
| 16384 | 8 | 530 | 2.6 | 9.0 | 0.29 |
| 65536 | 8 | 1250 | 9.7 | 20.8 | 0.47 |
| 262144 | 8 | 2017 | 46.7 | 37.5 | 1.25 |
| 1048576 | 8 | 5010 | 211 | 104 | 2.0 |
| 4194304 | 8 | 9678 | 684 | 230 | 3.0 |
| 16777216 | 8 | 14812 | 2910 | 429 | 6.8 |
| 67108864 | 8 | 46692 | 11634 | 1503 | 7.7 |
| 268435456 | 3 | 60746 | — | 2054 | — |

At the top, single clusters reached 17× ($2.8\cdot10^8$ sites: 24.2 s on the CPU, 1.39 s on the GPU).

- **Cost model.** GPU time ≈ **24.5 µs per layer + 1.7 ns per site** (least squares, $|C|\ge10^3$).
  One CPU thread costs **87 ns per site** (7.9 ns per neighbour read).
  The CPU rate is with the machine otherwise idle. Experiment 58 measured 22–30 ns/read per thread with 16 threads running, where memory traffic is shared.
- **Crossover** at $|C|\approx2\cdot10^5$, against one CPU thread.
- **The per-layer cost dominates.** A layer has to hold more than ~14,000 sites before its work (1.7 ns a site) outweighs the 24.5 µs.
  The mean layer of a $10^8$-site cluster has ~3,000 sites. So even the largest clusters are bound by launches and syncs, not by the kernel.
  For a $6.7\cdot10^7$ cluster that is 1.1 s of overhead against 0.11 s of work.
- **Where the 24.5 µs goes** (microbenchmark, empty kernel, 20,000 repetitions):

  | step | µs |
  |---|---|
  | launch | 4.1 |
  | 24-byte H2D counter reset | 2.1 |
  | 24-byte D2H counter read (the sync) | 5.8 |
  | all three | 11.9 |

  The remaining ~12 µs is the Python bookkeeping per layer plus the latency of a kernel that does dependent hash probes on a small frontier.

## Open (next step before the hybrid)

The GPU's work rate (1.7 ns a site) is already ~50× a CPU thread. What it lacks is a cheaper layer.
Option: a persistent kernel that runs many layers per launch, with a grid-wide barrier between layers. It would return to the host only when the table needs room or the frontier buffer could overflow.
That would cut the per-layer cost from 24.5 µs to a few µs, and move the crossover down.
That crossover sets the hybrid's size cut, so it should be measured after this change, not before.
