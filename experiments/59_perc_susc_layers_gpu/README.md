# 59 — One cluster grown on the GPU: when does it beat a CPU thread?

Model: `models/percolation_susceptibility_layers_gpu.py` (`MODELS["percolation_susceptibility_layers_gpu"]`).
The plan (user, 2026-11-12):
1. Build GPU growth of one cluster, parallel within a chemical layer. Find where it is faster than a CPU thread.
2. Build a hybrid: small clusters on the CPU, huge ones sent to the GPU.
3. Set the size cut from measured CPU and GPU timings and RAM↔VRAM transfer costs.

This README covers part 1. The hybrid is a separate model with its own experiment.

## Acceptance criteria

- **Bit-for-bit:** for the same seed, $|C|$, the depth and every layer size $N_\ell$ equal the CPU `_one_sample`'s.
  This must hold at $d=2..5$, with a small `max_live` that forces rebuilds, and with the hard load limit lowered to 1/2.
  In step 1 the lowered limit forced refused inserts and repeated layers. Since step 3 it makes launches end and resume often.
  The bench also asserts $|C|$ against the CPU for every timed cluster, up to $2.9\cdot10^8$ sites.
- **Cap:** a sample whose three live layers do not fit the table cap raises `MemoryError`.

## Step 1 — validation, one launch per layer (commit 3915b77; purus, RTX 5090, 2026-11-12)

```bash
.venv/bin/python experiments/59_perc_susc_layers_gpu/check_gpu.py validate
```

All identical:
- 3000 sample pairs, i.e. 5 configurations × 2 load limits × 300 seeds.
- $|C|$ up to $5.2\cdot10^6$ ($d=2$).
- 8,822 rebuilds and 1,894 repeated layers.

The cap check: the one `MemoryError` seen while choosing `max_live` was legitimate. CPU `need3` was 842 against a 768-site limit.
`tools/tests/test_percolation_susceptibility_layers_gpu.py` passes: 12 tests, including `simulate` against the CPU model and the cap.

## Step 2 — timing, one launch per layer, $d=5$, $\varepsilon_0=0.07$, $x=4096$ (purus)

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

## Step 3 — many layers per launch (purus, 2026-11-12)

Change: one cooperative launch runs layer after layer, with one `grid.sync()` between layers (170 blocks × 256 threads).
A barrier alone costs ~0.9 µs over the whole RTX 5090 (1000 barriers, 1 to 680 requested blocks; CuPy clamps a cooperative grid to 170).
The design and why it is exact are in the model's docstring. In short:
- the counters rotate over 3 slots;
- occupancy = occupancy at launch + the N appended since;
- the kernel stops before any layer whose worst case, $(2d-1)N_\ell$ new sites, could overfill the table or the frontier buffer.

So no insert is ever refused and the repeat path is gone.
The price: room is reserved for the worst case, so the `max_live` cap binds earlier than on the CPU. The validation case `max_live=1000` at $d=5$ became 5000.

**Bug found on the way:** `enable_cooperative_groups` applies to the whole CuPy module, so the `rebuild` kernel's grid was also clamped to 1,020 blocks.
Tables above 261k slots were then partly copied: entries were lost and clusters blew up.
The validation's tables were too small to show it. The bench caught it: a `MemoryError` claiming a worst case of 10⁸ new sites in one layer.
Fix: `rebuild` now lives in its own module and is grid-stride. `test_rebuild_large_table` (2²¹ slots, 300k keys) guards it.

Validation: `ALL IDENTICAL` again (3000 pairs, 12,513 rebuilds, 13,328 launches). 13 tests pass.

Bench: same scan, same clusters, same command.

| $\lvert C\rvert$ ≥ | n | depth | CPU ms | GPU ms | CPU/GPU | step 2 CPU/GPU |
|---|---|---|---|---|---|---|
| 1 | 8 | 0 | 0.007 | 0.123 | 0.06 | 0.06 |
| 64 | 8 | 26 | 0.014 | 0.180 | 0.08 | 0.03 |
| 1024 | 8 | 122 | 0.164 | 0.457 | 0.36 | 0.08 |
| 4096 | 8 | 236 | 0.619 | 1.10 | 0.56 | 0.18 |
| 16384 | 8 | 530 | 2.68 | 2.52 | **1.06** | 0.29 |
| 65536 | 8 | 1250 | 9.72 | 6.30 | 1.54 | 0.47 |
| 262144 | 8 | 2017 | 47.5 | 13.3 | 3.6 | 1.25 |
| 1048576 | 8 | 5010 | 199 | 32.6 | 6.1 | 2.0 |
| 4194304 | 8 | 9678 | 681 | 68.4 | 10.0 | 3.0 |
| 16777216 | 8 | 14812 | 2924 | 115 | 25 | 6.8 |
| 67108864 | 8 | 46692 | 11729 | 369 | 32 | 7.7 |
| 268435456 | 3 | 60746 | — | 579 | — | — |

The best single cluster ran 55× faster ($2.8\cdot10^8$ sites: 24.3 s on the CPU, 0.44 s on the GPU).

- **Cost model:** GPU time ≈ **5.8 µs per layer + 0.70 ns per site**, against 87 ns per site on one CPU thread.
- **Crossover** against one CPU thread at $|C|\approx1.6\cdot10^4$, about 12× lower than in step 2.
- **Fixed cost of a sample on the GPU:** ~0.12 ms (launch, the D2H reads, allocating the table and buffers).
  It is paid even for a single-site cluster. Small clusters stay on the CPU.
- **Not yet measured:** CPU throughput with all threads busy, which is the real competitor in the hybrid (experiment 58: ~3× slower per thread under load).
  Also not measured: the cost of handing a cluster from the CPU to the GPU mid-growth. Both belong to the hybrid's experiment.

## Later change (experiment 60, 2026-11-12)

A rebuild now grows the table to `_SLACK = 8` times the worst-case need.
Before, launches at $d=2$ stopped for room every ~3 layers. Validation was rerun afterwards: `ALL IDENTICAL`.
The step 3 timings above were taken before this change.
