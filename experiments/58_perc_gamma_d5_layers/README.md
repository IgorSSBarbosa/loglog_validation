# Experiment 58 — $\hat\gamma$ in $d=5$ by layered exploration (memory ~ three layers)

**Status: model validated bit for bit against the explore model; d=5 benchmark to
x=16384 done (2026-11-11). Pipeline not launched.**

## Question (user, 2026-11-11)

Experiment 39 stopped at $x=512$, and its benchmark (2026-11-11) put the ceiling at
$x\approx 2048$–$4096$: the explore model stores every open site of the cluster, and the
tail of $|C|$ (clusters $>10^8$ sites at $x\ge 4096$) is what runs out of memory.
The user's idea: once the table is full, drop every site that can never be looked up
again, and stop only if a clean-up frees nothing.

## Model

`models/percolation_susceptibility_layers.py` (registered as
`percolation_susceptibility_layers`). Same estimand, ladder, lattice and seed draw as
`percolation_susceptibility_explore` with one seed:
$p=p_c-\varepsilon_0/x$, $Y=|C(0)|\,1\{0\text{ open}\}/p$, $\mathbb E Y=S(p)$.

The cluster grows breadth first by chemical distance $\ell$. A stored site is only looked
up when an open neighbour is expanded, and neighbours' chemical distances differ by at
most 1. So while layer $\ell$ is expanded, every site in layer $\le\ell-2$ is dead. Each
table entry carries its layer. When the table holds `max_live` sites it is rebuilt with
layers $\ge\ell-1$ only. If the rebuild frees nothing, the sample raises MemoryError
(the user's stopping rule). $|C|$ is counted at expansion, so deletion never loses or
double-counts a site.

Per call it also returns the layer profile, $\sum_{\rm samples}N_\ell$ and
$\sum N_\ell^2$ (layers past `max_layers` are summed into the last bin), and per-sample
depth and `need3` $=\max_\ell(N_{\ell-1}+N_\ell+N_{\ell+1})$. `need3` is the least memory
any cap could use.

## Step 1 — validation (`check_layers.py validate`, purus, 4 threads)

With the same rng, the two models return **identical** $Y$ and neighbour-read counts,
sample by sample. That holds in $d=2,3,4,5$ (5000 samples per row), both with the default
cap and with `max_live` $=20{,}000$, which forces up to 1048 rebuilds per call. The
profile sums to $\sum|C|$ in every row. Log: `data/validate.log`.

```bash
.venv/bin/python experiments/58_perc_gamma_d5_layers/check_layers.py validate --n 5000
```

## Step 2 — benchmark, $d=5$, $\varepsilon_0=0.07$ (xingu, 16 threads)

```bash
NUMBA_NUM_THREADS=16 .venv/bin/python experiments/58_perc_gamma_d5_layers/check_layers.py bench \
    --threads 16 --x 512 1024 2048 4096 8192 16384 \
    --nx 1000000 1000000 1000000 1000000 1000000 200000 --seed 2026111158
```

This run used `max_live` $=2^{24}$; the default is now $2^{20}$, see below. Per-sample
arrays and profiles: `data/bench/bench_x*.npz`. Log: `data/bench.log`. One rng drives
the whole ladder, so the scales are not independent; this is a benchmark, not a
$\hat\gamma$ run.

| x | n | wall s | mean Y | se | cv | max $\lvert C\rvert$ | max need3 | need3/$\lvert C\rvert$ ($\lvert C\rvert>10^6$), median / max | max depth | ms/sample/thread | largest sample's share of reads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1e6 | 11.4 | 6347 | 261 | 41 | 8.0e6 | 1.0e4 | 0.0020 / 0.0040 | 16434 | 0.18 | 0.009 |
| 1024 | 1e6 | 34.7 | 1.303e4 | 740 | 57 | 2.7e7 | 2.1e4 | 0.0018 / 0.0041 | 20067 | 0.55 | 0.015 |
| 2048 | 1e6 | 86.9 | 3.044e4 | 2.1e3 | 69 | 8.7e7 | 3.3e4 | 0.0015 / 0.0046 | 37546 | 1.39 | 0.020 |
| 4096 | 1e6 | 163.0 | 5.402e4 | 5.2e3 | 96 | 2.9e8 | 4.6e4 | 0.0014 / 0.0044 | 80895 | 2.61 | 0.038 |
| 8192 | 1e6 | 761.9 | 1.694e5 | 2.7e4 | 156 | 2.5e9 | 1.2e5 | 0.0012 / 0.0042 | 229877 | 12.2 | 0.105 |
| 16384 | 2e5 | 729.5 | 4.47e5 | 1.7e5 | 171 | 4.0e9 | 2.4e5 | 0.0010 / 0.0040 | 194566 | 58.4 | 0.317 |

- **Memory is solved.** The three live layers never needed more than 0.5% of $|C|$, at
  most $2.4\cdot10^5$ sites. A $4\cdot10^9$-site cluster (about 120 GiB in the explore
  model) ran in 386 MiB. No sample hit the cap and no cluster reached the key range
  ($\pm2047$) up to $x=16384$, so 16-bit keys are still not needed.
- **Default cap.** Same ladder at $x=4096$ with other caps (`data/bench_cap.log`; this
  run drew different samples, so the comparison is not paired):

  | cap | ns/read |
  |---|---|
  | $2^{18}$ | 22 |
  | $2^{20}$ | 30 |
  | $2^{24}$ | 34 |

  The default is $2^{20}$ (24 MiB a thread), a 4x margin over the largest `need3`.
  $2^{18}$ would be within 7% of a MemoryError at $x=16384$.
- **The wall moves from memory to cost and variance.**
  - The cv grows with x: 41, 57, 69, 96, 156, 171. Experiment 39's bench capped at
    $2\cdot10^8$ sites and read 87 at $x=8192$, a truncated tail.
  - Above $x=4096$, single samples dominate. At $x=16384$ one cluster is 32% of all reads
    and costs about 10 minutes on one thread. A static-chunk prange then waits on it,
    which is why ns/read per thread rises to 51 and 93: that is idle threads, not slower
    reads.
  - Holding se(Y)/Y at 1% needs $n=(cv/0.01)^2$, i.e. $9\cdot10^7$ samples at 4096
    (about 66 thread-hours) and $2.4\cdot10^8$ at 8192 (about 820 thread-hours).

## Layer profile (the second exponent)

$\mathbb E N_\ell$ per sample (closed origin counted as 0) from the bench files. The
growth is a power law in $\ell$, cut off at a chemical correlation length that moves out
with $x$:

| x | $\ell$=2 | 8 | 32 | 128 | 512 | 1024 | local slopes, $\ell=2..256$ |
|---|---|---|---|---|---|---|---|
| 512 | 0.233 | 0.290 | 0.354 | 0.404 | 0.383 | 0.310 | 0.15 0.16 0.15 0.14 0.11 0.07 0.03 |
| 2048 | 0.236 | 0.294 | 0.364 | 0.443 | 0.491 | 0.499 | 0.16 0.16 0.16 0.15 0.15 0.14 0.09 |

Slope about 0.15 before the cutoff. What estimand it maps to, and how to fit it (the
$\ell$ rungs share samples; the cutoff depends on $x$), is open: not used yet, ground
rule 3.

## Step 3 — pilot (not launched)

`recipes/samples_gsusc_d5_layers.json`: the ladder is 4..4096, with $n=(cv/0.02)^2$ per
rung. That is about 24 thread-hours, roughly 1.5 h on 16 threads.

```bash
.venv/bin/python src/study/pilot.py -meta experiments/58_perc_gamma_d5_layers/recipes/samples_gsusc_d5_layers.json \
    --study gsusc_d5_layers --data-root experiments/58_perc_gamma_d5_layers/data --replicates 5
```

## Known values — acceptance criteria, not inputs (ground rule 4)

As experiment 39: $\gamma=1.1792(7)$; correction $\Delta_1\approx0.41$ (39's long-run AIC fit
gives 0.42).

Launched 2026-11-11 19:16 on xingu (idle: load 0.0, 166 GB available), 30 threads, seed
2026111158. Each replicate draws the recipe's full n (52.4M draws). Projected from x=512
(4.2M samples in 32.9 s): about 1 h per replicate, about 5 h in total. Log:
`data/gsusc_d5_layers_pilot.log`.
