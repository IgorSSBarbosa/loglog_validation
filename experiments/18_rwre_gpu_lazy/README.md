# Experiment 18 — `rwre` on a lazily revealed environment (GPU)

**Question.** Can `rwre`'s walk reach $k=N=2^{20}$ if the environment is only simulated
where the walker has looked? The design, and why it is exact, is
`plans/rwre_gpu_lazy.md` (§2: an unread site holds an unobserved initial value,
Bernoulli($\alpha$) independent of the past, because stirring acts on sites). This README
holds the acceptance criteria (copied from the plan's §6) and what was measured.

| | |
|---|---|
| model | `models/rwre_gpu_lazy.py`, phase A, the byte layout. One warp per sample. A deque of cells $\{U,0,1\}$ in `cap` bytes of shared memory, both ends always $U$, growing by `growth` cells, never shrinking. Philox4x32-10 keyed by what each draw is for, so the output doesn't depend on launch chunk, `growth` or `cap`. Skip-read on by default (`skip_read`). Not registered in `tools/models.py` yet (phase C) |
| parameters | `p`, `alpha`, `env_sweeps`, `swap_prob` as `rwre`; `skip_read` (default true); `cap` (default $2^{15}$), `growth` (default 32), performance only |
| observable | $Y_k=\lvert X_k\rvert$; `walk_stats` also returns reveals, ones, reads, deque width and ends |
| checks | `tools/tests/test_rwre_gpu_lazy.py` (L0, L2, L3, Philox known answers; local, gitignored), `local/rwre_lazy_ks.py` (L1a, L1), `local/rwre_lazy_l4.py` (L4, P1 timing) |

## Acceptance criteria (plans/rwre_gpu_lazy.md §6, decided 2026-10-02)

**L0. Bit-identity (hard).** Same seed, identical output across launch chunking and
prefix in $n$, across `growth` and across `cap` (short of an overflow). An overflow raises
and names the sample. Byte vs bit layout is phase B.

**L1a. Smoke KS against `rwre` (CPU).** Signed $X_k$, $n=4000$ per arm, independent
spawned seeds, `rwre` at its default window. $p=1/3$, $p=0$, $p=1/2$, frozen
`env_sweeps=0` at $p=0.2$, $\alpha=0.3$ at $p=0.3$, $p=1/3$ skip-read off, at $k=16,256$;
$p=1/3$ and $p=0$ at $k=1024$. PASS: KS $p\ge0.01$ and $\lvert\Delta\bar X\rvert\le3$
combined se in every arm.

**L1. Distribution against `rwre_gpu`.** $Y_k$, $k\in\{64,1024,16384\}$ × 13's five G1
parameter sets × skip-read on/off, $n=4000$, reference on the linear window $W=2k+100$.
Same PASS rule.

**L2. Exact references.** 13's G2 battery ($k=1$ law, frozen $k=2$, zero-check with
parity and range at $k=256,1024,16384$, $\lvert X\rvert$ same law at $p$ and $1-p$, closed
cases), plus at $p=1/2$ with skip-read: zero reveals and the Binomial law of $S_{64}$.

**L3. Lazy-specific invariants.** In a debug build, after every read and every sweep: known
cells in $[L,R]$ equal the reveals, ones equal the revealed ones, both ends $U$. Revealed
ones are Binomial(reveals, $\alpha$) pooled, within 4 se.

**L4. Measurements, not pass/fail.** Deque width $W(k)$, reveals $R(k)$ and overflow at
$k=2^8..2^{16}$, $p\in\{0,1/3,1/2\}$, to fix `cap` (phase A').

## L0, L1a, L1, L2, L3: PASS (2026-10-02)

RTX 5090, cupy 14.2.0, CUDA runtime 12.9.

**L0, L2, L3**: `python3 -m pytest tools/tests/test_rwre_gpu_lazy.py`, 42 passed. They
include Random123's three Philox4x32-10 known-answer vectors run through the kernel's own
device function, and two extra checks:
- reads per sample are Binomial($k$, $\lvert1-2p\rvert$) under skip-read (4 se);
- at $p=1/2$, a fixed seed gives the **same** walk with skip-read on and off (both
  branches go left iff $u<1/2$ on the same keyed uniform), while only the second
  reveals anything.

**L1a**: 14 of 14 pass (KS $p$ from 0.148 to 1.000, $\lvert z\rvert\le2.15$).

**L1**: 30 of 30 pass on $Y_k$. The lowest KS $p$ is 0.010, frozen with skip-read off at
$k=1024$ ($z=-2.40$), and the largest $\lvert z\rvert$ is 2.72, $\alpha=0.3$ with
skip-read off at $k=1024$. The 15 skip-read-off arms leaned low together (mean $z=-0.92$),
so they were rechecked at $50\times$ the sample size. None of it survived:

```
n=2e5, vs rwre_gpu at k=1024:   alpha=0.3,p=0.3 off  z=-0.36   frozen p=0.2 off  z=-0.05
                                p=1/3 off             z=-0.61   p=0 (k=256) off   z=-0.43
n=2e6, lazy only, k=1024:       p=1/2 off  E|S|=25.5233 vs exact 25.5261, z=-0.20
                                p=1/3      on 35.5326  off 35.5267  z=-0.23
                                alpha=0.3  on 134.0121 off 134.0370 z=+0.52
```

## L4: deque width and reveals (2026-10-02)

`local/rwre_lazy_l4.py widths`, $n=2^{14}$ per point, `cap` $=2^{16}$. No overflow.

| $p$ | $k$ | mean $W$ | $W/\sqrt k$ | max $W$ | max $\lvert L\rvert,R$ | reveals | $\overline{\lvert X\rvert}$ |
|---|---|---|---|---|---|---|---|
| 0 | $2^{8}$ | 138 | 8.61 | 225 | 128 | 45 | 22.7 |
| 0 | $2^{12}$ | 548 | 8.56 | 865 | 576 | 208 | 102.1 |
| 0 | $2^{16}$ | 2420 | 9.45 | 3809 | 2560 | 894 | 440.6 |
| 1/3 | $2^{8}$ | 128 | 8.00 | 225 | 128 | 32 | 16.0 |
| 1/3 | $2^{12}$ | 520 | 8.13 | 769 | 512 | 175 | 78.7 |
| 1/3 | $2^{16}$ | 2347 | 9.17 | 3393 | 2208 | 807 | 378.1 |
| 1/2 | any | 1 | — | 1 | 0 | 0 | srw |

- $W/\sqrt k$ rises slowly (local slope of $W$ ≈ 0.55 at the top), as the plan's
  $W\asymp k^{\max(\gamma,1/2)}$ predicts with $\gamma\approx0.55$–$0.58$.
- Reveals grow like $\sqrt k$ ($\approx3.1\sqrt k$ at $p=1/3$), a small fraction of the
  reads ($k/3$). Skip-read cut reads by $\lvert1-2p\rvert$, as designed.
- Extrapolated to $k=2^{20}$ with that slope, mean $W\approx1.1\times10^4$ and the largest
  half-span is $\approx1.1$–$1.3\times10^4$ cells. The buffer is centred, so the limit
  is $\mathtt{cap}/2$ per side: $2^{15}$ leaves only a ~1.3× margin there.

**Timing (P1, partial).** $k=2^{14}$, $n=2^{15}$, $p=1/3$, same seed so the same draws:

| `cap` | $2^{12}$ | $2^{13}$ | $2^{14}$ | $2^{15}$ | $2^{16}$ |
|---|---|---|---|---|---|
| seconds | 1.07 | 1.51 | 3.00 | 4.83 | 14.84 |

`growth` 1, 8, 32, 128, 256 at `cap` $2^{13}$: 1.56, 1.57, 1.51, 1.58, 1.72 s. **The
throughput is set by occupancy, not by `growth`**: each sample holds `cap` bytes of
shared memory for its whole life, so at $2^{15}$ only 3 warps share an SM. At
`cap` $=2^{15}$ a $k=2^{14}$ sample costs 147 µs. Scaled by
$k^{1.5}$–$k^{1.55}$, a $k=N=2^{20}$ rung is **≈ 22–27 h** in the byte layout. That
misses §9 Q6's "a few hours", so phase B (bit-packing: a quarter of the shared memory
per cell, so 4× the occupancy at the same `cap`, plus 32 cells per operation) is needed
for the top rung, as the plan anticipated.

**Open for sign-off (phase A'): the `cap` rule.** `cap` changes no draw, so it may depend
on $k$. Either keep the fixed $2^{15}$, or set it per $k$ from this table: the smallest
power of two at least $2\times1.5\times$ the largest half-span seen. That would be $2^{13}$
at $k=2^{14}$, 3.2× faster than $2^{15}$ there.
