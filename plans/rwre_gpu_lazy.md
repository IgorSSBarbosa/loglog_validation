# Plan — `rwre_gpu_lazy`: the walk on the SSEP with a lazily revealed environment

**Status: DESIGN ONLY, all §9 questions decided (2026-10-02). No code yet. To be
implemented on a machine with a GPU, phase by phase (§7).**
Proposed by Igor (the lazy environment came from the discussion with Roberto and Augusto;
the deque layout, the skip-read rule and the chunked-growth test are Igor's), with the
bit-packing and barrier proposals from the 2026-10-01 discussion. Written as a *new
model*, `MODELS["rwre_gpu_lazy"]`, and a new experiment, `experiments/18_rwre_gpu_lazy/`:
the random stream differs from `rwre_gpu`, so a seed recorded against one never
reproduces the other (same reasoning as `models/rwre_gpu.py`'s docstring).

---

## 1. The problem

Target: $k=2^{20}$ walker steps, $N\approx2^{20}$ samples per scale.

`models/rwre_gpu.py` simulates every site of a periodic window $W=2k+100$ (linear rule,
experiment 15) four sweeps per step. One sample is $4\cdot\tfrac W2\cdot k\approx4k^2$
bond decisions, and the measured device rate is $b\approx1.2$ ps per decision with the
card full (`experiments/13_rwre_gpu/README.md`, E1):

$$N\cdot4k^2\cdot1.2\,\text{ps}\approx2^{20}\cdot4.4\times10^{12}\cdot1.2\times10^{-12}\,\text{s}\approx64\text{ days}.$$

It also cannot run at all: the window lives in shared memory, so $k\le49094$ on the
linear window (`_MAX_W`). And the serial term $a=0.71$ ns per step (thread 0's read and
jump, the parity draw, 9 barriers per step) alone is $N k a\approx13$ minutes.

Faster per-site arithmetic alone cannot close this: even the bit-packed bound leaves
~13 h. The algorithm must do less work.

---

## 2. The lazy environment, and why it is exact

**Claim.** Conditional on everything the simulation has looked at — the walker path, the
values it has revealed, and the swap decisions of every bond that touched a revealed
site — the contents of the unrevealed sites are i.i.d. Bernoulli($\alpha$).

**Why.** `rwre.py` decision (a): swaps act on *sites*, chosen independently of the
configuration (the stirring construction). So the content of any site at time $t$ is the
initial value of the site at the end of its backward stirring path, and distinct sites
trace back to distinct initial sites. A site no one has read traces back to an initial
site whose value no one has observed: i.i.d. Bernoulli($\alpha$), independent of the
observed history. Swapping two unrevealed sites changes nothing observable, so it need
not be simulated.

**The rule.** Each site is in one of three states $\{0,1,U\}$, all $U$ at $t=0$.

- The walker needs $\eta_t(X_t)$ and the site is $U$: draw Bernoulli($\alpha$), the site
  becomes known.
- A bond with at least one known endpoint swaps with probability `swap_prob` (exactly
  `rwre`'s brick-wall sweep, parity drawn per sample per sweep, decision (a)).
- A bond with two $U$ endpoints: nothing to do. (Simulating it anyway is harmless, which
  the dense layout of §3 exploits.)
- Step order is `rwre`'s decision (b): read, jump, then `env_sweeps` sweeps.

**What it buys.** The known sites form an exclusion process fed by a source at the
walker. Near the source the density saturates towards 1 and fresh reveals become rare,
so the known region grows like the larger of the walker's spread and the diffusion of
revealed values (variance `env_sweeps * swap_prob` $=2$ per step at the defaults):

$$W(t)\asymp t^{\max(\gamma,\,1/2)},\qquad \text{cost}(k)=\sum_{t\le k}W(t)\asymp k^{1+\max(\gamma,1/2)}.$$

At $k=2^{20}$ that is roughly $\sqrt k/c\approx100$–$300\times$ less work than $2k^2$
(with $c\approx10$–$15$, to be measured, §6 L4).

**Two more consequences.** (i) The model is exact on $\mathbb Z$: there is no window
and no seam, so the wrap heuristics of `rwre.py` decision (d) and criterion A4 disappear.
(ii) The cost exponent is no longer known by construction (see §9 Q2).

### 2.1 Skip-read (Igor's idea 3)

With $u$ the walker's uniform, $P(\text{left})=p$ on a particle and $1-p$ on a hole. If
$u<\min(p,1-p)$ the walker goes left, and if $u\ge\max(p,1-p)$ it goes right, *whatever
the site holds*. Only on $[\min(p,1-p),\max(p,1-p))$, probability $\lvert1-2p\rvert$, is
the site read. A site not read is not revealed.

Exact, because the decision to read depends on $u$ alone, never on the value. On its own
it saves one shared-memory read per step (nothing). Combined with §2 it lowers the reveal
rate to $\lvert1-2p\rvert$ of the steps, shrinking the known region. At $p=1/2$ nothing
is ever revealed and the model *is* `srw` at zero environment cost — a free test (L2).

---

## 3. Data layout: Igor's deque

There is no window edge. Per sample, the environment is a **deque** covering
$[L,R]$. Cells inside may be $U$; cells outside are $U$ by definition.

**Invariant (Igor, corrected 2026-10-02): both end cells are always $U$.** The first and
last cells are never revealed sites. There is always at least one unrevealed cell beyond
the outermost known site on each side, so every bond that can move a known value is
inside the deque, and the walker never "feels" a finite environment. The walker itself
need not be inside the deque: outside it everything is $U$, and with skip-read it may
walk there without reading.

**Coordinates.** Absolute coordinate of deque index $j$ is $j-\text{off}$, where `off`
is how many cells were ever pushed on the left (Igor: "track how many left/right sites
were discovered"). The walker is stored in absolute coordinates; the origin never moves.
Bond parity is $x\bmod2$ in **absolute** coordinates, so it does not depend on the
layout.

**On a GPU a deque is a fixed buffer with two moving ends.** Shared memory is allocated
once per sample (capacity `cap` cells), the origin starts at the centre, and push-front /
push-back are $O(1)$ index moves into pre-zeroed ($U$) cells. Nothing is ever shifted.

**When the deque grows (always by pushing a chunk of $U$ cells).**
1. A sweep swaps a known value into an end cell: push a chunk of $U$ cells on that side
   before the next sweep. A known value moves at most one site per sweep, so a single $U$
   end cell is enough to make each sweep exact; checking once after every sweep restores
   the invariant in time.
2. The walker reads an end cell, or a site outside $[L,R]$: push chunks until the site
   is strictly inside, then reveal it. The new end cell is again $U$.

Both checks are local (look at the two end cells), so they cost $O(1)$ per sweep.

**Growth granularity $G$ (Igor's chunk test).** Grow by $G$ cells at a time,
$G\in\{1,32,256,\ldots\}$. Larger $G$ means fewer growth checks and more $U$ cells swept
for nothing. With the RNG of §4 keyed by absolute coordinates, **the output is
bit-identical for every $G$**, so the test is a pure timing comparison with a hard
correctness gate (L0). In the bit-packed layout (§5) the natural minimum is $G=32$ (one
word); $G=1$ is only meaningful in the byte layout.

**No shrinking.** If the outermost known values move inward, $U$ cells pile up at the
ends; they are kept (§9 Q5).

**Overflow.** If a sample needs more than `cap` cells it sets a flag. It may then be
rerun on a slow path (global memory, same kernel logic): because the RNG is keyed by
absolute coordinates and not by buffer layout, **the rerun is the same sample path**, so
overflow handling introduces no conditioning bias. Discarding the sample instead would.
Whether to build that path or simply raise is §9 Q4 (recommendation: raise in v1,
since overflow is not expected, §9.1).

---

## 4. The random stream: counter-based, keyed by meaning

Every draw is Philox4x32-10 at a counter that names *what* it is for, never *which
thread* or *which buffer slot* draws it:

| draw | key |
|---|---|
| parity of sweep $s$ | (seed, sample, $s$) |
| swap bits of sweep $s$, bonds $32w\ldots32w+31$ (absolute) | (seed, sample, $s$, $w$) |
| walker uniform at step $t$ | (seed, sample, $t$) |
| revealed value at step $t$ | (seed, sample, $t$) |

So the output is a pure function of (seed, sample, $k$, params). It does **not** depend
on launch chunking, $G$, `cap`, overflow reruns, byte vs bit layout, or threads per
sample. That gives the strongest test in the plan (L0). Every lane computes the parity
itself, so the parity draw needs no thread 0 and no barrier (§5).

`swap_prob` other than 1/2 needs more than one random bit per bond (bit-sliced Bernoulli:
compare a few random words against the binary digits of $q$). See §9 Q1.

---

## 5. Kernel design

**Two layouts, the first is the reference for the second.**

- **Byte layout** (phase A): one byte per cell, values $\{0,1,U\}$. Simple, the
  readable reference, and the only layout where $G=1$ makes sense.
- **Bit-packed layout** (phase B): two bit-planes, `known` and `value`, 32 cells per
  word. A sweep applies the same swap mask to both planes with the masked XOR

  $$t=(x\oplus(x\gg1))\,\&\,m\,\&\,\texttt{0x5555\ldots},\qquad x\leftarrow x\oplus(t\mid(t\ll1)),$$

  plus the carry across word boundaries for odd parity. Must be **bit-identical** to the
  byte layout.

**One warp per sample.** At $k=2^{20}$ the deque is about $10^4$ cells, a few hundred
words per plane: ~10–20 words per lane. Barriers become `__syncwarp` instead of
`__syncthreads`, and many samples share an SM. Combined with lane-local parity (§4) this
attacks the $a$ term, which after §2 and bit-packing is the same size as all the
remaining work.

**What is deliberately not in the plan.**

- *Deterministic alternating parity* (Igor's idea 2). It changes the law: $X_t\equiv
  t\pmod2$, so a parity tied to $t$ couples the walker's site to the active bonds
  (`rwre.py`, decision (a)). Its only speed gain, dropping the parity draw and barrier,
  comes from lane-local parity with the law unchanged.
- *Precompute the environment in a cone, then walk* (Igor's idea 4). The space-time field
  of one sample at $k=2^{20}$ is $\approx2k^2$ bits $=256$ GB; the cone saves at most
  $2\times$; and it computes the whole environment, while §2 wins precisely by letting
  the walker decide what is computed. Streaming it in time chunks reduces to the current
  kernel minus one barrier.
- *Multi-block per sample / temporal blocking.* Unnecessary once the deque fits one
  warp's slice of shared memory.

---

## 6. Acceptance criteria

**L0. Bit-identity (hard, exact).** Same seed gives identical output across: launch
chunking and prefix in $N$; $G\in\{1,32,256\}$ (byte) and $\{32,256\}$ (bit); forced
overflow (tiny `cap`) vs no overflow; byte vs bit layout.

**L1a. Smoke KS against the first CPU model, `MODELS["rwre"]` (few minutes).** A quick
sanity check, run first once phase A builds and again after every kernel change (also
phase B). A local script, outside git like 13's checks. Signed $X_k$ (not $\lvert
X_k\rvert$, so the reflection symmetry is tested too), $n=4000$ per arm, independent
spawned seeds, `rwre` at its default window $12\lceil\sqrt k\rceil$ (the wrap bound
$\approx1.5\times10^{-8}$ is negligible at these $k$):

| arms | $k$ | `rwre` CPU time (13's ~20–30 ns per step·site) |
|---|---|---|
| $p=1/3$; $p=0$; $p=1/2$; frozen `env_sweeps=0` at $p=0.2$; $\alpha=0.3$ at $p=0.3$; $p=1/3$ skip-read off | 16, 256 | ~5 s per arm |
| $p=1/3$; $p=0$ | 1024 | ~50 s per arm |

About 14 tests, ~3 minutes in all, dominated by the CPU side. PASS: two-sample KS
$p\ge0.01$ and $\lvert\bar X_{\rm lazy}-\bar X_{\rm rwre}\rvert\le3$ combined se in
every arm. With 14 tests at level 0.01 a single failure happens by chance ~13% of the
time: one failing arm is rerun with fresh seeds; a second failure in the same arm stops
the phase. (KS on integer-valued data is conservative, so it errs towards passing; L1
and L2 carry the power.)

**L1. Distribution against `rwre_gpu` / `rwre`.** $k\in\{64,1024,16384\}$, the five
parameter sets of 13's G1 (including `env_sweeps=0` and $\alpha=0.3$), plus skip-read on
and off. $n=4000$ per arm, independent seeds: two-sample KS $p\ge0.01$ and
$\lvert\bar Y_{\rm lazy}-\bar Y_{\rm ref}\rvert\le3$ combined se. The reference uses the
linear window $W=2k+100$, whose torus error is $\lesssim e^{-50}$ (`rwre.py`, (d)).

**L2. Exact references.** 13's G2 battery: $k=1$ law of $X_1$, frozen $k=2$, the
zero-check $\lvert\mathbb EX_k\rvert<4$ se, $\lvert X_k\rvert$ same law at $p$ and
$1-p$, the closed cases $\alpha\in\{0,1\}$. Plus: at $p=1/2$ with skip-read, **zero
reveals** and the Binomial law of $S_k$ ($\chi^2$, $p>10^{-3}$).

**L3. Lazy-specific invariants.**
- Conservation (hard assertion, every step in debug builds): popcount(`known`) equals the
  number of reveals.
- Values: revealed values are Bernoulli($\alpha$) independent of whether and when they
  were revealed, and swaps only move them, so the fraction of ones among known cells is
  Binomial(reveals, $\alpha$) / reveals exactly; test within 4 se pooled over samples.
- Deque (hard assertion after every sweep and every read): both end cells are $U$, so
  every known site lies strictly inside $(L,R)$.

**L4. Measurements that fix the budget (not pass/fail).** $W(t)$ (deque width), reveal
count $R(t)$, and the overflow rate, at $k=2^{8}\ldots2^{16}$ for $p\in\{0,1/3,1/2\}$,
fitted and extrapolated to $k=2^{20}$. These set `cap` and decide whether $W(t)\sim
t^{\max(\gamma,1/2)}$ holds.

**P1. Performance.** Throughput per sample against `rwre_gpu` at matched $k$; $G$
sweep; byte vs bit; warp vs block per sample. PASS: the full $k=N=2^{20}$ ladder top
fits the experiment's time budget (a few hours per scale at first, §9 Q6).

**E1. Cost exponent.** `measure_cost.py` on the new model: measured $d$ against
$1+\max(\gamma,1/2)$, using 13/15's measured $\gamma$ for each $p$.

---

## 7. Phases

| phase | content | gate |
|---|---|---|
| A | byte-layout kernel, one warp per sample, keyed RNG, $G$ parameter, `cap` with raise on overflow, skip-read flag (no trimming) | L1a first, then L0 (except bit vs byte), L1, L2, L3 |
| A' | L4 measurements → choose `cap`, default $G$ | sign-off on `cap` |
| B | bit-packed kernel | L1a, L0 bit vs byte, P1 |
| C | registration in `tools/models.py`, `cost_hint`, `max_steps` (none: no window) | E1 |

The science run (the $\gamma$ ladder to $k=2^{20}$ per $p$) is a separate experiment,
designed after C.

---

## 8. Budget estimate (assumes $W(t)\approx12\sqrt t$; L4 replaces it)

| version | $k=N=2^{20}$ |
|---|---|
| `rwre_gpu` today, full window | ~64 days (and cannot run past $k\approx49094$) |
| + lazy, same per-bond cost | ~5 h work + 13 min serial |
| + bit-packed | ~10 min work + 13 min serial |
| + warp per sample, lane-local parity | ~10–20 min; realistically under an hour |

---

## 9. Questions

### Decided (Igor, 2026-10-02)

1. **`swap_prob`: only 1/2 in the bit-packed kernel first.** One random bit per bond.
   General $q$ via bit-slicing later, if a study needs it; the byte layout may take
   general $q$.
2. **Cost exponent: fit $d$ only.** `cost_hint` declares no exact exponent; the cost
   model fits $d$ on the ladder, and E1 scores it against $1+\max(\gamma,1/2)$
   (`env_sweeps` $>0$: revealed values diffuse like $\sqrt t$ even when the walker is
   subdiffusive; frozen environment: $d\approx1$, only the range is revealed).
3. **Skip-read: on by default, with an off switch** so L1 can compare both arms.
4. **Overflow: raise in v1.** `cap` is a parameter; a sample exceeding it raises with
   its index. The global-memory rerun (§3) stays designed but unbuilt until an overflow
   is actually seen (§9.1: not expected).
5. **No trimming.** The deque never shrinks. Trimming has no physical meaning, needs a
   test that a whole chunk is unrevealed, and a single known value dancing near an end
   would cause repeated insert/delete churn costing more than the $U$ cells it saves.
   L4 records how many $U$ cells accumulate at the ends, so the cost is measured.
6. **Time budget: "the best we can", a few hours per $k=N=2^{20}$ scale at first.** So
   the plan is implemented and tested in parts: phase A (byte layout + lazy, ~5 h
   estimated) may already meet it; phase B (bit-packing) is the next step, not a
   prerequisite.
7. **Naming, following the repository pattern:** experiment
   `experiments/18_rwre_gpu_lazy/`, model `models/rwre_gpu_lazy.py`,
   `MODELS["rwre_gpu_lazy"]`.

### 9.1 Memory at $k=N=2^{20}$ (RTX 5090 specs; to be checked against L4)

| resource | size |
|---|---|
| global memory (GDDR7) | 32 GB |
| L2 | 96 MB |
| shared memory | ~100 KB per SM (99 KB opt-in per block), 170 SMs, ~17 MB in all |

Expected deque width at $k=2^{20}$: revealed values diffuse with std
$\sqrt{2k}\approx1450$ sites, the outermost of many reaching ~5–6 std each side, and the
walker spreads to a few thousand sites ($\gamma\approx0.58$). So **~$10^4$–$3\times10^4$
cells**: 2.5–8 KB per sample bit-packed (2 bits per cell), 10–30 KB in the byte layout.

- A capacity of $2^{16}$ cells is 16 KB bit-packed, 2–6× the expected width. That still
  fits ~6 samples per SM (~1000 in flight), enough to fill the card. In the **byte**
  layout (phase A) the same capacity is 64 KB, one sample per SM: phase A should start
  at `cap` $=2^{15}$ (32 KB, 3 per SM) and let L4 say whether that is enough.
- Hard worst case: the deque cannot outgrow the light cone, $\le2\cdot4k+1\approx8.4\times
  10^6$ cells = 2 MB bit-packed. Even that fits in global memory for thousands of samples
  at once, and **all $2^{20}$ samples at the typical width fit in global memory
  simultaneously** (~8 GB).
- Overflow at 2–6× the typical width is a far Gaussian tail; with $N=2^{20}$ samples it
  must have probability $\ll10^{-6}$ per sample, which L4 measures (the tail of the
  deque width, extrapolated) rather than assumes.

**Hence Q4: memory is not the bottleneck; raise in v1.** Make `cap` a
parameter, count the maximum width in L4, raise with the sample index if it is ever
exceeded. The global-memory rerun stays designed (§3) but unbuilt until an overflow is
actually seen.
