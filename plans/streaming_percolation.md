# Plan — a streaming percolation sampler: $O(i^{d-1})$ memory instead of $O(i^d)$

**Status: §3 IMPLEMENTED as `models/percolation_zd_stream.py` (2026-09-09), signed off by
the user. §4 (the parallel divide and conquer) is still design only. §6's central claim —
that streaming cannot be bit-identical — turned out to be WRONG, and the correction is
recorded there rather than edited away.**
Proposed by Igor, 2026-09-06, after `autopilot` on `percolation_zd` died at the cost
probe with an 80 GiB refusal. Written up as a *new model* rather than a change to the
existing ones — see §6, which is the part that most needs agreement.

---

## 1. The problem, in the two places it bites

`models/percolation_zd.py` and `models/percolation_tau_zd.py` both materialize the whole
box: `_draw_open` allocates $i^{\mathrm{dim}}$ float32 uniforms, `_label_block` a boolean
copy and an int32 label image. About 10 bytes per site, and `block_n` bounds the *number
of samples* in flight but cannot make one sample smaller. So one sample is irreducible,
and $i^{\mathrm{dim}}$ grows fast:

| | sites | working set |
|---|---|---|
| `percolation_zd`, $d=3$, $i=256$ | $1.7\times10^{7}$ | 160 MiB |
| `percolation_zd`, $d=3$, $i=2048$ | $8.6\times10^{9}$ | **80 GiB** |
| `percolation_zd`, $d=5$, $i=128$ | $3.4\times10^{10}$ | **320 GiB** |
| `percolation_tau_zd`, $d=8$, $s=512$ ($L=10$) | $1.0\times10^{8}$ | 954 MiB |

**Bite 1 — the ladder ceiling.** `_MAX_SITES_PER_SAMPLE` refuses anything past
`ndimage.label`'s int32 label space, which is also roughly where the memory becomes
absurd. That caps the ladders at $i\le256$ ($d=3$), $64$ ($d=4$), $32$ ($d=5$), $16$
($d=6$) — 6, 5, 4 and 4 rungs. `experiments/05_percolation_highd/README.md` traces three
separate weaknesses to exactly this: only one $\omega_1$ estimator above $d=2$ (the
bias-decay fit needs 4 windows of $\ge4$ scales, i.e. $\ge7$ rungs), a cost probe with a
short lever arm, and a criticality check with no discriminating power at $d=6$.

**Bite 2 — the memory wall is also a speed wall.** At $d=8$, $s=512$ the working set is
954 MiB, so `block_n = 1` and every sample streams a gigabyte through cache. Measured
throughput on the dimension sweep and by direct timing:

| dim | 2 | 3 | 4 | 7 | 8 |
|---|---|---|---|---|---|
| Msites/s | 25.7 | 17.7 | 14.8 | 1.9–3.0 | 0.6–1.1 |

**$25\times$ slower per site at $d=8$ than at $d=2$.** Part of that is real work —
`ndimage.label`'s footprint is $2d+1$ cells — but the collapse between $d=4$ and $d=8$ is
steeper than the footprint grows, and it tracks the working set crossing cache. The
$d=8$ leg of the sweep is $\approx2.3$ hours of the $\approx3.8$ total.

**Bite 3, already fixed separately.** `tools/cost_model.climb_to_target` doubled the
scale blindly and propagated the model's refusal, killing a pilot that had already spent
nine minutes drawing replicates. A refusal is now a ceiling on the climb (`refused_at`),
not an error. That is a driver fix and is independent of everything below.

---

## 2. Igor's proposal, and the one thing wrong with it

> Given an $i\times i$ box, for $i$ even, we can divide in two parts, count the $N$ sites
> connected to the north axis, see all the south edges connected to the north one. Then
> repeat the process in the lower part $S$, by starting only with the higher up connected
> vertices. In principle we only need $(\mathrm{dim}-1)\times i$ memory.

The memory bound is right and is the whole point. The **greedy** version of the recursion
is not: connectivity is not causal in the sweep direction. A site in the upper part can
reach the anchor face *only by going down and coming back up*, and a sweep that decides
"connected / not connected" as it passes will miss it.

Minimal counterexample, $3\times3$, anchor $=$ row 0 (verified against the production
path, 2026-09-06):

```
1 0 0
1 0 1     <- (1,2) reaches row 0 only via (2,2) -> (2,1) -> (2,0) -> (1,0)
1 1 1
```

true face-connected count **6**, greedy forward sweep **5**.

**The repair costs nothing in memory**: do not *decide* connectivity slab by slab, carry
*unresolved* components. That is Hoshen–Kopelman with a frontier, §3.

---

## 3. The sequential algorithm — frontier sweep

Sweep slabs $x_0 = 0, 1, \dots, i-1$. Carry, at slab $h$:

- `front`: the labels of slab $h$, an array of $i^{\mathrm{dim}-1}$ int32;
- a union-find over **live** labels only, each root holding `size`, `touches_anchor`,
  and `live_members` (how many of its labels still appear in the frontier).

Per slab:

1. Label slab $h$ on its own — a $(\mathrm{dim}-1)$-dimensional `ndimage.label` with
   `generate_binary_structure(dim-1, 1)`, plus the transverse periodic wraps merged by
   the existing `_wrap_roots` pointer-jumping over labels.
2. Union across the interface: wherever slab $h-1$ and slab $h$ are both open at the same
   transverse position, union their labels. Vectorized, one pass over $i^{\mathrm{dim}-1}$.
3. Accumulate `size`; set `touches_anchor` on every root present in slab 0.
4. Decrement `live_members` for each label of slab $h-1$ that has no successor. When a
   **root's** `live_members` hits zero the whole component is dead: it can never grow or
   merge again, so its flag and size are final. Bank it and free it.

At the end, add the still-live components. Output is identical to labelling the whole box
— that is a test, not a claim (§5).

**Memory**: two frontiers plus a union-find whose live entries are bounded by the frontier
size, so $O(i^{\mathrm{dim}-1})$ — exactly Igor's bound:

| | full box | frontier | ratio |
|---|---|---|---|
| $d=3$, $i=2048$ | 80 GiB | **32 MiB** | $2560\times$ |
| $d=5$, $i=128$ | 320 GiB | **2 GiB** | $160\times$ |
| τ, $d=8$, $s=512$ | 954 MiB | **76 MiB** | $12\times$ |

**Why this is easier for the cluster-number density, not harder.** `percolation_tau_zd`
has no anchor: it needs the *size histogram*, and a component's size is final exactly when
it dies — so step 4 increments a log-bin instead of a total, and the output is
$O(\#\mathrm{rungs})$ whatever $L$ is. Nothing accumulates. Two wrinkles: `geometry =
"torus"` wraps the sweep axis too, so slab 0's labels must stay live to the end (one extra
frontier, still $O(L^{\mathrm{dim}-1})$); and the per-rung box rule is unchanged, so
`cost_hint` and every allocation stay exactly as they are.

**Why the anchored observables are cheaper still.** `face`, `face_far` and `slab` need
only a boolean per component, never a size — a running count suffices. `face` in
particular seeds on slab 0, which is where the sweep starts.

---

## 4. Igor's divide and conquer is the parallel form of the same thing

Split $x_0$ into $P$ blocks, label each **independently** (each worker running §3 inside
its block, so its memory is a frontier, not a block), then merge across the $P-1$
interfaces with a union-find over *interface labels only* — which is exactly what
`_wrap_roots` already does for the periodic faces, the same vectorized pointer-jumping,
applied to internal interfaces instead. Then propagate `touches_anchor` through the merged
forest and finalize.

Labelling is embarrassingly parallel; the merge is $O(P\,i^{\mathrm{dim}-1})$. This is the
right structure for the "parallel generation" item that has been open in `TODO.md` since
`experiments/03_percolation_zd`, and it composes with replicate-level fan-out rather than
competing with it (ground rule 2's spawned streams already make that safe).

**Recommended order: §3 first, §4 only if the clock still demands it.** Sequential
streaming already unblocks the ladder, and it is the half that is hard to get wrong once
tested; the parallel merge adds a second union-find with its own failure modes.

---

## 5. Verification — what has to be true before this is believed

The repo's own history says where the bugs are: `percolation2d`'s label merge exited too
early on a termination test that looked right, and it took a `block_n` invariance failure
to surface it. A streaming rewrite has strictly more of that kind of surface.

1. **Exact agreement with `percolation_zd` on the same explicit lattices** — not the same
   seed. Random critical lattices at $d = 2,3,4$, every anchor, every geometry, every
   slab height. This is the load-bearing test and it replaces bit-identity (§6).
2. **The independent flood fill** already in `tools/tests/test_percolation_zd.py`, which
   shares no machinery with either implementation.
3. **Exact enumeration** of $\mathbb EY_i$ at $(\mathrm{dim},i) = (2,3),(3,2),(4,2)$.
4. **Slab-height invariance**: sweeping one slab at a time and $k$ at a time must give
   identical answers — the streaming analogue of `block_n` invariance, and the test most
   likely to catch a premature finalization.
5. **The full size multiset on a 3-torus**, as `test_percolation_tau_zd.py` already does:
   counts alone hide a merge that splits one cluster into two.
6. **A deliberate premature-finalization test**: a lattice with a component that dies and
   revives across a slab (impossible if `live_members` is right, and the assertion is that
   the implementation never banks such a component).

---

## 6. The decision that needs sign-off: a NEW model, not a rewrite

> **CORRECTION (2026-09-09, after implementing §3).** The premise below is false, and it
> was falsified by the first thing the implementation tried to do. numpy fills
> `rng.random(size=(rows,) + (i,)*dim)` in C order, which is **sample-major and
> plane-minor**: sample 0's planes in sequence, then sample 1's. A sweep that processes
> **one sample at a time**, drawing it plane by plane, consumes exactly that sequence —
> so it *is* bit-identical, at every anchor, geometry, $i$ and $\mathrm{dim}$. Verified
> directly (`rng.random((rows,i,i))` equals per-sample per-plane draws) and pinned by
> `test_bit_identical_to_percolation_zd`.
>
> What the argument below actually establishes is narrower: a sweep that **batches
> samples** cannot be bit-identical, because batching interleaves their planes. The first
> draft did batch, which is how the wrong conclusion got written down. Not batching costs
> the per-sample overhead — which is why streaming is ~1.4× *slower* at small $i$ — and
> buys the seed equivalence, the `slab_h` invariance `models/README.md` demands of a
> working-set knob, and a far stronger test than §5.1's.
>
> The **conclusion** stands unchanged, for the reasons in the last two paragraphs of this
> section: a separate model, because the performance profiles are opposite and a model
> with recorded results should not change its runtime characteristics underneath them.
> Only the reason changes.

**Streaming cannot be bit-identical to the current models.** `_draw_open` consumes
`rng.random(size=(rows,) + (i,)*dim)` in one call; a slab-by-slab sweep consumes the same
stream in a different order and produces different lattices from the same seed. Every
recorded run's seed would stop reproducing its numbers.

Proposal: **`models/percolation_zd_stream.py` and `models/percolation_tau_zd_stream.py`,
registered alongside the existing models**, which stay the reference implementation.

- Every recorded run stays valid and reproducible.
- The equivalence is stated where it is actually true — same *observable*, same
  distribution, exact agreement *on a given lattice* — and is a test rather than an
  assumption.
- `cost_hint` is unchanged, so allocations, budgets and `cost_unit_ratio` all carry over,
  and a streaming recipe is comparable to a materialized one at equal budget.
- The two can be run head to head by `src/estimate/compare_observables.py` at equal cost,
  which is how every other design question in this repo has been settled.

The cost of a second model is duplicated machinery, which `models/README.md` already
accepts as the rule (no model imports another).

**Alternative considered and rejected:** restructure `_draw_open` so the materialized path
also draws slab by slab, making the two bit-identical. It would invalidate every recorded
seed for a cosmetic gain, and it would slow the small-$i$ path, which is where most
samples are drawn.

---

## 7. Open questions for sign-off

| # | question | my recommendation |
|---|---|---|
| Q1 | New models, or change the existing ones? | **New** (§6) — **DONE**, and the reason changed: bit-identity turned out to be achievable, so the separation is about performance profile, not reproducibility |
| Q2 | Which first? | ~~τ~~ → **`percolation_zd` first, done 2026-09-09.** The τ sweep that motivated putting it first (H8) has since finished, and the blocked path is now `percolation_zd`: it is what an `autopilot` run walked into, and measuring $d_f$ above $d_c$ needs it, since hyperscaling cannot supply $d_f$ from τ up there. τ is the follow-up |
| Q3 | Sequential streaming only, or the parallel merge too? | **Sequential first**, §4 as a separate follow-up |
| Q4 | Is a Python-level per-slab loop fast enough? | **MEASURED 2026-09-09: fast enough, but never faster — there is no crossover.** 0.64×, 0.67×, 0.67× the materialized speed at $\mathrm{dim}=3$, $i=256/512/1024$, and 0.67×/0.68× at $\mathrm{dim}=4$, $i=64/96$: flat, not converging. The predicted “large win where the box leaves cache” did not appear — the box leaves cache in *both* models, and streaming pays per-sample overhead on top. So this stays a **big-$i$ path, not a default**: the win is memory and reach ($200$–$790\times$, and $i=2048$ at $\mathrm{dim}=3$ runs in 607 s where materializing refuses at 80 GiB), never throughput |
| Q5 | Does `_MAX_SITES_PER_SAMPLE` still apply? | Yes, but against the **frontier**, not the box: int32 labels bound the labels alive at once, not the sites ever visited. The ceiling moves from $i^{\mathrm{dim}} < 2^{31}$ to $i^{\mathrm{dim}-1} < 2^{31}$ |
| Q6 | Do the ladders change once it exists? | Not automatically. Widening them is what unblocks the second $\omega_1$ estimator and the $d=6$ criticality check, but each is its own experiment with its own acceptance criteria |

**Nothing here is implemented.** The only thing already changed is the `climb_to_target`
ceiling in §1's Bite 3, which is a driver fix that stands on its own.
