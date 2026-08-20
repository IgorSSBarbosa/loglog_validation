# Plan — three-experiment ladder: measure $d$, then $\omega_1$, then $\gamma$

Status: **decisions signed off 2026-08-20; §1 (shared prerequisites) and §3 (Experiment B)
implemented, run and PASSED. §4 (Experiment C) implemented and run: the RATE claim
passes, the POINT claim fails by a constant offset of ~3 in $m_0$ (~2.2-2.4x RMSE).
Experiment A not started (and may be redundant — see §1c).**

Experiment C result: measured error decay $-0.364$ against the predicted $-1/3$ (passes),
but `prop:opt`'s $m_0$ is ~3 too high at every budget, costing $2.18$-$2.39\times$ in
RMSE. At the empirical optimum $|\mathrm{bias}|/\mathrm{sd}\approx0.6$-$1.7$ as the
theorem's own balancing argument implies; at the $m_0$ it names, $0.09$-$0.30$ -- it
over-corrects for bias. Confirmed independently by an exact/analytic calculation that
draws no samples. Full write-up in `experiments/01_srw/README.md`.

Experiment B result: $\omega_1 = 1.0155 \pm 0.1050$, $\gamma = 0.5000 \pm 0.0003$,
$a_1 = -0.2748 \pm 0.0597$, $a_0 = 0.7979 \pm 0.0017$ over 5 independent replicates --
all four within half a standard error of the known truth. Full write-up and reproduction
steps in `experiments/01_srw/README.md`.

| decision | resolution (user, 2026-08-20) |
|---|---|
| D1 — write the closed form into `article.tex`? | **No.** Derivations of the first few correction-to-scaling exponents already exist elsewhere in the user's own notes; the article is left untouched, and the experiments do not consume these values as input. |
| D2 — add `target_fn`/`true_gamma_key` to `MODELS["srw"]`? | **No.** README-only acceptance criteria instead: the known $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$ are stated in `experiments/01_srw/README.md` and checked by hand when an experiment finishes, so the known answer never enters the code path the estimators run through. |
| D3 — default timing aggregator | **`median`**, with `min`/`mean`/`q95`/`iqmean` also available. |

Motivation (user, 2026-08-20): `larger_test`'s `estimates.png` is too noisy to show the
expected $\hat\gamma_i \approx \gamma + a\,i^{-\omega_1}$ decay. Getting enough samples
forces small scales; but at small scales the measured cost is *not* a power law
(`cost_probe`, `time_measure`), so the budget-allocation theorem cannot be applied.
The resolution is to split into three experiments, each measuring one thing.

---

## 0. Finding that reshapes this plan — needs sign-off

`experiments/01_srw/README.md` currently says the SRW gamma-estimation ladder is
"blocked on design" because `appendix-SimpleRandomWalk` in the article is an empty
stub, so there is no closed-form $\mathbb{E} Y_i$ to check against. **For
$Y_k=\lvert S_k\rvert$ there is an exact one**, verified here two independent ways:

$$\mathbb{E}\lvert S_k\rvert \;=\; k\,\binom{k-1}{\lfloor (k-1)/2\rfloor}\,2^{-(k-1)}$$

- Exhaustive check against $\mathbb{E}\lvert S_k\rvert = 2^{-k}\sum_j \binom{k}{j}\lvert 2j-k\rvert$
  for every $k=1,\dots,200$: exact agreement to machine precision.
- Monte Carlo ($2\times10^6$ samples) at $k\in\{1,2,3,4,5,8,9,16,32,100,101\}$:
  agreement within $1.6$ standard errors everywhere.

Its asymptotic expansion (verified numerically: $k(\text{ratio}-1)\to-0.25$ to 7 digits
by $k=10^6$) is

$$\mathbb{E}\lvert S_k\rvert=\sqrt{\tfrac{2k}{\pi}}\Bigl(1-\tfrac{1}{4k}+O(k^{-2})\Bigr)
=\sqrt{\tfrac{2}{\pi}}\;k^{1/2}\exp\Bigl(-\tfrac14 k^{-1}+O(k^{-2})\Bigr),$$

which is **exactly the article's Assumption 1 form, eq. (232)**
$\mathbb{E} Y_i=a_0 i^\gamma\exp(a_1 i^{-\omega_1}+\cdots)$, with

| object | value |
|---|---|
| $a_0$ | $\sqrt{2/\pi}\approx0.797885$ |
| $\gamma$ | $1/2$ (exact) |
| $\omega_1$ | $1$ (exact) |
| $a_1$ | $-1/4$ (exact) |

Pushing the expansion one order further (numerically, to 6 digits) gives something
unusually convenient for this project:

$$\log\frac{\mathbb{E}\lvert S_k\rvert}{\sqrt{2/\pi}\;k^{1/2}}
=-\frac{1}{4}k^{-1}+\frac{1}{24}k^{-3}+O(k^{-5}).$$

The $k^{-2}$ term **cancels identically** (verified: $k^2\times$ residual $\to0$, while
$k^3\times$ residual $\to 0.0416667=1/24$). So in eq. (232)'s notation the second
correction sits at $\omega_2=3$, not $2$ — three full orders below $\omega_1=1$.
Contamination of an $\omega_1$ estimate by the next term is therefore negligible over
any usable scale range, making this an unusually clean testbed for Experiment B:
$\omega_1$ is close to exactly identifiable rather than confounded.

**Why this matters:** it converts Experiment B from "estimate an unknown $\omega_1$"
into "estimate a *known* $\omega_1=1$" — i.e. a genuine validation checkpoint with a
numeric acceptance criterion, exactly what ground rule 1 asks for, instead of an
exploratory number nothing checks. It also unblocks the ladder in
`experiments/01_srw/README.md`.

**Decisions needed before any code:**

- **(D1)** Ground rule 4 says formulas come from the article by equation number, and
  this one does not — it is derived here. Do you want it (a) written into
  `appendix-SimpleRandomWalk` in `article_writting/article.tex` first, then used, or
  (b) used here immediately with a `derivations/srw_exact_mean.tex` note, article
  updated later? This is a paper-side decision, so I have not touched either repo.
- **(D2)** Adding `target_fn` + `true_gamma_key="gamma"` to `MODELS["srw"]` is what
  makes `plot_loglog.py` overlay the reference curve and stop printing the
  "exploratory" disclaimer. Confirm you want that switched on.
- **(D3)** Timing aggregator default: `median` or `iqmean`? You said either. Both get
  implemented; the question is only which one `measure_cost.py` uses by default.
  Recommendation: **`median`** — it has a clean distribution-free confidence interval
  (order statistics), so the cost curve gets honest error bars, which `iqmean` does not
  give as cheaply.

---

## 1. Shared code changes (prerequisites for all three experiments) — DONE

### 1a. `models/srw.py` — 4.4× faster, same cost structure — DONE

Replaced `rng.choice(_STEP_VALUES, size=(block, k), p=[1-q, q])` with a
`float32`-uniform draw, `rng.random(size=(rows,k), dtype=np.float32) < q`.
Benchmarked at $k=1024$, $n=2\times10^5$:

| variant | block-invariant | µs/sample | bytes/step |
|---|---|---|---|
| `choice(..., p=)` (old) | yes | 28.30 | 1 |
| `integers(0,2, dtype=int8)` | **no** | 5.16 | 1 |
| `integers(0,2, dtype=int64)` | yes | 7.65 | 8 |
| **`random(dtype=float32) < q`** (chosen) | **yes** | **6.63** | 4 |
| `random(dtype=float64) < q` | yes | 8.37 | 8 |

**The int8 integer draw was tried first and rejected**, despite being the fastest
option: numpy packs several small integers per 64-bit draw and *discards the leftover
bits at the end of each call*, so splitting rows into blocks consumes the bit stream
differently from one unblocked call and silently changes the output (reproduced at
$k=7$, $n=50$, `block_n=3`). That would have broken both `srw`'s own `block_n`
invariance and `src/generate.py`'s chunked path, which depends on it — i.e. it would
have quietly undone the OOM fix's correctness guarantee. A float32 uniform is one draw
per step with no packing, so row-blocking stays exact. float32 costs nothing
statistically here: 24 random mantissa bits, and for $q=1/2$ the comparison is an
exactly fair coin.

Verified after the switch: `block_n` invariance exact at five $(k,n,\text{block})$
combinations; sample means match the exact $\mathbb{E}\lvert S_k\rvert$ within
$\lvert z\rvert\le2.5$ at nine scales; parity and $[0,k]$ bounds hold; asymmetric
$q\in\{0.3,0.7\}$ matches an independent binomial reference.

Deliberately **not** adopted: the exact closed-form sampler
$\lvert S_k\rvert = \lvert 2\,\mathrm{Bin}(k,q)-k\rvert$, which is 375× faster and
distributionally identical (verified: identical support, total-variation distance
$\approx10^{-3}\sim n^{-1/2}$). Rejected on the user's instruction (2026-08-20)
because it samples every scale in ~constant time, destroying the $\Theta(k)$ cost
structure that makes `srw` a stand-in for a real percolation simulator — and with it,
the entire point of Experiments A and C. **The `integers` variant keeps cost
$\Theta(nk)$, so $d=1$ is preserved; only the constant shrinks.**

For $q\neq\tfrac12$ `integers` does not apply directly; keep a `rng.random(...) < q`
path for that case (still $\Theta(nk)$, still no `p=` slow path).

Note this **breaks bit-reproducibility of existing runs** (different RNG consumption).
`test_run`/`larger_test`/`cost_probe`/`time_measure` would need regenerating, or the
old path kept behind a `method=` flag. Recommend regenerating — they are all cheap
except `Huge_test`, which was never completed anyway.

### 1b. `tools/cost_model.py` — aggregator registry — DONE

`AGGREGATORS = {"min", "median", "mean", "q95", "iqmean"}`, mirroring the existing
`COST_ESTIMATORS` name-keyed-registry pattern, selectable per-recipe via an
`"aggregator"` key, defaulting to `median`. Measured on the existing runs the choice
barely moves $\hat d$ (min 0.884 / median 0.888 / iqmean 0.887 / mean 0.850 / q95 0.811
on `cost_probe`) — this is for honest error bars and for the user's stated preference,
not to change the estimate. `median_ci` supplies the distribution-free interval from
order statistics that motivated preferring `median` over `min`; verified by empirical
coverage (95.6% against a nominal 95% at $N=31,301$) rather than by a single-draw
bracket assertion, which would fail 5% of the time by construction.

### 1c. `tools/cost_model.py` — affine-plus-power fit — DONE

The real defect is **model misspecification, not aggregation**: over $k=2\dots1024$ the
measured cost is affine, $\approx 22\,\mu s + 0.025\,\mu s\cdot k$, so a pure power-law
fit returns $\hat d\approx0.10$ — meaningless. `estimate_cost_affine` fits
$\mathrm{cost}(i)=a+b\,i^d$ in log space and reports the overhead $a$ as an explicit
diagnostic.

This turns out to rescue the small-scale regime completely, which is the crux of the
original conundrum — small scales are where the samples are affordable, and they were
exactly where the cost model was failing:

| run | pure-power $\hat d$ | affine $\hat d$ | fitted overhead $a$ |
|---|---|---|---|
| `time_measure` ($k=2\dots1024$) | 0.103 | **0.948** | 23.7 µs |
| `cost_probe` ($k=256\dots10^6$), re-run after 1a | 0.771 | **1.006** | 11.2 µs |

Three independent routes now agree on $d=1$ for `srw`: the affine fit (1.0063), the
`drop_leading` local-slope limit (0.9980 at $m_0=5$), and the known $\Theta(k)$ ground
truth. Acceptance in `src/measure_cost.py` is therefore checked against the affine
$\hat d$ whenever the fit is available.

**Consequence for Experiment A:** the batched/amortized measurement below may now be
redundant — the affine fit already recovers $d$ correctly from $n=1$ timings. Worth
confirming rather than assuming; the two are independent corrections (batching removes
the overhead physically, the affine fit models it statistically), so agreement between
them would be a genuine cross-check.

---

## 2. Experiment A — the cost exponent $d$

**Question:** what is $d$ in Assumption 7, $\mathrm{cost}(i)=i^d$, for `srw`?

**Key change:** measure the **amortized, batched** per-sample cost, not
`simulate(i, n=1, ...)`. The current $n=1$ probe is dominated by a fixed $\approx22\,\mu s$
Python/NumPy call overhead — at $k=2$ essentially 100% of the measurement is overhead,
at $k=1024$ still ~46%. That overhead is an artifact of *how the probe calls the
simulator*, not of the simulation, and it amortizes to nothing in real generation,
where a whole scale is drawn in one call. Allocating a budget from the $n=1$ number
would be allocating against the wrong cost function.

So: time `simulate(i, n, ...)` at a moderate fixed $n$ (e.g. $n=10^4$) and report
elapsed$/n$. Prediction: this recovers $d\approx1$ cleanly across the *full* scale
range, including the small scales where the $n=1$ probe fails — which is precisely
what makes Experiment B's small-scale regime usable in Experiment C.

- Recipe: `experiments/01_srw/cost_amortized.json`, scales $2\dots65536$ (log grid),
  moderate repeats.
- Report: $\hat d$ under pure-power and affine fits, per-scale median with CI, and the
  fitted overhead $a$.
- **Acceptance:** $\hat d\in[0.9,1.1]$ over the full grid under the pure-power fit
  (tighter than the current $[0.8,1.2]$, justified because amortization should remove
  the overhead that forced the loose bound).
- Keep the existing $n=1$ probe as a separately-tagged *overhead* measurement. It is
  not wrong, it just answers a different question; label it as such rather than
  feeding it to `allocation.py`.

## 3. Experiment B — the correction-to-scaling exponent $\omega_1$ — DONE, PASSED

**Question:** estimate $\omega_1$, with known truth $\omega_1=1$ (§0).

Many samples, small scales — the regime where the correction term
$a_1 i^{-\omega_1}$ is large enough to see. Two estimators, cross-checked:

1. **Direct nonlinear fit** of $\log\overline Y_i$ against
   $\log a_0+\gamma\log i+a_1 i^{-\omega_1}$ over all scales, $(\gamma,\omega_1,a_0,a_1)$
   free. Recovers $\omega_1$ directly. New: `tools/correction.py`.
2. **Decay-of-bias fit** — the shape you originally wanted to see: compute
   $\hat\gamma_{(i)}$ from `gamma_drop_leading` as a function of the smallest scale $i$
   retained, then fit $\hat\gamma_{(i)}-\gamma \approx a\,i^{-\omega_1}$. This is
   estimator-specific (it measures the bias decay of *that* estimator) and is the one
   that directly justifies the $m_0$ choice in Experiment C.

### Sample allocation — the Neyman proposal was wrong, corrected to `snr`

The original proposal here was Neyman-style $n(i)\propto1/\sqrt{c(i)}$, on the
reasoning that it equalizes marginal variance reduction per unit cost and keeps the
cheap small scales "where the $\omega_1$ signal lives". **Running it showed that
reasoning is wrong**, and the fix matters more than any estimator detail:

Neyman minimizes the variance of $\overline Y_i$. But $\omega_1$ is not estimated from
$\overline Y_i$ — it is estimated from the *small correction* $a_1i^{-\omega_1}$ riding
on top of it, whose size **shrinks with $i$**. Equalizing the error of $\overline Y_i$
therefore over-samples exactly where the correction is already resolved hundreds of
times over, and starves the scales where it has sunk beneath the noise. Measured on the
first run (Neyman, $B=2\times10^9$, scales $2..1024$), the correction-term SNR
$\lvert a_1\rvert i^{-\omega_1}/\mathrm{sd}(\log\overline Y_i)$ ran:

| $k$ | 2 | 8 | 32 | 128 | 512 | 1024 |
|---|---|---|---|---|---|---|
| SNR | 460 | 98 | 18 | 3.3 | 0.59 | 0.25 |

Three orders of magnitude of imbalance; the large scales contributed nothing but noise
to the fit. Writing $\mathrm{sd}(\log\overline Y_i)=s_i/\sqrt{n_i}$ and holding the
correction's SNR constant across scales gives instead

$$n_i \;\propto\; s_i^2\, i^{2\omega_1},$$

i.e. for $\omega_1=1$, $n_i\propto i^{2}$ — **increasing** in $i$, the opposite of
Neyman's $i^{-1/2}$. Implemented as `tools/allocation.py`'s `snr_allocation`
(`"rule": "snr"` in a recipe, taking an extra `omega1` design input); verified to
flatten the predicted SNR to a constant across the grid. `neyman_allocation` is kept,
tested, and documented as the wrong tool for *this* job — a test asserts the two rules
trend in opposite directions, so they can never be silently conflated.

### The scale window is itself a bias–variance tradeoff

Fitting the one-correction truncation to the **exact** $\mathbb{E}\lvert S_k\rvert$
(zero sampling noise) isolates the truncation bias from the neglected $\omega_2=3$
term, and it caps what any amount of sampling can achieve:

| smallest scale kept | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| $\hat\omega_1$ ceiling | 0.959 | 0.987 | 0.996 | 0.999 | 1.000 |

At $k=2$ the $\omega_2$ term is 4.2% of the $\omega_1$ term, which is exactly the ~4%
deficit observed. So small scales must be dropped for accuracy — but dropping them
lowers the correction signal, which is why the allocation has to compensate by adding
samples at the larger scales. The recipe therefore uses scales $8\dots256$ (ceiling
$\approx0.995$) with the `snr` rule, rather than the wider, cheaper grid.

- **Acceptance:** the direct fit returns $\hat\omega_1\in[0.85,1.15]$, $\hat\gamma$
  within $0.01$ of $0.5$, and $\hat a_1$ within 10% of $-0.25$. The bias-decay
  estimator is held to a *looser* $\hat\omega_1\in[0.7,1.3]$ and is treated as a
  corroborating cross-check, not a co-equal measurement: it fits the convergence of
  nested OLS windows over shared samples, whose bias is a weighted mixture of
  correction terms rather than a clean $i^{-\omega_1}$, so it is expected to be the
  less accurate of the two. Disagreement beyond ~0.25 between them is the signal that
  the one-correction model does not describe the run.
- Report $\hat\omega_1$ **with the truncation ceiling for the chosen window alongside
  it** (see the table above), since the ceiling — not the sample count — is what
  bounds accuracy once the allocation is right.
- Independence rule (ground rule 2): the samples used here are a *separate draw* from
  Experiment C's — no reuse, no slicing.

## 4. Experiment C — $\gamma$ at maximum accuracy under an optimal budget — DONE

### The control arm, generalized: sweep $m_0$ instead of picking one alternative

The original design pitted `prop:opt`'s allocation against a single flat-$n$ arm. That
is weaker than it needs to be, because `prop:opt` *already* uses a uniform $n$ — the
thing it chooses is $m_0$, the offset of the scale window. So the honest control is not
"one other allocation" but **every other $m_0$ at the same budget**: sweep $m_0$, measure
RMSE of $\hat\gamma$ at each, and ask whether the theorem's $m_0$ is the argmin.

That also separates two claims which can come apart, and which a single-point
measurement could never distinguish:

- **RATE** — does the error fall like $B^{-\omega_1/(d+2\omega_1)}$?
- **POINT** — is the specific $m_0$ the theorem names the RMSE-minimizing one at a given
  finite $B$?

A rate theorem is asymptotic up to constants, so POINT may fail while RATE holds.

### Analytic prediction (before running anything)

Computing the bias exactly (the article's closed-form $w_{k,m}$ weights applied to the
*exact* $\mathbb{E}\lvert S_k\rvert$, no sampling) and the standard deviation
analytically (from $n$ and the half-normal CV $\sqrt{\pi/2-1}\approx0.7555$) predicts
that **`prop:opt` overshoots $m_0$ by 3–4 at every budget**:

| $B$ | `prop:opt` $m_0$ | argmin RMSE $m_0$ | RMSE penalty |
|---|---|---|---|
| $10^8$ | 8 | 5 | $2.4\times$ |
| $10^9$ | 9 | 6 | $2.3\times$ |
| $10^{10}$ | 11 | 7 | $3.2\times$ |
| $10^{11}$ | 12 | 8 | $3.0\times$ |
| $10^{12}$ | 13 | 9 | $2.9\times$ |

At the true optimum bias $\approx$ sd (e.g. $2.5\times10^{-4}$ vs $3.3\times10^{-4}$ at
$B=10^{10}$), exactly as the balancing argument says it should be. At `prop:opt`'s
$m_0$ the bias is ~80x *smaller* than the sd — it has over-corrected, buying far more
bias reduction than the variance can pay for.

But the **rate is right at both**: RMSE falls by a factor $0.46$–$0.47$ per decade of
budget at the argmin, against the predicted $10^{-1/3}=0.464$. So `prop:opt` looks
rate-correct with a suboptimal constant — which is precisely what a rate theorem
promises and no more. Experiment C measures whether simulation agrees.

### Implementation

**Question:** with $(d,\omega_1)$ measured, does `tools/allocation.py`'s
Proposition `prop:opt` allocation actually beat naive flat allocation at equal budget?

`src/allocation_experiment.py` + `experiments/01_srw/allocation_config.json`. For each
$(B, m_0)$ cell it sets $n$ to the largest uniform per-scale count the budget affords,
draws $R$ independent replicates (`SeedSequence.spawn`, ground rule 2), and estimates
$\gamma$ with the article's own closed-form $w_{k,m}$ weights (`gamma_closed_form`) —
the estimator `prop:opt` is stated for, and this ladder is exactly the consecutive
$\rho^k$ grid it requires. Generic OLS (`gamma_all_points`) is recorded alongside as a
cross-check. Samples are deliberately **not** persisted: the sweep draws far more than
is worth storing, and the per-cell $\hat\gamma$ values are the result; the base seed
is recorded and spawning is deterministic, so any cell regenerates exactly.

`true_gamma` scores finished estimates only and is never passed to an estimator,
consistent with decision D2.

- **Acceptance (RATE):** measured $\mathrm d\log\mathrm{RMSE}/\mathrm d\log B$ within
  $0.08$ of $-\omega_1/(d+2\omega_1) = -1/3$.
- **Acceptance (POINT):** reported, not asserted — the analytic pre-study predicts this
  one *fails* by 3–4 in $m_0$, so a test asserting it would be asserting the wrong
  thing. The experiment measures the gap and its RMSE cost.
- Verified: `tools/tests/test_allocation_experiment.py` (10 cases — the ladder is
  `def:alloc`'s grid, budget arithmetic never overspends, replicate streams reproduce
  and are independent across cells, unaffordable cells are marked skipped rather than
  faked, summary/rate machinery on planted inputs).

**Risk to flag now:** if Experiment A confirms $d\approx1$ and $\omega_1=1$, check
whether `prop:opt` degenerates for this parameter combination before committing to a
large run — a rule that says "put everything at one scale" is a valid answer but makes
a poor demonstration. Worth a cheap symbolic/numeric sanity check of
`optimal_allocation` at $(d,\omega_1)=(1,1)$ first.

---

## Ordering

A → B → C strictly: B's allocation needs A's $c(i)$, and C needs both. Each is a
separate `experiments/01_srw/` recipe + README acceptance criteria, committed and
pushed on completion.

## Out of scope

- Any change to `models/synthetic.py` or `experiments/00_synthetic/`.
- The `Huge_test.json` run itself — superseded by these three targeted recipes.
- Porting anything from `../loglog_experiments/` (PLAN.md, historical reference only).
