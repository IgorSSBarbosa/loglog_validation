# Plan — three-experiment ladder: measure $d$, then $\omega_1$, then $\gamma$

Status: **proposed, not started.** Per ground rule 3 (one experiment at a time, design
agreed before coding), nothing here is implemented until the open decisions in
§0 are signed off.

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

## 1. Shared code changes (prerequisites for all three experiments)

### 1a. `models/srw.py` — 5.2× faster, same cost structure

Replace `rng.choice(_STEP_VALUES, size=(block, k), p=[1-q, q])` with an
`rng.integers`-based draw. Benchmarked at $k=1024$, $n=2\times10^5$:

| variant | µs/sample | speedup |
|---|---|---|
| current `choice(..., p=)` | 28.70 | 1× |
| `choice(...)` without `p=` | 9.42 | 3× |
| `2*rng.integers(0,2,dtype=int8).sum(axis=1) - k` | 5.56 | **5.2×** |

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

### 1b. `tools/cost_model.py` — aggregator registry

Add `AGGREGATORS = {"min", "median", "mean", "q95", "iqmean"}`, mirroring the existing
`COST_ESTIMATORS` name-keyed-registry pattern, selectable from the recipe JSON.
Measured on the existing runs, the choice barely moves $\hat d$ (min 0.884 / median
0.888 / iqmean 0.887 / mean 0.850 / q95 0.811 on `cost_probe`) — this is for honest
error bars and for the user's stated preference, not to change the estimate.

### 1c. `tools/cost_model.py` — affine-plus-power fit

The real defect in `time_measure` is **model misspecification, not aggregation**: over
$k=2\dots1024$ the measured cost is affine, $\approx 22\,\mu s + 0.025\,\mu s\cdot k$,
so a pure power-law fit returns $\hat d\approx0.10$ — meaningless. Add
`estimate_cost_exponent_affine` fitting $\mathrm{cost}(i)=a+b\,i^d$, and report the
overhead $a$ explicitly as a diagnostic.

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

## 3. Experiment B — the correction-to-scaling exponent $\omega_1$

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

Sample allocation: **not** flat $n$ across scales like `larger_test.json`. With
$\mathrm{cost}(i)\propto i$ from Experiment A, use $n(i)\propto 1/\sqrt{c(i)}$
(Neyman-style: equalizes marginal variance reduction per unit cost) — cheap small
scales get many more replicates, which is exactly where the $\omega_1$ signal lives.
`generate.py` already accepts per-scale `n` (`normalize_scales_n`), so this needs no
new machinery, only a recipe.

- **Acceptance:** both estimators return $\hat\omega_1\in[0.85,1.15]$, and estimator 1
  additionally returns $\hat\gamma$ within $0.01$ of $0.5$ and $\hat a_1$ within $10\%$
  of $-0.25$.
- Independence rule (ground rule 2): the samples used here are a *separate draw* from
  Experiment C's — no reuse, no slicing.

## 4. Experiment C — $\gamma$ at maximum accuracy under an optimal budget

**Question:** with $(d,\omega_1)$ measured, does `tools/allocation.py`'s
Proposition `prop:opt` allocation actually beat naive flat allocation at equal budget?

- Feed $\hat d$ (Exp A) and $\hat\omega_1$ (Exp B) into `optimal_allocation(B, d, omega1, rho, m)`
  to get $(n, m_0, \theta_1, \theta_2)$; run `generate.py` on the resulting ladder.
- **Control arm:** the same total budget $B$ spent on a flat-$n$ ladder. Without this
  the result is unfalsifiable — "we got a $\hat\gamma$" is not evidence the allocation
  rule works.
- Replicate both arms $R$ times (independent seeds, ground rule 2) and compare
  RMSE of $\hat\gamma$ against the known $\gamma=1/2$.
- **Acceptance:** optimal-allocation RMSE $<$ flat-allocation RMSE at equal $B$, with
  the gap outside replicate noise. This is checkpoint 0.5.

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
