# models

Per-model simulation logic, one file per registered model (user's own framing,
2026-08-12): kept separate from both `tools/` — the helper functions models call, if any
— and `src/` — the scripts that call into models through `tools/models.py`'s registry.

---

## Adding a model

Two files change, and no driver does.

**1. Write `models/<name>.py`.** One required function:

```python
def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """n i.i.d. samples of Y_i at scale i."""
```

The contract, all of it load-bearing somewhere:

- **Returns exactly `n` values**, i.i.d., for the one scale `i`. Not a mean, not a
  summary — `src/generate/generate.py` does the reducing.
- **Draws only from `rng`.** Never `np.random.*`, never a module-level Generator: ground
  rule 2 (`PLAN.md`) hands each replicate its own spawned stream, and a model that
  reaches around it silently correlates replicates that are reported as independent.
- **Blocks over `n`, not over the scale**, if it blocks at all. Large `n` has to be split
  to bound memory, and splitting the *leading* axis consumes numpy's row-major stream in
  the same order one unblocked call would — so the output stays bit-identical at any
  block size. Splitting any other axis does not.
  `src/generate/generate.py`'s chunked path depends on this, so it belongs in a test
  (`test_block_n_matches_unblocked_for_same_seed`).
- **Anything that is not the randomness must not change the numbers.** A cost knob, a
  geometry flag, a working-set size: same seed, same draws. Also a test.
- **No `sys.path` games.** Only entry points under `src/` and `calibration/` touch
  `sys.path`; a model imports nothing from `tools/`, `src/`, or `experiments/`.

Two optional functions:

- `cost_hint(i, params) -> float` — the work **one** sample at scale `i` costs, in
  whatever unit the model naturally counts (steps, sites, operations). Only *ratios*
  across scales reach an allocation, so the unit is free and a constant factor cancels.
  Declaring it is what makes a model plannable: $d$ becomes known rather than fitted, and
  `src/estimate/measure_cost.py` scores the measured wall clock against it. Declare it
  whenever you can count the work exactly; leave it out and every recipe using the model
  has to state `"d"` by hand.
- `target_fn(i, params) -> np.ndarray` — the known closed-form $\mathbb{E}Y_i$. **Add
  this only when the article sanctions the formula**, and read the policy below first.

**2. Add one entry to `tools/models.py`'s `MODELS` dict:**

```python
"<name>": ModelSpec(
    simulate=model_<name>.simulate,
    cost_hint=model_<name>.cost_hint,      # omit if there is none
    target_fn=model_<name>.target_fn,      # omit unless the article gives one
    true_gamma_key="gamma",                # which params key holds the true gamma
),
```

**3. Write `tools/tests/test_<name>.py`** before pointing an experiment at it (ground
rule 1: a numeric check, not a plot). What has actually caught bugs here: degenerate
parameters with a known answer ($p=0$, $p=1$, $i=1$), an independent reimplementation
compared on random inputs, exhaustive enumeration of $\mathbb{E}Y_i$ at tiny $i$ where
the state space is small enough to sum, and the two invariance tests above.

**4. Optionally extend `calibration/exercise_all.py`** — add a `sec_<name>` and list it
under `"models"`. That harness asks whether each call *behaves as documented*, including
the error branches no test covers.

Nothing else changes: `generate.py`, `measure_cost.py`, `plot_loglog.py`,
`compare_observables.py` and the whole `src/study/` pipeline dispatch on the recipe's
`"model"` field.

### The `target_fn` policy — read before adding one

A missing `target_fn` is what stops `src/report/plot_loglog.py` overlaying a reference
curve or reporting a `true_gamma`. That absence is a *decision*, not a gap. For `srw`,
$\mathbb{E}|S_k|$ is known exactly; for `percolation2d`, $\gamma = d_f = 91/48$ is known
from the literature. Both are deliberately kept **out of the code path** (user,
2026-08-20; `plans/three_experiment_ladder.md` D1/D2) and written down as acceptance
criteria in the experiment README instead, so no estimator is ever handed the answer it
is supposed to be measuring. Truth enters at *reporting* time — `--expect-gamma`,
`--truth` — and nowhere else.

`synthetic` is the exception because its truth is *planted by the caller*: there is
nothing to discover, so `target_fn` is the point of the model rather than a leak.

---

## The recipe format

A recipe is a JSON file naming a model, its parameters, and what to draw. Three kinds
exist (`tools/artifacts.py`'s `RECIPES`); each declares its `kind`, which is **checked on
load**, so handing a cost probe to the sweep fails immediately instead of with a
`KeyError` three steps down. Recipes live in `experiments/<exp>/recipes/` and are named
`<prefix>_<name>.json`.

### `samples` — `generate.py`, `pilot.py`, `run.py`, `autopilot.py`, `compare_observables.py`

```jsonc
{
  "kind": "samples",
  "model": "percolation2d",              // a key of tools/models.py's MODELS
  "params": {"p": 0.59274605079210,      // passed straight to simulate()
             "anchor": "south",
             "geometry": "cylinder"},
  "scales": [8, 16, 32, 64, 128, 256, 512],
  "n": {"rule": "neyman", "budget": 4e9},
  "seed": 20260905                       // or null for fresh OS entropy
}
```

(The `//` comments above are for this page only — a recipe is strict JSON and will not
parse with them.)

**`scales`** is the ladder, and it is the recipe author's decision — `pilot.py` and
`autopilot.py` never change it. Use a geometric grid $\rho^k$: the article's closed-form
weights (eq. 526) are the OLS slope specialized to equally-spaced $k$, and the study
pipeline's `--rho`/`--m` assume one. Powers of two in practice. A ladder that mixes
regimes fails *silently* — see `experiments/01_srw/README.md` on how mixing odd and even
$k$ turned a staircase mean into $\hat\omega_1 \approx 17.8$ instead of $1$.

**`n`** takes three forms:

| form | meaning |
|---|---|
| `2000` | that many samples at every scale |
| `[500, 400, 300, ...]` | one entry per scale, same order |
| `{"rule": ..., "budget": ...}` | an allocation rule spends a budget across the ladder |

The rule form is preferred: it records *why* those counts were chosen, so the run can be
regenerated after a cost-model change by re-reading the recipe. Two rules, both in
`tools/allocation.py` and **not** interchangeable:

- `"neyman"` — $n_i \propto i^{-d/2}$, minimizing the variance of $\overline Y$ itself.
- `"snr"` — $n_i \propto i^{2\omega_1}$, equalizing the signal-to-noise ratio of the
  *correction* term, which is what Experiment B actually estimates. Needs an `"omega1"`.

Optional keys inside the rule: `"d"`, `"omega1"`, `"sigma"`, `"min_n"`. These are
**design inputs** — they decide how the budget is *split* and never reach an estimator.
`"d"` comes from the model's own `cost_hint` when it has one and does not need stating; a
missing `"omega1"` falls back to the uninformed $0$ and is *reported* as
`uninformed default`, never silently. A recipe must be runnable with no design constants
at all, so that the claim "these never reach an estimator" can be tested by running with
and without them.

**For `autopilot.py` specifically:** the pilot grows the recipe by doubling its draws, so
`"n"` must be growable — a `{"rule": ...}` with no `"budget"` has nothing to double, and
is refused with a message naming the field and the shape wanted. Scales are never
touched; only `n` moves.

**`seed`** is recorded resolved in every run's `samples_meta.json`. `--seed` on the CLI
overrides it, which is how replicates of one configuration get drawn without copying the
recipe.

**`replicates`** appears only in recipes written by `src/study/plan.py`; `generate.py`
ignores it, and `run.py` takes the count from `plan.json`. Those generated recipes are
stamped `"_generated_by"` and rewritten on every re-plan — edit the *pilot* recipe and
re-plan rather than editing them.

### `cost_probe` — `measure_cost.py`

```jsonc
{
  "kind": "cost_probe",
  "model": "percolation2d",
  "params": {"p": 0.59274605079210, "anchor": "south", "geometry": "box"},
  "scales": [16, 32, 64, 128, 256, 512, 1024],
  "repeats": 20,
  "aggregator": "median",       // optional; min | median | mean | q95 | iqmean
  "seed": 20260904
}
```

Times `simulate(i, n=1, ...)` — Assumption 7 defines $\mathrm{cost}(i)$ as the cost of
**one** sample. Reach higher than the sampling ladder: a fixed per-call overhead
dominates at small $i$ and biases the pure-power fit low, which is why the acceptance
check uses the affine $a + b\,i^d$ fit instead.

### `sweep` — `allocation_experiment.py`

Needs `model`, `budgets`, `m0_values`, `m`, `rho`; see
`experiments/01_srw/recipes/sweep_allocation.json`.

### Running one

```bash
python3 src/generate/generate.py -meta <recipe> --tag <run>          # just draw
python3 src/estimate/measure_cost.py -meta <cost recipe> --tag <run>
python3 src/study/autopilot.py -meta <recipe> --study <name> --time 30m
```

---

## The models

| | $Y_i$ | `cost_hint` | declared $d$ | measured (affine) | `target_fn` |
|---|---|---|---|---|---|
| `synthetic` | planted eq. (232) | `cost_scale·i**cost_d`, else 1 | any, else 0 | recovers the planted $d$ | **yes** |
| `srw` | $\lvert S_k\rvert$ | $i$ | 1 | $1.0028 \pm 0.0020$ | no |
| `rwre` | $\lvert X_k\rvert$ on an SSEP | $i\,W(i)$ | 3/2 | $1.4926$ (amortized; see below) | no |
| `percolation2d` | south-connected sites | $i^2$ | 2 | $2.029 \pm 0.018$ (box) | no |
| `percolation_tau` | clusters at size scale $s$, per site | $L(s)^2$ | $2\,$`box_exponent` (1) | $1.049 \pm 0.019$ (torus) | no |
| `percolation_zd` | face-connected sites on $\mathbb Z^{\texttt{dim}}$ | $i^{\texttt{dim}}$ | `dim` (2–6) | see `experiments/05_percolation_highd` | no |
| `percolation_zd_stream` | the same, swept in slabs | $i^{\texttt{dim}}$ | `dim` | — | no |
| `percolation_tau_zd` | clusters at size scale $s$ on $\mathbb Z^{\texttt{dim}}$ | $L(s)^{\texttt{dim}}$ | $\texttt{dim}\cdot$`box_exponent` ($\texttt{dim}/d_f$) | see `experiments/05_percolation_highd` | no |

### `synthetic.py` — the planted generator

$\mathbb{E}Y_i = a_0 i^\gamma \exp(\sum_j a_j i^{-\omega_j})$ (article eq. 232), realized
as $Y_i = \mathbb{E}Y_i\cdot\xi_i$ with $\xi_i>0$, $\mathbb{E}\xi_i=1$,
$\mathrm{Var}(\xi_i) = \sigma_\infty^2$ exactly — so Assumption 6 holds *by construction*
rather than only in the limit. Params: `gamma`, `a0`, `corrections` (a list of
$(a_j,\omega_j)$ pairs, validated for $0<\omega_1<\omega_2<\cdots$), `sigma_inf2`,
`family` (`NOISE_FAMILIES`, currently `lognormal`), `cost_d`, `cost_scale`.

Ground truth is planted, so this is the only model with a `target_fn` and a usable
`true_gamma_key`. It is the statistical testbed: the estimator, the CLT, the Wilson
interval and the allocation rule are all checked here before meeting a real process.

With no burn, `cost_hint(i) = 1` — the scale enters the *formula*, not the work, so
$d = 0$, which is not merely degenerate but *outside* the allocation formulas
(`allocation_constants` raises for $d\le0$). **The work burn** (`cost_scale`, `cost_d`)
fixes that: `simulate` spins for `cost_scale · i**cost_d` seconds per sample and
`cost_hint` declares exactly that, so the exponent is known and the clock has to find it.
It spins rather than sleeps, consumes no randomness, and dominates dispatch — the full
argument for each is in `_burn`'s docstring, and each is pinned by a test in
`tools/tests/test_synthetic.py`. This is what `calibration/check_no_leakage.py`'s `cost`
arm runs on, recovering $d$ from $\{0.5, 0.75, 1, 1.5, 2\}$ when the only $d$ anywhere in
the repo's constants is 1.

One trap, recorded because it cost a wrong measurement: `cost_hint`'s no-burn `1.0` is
one *work unit*, not one second. Feeding it to the burn made every call spin for a full
second per sample.

### `srw.py` — the cost testbed

`srw(k, n=1, q=0.5, rng=None, block_n=None)`, $n$ i.i.d. realizations of $|S_k|$ for a
$\pm1$ walk with $P(+1)=q$. Draws steps as `(block_n, k)` float32 blocks over the $n$
axis — bounding peak memory to a fixed byte budget, where the old unblocked `int64`
version needed 819 GiB at $n=10^8, k=1024$.

`cost_hint(i) = i`, exact: $i$ uniforms per sample, no early exit. That is the whole point
of the model — a known $\Theta(k)$ against which the *measurement procedure* for $d$ can
be validated before being pointed at an expensive simulator. Two faster draws were tried
and rejected, for reasons that generalize to any new model:
`rng.integers(0,2,dtype=int8)` packs several values per 64-bit draw and discards the
remainder per call, breaking block invariance; `rng.binomial` is ~375× faster and
distributionally identical but samples every scale in constant time, destroying the
$\Theta(k)$ cost the model exists to provide.

No `target_fn`, deliberately:
$\mathbb{E}|S_k| = \sqrt{2/\pi}\,k^{1/2}\exp(-\tfrac14 k^{-1}+\cdots)$ is known exactly,
giving $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$, $\omega_2=3$ — recorded as acceptance
criteria in `experiments/01_srw/README.md`. Verified: `tools/tests/test_srw.py`.

### `rwre.py` — the same observable, on a dynamic disordered environment

`simulate(i, n, params, rng)` returns $\lvert X_i\rvert$ for a walker on a 1-D SSEP at
density `alpha`, with $P(\text{left}\mid\text{particle}) = $ `p` and
$P(\text{left}\mid\text{hole}) = 1-$ `p`. Params: `p`, `alpha` (0.5), `env_sweeps` (4),
`swap_prob` (0.5), `window_c` (12.0). At `p = 0.5` the environment is unreadable and the
walk *is* `srw`, which makes `experiments/01_srw` a literal control arm — the calibration
criterion in `experiments/02_rwre/README.md` is that this reproduces its four exactly
known constants.

The environment is a brick-wall parity sweep applied to **sites**, i.e. the stirring
construction, which buys two closed forms a test can assert: product Bernoulli(`alpha`)
is *exactly* invariant at every time, and the particle count is conserved per sample.
The parity is drawn per sample — sharing it across a block would correlate rows that
ground rule 2 requires to be i.i.d.

`cost_hint(i) = i * W(i)` with $W(i)=$ `window_c` $\lceil\sqrt i\rceil$, so
$\mathrm{cost}(i) = $ `window_c` $\cdot i^{3/2}$ and Assumption 7's $d$ is **3/2 exactly**
— the first non-integer $d$ here. The periodic window's wrap error is
$2e^{-\texttt{window\_c}^2/8}\approx1.5\times10^{-8}$ at the default, which is what lets
the width be $\sqrt i$ with no $\sqrt{\log i}$ factor and hence the cost an exact power.

No `target_fn`, as for every model since `srw` — but for a different reason. Elsewhere
the truth is known and deliberately kept out of the code path; at `alpha = 0.5` there is
no known $\gamma$ to keep out. The proven CLT for this model
(Hilário–Kious–Teixeira, arXiv:1906.03167) covers only the densities where the speed is
non-zero, and this is exactly the zero-speed case it leaves open.

**It does not satisfy the blocking clause above, and says so.** One sample is a $k$-step
loop, so the RNG stream is interleaved across steps and row-blocking necessarily reorders
it; neither `srw`'s one-big-draw trick nor `percolation_zd_stream`'s one-sample-at-a-time
order is available. What holds instead, and what `tools/tests/test_rwre.py` checks, is
that the output is a pure function of `(seed, i, n, params)` — the block size is derived
from a fixed byte budget and $W(i)$, so there is no hidden knob — and that two very
different byte budgets agree *distributionally*. Keeping the clause verbatim would cost
about 3× at the top rung, where the budget goes; that was weighed and declined
(user, 2026-09-15). `generate.py`'s chunked path would need $>1.25\times10^8$ samples at
one scale to trigger here.

One measurement trap, recorded because it looks like a failure and is not: the cost probe
times `simulate(i, n=1)`, where this model runs *un-amortized* — every numpy call on a
$(1,W)$ array — so the affine fit returns $\hat d\approx1.20$, 20% below the declaration,
and `measure_cost.py` warns at 9.9σ. At $n=64$, the regime a run actually uses, the same
fit gives $1.4926$ against the declared $1.4928$. The probe's own drop-leading ladder is
the tell: $\hat d$ climbs monotonically with $m_0$ instead of sitting still.

### `percolation2d.py` — critical site percolation, the first real geometry

`percolation2d(i, n=1, p=P_C_SQUARE_SITE, anchor="south", geometry="box", rng=None,
block_n=None)`. Fills an $i\times i$ lattice with i.i.d. Bernoulli($p$) open sites and
counts those connected to the **south side** by 4-connected open paths — `PLAN.md` ground
rule 7's observable. $p_c = 0.59274605079210$ (Jacobsen 2015) is the **4-connected**
threshold; the 8-connected structure would silently simulate a supercritical system
($p_c^{(8)} = 0.4073$), which is why `_FOUR_CONNECTED` is a named constant carrying that
warning.

Two switches, both existing so the choice of observable can be *measured* rather than
argued (`experiments/03_percolation_zd/README.md`, P3 and P4):

- **`anchor`** — `"south"` (default) or `"origin"` (the cluster of the centre site, what
  `presentation18-05-2026` measured). Same exponent, but the origin's
  $\mathrm{Var}(\xi_i)\asymp 1/\pi_1(i)\to\infty$ **violates Assumption 6**, and its
  correction amplitude is far larger. At equal budget: $15.7\times$ worse in RMSE.
- **`geometry`** — `"box"` (default, four walls) or `"cylinder"` (periodic in $x$). For a
  south-anchored count the side walls are pure finite-size contamination. Measured on two
  matched $4\times10^9$-work-unit runs, $\overline Y_i/i^{91/48}$:

  | $i$ | 8 | 16 | 32 | 64 | 128 | 256 | 512 | drift |
  |---|---|---|---|---|---|---|---|---|
  | box | 0.4989 | 0.4842 | 0.4742 | 0.4695 | 0.4651 | 0.4660 | 0.4661 | $-6.6\%$ |
  | cylinder | 0.5578 | 0.5611 | 0.5633 | 0.5640 | 0.5671 | 0.5682 | 0.5698 | $+2.1\%$ |

  It does not *remove* the correction — a $+0.004$ residual in $\hat\gamma$ remains — but
  shrinking it threefold is enough to make the $m_0=2$ estimate variance-limited rather
  than bias-limited, and it is quieter too (cv $\approx0.37$ against $\approx0.43$), for
  $1.04\times$ the cost.

`cost_hint(i) = i**2`, exact and independent of `p`, the anchor and the geometry.
**This is the first model where Assumption 7's $\mathrm{cost}(i)=i^d$ is a geometric fact
about a real simulation rather than a stated formula** — declared $2$ against affine
$\hat d = 2.0285\pm0.0175$ (box, $+1.63\sigma$) and $1.9780\pm0.0388$ (cylinder,
$-0.57\sigma$). The wrap-merge changes the constant, not the exponent, which is why
`cost_hint` deliberately ignores the geometry.

Three implementation choices matter if you touch it, each with a test that pins it:

- **One `ndimage.label` call per block, not per sample.** A block's lattices are stacked
  vertically into one tall image with a blank separator row after each; no open path can
  cross a closed row, so labelling the stack once *is* the per-sample labelling, and the
  C-level union-find sees one large problem instead of $n$ tiny ones. A `neyman`
  allocation asks for ~500k samples at $i=8$, where per-call SciPy overhead would
  otherwise dominate.
- **float32 uniforms, blocked over the sample axis**, for exactly `srw.py`'s reasons. The
  cost is a realized $p$ within $2^{-24}\approx6\times10^{-8}$ of the requested one — a
  distance to criticality whose correlation length is $\approx4\times10^9$ sites, seven
  orders beyond any lattice this will run at.
- **The cylinder's wrap is a vectorized pointer-jumping union-find over *labels***, of
  which there are a few percent as many as there are sites, and the keep-mask is built
  per root and mapped back per label so no second full-size pass is paid (overhead
  1.01×–1.07×). The `root[root]` squaring step is not decoration: without it a
  three-label chain returns 9 where the answer is 12.

Assumption 2 ($Y_i>0$) is deliberately not asserted, unlike `synthetic`. For `"south"` it
fails **iff** row 0 is entirely closed — exactly $(1-p_c)^i$, which `zero_rate` returns in
closed form — and for `"origin"` it is the normal case.

No `target_fn`: $\gamma = d_f = 91/48$ is an acceptance criterion, not an input.
$\omega_1$ is genuinely unknown here (literature $\Omega=72/91\approx0.79$; measured
$\approx0.35$ on the box over $8\le i\le512$, almost certainly an effective exponent) and
is what this rung *measures*. Verified: `tools/tests/test_percolation2d.py` — exact
enumeration of $\mathbb{E}Y_i$ over all $2^{i^2}$ configurations at $i=2,3$ for **both**
geometries, an independent flood fill (wrapping or not) on random critical lattices, the
exact zero rate, `block_n` invariance, and the merge chain above.


### `percolation_tau.py` — the cluster-number density, and a ladder in cluster size

`percolation_tau(s, n=1, p=P_C_SQUARE_SITE, observable="bin", bin_ratio=2.0,
box_factor=16.0, box_exponent=0.5, geometry="torus", rng=None, block_n=None)`.

**The scale is not a box side.** At $p_c$ the number of clusters of size $s$ per site
decays as $n_s \asymp C s^{-\tau}$ with the Fisher exponent $\tau = 187/91$ in $d=2$
(Stauffer & Aharony; Malthe-Sørenssen, *Percolation Theory Using Python*, eq. 4.4 and
Fig. 4.2). The article's estimator fits $\log \mathbb E Y_i$ against $\log i$ on the
ladder $i=\rho^k$, so the only way to put $\tau$ in front of it is to make the ladder
variable the one the power law is *in*: here `simulate`'s first argument is a **cluster
size** $s$, and

$$Y_s = \frac{\#\{\text{clusters of size } \in [s,\,b\,s)\}}{L(s)^2}\ \ (\texttt{"bin"}),
\qquad
Y_s = \frac{\#\{\text{clusters of size} \ge s\}}{L(s)^2}\ \ (\texttt{"tail"}),$$

both with $\mathbb E Y_s \asymp a_0 s^{1-\tau}$, so $\gamma = 1-\tau$ and the conversion
at reporting time is $\boxed{\hat\tau = 1 - \hat\gamma}$, with acceptance value
$\gamma = -96/91 = -1.0549\ldots$ — **the first negative $\gamma$ in the repo**
($Y_s$ decays along the ladder; eq. (523)–(531) is an OLS slope and does not care, but a
plot reader does). Dividing by the box area is what makes that conversion carry no
design constant: it holds for *any* box rule. The bin width $\Delta s=(b-1)s$ is
deliberately **not** divided out, though the book's eq. (4.4) does divide by it — it is a
power of the scale, so dividing would move exactly one unit of the exponent inside the
simulator.

**The geometric ladder *is* Fig. 4.2's logarithmic binning.** Bin edges $a^i$ with $a=2$
are exactly a powers-of-two ladder with `bin_ratio` $=2$: one rung per bin, tiling $s$
with no gap and no overlap. The one difference from the book is ground rule 2 — each rung
is drawn on its **own independent boxes**. One lattice does supply every bin at once, and
that is what makes Fig. 4.2 cheap, but it correlates the rungs, and the CLT (eq. 583) the
error bars come from no longer applies to them.

**Why the box grows with the rung.** In a fixed $L\times L$ box, $n_s(L)$ is cut off
around $s\sim L^{d_f}$, so the top of the $s$-range bends down and has to be discarded —
and that is the wrong end for this estimator, which drops the $m_0$ *smallest* rungs and
keeps the largest. Tying the box to the rung,

$$L(s) = \lceil \texttt{box\_factor}\cdot s^{\texttt{box\_exponent}}\rceil,
\qquad \mathrm{cost}(s) = L(s)^2,$$

puts every rung at the same relative distance from its own cutoff, so **no rung is
thrown away** and $d = 2\,$`box_exponent` $= 1$ exactly at the default. Two knobs, both
measurable rather than argued:

- `box_exponent` — $1/2$ (default) needs no knowledge of $d_f$ and gives an exact
  $d=1$; the cutoff ratio $s/L^{d_f}$ then drifts as $s^{5/96}$ (13% per decade) and
  $\mathrm{Var}(\xi_s)\asymp1/\lambda$ drifts as $s^{5/91}$ — the same slow Assumption-6
  divergence, and nearly the same exponent, as `percolation2d`'s origin anchor.
  $48/91 = 1/d_f$ freezes both, at the price of importing $d_f$ as a **design** constant
  (the category `omega1` is in for an allocation: it fixes the geometry, never reaches an
  estimator, and getting it wrong changes the noise, not the fitted $\tau$).
- `box_factor` — nearly free precision-wise, since cost and count both scale as
  `box_factor`$^2$. Measured at $s=64$ (bin, torus): $\mathrm{cv}\cdot L$, the precision
  a unit of budget buys, is $73.5 / 73.6 / 76.3$ at `box_factor` $=8/16/32$, while the
  zero fraction falls $0.45 \to 0.04 \to 0.00$. Hence the default 16, not the 8 that
  first looked cheap.

The **fixed-box design is still reachable** and is the honest comparison arm:
`box_exponent = 0` with `box_factor = L` draws every rung on one $L\times L$ box (then
$d=0$, outside the allocation formulas, so such a recipe must state its own `n` list) and
the ladder has to stop well below $L^{d_f}$ by hand.

**`geometry = "torus"` is the default here**, unlike `percolation2d`, whose south-anchored
count *needs* a south wall. A cluster touching a wall is truncated, i.e. recorded at a
size below its own — which does not just lose clusters from a bin, it *feeds* the bin
from above, and since $n_s$ falls steeply the influx wins. Measured paired on the same
lattices, box/torus $= 1.571,\ 1.561,\ 1.565,\ 1.588$ at $s = 16, 64, 256, 1024$: an open
box reports **57% more clusters in every bin**. Because $L(s)$ grows with $s$ that is
almost purely an amplitude — it lands in $a_0$ — and the residual drift is $+0.0027$ in
$\gamma$, about half the pilot's statistical error. So `"box"` is usable and is kept for
the head-to-head, but it buys 57% of $a_0$ and a correction term for nothing.

**Assumption 2 is reported, never asserted** — and unlike `percolation2d`'s south anchor,
where a zero is exponentially rare, a zero here is *ordinary*: the count in one box is a
small near-Poisson number, so a fraction $\approx e^{-\lambda}$ of draws are $0$ by
design (4% at the default `box_factor`). `zero_fraction(draws)` reports it; there is no
closed form to check it against.

Implementation notes, each pinned by a test:

- **One `ndimage.label` per block**, stacked with a blank separator row, exactly as
  `percolation2d` does it and for the same reason. The code is duplicated rather than
  imported: no model file imports another.
- **The torus merge joins column $0$ to $L-1$ *and* row $0$ to $L-1$ of the same
  sample**, by the same vectorized pointer-jumping union-find over labels — but its
  termination test is on the **whole pass** (`root` unchanged after the unions *and* the
  squaring), not on the squaring alone. Testing only the squaring, which is what
  `percolation2d`'s `_wrap_roots` does, exits as soon as the pointer array is flat even
  when a further pass would still merge: on the $32\times32$ torus at `default_rng(3)` it
  reported 2 clusters where the flood fill finds 1, in 4 of 300 samples. It surfaced as a
  `block_n` invariance failure, since the label numbering a block produces decides
  whether the early exit is reachable. (Checked separately: with only `percolation2d`'s
  single wrap direction the chains are too short to reach it — current and strict merges
  agree on all 2976 cylinder samples tried, $i \le 256$.)
- **label → sample by one scatter over the sites** (`owner[flat] = sample index`), which
  is well defined because a cluster never spans two samples, and which assumes nothing
  about the order `ndimage.label` hands out labels.

**Measured, on the pilot pair of runs it was written for** (`samples_tau_torus.json` /
`samples_tau_tail.json`, $s = 8\ldots2048$, $4\times10^9$ sites each, 161 s):

| | cv over the ladder | zero fraction | $\hat\gamma\pm$ se at $m_0=3$ | $\hat\tau$ |
|---|---|---|---|---|
| `bin` | $0.578$–$0.617$ | $0.042$–$0.061$ | $-1.0394 \pm 0.0032$ | $2.0394$ |
| `tail` | $0.420$–$0.446$ | $0.000$ | $-1.0391 \pm 0.0023$ | $2.0391$ |

against $\tau = 187/91 = 2.05495$, i.e. both arms land 0.5% low and are **bias-limited,
not variance-limited** — $\hat\gamma$ climbs monotonically toward the truth as $m_0$
grows ($2.027 \to 2.041$ over $m_0 = 0\ldots5$ on the bin arm) and the direct fit of
eq. (232), which models the correction instead of dropping it, gives
$\hat\tau = 2.044$ with $\hat\omega_1 = 0.98$, $\hat a_1 = -0.89$. A correction with
$\omega_1 \approx 1$ is what the discreteness of $\sum_{u=s}^{2s-1}u^{-\tau}$ alone
would produce, so this rung has an $\omega_1$ worth measuring rather than a mystery.
The cost probe passes too: affine $\hat d = 1.049 \pm 0.019$ over $s = 16\ldots4096$
against the declared $0.9993$ ($+2.6\sigma$, 5.0% — inside the driver's 20% tolerance),
with the usual small-scale story (pure-power $\hat d = 0.88$, overhead 66% of the
measurement at $s=16$).

**The one unit trap.** A recipe's `budget` is charged at $i^d$ *with the scale itself*
(`tools/allocation.py`), not at `cost_hint`'s value — for `srw` and `percolation2d` those
coincide, and here they do not: one budget unit is `box_factor`$^2 = 256$ lattice sites.
Ask for $S/256$ to spend $S$ sites. The first run of this model asked for $4\times10^9$
meaning sites, and was on course to spend $10^{12}$ of them.

The same gap broke the **planner**, which is worse because it was silent:
`src/study/plan.py` divided a cost in allocation units by a throughput measured in
`cost_hint` units, and so predicted **740 s for a run that would have taken 42 hours**
($m_0=10$, $n=12\,086$, scales up to $s=65\,536$). Both directions of that conversion now
go through `tools/cost_model.cost_unit_ratio` — exactly $1.0$ for every model whose
`cost_hint` *is* $i^d$, so no `srw` or `percolation2d` plan changes — and the planner
bisects on predicted seconds rather than multiplying by throughput
(`plan.py: budget_for_seconds`). Verified after the fix: predicted 90.0 s, measured
89.9 s; a full autopilot predicted 412.8 s and drew in 404.6 s.

#### The cheap version: `shared_sampler`, one box for the whole ladder

`simulate` pays for a fresh box **per rung**. But one critical lattice already holds
clusters of every size below its own cutoff, so a box sized for the *top* rung can serve
the entire ladder — Fig. 4.2's own procedure. `shared_sampler(scales, n, params, rng)`
does that, driven by `src/generate/generate_shared.py` and a `samples_shared` recipe
(`n_lattices` replaces `n`: there is nothing to allocate when every rung reads the same
lattices). The box comes from the top rung,

$$L=\left\lceil\left(s_{\text{top edge}}/\kappa\right)^{1/d_f^{-}}\right\rceil,$$

with two **design** constants (neither reaches an estimator): $d_f^{-}$, a *lower* bound
on $d_f$ so the trusted region is conservative (default $1.85$ — below this repo's own
$1.9002/1.9059/1.9161$ and below $91/48$), and $\kappa$, the margin that keeps the top bin
out of the cutoff.

**$\kappa$ is measured, not guessed.** Hold one bin fixed, grow $L$, and watch
$\overline Y_s$ — a per-site density, so it must converge ($1.2\times10^9$ sites per point):

| $s_{\text{hi}}/L^{1.85}$ | 0.43 | 0.39 | 0.23 | 0.21 | 0.12 | $\le 0.10$ |
|---|---|---|---|---|---|---|
| bin $[64,128)$ | — | $+13.9\%$ | — | $-0.00\%$ | — | $<0.15\%$ |
| bin $[256,512)$ | $+20.4\%$ | — | $+0.39\%$ | — | $-0.23\%$ | $<0.6\%$ |

The two bins agree **in the scaled variable**, which is what had to be true for one
$\kappa$ to serve every rung. Bias is under the noise floor for $s\lesssim0.23\,L^{d_f}$
and explodes past $\approx0.4$; the default $\kappa = 0.05$ keeps a $4.6\times$ margin.
Note $\kappa$ costs nothing: at fixed top-rung precision the total is $n L^2 = C/c_{\rm top}$,
independent of $L$ — $\kappa$ only trades box size against lattice count.

**What it buys, measured on the same ladder ($s=8\ldots2048$) and the same estimator:**

| | sites | wall clock | $\hat\tau$, $m_0=3$ | $m_0=4$ | $m_0=5$ |
|---|---|---|---|---|---|
| independent rungs (`generate.py`) | $4.0\times10^9$ | 161 s | 2.0394 | 2.0391 | 2.0411 |
| shared lattice (`generate_shared.py`), mean of 4 | $1.25\times10^9$ | 49 s | 2.0441 | 2.0468 | **2.0491** |

against $\tau = 2.05495$. **3.2× cheaper and closer at every $m_0$** — and it keeps
converging as $m_0$ grows where the independent run flattens near 2.039. The likely
mechanism, stated as a hypothesis on one independent run against four shared ones: with
one box, any box-size effect is *common to every rung* and cancels from a slope whose
weights sum to zero (eq. 542), whereas the per-rung sampler changes the box with the
rung, so its finite-size effect does not cancel.

**What it costs: the rungs are correlated**, which is outside the article's independence
assumption and is the point of the experiment (user, 2026-09-05). Measured on the
$n=6100$ run, correlation of $Y_s$ across rungs:

| lag (doublings) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mean corr | $+0.124$ | $+0.119$ | $+0.110$ | $+0.105$ |

i.e. **a flat $\approx+0.11$ pedestal that does not decay in lag** — a common mode (a
lattice that happens to be busy is busy at every size), not a neighbour effect. That is
the benign case for this estimator: $\sum_k w_{k,m}=0$, so a common mode cancels, and the
covariance-corrected error bar
$\mathrm{se}^2 = w^\top\mathrm{Cov}(\log\overline Y)\,w$ differs from the naive
independent-rung one by $+6\%$ at $m_0=0$ and $\le1\%$ from $m_0=3$ on. Replicating the
whole experiment (16 replicates, $s\le512$) the empirical spread of $\hat\gamma$ is
$0.0033$ against a predicted $0.0041$ — the stated bar is if anything **conservative**
($\mathrm{sd}/\mathrm{se}=0.78\pm0.14$; a proper coverage study at $R\ge50$ is the
follow-up, `calibration/check_coverage.py` is the machinery for it).

No `target_fn`: $\tau = 187/91$, and the hyperscaling relation $\tau = 1 + d/d_f$ it
comes from, are acceptance criteria and `--expect-gamma` arguments, never inputs.
Verified: `tools/tests/test_percolation_tau.py` — exhaustive enumeration of
$\mathbb E Y_s$ over all $2^{L^2}$ configurations at $L=2,3$ for both geometries and both
observables, an independent flood fill (wrapping or not) on random critical lattices up
to $L=32$, the exact identity $\#[s,2s) = \#{\ge}s - \#{\ge}2s$ sample by sample,
`block_n` invariance, and the RNG-consumption invariance of both switches.

---

No model file imports another, or anything from `tools/`, `src/` or `experiments/` —
`tools/models.py` is the only thing that imports these, as `from models import srw`. Note
the two names that look alike and are not: `models` is this package of simulators,
`tools.models` is the registry that indexes them. Writing both out in full is what keeps
them apart.


### `percolation_zd.py` — the same lattice, with the dimension as a parameter

`percolation_zd(i, n=1, dim=2, p=None, anchor="face", geometry="box", rng=None,
block_n=None)`. Every algorithm in `percolation2d.py` — the blocked float32 draw, the
one-`ndimage.label`-per-block stacking, the vectorized pointer-jumping merge across
periodic boundaries, the per-root keep-mask, the exact zero rate — written for an
arbitrary number of axes, with `dim` moved out of the filename and into `params`.
**At `dim = 2` it reproduces `percolation2d.py` bit for bit at the same seed**
(`test_matches_percolation2d_bit_for_bit`, all four anchor × geometry combinations), so
every measured number in `experiments/03_percolation_zd/README.md` still describes this
code path too.

**Why this is the model the article's Assumption 7 wanted.** `cost_hint(i) = i**dim`
exactly, so *the article's cost exponent $d$ **is** the spatial dimension*: one recipe
field sweeps $d = 2,3,4,5,6$ without one simulator per dimension. Where
`percolation2d.py` made $d=2$ a geometric fact rather than a stated formula, this makes
$d$ a **knob** — which is what the budget-allocation theory (eq. 945–946, whose optimal
$n_i \propto i^{-d/2}$ and error rate $-\omega_1/(d+2\omega_1)$ both depend on $d$)
has never been tested against.

**Three anchors, because ground rule 7 is a $d\le4$ statement.** The 2-D derivation of
$\gamma = d_f$ generalizes literally, and generalizing it is what shows where it stops:

$$\mathbb E Y_i \;\asymp\; \sum_{h=1}^{i} i^{\,\texttt{dim}-1}\min(1,\pi_1(h)),
\qquad \pi_1(h)\asymp h^{-\beta/\nu},\quad \beta/\nu = \texttt{dim}-d_f,$$

a sum dominated by $h\sim i$ when $\beta/\nu<1$ and by $h = O(1)$ when $\beta/\nu>1$:

$$\boxed{\ \gamma_{\text{face}} = \max(d_f,\ \texttt{dim}-1)\ }$$

| `dim` | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|
| $\beta/\nu$ | 0.104 | 0.477 | 0.955 | 1.46 | 2 |
| $d_f$ | 1.896 | 2.523 | 3.045 | 3.54 | 4 |
| $\texttt{dim}-1$ | 1 | 2 | 3 | 4 | 5 |

So the face count measures $d_f$ up to `dim` $=4$ and the **trivial surface exponent**
`dim`$-1$ from `dim` $=5$ on (4 where $d_f=3.54$; 5 where $d_f=4$). `dim` $=4$ is nearly
degenerate on its own terms — $3.045$ against $3$ — so its crossover is slow.

`anchor="face_far"` is the fix: count only the sites with $x_0 \ge i/2$, so the sum runs
over $h\in[i/2,i)$ and gives $i^{d_f}$ in **every** dimension. It keeps both properties
that made the face anchor beat the origin anchor in 2-D — the count is a sum over
$\sim i^{\texttt{dim}-1}$ decorrelating columns, so $\mathrm{Var}(\xi_i) = O(1)$ and
Assumption 6 holds; and $Y_i=0$ is a crossing failure, an $O(1)$ probability rather than
one tending to $1$. Measured at `dim` $=3$ over $i=8\ldots64$ (cylinder,
$6\times10^{8}$ sites), local slopes:

| anchor | $8\!\to\!16$ | $16\!\to\!32$ | $32\!\to\!64$ | OLS |
|---|---|---|---|---|
| `face` | 2.657 | 2.614 | 2.600 | 2.622 |
| `face_far` | 2.553 | 2.523 | 2.520 | 2.531 |

against $d_f = 2.523$: `face_far` is already the better observable in three dimensions,
where `face` is still correct in principle. `anchor="origin"` remains as the comparison
arm, and `"south"` is accepted as an alias for `"face"` so a 2-D recipe reads the same
against either model.

**`anchor="slab"` makes the seed set's dimension a parameter** (`anchor_dim` $=k$: $0$
the centre site, $1$ a central axis, $2$ a central plane, `dim` every site — Igor's
proposal, 2026-09-06). The same depth sum for a $k$-slab gives

$$\gamma(k)=\begin{cases}k+\gamma/\nu, & k<\beta/\nu\\ d_f, & \beta/\nu\le k\le d_f\\ k, & k>d_f\end{cases}$$

so there is a **plateau in $k$** whose value is $d_f$ and whose two edges give
$\beta/\nu$ and $d_f$ — a measurement, not a fit. $k$ is deliberately *not* a design
constant: it changes the exponent being measured, so picking it from a literature
$\beta/\nu$ would assume the answer. $k=0$ reproduces `"origin"` identically (a test) and
$k=\texttt{dim}$ gives $\gamma=\texttt{dim}$ exactly with no percolation in it (a free
calibration point). Measured at `dim` $=3$: the axis lands at $2.540\pm0.026$ against
$d_f = 2.523$ where the $k=2$ slab is at $2.650$, and it takes the zero fraction from the
origin's $0.698$ to $0.000$ and the cv from $3.08$ to $0.66$. See
`experiments/05_percolation_highd/README.md`, H7.

**`anchor="origin"` does not measure $d_f$.** $\mathbb E|C(0)\cap B_i|$ is the
box-restricted susceptibility, $i^{\gamma/\nu} = i^{d-2\beta/\nu}$ — $43/24$ in two
dimensions, not $91/48$ — because $\mathbb E|C\cap B_i| = P(\text{reach }i)\cdot
\mathbb E[|C|\mid\text{reach }i] \asymp i^{-\beta/\nu}i^{d_f}$. `percolation2d.py`'s
docstring and `experiments/03_percolation_zd/README.md` said otherwise; both now carry
the correction, and P3's own measured $1.7593\ldots1.7955$ converges on $43/24$.

**Geometry** picks which axes are periodic: `"box"` none, `"cylinder"` every axis but
$x_0$ (so the anchor face and its opposite are the only walls — the direct
generalization of the 2-D cylinder), `"torus"` all of them. A `dim`-torus joins up to
`dim` pairs of faces at once, so the label chains are long and `_wrap_roots` uses the
**strict whole-pass termination test** `percolation_tau.py` had to introduce, not
`percolation2d.py`'s squaring-only one.

**`p_c` is a literature input, and the one real new risk.** In 2-D it is known to 14
digits and nothing reachable resolves it; above 2-D it is a *different number per
dimension* (`P_C_SITE_HYPERCUBIC`, with `P_C_SOURCE` recording the reference for each),
and a wrong entry simulates an off-critical system — a bias no estimator here can see,
because a slightly supercritical lattice still gives a clean power law with the wrong
exponent. `crossing_fraction` is the diagnostic that checks a threshold instead of
trusting it, and `src/estimate/check_criticality.py` is its driver.

Two guards that only exist because high `dim` reaches them: `_MAX_DIM = 13` (`_structure`
materializes a dense $3^{\texttt{dim}}$ array — 3.5 G entries at `dim` $=20$, from a
typo) and `_MAX_SITES_PER_SAMPLE = 2^{31}-1` (`ndimage.label` returns int32, so a larger
lattice cannot be labelled at all). Both raise naming `i` and `dim`.

No `target_fn`: $d_f(\texttt{dim})$, and the exact mean-field $d_f = 4$ for
`dim` $\ge 6$, are acceptance criteria in
`experiments/05_percolation_highd/README.md`. Verified:
`tools/tests/test_percolation_zd.py` — the bit-for-bit 2-D identity, exact enumeration of
$\mathbb EY_i$ at (`dim`,$i$) $=(2,3),(3,2),(4,2)$ for both anchors, an independent
$d$-dimensional flood fill on critical lattices for all three anchors and all three
geometries, the closed form $\mathbb EY_i = \sum_{k\le i}p^k$ at `dim` $=1$,
`block_n` invariance, and the exact zero rate $(1-p)^{i^{\texttt{dim}-1}}$.

### `percolation_tau_zd.py` — $\tau$ in any dimension, and the first exact target

`percolation_tau_zd(s, n=1, dim=2, p=None, observable="bin", bin_ratio=2.0,
box_factor=None, box_exponent=None, geometry="torus", rng=None, block_n=None)`.
`percolation_tau.py` generalized the same way, `shared_sampler` included; at `dim` $=2$
with `box_exponent=0.5, box_factor=16` it reproduces it bit for bit.

$\tau = 1 + \texttt{dim}/d_f$ and $\gamma = 1-\tau$, so:

| `dim` | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|
| $\tau$ | $187/91$ | 2.189 | 2.313 | 2.412 | $\mathbf{5/2}$ |
| $\gamma$ | $-96/91$ | $-1.189$ | $-1.313$ | $-1.412$ | $\mathbf{-3/2}$ |

**`dim` $\ge 6$ is why this model is worth having.** Above the upper critical dimension
$\tau = 5/2$ and $d_f = 4$ are *exact*, not literature best-fits — so a run there is
scored against a known rational. Until now the only exact target in this repo was
`synthetic`'s planted one; this is the first *real* simulated process with one.

**The 2-D box rule does not survive the generalization.** With
$L(s) = \lceil\texttt{box\_factor}\cdot s^{\texttt{box\_exponent}}\rceil$ the cutoff
ratio $s/L^{d_f}$ drifts as $s^{1-d_f\cdot\texttt{box\_exponent}}$, and
`percolation_tau.py`'s default $1/2$ is exactly $1/\texttt{dim}$ in two dimensions,
where the drift exponent $1-d_f/\texttt{dim} = 5/96$ is negligible. It is not elsewhere:

| `dim` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| $1-d_f/\texttt{dim}$ at $1/\texttt{dim}$ | 0.052 | 0.159 | 0.239 | 0.292 | 0.333 |

so the default here is `box_exponent` $= 1/$`DF_LOWER[dim]`, which freezes the ratio
(conservatively — `DF_LOWER` is a *lower* bound on $d_f$, so the box grows slightly
faster than needed). The option `percolation_tau.py` documents as *available* becomes
**mandatory** above two dimensions. Two consequences: the cost exponent is
$d = \texttt{dim}/d_f = \tau-1 = -\gamma$ — the article's cost exponent and the exponent
being measured are the same number — and $\lambda(s)$, the mean count per box, is
constant along the ladder, so Assumption 6 holds outright.

`DF_LOWER` is a **design** constant in exactly the sense an allocation rule's `omega1`
is: it sizes boxes, never reaches an estimator, and getting it wrong changes the noise
and the cost, not the fitted $\tau$. The experiment README states the test that shows it
(two runs at different `box_exponent`, same $\hat\tau$).

**`box_factor` is dimension-aware for the unit trap's sake.**
$\lambda \asymp \texttt{box\_factor}^{\texttt{dim}}C$, so a `box_factor` carried across
dimensions unchanged would move the count per box by orders of magnitude. What is carried
across instead is `box_factor**dim` — the lattice sites in one allocation budget unit,
i.e. `tools/cost_model.cost_unit_ratio` for this model — fixed at
`SITES_PER_BUDGET_UNIT = 256`, `percolation_tau.py`'s 2-D value. So **"ask for $S/256$ to
spend $S$ sites"** is the rule at `dim` $=2$ and at `dim` $=6$ alike, and the trap that
once made `plan.py` predict 740 s for a 42-hour run does not acquire a per-dimension
footnote. It does *not* equalize $\lambda$ across dimensions: $C$ is dimension-dependent
and is a measurement — read `zero_fraction` off a pilot and raise `box_factor` if it is
above $\approx10\%$.

Verified: `tools/tests/test_percolation_tau_zd.py` — the bit-for-bit 2-D identity, an
independent $d$-dimensional flood fill (counts *and* the full multiset of cluster sizes on
a 3-torus, which is what the multi-axis merge can silently get wrong), `block_n`
invariance, the box rule against its own definition, `binned_counts` against the per-rung
path on the same lattices, and that the shared sampler's rungs are correlated while
`simulate`'s are not.


### `percolation_zd_stream.py` — the same numbers, in $O(i^{d-1})$ memory

`percolation_zd_stream(i, n=1, dim=2, p=None, anchor="face", anchor_dim=None,
geometry="box", rng=None, slab_h=None)`. `percolation_zd`'s observable, swept plane by
plane along $x_0$ so the box is never materialized. Implements
`plans/streaming_percolation.md` §3; §4's parallel divide-and-conquer is not done.

**Why.** `percolation_zd` allocates $\approx10\,i^{\texttt{dim}}$ bytes per sample, and
`block_n` bounds how many *samples* are in flight but cannot make one smaller. That caps
the ladders at $i\le256$ (`dim` 3), $64$ (4), $32$ (5), $16$ (6) — 6, 5, 4, 4 rungs — and
`experiments/05_percolation_highd` traces three separate weaknesses to exactly that
(one $\omega_1$ estimator instead of two, a short cost-probe lever arm, no discriminating
power in the $\mathrm{dim}=6$ criticality check).

| `dim`, $i$ | box | frontier | |
|---|---|---|---|
| 3, 256 | 0.16 GiB | 0.8 MiB | $200\times$ |
| 3, 512 | 1.25 GiB | 3.2 MiB | $390\times$ |
| 3, 1024 | 10.0 GiB | 13.0 MiB | $790\times$ |
| 3, 2048 | 80 GiB — **refused** | 52 MiB | — |

**It is bit-identical to `percolation_zd` at the same seed**, which
`plans/streaming_percolation.md` §6 predicted was impossible. numpy fills
`rng.random(size=(rows,) + (i,)*dim)` in C order — sample-major, plane-minor — so a sweep
that processes **one sample at a time**, drawing it plane by plane, consumes precisely
that sequence. Only a sampler that *batches* samples has the problem the note described.
Pinned by `test_bit_identical_to_percolation_zd` over every dimension, geometry and
anchor; the independent lattice-agreement test (5760/5760) is kept as the cross-check.

**What that costs.** No batching, so the per-sample Python/SciPy overhead is not
amortized: measured **0.64–0.68× the speed** of the materialized model — 0.64/0.67/0.67
at `dim` $=3$, $i = 256/512/1024$ and 0.67/0.68 at `dim` $=4$, $i = 64/96$. That ratio is
flat, not converging: there is **no speed crossover**, and the "large win once the box
leaves cache" the design note expected never appeared, because the box leaves cache in
both models. The win is memory and reach, not throughput — $i=2048$ at `dim` $=3$ streams
in 607 s where the materialized model refuses at 80 GiB. **Use `percolation_zd` wherever
the box fits**; reach for this one when it does not.

**The algorithm, and the two traps.** Connectivity is not causal in the sweep direction,
so a greedy "decide as you pass" sweep undercounts (`plans/streaming_percolation.md` §2's
$3\times3$ counterexample: true 6, greedy 5). Instead the sweep carries *unresolved*
components — Hoshen–Kopelman with a frontier, each live component holding a size, a
"touches the seed set" flag, and a far-half count for `face_far`; a component with no
label left in the new frontier is dead and is banked. Freeing is by **compaction** (the
live roots are renumbered densely each slab) rather than refcounting, which is what makes
the memory bound hold. Two bugs found in testing, both torus-only and both now regression
tests:

- `first`, the plane-0 ids kept for the sweep-axis wrap, must be **renumbered through
  every compaction** or they point at the wrong components two slabs later;
- the interface union and each periodic transverse face must go into **one** merge pass —
  a later pass can pull an id out from under a root an earlier pass linked to it (found by
  brute force: a $4\times4$ torus losing one site of a six-site cluster). `_wrap_roots`
  in `percolation_zd.py` collects every axis's edges before its loop for the same reason.

`slab_h` trades frontier memory against the number of SciPy calls and **never changes the
numbers** (`models/README.md`'s working-set rule, tested). `block_n` is accepted for
signature parity and ignored — batching is exactly what would break bit-identity.
`cost_hint` is identical to `percolation_zd`'s, so allocations, budgets and
`cost_unit_ratio` carry over and the two are comparable at equal budget. The int32 label
ceiling now applies to the **frontier**, $i^{\texttt{dim}-1}$, not the box.

### `percolation2d_gpu.py` — `percolation2d` on a CUDA GPU

`percolation2d_gpu(i, n=1, p=p_c, anchor="south", geometry="box", rng=None)`. The same
observable, parameters and `cost_hint` as `percolation2d`, with draw, label, merge and
count run on the device (CuPy; `cupyx.scipy.ndimage.label` for the labelling). A
separate `MODELS` entry rather than a flag, because the random stream differs: each
call seeds cuRAND with one `rng.integers(0, 2**63)` from the driver's generator.

Two deliberate departures from the rules above, both for `*_gpu` models only (user,
2026-09-16):

- **It imports declarations from its CPU sibling** — p_c, the structure, the anchor and
  geometry names, `cost_hint`, `zero_rate` — so the declared cost is the same function
  object and cannot drift. The algorithm itself is not imported, only ported.
- **Not bit-identical at any block size.** cuRAND fills a block at once, so the block
  size is part of what a seed means. It is a fixed function of `i`
  (`block_rows`, a constant byte budget), never of free VRAM. What holds instead is
  equality *in distribution* with `percolation2d`: two-sample KS and a mean check,
  in `tools/tests/test_percolation2d_gpu.py`.

Importing it never imports cupy. `simulate` raises `RuntimeError` naming
`percolation2d` when cupy or a CUDA device is missing, and never falls back silently.
See `experiments/07_percolation2d_gpu/README.md`.

### `percolation_zd_gpu.py` — `percolation_zd` on a CUDA GPU

`percolation_zd_gpu(i, n=1, dim=2, p=None, anchor="face", anchor_dim=None,
geometry="box", rng=None)`, plus `crossing_fraction_gpu`. Everything above about
`percolation2d_gpu` applies: separate model, cuRAND seeded from the driver's rng, block
size fixed by `block_rows(i, dim)`, equal in distribution rather than bit for bit, and
cupy imported only inside `simulate`. It imports from `percolation_zd` the declarations
plus the three pure indexing helpers (`_periodic_axes`, `_wrap_faces`, `_seed_slab`), so
the geometry cannot disagree. The "origin" count uses the k = 0 seed-slab gather, the
same number without cupy's memory-hungry `bincount`. See
`experiments/08_percolation_zd_gpu/README.md`.

### `percolation_susceptibility.py` — the first off-critical model, and the first ladder in $\varepsilon$

Everything above is at $p = p_c$ and indexes its ladder by a LENGTH (a box side $i$, a
cluster size $s$). This one sits BELOW the critical point and indexes the ladder by the
distance to it:

$$\varepsilon(x) = \varepsilon_0/x, \qquad p(x) = p_c - \varepsilon(x), \qquad x = 1, 2, 4, \dots$$

A geometric ladder in $x$ is a geometric ladder in $1/\varepsilon$, so eq. (232) reads
$\mathbb{E}Y_x \sim a_0 x^\gamma$ with **$\gamma$ the susceptibility exponent**
$\gamma_{\text{susc}} = 43/18$ in $\dim = 2$ — the same letter in the same equation as
`percolation_zd`'s $d_f = 91/48$, a different number because the ladder is different.
$\varepsilon_0$ is a free amplitude: it multiplies $a_0$ and leaves $\gamma$ alone.
Everything downstream (the $m_0$ window, the estimators, `tuned_allocation`) runs
unmodified, which is the point of putting the experiment in these coordinates.

`eps0` gets no dimension-independent default and `p_at` REFUSES rather than clips:
$\varepsilon_0 = 1/2$ is `prompts/gamma_exponent.tex`'s own first rung and is fine at
$p_c = 0.593$, but $p_c = 0.3116$ in $\dim = 3$ and $0.1090$ in $\dim = 6$, where
$p_c - 1/2$ is negative. That is a real bug in the prompt's ladder, caught at the
constructor rather than becoming a silently different experiment.

**The observable is the whole lattice, not one cluster.** One sample is one $L^{\dim}$
torus and

$$Y = \frac{\sum_{\text{clusters}} s^{k+1}}{p\,L^{\dim}} = \frac{\sum_{\text{sites }x}|C(x)|^k}{p\,L^{\dim}}, \qquad \mathbb{E}Y = \mathbb{E}\!\left[|C(0)|^k \mid 0\text{ open}\right]$$

exactly — the denominator is deterministic, so unlike a ratio-of-means estimator this has
no ratio bias, and the $1/p$ (a known input) converts "cluster of a site" into "cluster of
an OPEN site" and removes a factor that would otherwise carry its own $x^{-1}$ correction
along the ladder. Drawing the textbook definition literally — one lattice, one origin, one
number — would waste the whole lattice; this uses every site of it and is one
`ndimage.label` call per block, reusing `percolation_zd`'s draw, stacking, and
pointer-jumping wrap merge unchanged.

`moment` $=2$ is not decoration. $\mathbb{E}|C| \sim s_\xi^{3-\tau}$ and
$\mathbb{E}|C|^2 \sim s_\xi^{4-\tau}$, so the two exponents give
$1/\sigma = e_2 - e_1$ and $\tau = 3 - e_1/(e_2-e_1)$: **one design, run at two moments,
measures $\gamma_{\text{susc}}$, $\sigma$ and $\tau$** — and by
`prompts/scaling_relations.tex`'s inversion, $(\tau,\sigma)$ determine every exponent
below $d_c$.

**The box side is declared, and the exponent it is declared with is the load-bearing
choice.** $L(x) = \lceil \text{box\_factor}\cdot x^{\nu_{\text{box}}}\rceil$, so
$L/\xi \sim x^{\nu_{\text{box}}-\nu}$:

| | $L/\xi$ along the ladder | effect on the fit |
|---|---|---|
| $\nu_{\text{box}} < \nu$ | shrinks | finite-size bias GROWS with $x$ — biases $\gamma$ down |
| $\nu_{\text{box}} = \nu$ | constant | bias is a constant FACTOR — moves $a_0$, leaves $\gamma$ alone |
| $\nu_{\text{box}} > \nu$ | grows | bias shrinks along the ladder — $\gamma$ slightly up, bounded by the first rung |

The default $1.5$ is $\ge\nu$ in every dimension ($\nu = 4/3$ at $\dim=2$, falling to
$1/2$ at $\dim\ge6$), i.e. safe without consulting a table of $\nu$. This is
`percolation_tau.py`'s `DF_LOWER` pattern, and it got the same treatment: measured, not
asserted. `experiments/06_susceptibility/calibrate_box.py` swept `box_factor` at three
rungs and found the finite-size deficit FLAT in $x$ at each `box_factor`
($-55\%$, $-18\%$, $-2\%$ at $1, 2, 4$ across $x = 8, 16, 32$) — i.e. a $-55\%$ box is
still an unbiased *exponent* measurement. The prompt's own rule ("increase $L$ until
$S(p,L)$ stabilizes") is deliberately NOT implemented: per sample it is a data-dependent
stopping rule, biased and with no cost knowable before the draw.

`cost_hint(x) = L(x)**dim` with the actual ceil'd $L$, a power law with declared exponent
$\nu_{\text{box}}\cdot\dim$ ($=8/3$ at the defaults) — so Assumption 7 holds by
construction and the cost probe has a number to score against, as for `srw` ($d=1$) and
`percolation_zd` ($d=\dim$).

Verified: `tools/tests/test_percolation_susceptibility.py` (18 cases) — the reduction
against a pure-Python BFS flood fill sharing no code with it (6 combinations of $\dim$,
$L$, moment, geometry); the mean against exhaustive enumeration of all $2^9$ torus
configurations, with the resolution stated so "agrees" is bounded; the mean against the
low-density lattice-animal series $S = p^{-1}\sum_A s^2p^s(1-p)^{t(A)}$ (animals to size
7), which is the infinite-lattice quantity the model actually claims; $p\to0$, $p=1$
($Y = L^{\dim}$ exactly), `block_n` invariance, the $\dim\ge3$ ladder refusal, cost-hint
exactness, and the $L/\xi$ growth law above.
