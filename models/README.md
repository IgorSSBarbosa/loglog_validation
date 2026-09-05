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
| `percolation2d` | south-connected sites | $i^2$ | 2 | $2.029 \pm 0.018$ (box) | no |
| `percolation_tau` | clusters at size scale $s$, per site | $L(s)^2$ | $2\,$`box_exponent` (1) | $1.049 \pm 0.019$ (torus) | no |

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
