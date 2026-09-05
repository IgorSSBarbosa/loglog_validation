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

---

No model file imports another, or anything from `tools/`, `src/` or `experiments/` —
`tools/models.py` is the only thing that imports these, as `from models import srw`. Note
the two names that look alike and are not: `models` is this package of simulators,
`tools.models` is the registry that indexes them. Writing both out in full is what keeps
them apart.
