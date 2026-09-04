# models

Per-model simulation logic, one file per registered model (user's own framing,
2026-08-12: kept separate from both `tools/` — the helper functions models call, if
any — and `src/` — the scripts that call into models via `tools/models.py`'s
registry). Each file exposes:

- `simulate(i, n, params, rng) -> np.ndarray` — required, `n` i.i.d. samples at scale `i`.
- `target_fn(i, params) -> np.ndarray` — optional, the article's known closed-form
  $\mathbb{E} Y_i$. Only present when there actually is an article-sanctioned closed
  form; its *absence* is what keeps `src/report/plot_loglog.py` from overlaying a
  reference curve or reporting a `true_gamma` for a model (the $\hat\gamma$
  estimators themselves still run either way, flagged exploratory when there's
  nothing known to check them against) — not a special case coded into any driver.
- `cost_hint(i, params) -> float` — optional, the work one sample at scale $i$ costs,
  in whatever unit the model counts in (steps, sites explored, operations). This is
  what makes a model usable by the budget machinery: it is a *declared* quantity, so
  $d$ is known rather than fitted, and `src/estimate/measure_cost.py` scores the
  measured wall-clock $d$ against it (`tools/cost_model.py`'s `compare_cost_models`).
  Both are kept deliberately — the declared count is exact where the clock is not
  (throughput varied 8.6x across scales on this machine), but only the clock can
  notice that a simulator has stopped being compute-bound.

- `true_gamma_key` (declared in `tools/models.py`'s registry entry, not here) —
  which key in `params` holds the true $\gamma$, when known.

`srw.py` — `srw(k, n=1, q=0.5, rng=None, block_n=None)`, $n$ i.i.d. realizations of
$|S_k|$. Draws steps as `(block_n, k)` float32 blocks over the $n$ axis and
accumulates row sums, instead of one $(n,k)$ matrix — bounds peak transient memory to
a fixed byte budget (`_DEFAULT_WORKING_SET_BYTES`) regardless of how large $n$ gets
(the old unblocked, `int64` version needed 819 GiB at $n=10^8,\,k=1024$ —
`experiments/01_srw/recipes/samples_huge.json`, fixed 2026-08-19). Blocking is over $n$, not $k$,
deliberately: splitting the leading axis into sequential row ranges consumes numpy's
row-major RNG stream in the same order a single unblocked call would, so results are
bit-identical for the same seed at any block size — splitting over $k$ would not have
this property (see the module's own docstring for the full argument).

The per-step draw is `rng.random(size=..., dtype=np.float32) < q`, counting $+1$s and
mapping $S_k=2(\#{+1})-k$ — 4.4x faster than the `rng.choice(..., p=[1-q,q])` it
replaced (6.6 vs 28.3 µs/sample at $k=1024$). Two faster alternatives were tried and
deliberately rejected, both documented in `_draw_heads`: `rng.integers(0,2,dtype=int8)`
(fastest, but numpy packs several values per 64-bit draw and discards the leftover bits
per call, so row-blocking stops being exact — it breaks the bit-identity guarantee
above and `src/generate/generate.py`'s chunked path with it), and `rng.binomial(k,q,size=n)`
(~375x faster and distributionally identical, but samples every scale in ~constant
time, destroying the $\Theta(k)$ per-sample cost that is this model's entire reason for
existing as a percolation stand-in — user's call, 2026-08-20).

`cost_hint(i) = i`, exact rather than estimated: `srw` draws $i$ uniforms per sample
and sums them, with no early exit. Measured against the clock, declared $d=1$ vs
affine $\hat d = 1.0028\pm0.0020$ — a 0.28% gap.

No `target_fn` — deliberate, not a gap. $\mathbb{E}\lvert S_k\rvert$ is known exactly
(see `experiments/01_srw/README.md` for the formula and its verification, giving
$\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$), but by the user's decision (2026-08-20) those
values stay out of the code path and serve as hand-checked README acceptance criteria,
so the estimators are never incidentally handed the answer they are measuring.
Verified: `tools/tests/test_srw.py` (shape/bounds/parity, classical
$\mathbb E|S_k|\sim\sqrt{2k/\pi}$ asymptotic, `block_n` exact-equivalence with the
unblocked path, and a large-$(n,k)$ case that would be gigabytes unblocked).

`synthetic.py` — the closed-form model (`SyntheticParams`, `NOISE_FAMILIES`, `mean_Y`,
article eq. 232: $\mathbb{E} Y_i = a_0 i^\gamma \exp(\sum_j a_j i^{-\omega_j})$). Ground
truth is planted and known, so this is currently the only model with a `target_fn` /
usable `true_gamma_key`. Verified indirectly via `tools/tests/test_loglog.py`
(checkpoint 0.2's noiseless-recovery checks) and `tools/tests/test_models.py`.

With no burn configured, `cost_hint(i) = 1` — drawing a sample is a closed-form
evaluation plus one noise draw, constant in $i$, so $d=0$: not merely degenerate but
*outside* the allocation formulas (`allocation_constants` raises for $d\le0$, and
`total_cost`'s $G$ divides by $\rho^d-1$).

**The work burn** (`cost_scale`, `cost_d`, added 2026-09-04) gives it any cost exponent
you like: `simulate` spins for `cost_scale * i**cost_d` seconds per sample and
`cost_hint` declares exactly that, so the exponent is *known* and the clock has to find
it. Off by default. Three properties make it usable, each pinned by a test in
`tools/tests/test_synthetic.py`:

- it consumes **no randomness** (`perf_counter` only), so the same seed gives
  bit-identical draws with the burn on and off — the same invariance `srw` guarantees
  over `block_n`, and for the same reason;
- it **spins rather than sleeps**. `time.sleep` would void the budget model's meaning
  (`throughput` converts work to seconds and only means something while elapsed time is
  proportional to work done), permanently trip `compare_cost_models`' "no longer
  compute-bound" diagnostic, add ~1 ms of *additive* jitter into exactly the overhead
  term $a$ of $a+b\,i^d$, and give free speedup under any future parallel run. An
  array burn was rejected too: past L2/L3 it falls out of cache and the cost per
  element rises, biasing the realized exponent *above* the declared one;
- the burn **dominates dispatch** at the scales timed: sized at 1 ms against ~33 µs of
  per-call overhead, the fitted $a$ is ~2% of the cheapest probe rung.

One trap, recorded because it cost a wrong measurement: `cost_hint`'s no-burn return of
`1.0` is one *work unit*, not one second. Feeding it to the burn (written that way
first) made every unburnt call spin for a full second per sample. `simulate` does its
own arithmetic from `cost_scale`/`cost_d` for that reason.

This is what `calibration/check_no_leakage.py`'s `cost` arm runs on: recovering $d$ from
$\{0.5, 0.75, 1, 1.5, 2\}$ when the only $d$ anywhere in the repo's constants is 1.

`percolation2d.py` — critical site percolation on the square lattice.
`percolation2d(i, n=1, p=P_C_SQUARE_SITE, anchor="south", rng=None, block_n=None)`
fills an $i\times i$ box with i.i.d. Bernoulli($p$) open sites and returns, per sample,
the number of open sites connected to the **south side** by 4-connected open paths
inside the box — `PLAN.md` ground rule 7's observable, and the reason this model
exists. `anchor="origin"` returns the cluster of the box's centre instead; it is the
comparison arm for `src/estimate/compare_observables.py`, not the default.

$p_c = 0.59274605079210$ (Jacobsen 2015) is the **4-connected** threshold; passing the
8-connected `structure` by mistake would silently simulate a supercritical system
($p_c^{(8)} = 0.4073$), which is why `_FOUR_CONNECTED` is a named module constant with
that warning on it.

Two implementation choices carry the whole cost story:

- **One `ndimage.label` call per block, not per sample.** A block's lattices are
  stacked vertically into one tall image with a blank separator row after each; no open
  path can cross a closed row, so labelling the stack once is exactly the per-sample
  labelling, and the C-level union-find sees one large problem instead of $n$ tiny
  ones. That matters because a `neyman` allocation asks for ~500k samples at $i=8$,
  where per-call SciPy overhead would otherwise dominate. `test_percolation2d.py`'s
  `test_block_stacking_does_not_leak_between_samples` is what pins the separator.
- **float32 uniforms, blocked over the sample axis**, for exactly `srw.py`'s reasons:
  one draw per site with no bit packing, so splitting into blocks consumes the RNG
  stream in the same order an unblocked call would and the output is bit-identical at
  any `block_n`. The cost is a realized $p$ within $2^{-24}\approx6\times10^{-8}$ of
  the requested one — a distance to criticality whose correlation length
  ($\xi\sim|p-p_c|^{-4/3}\approx4\times10^9$ sites) is seven orders beyond any box this
  will run at.

`cost_hint(i) = i**2`, exact: $i^2$ uniforms drawn, one near-linear union-find pass,
nothing depending on $p$ or the anchor. **This is the first model where article
Assumption 7's $\mathrm{cost}(i)=i^d$ is a geometric fact about a real simulation
rather than a stated formula** — measured against the clock, declared $d=2$ vs affine
$\hat d = 2.0285\pm0.0175$, a $+1.63\sigma$ gap (`experiments/03_percolation_zd/`, P1).

No `target_fn`/`true_gamma_key`, the same deliberate absence as `srw`: $\gamma = d_f =
91/48$ is known from the literature but stays out of the code path and lives as a
written acceptance criterion in `experiments/03_percolation_zd/README.md`. $\omega_1$
is genuinely *unknown* here (literature $\Omega=72/91\approx0.79$; measured $\approx0.35$
over $8\le i\le512$, almost certainly an effective exponent) — this rung measures it.

Assumption 2 ($Y_i>0$) is not asserted, unlike `synthetic.py`. For `"south"` the
violation is real but exponentially rare — $Y_i = 0$ **iff** row 0 is entirely closed,
exactly $(1-p_c)^i$, which `zero_rate` returns in closed form and
`test_zero_rate_is_exact` checks — and for `"origin"` it is the normal case. An assert
would either fire once in ten million runs for no actionable reason, or make the
comparison arm unrunnable.

Verified: `tools/tests/test_percolation2d.py` — **exact enumeration** of
$\mathbb{E}Y_i$ over all $2^{i^2}$ configurations at $i=2,3$ (a closed form, not
another Monte Carlo), an independent pure-Python flood fill matching site-for-site on
random critical lattices, $p\in\{0,1\}$, $i=1$ (Bernoulli), `block_n` invariance, the
exact zero rate, and the origin arm's failure of Assumption 2.

Neither file imports the other, or anything from `tools/`/`src/`/`experiments/` --
`tools/models.py` is the only thing that imports these, as `from models import srw`.
Note the two names that look alike and are not: `models` is this package of
simulators, `tools.models` is the registry that indexes them. Writing both out in
full is what keeps them apart.

Adding a new model: write `models/<name>.py` with at least a `simulate` function
(plus a `cost_hint` if the budget machinery is to be used on it), then add one
`ModelSpec(...)` entry to `tools/models.py`'s `MODELS` dict. No changes needed to `src/generate/generate.py`/`plot_loglog.py`/`measure_cost.py`/`plot_cost.py`.
