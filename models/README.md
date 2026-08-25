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
usable `true_gamma_key`. `cost_hint(i) = 1` — drawing a sample is a closed-form
evaluation plus one noise draw, constant in $i$, so $d=0$ and this model is
deliberately useless for testing the budget machinery. Verified indirectly via `tools/tests/test_loglog.py`
(checkpoint 0.2's noiseless-recovery checks) and `tools/tests/test_models.py`.

Neither file imports the other, or anything from `tools/`/`src/`/`experiments/` --
`tools/models.py` is the only thing that imports these, and it does so by adding this
directory to `sys.path` and importing `srw`/`synthetic` as bare top-level modules
(never through the name `models`, to avoid colliding with its own identity as
`tools/models.py` — see that file's docstring for the full explanation).

Adding a new model: write `models/<name>.py` with at least a `simulate` function
(plus a `cost_hint` if the budget machinery is to be used on it), then add one
`ModelSpec(...)` entry to `tools/models.py`'s `MODELS` dict. No changes needed to `src/generate/generate.py`/`plot_loglog.py`/`measure_cost.py`/`plot_cost.py`.
