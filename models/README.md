# models

Per-model simulation logic, one file per registered model (user's own framing,
2026-08-12: kept separate from both `tools/` — the helper functions models call, if
any — and `src/` — the scripts that call into models via `tools/models.py`'s
registry). Each file exposes:

- `simulate(i, n, params, rng) -> np.ndarray` — required, `n` i.i.d. samples at scale `i`.
- `target_fn(i, params) -> np.ndarray` — optional, the article's known closed-form
  $\mathbb{E} Y_i$. Only present when there actually is an article-sanctioned closed
  form; its *absence* is what keeps `src/plot_loglog.py` from overlaying a
  reference curve or reporting a `true_gamma` for a model (the $\hat\gamma$
  estimators themselves still run either way, flagged exploratory when there's
  nothing known to check them against) — not a special case coded into any driver.
- `true_gamma_key` (declared in `tools/models.py`'s registry entry, not here) —
  which key in `params` holds the true $\gamma$, when known.

`srw.py` — `srw(k, n=1, q=0.5, rng=None)`, $n$ i.i.d. realizations of $|S_k|$
(vectorized: one $(n,k)$ matrix of $\pm1$ steps, summed along the $k$ axis). No
`target_fn`: no article-sanctioned closed form for SRW yet (`appendix-SimpleRandomWalk`
is still an empty stub — see `experiments/01_srw/README.md`, PLAN.md "Open questions
before Phase 1"). Verified: `tools/tests/test_srw.py` (shape/bounds/parity, classical
$\mathbb E|S_k|\sim\sqrt{2k/\pi}$ asymptotic).

`synthetic.py` — the closed-form model (`SyntheticParams`, `NOISE_FAMILIES`, `mean_Y`,
article eq. 232: $\mathbb{E} Y_i = a_0 i^\gamma \exp(\sum_j a_j i^{-\omega_j})$). Ground
truth is planted and known, so this is currently the only model with a `target_fn` /
usable `true_gamma_key`. Verified indirectly via `tools/tests/test_loglog.py`
(checkpoint 0.2's noiseless-recovery checks) and `tools/tests/test_models.py`.

Neither file imports the other, or anything from `tools/`/`src/`/`experiments/` --
`tools/models.py` is the only thing that imports these, and it does so by adding this
directory to `sys.path` and importing `srw`/`synthetic` as bare top-level modules
(never through the name `models`, to avoid colliding with its own identity as
`tools/models.py` — see that file's docstring for the full explanation).

Adding a new model: write `models/<name>.py` with at least a `simulate` function,
then add one `ModelSpec(...)` entry to `tools/models.py`'s `MODELS` dict. No changes
needed to `src/generate.py`/`plot_loglog.py`/`measure_cost.py`/`plot_cost.py`.
