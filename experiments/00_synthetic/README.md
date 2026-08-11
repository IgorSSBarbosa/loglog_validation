# 00_synthetic

**Status: generator built (`generator.py`); how checkpoints get verified going forward is
an open question** (the earlier `run_checkpoint_0_1.py` acceptance script was removed —
see PLAN.md discussion). See `PLAN.md` (repo root) "Phase 0" for the full checkpoint
table with acceptance criteria — reproduced here for convenience, kept in sync manually.

Model: $\mathbb{E} Y_i = a_0\, i^\gamma \exp\!\big(\sum_j a_j\, i^{-\omega_j}\big)$ (eq. 232, general
$J$), realized multiplicatively as $Y_i = \mathbb{E} Y_i \cdot \xi_i$, $\xi_i>0$, $\mathbb{E}\xi_i=1$,
$Var(\xi_i)=\sigma_i^2 \to \sigma_\infty^2$ by construction (e.g. lognormal noise).
Ground truth $(\gamma, a_0, (a_j,\omega_j)_j, \sigma_\infty^2)$ is planted and known, so
this rung tests only the *statistical* machinery, not model fidelity.

| # | Checkpoint | Acceptance criterion |
|---|---|---|
| 0.1 | Generator matches its own formula | empirical mean at large $i$ within 3 MC standard errors of $a_0 i^\gamma\exp(a_1 i^{-\omega_1})$ |
| 0.2 | Estimator is algebraically correct | weight identities exact to float precision; noiseless data recovers $\gamma$ exactly |
| 0.3 | CLT holds empirically | $\mathbb{E}\hat\gamma\approx\gamma$, $Var(\hat\gamma)$ matches eq. (583) within bootstrap CI, over fresh independent replicates |
| 0.4 | $\omega_1$-bootstrap calibrated | 95% CI for $\omega_1$ covers truth in $\approx$93-97% of $\gtrsim$200 independent synthetic experiments |
| 0.5 | Error-decay law | RMSE$(B)$ log-log slope matches $-\omega_1/(d+2\omega_1)$ under optimal allocation (eq. 945-946); Wilson CI (eq. 720) coverage checked |

`generator.py` — pluggable-noise-family generator (`NOISE_FAMILIES`; only `lognormal`
implemented so far). `SyntheticParams.corrections` holds an arbitrary-length sequence of
$(a_j,\omega_j)$ pairs, not just a single term.

Two kinds of JSON, not to be confused: a hand-authored **recipe** (see `example_config.json`
for the agreed Phase-0 defaults: $\gamma=0.5$, $a_0=1$, one correction term
$a_1=1,\omega_1=1$, $\sigma_\infty^2=0.04$, lognormal) is read-only and never modified by
anything here; running it produces **output** — samples as `<tag>.npz` (compressed) plus
their reproducibility metadata as `<tag>.json`, same stem, written to `data/` (gitignored —
regenerable, not source). CLI:

```
python3 generator.py -meta example_config.json --tag demo_run
python3 plot_loglog.py -data data/demo_run.npz
```

`plot_loglog.py` takes a **data path** (`-data`, the `.npz`), not a JSON — one recipe can
produce many different runs (different tags/seeds), so pointing it at a JSON would be
ambiguous about which run's data you mean. Metadata (for the reference-curve overlay and
`true_gamma`) is read from the same stem's `.json` if present, but isn't required — missing
metadata just means no overlay, not a failure to plot. It also writes
`images/<stem>_estimates.png` (via `tools/loglog_plot.py`'s `estimates_plot`) and
`data/<stem>_results.json`, comparing four $\hat\gamma$ estimators from `tools/loglog.py`
over the same data (all-points OLS, two-point/$m{=}2$, drop-leading-$m_0$ sweep, and a
maximum-likelihood estimator — see `derivations/mle_gamma_estimator.tex` for its derivation)
— see `tools/README.md` for what's implemented and what's still open (the article's exact
closed-form estimator).

`generate(..., out_dir=..., tag=...)` is the equivalent programmatic entry point for batch
use (content-hash filenames by default, so an identical rerun overwrites rather than
accumulating). `load_samples(path)` reads a `.npz` back directly (fast); `reproduce(path)`
regenerates from the recorded recipe instead, as an independent correctness check that
saved data matches what its recipe actually produces.

Checkpoint 0.2 (estimator) is next, once we've agreed how checkpoints get verified.
