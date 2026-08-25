# 00_synthetic

**Status: checkpoint 0.2 (closed-form estimator) done via `tools/tests/test_loglog.py`;
how the remaining checkpoints (0.1, 0.3-0.5) get verified going forward is still an open
question** (the earlier `run_checkpoint_0_1.py` acceptance script was removed — see
PLAN.md discussion; 0.2 was small enough to settle directly as ordinary pytest unit
tests instead of waiting on that decision). See `PLAN.md` (repo root) "Phase 0" for the
full checkpoint table with acceptance criteria — reproduced here for convenience, kept
in sync manually.

Model: $\mathbb{E} Y_i = a_0\, i^\gamma \exp\!\big(\sum_j a_j\, i^{-\omega_j}\big)$ (eq. 232, general
$J$), realized multiplicatively as $Y_i = \mathbb{E} Y_i \cdot \xi_i$, $\xi_i>0$, $\mathbb{E}\xi_i=1$,
$Var(\xi_i)=\sigma_i^2 \to \sigma_\infty^2$ by construction (e.g. lognormal noise).
Ground truth $(\gamma, a_0, (a_j,\omega_j)_j, \sigma_\infty^2)$ is planted and known, so
this rung tests only the *statistical* machinery, not model fidelity. Registered as
`MODELS["synthetic"]` in `tools/models.py`; the actual formula/noise-family code lives in
`models/synthetic.py`, not in this folder — see `models/README.md`.

| # | Checkpoint | Acceptance criterion |
|---|---|---|
| 0.1 | Generator matches its own formula | empirical mean at large $i$ within 3 MC standard errors of $a_0 i^\gamma\exp(a_1 i^{-\omega_1})$ |
| 0.2 | Estimator is algebraically correct | weight identities exact to float precision; noiseless data recovers $\gamma$ exactly — **done**, `tools/tests/test_loglog.py` |
| 0.3 | CLT holds empirically | $\mathbb{E}\hat\gamma\approx\gamma$, $Var(\hat\gamma)$ matches eq. (583) within bootstrap CI, over fresh independent replicates |
| 0.4 | $\omega_1$-bootstrap calibrated | 95% CI for $\omega_1$ covers truth in $\approx$93-97% of $\gtrsim$200 independent synthetic experiments |
| 0.5 | Error-decay law | RMSE$(B)$ log-log slope matches $-\omega_1/(d+2\omega_1)$ under optimal allocation (eq. 945-946); Wilson CI (eq. 720) coverage checked |

## Running this experiment

This experiment has no scripts of its own any more — `src/generate.py` and
`src/plot_loglog.py` are single, shared drivers used by every experiment (see
`src/README.md`), dispatching on `samples_example.json`'s `"model": "synthetic"` field.
Two kinds of JSON, not to be confused: the hand-authored **recipe**
(`samples_example.json`, holding the agreed Phase-0 defaults: $\gamma=0.5$, $a_0=1$, one
correction term $a_1=1,\omega_1=1$, $\sigma_\infty^2=0.04$, lognormal) is read-only and
never modified by anything here; running it produces **output** — one `data/<tag>/`
directory per run (gitignored — regenerable, not source), holding `samples.npz` +
`metadata.json` (now also recording per-scale `timing_seconds`, the raw material for a
future meta-log-log plot of $\mathrm{cost}(i)$ vs $i$; see `tools/cost_model.py`).

```
cd src   # repo root -> src/, see src/README.md
python3 generate.py -meta ../experiments/00_synthetic/recipes/samples_example.json --tag demo_run
python3 plot_loglog.py -data ../experiments/00_synthetic/data/demo_run --estimates
```

`plot_loglog.py` takes a **run directory** (`-data`), not a recipe or a bare file — one
recipe can produce many different runs (different tags/seeds), so pointing it at the
recipe would be ambiguous about which run's data you mean. `data/<tag>/plot.png` always
overlays the all-points OLS fit (solid line, needs no known truth) and, since this
model has a known closed form (`MODELS["synthetic"].target_fn`), also the reference
$\mathbb{E} Y_i$ curve (dashed) for comparison. `data/<tag>/results.json` (four
$\hat\gamma$ estimators from `tools/loglog.py`: all-points OLS, two-point/$m{=}2$,
drop-leading-$m_0$ sweep, and a maximum-likelihood estimator — see
`derivations/mle_gamma_estimator.tex` for its derivation) is always written; the
`--estimates` flag additionally saves `data/<tag>/estimates.png`, the four-estimator
comparison chart (same folder as `samples.npz`/`metadata.json`/`plot.png` — everything
about one run in one place; `data/` is gitignored, copy a plot into `images/` to keep
it as evidence) — see `tools/README.md` for what's implemented (as of checkpoint 0.2,
also the article's own closed-form weighted estimator, not currently wired into
`compare_methods` since it requires scales on a consecutive $\rho^k$ grid) and what's
still open.

`tools.generate.generate(..., out_dir=..., tag=...)` is the equivalent programmatic entry
point for batch use (content-hash tags by default, so an identical rerun overwrites
rather than accumulating a new run directory). `tools.persistence.load_samples(run_dir)`
reads `samples.npz` back directly (fast); `tools.generate.reproduce(run_dir)` regenerates
from the recorded recipe instead, as an independent correctness check that saved data
matches what its recipe actually produces.

Checkpoint 0.2 is done. Next open item: agree how the remaining checkpoints (0.1, 0.3-0.5)
get verified going forward (see TODO.md), then resume with whichever of those is picked.
