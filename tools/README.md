# tools

Shared, experiment-agnostic utilities. May not import from `experiments/`. Every
function here needs a passing unit test in `tools/tests/` (checked against a closed
form, not just "runs") before any experiment is allowed to depend on it.

`loglog_plot.py` — generic log-log plot of $\overline Y_i$ vs $i$ (any `{scale:
samples}` dict, from any experiment) with $\pm 1$ SE error bars and an optional
overlay of a known $\E Y_i$ curve. No unit test yet (it's a plot, not a
numeric claim) — visually verified against `experiments/00_synthetic/`.

Not started yet — planned modules (see `PLAN.md` repo layout): `loglog.py` (the
weighted OLS *estimator*, eq. 523-531 — distinct from `loglog_plot.py` above),
`wilson.py`, `allocation.py`, `bootstrap.py`, `rng.py`, `io.py`.
