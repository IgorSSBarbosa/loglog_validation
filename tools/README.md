# tools

Shared, experiment-agnostic utilities. May not import from `experiments/`. Every
function here needs a passing unit test in `tools/tests/` (checked against a closed
form, not just "runs") before any experiment is allowed to depend on it.

Not started yet — planned modules (see `PLAN.md` repo layout): `loglog.py`,
`wilson.py`, `allocation.py`, `bootstrap.py`, `rng.py`, `io.py`.
