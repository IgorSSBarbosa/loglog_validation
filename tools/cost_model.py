"""Cost-model exponent estimation: article Assumption cost_is_power_law states
cost(i) = i**d, "the computational complexity of simulating one sample of
distribution Y_i". This has exactly the same log-log-linear form as
E[Y_i] = a0 * i**gamma (eq. 232), so d is recoverable the same way gamma is --
`_ols_cost_exponent` just calls `loglog.gamma_all_points` with elapsed time
standing in for Y_i.

`COST_ESTIMATORS` is a name -> function registry (mirrors
experiments/00_synthetic/generator.py's NOISE_FAMILIES) so a different
estimation approach can be added later as one more entry, without touching
callers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

# Self-contained: works whether this module is reached as `tools.cost_model`
# (repo root on sys.path, e.g. experiments/*/measure_cost.py) or as a bare
# `cost_model` (tools/ itself on sys.path, e.g. tools/tests/test_loglog.py's
# convention) -- either way, `loglog` needs its own directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loglog import gamma_all_points  # noqa: E402


def _ols_cost_exponent(scales: Sequence, elapsed: Sequence) -> float:
    return gamma_all_points(scales, elapsed)


COST_ESTIMATORS: dict[str, Callable[[Sequence, Sequence], float]] = {
    "ols": _ols_cost_exponent,
}


def estimate_cost_exponent(scales: Sequence, elapsed: Sequence, method: str = "ols") -> float:
    """Estimate d from cost(i) = c * i**d, given elapsed[i] measured at each scales[i]."""
    if method not in COST_ESTIMATORS:
        raise ValueError(f"unknown method {method!r}; known: {list(COST_ESTIMATORS)}")
    return COST_ESTIMATORS[method](scales, elapsed)
