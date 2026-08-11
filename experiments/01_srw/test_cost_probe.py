"""Cost-model probe's numeric acceptance criterion (see README), made executable.

Unlike the rest of this repo's tests, this one runs real wall-clock timing --
not instant, and in principle sensitive to a heavily loaded machine. That's a
deliberate, accepted exception here (see measure_cost.py's module docstring
for why min-of-repeats is used to resist ordinary OS/interpreter jitter),
not an oversight."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_cost import ACCEPTANCE_RANGE, DEFAULT_REPEATS, DEFAULT_SCALES, measure


def test_srw_cost_exponent_recovers_linear():
    result = measure(DEFAULT_SCALES, DEFAULT_REPEATS, seed=0)
    lo, hi = ACCEPTANCE_RANGE
    assert lo <= result["d_hat"] <= hi, (
        f"expected d_hat in [{lo}, {hi}] (ground truth d=1 for Theta(k) srw()), got {result['d_hat']}"
    )
