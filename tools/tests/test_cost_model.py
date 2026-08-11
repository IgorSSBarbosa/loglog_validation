"""Verifies estimate_cost_exponent against synthetic, noiseless cost curves --
decoupled from real timing noise (that's experiments/01_srw/test_cost_probe.py's
job), mirroring how test_loglog.py checks noiseless recovery of gamma."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from cost_model import COST_ESTIMATORS, estimate_cost_exponent


def test_recovers_d_exactly_noiseless():
    scales = np.array([4, 8, 16, 32, 64, 128], dtype=np.float64)
    for d in [0.5, 1.0, 1.7, 2.0]:
        elapsed = 3.0 * scales**d  # cost(i) = c * i**d, noiseless
        assert np.isclose(estimate_cost_exponent(scales, elapsed), d, atol=1e-9)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_cost_exponent([4, 8, 16], [1.0, 2.0, 4.0], method="nope")


def test_default_method_is_registered():
    assert "ols" in COST_ESTIMATORS
