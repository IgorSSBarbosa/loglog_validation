"""Correctness checks for srw's vectorized (n, k) -> n samples change."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from srw import srw


def test_shape_is_n():
    rng = np.random.default_rng(0)
    for n, k in [(1, 15), (5, 15), (1000, 50)]:
        assert srw(k, n=n, rng=rng).shape == (n,)


def test_values_bounded_and_parity_matches_k():
    """|S_k| in [0, k], and shares k's parity (sum of k +-1's)."""
    rng = np.random.default_rng(1)
    for k in [7, 8, 50]:
        vals = srw(k, n=200, rng=rng)
        assert np.all(vals >= 0) and np.all(vals <= k)
        assert np.all((vals - k) % 2 == 0)


def test_mean_matches_known_asymptotic():
    """E|S_k| ~ sqrt(2k/pi) for large k (classical SRW asymptotic)."""
    rng = np.random.default_rng(2)
    k, n = 2000, 20000
    vals = srw(k, n=n, rng=rng)
    expected = np.sqrt(2 * k / np.pi)
    assert np.isclose(vals.mean(), expected, rtol=0.05)
