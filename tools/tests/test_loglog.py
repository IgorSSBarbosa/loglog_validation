"""Checkpoint 0.2 (PLAN.md): the closed-form weighted estimator is algebraically
correct -- weight identities to float precision, and exact gamma recovery on
noiseless data, for a spread of (m, m0)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from loglog import closed_form_weights, gamma_all_points, gamma_closed_form

MS = [2, 3, 4, 5, 8, 13]
M0S = [0, 1, 4]


def test_weights_shift_invariant():
    """Lemma 'Elementary identities' (a): w_{k,m} depends only on j = k - m0."""
    for m in MS:
        base = closed_form_weights(m, m0=0)
        for m0 in M0S:
            assert np.array_equal(closed_form_weights(m, m0), base)


def test_weights_sum_to_zero():
    """Identity (b), eq. 542."""
    for m in MS:
        assert np.isclose(closed_form_weights(m).sum(), 0.0, atol=1e-12)


def test_weights_dot_k_is_one():
    """Identity (c), eq. 542."""
    for m in MS:
        w = closed_form_weights(m)
        for m0 in M0S:
            k = np.arange(m0 + 1, m0 + m + 1)
            assert np.isclose(w @ k, 1.0, atol=1e-12)


def test_weights_sum_of_squares():
    """Identity (d)."""
    for m in MS:
        w = closed_form_weights(m)
        assert np.isclose((w**2).sum(), 12.0 / (m * (m**2 - 1)))


def test_weights_max_abs():
    """Identity (e)."""
    for m in MS:
        w = closed_form_weights(m)
        assert np.isclose(np.abs(w).max(), 6.0 / (m * (m + 1)))


def test_noiseless_recovery_exact():
    """On a noiseless planted power law, gamma_closed_form recovers gamma to
    float precision, for a spread of (m, m0) -- checkpoint 0.2's acceptance
    criterion (PLAN.md)."""
    gamma, a0, rho = 0.5, 3.0, 2.0
    for m in MS:
        for m0 in M0S:
            k = np.arange(m0 + 1, m0 + m + 1)
            scales = rho**k
            y_bar = a0 * scales**gamma  # E[Y] = a0 * i**gamma, noiseless
            gamma_hat = gamma_closed_form(scales, y_bar, rho, m0=m0)
            assert np.isclose(gamma_hat, gamma, atol=1e-9)


def test_agrees_with_ols_on_consecutive_grid():
    """On a consecutive rho**k grid, the closed form and generic OLS slope
    coincide (they're the same estimator -- see module docstring)."""
    rho, m, m0 = 2.0, 6, 2
    k = np.arange(m0 + 1, m0 + m + 1)
    scales = rho**k
    rng = np.random.default_rng(0)
    y_bar = scales**0.7 * np.exp(0.05 * rng.standard_normal(m))
    assert np.isclose(
        gamma_closed_form(scales, y_bar, rho, m0=m0),
        gamma_all_points(scales, y_bar),
    )


def test_rejects_non_consecutive_grid():
    with pytest.raises(ValueError):
        gamma_closed_form([2**10, 2**15, 2**20], [1.0, 2.0, 3.0], rho=2.0)
