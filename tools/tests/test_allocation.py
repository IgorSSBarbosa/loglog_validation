"""Checks tools/allocation.py: the optimal allocation's exponents satisfy
Lemma lem:budget's feasibility constraint with equality (the proof's
"feasibility is active at the optimum"), the continuous allocation's total
cost matches the budget B exactly (Lemma lem:budget's closed-form cost
claim, combined with Proposition prop:opt), and the integer-discretized
allocation this module actually returns never exceeds B."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from allocation import feasible, optimal_allocation, total_cost

DM_PAIRS = [(1.0, 0.5), (2.0, 1.0), (0.5, 2.0), (1.5, 0.3)]


@pytest.mark.parametrize("d,omega1", DM_PAIRS)
def test_optimal_theta_active_at_feasibility_boundary(d, omega1):
    result = optimal_allocation(B=1e6, d=d, omega1=omega1, rho=2.0, m=5)
    assert np.isclose(result["theta1"] + d * result["theta2"], 1.0)
    assert result["theta_feasible"]
    assert feasible(result["theta1"], result["theta2"], d)


@pytest.mark.parametrize("B", [1e3, 1e5, 1e8])
def test_continuous_allocation_cost_equals_budget_exactly(B):
    """Lemma lem:budget's closed-form cost, at the Proposition's own
    (continuous, unrounded) n, m0, equals B**(theta1+d*theta2) = B**1 = B."""
    d, m, rho = 1.5, 4, 2.0
    result = optimal_allocation(B=B, d=d, omega1=0.7, rho=rho, m=m)
    cost = total_cost(result["n_exact"], result["m0_exact"], m, rho, d)
    assert np.isclose(cost, B, rtol=1e-9)


def test_feasible_discretized_allocation_never_exceeds_budget():
    """Holds whenever n_exact >= 1 (see module docstring's monotonicity
    argument); B=100 here already gives n_exact ~= 1.67 for this config."""
    for B in [100, 1e4, 1e6, 1e8]:
        result = optimal_allocation(B=B, d=1.0, omega1=0.5, rho=2.0, m=2)
        assert result["integer_feasible"]
        assert result["cost"] <= B + 1e-6
        assert result["n"] >= 1
        assert result["m0"] >= 0


def test_too_small_budget_flags_infeasible_not_overspending():
    """B=10 here gives continuous n_exact ~= 0.23 < 1: forcing n=1 would cost
    28 against a budget of 10. Must flag integer_feasible=False (n/m0/cost
    None) rather than silently overspend -- continuous quantities (theta1,
    theta2, n_exact, m0_exact) stay well-defined and returned regardless."""
    result = optimal_allocation(B=10, d=1.0, omega1=0.5, rho=2.0, m=3)
    assert not result["integer_feasible"]
    assert result["n"] is None
    assert result["m0"] is None
    assert result["cost"] is None
    assert result["n_exact"] < 1
    assert np.isclose(result["theta1"] + 1.0 * result["theta2"], 1.0)


def test_larger_budget_gives_more_samples_and_higher_offset():
    small = optimal_allocation(B=1e3, d=1.0, omega1=0.5, rho=2.0, m=4)
    large = optimal_allocation(B=1e9, d=1.0, omega1=0.5, rho=2.0, m=4)
    assert large["n"] > small["n"]
    assert large["m0"] >= small["m0"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": 0, "omega1": 0.5, "rho": 2.0, "m": 5},
        {"d": 1.0, "omega1": 0, "rho": 2.0, "m": 5},
        {"d": 1.0, "omega1": 0.5, "rho": 1.0, "m": 5},
        {"d": 1.0, "omega1": 0.5, "rho": 2.0, "m": 0},
        {"d": 1.0, "omega1": 0.5, "rho": 2.0, "m": 5, "B": 0.1},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    kwargs.setdefault("B", 1e6)
    with pytest.raises(ValueError):
        optimal_allocation(**kwargs)
