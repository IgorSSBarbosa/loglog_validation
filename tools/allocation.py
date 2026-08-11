"""Budget allocation rule: Definition def:alloc / Proposition prop:opt
(eq. 945-946) + cost accounting (Lemma lem:budget).

Given a compute budget B, cost-model exponent d (cost(i) = i**d, Assumption
cost_is_power_law -- see tools/cost_model.py for measuring it), correction
exponent omega1, scale ratio rho, and window size m, the rate-optimal
balanced allocation (Proposition prop:opt) samples n i.i.d. replicates at
each of the m scales k = m0+1, ..., m0+m, with

    theta1 = 2*omega1 / (d + 2*omega1)
    theta2 = 1 / (d + 2*omega1)

    n  = kappa * B**theta1,   kappa = (rho**d - 1) / (rho**(d*(m+1)) - rho**d)
    m0 = theta2 * log_rho(B)

giving the error-decay rate |beta_hat - beta| ~ B**(-omega1/(d+2*omega1))
(eq. errordecay). This is the exponents' proof-derived value: feasibility
(Lemma lem:budget, theta1 + d*theta2 <= 1) is active at the optimum, i.e.
theta1 + d*theta2 == 1 exactly here.

Discretization (not addressed by the theorem, which treats n/m0 as
continuous): n and m0 must be integers to name actual scales/sample counts.
This module floors both. Since total_cost is increasing in both n and m0
(rho > 1, d > 0), and the continuous (n_exact, m0_exact) costs exactly B
(Lemma lem:budget's own closed form, combined with feasibility being active
at the optimum -- see total_cost's docstring), flooring both can only move
cost downward: floor(n) <= n_exact and floor(m0) <= m0_exact together
guarantee total_cost(floor(n), floor(m0)) <= B -- but only when
n_exact >= 1 (floor(n_exact) >= 1 is then a real, budget-respecting sample
count). When n_exact < 1, there is no positive integer n at this scale
offset that respects B: `optimal_allocation` still returns the continuous
quantities (theta1/theta2/kappa/n_exact/m0_exact are well-defined for any
valid B, d, omega1, rho, m), but flags this with `integer_feasible=False`
and `n`/`m0`/`cost` set to None, rather than silently forcing n=1 and
overspending -- the same "diagnostics that must be checked before trusting
the result" pattern `gamma_mle` uses (see tools/loglog.py), not a raise,
since the continuous math itself isn't wrong, only the integer allocation
at this particular B.
"""

from __future__ import annotations

import math


def total_cost(n: float, m0: float, m: int, rho: float, d: float) -> float:
    """Total cost of Definition def:alloc's allocation (Lemma lem:budget):
    sum_{k=m0+1}^{m0+m} n*cost(rho**k), cost(i)=i**d, via the closed-form
    geometric sum n * rho**(d*(m0+1)) * (rho**(d*m)-1)/(rho**d-1).

    Works for continuous m0 (verifying the Lemma's own B**(theta1+d*theta2)
    claim at the exact, unrounded allocation) as well as the integer m0 an
    actual allocation uses.
    """
    return n * rho ** (d * (m0 + 1)) * (rho ** (d * m) - 1) / (rho**d - 1)


def feasible(theta1: float, theta2: float, d: float, *, tol: float = 1e-9) -> bool:
    """Budget-constraint feasibility (Lemma lem:budget): theta1 + d*theta2 <= 1."""
    return theta1 + d * theta2 <= 1.0 + tol


def optimal_allocation(B: float, d: float, omega1: float, rho: float, m: int) -> dict:
    """Rate-optimal allocation (Proposition prop:opt, eq. 945-946).

    Raises ValueError on parameters the formulas require: d > 0 (Assumption
    cost_is_power_law; d == 0 makes kappa's denominator vanish), omega1 > 0
    (article eq. 232's correction ordering), rho > 1 (scale ratio), m >= 1
    (window size), B >= 1 (budget).

    Returns the optimal exponents (theta1, theta2, kappa), the continuous
    n_exact/m0_exact the formula gives, `theta_feasible` (should always be
    True here -- a self-check, not a decision), and `integer_feasible`: when
    True, `n`/`m0`/`cost` are the floored, actually-usable allocation (cost
    <= B guaranteed, see module docstring); when False (B too small for this
    d/omega1/rho/m to admit any positive-integer allocation), `n`/`m0`/`cost`
    are None -- check `integer_feasible` before using them, the same
    pattern `gamma_mle`'s `trustworthy` uses.
    """
    if d <= 0:
        raise ValueError(f"d must be > 0 (Assumption cost_is_power_law); got {d}")
    if omega1 <= 0:
        raise ValueError(f"omega1 must be > 0 (article eq. 232); got {omega1}")
    if rho <= 1:
        raise ValueError(f"rho must be > 1; got {rho}")
    if m < 1:
        raise ValueError(f"m must be >= 1; got {m}")
    if B < 1:
        raise ValueError(f"B must be >= 1; got {B}")

    theta1 = 2 * omega1 / (d + 2 * omega1)
    theta2 = 1 / (d + 2 * omega1)

    kappa = (rho**d - 1) / (rho ** (d * (m + 1)) - rho**d)
    n_exact = kappa * B**theta1
    m0_exact = theta2 * math.log(B, rho)  # >= 0 always, since B >= 1 and rho > 1

    integer_feasible = n_exact >= 1
    n = math.floor(n_exact) if integer_feasible else None
    m0 = math.floor(m0_exact) if integer_feasible else None
    cost = total_cost(n, m0, m, rho, d) if integer_feasible else None

    return {
        "theta1": theta1,
        "theta2": theta2,
        "kappa": kappa,
        "n_exact": n_exact,
        "m0_exact": m0_exact,
        "n": n,
        "m0": m0,
        "cost": cost,
        "budget": B,
        "theta_feasible": feasible(theta1, theta2, d),
        "integer_feasible": integer_feasible,
    }
