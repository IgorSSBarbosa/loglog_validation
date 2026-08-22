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


def allocation_constants(d: float, omega1: float, rho: float, m: int,
                         a1: float, cv: float) -> dict:
    """The multiplicative constants Proposition prop:opt drops, in closed form.

    prop:opt is a RATE result: m0 = theta2 * log_rho(B) is correct up to an
    additive constant in m0, which the rate argument discards. Experiment C
    measured that constant to be ~ -3.3 for srw at (d, omega1, rho, m) =
    (1, 1, 2, 6) -- worth a factor 2.2-2.4 in RMSE, i.e. ~10x in budget. This
    function recovers it analytically. Writing the two error sources as

        |bias| = Cb * rho**(-m0*omega1)        (independent of n)
        sd     = Cs * n**(-1/2)                (independent of m0)
        B      = n * rho**(d*m0) * G           (Lemma lem:budget)

    the pieces are, with w_j the article's weights (eq. 526) and j = k - m0:

        Cb = |a1 * sum_j w_j rho**(-j*omega1)| / log(rho)
        Cs = cv * ||w|| / log(rho),   ||w||**2 = 12/(m(m**2-1))
        G  = rho**d * (rho**(d*m) - 1) / (rho**d - 1)

    `a1` is the correction amplitude of eq. (232) and `cv` the observable's
    coefficient of variation sd(Y_i)/E(Y_i) (assumed scale-free, which holds
    for srw's |S_k|: cv = sqrt(pi/2 - 1)). Both come from measurement --
    Experiment B supplies a1, and cv is read straight off the samples -- so
    this is a calibration, not extra theory.

    Returns Cb, Cs, G, kappa and `offset`, where

        m0_tuned = theta2 * (log_rho(B) + log_rho(kappa)) = m0_prop_opt + offset.
    """
    import numpy as np

    if m < 2:
        raise ValueError(f"m must be >= 2 for the weights to exist; got {m}")
    if rho <= 1:
        raise ValueError(f"rho must be > 1; got {rho}")
    if d <= 0 or omega1 <= 0:
        raise ValueError(f"d and omega1 must be > 0; got d={d}, omega1={omega1}")
    if cv <= 0:
        raise ValueError(f"cv must be > 0; got {cv}")

    j = np.arange(1, m + 1, dtype=float)
    w = 12.0 * (j - (m + 1) / 2.0) / (m * (m**2 - 1))
    w_norm_sq = float(np.sum(w**2))

    Cb = abs(a1 * float(np.sum(w * rho ** (-j * omega1)))) / math.log(rho)
    Cs = cv * math.sqrt(w_norm_sq) / math.log(rho)
    G = rho**d * (rho ** (d * m) - 1) / (rho**d - 1)

    theta2 = 1.0 / (d + 2 * omega1)
    if Cb == 0.0:
        # No correction term at all (a1 = 0): there is no bias to trade
        # against variance, so the balance that fixes m0 does not exist.
        return {"Cb": 0.0, "Cs": Cs, "G": G, "w_norm_sq": w_norm_sq,
                "kappa": None, "offset": None, "theta2": theta2}

    kappa = 2 * omega1 * Cb**2 / (d * Cs**2 * G)
    return {
        "Cb": Cb,
        "Cs": Cs,
        "G": G,
        "w_norm_sq": w_norm_sq,
        "kappa": kappa,
        "offset": theta2 * math.log(kappa) / math.log(rho),
        "theta2": theta2,
    }


def predict_error(n: float, m0: float, d: float, omega1: float, rho: float,
                  m: int, a1: float, cv: float) -> dict:
    """Predicted |bias|, sd and RMSE of gamma-hat for a given (n, m0) ladder."""
    c = allocation_constants(d, omega1, rho, m, a1, cv)
    bias = c["Cb"] * rho ** (-m0 * omega1)
    sd = c["Cs"] / math.sqrt(n)
    return {"bias": bias, "sd": sd, "rmse": math.sqrt(bias**2 + sd**2)}


def tuned_allocation(B: float, d: float, omega1: float, rho: float, m: int,
                     *, a1: float, cv: float) -> dict:
    """prop:opt's allocation with its dropped multiplicative constant restored.

    Same rate as `optimal_allocation` -- this only shifts m0 by the constant
    `allocation_constants` computes, which is exactly what the rate theorem
    leaves free. Verified against Experiment C's wide sweep (B = 1e4..1e9,
    R = 40, tag `allocation_wide`): the tuned m0 lands within ONE step of the
    measured argmin at all six budgets, costing 1.00-1.02x RMSE against the
    best m0 on the grid, where prop:opt's own m0 costs 2.02-3.35x.

    Quote that as a range, not as exact integers: which of two near-tied m0
    wins is itself noisy at R = 40, because the RMSE curve is flat near its
    minimum (at B = 1e6 the two best differ by 0.03%). An earlier 3-budget
    run gave argmins 3/5/6 at B = 1e7/1e8/1e9; the wide run gives 4/4/6.

    Returns the continuous m0, the floored integer actually usable, the
    resulting n, the realized cost, and the predicted error decomposition.
    `integer_feasible` is False when n floors below 1 -- check it, as with
    `optimal_allocation`.
    """
    if B < 1:
        raise ValueError(f"B must be >= 1; got {B}")
    c = allocation_constants(d, omega1, rho, m, a1, cv)
    if c["offset"] is None:
        raise ValueError("a1 == 0: no correction term, so no bias/variance balance to solve")

    m0_exact = c["theta2"] * math.log(B, rho) + c["offset"]
    # ROUND, not floor. `optimal_allocation` floors m0 because there the
    # budget guarantee needs it; here it does not. n is recomputed from
    # whatever integer m0 is chosen and then floored, so cost = n*G*rho^(d*m0)
    # <= B holds for ANY integer m0 -- leaving rounding free to pick the
    # nearer of the two candidates. It matters: against Experiment C's
    # measured argmins (3, 5, 6 at B = 1e7, 1e8, 1e9), rounding gives
    # (4, 5, 6) and flooring gives (3, 4, 5).
    m0 = max(0, int(round(m0_exact)))
    n = math.floor(B / (c["G"] * rho ** (d * m0)))
    feasible = n >= 1
    out = {
        "m0_exact": m0_exact,
        "m0": m0 if feasible else None,
        "n": n if feasible else None,
        "cost": total_cost(n, m0, m, rho, d) if feasible else None,
        "budget": float(B),
        "offset_vs_prop_opt": c["offset"],
        "integer_feasible": feasible,
        "constants": c,
    }
    if feasible:
        out.update(predict_error(n, m0, d, omega1, rho, m, a1, cv))
    return out


def neyman_allocation(
    scales,
    budget: float,
    d: float,
    *,
    sigma=None,
    min_n: int = 1,
) -> dict:
    """Per-scale sample counts n_i minimizing total variance at fixed budget.

    NOT Proposition prop:opt, and not from the article at all. The two rules
    answer different questions and must not be swapped:

    - `optimal_allocation` (prop:opt) minimizes the error of gamma-hat. The
      correction term a_1 i^-omega_1 is pure nuisance there, so the rule
      SLIDES THE SCALE WINDOW UPWARD as the budget grows, abandoning the
      small scales where that correction bites, and uses a uniform n.
    - This rule is for measuring omega_1 itself (Experiment B,
      plans/three_experiment_ladder.md section 3). That measurement needs the
      small scales KEPT, because they are the only place the correction is
      big enough to see. Applying prop:opt here would spend the budget
      precisely where the signal is not.

    Minimizing sum_i sigma_i^2 / n_i subject to sum_i n_i * cost(i) = budget,
    with cost(i) = i**d, gives (Lagrange multiplier, continuous relaxation)

        n_i  proportional to  sigma_i / sqrt(cost(i))  =  sigma_i * i**(-d/2),

    i.e. cheap small scales get many more replicates than expensive large
    ones. `sigma` defaults to constant across scales (reasonable for
    log Y_bar, whose variance is roughly scale-free); pass per-scale values
    to weight it.

    Returns {'scales', 'n', 'cost', 'budget', 'exhausted'}. Counts are
    floored and clamped to `min_n`, so the realized cost can exceed `budget`
    when the clamp binds -- `exhausted` reports cost/budget so the caller can
    check rather than assume (same "check the diagnostic" pattern as
    `integer_feasible` above).
    """
    import numpy as np

    i = np.asarray(scales, dtype=float)
    if i.size == 0:
        raise ValueError("scales must be non-empty")
    if np.any(i <= 0):
        raise ValueError("scales must be strictly positive")
    if d <= 0:
        raise ValueError(f"d must be > 0 (Assumption cost_is_power_law); got {d}")
    if budget <= 0:
        raise ValueError(f"budget must be > 0; got {budget}")
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1; got {min_n}")

    s = np.ones_like(i) if sigma is None else np.asarray(sigma, dtype=float)
    if s.shape != i.shape:
        raise ValueError("sigma must match scales in length")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")

    cost = i**d
    weights = s / np.sqrt(cost)
    # Scale the weights so the continuous allocation costs exactly `budget`.
    n_exact = weights * (budget / float(np.sum(weights * cost)))
    n = np.maximum(np.floor(n_exact).astype(np.int64), min_n)
    realized = float(np.sum(n * cost))

    return {
        "scales": [int(x) for x in np.asarray(scales)],
        "n": [int(x) for x in n],
        "cost": realized,
        "budget": float(budget),
        "exhausted": realized / float(budget),
    }


def snr_allocation(
    scales,
    budget: float,
    d: float,
    omega1: float,
    *,
    sigma=None,
    min_n: int = 1,
) -> dict:
    """Per-scale n_i equalizing the signal-to-noise ratio of the CORRECTION term.

    This is Experiment B's corrected rule, and it supersedes
    `neyman_allocation` for measuring omega_1 (measured 2026-08-20, see
    plans/three_experiment_ladder.md section 3).

    Neyman minimizes the variance of Y_bar_i. But omega_1 is not estimated
    from Y_bar_i -- it is estimated from the small correction a_1 * i^-omega_1
    riding on top of it, whose size SHRINKS with i. Equalizing the error of
    Y_bar therefore over-samples the small scales, where the correction is
    already resolved hundreds of times over, and starves the large scales,
    where it has sunk below the noise floor. Measured on the first Experiment
    B run (Neyman, budget 2e9, scales 2..1024), the correction-term SNR ran
    from 460 at k=2 down to 0.25 at k=1024 -- three orders of magnitude of
    imbalance, and the large scales contributed nothing but noise to the fit.

    Writing sd(log Y_bar_i) = s_i / sqrt(n_i), the correction's SNR at scale i
    is |a_1| i^-omega1 sqrt(n_i) / s_i. Holding that constant across scales,

        n_i  proportional to  s_i**2 * i**(2*omega1),

    which for omega1 = 1 puts n_i proportional to i**2 -- INCREASING in i, the
    opposite of Neyman's i**(-d/2). `omega1` here is a design input (the value
    you expect, or a prior guess); the allocation is not sensitive to getting
    it exactly right, but it does need the right sign of the trend.

    `sigma` is s_i, the per-scale coefficient of variation sd(Y_i)/E(Y_i); it
    defaults to constant, which holds well for srw's |S_k| (the coefficient of
    variation of |S_k| is scale-free in k).

    Returns the same shape as `neyman_allocation`, including `exhausted` --
    check it rather than assuming the budget was respected.
    """
    import numpy as np

    i = np.asarray(scales, dtype=float)
    if i.size == 0:
        raise ValueError("scales must be non-empty")
    if np.any(i <= 0):
        raise ValueError("scales must be strictly positive")
    if d <= 0:
        raise ValueError(f"d must be > 0 (Assumption cost_is_power_law); got {d}")
    if omega1 <= 0:
        raise ValueError(f"omega1 must be > 0 (article eq. 232); got {omega1}")
    if budget <= 0:
        raise ValueError(f"budget must be > 0; got {budget}")
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1; got {min_n}")

    s = np.ones_like(i) if sigma is None else np.asarray(sigma, dtype=float)
    if s.shape != i.shape:
        raise ValueError("sigma must match scales in length")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")

    cost = i**d
    weights = s**2 * i ** (2.0 * omega1)
    n_exact = weights * (budget / float(np.sum(weights * cost)))
    n = np.maximum(np.floor(n_exact).astype(np.int64), min_n)
    realized = float(np.sum(n * cost))

    return {
        "scales": [int(x) for x in np.asarray(scales)],
        "n": [int(x) for x in n],
        "cost": realized,
        "budget": float(budget),
        "exhausted": realized / float(budget),
    }


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
