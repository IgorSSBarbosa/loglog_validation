"""Simple random walk simulator -- see README's "Cost-model probe" section.

NOT Phase 1's gamma-estimation testbed. That remains blocked pending a
closed-form E[Y_i]/gamma/omega1 for the article's appendix-SimpleRandomWalk
(see PLAN.md "Open questions before Phase 1"). Here `srw` is used only as a
simulator whose per-call cost genuinely grows with scale -- unlike
experiments/00_synthetic/generator.py's draws, whose cost is ~constant in i
-- a fixture for validating tools/cost_model.py's estimator against a known
ground truth: generating k i.i.d. +-1 steps and summing them is Theta(k)
work, so the measured cost exponent d should recover close to 1.
"""

from __future__ import annotations

import numpy as np


def srw(k: int, n: int = 1, q: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of |S_k|, S_k = sum of k i.i.d. +-1 steps, P(+1)=q.

    Vectorized: draws one (n, k) matrix of steps and sums along the k axis
    (rather than looping n times), returning an array of n entries.

    `rng` defaults to a fresh, unseeded Generator if omitted; pass an
    explicit seeded one for reproducible runs (see measure_cost.py, which
    calls this with the default n=1 -- Assumption cost_is_power_law defines
    cost(i) as the cost of simulating *one* sample).
    """
    rng = rng if rng is not None else np.random.default_rng()
    steps = rng.choice([-1, 1], size=(n, k), p=[1 - q, q])
    return np.abs(steps.sum(axis=1))


if __name__ == "__main__":
    print(srw(15, n=5))
