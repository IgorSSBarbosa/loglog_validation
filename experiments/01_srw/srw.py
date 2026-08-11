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


def srw(k: int, q: float = 0.5, rng: np.random.Generator | None = None) -> int:
    """One realization of |S_k|, S_k = sum of k i.i.d. +-1 steps, P(+1)=q.

    `rng` defaults to a fresh, unseeded Generator if omitted; pass an
    explicit seeded one for reproducible runs (see measure_cost.py).
    """
    rng = rng if rng is not None else np.random.default_rng()
    steps = rng.choice([-1, 1], size=k, p=[1 - q, q])
    return int(np.abs(steps.sum()))


if __name__ == "__main__":
    print(srw(15))
