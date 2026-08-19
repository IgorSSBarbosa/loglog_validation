"""Simple random walk simulator -- registered as MODELS["srw"] in tools/models.py.

NOT Phase 1's gamma-estimation testbed. That remains blocked pending a
closed-form E[Y_i]/gamma/omega1 for the article's appendix-SimpleRandomWalk
(see PLAN.md "Open questions before Phase 1") -- MODELS["srw"] has no
target_fn/true_gamma_key for exactly this reason, so src/plot_loglog.py
never overlays a reference curve or reports a true_gamma for this model's
data, and prints an explicit "exploratory, not validated" note alongside
any gamma-hat it does compute (see that module's docstring). Here `srw` is
used as a simulator whose per-call cost
genuinely grows with scale -- unlike MODELS["synthetic"]'s draws, whose cost
is ~constant in i -- a fixture for validating tools/cost_model.py's
estimator against a known ground truth: generating k i.i.d. +-1 steps and
summing them is Theta(k) work, so the measured cost exponent d should
recover close to 1 (see experiments/01_srw/).
"""

from __future__ import annotations

import numpy as np

_STEP_VALUES = np.array([-1, 1], dtype=np.int8)

# Working-set budget for one (block_n, k) draw inside srw(), in bytes. Steps
# are int8 (1 byte each), so block_n = budget // k bounds the transient
# matrix to this size regardless of how large n gets -- fixes the O(n*k)
# blowup a single unblocked rng.choice(size=(n,k)) call used to cause (e.g.
# n=1e8, k=1024 was ~819 GiB with the old int64 dtype). Blocking is done
# over n, not k: numpy fills an (n, k) draw in row-major order (row 0's k
# columns, then row 1's, ...), so splitting along the leading axis n into
# sequential, non-overlapping row ranges consumes the RNG stream in exactly
# the same order a single unblocked call would -- bit-identical output for
# the same seed, at any block size. Splitting along k instead would
# interleave the stream differently and silently change the results.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024


def srw(
    k: int,
    n: int = 1,
    q: float = 0.5,
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. realizations of |S_k|, S_k = sum of k i.i.d. +-1 steps, P(+1)=q.

    Vectorized, but bounded to a fixed working set: draws (block_n, k) int8
    step matrices one row-block at a time (rather than one (n, k) matrix)
    and accumulates their row sums, so peak transient memory is
    ~block_n*k bytes regardless of how large n gets -- and, because the
    blocking is over n (see module docstring above), results are exactly
    the same as one unblocked call for the same seed and any block_n.
    `block_n` defaults to a size derived from a fixed byte budget
    (_DEFAULT_WORKING_SET_BYTES); pass an explicit value to override (e.g.
    in tests).

    `rng` defaults to a fresh, unseeded Generator if omitted; pass an
    explicit seeded one for reproducible runs. src/measure_cost.py calls
    this with n=1 -- Assumption cost_is_power_law defines cost(i) as the
    cost of simulating *one* sample.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // max(k, 1))
    block_n = min(block_n, n)

    total = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        this_block = min(block_n, n - offset)
        steps = rng.choice(_STEP_VALUES, size=(this_block, k), p=[1 - q, q])
        total[offset:offset + this_block] = steps.sum(axis=1, dtype=np.int64)
        offset += this_block
    return np.abs(total)


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["srw"].simulate: n i.i.d. samples of |S_i| at scale i (params: {"q": ...})."""
    return srw(i, n=n, q=params.get("q", 0.5), rng=rng)
