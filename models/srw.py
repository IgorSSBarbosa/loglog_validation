"""Simple random walk simulator -- registered as MODELS["srw"] in tools/models.py.

MODELS["srw"] deliberately has no target_fn/true_gamma_key, so
src/plot_loglog.py never overlays a reference curve or reports a true_gamma
for this model's data, and prints an explicit "exploratory, not validated"
note alongside any gamma-hat it does compute (see that module's docstring).

That absence is now a choice, not a gap. For Y_k = |S_k| the mean is known
in closed form,

    E|S_k| = k * C(k-1, floor((k-1)/2)) * 2^{-(k-1)}
           = sqrt(2/pi) * k^(1/2) * exp(-(1/4) k^-1 + (1/24) k^-3 + O(k^-5)),

i.e. article eq. (232) with a0 = sqrt(2/pi), gamma = 1/2, omega_1 = 1,
a_1 = -1/4 (and no k^-2 term at all, so omega_2 = 3). The user's decision
(2026-08-20, plans/three_experiment_ladder.md D1/D2) is to keep these values
OUT of the code path and state them instead as written acceptance criteria in
experiments/01_srw/README.md, checked by hand when an experiment finishes --
so the estimators are never even incidentally handed the answer they are
supposed to be measuring. Do not add target_fn here without revisiting that.

Here `srw` is used as a simulator whose per-call cost
genuinely grows with scale -- unlike MODELS["synthetic"]'s draws, whose cost
is ~constant in i -- a fixture for validating tools/cost_model.py's
estimator against a known ground truth: generating k i.i.d. +-1 steps and
summing them is Theta(k) work, so the measured cost exponent d should
recover close to 1 (see experiments/01_srw/).
"""

from __future__ import annotations

import numpy as np

# Working-set budget for one (block_n, k) draw inside srw(), in bytes, so
# block_n = budget // (bytes_per_step * k) bounds the transient matrix to this
# size regardless of how large n gets -- fixes the O(n*k) blowup a single
# unblocked rng.choice(size=(n,k)) call used to cause (e.g. n=1e8, k=1024 was
# ~819 GiB with the old int64 dtype). Blocking is done over n, not k: numpy
# fills an (n, k) draw in row-major order (row 0's k columns, then row 1's,
# ...), so splitting along the leading axis n into sequential, non-overlapping
# row ranges consumes the RNG stream in exactly the same order a single
# unblocked call would -- bit-identical output for the same seed, at any block
# size. Splitting along k instead would interleave the stream differently and
# silently change the results.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

# One float32 uniform per step (see _draw_heads).
_BYTES_PER_STEP = 4


def _draw_heads(rng: np.random.Generator, rows: int, k: int, q: float) -> np.ndarray:
    """Number of +1 steps in each of `rows` independent walks of length k.

    Returns an int64 array of shape (rows,), each entry in [0, k].

    Deliberately NOT rng.choice([-1, 1], p=[1-q, q]): choice's
    probability-vector path is ~4x slower for the same Theta(rows*k) work
    (28.3 vs 6.6 us/sample at k=1024, measured 2026-08-20).

    Equally deliberately NOT rng.binomial(k, q, size=rows), which would be a
    further ~375x faster and is distributionally identical: binomial samples
    every scale in ~constant time, which would destroy the Theta(k) per-sample
    cost that makes this model a stand-in for a real percolation simulator --
    and with it the whole point of the cost-exponent and budget-allocation
    experiments (user's call, 2026-08-20; see plans/three_experiment_ladder.md
    section 1a). The k steps are really generated, one per unit of scale.

    Also deliberately NOT rng.integers(0, 2, dtype=np.int8), which is another
    ~20% faster and only 1 byte/step: numpy packs several small ints per 64-bit
    draw and DISCARDS the leftover bits at the end of each call, so splitting
    the rows into blocks consumes the bit stream differently from one unblocked
    call and silently changes the output (verified 2026-08-20: k=7, n=50,
    block_n=3 diverges). That would break both srw()'s own block_n invariance
    and src/generate.py's chunked path, which relies on it. A float32 uniform
    is one draw per step with no packing, so row-blocking is exact.

    float32 (not float64) halves the working set at no statistical cost here:
    it carries 24 random mantissa bits, and for q=0.5 the comparison is an
    exactly fair coin (0.5 is a clean binary boundary). For other q the
    realized probability differs from q by at most 2^-24 ~ 6e-8.
    """
    heads = rng.random(size=(rows, k), dtype=np.float32) < q
    return heads.sum(axis=1, dtype=np.int64)


def srw(
    k: int,
    n: int = 1,
    q: float = 0.5,
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. realizations of |S_k|, S_k = sum of k i.i.d. +-1 steps, P(+1)=q.

    Vectorized, but bounded to a fixed working set: draws (block_n, k) step
    matrices one row-block at a time (rather than one (n, k) matrix) and
    accumulates their row sums via _draw_heads, so peak transient memory is
    bounded by _DEFAULT_WORKING_SET_BYTES regardless of how large n gets
    -- and, because the
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
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_STEP * max(k, 1)))
    block_n = min(block_n, n)

    total = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        this_block = min(block_n, n - offset)
        # S_k = (#+1 steps) - (#-1 steps) = 2 * heads - k.
        heads = _draw_heads(rng, this_block, k, q)
        total[offset:offset + this_block] = 2 * heads - k
        offset += this_block
    return np.abs(total)


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["srw"].simulate: n i.i.d. samples of |S_i| at scale i (params: {"q": ...})."""
    return srw(i, n=n, q=params.get("q", 0.5), rng=rng)


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample of |S_i|: exactly i steps.

    Exact, not an estimate -- `srw` draws i uniforms per sample and sums them,
    with no early exit. This is what makes srw a usable testbed for the budget
    machinery: d = 1 is known rather than fitted, so a measured d can be scored
    against it (see tools/models.py's ModelSpec.cost_hint for the numbers).

    Deliberately NOT rng.binomial, which would make the cost constant in i and
    void the whole cost model -- see this module's docstring.
    """
    return float(i)
