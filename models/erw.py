"""Elephant random walk -- MODELS["erw"].

The walk with complete memory of its own past (Schutz-Trimper 2004), as
defined in Bercu, "A martingale approach for the elephant random walk",
J. Phys. A 51 (2018) 015201, section 2:

    S_0 = 0, and X_1 = +1 or -1 with probability 1/2 each;
    for n >= 1, draw k uniformly from {1, ..., n} and set
        X_{n+1} = +X_k  with probability p,
        X_{n+1} = -X_k  with probability 1 - p;
    S_{n+1} = S_n + X_{n+1}                                     (eq. 2.1)

The observable is Y_k = |S_k| after k steps, the same observable as srw and
rwre, and the scale i IS the number of steps k. `p` is the memory parameter
and the model's only parameter.

What is fixed, and why
----------------------
(a) THE FIRST STEP IS FAIR. Bercu's X_1 is Rademacher(q) with q free; here
    q = 1/2 by the user's specification (2026-09-22), and q is deliberately
    not a parameter. It buys an exact zero-check: E S_{n+1} = gamma_n E S_n
    (eq. 2.3) and E S_1 = 0, so E S_k = 0 at every k and every p, and a
    measured drift is a bug rather than physics.

(b) p IS REQUIRED. There is no default: a recipe that forgets p would
    otherwise silently run a different walk. At p = 1/2 the sign alpha_n is a
    fair coin independent of everything before it, so X_{n+1} is a fair coin
    independent of the past and the walk IS srw in law -- which makes
    experiments/01_srw a literal control arm, as it is for rwre.

(c) NO target_fn, NO true_gamma_key, for two reasons at once. The known
    truths stay out of the code path (PLAN.md ground rule 4), and for |S_k|
    there is no closed form to keep out anyway: Bercu gives E S_k^2 exactly
    (via eq. A.3) but not E|S_k|. What IS known, per regime, belongs in the
    experiment README as acceptance criteria rather than here:
      - diffusive, p < 3/4: S_n / sqrt(n) -> N(0, 1/(3 - 4p))      (eq. 3.5)
      - critical,  p = 3/4: S_n / sqrt(n log n) -> N(0, 1)         (eq. 3.10)
        -- a logarithm that the article's eq. (232) cannot represent, so
        Assumption 1 fails exactly there;
      - superdiffusive, p > 3/4: S_n / n^(2p-1) -> L a.s. and in L^4
        (eqs. 3.11-3.12), L non-Gaussian.

The sampler: the literal rule, not the Markov reduction
-------------------------------------------------------
Bercu notes after eq. (2.1) that X_{n+1} = alpha_n X_{beta_n} with alpha_n
Rademacher(p), beta_n uniform on {1..n}, and alpha_n independent of the past.
So every "parent" beta_n and every sign alpha_n can be drawn up front: the
parents form a random recursive tree rooted at step 1, and X_t is X_1 times
the product of the signs on the path from t to the root. `walk` resolves
all k paths at once by pointer doubling -- each pass multiplies a node's sign
by its ancestor's and jumps to the ancestor's ancestor -- so a sample costs a
few vectorized passes over k entries, however long the walk.

The alternative is the Markov chain that eq. (2.2) implies,
P(X_{n+1} = +1 | F_n) = (1 + (2p - 1) S_n / n) / 2, one uniform per step.
It is exact too, and it is what tools/tests/test_erw.py reimplements
independently. It was measured and declined (2026-09-22): its loop runs over
the k steps, so it vectorizes only across samples, and the byte budget
starves it of rows at the top of the ladder. Per step per sample, each at
its own best working set (the chain at 256 MiB, this sampler at the 16 MiB
below):

      k          2^6    2^10   2^14   2^17   2^20
    markov       13     15     20     28     154   ns
    tree (this)  27     31     25     25      44   ns

The tree's step up at 2^20 is the cache, not the algorithm: one sample
there is ~40 MB on its own.

Blocking: the contract kept verbatim
------------------------------------
Each sample consumes exactly 2k - 1 float64 uniforms from one row of a
(rows, 2k - 1) draw, and rows never interact, so splitting n into row blocks
consumes numpy's row-major stream in the same order as one unblocked call:
bit-identical output at any `block_n` (models/README.md's contract, and
test_block_n_matches_unblocked_for_same_seed). Unlike rwre, nothing here had
to give. Pointer doubling runs until the deepest path in the block resolves;
the extra passes a shallower row sees are no-ops on it.

float64, not srw's float32: a parent index is floor(u * n) for n up to k, and
with float32's 24 bits the parents' probabilities are off by up to n / 2^24
relatively -- 0.6% at n = 10^5 -- and past n = 2^24 some cannot be drawn.
"""

from __future__ import annotations

import numpy as np

#: Working-set budget for one row-block, in bytes. Same role as srw's: it
#: bounds peak transient memory regardless of how large n gets, and -- as for
#: srw -- it is a free knob, because blocking does not change the draws.
#: 16 MiB rather than srw's 256: a block that stays in cache is 1.4x faster
#: per step at every k from 2^6 to 2^18 (measured 2026-09-22, i9-13900K:
#: 25-32 ns against 37-45), and no flatter or steeper in k.
_DEFAULT_WORKING_SET_BYTES = 16 * 1024 * 1024

#: The (rows, 2k - 1) float64 draw (16 bytes per step), the int64 ancestor
#: table (8), the int8 signs (1), and the two gathers each doubling pass
#: allocates (9). Only the order of magnitude matters.
_BYTES_PER_STEP = 40

_PARAMS = ("p",)


def _check(params: dict) -> float:
    """The memory parameter p, checked. Rejects unknown keys by name."""
    unknown = set(params) - set(_PARAMS)
    if unknown:
        raise ValueError(
            f"unknown erw params {sorted(unknown)}; known: {list(_PARAMS)}")
    if "p" not in params:
        raise ValueError("erw needs the memory parameter 'p' in params; "
                         "there is no default")
    p = float(params["p"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    return p


def _resolve(u: np.ndarray, k: int, p: float) -> np.ndarray:
    """S_k for each row of u, a (rows, 2k - 1) block of uniforms.

    Column 0 is X_1's coin; columns 1..k-1 pick the parents of steps 2..k;
    columns k..2k-2 are their signs. Node k is a sentinel: sign 1, its own
    ancestor, and the ancestor of step 1.
    """
    rows = u.shape[0]
    n_prev = np.arange(1, k, dtype=np.float64)

    sign = np.ones((rows, k + 1), dtype=np.int8)
    sign[:, 0] = np.where(u[:, 0] < 0.5, 1, -1)
    sign[:, 1:k] = np.where(u[:, k:] < p, 1, -1)

    anc = np.empty((rows, k + 1), dtype=np.int64)
    anc[:, 0] = k
    # Step t + 1 (0-based column t) copies one of the t steps before it. The
    # clip guards floor(u * t) == t, which float rounding allows when u is
    # within 2^-53 of 1.
    anc[:, 1:k] = np.minimum((u[:, 1:k] * n_prev).astype(np.int64),
                             np.arange(k - 1))
    anc[:, k] = k

    head = anc[:, :k]
    while not np.all(head == k):
        sign[:, :k] *= np.take_along_axis(sign, head, axis=1)
        anc[:, :k] = np.take_along_axis(anc, head, axis=1)
        head = anc[:, :k]
    return sign[:, :k].sum(axis=1, dtype=np.int64)


def walk(k: int, n: int, p: float, rng: np.random.Generator | None = None,
         block_n: int | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED position S_k. int64, shape (n,).

    `rng` defaults to a fresh unseeded Generator; pass a seeded one for
    reproducible runs. `block_n` defaults to a size derived from a fixed byte
    budget and does not change the output (see the module docstring).
    simulate() below returns |S_k|; this exists because the sign carries the
    exact zero-check E S_k = 0, and the tests need it.
    """
    p = _check({"p": p})
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    rng = rng if rng is not None else np.random.default_rng()
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_STEP * k))
    if block_n < 1:
        raise ValueError(f"block_n must be >= 1; got {block_n}")

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        out[offset:offset + rows] = _resolve(rng.random((rows, 2 * k - 1)), k, p)
        offset += rows
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["erw"].simulate: n i.i.d. samples of Y_i = |S_i| at scale i."""
    return np.abs(walk(i, n, _check(params), rng=rng))


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample of |S_i|: i steps.

    Every step is drawn and resolved -- no early exit, and no constant-time
    shortcut exists to be tempted by (S_k is not Markov in k alone). The one
    thing that is not a constant factor is the number of doubling passes,
    ceil(log2(h + 1)) for a tree of height h, and the height of a random
    recursive tree grows only logarithmically in i: that makes the true work
    i * O(log log i). Measured (2026-09-22): 4 passes at i = 2^6 and 5 from
    2^10 through 2^20, and the time per step flat at 25-31 ns from 2^6 to
    2^17 (the table in the module docstring) -- which is the resolution
    Assumption 7's d = 1 is scored at.

    What the clock sees depends on how it is asked. measure_cost.py times
    ONE sample per call, and over 2^10..2^18 at p = 0.9 two runs of the same
    recipe on the same idle host read d = 1.093 +/- 0.029 and
    0.924 +/- 0.031 -- both PASS at its 20% tolerance, on opposite sides of
    1, and each with a sigma warning. The small-k per-call overhead did not
    reproduce between them (45 vs 122 us); the top three rungs did, to 4%,
    and double exactly with i. Read the probe's drop-leading ladder, not
    its headline.
    """
    return float(i)
