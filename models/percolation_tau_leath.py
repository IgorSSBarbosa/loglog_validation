"""Fisher exponent by LEATH GROWTH on the infinite lattice -- registered as
MODELS["percolation_tau_leath"] in tools/models.py.

Why it exists (user, 2026-11-08): Mertens & Moore, arXiv:1806.08067 Sec. V,
measure tau by growing the cluster of the origin at p_c (Leath 1976) instead
of counting every cluster of a box, as models/percolation_tau_zd.py and
percolation_tau_gpu.py do. The point here is to see what the pipeline does
when its assumptions fail: with one cluster per sample, the variance of
Y_s / E Y_s diverges with s, which breaks Assumption sigma_converges of the
article (sigma_i^2 -> sigma_inf^2 in (0, inf)).

The observable
--------------
Site percolation on Z^dim at p (default the tabulated p_c), the origin
conditioned open. With K = ceil(batch_factor * s**batch_exponent) independent
lattices, one sample at scale s is

    Y_s = (1/K) sum_{k<K} 1{ |C_k(0)| >= s },     E Y_s = Q(s),

Q(s) = P(|C(0)| >= s | 0 open), Mertens & Moore's Q (their P(s) is the law of
the size of an occupied vertex's cluster). Since P(s) ~ s n_s ~ s**(1-tau),
Q(s) ~ s**(2-tau): the pipeline's gamma is 2 - tau, not the 1 - tau of the
box models. E Y_s is Q(s) exactly for any K; K only sets the variance:

    Var(Y_s / Q) = (1 - Q) / (K Q)  ~  s**(tau - 2 - batch_exponent).

batch_exponent = 0 (the default, one cluster a sample) is the deliberate
violation: sigma_s^2 grows like s**(tau-2) (0.055 at dim 2, 0.19 at dim 3).
batch_exponent = 1/2 uses only tau <= 5/2 and sends sigma_s^2 to 0 instead;
any value from tau would put the answer in the code path (ground rule 4).

Each scale grows fresh clusters (ground rule 2): unlike Mertens & Moore, who
read Q(s) at every s off one set of runs, a cluster here is grown only until
it has s open sites or dies, and nothing is shared across scales. Stopping at
s is exact for the indicator: any exploration order finds s open sites iff
the cluster has at least s.

Randomness: as models/percolation_susceptibility_explore.py
-----------------------------------------------------------
open(site) <=> splitmix64_output(cluster seed, key(site)) < p * 2**64, a pure
function of (seed, site), so only open sites are stored (an open-addressing
table, cleared between the K clusters by bumping a generation stamp). Each
sample takes one `rng.integers(0, 2**63)` from the driver's Generator; its
k-th lattice uses the seed mix(sample seed + (k+1) * golden). Keys pack the
coordinates into 63 // dim bits apiece; a cluster that reaches the edge of
that range (+-2**30 at dim 2, +-2**20 at dim 3) raises, never wraps. At dim 2
that cannot happen below s = 2**30, since a cluster stopped at s sites has
radius < s.

Cost
----
No cost_hint. The work is E min(|C|, s) ~ s**(3-tau) clusters' sites times K,
and tau is the estimand (ground rule 4); the planner times it, batched and
latency-bound as the exploration model. Memory per thread ~ 24 bytes * s at
the top of a cluster (table at load <= 1/2, plus the BFS queue).
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit, prange

#: dimensions the key packing supports (bits per coordinate = 63 // dim)
DIMS = (2, 3, 4, 5)

#: Site-percolation thresholds on Z^dim, nearest-neighbour. Restated from
#: models/percolation_zd.py (no model file imports another); sources in its
#: P_C_SOURCE: dim 2 Jacobsen (2015) 0.59274605079210(2), dim 3 Xu, Wang, Lv
#: & Deng (2014) 0.3116077(2), dims 4-5 Mertens & Moore (2018).
P_C_SITE_HYPERCUBIC: dict[int, float] = {
    2: 0.59274605079210,
    3: 0.3116077,
    4: 0.19688561,
    5: 0.14079633,
}

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

# status codes per sample, raised on in Python (a raise inside prange is lost)
_OK, _EDGE = 0, 1


@njit(inline="always")
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


@njit(inline="always")
def _is_open(seed, key, thresh):
    return _mix(seed + np.uint64(key) * _GOLDEN) < thresh


@njit(inline="always")
def _slot(keys, stamp, gen, key):
    """Index of `key` in the table, or of the free slot where it would go.

    A slot is occupied iff its stamp is the current generation, so bumping
    `gen` empties the table in O(1) between clusters.
    """
    mask = keys.size - 1
    h = np.int64(_mix(np.uint64(key)) & np.uint64(mask))
    while stamp[h] == gen and keys[h] != key:
        h = (h + 1) & mask
    return h


@njit
def _rehash(keys, stamp, gen, queue, tail):
    """Double the table, keeping the current cluster (the first tail queue keys)."""
    new_keys = np.empty(keys.size * 2, dtype=np.int64)
    new_stamp = np.zeros(keys.size * 2, dtype=np.int32)
    for t in range(tail):
        g = _slot(new_keys, new_stamp, np.int32(1), queue[t])
        new_keys[g] = queue[t]
        new_stamp[g] = 1
    return new_keys, new_stamp


@njit
def _one_sample(seed, thresh, s, n_clusters, bits, dim, origin):
    """(clusters with >= s open sites, neighbour reads, peak bytes, status)."""
    fmask = np.int64((1 << bits) - 1)
    keys = np.empty(1 << 10, dtype=np.int64)
    stamp = np.zeros(1 << 10, dtype=np.int32)
    queue = np.empty(1 << 9, dtype=np.int64)
    gen = np.int32(0)
    hits = 0
    reads = 0
    peak = 0
    for c in range(n_clusters):
        cseed = _mix(seed + np.uint64(c + 1) * _GOLDEN)
        gen += 1
        if gen == np.int32(2 ** 31 - 1):  # stamps would wrap: start clean
            stamp[:] = 0
            gen = np.int32(1)
        g = _slot(keys, stamp, gen, origin)
        keys[g] = origin
        stamp[g] = gen
        queue[0] = origin
        head = 0
        tail = 1
        reached = tail >= s
        while head < tail and not reached:
            k = queue[head]
            head += 1
            for j in range(dim):
                shift = bits * j
                f = (k >> shift) & fmask
                if f == 0 or f == fmask:
                    return hits, reads, peak, _EDGE
                step = np.int64(1) << shift
                for nk in (k + step, k - step):
                    reads += 1
                    if not _is_open(cseed, nk, thresh):
                        continue
                    g = _slot(keys, stamp, gen, nk)
                    if stamp[g] == gen:
                        continue
                    keys[g] = nk
                    stamp[g] = gen
                    if tail == queue.size:
                        queue = np.concatenate(
                            (queue, np.empty(queue.size, np.int64)))
                    queue[tail] = nk
                    tail += 1
                    if tail >= s:
                        reached = True
                        break
                    if 2 * tail > keys.size:
                        keys, stamp = _rehash(keys, stamp, gen, queue, tail)
                        gen = np.int32(1)
                if reached:
                    break
        if reached:
            hits += 1
        b = keys.size * 12 + queue.size * 8
        if b > peak:
            peak = b
    return hits, reads, peak, _OK


@njit(parallel=True)
def _samples(seeds, thresh, s, n_clusters, bits, dim, origin):
    n = seeds.size
    hits = np.empty(n, dtype=np.int64)
    reads = np.empty(n, dtype=np.int64)
    peak = np.empty(n, dtype=np.int64)
    status = np.empty(n, dtype=np.int64)
    for r in prange(n):
        hits[r], reads[r], peak[r], status[r] = _one_sample(
            seeds[r], thresh, s, n_clusters, bits, dim, origin)
    return hits, reads, peak, status


#: samples per numba thread in one probe call; the reason is the exploration
#: model's (models/percolation_susceptibility_explore.py): prange chunks are
#: static and a sample's cost is heavy-tailed.
PROBE_SAMPLES_PER_THREAD = 2048


def device_fill() -> int:
    """ModelSpec.device_fill: the smallest call `probe_batched` may time."""
    return PROBE_SAMPLES_PER_THREAD * numba.get_num_threads()


def critical_p(dim: int) -> float:
    """p_c on Z^dim, from the table -- raising, never guessing, if absent."""
    if dim not in P_C_SITE_HYPERCUBIC:
        raise ValueError(f"no tabulated p_c for dim = {dim}; known: "
                         f"{sorted(P_C_SITE_HYPERCUBIC)}. Pass params['p'].")
    return P_C_SITE_HYPERCUBIC[dim]


def n_clusters(s: int, batch_factor: float = 1.0,
               batch_exponent: float = 0.0) -> int:
    """K(s) = ceil(batch_factor * s**batch_exponent); the default is one."""
    if batch_factor <= 0 or batch_exponent < 0:
        raise ValueError(f"need batch_factor > 0, batch_exponent >= 0; got "
                         f"{batch_factor}, {batch_exponent}")
    return int(math.ceil(float(batch_factor) * float(s) ** float(batch_exponent)))


def leath(s: int, n: int = 1, dim: int = 2, p: float | None = None,
          batch_factor: float = 1.0, batch_exponent: float = 0.0,
          rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """n i.i.d. samples of Y_s, plus per-sample work and memory.

    Returns (Y, info) with info = {"reads": neighbour states read per sample,
    "peak_bytes": largest table+queue footprint per sample, "K": clusters per
    sample, "p": p}. `simulate` keeps only Y; a benchmark wants the rest.
    """
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}; got {dim}")
    s = int(s)
    if s < 1:
        raise ValueError(f"s must be >= 1; got {s}")
    p = critical_p(dim) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    K = n_clusters(s, batch_factor, batch_exponent)
    bits = 63 // dim
    origin = sum((1 << (bits - 1)) << (bits * j) for j in range(dim))

    rng = rng if rng is not None else np.random.default_rng()
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    # P(U < thresh) = p for U uniform on [0, 2**64), to 2**-64; p = 1 opens all
    thresh = np.uint64(min(int(p * 2.0 ** 64), 2 ** 64 - 1))
    hits, reads, peak, status = _samples(seeds, thresh, s, K, bits, dim,
                                         np.int64(origin))
    if np.any(status == _EDGE):
        raise RuntimeError(
            f"a cluster reached the +-2**{bits - 1} key range at s = {s}, "
            f"dim = {dim} (p = {p:.8g}); it is not wrapped or truncated, so "
            f"the sample cannot be reported")
    return hits / K, {"reads": reads, "peak_bytes": peak, "K": K, "p": p}


def simulate(s: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_tau_leath"].simulate.

    params: {"dim": 2..5 (default 2), "p": float (default the tabulated p_c),
             "batch_factor": float (default 1), "batch_exponent": float
             (default 0: one cluster a sample)}.
    """
    y, _ = leath(s, n=n, dim=int(params.get("dim", 2)), p=params.get("p"),
                 batch_factor=float(params.get("batch_factor", 1.0)),
                 batch_exponent=float(params.get("batch_exponent", 0.0)),
                 rng=rng)
    return y
