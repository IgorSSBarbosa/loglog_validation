"""Off-critical mean cluster size by EXPLORATION on the infinite lattice, from
seeds on an axis segment -- registered as
MODELS["percolation_susceptibility_explore"] in tools/models.py.

Same estimand, same ladder as models/percolation_susceptibility.py:
p = p_c - eps0/x, and

    E Y = E[ |C(0)|**moment | 0 open ] = S(p)      (moment = 1)

exactly, on Z^dim itself -- no box, no torus, no wrap correction and no
ceil'd-L rounding (the shift of experiments/09 Step 3 has nothing to act on).
Why it exists (user, 2026-11-03): the torus model needs the whole L**dim box
in memory, ~21 bytes a site on the GPU, which stops dim = 3 at x = 512
(experiment 51). Below p_c every cluster is finite almost surely, so the
cluster of a seed can be grown site by site and only what it touches is ever
stored: memory ~ cluster, not box.

The observable
--------------
Seeds are m = ceil(axis_factor * x**nu_axis) sites on the x_0 axis,
s_k = (k * axis_spacing, 0, ..., 0), k = 0..m-1. One sample is

    Y = sum_k 1{s_k open} |C(s_k)|**moment / (p * m),

and since every term has expectation p * E[|C|^moment | open], E Y is the
estimand exactly, whatever m and the spacing -- they only change the variance
and the cost. Seeds in the same cluster explore it ONCE: the cluster gets an
id, and later seeds that land in it reuse its size. That sharing is the point
of the axis (the d_f experiments' "connected to a face" in this setting):
within a correlation length the seeds hit the same few big clusters, so
m seeds cost much less than m explorations.

Randomness: one splitmix64 stream per sample, read at random access
-------------------------------------------------------------------
A lazily explored lattice must give every site ONE state however often it is
looked at. Storing the state of every touched site would store the closed
boundary too (most of what is touched). Instead the state is a pure function
of (sample seed, site):

    open(site)  <=>  splitmix64_output(seed, index = key(site)) < p * 2**64

where splitmix64_output(seed, k) = mix(seed + k * 0x9E3779B97F4A7C15) is the
k-th output of the splitmix64 generator started at `seed` (Steele, Lea &
Flood 2014), i.e. the sites read DISTINCT positions of one splitmix64
sequence. Each sample's seed is one `rng.integers(0, 2**63)` from the
driver's Generator -- the same one-integer-per-call contract as the *_gpu
models (ground rule 2). Only open sites are stored: key -> cluster id in an
open-addressing table.

Keys pack the coordinates, each offset by 2**(bits-1), into bits = 63 // dim
bits apiece, so a neighbour is key +- 2**(bits*j). A cluster that reaches the
edge of that range (+-2**20 sites at dim 3) is reported, never wrapped:
`simulate` raises. dim is 2..4; at dim >= 5 the range is too short to trust.

A cluster larger than `max_sites` open sites (default 2e8, ~6 GiB of table)
raises MemoryError instead of growing on. Near p_c one seed can hit a cluster
of billions of sites; the first pilot (2026-11-03) had its cost probe climb to
x = 32768 and was killed, by the OOM killer as far as could be told, with five
replicates drawn and unsaved. MemoryError is what the cost probes already read
as "this scale is above the model's ceiling"; in a real draw it fails loudly.

Cost
----
No cost_hint. The work is the number of neighbour states read, ~ m * S(p),
and S(p) ~ x**gamma_susc is the estimand -- declaring it would put the
answer in the code path (ground rule 4). The planner times it instead, as a
`batched_cost`, `latency_bound` model: one sample's cost is heavy-tailed and
usually ~0 (a closed seed), so one call per scale measures nothing, and the
generic probe climbed to x = 32768 looking for a call slow enough to time.
The batched probe reads the cost of one more sample from large calls on the
pilot's own scales.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit, prange

from models.percolation_susceptibility import _MAX_MOMENT, p_at

#: dimensions the key packing supports (bits per coordinate = 63 // dim)
DIMS = (2, 3, 4)

#: open sites one sample may store before MemoryError (12 bytes a table slot,
#: 8 with one seed, load <= 1/2: ~6 GiB at the cap). See the docstring.
DEFAULT_MAX_SITES = 200_000_000

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_EMPTY = np.int64(-1)

# status codes per sample, raised on in Python (a raise inside prange is lost)
_OK, _EDGE, _CAP = 0, 1, 2


@njit(inline="always")
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


@njit(inline="always")
def _is_open(seed, key, thresh):
    return _mix(seed + np.uint64(key) * _GOLDEN) < thresh


@njit(inline="always")
def _slot(keys, key):
    """Index of `key` in the table, or of the empty slot where it would go."""
    mask = keys.size - 1
    h = np.int64(_mix(np.uint64(key)) & np.uint64(mask))
    while keys[h] != _EMPTY and keys[h] != key:
        h = (h + 1) & mask
    return h


@njit
def _grow(keys, vals):
    new_keys = np.full(keys.size * 2, _EMPTY, dtype=np.int64)
    new_vals = np.empty(keys.size * 2, dtype=np.int32)
    for h in range(keys.size):
        if keys[h] != _EMPTY:
            g = _slot(new_keys, keys[h])
            new_keys[g] = keys[h]
            new_vals[g] = vals[h]
    return new_keys, new_vals


@njit
def _grow_keys(keys):
    new_keys = np.full(keys.size * 2, _EMPTY, dtype=np.int64)
    for h in range(keys.size):
        if keys[h] != _EMPTY:
            new_keys[_slot(new_keys, keys[h])] = keys[h]
    return new_keys


@njit
def _one_sample(seed, thresh, seed_keys, bits, dim, moment, max_sites):
    """(sum_k 1{open} |C(s_k)|**moment, neighbour reads, peak bytes, status)."""
    fmask = np.int64((1 << bits) - 1)
    keys = np.full(1 << 12, _EMPTY, dtype=np.int64)
    # cluster ids only tell a later seed which cluster it landed in; with one
    # seed nothing reads them, and dropping them is a third of the table
    # (user, 2026-11-04: the x = 2048 long run was OOM-killed at 16 threads)
    track = seed_keys.size > 1
    vals = np.empty((1 << 12) if track else 1, dtype=np.int32)
    slot_bytes = 12 if track else 8
    used = 0
    sizes = np.empty(64, dtype=np.int64)
    n_clusters = 0
    stack = np.empty(1 << 10, dtype=np.int64)
    total = 0.0
    reads = 0
    peak = 0
    for s in range(seed_keys.size):
        key0 = seed_keys[s]
        reads += 1
        if not _is_open(seed, key0, thresh):
            continue
        h = _slot(keys, key0)
        if keys[h] == key0:
            c = vals[h]
        else:
            c = n_clusters
            if c == sizes.size:
                sizes = np.concatenate((sizes, np.empty(sizes.size, np.int64)))
            n_clusters += 1
            keys[h] = key0
            if track:
                vals[h] = c
            used += 1
            top = 0
            stack[top] = key0
            top += 1
            size = 0
            while top > 0:
                top -= 1
                k = stack[top]
                size += 1
                for j in range(dim):
                    shift = bits * j
                    f = (k >> shift) & fmask
                    if f == 0 or f == fmask:
                        return 0.0, reads, peak, _EDGE
                    step = np.int64(1) << shift
                    for nk in (k + step, k - step):
                        reads += 1
                        if not _is_open(seed, nk, thresh):
                            continue
                        g = _slot(keys, nk)
                        if keys[g] == nk:
                            continue
                        keys[g] = nk
                        if track:
                            vals[g] = c
                        used += 1
                        if used > max_sites:
                            return 0.0, reads, peak, _CAP
                        if 2 * used > keys.size:
                            if track:
                                keys, vals = _grow(keys, vals)
                            else:
                                keys = _grow_keys(keys)
                        if top == stack.size:
                            stack = np.concatenate(
                                (stack, np.empty(stack.size, np.int64)))
                        stack[top] = nk
                        top += 1
            sizes[c] = size
        total += float(sizes[c]) ** moment
        b = keys.size * slot_bytes + stack.size * 8 + sizes.size * 8
        if b > peak:
            peak = b
    return total, reads, peak, _OK


@njit(parallel=True)
def _samples(seeds, thresh, seed_keys, bits, dim, moment, max_sites):
    n = seeds.size
    tot = np.empty(n, dtype=np.float64)
    reads = np.empty(n, dtype=np.int64)
    peak = np.empty(n, dtype=np.int64)
    status = np.empty(n, dtype=np.int64)
    for r in prange(n):
        tot[r], reads[r], peak[r], status[r] = _one_sample(
            seeds[r], thresh, seed_keys, bits, dim, moment, max_sites)
    return tot, reads, peak, status


#: samples per numba thread in one probe call. prange hands each thread a
#: static chunk, so a call's time is its slowest chunk's, and a chunk's cost is
#: a sum of heavy-tailed sample costs (cv ~ 10). 2048 a thread puts a call's
#: spread at a few percent; with 128 samples a call (the latency walk's own
#: choice, 2026-11-03) 4n samples timed no slower than n.
PROBE_SAMPLES_PER_THREAD = 2048


def device_fill() -> int:
    """ModelSpec.device_fill: the smallest call `probe_batched` may time."""
    return PROBE_SAMPLES_PER_THREAD * numba.get_num_threads()


def n_seeds(x: int, axis_factor: float = 1.0, nu_axis: float = 0.0) -> int:
    """m(x) = ceil(axis_factor * x**nu_axis); the default is the single origin."""
    if axis_factor <= 0 or nu_axis < 0:
        raise ValueError(f"need axis_factor > 0, nu_axis >= 0; got "
                         f"{axis_factor}, {nu_axis}")
    return int(math.ceil(float(axis_factor) * float(x) ** float(nu_axis)))


def _seed_keys(m: int, spacing: int, dim: int) -> tuple[np.ndarray, int]:
    bits = 63 // dim
    off = 1 << (bits - 1)
    if (m - 1) * spacing >= off - 1:
        raise ValueError(f"axis of {m} seeds at spacing {spacing} does not fit "
                         f"the +-{off} coordinate range at dim = {dim}")
    base = sum(off << (bits * j) for j in range(dim))
    return base + np.arange(m, dtype=np.int64) * spacing, bits


def explore(x: int, n: int = 1, dim: int = 3, eps0: float = 0.15,
            p: float | None = None, p_c: float | None = None,
            axis_factor: float = 1.0, nu_axis: float = 0.0,
            axis_spacing: int = 1, moment: int = 1,
            max_sites: int = DEFAULT_MAX_SITES,
            rng: np.random.Generator | None = None,
            ) -> tuple[np.ndarray, dict]:
    """n i.i.d. samples of Y, plus per-sample work and memory.

    Returns (Y, info) with info = {"reads": neighbour states read per sample,
    "peak_bytes": largest table+stack+sizes footprint per sample, "m": seeds,
    "p": p}. `simulate` keeps only Y; the benchmark wants the rest.
    """
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}; got {dim}")
    moment = int(moment)
    if not 1 <= moment <= _MAX_MOMENT:
        raise ValueError(f"moment must be in 1..{_MAX_MOMENT}; got {moment}")
    x = int(x)
    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    m = n_seeds(x, axis_factor, nu_axis)
    spacing = int(axis_spacing)
    if spacing < 1:
        raise ValueError(f"axis_spacing must be >= 1; got {spacing}")
    seed_keys, bits = _seed_keys(m, spacing, dim)

    rng = rng if rng is not None else np.random.default_rng()
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    # P(U < thresh) = p for U uniform on [0, 2**64), to 2**-64
    thresh = np.uint64(min(int(p * 2.0 ** 64), 2 ** 64 - 1))
    tot, reads, peak, status = _samples(seeds, thresh, seed_keys, bits, dim,
                                        moment, int(max_sites))
    if np.any(status == _CAP):
        raise MemoryError(
            f"a sample stored more than max_sites = {int(max_sites):,} open "
            f"sites at x = {x}, dim = {dim} (p = {p:.8g})")
    if np.any(status == _EDGE):
        raise RuntimeError(
            f"a cluster reached the +-2**{bits - 1} key range at x = {x}, "
            f"dim = {dim} (p = {p:.8g}); it is not wrapped or truncated, so "
            f"the sample cannot be reported")
    return tot / (p * m), {"reads": reads, "peak_bytes": peak, "m": m, "p": p}


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_explore"].simulate.

    params: {"dim": 2|3|4 (default 3), "eps0": float (default 0.15), "p",
             "p_c": as percolation_susceptibility, "axis_factor": float
             (default 1), "nu_axis": float (default 0: one seed),
             "axis_spacing": int (default 1), "moment": 1|2|3,
             "max_sites": int (default DEFAULT_MAX_SITES)}.
    """
    y, _ = explore(
        x, n=n, dim=int(params.get("dim", 3)),
        eps0=float(params.get("eps0", 0.15)),
        p=params.get("p"), p_c=params.get("p_c"),
        axis_factor=float(params.get("axis_factor", 1.0)),
        nu_axis=float(params.get("nu_axis", 0.0)),
        axis_spacing=int(params.get("axis_spacing", 1)),
        moment=int(params.get("moment", 1)),
        max_sites=int(params.get("max_sites", DEFAULT_MAX_SITES)), rng=rng)
    return y
