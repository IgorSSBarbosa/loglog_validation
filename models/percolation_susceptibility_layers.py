"""Off-critical mean cluster size by LAYERED exploration with memory recycling --
registered as MODELS["percolation_susceptibility_layers"] in tools/models.py.

The same estimand, ladder, lattice and site hash as
models/percolation_susceptibility_explore.py with one seed (the origin), which is
what every recipe of that model uses:

    p = p_c - eps0/x,   Y = |C(0)| 1{0 open} / p,   E Y = S(p).

Why it exists (user, 2026-11-11): the explore model stores every open site of the
cluster, so its memory is ~|C|, and near p_c the tail of |C| sets the ceiling
(d = 3 OOM kills, experiment 51; at d = 5, experiment 39's benchmark of
2026-11-11 had P(|C| > 1e8) ~ 2e-5 at x = 4096). The ceiling is a tail
problem, not a mean-cost one.

Layers, and why old ones can be dropped
---------------------------------------
The cluster is grown breadth first by chemical distance l (the length of the
shortest open path from the origin): layer l+1 is the set of unseen open
neighbours of layer l. A stored site is only ever LOOKED UP when one of its open
neighbours is expanded, and two neighbouring open sites have chemical distances
differing by at most 1. So while layer l is expanded, lookups only hit layers
l-1, l, l+1, and every site of layer <= l-2 can never be looked up again: it may
be deleted. Deleting it is exact -- |C| is counted when a site is expanded, not
read off the table, so nothing is lost or counted twice.

Each table entry stores its layer. The table grows (doubling, load <= 1/2) as in
the explore model until it holds `max_live` sites; then it is rebuilt keeping
only layers >= l-1 (a rebuild, not in-place deletion, so linear probing needs
no tombstones). If the rebuild frees nothing, the three live layers alone fill
the cap and the sample stops with MemoryError (user's rule, 2026-11-11).
Within one layer a second rebuild can free nothing, so rebuilds are at most one
per layer.

Live memory is then ~ three layers instead of |C| (measured at d = 5: at most
0.5 % of |C| for clusters above 1e6 sites, experiment 58). With d_f ~ 3.5 and
d_min ~ 1.8 at d = 5 (values quoted from memory, not used anywhere), a layer
is ~ |C|**0.5 sites. What it does NOT change: the work, still 1 + 2 dim |C|
neighbour reads a sample, and the key range (+-2047 sites a coordinate at
dim = 5), which becomes the next wall. A cluster reaching it raises, as in the
explore model.

Layer profile (user, 2026-11-11: "two exponents at the price of one")
---------------------------------------------------------------------
Counting the sites of each layer, N_l, is free. `explore_layers` returns, per
call, the sums over samples of N_l and N_l**2 for l = 0..max_layers-1 and the
depth (last layer) of each sample. Storing N_l per sample is not cheap (a
variable-length row per sample, 7e8 samples a scale), so only the per-call sums
are kept. `simulate` returns Y only; the generic drivers never see the
profile.

Bit-for-bit check: the site states are the explore model's (same splitmix64
hash of (sample seed, key), same one-integer-per-sample seed draw), so for the
same rng both models return the SAME Y and the same reads, sample by sample.
Only the visiting order and the memory differ.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from models.percolation_susceptibility import p_at
from models.percolation_susceptibility_explore import (
    DIMS, PROBE_SAMPLES_PER_THREAD, _EMPTY, _is_open, _slot)

#: live open sites one sample may hold before a rebuild (12 bytes a slot at
#: load <= 1/2: ~24 MiB a thread). Bench 2026-11-11 (experiment 58, d = 5,
#: eps0 = 0.07): the three live layers peaked at 2.4e5 sites over 2e5 samples
#: at x = 16384 (largest cluster 4e9 sites), so 2**20 leaves a 4x margin;
#: 2**18 would have been within 7 % of a MemoryError there.
DEFAULT_MAX_LIVE = 1 << 20

#: layers tallied one by one in the profile; deeper layers are summed into
#: the last bin. The sample itself is unaffected. (Bench 2026-11-11, d = 5,
#: x = 512: a cluster of 8e6 sites was 16,434 layers deep.)
DEFAULT_MAX_LAYERS = 1 << 17

_OK, _EDGE, _CAP = 0, 1, 2


@njit
def _rebuild(keys, lay, size, keep_from):
    """Table of `size` slots holding only the entries with layer >= keep_from."""
    nk = np.full(size, _EMPTY, dtype=np.int64)
    nl = np.empty(size, dtype=np.int32)
    live = 0
    for h in range(keys.size):
        if keys[h] != _EMPTY and lay[h] >= keep_from:
            g = _slot(nk, keys[h])
            nk[g] = keys[h]
            nl[g] = lay[h]
            live += 1
    return nk, nl, live


@njit
def _one_sample(seed, thresh, key0, bits, dim, max_live, prof, prof2):
    """(|C| or 0, reads, peak live, need3, peak bytes, depth, rebuilds, status).

    need3 = max_l (N_{l-1} + N_l + N_{l+1}), the fewest live sites any cap
    could get away with. Adds N_l, N_l**2 into prof, prof2 (one chunk's row),
    layers past the last bin summed into it."""
    fmask = np.int64((1 << bits) - 1)
    reads = 1
    if not _is_open(seed, key0, thresh):
        return 0, reads, 0, 0, 0, -1, 0, _OK
    keys = np.full(1 << 12, _EMPTY, dtype=np.int64)
    lay = np.empty(1 << 12, dtype=np.int32)
    g = _slot(keys, key0)
    keys[g] = key0
    lay[g] = 0
    live = 1
    peak_live = 1
    peak_bytes = 0
    rebuilds = 0
    cur = np.empty(1 << 10, dtype=np.int64)
    nxt = np.empty(1 << 10, dtype=np.int64)
    cur[0] = key0
    n_cur = 1
    size = 0
    layer = 0
    n_prev = 0
    need3 = 0
    last_bin = prof.size - 1
    while n_cur > 0:
        b_l = layer if layer < last_bin else last_bin
        prof[b_l] += n_cur
        prof2[b_l] += float(n_cur) * float(n_cur)
        size += n_cur
        n_nxt = 0
        for q in range(n_cur):
            k = cur[q]
            for j in range(dim):
                shift = bits * j
                f = (k >> shift) & fmask
                if f == 0 or f == fmask:
                    return (0, reads, peak_live, need3, peak_bytes, layer,
                            rebuilds, _EDGE)
                step = np.int64(1) << shift
                for nk in (k + step, k - step):
                    reads += 1
                    if not _is_open(seed, nk, thresh):
                        continue
                    h = _slot(keys, nk)
                    if keys[h] == nk:
                        continue
                    keys[h] = nk
                    lay[h] = layer + 1
                    live += 1
                    if live > max_live:
                        # layers <= layer-2 can no longer be looked up
                        keys, lay, kept = _rebuild(keys, lay, keys.size, layer - 1)
                        rebuilds += 1
                        if kept == live:
                            return (0, reads, peak_live, need3, peak_bytes,
                                    layer, rebuilds, _CAP)
                        live = kept
                    elif 2 * live > keys.size:
                        keys, lay, _ = _rebuild(keys, lay, 2 * keys.size, 0)
                    if live > peak_live:
                        peak_live = live
                    if n_nxt == nxt.size:
                        nxt = np.concatenate((nxt, np.empty(nxt.size, np.int64)))
                    nxt[n_nxt] = nk
                    n_nxt += 1
        b = keys.size * 12 + (cur.size + nxt.size) * 8
        if b > peak_bytes:
            peak_bytes = b
        if n_prev + n_cur + n_nxt > need3:
            need3 = n_prev + n_cur + n_nxt
        n_prev = n_cur
        cur, nxt = nxt, cur
        n_cur = n_nxt
        layer += 1
    return size, reads, peak_live, need3, peak_bytes, layer - 1, rebuilds, _OK


@njit(parallel=True)
def _samples(seeds, thresh, key0, bits, dim, max_live, max_layers, n_chunks):
    n = seeds.size
    size = np.empty(n, dtype=np.int64)
    reads = np.empty(n, dtype=np.int64)
    peak_live = np.empty(n, dtype=np.int64)
    need3 = np.empty(n, dtype=np.int64)
    peak_bytes = np.empty(n, dtype=np.int64)
    depth = np.empty(n, dtype=np.int64)
    rebuilds = np.empty(n, dtype=np.int64)
    status = np.empty(n, dtype=np.int64)
    # one profile row per chunk: prange over chunks gives each row one writer
    prof = np.zeros((n_chunks, max_layers), dtype=np.float64)
    prof2 = np.zeros((n_chunks, max_layers), dtype=np.float64)
    for c in prange(n_chunks):
        for r in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            (size[r], reads[r], peak_live[r], need3[r], peak_bytes[r],
             depth[r], rebuilds[r], status[r]) = _one_sample(
                seeds[r], thresh, key0, bits, dim, max_live, prof[c], prof2[c])
    return (size, reads, peak_live, need3, peak_bytes, depth, rebuilds,
            status, prof.sum(axis=0), prof2.sum(axis=0))


def device_fill() -> int:
    """ModelSpec.device_fill, as the explore model's."""
    return PROBE_SAMPLES_PER_THREAD * numba.get_num_threads()


def explore_layers(x: int, n: int = 1, dim: int = 3, eps0: float = 0.15,
                   p: float | None = None, p_c: float | None = None,
                   max_live: int = DEFAULT_MAX_LIVE,
                   max_layers: int = DEFAULT_MAX_LAYERS,
                   rng: np.random.Generator | None = None,
                   ) -> tuple[np.ndarray, dict]:
    """n i.i.d. samples of Y, plus per-sample work, memory and depth, and the
    call's layer profile sums. info keys: size, reads, peak_live, peak_bytes,
    need3, depth (-1 for a closed origin), rebuilds, prof_sum, prof_sumsq
    (the last bin holds every layer >= max_layers - 1), p."""
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}; got {dim}")
    if int(max_live) < 1 or int(max_layers) < 1:
        raise ValueError(f"need max_live, max_layers >= 1; got {max_live}, {max_layers}")
    x = int(x)
    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    bits = 63 // dim
    off = 1 << (bits - 1)
    key0 = np.int64(sum(off << (bits * j) for j in range(dim)))

    rng = rng if rng is not None else np.random.default_rng()
    # the explore model's draw, so both models see the same lattices
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    thresh = np.uint64(min(int(p * 2.0 ** 64), 2 ** 64 - 1))
    n_chunks = max(1, min(int(n), numba.get_num_threads()))
    (size, reads, peak_live, need3, peak_bytes, depth, rebuilds, status,
     prof, prof2) = _samples(seeds, thresh, key0, bits, dim, int(max_live),
                             int(max_layers), n_chunks)
    if np.any(status == _CAP):
        raise MemoryError(
            f"a sample's three live layers exceeded max_live = {int(max_live):,} "
            f"open sites at x = {x}, dim = {dim} (p = {p:.8g})")
    if np.any(status == _EDGE):
        raise RuntimeError(
            f"a cluster reached the +-2**{bits - 1} key range at x = {x}, "
            f"dim = {dim} (p = {p:.8g}); it is not wrapped or truncated, so "
            f"the sample cannot be reported")
    last = min(int(depth.max()) + 1, int(max_layers)) if n else 0
    return size / p, {"size": size, "reads": reads, "peak_live": peak_live,
                      "need3": need3,
                      "peak_bytes": peak_bytes, "depth": depth,
                      "rebuilds": rebuilds, "prof_sum": prof[:last],
                      "prof_sumsq": prof2[:last], "p": p}


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_layers"].simulate.

    params: {"dim": 2..5 (default 3), "eps0": float (default 0.15), "p", "p_c":
             as percolation_susceptibility, "max_live": int (default 2**20),
             "max_layers": int (default 2**17)}.
    """
    y, _ = explore_layers(
        x, n=n, dim=int(params.get("dim", 3)),
        eps0=float(params.get("eps0", 0.15)),
        p=params.get("p"), p_c=params.get("p_c"),
        max_live=int(params.get("max_live", DEFAULT_MAX_LIVE)),
        max_layers=int(params.get("max_layers", DEFAULT_MAX_LAYERS)), rng=rng)
    return y
