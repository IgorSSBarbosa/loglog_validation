"""models/percolation_susceptibility_layers.py with each cluster grown on a CUDA
GPU, one chemical layer per kernel launch -- registered as
MODELS["percolation_susceptibility_layers_gpu"] in tools/models.py.

Why (user, 2026-11-12): at d = 5 the cost is set by a few huge clusters
(experiment 58: at x = 16384 one cluster was 32 % of all reads, ~10 min on one
CPU thread). Parallelising WITHIN a cluster, not across clusters, keeps the
memory of one exploration (three live layers) while spreading its work. This
module is the first part of a plan: measure where GPU growth beats a CPU thread,
then send only clusters above a size cut to the GPU (the hybrid, not built yet).

Same sample as the CPU model, bit for bit
-----------------------------------------
Site states are the explore/layers models' pure function of (sample seed, key)
(splitmix64, see models/percolation_susceptibility_explore.py), and the seeds are
the same one `rng.integers(0, 2**63)` per sample. The cluster is a function of
the lattice alone, so |C|, the depth and every layer size N_l equal the CPU
model's, sample by sample. Only the order inside a frontier differs, and it
changes nothing that is reported.

Why no colouring of the frontier
--------------------------------
Two frontier sites at distance 2 can discover the same neighbour at once (same
layer sites are never adjacent: Z^d is bipartite, a layer has one parity). A
colouring with no repeated colour at distance <= 2 would serialise them (2d+1
colours suffice: sum_i i x_i mod 2d+1). But the visited set is an
open-addressing table, where unrelated keys collide in a slot too, so inserts
need atomics anyway; with atomicCAS on the slot the thread that claims a key
is the only one to append it to the next frontier, and one pass per layer
suffices.

The table on the device
-----------------------
keys (int64, -1 empty) and the layer of each key, linear probing, as the CPU
model. While layer l is expanded only layers l-1, l, l+1 are looked up, so a
rebuild keeping layers >= l-1 is exact. Before each layer the host, which reads
N_{l+1} after every launch, rebuilds when the table is more than half full
(dropping old layers, and doubling if that is not enough). Inside a launch an
insert that would pass 3/4 load is refused and flagged; the host then grows or
rebuilds and REPEATS the layer. A repeat is idempotent: keys already inserted
are found and skipped (they are in the frontier already), the refused ones are
inserted now. The table stops growing at the first power of two >= 2 max_live
slots (so up to ~1.5 max_live live sites at 3/4 load); if dropping old layers
frees nothing there, the sample fails with MemoryError, as the CPU model.

Cost of a layer: one launch and one 24-byte device->host read of the counters
(the synchronisation that tells the host N_{l+1}). That is the price per layer that a CPU thread
does not pay, and why small clusters belong on the CPU.
"""

from __future__ import annotations

import numpy as np

from models.percolation_susceptibility import p_at
from models.percolation_susceptibility_explore import DIMS, _is_open

#: live sites one sample may hold (table of 2*max_live slots, 12 bytes a slot:
#: 1.5 GiB of VRAM at the default). Experiment 58 never needed 2.4e5.
DEFAULT_MAX_LIVE = 1 << 26

#: first table size (slots); it doubles as needed
_T0 = 1 << 14

_THREADS = 256

#: inserts past this load are refused and the layer repeated (tests lower it
#: to exercise the repeat path, which a 3/4 limit almost never reaches)
_HARD_LOAD = 0.75
_EDGE, _OVER, _BUF = 1, 2, 4

_KERNEL = r"""
typedef unsigned long long u64;
#define EMPTY 0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__ u64 mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// one thread per (frontier site, direction)
extern "C" __global__ void expand(
        const long long* cur, const long long n_cur,
        u64* keys, int* lay, const u64 mask,
        long long* nxt, const u64 nxt_cap, u64* ctr, const u64 occ_limit,
        const u64 seed, const u64 thresh,
        const int bits, const int dim, const int layer) {
    // ctr[0] = sites appended to nxt, ctr[1] = table occupancy, ctr[2] = flags
    u64* n_nxt = ctr;
    u64* occ = ctr + 1;
    u64* flags = ctr + 2;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_cur * 2 * dim) return;
    const long long k = cur[t / (2 * dim)];
    const int r = (int)(t % (2 * dim));
    const int shift = bits * (r >> 1);
    const long long fmask = (1LL << bits) - 1;
    const long long f = (k >> shift) & fmask;
    if (f == 0 || f == fmask) { atomicOr(flags, 1ULL); return; }
    const long long step = 1LL << shift;
    const long long nk = (r & 1) ? k - step : k + step;
    if (mix(seed + (u64)nk * 0x9E3779B97F4A7C15ULL) >= thresh) return;
    u64 h = mix((u64)nk) & mask;
    bool reserved = false;
    while (true) {
        u64 cur_key = keys[h];
        if (cur_key == (u64)nk) break;
        if (cur_key == EMPTY) {
            if (!reserved) {
                if (atomicAdd(occ, 1ULL) >= occ_limit) {
                    atomicAdd(occ, (u64)-1LL);
                    atomicOr(flags, 2ULL);
                    return;
                }
                reserved = true;
            }
            u64 old = atomicCAS(keys + h, EMPTY, (u64)nk);
            if (old == EMPTY) {
                lay[h] = layer + 1;
                u64 i = atomicAdd(n_nxt, 1ULL);
                if (i < nxt_cap) nxt[i] = nk; else atomicOr(flags, 4ULL);
                return;
            }
            if (old == (u64)nk) break;
        }
        h = (h + 1) & mask;
    }
    if (reserved) atomicAdd(occ, (u64)-1LL);
}

// copy the entries with layer >= keep_from into an empty table
extern "C" __global__ void rebuild(
        const u64* keys, const int* lay, const long long size,
        u64* nkeys, int* nlay, const u64 nmask, const int keep_from, u64* kept) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= size) return;
    const u64 k = keys[t];
    if (k == EMPTY || lay[t] < keep_from) return;
    u64 h = mix(k) & nmask;
    while (atomicCAS(nkeys + h, EMPTY, k) != EMPTY) h = (h + 1) & nmask;
    nlay[h] = lay[t];
    atomicAdd(kept, 1ULL);
}
"""


def _cupy():
    """cupy, or a RuntimeError naming the CPU model."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["percolation_susceptibility_layers_gpu"] needs cupy (see '
            'requirements.txt). Without a GPU use '
            'MODELS["percolation_susceptibility_layers"], the same samples on CPU.'
        ) from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f"no usable CUDA device ({err}); use "
            'MODELS["percolation_susceptibility_layers"], the same samples on CPU.'
        ) from err
    return cupy


_compiled: dict = {}


def _kernels(cp):
    """(expand, rebuild), compiled once per process."""
    if not _compiled:
        mod = cp.RawModule(code=_KERNEL)
        _compiled["expand"] = mod.get_function("expand")
        _compiled["rebuild"] = mod.get_function("rebuild")
    return _compiled["expand"], _compiled["rebuild"]


def _blocks(n: int) -> tuple[int]:
    return ((n + _THREADS - 1) // _THREADS,)


class _Table:
    """The device hash table of one sample: keys, layers, occupancy."""

    def __init__(self, cp, key0: int, size: int):
        self.cp = cp
        self.keys = cp.full(size, -1, dtype=cp.int64)
        self.lay = cp.zeros(size, dtype=cp.int32)
        self.kept = cp.zeros(1, dtype=cp.uint64)
        self.n_occ = 1
        self.rebuilds = 0
        h = int(_mix_host(np.uint64(key0)) & np.uint64(size - 1))
        self.keys[h] = key0

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def rebuild(self, new_size: int, keep_from: int) -> int:
        cp = self.cp
        _, rebuild = _kernels(cp)
        nkeys = cp.full(new_size, -1, dtype=cp.int64)
        nlay = cp.zeros(new_size, dtype=cp.int32)
        self.kept[0] = 0
        rebuild(_blocks(self.size), (_THREADS,),
                (self.keys, self.lay, np.int64(self.size), nkeys, nlay,
                 np.uint64(new_size - 1), np.int32(keep_from), self.kept))
        self.keys, self.lay = nkeys, nlay
        self.n_occ = int(self.kept.get()[0])
        self.rebuilds += 1
        return self.n_occ

    def make_room(self, layer: int, table_cap: int) -> None:
        """Bring the load to <= 1/2 before expanding `layer` if it can be:
        drop layers <= layer-2, then double up to the cap. A full table at the
        cap is left to the kernel's 3/4 limit and the repeat path."""
        if 2 * self.n_occ <= self.size:
            return
        self.rebuild(self.size, max(layer - 1, 0))
        while 2 * self.n_occ > self.size and self.size < table_cap:
            self.rebuild(2 * self.size, 0)


def _mix_host(z: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def grow_cluster(seed: int, thresh: int, key0: int, bits: int, dim: int,
                 max_live: int = DEFAULT_MAX_LIVE) -> dict:
    """Grow the cluster of the origin for one sample seed on the GPU.

    Returns {"size", "depth", "layers" (N_l, int64 array), "rebuilds",
    "repeats" (layers repeated after a refused insert), "peak_bytes"}.
    size 0 and depth -1 for a closed origin (checked on the host)."""
    if not _is_open(np.uint64(seed), np.int64(key0), np.uint64(thresh)):
        return {"size": 0, "depth": -1, "layers": np.zeros(0, np.int64),
                "rebuilds": 0, "repeats": 0, "peak_bytes": 0}
    cp = _cupy()
    expand, _ = _kernels(cp)
    table_cap = 1 << max(int(2 * max_live - 1).bit_length(), 1)
    tab = _Table(cp, key0, min(_T0, table_cap))
    cur = cp.full(1024, key0, dtype=cp.int64)
    nxt = cp.empty(1024, dtype=cp.int64)
    ctr = cp.zeros(3, dtype=cp.uint64)
    n_cur = 1
    layer = 0
    layers = []
    repeats = 0
    peak_bytes = 0
    seed_u, thresh_u = np.uint64(seed), np.uint64(thresh)
    while n_cur > 0:
        layers.append(n_cur)
        # a layer adds at most (2 dim - 1) n_cur sites (2 dim at layer 0)
        bound = 2 * dim * n_cur
        if nxt.size < bound:
            nxt = cp.empty(1 << (bound - 1).bit_length(), dtype=cp.int64)
        n_new = 0
        while True:
            tab.make_room(layer, table_cap)
            ctr.set(np.array([n_new, tab.n_occ, 0], dtype=np.uint64))
            expand(_blocks(bound), (_THREADS,),
                   (cur, np.int64(n_cur), tab.keys, tab.lay,
                    np.uint64(tab.size - 1), nxt, np.uint64(nxt.size), ctr,
                    np.uint64(int(_HARD_LOAD * tab.size)), seed_u, thresh_u,
                    np.int32(bits), np.int32(dim), np.int32(layer)))
            n_new, tab.n_occ, fl = (int(v) for v in ctr.get())
            if fl & _EDGE:
                raise RuntimeError(
                    f"a cluster reached the +-2**{bits - 1} key range (dim = {dim}); "
                    "it is not wrapped or truncated, so the sample cannot be reported")
            if fl & _BUF:
                raise AssertionError("frontier buffer overflow: the bound is wrong")
            if not fl & _OVER:
                break
            # refused inserts: force room (half load), then repeat the layer
            repeats += 1
            if tab.size >= table_cap:
                before = tab.n_occ
                if tab.rebuild(tab.size, max(layer - 1, 0)) >= before:
                    raise MemoryError(
                        f"three live layers fill the table cap ({table_cap:,} slots)")
            else:
                tab.rebuild(2 * tab.size, 0)
        b = tab.size * 12 + (cur.size + nxt.size) * 8
        peak_bytes = max(peak_bytes, b)
        cur, nxt = nxt, cur
        n_cur = n_new
        layer += 1
    lay = np.asarray(layers, dtype=np.int64)
    return {"size": int(lay.sum()), "depth": layer - 1, "layers": lay,
            "rebuilds": tab.rebuilds, "repeats": repeats, "peak_bytes": peak_bytes}


def _geometry(dim: int) -> tuple[int, int]:
    bits = 63 // dim
    off = 1 << (bits - 1)
    return bits, int(sum(off << (bits * j) for j in range(dim)))


def explore_layers_gpu(x: int, n: int = 1, dim: int = 3, eps0: float = 0.15,
                       p: float | None = None, p_c: float | None = None,
                       max_live: int = DEFAULT_MAX_LIVE,
                       rng: np.random.Generator | None = None,
                       ) -> tuple[np.ndarray, dict]:
    """n samples of Y = |C(0)| / p, as explore_layers (same rng -> same Y).
    info: size, depth, reads (1 + 2 dim |C|, the CPU model's count), rebuilds,
    repeats, peak_bytes, p."""
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}; got {dim}")
    x = int(x)
    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    bits, key0 = _geometry(dim)
    rng = rng if rng is not None else np.random.default_rng()
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    thresh = min(int(p * 2.0 ** 64), 2 ** 64 - 1)
    out = {k: np.zeros(n, dtype=np.int64)
           for k in ("size", "depth", "rebuilds", "repeats", "peak_bytes")}
    for r, s in enumerate(seeds):
        g = grow_cluster(int(s), thresh, key0, bits, dim, max_live)
        for k in out:
            out[k][r] = g[k]
    out["reads"] = np.where(out["size"] > 0, 1 + 2 * dim * out["size"], 1)
    out["p"] = p
    return out["size"] / p, out


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_layers_gpu"].simulate; params as
    percolation_susceptibility_layers (max_live default 2**26)."""
    y, _ = explore_layers_gpu(
        x, n=n, dim=int(params.get("dim", 3)),
        eps0=float(params.get("eps0", 0.15)),
        p=params.get("p"), p_c=params.get("p_c"),
        max_live=int(params.get("max_live", DEFAULT_MAX_LIVE)), rng=rng)
    return y
