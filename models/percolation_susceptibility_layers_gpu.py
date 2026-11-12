"""models/percolation_susceptibility_layers.py with each cluster grown on a CUDA
GPU, many chemical layers per kernel launch -- registered as
MODELS["percolation_susceptibility_layers_gpu"] in tools/models.py.

Why (user, 2026-11-12): at d = 5 the cost is set by a few huge clusters
(experiment 58: at x = 16384 one cluster was 32 % of all reads, ~10 min on one
CPU thread). Parallelising WITHIN a cluster, not across clusters, keeps the
memory of one exploration (three live layers) while spreading its work. This
module is the first part of a plan: measure where GPU growth beats a CPU thread,
then send only clusters above a size cut to the GPU (a separate hybrid model).

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

One launch, many layers (experiment 59, step 3)
-----------------------------------------------
The first version launched one kernel per layer and read N_{l+1} back each
time: 24.5 us a layer, of which a layer's actual work was a small part (step
2). Now one cooperative launch runs layer after layer with a grid-wide barrier
(`grid.sync()`, ~0.9 us on an RTX 5090) between them. What makes it exact:

- the counters (sites appended, edge flag) rotate over three slots: layer l
  appends to slot l+1, reads slot l+1 after the barrier, and thread 0 clears
  slot l+2 during layer l, when no thread reads it any more;
- every thread takes the same stop decision, from values no thread changes
  until the next barrier: N_{l+1} and the occupancy, which needs no atomic
  counter because every successful CAS appends exactly one site, so
  occupancy = occupancy at launch + sum of the N appended since;
- the kernel stops BEFORE a layer whose worst case could overfill the table
  or the frontier buffer: a layer l >= 1 adds at most (2d-1) N_l sites (each
  site's parent is already in the table), layer 0 at most 2d. So inserts are
  never refused and nothing is ever redone.

The host then makes room and launches again. It does so when the table would
be more than 3/4 full in the worst case, when the frontier buffer could
overflow, or after `max_iters` layers (the size of the per-launch N_l record).

The table on the device
-----------------------
keys (int64, -1 empty) and the layer of each key, linear probing, as the CPU
model. While layer l is expanded only layers l-1, l, l+1 are looked up, so a
rebuild keeping layers >= l-1 is exact. Between launches the host rebuilds when
occupancy + worst-case growth would pass half the table: it drops old layers,
then grows the table to _SLACK times that need, so the next launch runs many
layers before it stops for room. The table stops growing at the first power of two >= 2
max_live slots; if worst-case growth cannot fit in 3/4 of it after dropping old
layers, the sample fails with MemoryError. Because the room is reserved for
the worst case, this cap binds earlier than the CPU model's cap does (a table
~2(2d+2) N_l slots, not ~6 N_l); at the default max_live it is far away.
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

#: threads a block; blocks = one per SM (the cooperative launch's limit for
#: this kernel on the RTX 5090, see experiment 59)
_THREADS = 256

#: layers one launch may run (it records N_l for each)
_MAX_ITERS = 1 << 14

#: the kernel stops before a layer whose worst case would pass this load (tests
#: lower it to make launches end often)
_HARD_LOAD = 0.75

#: a rebuild that grows the table makes it this many times the worst-case need
_SLACK = 8

_DONE, _ROOM, _BUF, _ITERS, _EDGE = 0, 1, 2, 3, 4

_KERNEL = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
typedef unsigned long long u64;
#define EMPTY 0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__ u64 mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Grow layers layer0, layer0+1, ... from the n_cur sites in buf0.
// slots[3][2]: per rotating slot, (sites appended, edge flag); slot 1 zero on entry.
// out[0..2] = (layers run, n_cur after them, status), out[3 + i] = N of layer layer0+i+1.
extern "C" __global__ void grow(
        long long* buf0, long long* buf1, const long long buf_cap,
        u64* keys, int* lay, const u64 mask, const long long occ_limit, long long occ,
        long long n_cur, const int layer0, const int max_iters,
        const u64 seed, const u64 thresh, const int bits, const int dim,
        u64* slots, long long* out) {
    cg::grid_group grid = cg::this_grid();
    const long long tid = grid.thread_rank();
    const long long nth = grid.size();
    const long long fmask = (1LL << bits) - 1;
    int it = 0, status;
    while (true) {
        const int l = layer0 + it;
        const long long B = (l == 0) ? 2 * dim : 2 * dim - 1;
        if (n_cur == 0) { status = 0; break; }
        if (occ + B * n_cur > occ_limit) { status = 1; break; }
        if (B * n_cur > buf_cap) { status = 2; break; }
        if (it == max_iters) { status = 3; break; }
        const long long* cur = (it & 1) ? buf1 : buf0;
        long long* nxt = (it & 1) ? buf0 : buf1;
        u64* s = slots + 2 * ((it + 1) % 3);
        if (tid == 0) { u64* z = slots + 2 * ((it + 2) % 3); z[0] = 0; z[1] = 0; }
        for (long long t = tid; t < n_cur * 2 * dim; t += nth) {
            const long long k = __ldcg(cur + t / (2 * dim));
            const int r = (int)(t % (2 * dim));
            const int shift = bits * (r >> 1);
            const long long f = (k >> shift) & fmask;
            if (f == 0 || f == fmask) { atomicOr(s + 1, 1ULL); continue; }
            const long long step = 1LL << shift;
            const long long nk = (r & 1) ? k - step : k + step;
            if (mix(seed + (u64)nk * 0x9E3779B97F4A7C15ULL) >= thresh) continue;
            u64 h = mix((u64)nk) & mask;
            while (true) {
                // a stale cached read can only be EMPTY (no key is removed
                // within a launch), and the CAS below returns the true value
                const u64 c = keys[h];
                if (c == (u64)nk) break;
                if (c == EMPTY) {
                    const u64 old = atomicCAS(keys + h, EMPTY, (u64)nk);
                    if (old == EMPTY) {
                        lay[h] = l + 1;
                        nxt[atomicAdd(s, 1ULL)] = nk;
                        break;
                    }
                    if (old == (u64)nk) break;
                }
                h = (h + 1) & mask;
            }
        }
        grid.sync();
        const long long n_new = (long long)__ldcg(s);
        const u64 edge = __ldcg(s + 1);
        if (tid == 0) out[3 + it] = n_new;
        occ += n_new;
        n_cur = n_new;
        ++it;
        if (edge) { status = 4; break; }
    }
    if (tid == 0) { out[0] = it; out[1] = n_cur; out[2] = status; }
}

"""

# Compiled apart from `grow`: in a cooperative module CuPy clamps EVERY
# kernel's grid to the co-resident limit, and a one-thread-a-slot rebuild then
# skipped the slots past it (found by experiment 59's bench, 2026-11-12).
# Grid-stride as well, so no clamp can drop a slot.
_REBUILD = r"""
typedef unsigned long long u64;
#define EMPTY 0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__ u64 mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// copy the entries with layer >= keep_from into an empty table
extern "C" __global__ void rebuild(
        const u64* keys, const int* lay, const long long size,
        u64* nkeys, int* nlay, const u64 nmask, const int keep_from, u64* kept) {
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < size;
         t += (long long)gridDim.x * blockDim.x) {
        const u64 k = keys[t];
        if (k == EMPTY || lay[t] < keep_from) continue;
        u64 h = mix(k) & nmask;
        while (atomicCAS(nkeys + h, EMPTY, k) != EMPTY) h = (h + 1) & nmask;
        nlay[h] = lay[t];
        atomicAdd(kept, 1ULL);
    }
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
    """(grow, rebuild, blocks for grow), compiled once per process."""
    if not _compiled:
        mod = cp.RawModule(code=_KERNEL, enable_cooperative_groups=True)
        _compiled["grow"] = mod.get_function("grow")
        _compiled["rebuild"] = cp.RawModule(code=_REBUILD).get_function("rebuild")
        _compiled["blocks"] = cp.cuda.Device().attributes["MultiProcessorCount"]
    return _compiled["grow"], _compiled["rebuild"], _compiled["blocks"]


def _mix_host(z: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class _Table:
    """The device hash table of one sample: keys, layers, occupancy."""

    def __init__(self, cp, key0: int, size: int):
        self.cp = cp
        self.keys = cp.full(size, -1, dtype=cp.int64)
        self.lay = cp.zeros(size, dtype=cp.int32)
        self.kept = cp.zeros(1, dtype=cp.uint64)
        self.occ = 1
        self.rebuilds = 0
        self.keys[int(_mix_host(np.uint64(key0)) & np.uint64(size - 1))] = key0

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def rebuild(self, new_size: int, keep_from: int) -> None:
        cp = self.cp
        _, rebuild, _ = _kernels(cp)
        nkeys = cp.full(new_size, -1, dtype=cp.int64)
        nlay = cp.zeros(new_size, dtype=cp.int32)
        self.kept[0] = 0
        rebuild((min((self.size + 255) // 256, 1 << 16),), (256,),
                (self.keys, self.lay, np.int64(self.size), nkeys, nlay,
                 np.uint64(new_size - 1), np.int32(keep_from), self.kept))
        self.keys, self.lay = nkeys, nlay
        self.occ = int(self.kept.get()[0])
        self.rebuilds += 1

    def make_room(self, layer: int, grow_by: int, table_cap: int) -> None:
        """Before expanding `layer`, which adds at most `grow_by` sites: if the
        worst case passes half the table, drop layers <= layer-2, then double up
        to the cap. MemoryError if the worst case still passes the hard load."""
        if self.occ + grow_by <= self.size // 2:
            return
        self.rebuild(self.size, max(layer - 1, 0))
        # slack, so the next launch runs many layers before it stops for room
        # (sized just to fit, d = 2 launches ran ~3 layers each: experiment 60)
        target = min(1 << (_SLACK * (self.occ + grow_by) - 1).bit_length(), table_cap)
        if target > self.size:
            self.rebuild(target, 0)
        if self.occ + grow_by > int(_HARD_LOAD * self.size):
            raise MemoryError(
                f"layer {layer}: {self.occ:,} live sites plus a worst case of "
                f"{grow_by:,} new ones do not fit the table cap ({table_cap:,} slots)")


def grow_cluster(seed: int, thresh: int, key0: int, bits: int, dim: int,
                 max_live: int = DEFAULT_MAX_LIVE) -> dict:
    """Grow the cluster of the origin for one sample seed on the GPU.

    Returns {"size", "depth", "layers" (N_l, int64 array), "rebuilds",
    "launches", "peak_bytes"}. size 0 and depth -1 for a closed origin
    (checked on the host, no launch)."""
    if not _is_open(np.uint64(seed), np.int64(key0), np.uint64(thresh)):
        return {"size": 0, "depth": -1, "layers": np.zeros(0, np.int64),
                "rebuilds": 0, "launches": 0, "peak_bytes": 0}
    cp = _cupy()
    grow, _, blocks = _kernels(cp)
    table_cap = 1 << max(int(2 * max_live - 1).bit_length(), 1)
    tab = _Table(cp, key0, min(_T0, table_cap))
    buf_cap = 1 << 12
    buf0 = cp.full(buf_cap, key0, dtype=cp.int64)
    buf1 = cp.empty(buf_cap, dtype=cp.int64)
    slots = cp.zeros(6, dtype=cp.uint64)
    out = cp.zeros(3 + _MAX_ITERS, dtype=cp.int64)
    n_cur, layer, launches, peak_bytes = 1, 0, 0, 0
    layers = [np.array([1], dtype=np.int64)]
    seed_u, thresh_u = np.uint64(seed), np.uint64(thresh)
    while n_cur > 0:
        bound = (2 * dim if layer == 0 else 2 * dim - 1) * n_cur
        tab.make_room(layer, bound, table_cap)
        if bound > buf_cap:
            # slack: a launch should run many layers before the next resize
            buf_cap = 1 << (8 * bound - 1).bit_length()
            nb = cp.empty(buf_cap, dtype=cp.int64)
            nb[:n_cur] = buf0[:n_cur]
            buf0, buf1 = nb, cp.empty(buf_cap, dtype=cp.int64)
        slots.fill(0)
        grow((blocks,), (_THREADS,),
             (buf0, buf1, np.int64(buf_cap), tab.keys, tab.lay,
              np.uint64(tab.size - 1), np.int64(int(_HARD_LOAD * tab.size)),
              np.int64(tab.occ), np.int64(n_cur), np.int32(layer),
              np.int32(_MAX_ITERS), seed_u, thresh_u, np.int32(bits),
              np.int32(dim), slots, out))
        launches += 1
        head = out[:3].get()
        it, n_cur, status = (int(v) for v in head)
        if status == _EDGE:
            raise RuntimeError(
                f"a cluster reached the +-2**{bits - 1} key range (dim = {dim}); "
                "it is not wrapped or truncated, so the sample cannot be reported")
        new = out[3:3 + it].get()
        tab.occ += int(new.sum())
        layers.append(new)
        layer += it
        if it & 1:
            buf0, buf1 = buf1, buf0
        peak_bytes = max(peak_bytes, tab.size * 12 + 2 * buf_cap * 8)
    lay = np.concatenate(layers)[:-1]   # the last entry is the empty layer
    return {"size": int(lay.sum()), "depth": layer - 1, "layers": lay,
            "rebuilds": tab.rebuilds, "launches": launches, "peak_bytes": peak_bytes}


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
    launches, peak_bytes, p."""
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
           for k in ("size", "depth", "rebuilds", "launches", "peak_bytes")}
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
