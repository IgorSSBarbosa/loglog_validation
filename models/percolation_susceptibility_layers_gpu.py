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

Two-word keys (user, 2026-11-12)
--------------------------------
The CPU models pack a site into one int64, 63 // d bits a coordinate: at
d = 5 that is +-2**11 sites from the origin, and at x = 65536 a cluster
reached it within 4e5 samples. Here a key is two words (unsigned __int128 on
the device), _KEY_BITS = 21 bits a coordinate at d = 4, 5 (+-2**20); d = 2, 3
keep the one-word layout, which already gives them >= 21 bits.

The lattice must not change where the CPU models can see it, so a site's
state depends on where it is:
- inside the old key's box: mix(seed + K * GOLDEN) < thresh with K the old
  one-word key, exactly as before -- every cluster that stays inside (all
  that the CPU models can report) is the same sample;
- outside it: the same test with K = mix(lo ^ mix(hi ^ salt)) | 2**63. No old
  key has the top bit (they are < 2**63) and GOLDEN is odd, so the hash input
  never equals an inside site's; two outside sites share one only if their
  64-bit fold collides (~2**-63 a pair).
The table and the frontier hold 16-byte keys and atomicCAS works on all 16
bytes (NVRTC --device-int128). Reads are one 16-byte load, which the hardware
does not promise to be atomic; a valid key never has an all-ones word, so a
read that is half EMPTY is recognised as torn and repeated. Cost: ~1.26x the
one-word kernel's GPU time on the same clusters (experiment 60).

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
keys (two words, all ones empty) and the layer of each key, linear probing,
as the CPU model. While layer l is expanded only layers l-1, l, l+1 are looked up, so a
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

#: live sites one sample may hold (table of 2*max_live slots, 20 bytes a slot:
#: 2.5 GiB of VRAM at the default). Experiment 58 never needed 2.4e5; a
#: 1.2e10-site cluster at x = 65536 peaked at 192 MiB (experiment 60).
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

#: bits a coordinate in the two-word key at d = 4, 5: +-2**20 sites from the
#: origin, where the one-word key of the CPU models stops at +-2**11 (d = 5)
_KEY_BITS = 21

#: bits a coordinate of the old one-word key, None = 63 // dim. Tests lower
#: it, so the far-site hash and the CPU side's hand-off at the edge are
#: exercised on small clusters; it changes the lattice outside that box.
_COMPAT_BITS = None

_GOLDEN = 0x9E3779B97F4A7C15

_KERNEL = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
typedef unsigned long long u64;
typedef unsigned __int128 u128;
#define EMPTY (~(u128)0)
#define GOLDEN 0x9E3779B97F4A7C15ULL
#define FAR_SALT 0xD1B54A32D192ED03ULL

__device__ __forceinline__ u64 mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ u64 slot_of(const u128 k) {
    return mix((u64)k ^ mix((u64)(k >> 64) + GOLDEN));
}

// One 16-byte load, past L1 (other SMs write the table). It is not promised
// to be single-copy atomic, so the caller re-reads a torn value.
__device__ __forceinline__ u128 load(const u128* p) {
    const ulonglong2 v = __ldcg((const ulonglong2*)p);
    return ((u128)v.y << 64) | (u128)v.x;
}

// Is site k open? Inside the old one-word key's box the state is the CPU
// models' mix(seed + key * GOLDEN) of that old key; outside it the "key" is a
// hash of both words with the top bit set, which no old key has (< 2**63).
__device__ __forceinline__ bool open_site(const u128 k, const u64 seed, const u64 thresh,
        const int dim, const int nb, const int per, const int ob) {
    const u64 lo = (u64)k, hi = (u64)(k >> 64);
    u64 key = 0;
    if (ob == nb && dim <= per) {
        key = lo;                       // d = 2, 3: the layout IS the old key
    } else {
        const long long nmask = (1LL << nb) - 1;
        const long long shift_back = (1LL << (ob - 1)) - (1LL << (nb - 1));
        const long long ospan = 1LL << ob;
        for (int j = 0; j < dim; ++j) {
            const u64 w = (j < per) ? lo : hi;
            const int sh = nb * ((j < per) ? j : j - per);
            const long long g = (long long)((w >> sh) & nmask) + shift_back;
            if (g < 0 || g >= ospan) { key = mix(lo ^ mix(hi ^ FAR_SALT)) | (1ULL << 63); break; }
            key |= (u64)g << (ob * j);
        }
    }
    return mix(seed + key * GOLDEN) < thresh;
}

// Grow layers layer0, layer0+1, ... from the n_cur sites in buf0.
// Keys are two words: coordinate j is nb bits at nb*(j % per) of word j / per.
// slots[3][2]: per rotating slot, (sites appended, edge flag); slot 1 zero on entry.
// out[0..2] = (layers run, n_cur after them, status), out[3 + i] = N of layer layer0+i+1.
extern "C" __global__ void grow(
        u128* buf0, u128* buf1, const long long buf_cap,
        u128* keys, int* lay, const u64 mask, const long long occ_limit, long long occ,
        long long n_cur, const int layer0, const int max_iters,
        const u64 seed, const u64 thresh, const int dim, const int nb, const int per,
        const int ob, u64* slots, long long* out) {
    cg::grid_group grid = cg::this_grid();
    const long long tid = grid.thread_rank();
    const long long nth = grid.size();
    const u64 fmask = (1ULL << nb) - 1;
    int it = 0, status;
    while (true) {
        const int l = layer0 + it;
        const long long B = (l == 0) ? 2 * dim : 2 * dim - 1;
        if (n_cur == 0) { status = 0; break; }
        if (occ + B * n_cur > occ_limit) { status = 1; break; }
        if (B * n_cur > buf_cap) { status = 2; break; }
        if (it == max_iters) { status = 3; break; }
        const u128* cur = (it & 1) ? buf1 : buf0;
        u128* nxt = (it & 1) ? buf0 : buf1;
        u64* s = slots + 2 * ((it + 1) % 3);
        if (tid == 0) { u64* z = slots + 2 * ((it + 2) % 3); z[0] = 0; z[1] = 0; }
        for (long long t = tid; t < n_cur * 2 * dim; t += nth) {
            const u128 k = load(cur + t / (2 * dim));
            const int r = (int)(t % (2 * dim));
            const int j = r >> 1;
            const int hiw = j >= per;
            const int shift = nb * (hiw ? j - per : j);
            const u64 f = ((u64)(k >> (hiw ? 64 : 0)) >> shift) & fmask;
            if (f == 0 || f == fmask) { atomicOr(s + 1, 1ULL); continue; }
            const u128 step = (u128)(1ULL << shift) << (hiw ? 64 : 0);
            const u128 nk = (r & 1) ? k - step : k + step;
            if (!open_site(nk, seed, thresh, dim, nb, per, ob)) continue;
            u64 h = slot_of(nk) & mask;
            while (true) {
                // a stale read can only be EMPTY (no key is removed within a
                // launch), and the CAS below returns the true value
                const u128 c = load(keys + h);
                if (c == nk) break;
                if (c == EMPTY) {
                    const u128 old = atomicCAS(keys + h, EMPTY, nk);
                    if (old == EMPTY) {
                        lay[h] = l + 1;
                        nxt[atomicAdd(s, 1ULL)] = nk;
                        break;
                    }
                    if (old == nk) break;
                } else if ((u64)c == ~0ULL || (u64)(c >> 64) == ~0ULL) {
                    continue;           // torn mid-CAS (no key has an all-ones word): re-read
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
typedef unsigned __int128 u128;
#define EMPTY (~(u128)0)
#define GOLDEN 0x9E3779B97F4A7C15ULL

__device__ __forceinline__ u64 mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ u64 slot_of(const u128 k) {
    return mix((u64)k ^ mix((u64)(k >> 64) + GOLDEN));
}

// copy the entries with layer >= keep_from into an empty table
extern "C" __global__ void rebuild(
        const u128* keys, const int* lay, const long long size,
        u128* nkeys, int* nlay, const u64 nmask, const int keep_from, u64* kept) {
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < size;
         t += (long long)gridDim.x * blockDim.x) {
        const u128 k = keys[t];
        if (k == EMPTY || lay[t] < keep_from) continue;
        u64 h = slot_of(k) & nmask;
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
        opts = ("--device-int128",)     # the two-word keys and their atomicCAS
        mod = cp.RawModule(code=_KERNEL, options=opts, enable_cooperative_groups=True)
        _compiled["grow"] = mod.get_function("grow")
        _compiled["rebuild"] = cp.RawModule(code=_REBUILD, options=opts).get_function("rebuild")
        _compiled["blocks"] = cp.cuda.Device().attributes["MultiProcessorCount"]
    return _compiled["grow"], _compiled["rebuild"], _compiled["blocks"]


def _mix_host(z: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _slot_host(lo: int, hi: int) -> int:
    """The kernels' slot_of for a two-word key, before masking."""
    with np.errstate(over="ignore"):
        return int(_mix_host(np.uint64(lo) ^ _mix_host(np.uint64(hi) + np.uint64(_GOLDEN))))


class _Table:
    """The device hash table of one sample: keys (two int64 words a slot, low
    word first, as the kernels' unsigned __int128), layers, occupancy."""

    def __init__(self, cp, key0: tuple[int, int], size: int):
        self.cp = cp
        self.keys = cp.full(2 * size, -1, dtype=cp.int64)
        self.lay = cp.zeros(size, dtype=cp.int32)
        self.kept = cp.zeros(1, dtype=cp.uint64)
        self.occ = 1
        self.rebuilds = 0
        h = _slot_host(*key0) & (size - 1)
        self.keys[2 * h:2 * h + 2] = cp.asarray(key0, dtype=cp.int64)

    @property
    def size(self) -> int:
        return int(self.lay.size)

    def rebuild(self, new_size: int, keep_from: int) -> None:
        cp = self.cp
        _, rebuild, _ = _kernels(cp)
        nkeys = cp.full(2 * new_size, -1, dtype=cp.int64)
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


def grow_cluster(seed: int, thresh: int, dim: int,
                 max_live: int = DEFAULT_MAX_LIVE) -> dict:
    """Grow the cluster of the origin for one sample seed on the GPU.

    Returns {"size", "depth", "layers" (N_l, int64 array), "rebuilds",
    "launches", "peak_bytes"}. size 0 and depth -1 for a closed origin
    (checked on the host, no launch)."""
    ob, old0 = _geometry(dim)
    if not _is_open(np.uint64(seed), np.int64(old0), np.uint64(thresh)):
        return {"size": 0, "depth": -1, "layers": np.zeros(0, np.int64),
                "rebuilds": 0, "launches": 0, "peak_bytes": 0}
    nb, per, key0 = _layout(dim)
    cp = _cupy()
    grow, _, blocks = _kernels(cp)
    table_cap = 1 << max(int(2 * max_live - 1).bit_length(), 1)
    tab = _Table(cp, key0, min(_T0, table_cap))
    buf_cap = 1 << 12
    buf0 = cp.empty(2 * buf_cap, dtype=cp.int64)
    buf0[:2] = cp.asarray(key0, dtype=cp.int64)
    buf1 = cp.empty(2 * buf_cap, dtype=cp.int64)
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
            nbuf = cp.empty(2 * buf_cap, dtype=cp.int64)
            nbuf[:2 * n_cur] = buf0[:2 * n_cur]
            buf0, buf1 = nbuf, cp.empty(2 * buf_cap, dtype=cp.int64)
        slots.fill(0)
        grow((blocks,), (_THREADS,),
             (buf0, buf1, np.int64(buf_cap), tab.keys, tab.lay,
              np.uint64(tab.size - 1), np.int64(int(_HARD_LOAD * tab.size)),
              np.int64(tab.occ), np.int64(n_cur), np.int32(layer),
              np.int32(_MAX_ITERS), seed_u, thresh_u, np.int32(dim),
              np.int32(nb), np.int32(per), np.int32(ob), slots, out))
        launches += 1
        head = out[:3].get()
        it, n_cur, status = (int(v) for v in head)
        if status == _EDGE:
            raise RuntimeError(
                f"a cluster reached the +-2**{nb - 1} key range (dim = {dim}); "
                "it is not wrapped or truncated, so the sample cannot be reported")
        new = out[3:3 + it].get()
        tab.occ += int(new.sum())
        layers.append(new)
        layer += it
        if it & 1:
            buf0, buf1 = buf1, buf0
        peak_bytes = max(peak_bytes, tab.size * 20 + 2 * buf_cap * 16)
    lay = np.concatenate(layers)[:-1]   # the last entry is the empty layer
    return {"size": int(lay.sum()), "depth": layer - 1, "layers": lay,
            "rebuilds": tab.rebuilds, "launches": launches, "peak_bytes": peak_bytes}


def _geometry(dim: int) -> tuple[int, int]:
    """(bits, origin key) of the CPU models' one-word key: the box inside which
    a site's state is theirs, and the CPU side of the hybrid's layout."""
    bits = _COMPAT_BITS or 63 // dim
    off = 1 << (bits - 1)
    return bits, int(sum(off << (bits * j) for j in range(dim)))


def _layout(dim: int) -> tuple[int, int, tuple[int, int]]:
    """(bits a coordinate, coordinates a word, origin as (low, high) word) of
    the two-word key. d = 2, 3 keep the one-word layout (all in the low word,
    which is then the old key); d = 4, 5 get _KEY_BITS a coordinate."""
    nb = max(63 // dim, _KEY_BITS)
    per = 63 // nb
    if dim > 2 * per:
        raise ValueError(f"dim {dim} does not fit two words at {nb} bits")
    off = 1 << (nb - 1)
    lo = sum(off << (nb * j) for j in range(min(dim, per)))
    hi = sum(off << (nb * j) for j in range(dim - per)) if dim > per else 0
    return nb, per, (int(lo), int(hi))


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
    rng = rng if rng is not None else np.random.default_rng()
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    thresh = min(int(p * 2.0 ** 64), 2 ** 64 - 1)
    out = {k: np.zeros(n, dtype=np.int64)
           for k in ("size", "depth", "rebuilds", "launches", "peak_bytes")}
    for r, s in enumerate(seeds):
        g = grow_cluster(int(s), thresh, dim, max_live)
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
