"""models/rwre.py's walk on a lazily revealed environment, on a CUDA GPU.

Phase A of plans/rwre_gpu_lazy.md: the byte layout. Not yet registered in
tools/models.py (that is phase C, with `cost_hint`); until then it is reached
as `from models import rwre_gpu_lazy`. Acceptance criteria and measured
results: experiments/18_rwre_gpu_lazy/README.md.

Same walk, same environment, same observable Y_k = |X_k| as MODELS["rwre"] and
MODELS["rwre_gpu"] -- the walker on a stirred SSEP, decisions (a)-(c) of
models/rwre.py -- but on Z itself: there is no window, so decision (d) and its
wrap heuristics do not apply. A separate model because the random stream
differs: a seed recorded against rwre_gpu does not reproduce this one.

Why the lazy environment is exact (plan §2)
-------------------------------------------
Swaps act on SITES, chosen independently of the configuration (rwre.py (a)),
so the content of a site at time t is the initial value at the end of its
backward stirring path, and distinct sites trace back to distinct initial
sites. A site no one has read therefore holds an initial value no one has
observed: Bernoulli(alpha), independent of everything seen so far. So each
cell is U (unrevealed), hole or particle, all U at t = 0; reading a U cell
draws its value; and a swap of two U cells changes nothing observable and
need not be simulated. Bonds touching a known cell are swapped exactly as in
rwre.py: brick-wall parity drawn per sample per sweep, each bond of that
parity with probability swap_prob, order read -> jump -> env_sweeps sweeps.

Skip-read (plan §2.1, on by default): with u the walker's uniform, the walker
goes left if u < min(p, 1-p) and right if u >= max(p, 1-p) whatever the site
holds, so only on the rest, probability |1 - 2p|, is the site read -- and
revealed. Exact, because whether to read depends on u alone. At p = 1/2
nothing is ever revealed and the walk is srw. `skip_read=False` reads every
step, for the distribution check against it (L1).

The deque (plan §3)
-------------------
Per sample, the known cells live in a buffer of `cap` bytes in shared memory,
absolute site a at buffer index a + cap/2. The deque [L, R] is the part that
is swept. Its invariant: both end cells are U. A known value moves at most one
site per sweep, so a sweep over the bonds inside [L, R] moves every known
value exactly, and checking the two ends after each sweep (and before each
read) keeps the invariant. The deque grows `growth` cells at a time on the
side that needs it and never shrinks (plan §9 Q5). Cells outside [L, R] are U
by construction: the buffer starts zeroed (U) and only bonds inside are
applied. A sample that would need a cell outside the buffer raises with its
index (plan §9 Q4); it is never discarded, which would condition the sample.

The random stream (plan §4)
---------------------------
Every draw is Philox4x32-10 (`_philox` below, checked against Random123's
known-answer vectors) keyed by the call's seed, at a counter that names what
the draw is FOR -- never which thread, buffer slot or launch draws it:

    walker uniform and revealed value at step t   (t,         0, sample, STEP)
    parity of sweep s (bit s mod 128)             (s / 128,   0, sample, PARITY)
    swap bits, bonds 128w .. 128w + 127 at sweep s (w,        s, sample, SWAP)
    swap uniforms (swap_prob != 1/2), bond 4r+j   (w,         s, sample, SWAPQ | r << 8)

with w the ABSOLUTE 128-site block (bond x = left endpoint, w = floor(x/128),
bit x mod 128; plan §4 wrote 32-bond words, the 128 bits of one Philox call
cover four of them at a quarter of the cost). So the output is a pure function
of (seed, sample index, k, p, alpha, env_sweeps, swap_prob, skip_read): it
does not depend on the launch chunk, on `growth`, or on `cap` short of an
overflow -- the L0 checks. At swap_prob = 1/2 a bond swaps iff its bit is
set, one bit per bond, which is what the bit-packed layout (phase B) will use;
otherwise it draws a 24-bit float32 uniform per bond, as rwre's float32
comparison does. The walker's uniform is a 53-bit double on [0, 1), the
revealed value a 24-bit float32 uniform against alpha, as on CPU.

One warp per sample (plan §5): every lane computes every per-step draw itself
(the same counter gives the same value, at no extra latency), so there is no
thread 0 and no barrier but `__syncwarp`.

No GPU present: as models/rwre_gpu.py, importing never imports cupy and `walk`
raises RuntimeError naming MODELS["rwre"].
"""

from __future__ import annotations

import numpy as np

__all__ = ["walk", "walk_stats", "simulate", "philox", "STATS"]

_DEFAULTS = {
    "p": 0.5,
    "alpha": 0.5,
    "env_sweeps": 4,
    "swap_prob": 0.5,
    "skip_read": True,
    # Performance only: neither changes a draw (module docstring). cap = 2**15
    # bytes is 3 samples per SM on the RTX 5090; plan §9.1 expects 1e4-3e4
    # cells at k = 2**20, and phase A' (L4) sets it from measured widths.
    "cap": 1 << 15,
    "growth": 32,
}

#: Per-sample output columns of `walk_stats`, in order.
STATS = ("x", "reveals", "ones", "reads", "width", "left", "right", "error")

#: error codes in the "error" column. DEBUG is only ever set by a debug build.
_OK, _OVERFLOW, _DEBUG_FAILED = 0, 1, 2

#: The most a block gets without opting in, and the fixed opt-in ceiling (fits
#: the RTX 5090's 99 KiB), as in models/rwre_gpu.py.
_DEFAULT_SHARED = 48 * 1024
_MAX_SHARED = 96 * 1024

#: Samples per launch. Only memory and responsiveness depend on it.
_CHUNK = 1 << 16

_SOURCE = r"""
#define TAG_STEP   1u
#define TAG_PARITY 2u
#define TAG_SWAP   3u
#define TAG_SWAPQ  4u
#define NSTAT 8
#define FULL 0xffffffffu

__device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r) { k.x += 0x9E3779B9u; k.y += 0xBB67AE85u; }
        const unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        const unsigned hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

__device__ __forceinline__ unsigned word(const uint4 r, const int i) {
    return i == 0 ? r.x : i == 1 ? r.y : i == 2 ? r.z : r.w;
}

__device__ __forceinline__ float unit24(const unsigned r) {
    return (float)(r >> 8) * 0x1p-24f;
}

extern "C" __global__ void philox_kat(const uint4* ctr, const uint2* key, uint4* out, int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = philox(ctr[i], key[i]);
}

// Swap cells j, j+1 if the bond lies in the deque. Bonds outside [L, R] join
// two U cells, a no-op.
__device__ __forceinline__ void swap_if(unsigned char* env, const int j, const bool on,
                                        const int L, const int R) {
    if (on && j >= L && j < R) {
        const unsigned char a = env[j];
        env[j] = env[j + 1];
        env[j + 1] = a;
    }
}

// One brick-wall sweep over the bonds of absolute parity b inside [L, R].
// Buffer index j is absolute j - c, and c is a multiple of 128, so the
// 128-cell blocks are aligned in both. Lane l takes bonds at offsets 2l + b
// and 2l + b + 64 of each block: the 64 bonds of a block are disjoint, and so
// are those of different blocks (a bond crossing into the next block has odd
// offset 127 and meets offset 0 of the next only at the other parity).
__device__ void sweep(unsigned char* env, const int L, const int R, const int c,
                      const int b, const unsigned s, const unsigned sample,
                      const uint2 key, const bool bits, const float q,
                      const unsigned lane) {
    const int B0 = L >> 7, B1 = (R - 1) >> 7, o = 2 * (int)lane + b;
    if (bits) {
        for (int base = B0; base <= B1; base += 32) {
            // Each lane draws one block's 128 bits; the warp then walks the
            // blocks together, so accesses stay on distinct banks.
            const uint4 r = philox(make_uint4((unsigned)(base + (int)lane - (c >> 7)), s,
                                              sample, TAG_SWAP), key);
            const int nb = min(32, B1 - base + 1);
            for (int i = 0; i < nb; ++i) {
                const unsigned w0 = __shfl_sync(FULL, r.x, i), w1 = __shfl_sync(FULL, r.y, i);
                const unsigned w2 = __shfl_sync(FULL, r.z, i), w3 = __shfl_sync(FULL, r.w, i);
                const int j = (base + i) << 7;
                swap_if(env, j + o, ((o < 32 ? w0 : w1) >> (o & 31)) & 1u, L, R);
                swap_if(env, j + o + 64, ((o < 32 ? w2 : w3) >> (o & 31)) & 1u, L, R);
            }
        }
    } else {
        for (int B = B0; B <= B1; ++B) {
            const unsigned w = (unsigned)(B - (c >> 7)), j = (unsigned)(B << 7);
            for (int h = 0; h < 2; ++h) {
                const int bond = (int)lane + 32 * h;          // bond index within the block
                const uint4 r = philox(make_uint4(w, s, sample, TAG_SWAPQ | ((unsigned)(bond >> 2) << 8)), key);
                swap_if(env, (int)j + 2 * bond + b, unit24(word(r, bond & 3)) < q, L, R);
            }
        }
    }
}

// Debug build only: known cells in [L, R] must number exactly the reveals, and
// the ones exactly the revealed ones (swaps only move values), and both end
// cells must be U.
__device__ bool deque_ok(const unsigned char* env, const int L, const int R,
                         const long long reveals, const long long ones, const unsigned lane) {
    int known = 0, one = 0;
    for (int j = L + (int)lane; j <= R; j += 32) {
        known += env[j] & 1;
        one += env[j] >> 1;
    }
    known = __reduce_add_sync(FULL, known);
    one = __reduce_add_sync(FULL, one);
    return known == reveals && one == ones && !(env[L] & 1) && !(env[R] & 1);
}

extern "C" __global__ void lazy_walk(
        const unsigned long long seed, const long long first, const long long n,
        const long long k, const int sweeps, const double p, const float alpha,
        const float q, const int skip_read, const int cap, const int growth,
        const int debug, long long* out) {
    extern __shared__ unsigned char env[];     // 0 = U, 1 = hole, 3 = particle
    if (blockIdx.x >= n) return;
    const unsigned lane = threadIdx.x;
    const unsigned sample = (unsigned)(first + blockIdx.x);
    const uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
    const int c = cap / 2;
    const bool bits = q == 0.5f;
    const double lo = fmin(p, 1.0 - p), hi = fmax(p, 1.0 - p);

    for (int j = (int)lane; j < cap / 4; j += 32) ((unsigned*)env)[j] = 0u;
    __syncwarp();

    long long x = 0, reveals = 0, ones = 0, reads = 0;
    int L = c, R = c, err = 0;
    unsigned s = 0;                             // sweeps so far; k * sweeps < 2**32
    for (long long t = 0; t < k && !err; ++t) {
        const uint4 r = philox(make_uint4((unsigned)t, (unsigned)(t >> 32), sample, TAG_STEP), key);
        const double u = ((r.x >> 5) * 67108864.0 + (r.y >> 6)) * (1.0 / 9007199254740992.0);
        bool left;
        if (skip_read && u < lo) {
            left = true;
        } else if (skip_read && u >= hi) {
            left = false;
        } else {
            const long long a = x + c;
            if (a < 0 || a >= cap) { err = 1; break; }
            const int j = (int)a;
            while (j <= L) L -= growth;         // keep both ends U around the read
            while (j >= R) R += growth;
            if (L < 0 || R >= cap) { err = 1; break; }
            unsigned char v = env[j];
            if (!(v & 1)) {
                v = unit24(r.z) < alpha ? 3 : 1;
                ++reveals;
                ones += v >> 1;
                __syncwarp();                   // every lane has read env[j] as U
                if (lane == 0) env[j] = v;
            }
            ++reads;
            left = u < ((v >> 1) ? p : 1.0 - p);
        }
        x += left ? -1 : 1;
        __syncwarp();                           // the reveal lands before anything moves
        if (debug && !deque_ok(env, L, R, reveals, ones, lane)) { err = 2; break; }
        for (int i = 0; i < sweeps; ++i, ++s) {
            const uint4 pr = philox(make_uint4(s >> 7, 0u, sample, TAG_PARITY), key);
            const int b = (word(pr, (s >> 5) & 3) >> (s & 31)) & 1u;
            sweep(env, L, R, c, b, s, sample, key, bits, q, lane);
            __syncwarp();                       // this sweep lands before the next reads
            // A known value moved at most one site, so at worst onto an end.
            if (env[L] & 1) L -= growth;
            if (env[R] & 1) R += growth;
            if (L < 0 || R >= cap) { err = 1; break; }
            if (debug && !deque_ok(env, L, R, reveals, ones, lane)) { err = 2; break; }
        }
    }
    if (lane == 0) {
        long long* o = out + (long long)blockIdx.x * NSTAT;
        o[0] = x; o[1] = reveals; o[2] = ones; o[3] = reads;
        o[4] = R - L + 1; o[5] = L - c; o[6] = R - c; o[7] = err;
    }
}
"""


def _check(params: dict) -> dict:
    """Merge with defaults and reject anything outside its range."""
    q = dict(_DEFAULTS)
    unknown = set(params) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown rwre_gpu_lazy params {sorted(unknown)}; "
                         f"known: {sorted(_DEFAULTS)}")
    q.update(params)
    for name in ("p", "alpha", "swap_prob"):
        if not 0.0 <= q[name] <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {q[name]}")
    if int(q["env_sweeps"]) != q["env_sweeps"] or q["env_sweeps"] < 0:
        raise ValueError(f"env_sweeps must be a non-negative integer; got {q['env_sweeps']}")
    q["env_sweeps"] = int(q["env_sweeps"])
    if not isinstance(q["skip_read"], bool):
        raise ValueError(f"skip_read must be true or false; got {q['skip_read']!r}")
    # cap/2 must be a multiple of 128 so the swap blocks align in the buffer.
    if int(q["cap"]) != q["cap"] or q["cap"] < 256 or q["cap"] % 256 or q["cap"] > _MAX_SHARED:
        raise ValueError(f"cap must be a multiple of 256 in [256, {_MAX_SHARED}]; got {q['cap']}")
    if int(q["growth"]) != q["growth"] or not 1 <= q["growth"] <= q["cap"] // 4:
        raise ValueError(f"growth must be an integer in [1, cap/4]; got {q['growth']}")
    q["cap"], q["growth"] = int(q["cap"]), int(q["growth"])
    return q


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            "rwre_gpu_lazy needs cupy (pip install cupy-cuda12x, see requirements.txt). "
            'Without a GPU use MODELS["rwre"], the same walk on CPU.') from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(f"rwre_gpu_lazy found no usable CUDA device ({err}). "
                           'Use MODELS["rwre"], the same walk on CPU.') from err
    return cupy


_module = None


def _compiled(cp):
    """The compiled module, built once per process (NVRTC takes ~1 s)."""
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SOURCE)
    return _module


def philox(ctr: np.ndarray, key: np.ndarray) -> np.ndarray:
    """The kernel's own Philox4x32-10 on the device: uint32 (m, 4), (m, 2) -> (m, 4).

    Exposed so the known-answer test runs the exact device function the walk uses.
    """
    cp = _cupy()
    ctr = np.ascontiguousarray(ctr, dtype=np.uint32).reshape(-1, 4)
    key = np.ascontiguousarray(key, dtype=np.uint32).reshape(-1, 2)
    m = len(ctr)
    out = cp.empty((m, 4), dtype=cp.uint32)
    _compiled(cp).get_function("philox_kat")(
        ((m + 127) // 128,), (128,), (cp.asarray(ctr), cp.asarray(key), out, cp.int32(m)))
    return out.get()


def walk_stats(k: int, n: int = 1, params: dict | None = None,
               rng: np.random.Generator | None = None, *, debug: bool = False) -> dict:
    """n i.i.d. samples of the walk, with the deque's bookkeeping. dict of int64 (n,).

    Keys are `STATS`: the signed displacement x = X_k; reveals (values drawn),
    ones (of them, particles), reads (steps that read the site); the final
    deque width = R - L + 1 and its ends in absolute coordinates; error (0).
    With debug=True every read and sweep also asserts the deque invariants
    (plan §6, L3) -- an O(width) scan each time, for tests only. Raises on an
    overflow of `cap` or a failed assertion, naming the first sample.
    """
    q = _check(dict(params or {}))
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if not 0 <= n < 2 ** 32:
        raise ValueError(f"n must be in [0, 2**32); got {n}")
    if k * q["env_sweeps"] >= 2 ** 32:
        raise ValueError(f"k * env_sweeps must be < 2**32 (the sweep counter); got {k} * "
                         f"{q['env_sweeps']}")

    cp = _cupy()
    kernel = _compiled(cp).get_function("lazy_walk")
    if q["cap"] > _DEFAULT_SHARED:
        optin = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        if q["cap"] > optin:
            raise RuntimeError(f"cap = {q['cap']} bytes of shared memory per sample; this "
                               f"device allows {optin}")
        kernel.max_dynamic_shared_size_bytes = q["cap"]
    rng = rng if rng is not None else np.random.default_rng()
    seed = int(rng.integers(0, 2 ** 63))

    out = np.empty((n, len(STATS)), dtype=np.int64)
    dev = cp.empty((min(n, _CHUNK), len(STATS)), dtype=cp.int64)
    for first in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - first)
        kernel((rows,), (32,),
               (cp.uint64(seed), cp.int64(first), cp.int64(rows), cp.int64(k),
                cp.int32(q["env_sweeps"]), cp.float64(q["p"]), cp.float32(q["alpha"]),
                cp.float32(q["swap_prob"]), cp.int32(q["skip_read"]), cp.int32(q["cap"]),
                cp.int32(q["growth"]), cp.int32(debug), dev),
               shared_mem=q["cap"])
        out[first:first + rows] = dev[:rows].get()

    err = out[:, STATS.index("error")]
    bad = np.flatnonzero(err)
    if bad.size:
        i = int(bad[0])
        what = {_OVERFLOW: f"needed more than cap = {q['cap']} cells",
                _DEBUG_FAILED: "broke a deque invariant (debug build)"}[int(err[i])]
        raise RuntimeError(f"sample {i} of {n} at k = {k} {what}; {bad.size} sample(s) affected")
    return {name: out[:, j] for j, name in enumerate(STATS)}


def walk(k: int, n: int = 1, params: dict | None = None,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED displacement X_k, on the GPU. int64, shape (n,).

    Annealed, as models/rwre.py's `walk`: every sample its own environment.
    """
    return walk_stats(k, n, params, rng)["x"]


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """n i.i.d. samples of Y_i = |X_i| at scale i (the scale is the step count k)."""
    return np.abs(walk(i, n=n, params=params, rng=rng))
