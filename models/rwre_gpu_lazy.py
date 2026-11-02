"""models/rwre.py's walk on a lazily revealed environment, on a CUDA GPU.

Phases A and B of plans/rwre_gpu_lazy.md: the byte and bit-packed layouts
(`layout`, default "bit"), bit-identical to each other. Registered as
MODELS["rwre_gpu_lazy"] (phase C) with no `cost_hint` and no `max_steps`: the
work is k * W(k) with W the deque width, an exponent the clock fits rather
than one the model can declare (plan §9 Q2), and there is no window to bound
k -- a sample that outgrows `cap` raises. Acceptance criteria and measured
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
Per sample, the known cells live in a buffer of `cap` cells in shared memory,
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
set, one bit per bond, which is what the bit-packed layout uses;
otherwise it draws a 24-bit float32 uniform per bond, as rwre's float32
comparison does. The walker's uniform is a 53-bit double on [0, 1), the
revealed value a 24-bit float32 uniform against alpha, as on CPU.

Two layouts (plan §5), same draws, same output
--------------------------------------------------
"byte": one byte per cell, cap bytes of shared memory, any swap_prob. The
readable reference. "bit": two bit-planes (known, value), cap / 4 bytes; a
lane sweeps 128 cells per Philox draw with the masked XOR on both planes,
t = (x ^ (x >> 1)) & m, x ^= t | (t << 1), plus the bond across each word
boundary at odd parity. swap_prob = 1/2 only (plan §9 Q1). It is ~7.7x
faster at cap = 2**15, mostly from occupancy: a sample holds 8 KiB of shared
memory instead of 32 KiB.

One warp per sample (plan §5): every lane computes every per-step draw itself
(the same counter gives the same value, at no extra latency), so there is no
thread 0 and no barrier but `__syncwarp`.

The pooled kernel (`kernel` "pool", user, 2026-11-02)
----------------------------------------------------
A sample's deque only grows, so in the warp kernel every sample reserves the
buffer its widest moment needs, and samples launched together peak together.
In `lazy_walk_pool` the `warps` warps of a block share a pool of 512-byte
pages (2048 cells in both bit-planes), mapped per sample through a page table
that covers the same [-cap/2, cap/2) as the warp kernel's buffer (so the same
samples overflow); warp w starts its first sample once warp 0 is
`stagger` * w / warps of the way through its own, and each warp then claims the
next sample as it finishes one, so the warps' ages stay spread. A warp that
finds no free page waits; if every holder waits, the block aborts and its
unfinished samples are rerun with half the warps. Bits, blocks and Philox
counters are BitEnv's, so the output is bit-identical to `layout` "bit".

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
    # Performance only: none changes a draw (module docstring). cap is in
    # cells: 2**15 is 8 KiB of shared memory in the bit layout, 32 KiB (3
    # samples per SM on the RTX 5090) in the byte layout. Kept at 2**15 for now
    # (Igor, 2026-10-02); L4 extrapolates ~1.3e4 cells per side at k = 2**20.
    "cap": 1 << 15,
    "growth": 32,
    "layout": "bit",
    # kernel "pool" (user, 2026-11-02): `warps` samples per block share `pool`
    # bytes of pages (0: as many as the block may have), first samples delayed
    # by stagger * w / warps of a sample (`lazy_walk_pool`). Same draws.
    "kernel": "warp",
    "warps": 32,
    "pool": 0,
    "stagger": 1.0,
}

#: Per-sample output columns of `walk_stats`, in order.
STATS = ("x", "reveals", "ones", "reads", "width", "left", "right", "error")

#: error codes in the "error" column. DEBUG is only ever set by a debug build;
#: POOL (a block's pool ran dry with every warp waiting) is rerun, never returned.
_OK, _OVERFLOW, _DEBUG_FAILED, _POOL = 0, 1, 2, 3

#: The most a block gets without opting in, and the fixed opt-in ceiling (fits
#: the RTX 5090's 99 KiB), as in models/rwre_gpu.py.
_DEFAULT_SHARED = 48 * 1024
_MAX_SHARED = 96 * 1024

#: Shared memory per cell: one byte, or one bit in each of two planes.
_BYTES_PER_CELL = {"byte": 1.0, "bit": 0.25}

#: Samples per launch. Only memory and responsiveness depend on it.
_CHUNK = 1 << 16

#: The pooled kernel's page (cells), its bytes (two planes), the block's
#: bookkeeping (sizeof PoolMeta) and the most pages a free map of 8 words holds.
_PAGE_CELLS = 2048
_PAGE_BYTES = _PAGE_CELLS // 4
_POOL_META = 80
_MAX_PAGES = 256

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

// ------------------------------------------------------------ byte layout
// One byte per cell: 0 = U, 1 = hole, 3 = particle. The reference for BitEnv.
struct ByteEnv {
    unsigned char* e;

    __device__ void clear(const int cap, const unsigned lane) const {
        for (int j = (int)lane; j < cap / 4; j += 32) ((unsigned*)e)[j] = 0u;
    }
    __device__ unsigned known(const int j) const { return e[j] & 1u; }
    __device__ unsigned one(const int j) const { return e[j] >> 1; }
    __device__ void reveal(const int j, const unsigned one) const { e[j] = one ? 3 : 1; }

    // Swap cells j, j+1 if the bond lies in the deque. Bonds outside [L, R]
    // join two U cells, a no-op.
    __device__ __forceinline__ void swap_if(const int j, const bool on, const int L,
                                            const int R) const {
        if (on && j >= L && j < R) {
            const unsigned char a = e[j];
            e[j] = e[j + 1];
            e[j + 1] = a;
        }
    }

    // One brick-wall sweep over the bonds of absolute parity b inside [L, R].
    // Buffer index j is absolute j - c, and c is a multiple of 128, so the
    // 128-cell blocks are aligned in both. Lane l takes bonds at offsets 2l + b
    // and 2l + b + 64 of each block: the 64 bonds of a block are disjoint, and
    // so are those of different blocks (a bond crossing into the next block has
    // odd offset 127 and meets offset 0 of the next only at the other parity).
    __device__ void sweep(const int L, const int R, const int c, const int b,
                          const unsigned s, const unsigned sample, const uint2 key,
                          const float q, const unsigned lane) const {
        const int B0 = L >> 7, B1 = (R - 1) >> 7, o = 2 * (int)lane + b;
        if (q == 0.5f) {
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
                    swap_if(j + o, ((o < 32 ? w0 : w1) >> (o & 31)) & 1u, L, R);
                    swap_if(j + o + 64, ((o < 32 ? w2 : w3) >> (o & 31)) & 1u, L, R);
                }
            }
        } else {
            for (int B = B0; B <= B1; ++B) {
                const unsigned w = (unsigned)(B - (c >> 7)), j = (unsigned)(B << 7);
                for (int h = 0; h < 2; ++h) {
                    const int bond = (int)lane + 32 * h;          // bond index within the block
                    const uint4 r = philox(make_uint4(w, s, sample, TAG_SWAPQ | ((unsigned)(bond >> 2) << 8)), key);
                    swap_if((int)j + 2 * bond + b, unit24(word(r, bond & 3)) < q, L, R);
                }
            }
        }
    }

    // Known cells and ones in [L, R]; cells outside are never touched.
    __device__ void count(const int L, const int R, const unsigned lane, int& known,
                          int& one) const {
        known = one = 0;
        for (int j = L + (int)lane; j <= R; j += 32) {
            known += e[j] & 1;
            one += e[j] >> 1;
        }
    }
};

// ------------------------------------------------------------ bit layout
// Two bit-planes of cap / 32 words, K (known) and V (value), cell j at bit
// j mod 32 of word j / 32; U is K = V = 0. Bit t of word 4B + i is the bond
// with left endpoint at cell 128B + 32i + t, which is bit 32i + t of block B's
// Philox draw: exactly the bit ByteEnv reads for that bond, so the two layouts
// make the same swaps (L0).
struct BitEnv {
    unsigned* K;
    unsigned* V;

    __device__ void clear(const int cap, const unsigned lane) const {
        for (int j = (int)lane; j < cap / 16; j += 32) K[j] = 0u;    // both planes
    }
    __device__ unsigned known(const int j) const { return (K[j >> 5] >> (j & 31)) & 1u; }
    __device__ unsigned one(const int j) const { return (V[j >> 5] >> (j & 31)) & 1u; }
    __device__ void reveal(const int j, const unsigned one) const {
        K[j >> 5] |= 1u << (j & 31);
        V[j >> 5] |= one << (j & 31);
    }

    // Swap bits t, t+1 for every left endpoint t set in m.
    static __device__ __forceinline__ void swap_in(unsigned& x, const unsigned m) {
        const unsigned t = (x ^ (x >> 1)) & m;
        x ^= t | (t << 1);
    }
    // Swap bit 31 of x with bit 0 of y if bit 31 of m is set.
    static __device__ __forceinline__ void swap_across(unsigned& x, unsigned& y, const unsigned m) {
        const unsigned d = ((x >> 31) ^ y) & (m >> 31) & 1u;
        x ^= d << 31;
        y ^= d;
    }
    static __device__ __forceinline__ void swap_block(uint4& x, const uint4 m, const int b) {
        // Parity 0: pairs (2i, 2i+1), all inside a word. Parity 1: pairs
        // (2i+1, 2i+2) inside a word for bits 1..30, and bit 31 with bit 0 of
        // the next word. Neither touches the bits the other one moves.
        const unsigned in = b ? 0x2AAAAAAAu : 0x55555555u;
        swap_in(x.x, m.x & in); swap_in(x.y, m.y & in);
        swap_in(x.z, m.z & in); swap_in(x.w, m.w & in);
        if (b) {
            swap_across(x.x, x.y, m.x);
            swap_across(x.y, x.z, m.y);
            swap_across(x.z, x.w, m.z);
        }
    }

    // One sweep over the blocks B0 .. B1 covering [L, R], lane l on block
    // base + l, 128 cells and one Philox draw per lane. Swapping whole blocks
    // also swaps bonds outside [L, R]: those join two U cells (both ends of the
    // deque are U, and everything beyond), a no-op, as in ByteEnv.
    __device__ void sweep(const int L, const int R, const int c, const int b,
                          const unsigned s, const unsigned sample, const uint2 key,
                          const float q, const unsigned lane) const {
        uint4* K4 = (uint4*)K;
        uint4* V4 = (uint4*)V;
        const int B0 = L >> 7, B1 = R >> 7;
        // Bit 31 of the previous chunk's last word, before this sweep, and its
        // mask (0 for the first chunk: the bond into block B0 joins two U cells).
        unsigned pk = 0u, pv = 0u, pm = 0u;
        for (int base = B0; base <= B1; base += 32) {
            const int B = base + (int)lane;
            const bool on = B <= B1;
            uint4 k = make_uint4(0u, 0u, 0u, 0u), v = k, m = k;
            if (on) {
                k = K4[B];
                v = V4[B];
                m = philox(make_uint4((unsigned)(B - (c >> 7)), s, sample, TAG_SWAP), key);
            }
            unsigned prevk = __shfl_up_sync(FULL, k.w, 1), prevv = __shfl_up_sync(FULL, v.w, 1);
            unsigned prevm = __shfl_up_sync(FULL, m.w, 1);
            unsigned nextk = __shfl_down_sync(FULL, k.x, 1), nextv = __shfl_down_sync(FULL, v.x, 1);
            if (lane == 0) { prevk = pk; prevv = pv; prevm = pm; }
            if (lane == 31) {
                // The next chunk's first block, not yet swapped; past B1 the
                // bond joins two U cells.
                nextk = B < B1 ? K[4 * (B + 1)] : 0u;
                nextv = B < B1 ? V[4 * (B + 1)] : 0u;
            }
            pk = __shfl_sync(FULL, k.w, 31);
            pv = __shfl_sync(FULL, v.w, 31);
            pm = __shfl_sync(FULL, m.w, 31);
            if (b) {
                // The bonds into and out of the block, from the old neighbours.
                // The neighbour lane makes the same decision on the same bits.
                swap_across(prevk, k.x, prevm);
                swap_across(prevv, v.x, prevm);
                swap_across(k.w, nextk, m.w);
                swap_across(v.w, nextv, m.w);
            }
            swap_block(k, m, b);
            swap_block(v, m, b);
            if (on) { K4[B] = k; V4[B] = v; }
            __syncwarp();
        }
    }

    // Known cells and ones in every block a sweep touches, so this also checks
    // that the no-op swaps outside [L, R] left only U there.
    __device__ void count(const int L, const int R, const unsigned lane, int& known,
                          int& one) const {
        known = one = 0;
        for (int w = 4 * (L >> 7) + (int)lane; w < 4 * ((R >> 7) + 1); w += 32) {
            known += __popc(K[w]);
            one += __popc(V[w]);
        }
    }
};

// ------------------------------------------------------------ the walk

// Debug build only: known cells must number exactly the reveals, and the ones
// exactly the revealed ones (swaps only move values), and both end cells must
// be U.
template <class Env>
__device__ bool deque_ok(const Env& env, const int L, const int R, const long long reveals,
                         const long long ones, const unsigned lane) {
    int known, one;
    env.count(L, R, lane, known, one);
    known = __reduce_add_sync(FULL, known);
    one = __reduce_add_sync(FULL, one);
    return known == reveals && one == ones && !env.known(L) && !env.known(R);
}

template <class Env>
__device__ void lazy_walk(
        const Env env, const unsigned long long seed, const long long first,
        const long long k, const int sweeps, const double p, const float alpha,
        const float q, const int skip_read, const int cap, const int growth,
        const int debug, long long* out) {
    const unsigned lane = threadIdx.x;
    const unsigned sample = (unsigned)(first + blockIdx.x);
    const uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
    const int c = cap / 2;
    const double lo = fmin(p, 1.0 - p), hi = fmax(p, 1.0 - p);

    env.clear(cap, lane);
    __syncwarp();

    long long x = 0, reveals = 0, ones = 0, reads = 0;
    int L = c, R = c, err = 0;
    unsigned s = 0;                             // sweeps so far; k * sweeps < 2**32
    uint4 pr;                                   // parities of sweeps 128(s/128) ..
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
            unsigned one;
            if (env.known(j)) {
                one = env.one(j);
            } else {
                one = unit24(r.z) < alpha;
                ++reveals;
                ones += one;
                __syncwarp();                   // every lane has read cell j as U
                if (lane == 0) env.reveal(j, one);
            }
            ++reads;
            left = u < (one ? p : 1.0 - p);
        }
        x += left ? -1 : 1;
        __syncwarp();                           // the reveal lands before anything moves
        if (debug && !deque_ok(env, L, R, reveals, ones, lane)) { err = 2; break; }
        for (int i = 0; i < sweeps; ++i, ++s) {
            if (!(s & 127)) pr = philox(make_uint4(s >> 7, 0u, sample, TAG_PARITY), key);
            const int b = (word(pr, (s >> 5) & 3) >> (s & 31)) & 1u;
            env.sweep(L, R, c, b, s, sample, key, q, lane);
            __syncwarp();                       // this sweep lands before the next reads
            // A known value moved at most one site, so at worst onto an end.
            if (env.known(L)) L -= growth;
            if (env.known(R)) R += growth;
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

extern __shared__ uint4 smem[];

#define ARGS const unsigned long long seed, const long long first, const long long n, \
        const long long k, const int sweeps, const double p, const float alpha, \
        const float q, const int skip_read, const int cap, const int growth, \
        const int debug, long long* out
#define PASS seed, first, k, sweeps, p, alpha, q, skip_read, cap, growth, debug, out

// cap bytes of shared memory.
extern "C" __global__ void lazy_walk_byte(ARGS) {
    if (blockIdx.x >= n) return;
    lazy_walk(ByteEnv{(unsigned char*)smem}, PASS);
}

// cap / 4 bytes of shared memory: two planes of cap bits.
extern "C" __global__ void lazy_walk_bit(ARGS) {
    if (blockIdx.x >= n) return;
    lazy_walk(BitEnv{(unsigned*)smem, (unsigned*)smem + cap / 32}, PASS);
}

// ------------------------------------------------------------ pooled bit layout
// kernel = "pool" (user, 2026-11-02): the warps of a block share one pool of
// pages, and start their first samples staggered, so their widths (which only
// grow) do not peak together. Same cells, same bits, same Philox counters as
// BitEnv; only where a block's 128 cells live differs.
#define PAGE_SHIFT 11                    // 2048 cells = 16 blocks of 128 per page
#define PAGE_WORDS 128                   // 64 words of K, then 64 of V: 512 bytes
#define NO_PAGE 0xffu
#define POOL_ABORT 3

struct PoolMeta {
    long long lead;                      // steps warp 0 has made in its first sample
    long long sum, cnt;                  // pages in use, sampled by warp 0
    int used, peak, stalled, holders, abort, pad;
    unsigned free[8];                    // bit set = page free; at most 256 pages
};

// A warp's view: absolute cell j lives in logical page j >> PAGE_SHIFT, which
// tab maps to a pool page. tab covers cells [-cap/2, cap/2), the warp kernel's
// buffer, so both kernels overflow on exactly the same samples.
struct PoolEnv {
    unsigned* pool;
    unsigned char* tab;
    int half;                            // logical page lp is tab[lp + half]

    __device__ __forceinline__ unsigned* pg(const int lp) const {
        return pool + PAGE_WORDS * tab[lp + half];
    }
    __device__ __forceinline__ uint4* kb(const int B) const {     // K of block B; V is +16
        return (uint4*)pg(B >> 4) + (B & 15);
    }
    __device__ unsigned known(const int j) const {
        const unsigned o = j & 2047;
        return (pg(j >> PAGE_SHIFT)[o >> 5] >> (o & 31)) & 1u;
    }
    __device__ unsigned one(const int j) const {
        const unsigned o = j & 2047;
        return (pg(j >> PAGE_SHIFT)[64 + (o >> 5)] >> (o & 31)) & 1u;
    }
    __device__ void reveal(const int j, const unsigned one) const {
        const unsigned o = j & 2047;
        unsigned* q = pg(j >> PAGE_SHIFT);
        q[o >> 5] |= 1u << (o & 31);
        q[64 + (o >> 5)] |= one << (o & 31);
    }

    // BitEnv::sweep with absolute blocks: K4[B] -> kb(B), and the counter's
    // block is B itself (BitEnv's B - c/128).
    __device__ void sweep(const int L, const int R, const int b, const unsigned s,
                          const unsigned sample, const uint2 key, const unsigned lane) const {
        const int B0 = L >> 7, B1 = R >> 7;
        unsigned pk = 0u, pv = 0u, pm = 0u;
        for (int base = B0; base <= B1; base += 32) {
            const int B = base + (int)lane;
            const bool on = B <= B1;
            uint4 k = make_uint4(0u, 0u, 0u, 0u), v = k, m = k;
            uint4* kp = nullptr;
            if (on) {
                kp = kb(B);
                k = kp[0];
                v = kp[16];
                m = philox(make_uint4((unsigned)B, s, sample, TAG_SWAP), key);
            }
            unsigned prevk = __shfl_up_sync(FULL, k.w, 1), prevv = __shfl_up_sync(FULL, v.w, 1);
            unsigned prevm = __shfl_up_sync(FULL, m.w, 1);
            unsigned nextk = __shfl_down_sync(FULL, k.x, 1), nextv = __shfl_down_sync(FULL, v.x, 1);
            if (lane == 0) { prevk = pk; prevv = pv; prevm = pm; }
            if (lane == 31) {
                nextk = B < B1 ? kb(B + 1)[0].x : 0u;
                nextv = B < B1 ? kb(B + 1)[16].x : 0u;
            }
            pk = __shfl_sync(FULL, k.w, 31);
            pv = __shfl_sync(FULL, v.w, 31);
            pm = __shfl_sync(FULL, m.w, 31);
            if (b) {
                BitEnv::swap_across(prevk, k.x, prevm);
                BitEnv::swap_across(prevv, v.x, prevm);
                BitEnv::swap_across(k.w, nextk, m.w);
                BitEnv::swap_across(v.w, nextv, m.w);
            }
            BitEnv::swap_block(k, m, b);
            BitEnv::swap_block(v, m, b);
            if (on) { kp[0] = k; kp[16] = v; }
            __syncwarp();
        }
    }

    __device__ void count(const int L, const int R, const unsigned lane, int& known,
                          int& one) const {
        known = one = 0;
        const int B0 = L >> 7;
        for (int w = (int)lane; w < 4 * ((R >> 7) - B0 + 1); w += 32) {
            const unsigned* q = (const unsigned*)kb(B0 + w / 4);
            known += __popc(q[w & 3]);
            one += __popc(q[64 + (w & 3)]);
        }
    }
};

// A free page, or -1 once the block aborts. Lane 0 claims a bit of the free
// map; while none is free the warp waits. If every warp holding pages waits,
// none will ever free one: the block aborts and the host reruns its unfinished
// samples with fewer warps (the same draws: they are keyed by sample, not slot).
// A finishing warp frees its pages before it stops counting as a holder, so a
// wait that sees stalled == holders four times running is a real deadlock.
__device__ int pool_alloc(PoolMeta* m, const unsigned lane) {
    int pg = -1;
    if (lane == 0) {
        volatile PoolMeta* v = m;
        bool stalled = false;
        int strikes = 0;
        for (;;) {
            for (int w = 0; w < 8 && pg < 0; ++w) {
                unsigned f = v->free[w];
                while (f) {
                    const unsigned bit = f & (0u - f);
                    const unsigned old = atomicAnd(&m->free[w], ~bit);
                    if (old & bit) { pg = 32 * w + __ffs(bit) - 1; break; }
                    f = old & ~bit;
                }
            }
            if (pg >= 0 || v->abort) break;
            if (!stalled) { atomicAdd(&m->stalled, 1); stalled = true; }
            __threadfence_block();
            if (v->stalled == v->holders) {
                if (++strikes == 4) { atomicExch(&m->abort, 1); break; }
            } else {
                strikes = 0;
            }
            __nanosleep(1000);
        }
        if (stalled) atomicSub(&m->stalled, 1);
        if (pg >= 0) atomicMax(&m->peak, atomicAdd(&m->used, 1) + 1);
    }
    return __shfl_sync(FULL, pg, 0);
}

// Map every unmapped logical page of [L, R], each zeroed (U). 0 or POOL_ABORT.
__device__ int pool_map(PoolMeta* m, const PoolEnv& env, const int L, const int R,
                        int& PL, int& PR, const unsigned lane) {
    for (int lp = L >> PAGE_SHIFT; lp <= (R >> PAGE_SHIFT); ++lp) {
        if (env.tab[lp + env.half] != NO_PAGE) continue;
        const int pg = pool_alloc(m, lane);
        if (pg < 0) return POOL_ABORT;
        unsigned* q = env.pool + PAGE_WORDS * pg;
        for (int w = (int)lane; w < PAGE_WORDS; w += 32) q[w] = 0u;
        __syncwarp();
        if (lane == 0) env.tab[lp + env.half] = (unsigned char)pg;
        __syncwarp();
    }
    PL = L >> PAGE_SHIFT;
    PR = R >> PAGE_SHIFT;
    return 0;
}

// Return every page in the table (a map cut short by an abort leaves holes).
__device__ void pool_free(PoolMeta* m, const PoolEnv& env, const int T, const unsigned lane) {
    int freed = 0;
    for (int i = (int)lane; i < T; i += 32) {
        const unsigned pg = env.tab[i];
        if (pg != NO_PAGE) {
            atomicOr(&m->free[pg >> 5], 1u << (pg & 31));
            env.tab[i] = NO_PAGE;
            ++freed;
        }
    }
    freed = __reduce_add_sync(FULL, freed);
    if (lane == 0) atomicSub(&m->used, freed);
    __threadfence_block();
    __syncwarp();
}

// Persistent: each warp claims samples ids[i] from a global counter until none
// are left. Rows of out are indexed by sample; error -1 (set by the host) marks
// a row not reached. lazy_walk's step loop, in absolute coordinates: the same
// overflow tests (|x|, L, R against cap/2), growth and order of draws.
#define POOL_ARGS const unsigned long long seed, const unsigned* ids, const long long rows, \
        const long long k, const int sweeps, const double p, const float alpha, \
        const int skip_read, const int cap, const int growth, const int debug, \
        const float stagger, const int npages, unsigned long long* next, long long* out, \
        long long* bstat
#define POOL_PASS seed, ids, rows, k, sweeps, p, alpha, skip_read, cap, growth, debug, \
        stagger, npages, next, out, bstat

__device__ __forceinline__ void pool_walk(POOL_ARGS) {
    PoolMeta* m = (PoolMeta*)smem;
    volatile PoolMeta* vm = m;
    unsigned* pool = (unsigned*)(m + 1);
    const int T = cap >> PAGE_SHIFT;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5, W = blockDim.x >> 5;
    unsigned char* tabs = (unsigned char*)(pool + PAGE_WORDS * npages);
    if (threadIdx.x == 0) {
        m->lead = m->sum = m->cnt = 0;
        m->used = m->peak = m->stalled = m->holders = m->abort = 0;
        for (int w = 0; w < 8; ++w) {
            const int left = npages - 32 * w;
            m->free[w] = left >= 32 ? FULL : left > 0 ? (1u << left) - 1u : 0u;
        }
    }
    for (int i = (int)threadIdx.x; i < (int)W * T; i += (int)blockDim.x) tabs[i] = NO_PAGE;
    __syncthreads();

    const PoolEnv env{pool, tabs + warp * T, T / 2};
    const uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
    const int c = cap / 2;
    const double lo = fmin(p, 1.0 - p), hi = fmax(p, 1.0 - p);

    // The delay: warp w starts once warp 0 is stagger * w / W of the way
    // through its first sample, so the first wave's ages are spread over a
    // sample's length, and later waves keep the spread.
    if (lane == 0 && warp > 0) {
        const long long wait = (long long)((double)stagger * warp / W * (double)k);
        while (vm->lead < wait && !vm->abort) __nanosleep(4096);
    }
    __syncwarp();

    bool leader = warp == 0;
    for (;;) {
        unsigned long long i = (unsigned long long)rows;
        if (lane == 0 && !vm->abort) i = atomicAdd(next, 1ull);
        i = __shfl_sync(FULL, i, 0);
        if (i >= (unsigned long long)rows) break;
        const unsigned sample = ids[i];
        if (lane == 0) atomicAdd(&m->holders, 1);

        long long x = 0, reveals = 0, ones = 0, reads = 0;
        int L = 0, R = 0, PL = 0, PR = -1;
        int err = pool_map(m, env, L, R, PL, PR, lane);
        unsigned s = 0;
        uint4 pr;
        for (long long t = 0; t < k && !err; ++t) {
            if (!(t & 63)) {
                if (__shfl_sync(FULL, lane == 0 ? vm->abort : 0, 0)) { err = POOL_ABORT; break; }
                if (lane == 0 && warp == 0 && !(t & 1023)) {
                    if (leader) vm->lead = t;
                    m->sum += vm->used;
                    ++m->cnt;
                }
            }
            const uint4 r = philox(make_uint4((unsigned)t, (unsigned)(t >> 32), sample, TAG_STEP), key);
            const double u = ((r.x >> 5) * 67108864.0 + (r.y >> 6)) * (1.0 / 9007199254740992.0);
            bool left;
            if (skip_read && u < lo) {
                left = true;
            } else if (skip_read && u >= hi) {
                left = false;
            } else {
                if (x < -c || x >= c) { err = 1; break; }
                const int j = (int)x;
                while (j <= L) L -= growth;
                while (j >= R) R += growth;
                if (L < -c || R >= c) { err = 1; break; }
                if ((L >> PAGE_SHIFT) < PL || (R >> PAGE_SHIFT) > PR) {
                    err = pool_map(m, env, L, R, PL, PR, lane);
                    if (err) break;
                }
                unsigned one;
                if (env.known(j)) {
                    one = env.one(j);
                } else {
                    one = unit24(r.z) < alpha;
                    ++reveals;
                    ones += one;
                    __syncwarp();
                    if (lane == 0) env.reveal(j, one);
                }
                ++reads;
                left = u < (one ? p : 1.0 - p);
            }
            x += left ? -1 : 1;
            __syncwarp();
            if (debug && !deque_ok(env, L, R, reveals, ones, lane)) { err = 2; break; }
            for (int i = 0; i < sweeps; ++i, ++s) {
                if (!(s & 127)) pr = philox(make_uint4(s >> 7, 0u, sample, TAG_PARITY), key);
                const int b = (word(pr, (s >> 5) & 3) >> (s & 31)) & 1u;
                env.sweep(L, R, b, s, sample, key, lane);
                __syncwarp();
                if (env.known(L)) L -= growth;
                if (env.known(R)) R += growth;
                if (L < -c || R >= c) { err = 1; break; }
                if ((L >> PAGE_SHIFT) < PL || (R >> PAGE_SHIFT) > PR) {
                    err = pool_map(m, env, L, R, PL, PR, lane);
                    if (err) break;
                }
                if (debug && !deque_ok(env, L, R, reveals, ones, lane)) { err = 2; break; }
            }
        }
        pool_free(m, env, T, lane);
        if (lane == 0) {
            atomicSub(&m->holders, 1);
            if (leader) vm->lead = 0x7fffffffffffffffLL;
            long long* o = out + (long long)sample * NSTAT;
            o[0] = x; o[1] = reveals; o[2] = ones; o[3] = reads;
            o[4] = R - L + 1; o[5] = L; o[6] = R; o[7] = err;
        }
        leader = false;
    }
    if (lane == 0 && warp == 0) vm->lead = 0x7fffffffffffffffLL;
    __syncthreads();
    if (threadIdx.x == 0) {
        long long* b = bstat + 4 * (long long)blockIdx.x;
        b[0] = m->peak; b[1] = m->sum; b[2] = m->cnt; b[3] = m->abort;
    }
}

// Two register budgets for one body (80 registers unbounded allow only 24
// warps): up to 32 warps in one block per SM (64 registers), or up to 16
// warps in each of three (40 registers, 48 warps per SM, a small spill).
extern "C" __global__ void __launch_bounds__(1024, 1) lazy_walk_pool(POOL_ARGS) {
    pool_walk(POOL_PASS);
}
extern "C" __global__ void __launch_bounds__(512, 3) lazy_walk_pool3(POOL_ARGS) {
    pool_walk(POOL_PASS);
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
    if q["layout"] not in _BYTES_PER_CELL:
        raise ValueError(f"layout must be one of {sorted(_BYTES_PER_CELL)}; got {q['layout']!r}")
    # Plan §9 Q1: the bit layout draws one bit per bond, so swap_prob = 1/2 only.
    if q["layout"] == "bit" and q["swap_prob"] != 0.5:
        raise ValueError(f"layout 'bit' needs swap_prob = 0.5; got {q['swap_prob']}. "
                         "Use layout 'byte' for any other swap_prob.")
    # cap/2 must be a multiple of 128 so the swap blocks align in the buffer.
    top = (int(_MAX_SHARED / _BYTES_PER_CELL[q["layout"]]) if q.get("kernel") != "pool"
           else 1 << 21)
    if int(q["cap"]) != q["cap"] or q["cap"] < 256 or q["cap"] % 256 or q["cap"] > top:
        raise ValueError(f"cap must be a multiple of 256 in [256, {top}] for layout "
                         f"{q['layout']!r}; got {q['cap']}")
    if int(q["growth"]) != q["growth"] or not 1 <= q["growth"] <= q["cap"] // 4:
        raise ValueError(f"growth must be an integer in [1, cap/4]; got {q['growth']}")
    q["cap"], q["growth"] = int(q["cap"]), int(q["growth"])
    if q["kernel"] not in ("warp", "pool"):
        raise ValueError(f"kernel must be 'warp' or 'pool'; got {q['kernel']!r}")
    if q["kernel"] == "pool":
        if q["layout"] != "bit":
            raise ValueError("kernel 'pool' has the bit layout only")
        # The page table covers [-cap/2, cap/2) in whole pages, an even number.
        if q["cap"] % (2 * _PAGE_CELLS):
            raise ValueError(f"kernel 'pool' needs cap a multiple of {2 * _PAGE_CELLS}; "
                             f"got {q['cap']}")
        if int(q["warps"]) != q["warps"] or not 1 <= q["warps"] <= 32:
            raise ValueError(f"warps must be an integer in [1, 32]; got {q['warps']}")
        if not 0.0 <= q["stagger"] <= 1.0:
            raise ValueError(f"stagger must be in [0, 1]; got {q['stagger']}")
        if int(q["pool"]) != q["pool"] or q["pool"] < 0:
            raise ValueError(f"pool must be a non-negative integer (bytes); got {q['pool']}")
        q["warps"], q["pool"] = int(q["warps"]), int(q["pool"])
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


def _walk_pool(cp, q: dict, k: int, n: int, seed: int, debug: bool,
               report: list | None) -> np.ndarray:
    """All n samples through `lazy_walk_pool`: (n, len(STATS)) int64.

    A block whose pool runs dry with every warp waiting aborts; its unfinished
    samples are rerun with half the warps, the same seed and sample indices, so
    the same draws. One warp that still runs dry needs more than the pool alone.
    Each launch appends its memory figures to `report`.
    """
    attr = cp.cuda.Device().attributes
    optin, per_sm = attr["MaxSharedMemoryPerBlockOptin"], attr["MaxSharedMemoryPerMultiprocessor"]
    T = q["cap"] // _PAGE_CELLS
    out = cp.full((n, len(STATS)), -1, dtype=cp.int64)
    todo = np.arange(n, dtype=np.uint32)
    warps = q["warps"]
    while todo.size:
        blocks = 3 if warps <= 16 else 1            # the variant's launch bounds
        kernel = _compiled(cp).get_function("lazy_walk_pool3" if blocks == 3 else "lazy_walk_pool")
        tables = warps * T
        room = (min(optin, per_sm // blocks - 1024) - _POOL_META - tables) // _PAGE_BYTES
        npages = min(_MAX_PAGES, room, q["pool"] // _PAGE_BYTES if q["pool"] else room)
        if npages < 1:
            raise RuntimeError(f"no room for a page: {warps} page tables of {T} entries")
        shared = _POOL_META + npages * _PAGE_BYTES + tables
        kernel.max_dynamic_shared_size_bytes = shared
        # Blocks per SM: the shared memory (1 KiB reserved per block), 48 warps,
        # and the kernel's register budget.
        per = max(1, min(attr["MaxThreadsPerMultiProcessor"] // (32 * warps),
                         per_sm // (shared + 1024),
                         attr["MaxRegistersPerMultiprocessor"] // (kernel.num_regs * 32 * warps)))
        grid = attr["MultiProcessorCount"] * per
        bstat = cp.zeros((grid, 4), dtype=cp.int64)
        kernel((grid,), (32 * warps,),
               (cp.uint64(seed), cp.asarray(todo), cp.int64(todo.size), cp.int64(k),
                cp.int32(q["env_sweeps"]), cp.float64(q["p"]), cp.float32(q["alpha"]),
                cp.int32(q["skip_read"]), cp.int32(q["cap"]), cp.int32(q["growth"]),
                cp.int32(debug), cp.float32(q["stagger"]), cp.int32(npages),
                cp.zeros(1, dtype=cp.uint64), out, bstat),
               shared_mem=shared)
        err = out[:, STATS.index("error")].get()[todo]
        rerun = todo[(err == _POOL) | (err == -1)]
        b = bstat.get()
        if report is not None:
            report.append({"warps": warps, "pages": int(npages), "page_bytes": _PAGE_BYTES,
                           "shared": int(shared), "blocks_per_sm": int(per), "samples": int(todo.size),
                           "peak_pages": b[:, 0].tolist(),
                           "mean_pages": (b[:, 1] / np.maximum(b[:, 2], 1)).tolist(),
                           "aborted_blocks": int(b[:, 3].sum()), "rerun": int(rerun.size)})
        if rerun.size and warps == 1:
            raise RuntimeError(f"sample {int(rerun[0])} at k = {k} needs more than a pool of "
                               f"{npages} pages ({npages * _PAGE_CELLS} cells)")
        todo, warps = rerun, max(1, warps // 2)
    return out.get()


def walk_stats(k: int, n: int = 1, params: dict | None = None,
               rng: np.random.Generator | None = None, *, debug: bool = False,
               report: list | None = None) -> dict:
    """n i.i.d. samples of the walk, with the deque's bookkeeping. dict of int64 (n,).

    Keys are `STATS`: the signed displacement x = X_k; reveals (values drawn),
    ones (of them, particles), reads (steps that read the site); the final
    deque width = R - L + 1 and its ends in absolute coordinates; error (0).
    With debug=True every read and sweep also asserts the deque invariants
    (plan §6, L3) -- an O(width) scan each time, for tests only. Raises on an
    overflow of `cap` or a failed assertion, naming the first sample. With
    kernel "pool", each launch appends its memory figures to `report` if given.
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
    rng = rng if rng is not None else np.random.default_rng()
    if q["kernel"] == "pool":
        out = _walk_pool(cp, q, k, n, int(rng.integers(0, 2 ** 63)), debug, report)
        return _checked(out, q, k, n)
    kernel = _compiled(cp).get_function(f"lazy_walk_{q['layout']}")
    shared = int(q["cap"] * _BYTES_PER_CELL[q["layout"]])
    if shared > _DEFAULT_SHARED:
        optin = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        if shared > optin:
            raise RuntimeError(f"cap = {q['cap']} cells is {shared} bytes of shared memory "
                               f"per sample; this device allows {optin}")
        kernel.max_dynamic_shared_size_bytes = shared
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
               shared_mem=shared)
        out[first:first + rows] = dev[:rows].get()
    return _checked(out, q, k, n)


def _checked(out: np.ndarray, q: dict, k: int, n: int) -> dict:
    """walk_stats' columns, or the error of the first sample that failed."""
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
