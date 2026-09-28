"""The elephant random walk as an urn, on a CUDA GPU -- registered as
MODELS["erw_urn_gpu"] in tools/models.py.

Same walk, same observable Y_k = |S_k|, same parameter p, same `cost_hint` as
models/erw.py; a different sampler, a different device and a different random
stream. The model -- Bercu (2018) eq. (2.1), the fair first step, why p has
no default, why there is no target_fn -- is explained in models/erw.py and
not repeated here.

The sampler: the walk as a two-colour urn (user, 2026-09-28)
------------------------------------------------------------
To draw X_{m+1} the elephant picks one of its m past steps uniformly and
repeats it with probability p or reverses it with 1 - p. Which past step it
picks never matters, only its colour, so the whole past is summarised by
(m, U_m), U_m = #{t <= m : X_t = +1}, and one step is

    copied_up = (a uniform past step is a +1)    probability U_m / m
    flip      = (reverse it)                     probability 1 - p
    X_{m+1}   = +1  iff  copied_up != flip

which is the Markov chain of Bercu eq. (2.2), P(X_{m+1} = +1 | F_m) =
p U_m/m + (1 - p)(1 - U_m/m) = (1 + (2p - 1) S_m / m) / 2, drawn as the two
independent coins it is made of. S_k = 2 U_k - k.

models/erw.py measured this chain and declined it on CPU (2026-09-22): its
loop over the k steps vectorizes only across samples. On a GPU that is the
right shape: one thread per sample, its whole state two registers, no
trajectory in memory, all k steps in one launch. Against MODELS["erw_gpu"]
(the tree sampler in CuPy) it is ~240x faster per step at a full device --
2.35 against 560-590 ps -- and its memory is 8 bytes per SAMPLE instead of
~46 bytes per STEP (experiments/14_erw_urn_gpu/README.md).

Both coins in integers, not doubles
-----------------------------------
Each step takes one Philox4x32-10 draw, 128 bits, as two 64-bit words a, b:

    copied_up = umulhi(a, m) < U_m      umulhi(a, m) = floor(a m / 2^64) is
                                        uniform on {0..m-1} to within m/2^64
    flip      = b < F,  F = floor((1 - p) 2^64), computed exactly on the host

so P(flip) is 1 - p to 2^-64, and p = 1 (F = 0) never flips and p = 0
(`always_flip`, since 2^64 does not fit) always does -- the closed cases hold
by construction. No floating point runs on the device at all, which matters
on a consumer card: the RTX 5090 does float64 at 1/64 of its float32 rate,
and the float64 draft of this kernel (one curand_uniform_double against the
folded probability of eq. 2.2) ran at 23 ps per step, 10x slower.

The random stream
-----------------
Each `simulate` call takes ONE integer from the driver's np.random.Generator,
`rng.integers(0, 2**63)`, as every *_gpu model does, and uses it as the
Philox seed. Sample j draws from subsequence j, so -- as for
models/rwre_gpu.py -- sample j is the same draw however the call is cut into
launches, and the output is a pure function of (seed, i, n, p) on a fixed
cupy/CUDA version (ground rule 5). What is tested against MODELS["erw"] is
equality in DISTRIBUTION.

How many samples a call needs
-----------------------------
A sample is a serial chain of k steps on one thread, so a call only fills the
device with as many samples as it has resident threads (`resident_threads`:
multiprocessors x threads per multiprocessor, 170 x 1536 = 261,120 on the RTX
5090). Measured at k = 2^16 (2026-09-28): a call takes 6.8 ms for every n
from 2^8 to 2^15 -- one sample's latency -- and then grows with n, at 2.9 ps
per step at 2^16-2^17 and 2.35 from 2^19 on. So below ~2^16 samples the
missing ones are time the device spends idle, and from resident_threads()
up a call is within ~7% of the device's throughput. The registry sets
`latency_bound`, so the cost probe walks n up to a full device; keeping the
PLANNED n above it is the experiment's job, by capping the ladder
(experiments/14_erw_urn_gpu/sweep_p.py).

No GPU present
--------------
Importing this module never imports cupy. `walk` imports it lazily and, if
cupy is missing or no CUDA device is visible, raises RuntimeError naming
MODELS["erw"] as the CPU model to use. It never falls back silently.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from models.erw import _check, cost_hint

__all__ = ["walk", "simulate", "cost_hint", "resident_threads"]

#: Threads per CUDA block. Only occupancy depends on it: sample j's draws do
#: not (subsequence j, whatever block it lands in).
_THREADS = 256

#: Samples per launch. Only memory (8 bytes each) and how long one launch
#: runs depend on it, not the draws. Four times the RTX 5090's resident
#: threads, so a full launch keeps the device full with little tail.
_CHUNK = 1 << 20

_KERNEL = r"""
#include <curand_kernel.h>

extern "C" __global__ void erw_urn(
        const unsigned long long seed, const long long first, const long long n,
        const long long k, const unsigned long long flip_below,
        const int always_flip, long long* out) {
    const long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= n) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)(first + j), 0ULL, &st);

    uint4 r = curand4(&st);
    unsigned long long up = r.x & 1u;                  // X_1, a fair coin
    for (unsigned long long m = 1; m < (unsigned long long)k; ++m) {
        r = curand4(&st);
        const unsigned long long a = ((unsigned long long)r.x << 32) | r.y;
        const unsigned long long b = ((unsigned long long)r.z << 32) | r.w;
        const bool copied_up = __umul64hi(a, m) < up;  // P = up / m
        const bool flip = always_flip || b < flip_below;  // P = 1 - p
        up += copied_up != flip;
    }
    out[j] = 2 * (long long)up - k;
}
"""


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["erw_urn_gpu"] needs cupy (pip install cupy-cuda12x, see '
            'requirements.txt). Without a GPU use MODELS["erw"], the same '
            "walk on CPU.") from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["erw_urn_gpu"] found no usable CUDA device ({err}). '
            'Use MODELS["erw"], the same walk on CPU.') from err
    return cupy


_compiled = None


def _kernel(cp):
    """The compiled kernel, built once per process (NVRTC takes ~1 s)."""
    global _compiled
    if _compiled is None:
        _compiled = cp.RawKernel(_KERNEL, "erw_urn")
    return _compiled


def flip_threshold(p: float) -> tuple[int, bool]:
    """(F, always_flip) with P(flip) = F / 2^64 = 1 - p exactly to 2^-64.

    p is a float64, so Fraction(p) is exact and so is the floor.
    """
    f = int((1 - Fraction(p)) * 2 ** 64)
    return (0, True) if f == 2 ** 64 else (f, False)


def resident_threads() -> int:
    """How many samples fill the current device: one thread each, and this
    many threads can be resident at once. A call of fewer samples costs the
    same wall clock as a call of this many."""
    cp = _cupy()
    attrs = cp.cuda.Device().attributes
    return int(attrs["MultiProcessorCount"]) * int(attrs["MaxThreadsPerMultiProcessor"])


def walk(k: int, n: int, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED position S_k, on the GPU. int64, shape (n,).

    Arguments and validation are models/erw.py's `walk`, less `block_n`: a
    sample's draws do not depend on how the call is cut, so there is nothing
    to set. The sign is kept for the exact zero-check E S_k = 0.
    """
    p = _check({"p": p})
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")

    cp = _cupy()
    kernel = _kernel(cp)
    rng = rng if rng is not None else np.random.default_rng()
    seed = int(rng.integers(0, 2 ** 63))
    flip_below, always_flip = flip_threshold(p)

    out = np.empty(n, dtype=np.int64)
    dev = cp.empty(min(n, _CHUNK), dtype=cp.int64)
    for first in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - first)
        kernel(((rows + _THREADS - 1) // _THREADS,), (_THREADS,),
               (cp.uint64(seed), cp.int64(first), cp.int64(rows), cp.int64(k),
                cp.uint64(flip_below), cp.int32(always_flip), dev))
        out[first:first + rows] = dev[:rows].get()
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["erw_urn_gpu"].simulate: n i.i.d. samples of Y_i = |S_i| at scale i."""
    return np.abs(walk(i, n, _check(params), rng=rng))
