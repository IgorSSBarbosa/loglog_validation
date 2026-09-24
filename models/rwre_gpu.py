"""models/rwre.py on a CUDA GPU -- registered as MODELS["rwre_gpu"] in
tools/models.py.

Same walk, same environment, same observable Y_k = |X_k|, same parameters and
defaults (p, alpha, env_sweeps, swap_prob, window_c), same `cost_hint`; a
different device and a different random stream. The model -- the walker on a
stirred SSEP, the four decisions (a)-(d) it rests on, why the parity is drawn
per sample, why there is no target_fn -- is explained in models/rwre.py and
experiments/02_rwre/README.md and is not repeated here. This file only says
what changes on a GPU, following the conventions of the other *_gpu models
(models/erw_gpu.py, models/percolation2d_gpu.py; user, 2026-09-16).

A separate model, not a backend flag
------------------------------------
The random stream differs, so a seed recorded against MODELS["rwre"] does not
reproduce a GPU run. With a separate name the recipe's "model" field records
the device, and no driver learns about hardware.

What is imported from the CPU sibling, and what is not
------------------------------------------------------
DECLARATIONS are imported: the parameter check `_check`, `window_width` and
`cost_hint`. So MODELS["rwre_gpu"].cost_hint IS MODELS["rwre"].cost_hint --
the declared d = 3/2 cannot drift, and neither can the window the walk runs
in. The ALGORITHM is a CUDA kernel, `_KERNEL` below.

Why one kernel launch per call, not CuPy ops (user, 2026-09-23)
---------------------------------------------------------------
The first version ported models/rwre.py's step loop to CuPy one to one, as
models/erw_gpu.py does. It sampled the right law (G1/G2 passed) but every
step is ~40 small kernels (read, jump, four sweeps of roll/where), so a call
cost ~0.55 ms * k in LAUNCHES whatever n was: 545 ms at k = 1024 for n = 4,
579 ms for n = 256 (experiments/13_rwre_gpu/README.md). That overhead grows
with k, so it is not the fixed per-call cost `ModelSpec.batched_cost` assumes,
and tools/cost_model.py's `probe_batched` -- which the pilot runs on the
pilot's own scales -- could not see one more sample under it and refused.

The kernel runs all k steps of a sample inside one CUDA block, the window in
shared memory, so a call is one launch per chunk and the only per-call cost
is fixed. Per step, exactly models/rwre.py's order (b):

  1. read: thread 0 reads eta(X) at the walker's site and jumps, left with
     probability p on a particle and 1 - p on a hole (a float64 uniform);
  2. stir, env_sweeps times: thread 0 draws the sweep's parity b for THIS
     sample (per sample, for rwre.py's reason (a)); every bond (j, j+1 mod W)
     with j = b mod 2 swaps its two sites with probability swap_prob. W is
     even, so the bonds of one parity are disjoint, and the threads swap them
     in place with no conflicts -- the same permutation rwre.py's `_stir`
     builds out of two rolls;

with a barrier between phases. The initial window is Bernoulli(alpha) per
site. The comparisons against alpha and swap_prob are float32 and the one
against p is float64, as on CPU; each uniform is 1 - curand_uniform, which is
on [0, 1) like numpy's (curand_uniform itself is on (0, 1]), so `u < q` has
probability q exactly at q = 0 and q = 1.

The random stream
-----------------
Each `simulate` call takes ONE integer from the driver's np.random.Generator,
`rng.integers(0, 2**63)`, as every *_gpu model does, and uses it as the seed of
cuRAND's Philox4x32-10 device generator. Thread t of sample j draws from
subsequence j * 256 + t, and how many threads a sample gets is a function of
its window width alone. So the output is a pure function of (seed, i, n,
params) on a fixed cupy/CUDA version (ground rule 5), and -- unlike
models/rwre.py and the other *_gpu models -- it does NOT depend on how the
samples are cut into launches: sample j is the same draw at any `_CHUNK`.
What is tested against MODELS["rwre"] is equality in DISTRIBUTION
(experiments/13_rwre_gpu/README.md, G1).

No GPU present
--------------
Importing this module never imports cupy. `walk` imports it lazily and, if
cupy is missing or no CUDA device is visible, raises RuntimeError naming
MODELS["rwre"] as the CPU model to use. It never falls back silently: a run
labelled rwre_gpu was drawn on a GPU.
"""

from __future__ import annotations

import math

import numpy as np

from models.rwre import _check, cost_hint, window_width

__all__ = ["walk", "simulate", "cost_hint", "threads_per_sample", "max_steps"]

#: The window lives in a block's shared memory, one byte per site, and 48 KiB
#: is what CUDA gives a block without an opt-in. At the default window_c = 12
#: that allows k up to 4096**2, about 1.7e7 steps, and 2**16 at
#: window_exponent = 3/4 (see `max_steps`).
_MAX_W = 48 * 1024

#: Threads per sample are at most this, and it is also the stride between
#: samples in Philox subsequences, so it may never change without changing
#: every draw.
_MAX_THREADS = 256

#: Samples per launch. Only memory and responsiveness depend on it: a
#: sample's draws do not (see the module docstring).
_CHUNK = 1 << 16

_KERNEL = r"""
#include <curand_kernel.h>

extern "C" __global__ void rwre_walk(
        const unsigned long long seed, const long long first, const long long n,
        const long long k, const int w, const int sweeps,
        const double p, const float alpha, const float swap_prob,
        long long* out) {
    extern __shared__ unsigned char env[];
    __shared__ int parity;
    const long long sample = first + blockIdx.x;
    if (blockIdx.x >= n) return;
    const int t = threadIdx.x, T = blockDim.x, bonds = w / 2, half = w / 2;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)sample * 256ULL + t, 0ULL, &st);

    for (int j = t; j < w; j += T)
        env[j] = (1.0f - curand_uniform(&st)) < alpha;
    __syncthreads();

    long long x = 0;
    for (long long step = 0; step < k; ++step) {
        if (t == 0) {
            long long site = (x + half) % w;
            if (site < 0) site += w;
            const double p_left = env[site] ? p : 1.0 - p;
            x += ((1.0 - curand_uniform_double(&st)) < p_left) ? -1 : 1;
        }
        __syncthreads();                    // the walker read before anything moves
        for (int s = 0; s < sweeps; ++s) {
            if (t == 0) parity = (1.0f - curand_uniform(&st)) < 0.5f;
            __syncthreads();
            const int b = parity;
            for (int m = t; m < bonds; m += T) {
                const int j = 2 * m + b, j1 = (j + 1 == w) ? 0 : j + 1;
                if ((1.0f - curand_uniform(&st)) < swap_prob) {
                    const unsigned char a = env[j];
                    env[j] = env[j1];
                    env[j1] = a;
                }
            }
            __syncthreads();                // this sweep lands before the next reads
        }
    }
    if (t == 0) out[blockIdx.x] = x;
}
"""


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["rwre_gpu"] needs cupy (pip install cupy-cuda12x, see '
            'requirements.txt). Without a GPU use MODELS["rwre"], the same '
            "walk on CPU.") from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["rwre_gpu"] found no usable CUDA device ({err}). '
            'Use MODELS["rwre"], the same walk on CPU.') from err
    return cupy


_compiled = None


def _kernel(cp):
    """The compiled kernel, built once per process (NVRTC takes ~1 s)."""
    global _compiled
    if _compiled is None:
        _compiled = cp.RawKernel(_KERNEL, "rwre_walk")
    return _compiled


def threads_per_sample(w: int) -> int:
    """CUDA threads for one sample of window width w: one per bond of a parity,
    rounded up to a warp and capped at _MAX_THREADS. A pure function of w,
    because it decides which thread's stream draws which site."""
    return min(_MAX_THREADS, 32 * math.ceil((w // 2) / 32))


def max_steps(params: dict | None = None) -> int:
    """The largest k whose window fits one block's shared memory, for these params.

    What autopilot's --max-scale should be for this model: `walk` raises above
    it rather than silently moving the window to slower global memory.
    """
    q = _check(dict(params or {}))
    lo, hi = 1, 2
    while window_width(hi, q["window_c"], q["window_exponent"]) <= _MAX_W:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:                  # window_width is non-decreasing in k
        mid = (lo + hi) // 2
        if window_width(mid, q["window_c"], q["window_exponent"]) <= _MAX_W:
            lo = mid
        else:
            hi = mid
    return lo


def walk(k: int, n: int = 1, params: dict | None = None,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED displacement X_k, on the GPU. int64, shape (n,).

    Arguments, validation and the annealed measure are models/rwre.py's
    `walk`. The sign is kept for the exact zero-check E X_k = 0.
    """
    q = _check(dict(params or {}))
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    w = window_width(k, q["window_c"], q["window_exponent"])
    if w > _MAX_W:
        raise ValueError(f"window width {w} at k = {k} exceeds the {_MAX_W} bytes of "
                         "shared memory one sample may use; use MODELS[\"rwre\"]")

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    seed = int(rng.integers(0, 2 ** 63))
    kernel = _kernel(cp)
    threads = threads_per_sample(w)

    out = np.empty(n, dtype=np.int64)
    dev = cp.empty(min(n, _CHUNK), dtype=cp.int64)
    for first in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - first)
        kernel((rows,), (threads,),
               (cp.uint64(seed), cp.int64(first), cp.int64(rows), cp.int64(k),
                cp.int32(w), cp.int32(q["env_sweeps"]), cp.float64(q["p"]),
                cp.float32(q["alpha"]), cp.float32(q["swap_prob"]), dev),
               shared_mem=w)
        out[first:first + rows] = dev[:rows].get()
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["rwre_gpu"].simulate: n i.i.d. samples of Y_i = |X_i| at scale i."""
    return np.abs(walk(i, n=n, params=params, rng=rng))
