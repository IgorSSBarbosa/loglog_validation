"""models/erw.py on a CUDA GPU -- registered as MODELS["erw_gpu"] in
tools/models.py.

Same walk, same observable Y_k = |S_k|, same parameter p, same `cost_hint`,
a different device and a different random stream. The model -- Bercu (2018)
eq. (2.1), the fair first step, why p has no default, why there is no
target_fn -- and the sampler -- the parents and signs drawn up front as a
random recursive tree, resolved by pointer doubling -- are explained in
models/erw.py and not repeated here. This file only says what changes on a
GPU, following the conventions of the other *_gpu models
(models/percolation2d_gpu.py; user, 2026-09-16).

A separate model, not a backend flag
------------------------------------
The random stream differs, so a seed recorded against MODELS["erw"] does not
reproduce a GPU run. With a separate name the recipe's "model" field records
the device, and no driver learns about hardware.

What is imported from the CPU sibling, and what is not
------------------------------------------------------
DECLARATIONS are imported: the parameter check `_check` and `cost_hint`. So
MODELS["erw_gpu"].cost_hint IS MODELS["erw"].cost_hint -- the declared d = 1
cannot drift. The ALGORITHM is not imported: `_resolve` is a CuPy port of the
CPU function of the same name, kept one to one (user, 2026-09-23: "adapt the
same cpu model", CuPy ops rather than a RawKernel) so the two read side by
side. Each doubling pass is two `take_along_axis` gathers over the block,
exactly as on CPU; the rows of a block are resolved in parallel, and so are
the k nodes of each row.

Two dtypes differ from the CPU module, neither in the law (user, 2026-09-23):
  - the uniforms stay float64, for models/erw.py's reason (a parent index is
    floor(u * n)). cupy's Generator.random draws float64 on [0, 1) with the
    full 53 bits -- the same interval as numpy's, so `u < p` and the clip on
    floor(u * t) carry over unchanged. Checked 2026-09-23 (cupy 14.2.0):
    over 1e7 draws, u * 2**53 is an integer for 50% of them and u * 2**40
    for 0.013%, i.e. the grid is 2**-53, not the 2**-32 of a single cuRAND
    word;
  - the ancestor table is int32, not int64: half the device bytes, and k is
    capped below 2**31 - 1 so that the sentinel k fits.

The random stream, and the contract this model does not keep
------------------------------------------------------------
Each `simulate` call takes ONE integer from the driver's np.random.Generator,
`rng.integers(0, 2**63)`, and seeds CuPy's default cuRAND generator (XORWOW)
with it, as every *_gpu model does. The driver's SeedSequence spawning and
seed recording are untouched, so ground rule 5 holds: the output is a pure
function of (seed, i, n, p) on a fixed cupy/CUDA version (requirements.txt
pins cupy for that reason).

models/erw.py is bit-identical at any `block_n`. This model is not, for the
reason percolation2d_gpu is not: cuRAND fills a block's (rows, 2k - 1)
uniforms in one call, and that call is not the concatenation of smaller ones.
So `block_rows(k)` is a fixed function of k -- a CONSTANT byte budget over the
bytes one sample needs -- and never of free VRAM, and there is no `block_n`
argument. What holds instead is equality in DISTRIBUTION with MODELS["erw"],
tested in tools/tests/test_erw_gpu.py.

No GPU present
--------------
Importing this module never imports cupy. `walk` imports it lazily and, if
cupy is missing or no CUDA device is visible, raises RuntimeError naming
MODELS["erw"] as the CPU model to use. It never falls back silently: a run
labelled erw_gpu was drawn on a GPU.
"""

from __future__ import annotations

import numpy as np

from models.erw import _check, cost_hint

__all__ = ["walk", "simulate", "cost_hint", "block_rows"]

#: Device working-set budget for one block, in bytes. A FIXED constant, not a
#: fraction of free memory: block_rows decides how the cuRAND stream is cut, so
#: it must be a pure function of k (see the module docstring). 2 GiB, as the
#: other *_gpu models, leaves room on a shared 32 GiB card.
_GPU_WORKING_SET_BYTES = 2 * 1024 ** 3

#: Device bytes per step across one block: the (rows, 2k - 1) float64 draw
#: (16), its float64 product and int32 casts while the parents are built, the
#: int32 ancestor table and int8 signs, and each pass's gathers with the int64
#: index grids take_along_axis builds. The CuPy pool peaked at 45.8-46.1 over
#: k = 2^6..2^20, one full block each (2026-09-23,
#: experiments/12_erw_gpu/README.md step 1); 64 is that with headroom.
#: Changing it changes the draws -- see block_rows.
_BYTES_PER_STEP = 64

#: The sentinel node k must fit the int32 ancestor table.
_MAX_K = 2 ** 31 - 2


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["erw_gpu"] needs cupy (pip install cupy-cuda12x, see '
            'requirements.txt). Without a GPU use MODELS["erw"], the same '
            "walk on CPU.") from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["erw_gpu"] found no usable CUDA device ({err}). '
            'Use MODELS["erw"], the same walk on CPU.') from err
    return cupy


def _resolve(cp, u, k: int, p: float):
    """S_k for each row of u, a (rows, 2k - 1) device block of uniforms.

    models/erw.py's `_resolve`, line for line: column 0 is X_1's coin, columns
    1..k-1 pick the parents of steps 2..k, columns k..2k-2 are their signs,
    and node k is the sentinel (sign 1, its own ancestor, the ancestor of
    step 1). Returns int64 on device.
    """
    rows = u.shape[0]
    n_prev = cp.arange(1, k, dtype=cp.float64)

    sign = cp.ones((rows, k + 1), dtype=cp.int8)
    sign[:, 0] = cp.where(u[:, 0] < 0.5, 1, -1)
    sign[:, 1:k] = cp.where(u[:, k:] < p, 1, -1)

    anc = cp.empty((rows, k + 1), dtype=cp.int32)
    anc[:, 0] = k
    # The clip guards floor(u * t) == t, as on CPU.
    anc[:, 1:k] = cp.minimum((u[:, 1:k] * n_prev).astype(cp.int32),
                             cp.arange(k - 1, dtype=cp.int32))
    anc[:, k] = k
    del u, n_prev

    head = anc[:, :k]
    while not bool(cp.all(head == k)):
        sign[:, :k] *= cp.take_along_axis(sign, head, axis=1)
        anc[:, :k] = cp.take_along_axis(anc, head, axis=1)
        head = anc[:, :k]
    return sign[:, :k].sum(axis=1, dtype=cp.int64)


def block_rows(k: int) -> int:
    """Samples per device block at k steps -- a pure function of k, by design."""
    return max(1, _GPU_WORKING_SET_BYTES // (_BYTES_PER_STEP * k))


def walk(k: int, n: int, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED position S_k, on the GPU. int64, shape (n,).

    Arguments and validation are models/erw.py's `walk`, less `block_n`: the
    block size is part of what the seed means here (see `block_rows`), so it
    is not a knob. The sign is kept for the exact zero-check E S_k = 0.
    """
    p = _check({"p": p})
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if k > _MAX_K:
        raise ValueError(f"k must be <= {_MAX_K} (int32 ancestor table); got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(k)

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        s = _resolve(cp, gen.random(size=(rows, 2 * k - 1), dtype=cp.float64), k, p)
        out[offset:offset + rows] = s.get()
        offset += rows
        del s
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["erw_gpu"].simulate: n i.i.d. samples of Y_i = |S_i| at scale i."""
    return np.abs(walk(i, n, _check(params), rng=rng))
