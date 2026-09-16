"""models/percolation2d.py on a CUDA GPU -- registered as
MODELS["percolation2d_gpu"] in tools/models.py.

Same observable, same parameters, same `cost_hint`, a different device and a
different random stream. The physics -- why "south" measures d_f = 91/48, why
"origin" measures 43/24, why the cylinder removes most of the correction -- is
in models/percolation2d.py and is not repeated here. This file only says what
changes when the pipeline runs on a GPU, and why each choice was made
(plan/gpu_models.md, experiments/07_percolation2d_gpu/README.md).

A separate model, not a backend flag (user, 2026-09-16)
-------------------------------------------------------
For the reason percolation_zd_stream is separate from percolation_zd: the
performance profile is different, and here the random stream is too, so a
seed recorded against MODELS["percolation2d"] does not reproduce a GPU run and
must never be read as if it did. With a separate name the recipe's "model"
field records the device, and no driver learns about hardware.

What is imported from the CPU sibling, and what is not
------------------------------------------------------
DECLARATIONS are imported: p_c, the 4-connected structure, the anchor and
geometry names, the merge-pass cap, `cost_hint` and `zero_rate`. So
MODELS["percolation2d_gpu"].cost_hint IS MODELS["percolation2d"].cost_hint --
the declared work i**2 cannot drift between the two, which is the one thing a
device change must not touch. This is a stated exception to models/README.md's
"no model file imports another", for *_gpu models only (user, 2026-09-16).

The ALGORITHM is not imported: draw, label, merge and count are CuPy ports of
the CPU functions of the same names, kept one to one so the two files can be
read side by side.

The random stream, and the one contract this model does not keep
----------------------------------------------------------------
Each `simulate` call takes ONE integer from the driver's np.random.Generator,
`rng.integers(0, 2**63)`, and seeds CuPy's default cuRAND generator (XORWOW)
with it (user, 2026-09-16). The driver's SeedSequence spawning and seed
recording are untouched, so ground rule 5 holds: the output is a pure function
of (seed, i, n, params) -- on a fixed cupy/CUDA version, since cuRAND does not
promise stream stability across library versions (the experiment README
records both).

cuRAND fills a block's uniforms in one call, so the draws depend on how `n` is
cut into blocks. `block_n` is therefore a fixed function of i (a CONSTANT byte
budget divided by the bytes one sample needs) and is NOT derived from free
VRAM -- the card is shared, and free memory moves. The models/README.md
contract "bit-identical at any block size" does not hold here, exactly as it
does not for models/rwre.py; what holds instead is equality in DISTRIBUTION,
which is what tools/tests/test_percolation2d_gpu.py tests (two-sample KS and a
mean-difference check against the CPU model). If src/generate/generate.py's
chunked path re-enters `simulate` with a smaller n, the run is still valid
i.i.d. sampling, just not bit-comparable to an unchunked one.

cuRAND's float32 uniforms lie in (0, 1] where NumPy's lie in [0, 1). `u < p`
differs between the two only at u = 1.0, which it never selects for p < 1, so
the Bernoulli(p) law is the same up to the 2**-24 float32 grid the CPU
docstring already accounts for.

Staying on the device
---------------------
Labelling is no longer the bottleneck on a GPU (cupyx labels ~1.3e10 sites/s
at dim 2 against 1.5e8 for scipy on this machine, 2026-09-16), so every step --
draw, pad, label, merge, count -- runs on device and only the n per-sample
counts are copied back. A host round trip per block is the whole overhead.

No GPU present
--------------
Importing this module never imports cupy. `simulate` imports it lazily and, if
cupy is missing or no CUDA device is visible, raises RuntimeError naming
MODELS["percolation2d"] as the CPU model to use. It never falls back silently:
a run labelled percolation2d_gpu must have been drawn on a GPU.
"""

from __future__ import annotations

import numpy as np

from models.percolation2d import (
    _FOUR_CONNECTED,
    _MAX_MERGE_PASSES,
    ANCHORS,
    GEOMETRIES,
    P_C_SQUARE_SITE,
    cost_hint,
    zero_rate,
)

__all__ = ["percolation2d_gpu", "simulate", "cost_hint", "zero_rate",
           "ANCHORS", "GEOMETRIES", "P_C_SQUARE_SITE"]

#: Device working-set budget for one block, in bytes. A FIXED constant, not a
#: fraction of free memory: block_n decides how the cuRAND stream is cut, so it
#: must be a pure function of i (see the module docstring). 2 GiB leaves room
#: on a shared 32 GiB card.
_GPU_WORKING_SET_BYTES = 2 * 1024 ** 3

#: Device bytes per padded site across one block: float32 uniform (4), boolean
#: lattice and padded copy (2), int32 labels (4), and the boolean keep gather.
#: The CuPy pool peaked at 9.0-10.1 over i = 8..1024, every anchor and
#: geometry (2026-09-16, experiments/07_percolation2d_gpu/README.md step 1);
#: 16 is that with headroom. Changing it changes the draws -- see block_rows.
_BYTES_PER_SITE = 16


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["percolation2d_gpu"] needs cupy (pip install cupy-cuda12x, '
            'see requirements.txt). Without a GPU use MODELS["percolation2d"], '
            "the same observable on CPU.") from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["percolation2d_gpu"] found no usable CUDA device ({err}). '
            'Use MODELS["percolation2d"], the same observable on CPU.') from err
    return cupy


def _draw_open(cp, gen, rows: int, i: int, p: float):
    """`rows` independent i x i boolean lattices on device, open w.p. p."""
    return gen.random(size=(rows, i, i), dtype=cp.float32) < p


def _label_block(cp, open_grid):
    """models/percolation2d.py's stacking trick on device: one label call.

    Returns int32 (rows, i+1, i); `[:, i, :]` is each sample's separator row.
    """
    from cupyx.scipy import ndimage as cndimage

    rows, i, _ = open_grid.shape
    padded = cp.zeros((rows * (i + 1), i), dtype=cp.bool_)
    padded.reshape(rows, i + 1, i)[:, :i, :] = open_grid
    labels, _ = cndimage.label(padded, structure=cp.asarray(_FOUR_CONNECTED))
    return labels.reshape(rows, i + 1, i)


def _wrap_roots(cp, lab, i: int):
    """label -> canonical label after the periodic x-boundary merge, on device.

    The CPU pointer-jumping union-find over labels, with the STRICT
    termination test (root unchanged after the unions and the squaring) that
    models/percolation_zd.py uses -- not percolation2d.py's squaring-only
    test, which is safe for one wrap direction but is not the version worth
    porting.
    """
    nlab = int(lab.max()) + 1
    root = cp.arange(nlab, dtype=lab.dtype)
    left, right = lab[:, :i, 0], lab[:, :i, i - 1]
    both = (left > 0) & (right > 0)
    if not bool(both.any()):
        return root
    a, b = left[both], right[both]
    lo, hi = cp.minimum(a, b), cp.maximum(a, b)
    for _ in range(_MAX_MERGE_PASSES):
        before = root.copy()
        cp.minimum.at(root, hi, root[lo])
        cp.minimum.at(root, lo, root[hi])
        root = root[root]
        if bool(cp.array_equal(before, root)):
            return root
    raise RuntimeError(
        f"the periodic-boundary label merge did not converge in "
        f"{_MAX_MERGE_PASSES} passes ({nlab} labels, {a.size} wrap edges). "
        f"This should be unreachable -- see models/percolation2d.py's "
        f"_MAX_MERGE_PASSES.")


def _roots(cp, lab, i: int, geometry: str):
    if geometry == "box":
        return cp.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(cp, lab, i)


def _seeded_counts(cp, lab, roots, seeds):
    """Per sample, open sites whose cluster contains a site of `seeds`.

    One keep-mask built per root and one gather over the lattice, for both
    anchors. A cluster never spans two samples (separator rows, and the wrap
    joins a sample only to itself), so marking every sample's seed roots at
    once and summing per sample counts each sample's own clusters only.
    """
    keep_root = cp.zeros(roots.size, dtype=cp.bool_)
    keep_root[roots[seeds]] = True
    keep = keep_root[roots]
    keep[0] = False                   # label 0 is "closed", and the separators
    return keep[lab].sum(axis=(1, 2), dtype=cp.int64)


def _south_counts(cp, lab, roots):
    return _seeded_counts(cp, lab, roots, lab[:, 0, :])


def _origin_counts(cp, lab, i: int, roots):
    """Size of the centre site's cluster, per sample; 0 when it is closed.

    Not the CPU module's two bincounts: cupy's bincount peaks at ~118 device
    bytes per site at i = 256 (measured 2026-09-16), 12x the rest of the
    pipeline, and the seed-set gather gives the same number with no
    per-label histogram.
    """
    return _seeded_counts(cp, lab, roots, lab[:, i // 2, i // 2])


def block_rows(i: int) -> int:
    """Samples per device block at side i -- a pure function of i, by design."""
    return max(1, _GPU_WORKING_SET_BYTES // (_BYTES_PER_SITE * (i + 1) * i))


def percolation2d_gpu(
    i: int,
    n: int = 1,
    p: float = P_C_SQUARE_SITE,
    anchor: str = "south",
    geometry: str = "box",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_i, drawn and labelled on the GPU. int64, shape (n,).

    Arguments and validation are models/percolation2d.py's. There is no
    `block_n` argument: the block size is part of what the seed means here
    (see `block_rows`), so it is not a knob.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; known: {list(ANCHORS)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(i)

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, i, p))
        roots = _roots(cp, lab, i, geometry)
        counts = (_south_counts(cp, lab, roots) if anchor == "south"
                  else _origin_counts(cp, lab, i, roots))
        out[offset:offset + rows] = counts.get()
        offset += rows
        del lab, roots, counts
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation2d_gpu"].simulate. params: as MODELS["percolation2d"]."""
    return percolation2d_gpu(
        i,
        n=n,
        p=float(params.get("p", P_C_SQUARE_SITE)),
        anchor=params.get("anchor", "south"),
        geometry=params.get("geometry", "box"),
        rng=rng,
    )
