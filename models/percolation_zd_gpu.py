"""models/percolation_zd.py on a CUDA GPU -- registered as
MODELS["percolation_zd_gpu"] in tools/models.py.

The same observable in any dimension, every anchor ("face", "face_far",
"origin", "slab") and every geometry ("box", "cylinder", "torus"), with the
same parameters and the same `cost_hint` (i**dim). Which anchor measures which
exponent, and why, is in models/percolation_zd.py and is not repeated here.

Everything models/percolation2d_gpu.py says about the device applies
unchanged, and is only summarized:

  - a separate model because the random stream differs: one
    `rng.integers(0, 2**63)` per call seeds cuRAND XORWOW, so a seed recorded
    for MODELS["percolation_zd"] does not reproduce this model;
  - the block size is part of what that seed means, so `block_rows(i, dim)` is
    a fixed byte budget, never free VRAM, and the models/README.md contract
    "bit-identical at any block size" does not hold -- equality in
    DISTRIBUTION with the CPU model does, and is what
    tools/tests/test_percolation_zd_gpu.py tests;
  - draw, stack, label (`cupyx.scipy.ndimage.label`), merge and count all stay
    on device, and only per-sample results are copied back;
  - importing never imports cupy; without cupy or a CUDA device `simulate`
    raises RuntimeError naming MODELS["percolation_zd"].

Imported from the CPU sibling (user, 2026-09-16): the p_c table and its
sources, the anchor/geometry names and alias, every validator, `cost_hint`,
`zero_rate`, `expected_face_count_1d`, and the three pure INDEXING
declarations -- `_periodic_axes` (which axes a geometry wraps), `_wrap_faces`
(which two faces a wrap identifies) and `_seed_slab` (where the k-slab sits).
Those three only slice an array, so they apply to a cupy array unchanged, and
importing them means the GPU cannot disagree with the CPU about the geometry.
The draw, the labelling, the merge and the counts are ports, never imports.

The one count that is not a line-for-line port: "origin" uses the seed-set
gather ("slab" with k = 0) instead of the CPU module's two bincounts, which on
cupy peak at ~118 device bytes per site (experiments/07_percolation2d_gpu).
The number is the same -- models/percolation_zd.py itself notes that k = 0 IS
the origin anchor.

`crossing_fraction_gpu` ports the CPU module's criticality diagnostic
(Experiment H0) on the same kernel, because a spanning probability that is
flat in i is the check on the tabulated p_c, and a GPU makes it affordable at
dim = 6 (user, 2026-09-16). It is a probability, not a registered observable.
"""

from __future__ import annotations

import numpy as np

from models.percolation_zd import (
    _MAX_MERGE_PASSES,
    ANCHORS,
    GEOMETRIES,
    P_C_SITE_HYPERCUBIC,
    P_C_SOURCE,
    _check_anchor_dim,
    _check_dim,
    _check_size,
    _periodic_axes,
    _resolve_anchor,
    _seed_slab,
    _structure,
    _wrap_faces,
    cost_hint,
    critical_p,
    expected_face_count_1d,
    zero_rate,
)

__all__ = ["percolation_zd_gpu", "crossing_fraction_gpu", "simulate", "cost_hint",
           "critical_p", "zero_rate", "expected_face_count_1d", "block_rows",
           "ANCHORS", "GEOMETRIES", "P_C_SITE_HYPERCUBIC", "P_C_SOURCE"]

#: Device working-set budget for one block, in bytes. Fixed, never a fraction
#: of free memory: it decides how cuRAND's stream is cut (see block_rows).
_GPU_WORKING_SET_BYTES = 2 * 1024 ** 3

#: Device bytes per padded site across one block. The CuPy pool peaked at
#: 8.6-9.6 for dim = 2..8, box and torus, face_far and origin (2026-09-16,
#: experiments/08_percolation_zd_gpu step 1) -- flat in dim, so cupyx's label
#: buffers do not grow with the 3**dim structure. 16 is headroom. Changing it
#: changes the draws.
_BYTES_PER_SITE = 16


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["percolation_zd_gpu"] needs cupy (pip install cupy-cuda12x, '
            'see requirements.txt). Without a GPU use MODELS["percolation_zd"], '
            "the same observable on CPU.") from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["percolation_zd_gpu"] found no usable CUDA device ({err}). '
            'Use MODELS["percolation_zd"], the same observable on CPU.') from err
    return cupy


def block_rows(i: int, dim: int) -> int:
    """Samples per device block -- a pure function of (i, dim), by design."""
    return max(1, _GPU_WORKING_SET_BYTES
               // (_BYTES_PER_SITE * (i + 1) * i ** (dim - 1)))


def _draw_open(cp, gen, rows: int, i: int, dim: int, p: float):
    return gen.random(size=(rows,) + (i,) * dim, dtype=cp.float32) < p


def _label_block(cp, open_grid, dim: int):
    """One label call per block, samples stacked along x_0 with a blank slab
    after each. int32 of shape (rows, i+1) + (i,)*(dim-1)."""
    from cupyx.scipy import ndimage as cndimage

    rows = open_grid.shape[0]
    spatial = open_grid.shape[1:]
    i = spatial[0]
    padded = cp.zeros((rows * (i + 1),) + spatial[1:], dtype=cp.bool_)
    padded.reshape((rows, i + 1) + spatial[1:])[:, :i] = open_grid
    labels, _ = cndimage.label(padded, structure=cp.asarray(_structure(dim)))
    return labels.reshape((rows, i + 1) + spatial[1:])


def _wrap_roots(cp, lab, i: int, axes: tuple[int, ...]):
    """The CPU pointer-jumping merge over labels, with its strict whole-pass
    termination test, on device."""
    nlab = int(lab.max()) + 1
    root = cp.arange(nlab, dtype=lab.dtype)
    a_parts, b_parts = [], []
    for axis in axes:
        u, v = _wrap_faces(lab, i, axis)
        both = (u > 0) & (v > 0)
        if bool(both.any()):
            a_parts.append(u[both])
            b_parts.append(v[both])
    if not a_parts:
        return root
    a = cp.concatenate(a_parts)
    b = cp.concatenate(b_parts)
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
        f"{_MAX_MERGE_PASSES} passes ({nlab} labels, {a.size} wrap edges, "
        f"axes {list(axes)}). This should be unreachable -- see "
        f"models/percolation_zd.py's _MAX_MERGE_PASSES.")


def _roots(cp, lab, i: int, dim: int, geometry: str):
    axes = _periodic_axes(dim, geometry)
    if not axes:
        return cp.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(cp, lab, i, axes)


def _anchor_keep(cp, roots, seeds):
    """Per-label mask: does this label's cluster touch the seed set?"""
    keep_root = cp.zeros(roots.size, dtype=cp.bool_)
    keep_root[roots[seeds]] = True
    keep = keep_root[roots]
    keep[0] = False                   # label 0 is "closed", and the separators
    return keep


def _counts(cp, lab, i: int, keep, depth_from: int):
    """Open sites with keep set and x_0 >= depth_from, per sample."""
    sel = lab[:, depth_from:i]
    return keep[sel].sum(axis=tuple(range(1, sel.ndim)), dtype=cp.int64)


def _seeds(lab, i: int, dim: int, anchor: str, k: int | None):
    if anchor in ("face", "face_far"):
        return lab[:, 0]
    if anchor == "origin":
        return _seed_slab(lab, i, dim, 0)
    return _seed_slab(lab, i, dim, k)


def percolation_zd_gpu(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    anchor: str = "face",
    anchor_dim: int | None = None,
    geometry: str = "box",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_i on an i**dim lattice, on the GPU. int64, (n,).

    Arguments and validation are models/percolation_zd.py's, minus `block_n`,
    which is not a knob here (see block_rows).
    """
    anchor = _resolve_anchor(anchor)
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
    k = _check_anchor_dim(anchor_dim, dim) if anchor == "slab" else None
    i = int(i)
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    p = critical_p(dim) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    _check_size(i, dim)

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(i, dim)
    depth_from = (i // 2) if anchor == "face_far" else 0

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, i, dim, p), dim)
        roots = _roots(cp, lab, i, dim, geometry)
        keep = _anchor_keep(cp, roots, _seeds(lab, i, dim, anchor, k))
        out[offset:offset + rows] = _counts(cp, lab, i, keep, depth_from).get()
        offset += rows
        del lab, roots, keep
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_zd_gpu"].simulate. params: as MODELS["percolation_zd"]."""
    return percolation_zd_gpu(
        i,
        n=n,
        dim=int(params.get("dim", 2)),
        p=params.get("p"),
        anchor=params.get("anchor", "face"),
        anchor_dim=params.get("anchor_dim"),
        geometry=params.get("geometry", "box"),
        rng=rng,
    )


def crossing_fraction_gpu(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    geometry: str = "box",
    rng: np.random.Generator | None = None,
) -> float:
    """models/percolation_zd.py's `crossing_fraction`, on the GPU.

    Fraction of samples in which one cluster touches both x_0 = 0 and
    x_0 = i-1. The same random-stream caveats as `simulate`.
    """
    dim = _check_dim(dim)
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    i = int(i)
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    p = critical_p(dim) if p is None else float(p)
    _check_size(i, dim)

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(i, dim)

    spanned = 0
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, i, dim, p), dim)
        roots = _roots(cp, lab, i, dim, geometry)
        both = _anchor_keep(cp, roots, lab[:, 0]) & _anchor_keep(cp, roots, lab[:, i - 1])
        spanned += int(both[lab[:, 0]].reshape(rows, -1).any(axis=1).sum())
        offset += rows
        del lab, roots, both
    return spanned / float(n)
