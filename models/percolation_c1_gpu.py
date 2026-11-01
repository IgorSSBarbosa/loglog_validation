"""Largest-cluster size C1 of critical site percolation in dim dimensions, on the
GPU -- registered as MODELS["percolation_c1_gpu"] in tools/models.py.

Y_i = number of sites in the largest open cluster of an i**dim lattice at p_c.
E Y_i ~ i**d_f, so gamma = d_f in every dim (d_f = 91/48 at dim = 2).

Why this model (user, 2026-10-31). Zhang, Hou, Fang, Hu & Deng, arXiv:2004.11289,
measure d_f from C1 on a periodic box, and their leading correction for it is fast,
y1 = -2.0(2) at dim = 5. This repo's "face_far" observable has a slow one
(omega1 ~ 0.4-0.6 on experiment 43's 5D replicate), which is what sinks the log-log
slope there. C1 is the observable whose corrections Zhang's allocation was built on.
It departs from PLAN.md's ground rule 7 (the face-connected count), by the user's
decision for experiment 47.

Not Zhang's model either: SITE percolation, not bond. The labeller
(`cupyx.scipy.ndimage.label`) is site-based, and labelling is ~90% of a sample's
cost at dim = 5 (experiment 43, 2026-10-31), so bonds would mean a hand-written
labeller on the expensive step. Site and bond share d_f, but not the correction
amplitudes: y1 for this observable has to be measured, not imported.

Geometry defaults to "torus", Zhang's periodic box. "box" and "cylinder" are
accepted for comparison.

The draw, label, wrap merge and block size are percolation_zd_gpu's own functions,
imported, so the two models label the same lattice the same way and differ only in
what they count. The count:

  - per label, its site count (bincount over the label image);
  - per merged cluster, the sum over its labels (the wrap merge's roots);
  - per sample, the maximum over its clusters.

The last step needs the sample each label belongs to. Samples are stacked along x_0
with a blank slab between them, and `label` numbers clusters by their root's flat
index (sorted), so each sample's labels are one contiguous, increasing range. That is
cupyx's behaviour, not its contract, so `_sample_of_label` checks it on every block
and raises if it ever fails.
"""

from __future__ import annotations

import numpy as np

from models.percolation_zd import GEOMETRIES, _check_dim, _check_size
from models.percolation_zd_gpu import (
    _cupy,
    _draw_open,
    _label_block,
    _roots,
    block_rows,
    cost_hint,
    critical_p,
)

__all__ = ["percolation_c1_gpu", "largest_cluster", "simulate", "cost_hint", "critical_p"]


_KERNELS: dict = {}


def _kernels(cp) -> dict:
    """Built on first use, so importing this module never imports cupy.

    Why custom kernels: ~86% of sites are closed (label 0) at dim = 5, so
    `cp.bincount(lab)` sends most of its atomic adds to one counter, and
    `cp.where(lab > 0, ...)` before a min builds a lattice-sized temporary. At
    dim = 5, L = 64 those took 442 and 236 ms of an 830 ms sample (purus,
    2026-10-31), against 96 ms for the labelling itself.
    """
    if not _KERNELS:
        _KERNELS["count"] = cp.ElementwiseKernel(
            "int32 l", "raw int32 counts",
            "if (l > 0) atomicAdd(&counts[l], 1)", "c1_count_open")
        _KERNELS["lo"] = cp.ReductionKernel(
            "int32 l", "int32 y", "l > 0 ? l : 2147483647", "min(a, b)",
            "y = a", "2147483647", "c1_min_positive")
    return _KERNELS


def _sample_of_label(cp, lab):
    """Sample index of every label (label 0 -> 0, unused). int64, (nlab,)."""
    rows = lab.shape[0]
    flat = lab.reshape(rows, -1)
    hi = flat.max(axis=1)
    nlab = int(hi.max()) + 1
    if rows == 1:
        return cp.zeros(nlab, dtype=cp.int64)
    lo = _kernels(cp)["lo"](flat, axis=1)
    has = hi > 0
    lo_h, hi_h = lo[has], hi[has]
    if lo_h.size > 1 and not bool(cp.all(lo_h[1:] > hi_h[:-1])):
        raise RuntimeError(
            "cupyx.scipy.ndimage.label did not number the stacked samples in "
            "contiguous ranges, so labels cannot be assigned to samples. "
            "See models/percolation_c1_gpu.py's docstring.")
    if not lo_h.size:
        return cp.zeros(nlab, dtype=cp.int64)
    # Over the non-empty rows hi is increasing, so a label's sample is the first
    # of them whose max reaches it. Label 0 lands on the first; its size is 0.
    k = cp.searchsorted(hi_h, cp.arange(nlab, dtype=hi_h.dtype), side="left")
    return cp.flatnonzero(has)[k]


def _largest(cp, lab, roots):
    """Largest merged cluster per sample, int64 (rows,)."""
    rows = lab.shape[0]
    nlab = roots.size
    per_label = cp.zeros(nlab, dtype=cp.int32)  # label 0 (closed, separators) stays 0
    _kernels(cp)["count"](lab, per_label)
    per_root = cp.zeros(nlab, dtype=cp.int64)
    cp.add.at(per_root, roots, per_label)
    out = cp.zeros(rows, dtype=cp.int64)
    cp.maximum.at(out, _sample_of_label(cp, lab), per_root)
    return out


def largest_cluster(open_grid, geometry: str = "torus") -> np.ndarray:
    """C1 of each lattice in `open_grid` (bool, (rows,) + (i,)*dim), on the GPU.

    The model's own count on a given lattice, so a test can compare it with an
    independent one on the same sites.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    cp = _cupy()
    grid = cp.asarray(open_grid, dtype=cp.bool_)
    i, dim = grid.shape[1], grid.ndim - 1
    lab = _label_block(cp, grid, dim)
    return _largest(cp, lab, _roots(cp, lab, i, dim, geometry)).get()


def percolation_c1_gpu(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of C1 on an i**dim lattice. int64, (n,)."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
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

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, i, dim, p), dim)
        out[offset:offset + rows] = _largest(cp, lab, _roots(cp, lab, i, dim, geometry)).get()
        offset += rows
        del lab
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_c1_gpu"].simulate. params: dim, p (default p_c), geometry."""
    return percolation_c1_gpu(
        i,
        n=n,
        dim=int(params.get("dim", 2)),
        p=params.get("p"),
        geometry=params.get("geometry", "torus"),
        rng=rng,
    )
