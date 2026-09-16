"""models/percolation_tau.py on a CUDA GPU -- registered as
MODELS["percolation_tau_gpu"] in tools/models.py.

The same cluster-number density Y_s (clusters of size in [s, bin_ratio*s), or
>= s, per site of an L(s) x L(s) critical lattice), with the same parameters,
the same box rules and the same `cost_hint`; and the same `shared_sampler`,
which reads every rung off one box and stamps `shared_lattice` in its info.
Why the ladder variable is a cluster size, why the torus, why box_factor = 16
and what the shared box costs in correlation are in models/percolation_tau.py
and are not repeated here.

The device design is models/percolation2d_gpu.py's, unchanged: a separate
model because the random stream differs (one `rng.integers(0, 2**63)` per call
seeds cuRAND, for `simulate` and `shared_sampler` alike); `block_rows` a fixed
byte budget so a seed means one thing; equal in DISTRIBUTION to the CPU model
-- per rung, and for `shared_sampler` also in the covariance between rungs --
which is what tools/tests/test_percolation_tau_gpu.py tests; cupy imported
only inside the samplers, which raise without a GPU and name the CPU model.

Imports
-------
Declarations come from the CPU sibling (user, 2026-09-16): p_c, the box rule
`box_side`, `shared_box_side`, `bin_edges`, the design defaults (box_factor,
box_exponent, bin_ratio, df_lower, cut_fraction), `cost_hint` and
`zero_fraction`. So L(s), the shared box and the bin windows cannot drift from
the CPU model's. Draw, stack-and-label and the torus merge are
percolation_zd_gpu's at dim = 2 (the 4-connected cross, both axes periodic on
the torus), the kernels experiments 08 and 09 validated.

Counting without bincount
-------------------------
The CPU module's per-label size and label -> sample map are kept, as
`cupy.add.at` accumulations and one scatter over the sites; cupy's `bincount`
is not used (it peaks at ~118-184 extra device bytes per site, experiments
07 and 09). Measured with the counting in place, the block peaks at 9.1-9.2
device bytes per site, the labelling's own peak (L = 64..2048, torus and box,
2026-09-16, experiments/10_percolation_tau_gpu step 1). Every window after
that is label-sized. Sizes are float64 sums of 1.0, exact integers; nothing
here raises to a power (experiment 09's `**` trap).
"""

from __future__ import annotations

import math

import numpy as np

from models.percolation_tau import (
    DEFAULT_BIN_RATIO,
    DEFAULT_BOX_EXPONENT,
    DEFAULT_BOX_FACTOR,
    DEFAULT_CUT_FRACTION,
    DEFAULT_DF_LOWER,
    GEOMETRIES,
    OBSERVABLES,
    P_C_SQUARE_SITE,
    bin_edges,
    box_side,
    cost_hint,
    shared_box_side,
    zero_fraction,
)
from models.percolation_zd_gpu import _draw_open, _label_block, _roots

__all__ = ["percolation_tau_gpu", "simulate", "shared_sampler", "binned_counts_gpu",
           "cost_hint", "block_rows", "box_side", "shared_box_side", "bin_edges",
           "zero_fraction", "OBSERVABLES", "GEOMETRIES", "P_C_SQUARE_SITE"]

#: Device working-set budget for one block. Fixed: it decides how cuRAND's
#: stream is cut (see block_rows).
_GPU_WORKING_SET_BYTES = 2 * 1024 ** 3

#: Device bytes per padded site. The CuPy pool peaked at 9.1-9.2 with the
#: counting in place (experiments/10_percolation_tau_gpu step 1); 16 is that
#: with headroom, as in percolation2d_gpu. Changing it changes the draws.
_BYTES_PER_SITE = 16

_DIM = 2


def _cupy():
    """cupy, or a RuntimeError saying which CPU model to use instead."""
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            'MODELS["percolation_tau_gpu"] needs cupy (pip install cupy-cuda12x, '
            'see requirements.txt). Without a GPU use MODELS["percolation_tau"], '
            'the same observable on CPU.') from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["percolation_tau_gpu"] found no usable CUDA device ({err}). '
            f'Use MODELS["percolation_tau"], the same observable on CPU.') from err
    return cupy


def block_rows(L: int) -> int:
    """Samples per device block at box side L -- a pure function of L, by design."""
    return max(1, _GPU_WORKING_SET_BYTES // (_BYTES_PER_SITE * (L + 1) * L))


def _sizes_and_owner(cp, lab, L: int, roots):
    """Per-label cluster size (0 for non-roots) and per-label sample index.

    The CPU function of the same name. The owner scatter is well defined for
    the CPU's reason: a cluster never spans two samples.
    """
    rows = lab.shape[0]
    by_label = cp.zeros(roots.size, dtype=cp.float64)
    cp.add.at(by_label, lab.ravel(), 1.0)
    size = cp.zeros(roots.size, dtype=cp.float64)
    cp.add.at(size, roots, by_label)
    size[0] = 0.0                      # label 0 is "closed", and the separators

    owner = cp.zeros(roots.size, dtype=cp.int32)
    owner[lab[:, :L]] = cp.arange(rows, dtype=cp.int32).reshape(rows, 1, 1)
    return size, owner


def _count_window(cp, size, owner, rows: int, s_lo: int, s_hi: int | None):
    """Clusters per sample with size in [s_lo, s_hi) (s_hi None = no cap)."""
    in_range = size >= s_lo
    if s_hi is not None:
        in_range &= size < s_hi
    counts = cp.zeros(rows, dtype=cp.int64)
    cp.add.at(counts, owner[in_range], 1)
    return counts


def percolation_tau_gpu(
    s: int,
    n: int = 1,
    p: float = P_C_SQUARE_SITE,
    observable: str = "bin",
    bin_ratio: float = DEFAULT_BIN_RATIO,
    box_factor: float = DEFAULT_BOX_FACTOR,
    box_exponent: float = DEFAULT_BOX_EXPONENT,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_s, drawn and labelled on the GPU. float64, shape (n,).

    Arguments and validation are models/percolation_tau.py's, minus `block_n`,
    which is not a knob here (see block_rows).
    """
    if observable not in OBSERVABLES:
        raise ValueError(f"unknown observable {observable!r}; known: {list(OBSERVABLES)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    s = int(s)
    L = box_side(s, box_factor, box_exponent)     # validates s, box_factor, box_exponent
    if L * L < s:
        raise ValueError(
            f"box rule gives L = {L} at s = {s}, so a cluster of size {s} does "
            f"not fit in the box at all and Y_s is identically 0; raise "
            f"box_factor (currently {box_factor}) or box_exponent "
            f"(currently {box_exponent})")

    s_hi: int | None = None
    if observable == "bin":
        if bin_ratio <= 1.0:
            raise ValueError(f"bin_ratio must be > 1; got {bin_ratio}")
        s_hi = int(math.floor(bin_ratio * s + 0.5))
        if s_hi <= s:
            raise ValueError(
                f"bin [s, bin_ratio*s) is empty at s = {s}, bin_ratio = "
                f"{bin_ratio}: upper edge rounds to {s_hi}")

    counts = binned_counts_gpu(L, n, [(s, s_hi)], p=p, geometry=geometry, rng=rng)
    return counts[:, 0] / float(L) ** 2


def simulate(s: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_tau_gpu"].simulate. params: as MODELS["percolation_tau"]."""
    return percolation_tau_gpu(
        s,
        n=n,
        p=float(params.get("p", P_C_SQUARE_SITE)),
        observable=params.get("observable", "bin"),
        bin_ratio=float(params.get("bin_ratio", DEFAULT_BIN_RATIO)),
        box_factor=float(params.get("box_factor", DEFAULT_BOX_FACTOR)),
        box_exponent=float(params.get("box_exponent", DEFAULT_BOX_EXPONENT)),
        geometry=params.get("geometry", "torus"),
        rng=rng,
    )


def binned_counts_gpu(
    L: int,
    n: int,
    windows,
    p: float = P_C_SQUARE_SITE,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(n, len(windows)) int64 counts, every window read off the SAME n lattices.

    models/percolation_tau.py's `binned_counts` on device. `simulate` is this
    with one window, as on the CPU the two paths share the counting, so a
    window's column does not depend on which other windows were asked for.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if L < 1:
        raise ValueError(f"box side must be >= 1; got {L}")
    windows = [(int(lo), None if hi is None else int(hi)) for lo, hi in windows]
    if not windows:
        raise ValueError("windows must be non-empty")

    cp = _cupy()
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(L)

    out = np.empty((n, len(windows)), dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, L, _DIM, p), _DIM)
        size, owner = _sizes_and_owner(cp, lab, L, _roots(cp, lab, L, _DIM, geometry))
        del lab
        block = cp.stack([_count_window(cp, size, owner, rows, s_lo, s_hi)
                          for s_lo, s_hi in windows], axis=1)
        out[offset:offset + rows] = block.get()
        offset += rows
        del size, owner, block
    return out


def shared_sampler(scales, n: int, params: dict,
                   rng: np.random.Generator) -> tuple[dict, dict]:
    """MODELS["percolation_tau_gpu"].shared_sampler -- the whole ladder, one box.

    models/percolation_tau.py's `shared_sampler` on device: the same box, the
    same windows, the same normalization by L**2 and the same `info`, stamped
    `shared_lattice`. The rungs are correlated by construction, exactly as on
    the CPU.
    """
    scales = [int(s) for s in scales]
    if sorted(scales) != scales or len(set(scales)) != len(scales):
        raise ValueError(f"scales must be strictly increasing; got {scales}")
    windows = bin_edges(scales, float(params.get("bin_ratio", DEFAULT_BIN_RATIO)))
    df_lower = float(params.get("df_lower", DEFAULT_DF_LOWER))
    cut_fraction = float(params.get("cut_fraction", DEFAULT_CUT_FRACTION))
    L = shared_box_side(windows[-1][1], df_lower, cut_fraction)

    counts = binned_counts_gpu(L, n, windows,
                               p=float(params.get("p", P_C_SQUARE_SITE)),
                               geometry=params.get("geometry", "torus"),
                               rng=rng)
    area = float(L) ** 2
    samples = {s: counts[:, j] / area for j, s in enumerate(scales)}
    info = {
        "shared_lattice": True,
        "L": L,
        "df_lower": df_lower,
        "cut_fraction": cut_fraction,
        "n_lattices": int(n),
        "sites": int(n) * L * L,
        "windows": [list(w) for w in windows],
        "cut_ratio": [w[1] / L ** df_lower for w in windows],
    }
    return samples, info
