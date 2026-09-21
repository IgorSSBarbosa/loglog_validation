"""models/percolation_susceptibility.py on a CUDA GPU -- registered as
MODELS["percolation_susceptibility_gpu"] in tools/models.py.

The same off-critical observable, Y = sum_clusters s**(moment+1) / (p L**dim)
on an L(x)**dim lattice at p = p_c - eps0/x, with the same parameters, the
same declared box rule and the same `cost_hint` (L(x)**dim with the ceil'd
L). Why the torus, why nu_box, and what E Y is are in
models/percolation_susceptibility.py and are not repeated here.

The device design is models/percolation2d_gpu.py's, unchanged: a separate
model because the random stream differs (one `rng.integers(0, 2**63)` per call
seeds cuRAND); `block_rows` a fixed byte budget so a seed means one thing;
equal in DISTRIBUTION to the CPU model, which is what
tools/tests/test_percolation_susceptibility_gpu.py tests; cupy imported only
inside `simulate`, which raises without a GPU and names the CPU model.

Imports, mirroring the CPU module's own
---------------------------------------
The CPU module imports percolation_zd's draw, stack-and-label and merge; this
one imports the same three from percolation_zd_gpu, the kernels experiment 08
validated. Declarations -- the box rule, p_at, the moment cap, `cost_hint` --
come from the CPU sibling (user, 2026-09-16), so L(x) and the declared cost
cannot drift.

Cluster sizes without bincount
------------------------------
The CPU module sums s**(moment+1) per CLUSTER, which needs a label -> sample
owner map and two bincounts. On CuPy, `bincount` peaks at ~184 extra device
bytes per site (measured 2026-09-16, experiments/09_percolation_susceptibility_gpu),
so this module computes the SAME sum per SITE instead,

    sum_clusters s**(moment+1) = sum_{open sites x} |C(x)|**moment,

with the size per label from `cupy.add.at` (no extra memory) and one
float64 gather of size**moment over the lattice (8 bytes per site). No owner
map is needed: the gather is summed per sample over that sample's own sites.
"""

from __future__ import annotations

import numpy as np

from models.percolation_susceptibility import (
    _MAX_MOMENT,
    DEFAULT_BOX_FACTOR,
    DEFAULT_NU_BOX,
    GEOMETRIES,
    _check_dim,
    _check_size,
    box_side,
    cost_hint,
    critical_p,
    declared_cost_exponent,
    epsilon,
    p_at,
)
from models.percolation_zd_gpu import _draw_open, _label_block, _roots

__all__ = ["percolation_susceptibility_gpu", "simulate", "cost_hint", "block_rows",
           "box_side", "p_at", "epsilon", "declared_cost_exponent", "critical_p",
           "DEFAULT_BOX_FACTOR", "DEFAULT_NU_BOX", "GEOMETRIES"]

#: Device working-set budget for one block. Fixed: it decides how cuRAND's
#: stream is cut (see block_rows).
_GPU_WORKING_SET_BYTES = 2 * 1024 ** 3

#: Device bytes per padded site: percolation_zd_gpu's ~10 plus the float64
#: size**moment gather (8). The CuPy pool peaked at 17.2-17.8 at dim 2-4,
#: torus, moment 2 (2026-09-16, experiments/09_percolation_susceptibility_gpu
#: step 1); 24 is headroom. Changing it changes the draws.
_BYTES_PER_SITE = 24


_NAME = "percolation_susceptibility_gpu"
_CPU_SIBLING = "percolation_susceptibility"


def _cupy(name: str = _NAME, cpu: str | None = _CPU_SIBLING):
    """cupy, or a RuntimeError saying which CPU model to use instead (if any)."""
    instead = (f'Without a GPU use MODELS["{cpu}"], the same observable on CPU.'
               if cpu else "This model has no CPU version.")
    try:
        import cupy
    except ImportError as err:
        raise RuntimeError(
            f'MODELS["{name}"] needs cupy (pip install cupy-cuda12x, see '
            f'requirements.txt). {instead}') from err
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as err:
        raise RuntimeError(
            f'MODELS["{name}"] found no usable CUDA device ({err}). '
            f'{instead}') from err
    return cupy


def block_rows(L: int, dim: int) -> int:
    """Samples per device block -- a pure function of (L, dim), by design."""
    return max(1, _GPU_WORKING_SET_BYTES
               // (_BYTES_PER_SITE * (L + 1) * L ** (dim - 1)))


def cluster_moment_sum(cp, lab, L: int, roots, moment: int):
    """sum_clusters s**(moment+1) per sample, as sum over sites of |C(x)|**moment."""
    by_label = cp.zeros(roots.size, dtype=cp.float64)
    cp.add.at(by_label, lab.ravel(), 1.0)
    size = cp.zeros(roots.size, dtype=cp.float64)
    cp.add.at(size, roots, by_label)
    size_of_label = size[roots]
    # size**moment by repeated multiplication, NOT cupy's `**`: that goes
    # through the device pow(), which is inexact on integers (3.0**1 ->
    # 2.9999999999999996, 19.0**3 one ulp low), so the same Y came out in two
    # representations one ulp apart and a two-sample KS against the CPU model
    # read the split atom as a distribution difference (p = 7.5e-5 at a 4x4
    # torus; experiments/09 step 2). Products of integers are exact below
    # 2**53, as numpy's power is.
    weight = size_of_label.copy()          # per label: its cluster's size**moment
    for _ in range(moment - 1):
        weight *= size_of_label
    weight[0] = 0.0                        # closed sites and the separators
    sel = lab[:, :L]
    return weight[sel].sum(axis=tuple(range(1, sel.ndim)))


def _check_design(dim: int, moment: int, geometry: str) -> tuple[int, int]:
    """Validated (dim, moment); the checks every caller of the core needs."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
    moment = int(moment)
    if not 1 <= moment <= _MAX_MOMENT:
        raise ValueError(f"moment must be in 1..{_MAX_MOMENT}; got {moment}")
    return dim, moment


def _susceptibility_gpu_at(L: int, p: float, n: int, dim: int, moment: int,
                           geometry: str, rng: np.random.Generator | None,
                           *, name: str = _NAME,
                           cpu: str | None = _CPU_SIBLING) -> np.ndarray:
    """n samples of Y at an explicit torus side `L` and occupation `p`.

    The draw, and nothing about how (L, p) were chosen: the ladder that maps a
    rung to (L, p) belongs to the caller. `percolation_susceptibility_gpu`
    indexes rungs by the distance to criticality x, and
    models/percolation_susceptibility_L_gpu.py by the box side L. `dim`,
    `moment` and `geometry` arrive validated (`_check_design`). Consumes
    exactly one integer of `rng`, as always. `name` and `cpu` only word the
    error raised without a GPU (`cpu=None`: the caller has no CPU version).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    _check_size(L, dim)

    cp = _cupy(name, cpu)
    rng = rng if rng is not None else np.random.default_rng()
    gen = cp.random.default_rng(int(rng.integers(0, 2 ** 63)))
    rows_per_block = block_rows(L, dim)

    sites = float(L) ** dim
    out = np.empty(n, dtype=np.float64)
    offset = 0
    while offset < n:
        rows = min(rows_per_block, n - offset)
        lab = _label_block(cp, _draw_open(cp, gen, rows, L, dim, p), dim)
        roots = _roots(cp, lab, L, dim, geometry)
        out[offset:offset + rows] = (
            cluster_moment_sum(cp, lab, L, roots, moment) / (p * sites)).get()
        offset += rows
        del lab, roots
    return out


def percolation_susceptibility_gpu(
    x: int,
    n: int = 1,
    dim: int = 2,
    eps0: float = 0.5,
    p: float | None = None,
    p_c: float | None = None,
    box_factor: float = DEFAULT_BOX_FACTOR,
    nu_box: float = DEFAULT_NU_BOX,
    moment: int = 1,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y = sum_clusters s^(moment+1) / (p * L^dim), on the GPU.

    Arguments and validation are models/percolation_susceptibility.py's, minus
    `block_n`, which is not a knob here (see block_rows). float64, shape (n,).
    """
    dim, moment = _check_design(dim, moment, geometry)
    x = int(x)
    if x < 1:
        raise ValueError(f"ladder position x must be >= 1; got {x}")

    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    L = box_side(x, box_factor, nu_box)
    return _susceptibility_gpu_at(L, p, n, dim, moment, geometry, rng)


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_gpu"].simulate.

    params: as MODELS["percolation_susceptibility"].
    """
    return percolation_susceptibility_gpu(
        x,
        n=n,
        dim=int(params.get("dim", 2)),
        eps0=float(params.get("eps0", 0.5)),
        p=params.get("p"),
        p_c=params.get("p_c"),
        box_factor=float(params.get("box_factor", DEFAULT_BOX_FACTOR)),
        nu_box=float(params.get("nu_box", DEFAULT_NU_BOX)),
        moment=int(params.get("moment", 1)),
        geometry=params.get("geometry", "torus"),
        rng=rng,
    )
