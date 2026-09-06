"""Cluster-number density of critical site percolation on Z^dim -- the Fisher
exponent tau in a spatial dimension that is a MODEL PARAMETER. Registered as
MODELS["percolation_tau_zd"] in tools/models.py.

This is models/percolation_tau.py generalized the way models/percolation_zd.py
generalizes models/percolation2d.py: same observable, same box rule, same
one-`ndimage.label`-per-block stacking, same vectorized label merge, same
shared-lattice sampler -- with `dim` moved out of the filename and into
`params`. At dim = 2 with `box_exponent = 0.5` it reproduces
models/percolation_tau.py bit for bit at the same seed
(`test_matches_percolation_tau_bit_for_bit`).

The observable, unchanged
--------------------------
At p = p_c the number of clusters of size s per lattice site decays as a pure
power (Stauffer & Aharony; Malthe-Sorenssen eq. 4.4),

    n_s ~ C s**(-tau),

and to put tau in front of an estimator that fits log E Y_i against log i, the
LADDER VARIABLE must be a cluster size. So `simulate`'s first argument is a
cluster size s, and

    Y_s = #{clusters with size in [s, b*s)} / L(s)**dim        ("bin")
    Y_s = #{clusters with size >= s}       / L(s)**dim        ("tail")

with E Y_s ~ a0 * s**(1-tau), i.e. gamma = 1 - tau and the reporting-time
conversion is tau-hat = 1 - gamma-hat. Dividing by the exactly-known box
VOLUME (L**dim here, L**2 there) is what makes that conversion carry no design
constant. The bin width is deliberately not divided out; see
models/percolation_tau.py for that argument, which is dimension-free.

What tau is, per dimension
---------------------------
Hyperscaling gives tau = 1 + dim/d_f, and above the upper critical dimension
6 both saturate at the mean-field values:

    dim         2         3        4       5       6+
    d_f       91/48     2.523    3.045   3.54      4
    tau      187/91     2.189    2.313   2.412    5/2
    gamma   -96/91     -1.189   -1.313  -1.412   -3/2

**dim >= 6 is the interesting rung**: tau = 5/2 and d_f = 4 are EXACT there,
not numerical estimates, so a run in dim = 6 or 7 is scored against a known
rational rather than against a literature best-fit. This repo has had exactly
one such target so far (the planted `synthetic` model); it is the first time a
real simulated process has one.

None of these numbers is in the code path. They are acceptance criteria in
experiments/05_percolation_highd/README.md and --expect-gamma / --truth flags
at reporting time (user's decision 2026-08-20; plans/three_experiment_ladder.md
D1/D2). The exception is DF_LOWER below, which is a DESIGN constant -- it sizes
boxes and never reaches an estimator, the same category as an allocation
rule's `omega1` -- and the experiment README states the test that shows it:
two runs at different `box_exponent` must agree on tau-hat.

The box rule, and why dim = 2's default does not survive
---------------------------------------------------------
    L(s) = ceil(box_factor * s**box_exponent),   cost(s) = L(s)**dim sites,

so the article's cost exponent is d = dim * box_exponent. The finite box cuts
n_s off around s ~ L**d_f, and what matters is that every rung sits at the same
relative distance from ITS OWN cutoff, i.e. that

    s / L**d_f  ~  s**(1 - d_f*box_exponent) / box_factor**d_f

does not drift along the ladder. models/percolation_tau.py's default,
box_exponent = 1/2, is exactly 1/dim in two dimensions, and its drift exponent
is 1 - d_f/dim = 1 - (91/48)/2 = 5/96 = 0.052 -- small enough to live with,
which is why 1/2 was a defensible default there. The same choice in higher
dimensions is not:

    dim                      2       3       4       5       6
    1 - d_f/dim  at 1/dim  0.052   0.159   0.239   0.292   0.333

a third of a decade of drift per decade of s at dim = 6. So the default here is

    box_exponent = 1 / DF_LOWER[dim],

which freezes the ratio (conservatively, since DF_LOWER is a LOWER bound on
d_f, so L grows slightly faster than strictly needed and the ladder stays on
the safe side of its cutoff). The option models/percolation_tau.py documents
as available becomes mandatory above two dimensions. Two consequences:

  - the cost exponent is d = dim/d_f = tau - 1, so the article's cost exponent
    and the exponent being measured are the same number up to a sign and a
    shift: d = -gamma. At dim >= 6 that is exactly 3/2.
  - Assumption 6 holds outright rather than drifting: the mean count per box
    is lambda(s) = L**dim * C s**(1-tau) ~ box_factor**dim * C, constant along
    the ladder, and for a near-Poisson count Var(xi_s) ~ 1/lambda.

`box_factor` and the unit trap, made dimension-free
----------------------------------------------------
lambda ~ box_factor**dim * C, so a box_factor carried across dimensions
unchanged would change the count per box by orders of magnitude. What is
carried across instead is box_factor**dim -- the number of lattice SITES in
one allocation budget unit, which is also `tools/cost_model.cost_unit_ratio`
for this model. Fixing it at models/percolation_tau.py's value,

    box_factor = SITES_PER_BUDGET_UNIT ** (1/dim),   SITES_PER_BUDGET_UNIT = 256,

makes "one budget unit is 256 lattice sites" true in every dimension, so the
recipe-writing rule (ask for S/256 to spend S sites) does not become a
per-dimension footnote. It does NOT make lambda equal across dimensions: C is
dimension-dependent and is a measurement, not a constant this file can state.
Read `zero_fraction` off a pilot and raise `box_factor` if it is above ~10%;
that is the same procedure the 2-D model's box_factor = 16 came from, and the
experiment README records the numbers per dimension.

Boundaries: torus by default, for models/percolation_tau.py's reason
---------------------------------------------------------------------
A cluster touching a wall is truncated, i.e. recorded at a size below its own,
which does not merely lose clusters from a bin but FEEDS the bin from above --
and since n_s falls steeply, the influx wins (measured 57% more clusters in
every bin, in 2-D). A torus has no wall. The effect grows with dimension,
since the fraction of a box within one correlation length of some wall grows
with the number of faces, so the default matters more here, not less.
`geometry = "box"` is kept so the two can be run head to head.

The shared-lattice sampler
---------------------------
`shared_sampler` is models/percolation_tau.py's cheap version carried over: one
box, sized by the TOP rung, serves the whole ladder, at the price of
correlating the rungs. In 2-D it was measured 3.2x cheaper AND closer to tau at
every m0 (experiments/03_percolation_zd), with a flat +0.11 correlation
pedestal that cancels from a slope whose weights sum to zero. The saving grows
with dimension: the per-rung sampler pays sum_k n_k L(s_k)**dim, and L**dim
grows faster in higher dim, so the ratio between the two designs widens.
Whether the pedestal stays benign in high dim is one of the questions
experiments/05_percolation_highd asks.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

#: Site-percolation thresholds on Z^dim with nearest-neighbour connectivity.
#: The same table as models/percolation_zd.py's, restated rather than imported:
#: models/README.md's rule is that no model file imports another. See
#: models/percolation_zd.py's P_C_SOURCE for the reference behind each entry
#: and `crossing_fraction` there for the diagnostic that checks one.
P_C_SITE_HYPERCUBIC: dict[int, float] = {
    1: 1.0,
    2: 0.59274605079210,
    3: 0.3116077,
    4: 0.19688561,
    5: 0.14079633,
    6: 0.109016661,
    7: 0.088951121,
    8: 0.075210128,
    9: 0.065209595,
    10: 0.057592949,
    11: 0.051589684,
    12: 0.046730976,
    13: 0.042715021,
}

#: LOWER bounds on the fractal dimension d_f, per spatial dimension. A DESIGN
#: constant in the sense tools/allocation.py's `omega1` is one: it sizes boxes
#: and decides which rungs a box can serve, it never reaches an estimator, and
#: getting it wrong changes the noise and the cost, not the fitted tau. Being
#: a LOWER bound is what makes it safe -- too small keeps the ladder further
#: below its cutoff (wasteful, never wrong); too large keeps rungs the box has
#: already cut off.
#:
#: Each entry sits ~2% below the literature d_f, which is the margin
#: models/percolation_tau.py chose in 2-D (1.85 against 91/48 = 1.8958, and
#: below this repo's own measured 1.9002/1.9059/1.9161). d_f = 4 is EXACT for
#: dim >= 6 (mean field), so 3.92 there is margin against finite-size drift
#: rather than against an uncertain value.
DF_LOWER: dict[int, float] = {
    2: 1.85,     # d_f = 91/48 = 1.8958
    3: 2.47,     # d_f ~ 2.523
    4: 2.98,     # d_f ~ 3.045
    5: 3.46,     # d_f ~ 3.54
    6: 3.92,     # d_f = 4 exactly (mean field, dim >= 6)
    7: 3.92,
    8: 3.92,
}

#: Lattice sites in one allocation budget unit, held fixed across dimensions:
#: box_factor = SITES_PER_BUDGET_UNIT**(1/dim), so box_factor**dim is this
#: number whatever dim is. 256 is models/percolation_tau.py's 2-D value
#: (box_factor 16), carried over so that "ask for S/256 to spend S sites" --
#: the unit trap that broke the planner once -- reads the same in every
#: dimension. See the module docstring.
SITES_PER_BUDGET_UNIT = 256.0

OBSERVABLES = ("bin", "tail")
GEOMETRIES = ("torus", "box")

#: Bin upper edge, [s, bin_ratio*s), for observable = "bin". 2.0 tiles a
#: powers-of-two ladder exactly -- the book's a = 2 logarithmic binning.
DEFAULT_BIN_RATIO = 2.0

#: kappa in s <= kappa * L**df_lower, for the shared-lattice sampler. MEASURED
#: in 2-D (models/percolation_tau.py's CUT_FRACTION_CALIBRATION: bias under the
#: noise floor for s/L**d_f <~ 0.23, exploding past ~0.4, so 0.05 keeps a 4.6x
#: margin). Carried over unchanged because the quantity it bounds is the
#: dimensionless ratio s/L**d_f, which is what the 2-D calibration showed the
#: cutoff is a function of -- but that transfer is an assumption, and
#: experiments/05_percolation_highd/README.md's H5 is the check on it.
DEFAULT_CUT_FRACTION = 0.05

#: Safety cap on the pointer-jumping loop in `_wrap_roots`; it doubles its
#: reach each pass and provably terminates, so this only ever fires on a bug.
_MAX_MERGE_PASSES = 64

#: Working-set budget for one (block_n, L, dim) draw, in bytes -- the same
#: guard, and the same reason, as models/srw.py, models/percolation2d.py and
#: models/percolation_zd.py.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: Per site, per sample: 4 bytes for the float32 uniform, 1 for the boolean
#: padded lattice, 4 for the int32 label image. Rounded up, ignoring the
#: (L+1)/L separator overhead.
_BYTES_PER_SITE = 10

#: `ndimage.label` returns int32 labels, so a lattice of 2**31 or more sites
#: cannot be labelled at all. Refused up front rather than as a MemoryError.
_MAX_SITES_PER_SAMPLE = 2 ** 31 - 1

#: Largest spatial dimension this file will build, for models/percolation_zd.py's
#: reason: `_structure` materializes a dense 3**dim neighbourhood array, 1.6 M
#: entries at dim = 13 and 3.5 G at dim = 20 -- an unrecoverable allocation from
#: a typo, before any lattice is drawn.
_MAX_DIM = 13

#: ceil() with a tolerance, so that box_factor * s**box_exponent landing on an
#: integer from below (8 * 16**0.5 = 32.000000000000004 on some libms) does not
#: cost a whole extra row and column.
_CEIL_TOL = 1e-9


def critical_p(dim: int) -> float:
    """p_c on Z^dim, from the table -- raising, never guessing, if absent."""
    if dim not in P_C_SITE_HYPERCUBIC:
        raise ValueError(
            f"no tabulated p_c for dim = {dim}; known dimensions: "
            f"{sorted(P_C_SITE_HYPERCUBIC)}. Pass params['p'] explicitly, and "
            f"record where the value came from.")
    return P_C_SITE_HYPERCUBIC[dim]


def df_lower(dim: int) -> float:
    """The design lower bound on d_f in `dim` dimensions (see DF_LOWER)."""
    if dim not in DF_LOWER:
        raise ValueError(
            f"no design d_f bound for dim = {dim}; known: {sorted(DF_LOWER)}. "
            f"Pass params['box_exponent'] (and params['df_lower'] for the "
            f"shared sampler) explicitly.")
    return DF_LOWER[dim]


def default_box_exponent(dim: int) -> float:
    """1 / DF_LOWER[dim] -- the exponent that freezes the cutoff ratio.

    At dim = 2 this is 0.5405, NOT models/percolation_tau.py's 0.5. The two
    differ deliberately: 0.5 is 1/dim, whose drift 1 - d_f/dim is negligible in
    two dimensions and is not in higher ones (module docstring). Pass
    `box_exponent = 0.5` explicitly to reproduce the 2-D model exactly.
    """
    return 1.0 / df_lower(dim)


def default_box_factor(dim: int) -> float:
    """SITES_PER_BUDGET_UNIT**(1/dim): 16 at dim=2, 6.35 at 3, 4 at 4, ...

    Holds box_factor**dim -- the sites in one allocation budget unit, i.e.
    `tools/cost_model.cost_unit_ratio` for this model -- fixed at 256 across
    dimensions. It does NOT hold the mean count per box fixed; see the module
    docstring on why that is a measurement rather than a constant.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    return float(SITES_PER_BUDGET_UNIT) ** (1.0 / dim)


def box_side(s: int, box_factor: float, box_exponent: float) -> int:
    """Side of the box a rung at cluster size `s` is drawn on.

    L(s) = ceil(box_factor * s**box_exponent), at least 1. Deterministic, and
    the single place the box rule is defined -- `cost_hint` returns its dim'th
    power and `simulate` divides its counts by the same.
    """
    if s < 1:
        raise ValueError(f"cluster size s must be >= 1; got {s}")
    if box_factor <= 0:
        raise ValueError(f"box_factor must be > 0; got {box_factor}")
    if box_exponent < 0:
        raise ValueError(f"box_exponent must be >= 0; got {box_exponent}")
    return max(1, int(math.ceil(box_factor * float(s) ** box_exponent - _CEIL_TOL)))


def _box_rule(params: dict, dim: int) -> tuple[float, float]:
    """(box_factor, box_exponent), each from `params` or its dim-aware default."""
    factor = params.get("box_factor")
    exponent = params.get("box_exponent")
    return (default_box_factor(dim) if factor is None else float(factor),
            default_box_exponent(dim) if exponent is None else float(exponent))


def _check_dim(dim: int) -> int:
    """The one place a spatial dimension is range-checked."""
    dim = int(dim)
    if not 1 <= dim <= _MAX_DIM:
        raise ValueError(
            f"dim must be in [1, {_MAX_DIM}]; got {dim}. The upper bound is "
            f"where the dense 3**dim neighbourhood `_structure` builds stops "
            f"being allocatable.")
    return dim


def _structure(dim: int) -> np.ndarray:
    """Nearest-neighbour (2*dim-)connectivity -- the one p_c is tabulated for."""
    return ndimage.generate_binary_structure(dim, 1)


def _check_size(L: int, dim: int) -> int:
    """Sites in one padded sample, refusing what int32 labels cannot hold."""
    sites = (L + 1) * L ** (dim - 1)
    if sites > _MAX_SITES_PER_SAMPLE:
        raise ValueError(
            f"one sample at L = {L}, dim = {dim} is {sites:.3g} sites, past "
            f"ndimage.label's int32 label space ({_MAX_SITES_PER_SAMPLE}) and "
            f"{sites * _BYTES_PER_SITE / 2**30:.1f} GiB of working set. Lower "
            f"the top of the ladder, or box_factor.")
    return sites


def _draw_open(rng: np.random.Generator, rows: int, L: int, dim: int,
               p: float) -> np.ndarray:
    """`rows` independent L**dim boolean lattices, each site open w.p. p.

    float32 uniforms, one draw per site, so that splitting `rows` into
    sequential blocks consumes the RNG stream exactly as one unblocked call
    would and the output is bit-identical at any `block_n` (models/srw.py's
    `_draw_heads` has the full argument; bit-packed integer draws break it).
    """
    return rng.random(size=(rows,) + (L,) * dim, dtype=np.float32) < p


def _label_block(open_grid: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Label every sample's open clusters in ONE ndimage.label call.

    The block's lattices are stacked along the first spatial axis with a blank
    separator SLAB after each; no open path can cross a closed slab, so
    labelling the stack once IS the per-sample labelling. Returns int32 of
    shape (rows, L+1) + (L,)*(dim-1).

    Mirrors models/percolation_zd.py's function of the same name, and
    models/percolation2d.py's before it. Deliberately duplicated rather than
    imported: models/README.md's rule is that no model file imports another.
    """
    rows = open_grid.shape[0]
    spatial = open_grid.shape[1:]
    L = spatial[0]
    padded = np.zeros((rows * (L + 1),) + spatial[1:], dtype=bool)
    padded.reshape((rows, L + 1) + spatial[1:])[:, :L] = open_grid
    labels, _ = ndimage.label(padded, structure=structure)
    return labels.reshape((rows, L + 1) + spatial[1:])


def _wrap_faces(lab: np.ndarray, L: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """The two opposite faces that making spatial `axis` periodic identifies.

    Spatial axis `a` is array axis `a+1`; the stack axis (a = 0) carries the
    separator slab at index L, which is why the slice is [0, L). Both faces
    come from the same sample, so the wrap never joins two samples.
    """
    lo: list = [slice(None)] * lab.ndim
    lo[1] = slice(0, L)
    hi = list(lo)
    lo[axis + 1] = 0
    hi[axis + 1] = L - 1
    return lab[tuple(lo)], lab[tuple(hi)]


def _wrap_roots(lab: np.ndarray, L: int, dim: int) -> np.ndarray:
    """label -> canonical label after joining what the torus joins.

    Union-find over LABELS (a few percent as many as there are sites),
    vectorized by pointer jumping: each pass pushes every pair's minimum across
    every periodic face in both directions and then squares the pointer array
    (`root[root]`), so a chain of length C resolves in O(log C) passes.
    `root[x] <= x` is an invariant, which makes the loop terminate at the
    component's smallest label.

    The termination test is on the WHOLE pass -- `root` unchanged after the
    unions AND the squaring. Testing only the squaring exits as soon as the
    pointer array is flat, which can happen on a pass where the union step
    still moved something: it returned 2 clusters where a flood fill finds 1 on
    a 32x32 torus (models/percolation_tau.py, 4 samples in 300). With `dim`
    directions merged at once the chains are longer still.
    """
    nlab = int(lab.max()) + 1
    root = np.arange(nlab, dtype=lab.dtype)
    a_parts, b_parts = [], []
    for axis in range(dim):
        u, v = _wrap_faces(lab, L, axis)
        both = (u > 0) & (v > 0)
        if both.any():
            a_parts.append(u[both])
            b_parts.append(v[both])
    if not a_parts:
        return root
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for _ in range(_MAX_MERGE_PASSES):
        before = root.copy()
        np.minimum.at(root, hi, root[lo])
        np.minimum.at(root, lo, root[hi])
        root = root[root]
        if np.array_equal(before, root):
            return root
    raise RuntimeError(
        f"the periodic-boundary label merge did not converge in "
        f"{_MAX_MERGE_PASSES} passes ({nlab} labels, {a.size} wrap edges, "
        f"dim {dim}). This should be unreachable -- see _MAX_MERGE_PASSES.")


def _roots(lab: np.ndarray, L: int, dim: int, geometry: str) -> np.ndarray:
    """label -> canonical label for this geometry (identity for a box)."""
    if geometry == "box":
        return np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(lab, L, dim)


def _sizes_and_owner(lab: np.ndarray, L: int,
                     roots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-label cluster size and per-label sample index, for a whole block.

    Sizes are accumulated per label and re-accumulated per root -- two cheap
    operations on something label-sized -- rather than by relabelling the
    lattice. Non-root labels come back with size 0, which is what makes them
    invisible to every size window (all windows start at s >= 1).

    The label -> sample map is a single scatter over the sites, well defined
    precisely because a cluster never spans two samples: the separator slabs
    block the stack direction and the wrap joins a sample only to itself. It
    assumes nothing about the ORDER ndimage.label hands out its labels.

    Split out so the one-window path (`simulate`) and the every-window path
    (`binned_counts`) share it: the passes over the lattice happen once per
    block, not once per window, which is the whole economy of the
    shared-lattice sampler.
    """
    rows = lab.shape[0]
    by_label = np.bincount(lab.ravel(), minlength=roots.size)
    size = np.bincount(roots, weights=by_label, minlength=roots.size)
    size[0] = 0.0                     # label 0 is "closed", and the separators

    flat = lab[:, :L].reshape(rows, -1)
    owner = np.zeros(roots.size, dtype=np.int64)
    owner[flat] = np.broadcast_to(np.arange(rows, dtype=np.int64)[:, None],
                                  flat.shape)
    return size, owner


def _count_window(size: np.ndarray, owner: np.ndarray, rows: int,
                  s_lo: int, s_hi: int | None) -> np.ndarray:
    """Clusters per sample with size in [s_lo, s_hi) (s_hi None = no cap)."""
    in_range = size >= s_lo
    if s_hi is not None:
        in_range = in_range & (size < s_hi)
    sel = np.nonzero(in_range)[0]
    if sel.size == 0:
        return np.zeros(rows, dtype=np.int64)
    return np.bincount(owner[sel], minlength=rows)


def percolation_tau_zd(
    s: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    observable: str = "bin",
    bin_ratio: float = DEFAULT_BIN_RATIO,
    box_factor: float | None = None,
    box_exponent: float | None = None,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_s: clusters at size scale `s`, per lattice site.

    Each sample is one fresh L(s)**dim critical lattice, union-found into
    clusters; the sample's value is the number of its clusters whose size is
    in [s, bin_ratio*s) (observable="bin") or >= s (observable="tail"),
    DIVIDED BY L(s)**dim. Both have E Y_s ~ a0 * s**(1-tau).

    `box_factor` and `box_exponent` default to the dimension-aware values
    (`default_box_factor`, `default_box_exponent`); `p` to the tabulated p_c.
    `rng` defaults to a fresh unseeded Generator. `block_n` defaults to a size
    derived from a fixed byte budget; results are bit-identical for any block
    size at the same seed, because the blocking is over the leading (sample)
    axis, and neither `observable` nor `geometry` touches the RNG.
    """
    if observable not in OBSERVABLES:
        raise ValueError(f"unknown observable {observable!r}; known: {list(OBSERVABLES)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
    p = critical_p(dim) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if box_factor is None:
        box_factor = default_box_factor(dim)
    if box_exponent is None:
        box_exponent = default_box_exponent(dim)

    s = int(s)
    L = box_side(s, box_factor, box_exponent)  # validates s, box_factor, exponent
    if L ** dim < s:
        raise ValueError(
            f"box rule gives L = {L} at s = {s}, dim = {dim}, so a cluster of "
            f"size {s} does not fit in the box at all and Y_s is identically "
            f"0; raise box_factor (currently {box_factor}) or box_exponent "
            f"(currently {box_exponent})")
    _check_size(L, dim)

    s_hi: int | None = None
    if observable == "bin":
        if bin_ratio <= 1.0:
            raise ValueError(f"bin_ratio must be > 1; got {bin_ratio}")
        s_hi = int(math.floor(bin_ratio * s + 0.5))
        if s_hi <= s:
            raise ValueError(
                f"bin [s, bin_ratio*s) is empty at s = {s}, bin_ratio = "
                f"{bin_ratio}: upper edge rounds to {s_hi}")

    rng = rng if rng is not None else np.random.default_rng()
    structure = _structure(dim)
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES
                      // (_BYTES_PER_SITE * L ** dim))
    block_n = min(block_n, n)

    out = np.empty(n, dtype=np.float64)
    volume = float(L) ** dim
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, L, dim, p), structure)
        size, owner = _sizes_and_owner(lab, L, _roots(lab, L, dim, geometry))
        out[offset:offset + rows] = _count_window(size, owner, rows,
                                                  s, s_hi) / volume
        offset += rows
        del lab, size, owner
    return out


def simulate(s: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_tau_zd"].simulate.

    params: {"dim": int (default 2), "p": float (default the tabulated p_c),
             "observable": "bin"|"tail", "bin_ratio": float,
             "box_factor": float, "box_exponent": float,
             "geometry": "torus"|"box"}.

    Deliberately does NOT assert Assumption 2 (Y_s > 0): the count of clusters
    at one size scale in one box is a small near-Poisson number, so a fraction
    ~exp(-lambda) of draws are 0 by design. `zero_fraction` reports it from a
    finished sample; `box_factor` is the knob that sets it.
    """
    dim = int(params.get("dim", 2))
    box_factor, box_exponent = _box_rule(params, dim)
    return percolation_tau_zd(
        s,
        n=n,
        dim=dim,
        p=params.get("p"),
        observable=params.get("observable", "bin"),
        bin_ratio=float(params.get("bin_ratio", DEFAULT_BIN_RATIO)),
        box_factor=box_factor,
        box_exponent=box_exponent,
        geometry=params.get("geometry", "torus"),
        rng=rng,
    )


def cost_hint(s: int, params: dict | None = None) -> float:
    """Work for one sample: L(s)**dim sites, exactly.

    Both halves of the pipeline are linear in the number of sites -- L**dim
    uniforms drawn, one near-linear union-find pass over L**dim sites -- and
    nothing here depends on p, on the observable or on the geometry: the whole
    lattice is labelled either way, and the torus adds one merge over
    something label-sized, a constant factor which cancels in an allocation
    since only RATIOS across scales reach one.

    So d = dim * box_exponent, which at the dimension-aware default
    box_exponent = 1/d_f is d = dim/d_f = tau - 1 = -gamma -- the article's
    cost exponent and the exponent being measured become the same number. The
    `ceil` in `box_side` makes cost(s) only ASYMPTOTICALLY a pure power; the
    wobble is a relative 1/L and is absorbed by the OLS-on-logs fit that
    tools/cost_model.py's `declared_exponent` runs over the recipe's ladder.

    params keys read: "dim", "box_factor", "box_exponent".

    THE UNIT TRAP, dimension-free by construction. Only ratios across scales
    reach an allocation, so the unit of this function is free -- but a recipe's
    `budget` is NOT in these units: tools/allocation.py charges cost(i) = i**d
    with the SCALE itself. One budget unit is box_factor**dim lattice sites,
    which `default_box_factor` deliberately holds at SITES_PER_BUDGET_UNIT =
    256 in every dimension, so "ask for S/256 to spend S sites" is the rule at
    dim = 2 and at dim = 6 alike. Both directions of that conversion go through
    tools/cost_model.cost_unit_ratio, which is what src/study/plan.py bisects
    on; getting it wrong once predicted 740 s for a 42-hour run.
    """
    params = params or {}
    dim = _check_dim(params.get("dim", 2))
    box_factor, box_exponent = _box_rule(params, dim)
    return float(box_side(int(s), box_factor, box_exponent)) ** dim


# ---------------------------------------------------------------------------
# The shared-lattice sampler: one box, every rung
# ---------------------------------------------------------------------------
#
# `percolation_tau_zd` above draws a fresh box FOR EACH rung, which is what
# makes the rungs independent and the article's CLT apply to them verbatim. It
# is also what makes it expensive: the ladder costs sum_k n_k L(s_k)**dim, and
# L**dim grows faster the higher dim is.
#
# One lattice already contains clusters of every size below its own cutoff, so
# a single box sized by the TOP rung can serve the whole ladder,
#
#     L = ceil( (s_top_edge / cut_fraction) ** (1/df_lower) ),
#
# for a total of n * L**dim. In 2-D that was measured 3.2x cheaper AND closer
# to tau at every m0 (experiments/03_percolation_zd/README.md).
#
# What is paid for it is independence: the rungs come from the same lattices,
# so Cov(Ybar_s, Ybar_s') != 0 and the CLT of eq. (583) does not apply as
# written. That is the experiment (user, 2026-09-05). Every artifact drawn this
# way is stamped `shared_lattice` in its metadata so no later reader mistakes
# it for an independent run.


def shared_box_side(s_top_edge: int, df_low: float,
                    cut_fraction: float = DEFAULT_CUT_FRACTION) -> int:
    """Smallest box that can serve a ladder whose top bin ends at `s_top_edge`.

    Inverts s <= cut_fraction * L**df_low. Dimension enters only through
    `df_low`: the box is sized by the largest cluster size the ladder asks
    about, and `cut_fraction` is the margin that keeps that top bin out of the
    cutoff.
    """
    if s_top_edge < 1:
        raise ValueError(f"s_top_edge must be >= 1; got {s_top_edge}")
    if not 0.0 < cut_fraction <= 1.0:
        raise ValueError(f"cut_fraction must be in (0, 1]; got {cut_fraction}")
    if df_low <= 0:
        raise ValueError(f"df_lower must be > 0; got {df_low}")
    return max(1, int(math.ceil((s_top_edge / cut_fraction) ** (1.0 / df_low)
                                - _CEIL_TOL)))


def bin_edges(scales, bin_ratio: float = DEFAULT_BIN_RATIO) -> list[tuple[int, int]]:
    """[(s, upper edge)] for each rung, the same windows `simulate` uses."""
    out = []
    for s in scales:
        s = int(s)
        if s < 1:
            raise ValueError(f"cluster size s must be >= 1; got {s}")
        hi = int(math.floor(bin_ratio * s + 0.5))
        if hi <= s:
            raise ValueError(
                f"bin [s, bin_ratio*s) is empty at s = {s}, bin_ratio = "
                f"{bin_ratio}: upper edge rounds to {hi}")
        out.append((s, hi))
    return out


def binned_counts(
    L: int,
    n: int,
    windows,
    dim: int = 2,
    p: float | None = None,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """(n, len(windows)) counts: every window read off the SAME n lattices.

    `windows` is [(s_lo, s_hi or None)], one per rung. Each row is one L**dim
    lattice, labelled once; the row's entries are that lattice's cluster counts
    in each window, so entries in one row are correlated by construction and
    rows are i.i.d.

    Identical, window by window, to calling the per-rung path on the same
    lattices -- the block draws the same uniforms in the same order and the
    counting shares `_sizes_and_owner`. That identity is a test, and it is what
    makes the cheap sampler a change of BUDGET rather than of observable.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
    p = critical_p(dim) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if L < 1:
        raise ValueError(f"box side must be >= 1; got {L}")
    _check_size(L, dim)
    windows = [(int(lo), None if hi is None else int(hi)) for lo, hi in windows]
    if not windows:
        raise ValueError("windows must be non-empty")

    rng = rng if rng is not None else np.random.default_rng()
    structure = _structure(dim)
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES
                      // (_BYTES_PER_SITE * L ** dim))
    block_n = min(block_n, n)

    out = np.empty((n, len(windows)), dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, L, dim, p), structure)
        size, owner = _sizes_and_owner(lab, L, _roots(lab, L, dim, geometry))
        for j, (s_lo, s_hi) in enumerate(windows):
            out[offset:offset + rows, j] = _count_window(size, owner, rows,
                                                         s_lo, s_hi)
        offset += rows
        del lab, size, owner
    return out


def shared_sampler(scales, n: int, params: dict,
                   rng: np.random.Generator) -> tuple[dict, dict]:
    """MODELS["percolation_tau_zd"].shared_sampler -- the ladder, one box.

    Returns ({scale: array of n values of Y_s}, info), where `info` records the
    box side, the dimension, the two design constants behind the box, and the
    site count -- everything a reader needs to tell this run from an
    independent one.

    Y_s is normalized exactly as `simulate`'s is (count / L**dim), so the two
    samplers produce the SAME observable and their gamma-hats are directly
    comparable; only the joint law across rungs differs.

    params: the same keys `simulate` reads (dim, p, bin_ratio, geometry -- the
    box-rule keys are ignored, the box comes from the ladder here), plus
    "df_lower" and "cut_fraction".
    """
    scales = [int(s) for s in scales]
    if sorted(scales) != scales or len(set(scales)) != len(scales):
        raise ValueError(f"scales must be strictly increasing; got {scales}")
    dim = int(params.get("dim", 2))
    windows = bin_edges(scales, float(params.get("bin_ratio", DEFAULT_BIN_RATIO)))
    df_low = float(params["df_lower"]) if params.get("df_lower") is not None \
        else df_lower(dim)
    cut_fraction = float(params.get("cut_fraction", DEFAULT_CUT_FRACTION))
    L = shared_box_side(windows[-1][1], df_low, cut_fraction)

    counts = binned_counts(L, n, windows, dim=dim, p=params.get("p"),
                           geometry=params.get("geometry", "torus"), rng=rng)
    volume = float(L) ** dim
    samples = {s: counts[:, j] / volume for j, s in enumerate(scales)}
    info = {
        "shared_lattice": True,
        "dim": dim,
        "L": L,
        "df_lower": df_low,
        "cut_fraction": cut_fraction,
        "n_lattices": int(n),
        "sites": int(n) * L ** dim,
        "windows": [list(w) for w in windows],
        "cut_ratio": [w[1] / L ** df_low for w in windows],
    }
    return samples, info


def zero_fraction(draws: np.ndarray) -> float:
    """Fraction of samples with Y_s = 0 -- the Assumption 2 diagnostic.

    Not a closed form: the count of clusters at one size scale has no simple
    exact law. It is ~exp(-lambda) for the mean count lambda per box, i.e. it
    is `box_factor` that controls it -- and since lambda ~ box_factor**dim * C
    with a dimension-dependent amplitude C, the value of box_factor that keeps
    it near 4% has to be READ OFF A PILOT in each new dimension rather than
    carried over. That is what this function is for.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.size == 0:
        raise ValueError("zero_fraction of an empty sample")
    return float((draws == 0.0).mean())
