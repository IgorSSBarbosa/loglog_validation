"""Critical site percolation on the hypercubic lattice Z^d, for a spatial
dimension `dim` that is a MODEL PARAMETER -- registered as
MODELS["percolation_zd"] in tools/models.py.

This is models/percolation2d.py generalized: every algorithm there (the
blocked float32 draw, the one-`ndimage.label`-per-block stacking, the
vectorized pointer-jumping merge across periodic boundaries, the per-root
keep-mask, the exact zero rate, the declared cost) is written here for an
arbitrary number of axes, and `dim` moves from the filename into `params`.
At `dim = 2` it reproduces models/percolation2d.py BIT FOR BIT at the same
seed, which is a test (`test_matches_percolation2d_bit_for_bit`), not a hope.

Why generalizing is worth doing
-------------------------------
`cost(i) = i**d` is article Assumption 7, and in models/percolation2d.py the
exponent d = 2 is a geometric fact about the simulator rather than a stated
formula (PLAN.md ladder step 4). Here **the article's cost exponent d IS the
spatial dimension**: one sample draws i**dim uniforms and runs one union-find
pass over i**dim sites, so `cost_hint(i) = i**dim` exactly, and a single model
sweeps d = 2, 3, 4, 5, 6 by changing one number in a recipe. That is what
makes the budget-allocation theory testable across a range of d without a new
simulator per dimension -- PLAN.md's ladder step 4 asks for exactly this, and
experiments/03_percolation_zd/README.md's "Open" list names it.

The observable, and the one place ground rule 7 needs restating in d > 4
--------------------------------------------------------------------------
PLAN.md ground rule 7 fixes the observable: sites connected to a full FACE of
the box, not the cluster of the origin. In d dimensions the face is the slab
x_0 = 0, and

    Y_i = #{ x in [0,i)^dim : x open and x <-> slab x_0 = 0 inside the box }.

The 2-D derivation of gamma = d_f generalizes literally, and generalizing it
is what shows that it stops working:

    E Y_i  ~  sum_{h=1}^{i} i**(dim-1) * min(1, pi_1(h)),
    pi_1(h) ~ h**(-beta/nu),   beta/nu = dim - d_f,

because the i**(dim-1) sites at depth h reach the face with probability of the
order of the bulk one-arm probability (reach distance h, then hit a full
hyperplane with conditional probability bounded below). The sum is a geometric
series in disguise and has two regimes:

    beta/nu < 1  ->  dominated by h ~ i:   E Y_i ~ i**(dim - beta/nu) = i**d_f
    beta/nu > 1  ->  dominated by h = O(1): E Y_i ~ i**(dim - 1)

so, writing gamma_face for what the FACE anchor actually measures,

    gamma_face = max(d_f, dim - 1),

and beta/nu < 1 holds only up to dim = 4:

    dim         2        3        4        5        6+
    beta/nu   0.104    0.477    0.955     1.46      2
    d_f       1.896    2.523    3.045     3.54      4
    dim-1       1        2        3        4        5

**The face anchor measures d_f for dim <= 4 and the trivial surface exponent
dim - 1 for dim >= 5.** In dim = 5 it would report 4 where d_f = 3.54; in
dim = 6, 5 where d_f = 4. Ground rule 7 was written for a 2-D testbed and is
correct there; it is not a statement that survives to high dimension, and
finding that out is one of the things experiments/05_percolation_highd is for.
Note also that dim = 4 is nearly degenerate on its own terms -- d_f = 3.045
against dim - 1 = 3, two exponents 0.045 apart -- so the crossover is slow and
dim = 4 is expected to be hard even though it is on the good side of it.

`anchor = "face_far"`: the fix that keeps d_f in every dimension
----------------------------------------------------------------
The high-d failure above is a BOUNDARY-LAYER effect: when pi_1 is summable,
almost every face-connected site sits within O(1) of the face, and the count
degenerates into "open sites next to the face", of which there are ~i**(dim-1).
Discarding that layer restores the bulk exponent. `anchor = "face_far"` counts
only the sites in the FAR HALF of the box,

    Y_i = #{ x : x_0 >= floor(dim_side/2), x open, x <-> slab x_0 = 0 },

for which the same sum runs over h in [i/2, i) and gives, in every dimension,

    E Y_i ~ i**(dim-1) * i * i**(-beta/nu) = i**d_f.

It keeps the two properties that made the face anchor better than the origin
anchor in 2-D (models/percolation2d.py, experiments/03_percolation_zd P3): the
count is a sum over ~i**(dim-1) columns that decorrelate beyond the correlation
length, so Var(xi_i) stays O(1) and article Assumption 6 holds; and Y_i = 0 is
the event that no cluster crosses half the box, whose probability at p_c is a
CONSTANT bounded away from 0 and 1 rather than one tending to 1 (the origin
anchor's failure mode, Var(xi_i) ~ 1/pi_1(i) -> infinity).

So there are three anchors here and they answer different questions:

    "face"      ground rule 7's observable, verbatim. gamma = max(d_f, dim-1).
    "face_far"  the same, restricted to the far half. gamma = d_f in every dim.
    "origin"    the cluster of the centre site, gamma = d_f, Assumption 6
                fails; kept only as the comparison arm, as in 2-D.

None of the three is preferred in the code: `simulate` defaults to "face"
because that is what ground rule 7 says and what dim = 2 must reproduce, and
the choice between them is something the experiments MEASURE.

Geometry
--------
`params["geometry"]` picks which axes are periodic:

    "box"       none -- 2*dim walls, the direct generalization of the 2-D box.
    "cylinder"  every axis EXCEPT x_0 -- so the anchor face and its opposite
                are the only walls left. The generalization of the 2-D
                cylinder (periodic in x, walls at south and north), and for
                the same reason: for a face-anchored count the transverse
                walls are pure finite-size contamination, worth 3.7x in RMSE
                in 2-D at 1.04x the cost (experiments/03_percolation_zd P4).
    "torus"     every axis. No boundary at all, so no face is a boundary
                either -- the anchor slab x_0 = 0 is still a well-defined set
                of sites and the count is still well defined, but the
                derivation above no longer applies to it. Meant for the
                "origin" anchor, and for the shared machinery that
                models/percolation_tau_zd.py needs.

The wrap costs one merge of the labels joined across each periodic face, done
by a vectorized pointer-jumping union-find over LABELS (a few percent as many
as there are sites), so it is a few passes over something small rather than
another pass over the lattice. Unlike models/percolation2d.py's single wrap
direction, a dim-dimensional torus joins up to dim faces at once and the
resulting label chains are long, so the merge here uses the STRICT
whole-pass termination test that models/percolation_tau.py's had to introduce
-- see `_wrap_roots`.

p_c is a literature input, and the one real new risk
-----------------------------------------------------
In 2-D, p_c is known to 14 digits and nothing about a lattice this code can
build resolves the uncertainty. In higher dimensions the thresholds are
still known to 8-10 digits (Mertens & Moore 2018), which is likewise far
beyond reach -- but they are DIFFERENT NUMBERS per dimension and a wrong entry
in the table simulates a systematically off-critical system, which is a bias
no estimator can see. `P_C_SITE_HYPERCUBIC` is that table, `P_C_SOURCE`
records where each entry comes from, and `crossing_fraction` is the
diagnostic that checks a threshold rather than trusting it: at p_c the
probability that some cluster spans the box is asymptotically independent of
i, and it drifts to 0 or 1 off criticality. That check is Experiment H0 in
experiments/05_percolation_highd/README.md, and it is the first thing to run
in a new dimension.

No target_fn -- deliberately, as for srw, percolation2d and percolation_tau
----------------------------------------------------------------------------
d_f(dim), tau(dim), beta/nu(dim) and the mean-field values d_f = 4, tau = 5/2
for dim >= 6 are all known, and are all kept OUT of the code path (user's
decision 2026-08-20; plans/three_experiment_ladder.md D1/D2). They are written
down as acceptance criteria in experiments/05_percolation_highd/README.md and
enter only as --expect-gamma / --truth at reporting time. The single exponent
this file states is `cost_hint`'s, which is a declared COST: it is an input to
the allocation and reaches no estimator of gamma.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Site-percolation thresholds on the hypercubic lattice Z^dim with
#: nearest-neighbour (2*dim-)connectivity -- the connectivity `_structure`
#: builds, and the one these numbers are the threshold FOR. Using a larger
#: neighbourhood (the "matching" lattice in 2-D, p_c = 0.407) with these values
#: silently simulates a supercritical system, which is why the structure is
#: built from `ndimage.generate_binary_structure(dim, 1)` and never from a
#: hand-written array.
#:
#: dim = 1 is exact and degenerate: an infinite open run needs every site, so
#: p_c = 1. It is in the table because it makes a closed-form test possible
#: (`expected_face_count_1d`), not because it is a percolation model.
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

#: Where each entry of P_C_SITE_HYPERCUBIC comes from. Recorded per dimension
#: because these are LITERATURE INPUTS, not measurements this repo makes: a
#: wrong entry is an off-critical simulation, i.e. a bias that no estimator
#: here can detect. Check a new dimension with `crossing_fraction` before
#: trusting it (Experiment H0).
P_C_SOURCE: dict[int, str] = {
    1: "exact (p_c = 1 in one dimension)",
    2: "Jacobsen (2015), 0.59274605079210(2)",
    3: "Xu, Wang, Lv & Deng (2014), 0.3116077(2)",
    4: "Mertens & Moore (2018), 0.19688561(3)",
    5: "Mertens & Moore (2018), 0.14079633(4)",
    6: "Mertens & Moore (2018), 0.109016661(8)",
    7: "Mertens & Moore (2018), 0.088951121(1)",
    8: "Mertens & Moore (2018), 0.075210128(1)",
    9: "Mertens & Moore (2018), 0.065209595(1)",
    10: "Mertens & Moore (2018), 0.0575929488(4)",
    11: "Mertens & Moore (2018), 0.0515896843(2)",
    12: "Mertens & Moore (2018), 0.0467309755(1)",
    13: "Mertens & Moore (2018), 0.0427150211(1)",
}

ANCHORS = ("face", "face_far", "origin")
GEOMETRIES = ("box", "cylinder", "torus")

#: "south" is models/percolation2d.py's name for the x_0 = 0 face. Accepted
#: here so a 2-D recipe reads the same against either model; "face" is the
#: name that means anything in dim != 2.
_ANCHOR_ALIASES = {"south": "face"}

#: Safety cap on `_wrap_roots`' pointer-jumping loop. The loop provably
#: terminates (every entry of `root` is non-increasing and bounded below by 0)
#: and doubles its reach each pass, so 64 is astronomically more than any real
#: block needs. It raises rather than breaking out: a merge that has not
#: converged returns silently wrong cluster sizes, which is exactly the class
#: of bug this repo exists to not have.
_MAX_MERGE_PASSES = 64

#: Working-set budget for one (block_n, i, dim) draw, in bytes, so
#: block_n = budget // (bytes_per_site * i**dim) bounds the transient arrays
#: however large n gets -- the same guard, and the same reason, as
#: models/srw.py and models/percolation2d.py: an allocation rule hands the
#: smallest scales enormous n. In high dim it binds at once: at dim = 5,
#: i = 32 one sample is already 33.5 M sites.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: Per site, per sample, across the whole pipeline: 4 bytes for the float32
#: uniform, 1 for the boolean padded lattice, 4 for the int32 label image, 1
#: for the boolean keep-mask. Rounded up, and ignoring the (i+1)/i separator
#: overhead, which is under 13% for every i >= 8.
_BYTES_PER_SITE = 10

#: Largest spatial dimension this file will build. `_structure` materializes a
#: dense 3**dim neighbourhood array, which is 1.6 M entries at dim = 13 and
#: 3.5 G at dim = 20 -- an unrecoverable allocation, from a typo, before any
#: lattice is drawn. 13 is also the extent of P_C_SITE_HYPERCUBIC, so nothing
#: reachable is lost.
_MAX_DIM = 13

#: `ndimage.label` returns int32 labels, so a lattice with 2**31 or more sites
#: cannot be labelled at all -- and one sample of it would need 8 GiB for the
#: label image alone. Refused up front with a message naming i and dim,
#: because the alternative is an opaque MemoryError or, worse, an overflow.
_MAX_SITES_PER_SAMPLE = 2 ** 31 - 1


def _check_dim(dim: int) -> int:
    """The one place a spatial dimension is range-checked."""
    dim = int(dim)
    if not 1 <= dim <= _MAX_DIM:
        raise ValueError(
            f"dim must be in [1, {_MAX_DIM}]; got {dim}. The upper bound is "
            f"where the dense 3**dim neighbourhood `_structure` builds stops "
            f"being allocatable, and is also the extent of the p_c table.")
    return dim


def _structure(dim: int) -> np.ndarray:
    """Nearest-neighbour (2*dim-)connectivity on Z^dim.

    `generate_binary_structure(dim, 1)` is the +/-1-along-one-axis
    neighbourhood: 4 neighbours in 2-D, 6 in 3-D, 2*dim in general. This is
    the connectivity P_C_SITE_HYPERCUBIC is tabulated for. Built rather than
    written out so that no dimension can silently get the wrong one -- at
    dim = 2 it equals models/percolation2d.py's `_FOUR_CONNECTED`, which is a
    test.
    """
    return ndimage.generate_binary_structure(dim, 1)


def _draw_open(rng: np.random.Generator, rows: int, i: int, dim: int,
               p: float) -> np.ndarray:
    """`rows` independent i**dim boolean lattices, each site open w.p. p.

    float32 uniforms, one draw per site, for the reason models/srw.py's
    `_draw_heads` gives: splitting `rows` into sequential blocks then consumes
    the RNG stream exactly as one unblocked call would, so the output is
    bit-identical at any `block_n`. Bit-packed integer draws are not, and
    8-bit ones cannot even hold p_c.

    The shape is (rows,) + (i,)*dim, which at dim = 2 is literally
    models/percolation2d.py's (rows, i, i) -- same call, same stream, same
    lattices.

    The realized open probability differs from `p` by at most 2**-24 ~ 6e-8.
    In 2-D that is a distance to criticality with correlation length ~4e9
    sites; in higher dimensions nu is SMALLER (0.876 in 3-D, 1/2 at and above
    6-D) so xi ~ |p-p_c|**(-nu) is smaller too, but 6e-8 still puts xi at
    10**4 sites even at nu = 1/2 -- orders beyond any box this model can
    allocate.
    """
    return rng.random(size=(rows,) + (i,) * dim, dtype=np.float32) < p


def _label_block(open_grid: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Label every sample's open clusters in ONE ndimage.label call.

    Exactly models/percolation2d.py's trick with one more index: the block's
    lattices are stacked along the FIRST SPATIAL AXIS into one tall image with
    a blank separator SLAB after each, and that image is labelled once. No
    open path can cross a closed slab, so labelling the stack is the
    per-sample labelling, and the C-level union-find sees one large problem
    instead of `rows` tiny ones -- which is what makes the cheapest rungs
    affordable when an allocation rule asks for millions of samples there.

    Returns int32 of shape (rows, i+1) + (i,)*(dim-1): `[:, :i]` is the sample
    and `[:, i]` its separator slab, all label 0.
    """
    rows = open_grid.shape[0]
    spatial = open_grid.shape[1:]
    i = spatial[0]
    stacked_shape = (rows * (i + 1),) + spatial[1:]
    padded = np.zeros(stacked_shape, dtype=bool)
    padded.reshape((rows, i + 1) + spatial[1:])[:, :i] = open_grid
    labels, _ = ndimage.label(padded, structure=structure)
    return labels.reshape((rows, i + 1) + spatial[1:])


def _wrap_faces(lab: np.ndarray, i: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """The two opposite faces that making spatial `axis` periodic identifies.

    `lab` is (rows, i+1) + (i,)*(dim-1), so spatial axis `a` is array axis
    `a+1`, and the stack axis (a = 0) is the one carrying the separator slab
    at index i -- which is why the slice is [0, i) and not [0, i]. Both faces
    are taken from the SAME sample, so the wrap never joins two samples.
    """
    lo: list = [slice(None)] * lab.ndim
    lo[1] = slice(0, i)
    hi = list(lo)
    lo[axis + 1] = 0
    hi[axis + 1] = i - 1
    return lab[tuple(lo)], lab[tuple(hi)]


def _periodic_axes(dim: int, geometry: str) -> tuple[int, ...]:
    """Which spatial axes this geometry makes periodic.

    "box" none; "cylinder" every axis but x_0, so the anchor face and its
    opposite stay walls; "torus" all of them. At dim = 2 the cylinder is
    axis 1 alone, i.e. models/percolation2d.py's periodic x.
    """
    if geometry == "box":
        return ()
    if geometry == "cylinder":
        return tuple(range(1, dim))
    return tuple(range(dim))


def _wrap_roots(lab: np.ndarray, i: int, axes: tuple[int, ...]) -> np.ndarray:
    """label -> canonical label, after joining what the periodic axes join.

    `ndimage.label` cannot wrap, so a periodic lattice is labelled as a box
    and the labels of the identified faces are merged afterwards. The merge is
    a union-find over LABELS -- a few percent as many as there are sites -- so
    it costs a few passes over something small rather than another pass over
    the lattice.

    Vectorized by pointer jumping rather than a Python `find` loop: each pass
    pushes every pair's minimum across every edge in both directions and then
    squares the pointer array (`root[root]`), so a chain of length C resolves
    in O(log C) passes. `root[x] <= x` is an invariant (both operations only
    ever assign a smaller value), which is what makes the loop terminate and
    makes the fixed point the component's smallest label.

    The termination test is on the WHOLE pass -- `root` unchanged after the
    unions AND the squaring -- not on the squaring alone. That distinction is
    not cosmetic here. models/percolation2d.py tests only the squaring and
    gets away with it because a cylinder has ONE wrap direction and its label
    chains are short; with up to `dim` directions joined at once the chains are
    long, and testing only the squaring exits on a pass where the union step
    still moved something, returning a PIECE of a cluster as if it were the
    cluster. models/percolation_tau.py hit that on a 32x32 torus (4 samples in
    300, found as a block_n invariance failure), which is where the strict
    test comes from.
    """
    nlab = int(lab.max()) + 1
    root = np.arange(nlab, dtype=lab.dtype)
    a_parts, b_parts = [], []
    for axis in axes:
        u, v = _wrap_faces(lab, i, axis)
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
        f"axes {list(axes)}). This should be unreachable -- see "
        f"_MAX_MERGE_PASSES.")


def _roots(lab: np.ndarray, i: int, dim: int, geometry: str) -> np.ndarray:
    """label -> canonical label for this geometry (identity for a box)."""
    axes = _periodic_axes(dim, geometry)
    if not axes:
        return np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(lab, i, axes)


def _anchor_keep(lab: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """Per-LABEL mask: does this label's cluster touch the slab x_0 = 0?

    Built per ROOT and mapped back per label, so the only full-size operation
    left to the caller is the single `keep[lab]` gather a box already pays --
    a periodic geometry adds no second pass over the lattice.
    """
    keep_root = np.zeros(roots.size, dtype=bool)
    keep_root[roots[lab[:, 0]]] = True     # every root touching the anchor slab
    keep = keep_root[roots]
    keep[0] = False                   # label 0 is "closed", and the separators
    return keep


def _face_counts(lab: np.ndarray, i: int, keep: np.ndarray,
                 depth_from: int) -> np.ndarray:
    """Open sites connected to slab x_0 = 0 with x_0 >= depth_from, per sample.

    `depth_from = 0` is the "face" anchor (ground rule 7's observable);
    `depth_from = i // 2` is "face_far", the far half only. The separator slab
    at index i is excluded by the slice -- it is label 0 and would be dropped
    by `keep` anyway, but slicing says so.
    """
    sel = lab[:, depth_from:i]
    return keep[sel].sum(axis=tuple(range(1, sel.ndim)), dtype=np.int64)


def _origin_counts(lab: np.ndarray, i: int, dim: int,
                   roots: np.ndarray) -> np.ndarray:
    """Size of the cluster containing the box's centre site, per sample.

    0 when the centre is closed -- which is most of the time, and is the whole
    point of the comparison this anchor exists for. Sizes are accumulated per
    label and then re-accumulated per root, both cheap array operations,
    rather than by relabelling the lattice.
    """
    by_label = np.bincount(lab.ravel(), minlength=roots.size)
    by_root = np.bincount(roots, weights=by_label, minlength=roots.size)
    centre = roots[lab[(slice(None), i // 2) + (i // 2,) * (dim - 1)]]
    return np.where(centre > 0, by_root[centre], 0).astype(np.int64)


def _resolve_anchor(anchor: str) -> str:
    anchor = _ANCHOR_ALIASES.get(anchor, anchor)
    if anchor not in ANCHORS:
        raise ValueError(
            f"unknown anchor {anchor!r}; known: {list(ANCHORS)} "
            f"(aliases: {sorted(_ANCHOR_ALIASES)})")
    return anchor


def critical_p(dim: int) -> float:
    """p_c on Z^dim, from the table -- raising, never guessing, if absent."""
    if dim not in P_C_SITE_HYPERCUBIC:
        raise ValueError(
            f"no tabulated p_c for dim = {dim}; known dimensions: "
            f"{sorted(P_C_SITE_HYPERCUBIC)}. Pass params['p'] explicitly, and "
            f"record where the value came from -- an off-critical p is a bias "
            f"no estimator in this repo can see.")
    return P_C_SITE_HYPERCUBIC[dim]


def _check_size(i: int, dim: int) -> int:
    """Sites in one padded sample, refusing what int32 labels cannot hold."""
    sites = (i + 1) * i ** (dim - 1)
    if sites > _MAX_SITES_PER_SAMPLE:
        raise ValueError(
            f"one sample at i = {i}, dim = {dim} is {sites:.3g} sites, past "
            f"ndimage.label's int32 label space ({_MAX_SITES_PER_SAMPLE}) and "
            f"{sites * _BYTES_PER_SITE / 2**30:.1f} GiB of working set. Lower "
            f"the top of the ladder: i**dim is the whole cost model here.")
    return sites


def percolation_zd(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    anchor: str = "face",
    geometry: str = "box",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_i on an i**dim critical site-percolation lattice.

    anchor="face" (the default, PLAN.md ground rule 7, alias "south"): open
    sites connected to the slab x_0 = 0. anchor="face_far": the same count
    restricted to x_0 >= i//2, which is what keeps gamma = d_f above dim = 4.
    anchor="origin": open sites in the cluster of the centre site -- the
    comparison arm only. See the module docstring for which measures what.

    geometry="box" (the default): 2*dim walls. "cylinder": periodic in every
    axis but x_0. "torus": periodic in all of them.

    `p` defaults to the tabulated p_c for `dim`. `rng` defaults to a fresh
    unseeded Generator; pass a seeded one for reproducible runs. `block_n`
    defaults to a size derived from a fixed byte budget
    (_DEFAULT_WORKING_SET_BYTES); results are bit-identical for any block size
    at the same seed, because the blocking is over the leading (sample) axis
    (see `_draw_open`) -- and neither the anchor nor the geometry touches the
    RNG, so all combinations see the same lattices at the same seed.
    """
    anchor = _resolve_anchor(anchor)
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

    rng = rng if rng is not None else np.random.default_rng()
    structure = _structure(dim)
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES
                      // (_BYTES_PER_SITE * i ** dim))
    block_n = min(block_n, n)

    depth_from = (i // 2) if anchor == "face_far" else 0
    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, i, dim, p), structure)
        roots = _roots(lab, i, dim, geometry)
        if anchor == "origin":
            out[offset:offset + rows] = _origin_counts(lab, i, dim, roots)
        else:
            out[offset:offset + rows] = _face_counts(
                lab, i, _anchor_keep(lab, roots), depth_from)
        offset += rows
        del lab, roots
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_zd"].simulate.

    params: {"dim": int (default 2), "p": float (default the tabulated p_c for
             that dim), "anchor": "face"|"face_far"|"origin" ("south" aliases
             "face"), "geometry": "box"|"cylinder"|"torus"}.

    Deliberately does NOT assert Y_i > 0 the way models/synthetic.py asserts
    Assumption 2, for models/percolation2d.py's reasons carried over: for
    "face" the violation is real but exponentially rare (`zero_rate` gives it
    exactly), for "face_far" it is an O(1) fraction by design, and for
    "origin" it is the normal case. An assert would either never fire or fire
    immediately, and in neither case say anything actionable. The rate is a
    measurable quantity instead.
    """
    return percolation_zd(
        i,
        n=n,
        dim=int(params.get("dim", 2)),
        p=params.get("p"),
        anchor=params.get("anchor", "face"),
        geometry=params.get("geometry", "box"),
        rng=rng,
    )


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample: i**dim, so the article's d IS the dimension.

    Both halves of the pipeline are linear in the number of sites: i**dim
    uniforms are drawn, and one union-find pass visits each site O(1) times
    (ndimage.label, near-linear with path compression). Nothing here depends
    on p, on the anchor, or on the geometry -- the full lattice is labelled
    either way, and a periodic wrap adds one merge over something label-sized,
    a nearly constant factor which cancels exactly in an allocation since only
    RATIOS across scales reach one.

    Exact by construction, like models/srw.py's and models/percolation2d.py's,
    which is what makes a MEASURED d scoreable against it rather than merely
    plausible. What is new here is that a single model covers a RANGE of d by
    a parameter: Assumption 7's cost exponent and the spatial dimension are
    the same number, so recipes at dim = 2..6 exercise the allocation theory
    at d = 2..6 without one simulator per dimension (PLAN.md, ladder step 4).

    params keys read: "dim" (nothing else).
    """
    return float(i) ** _check_dim((params or {}).get("dim", 2))


def zero_rate(i: int, dim: int = 2, p: float | None = None) -> float:
    """P(Y_i = 0) for anchor="face": exactly (1-p)**(i**(dim-1)).

    Y_i = 0 iff no site of the anchor slab x_0 = 0 is open, since any open
    site of that slab is itself connected to it. The slab holds i**(dim-1)
    sites, so the rate is exact rather than asymptotic -- and it collapses
    faster with dim than the 2-D (1-p)**i does, even though p_c is smaller:
    at dim = 3, i = 32 it is 0.686**1024 ~ 10**(-168).

    Independent of the geometry (wrapping joins clusters but opens no site)
    and NOT the rate for the other two anchors: "face_far" needs a crossing of
    half the box, an O(1) event with no closed form, and "origin" needs the
    centre site open at all.

    Used in experiments/05_percolation_highd/README.md's Assumption-2
    criterion, and checked against Monte Carlo in
    tools/tests/test_percolation_zd.py.
    """
    p = critical_p(int(dim)) if p is None else float(p)
    return float(1.0 - p) ** (int(i) ** (int(dim) - 1))


def expected_face_count_1d(i: int, p: float) -> float:
    """E Y_i for anchor="face" at dim = 1, exactly: sum_{k=1}^{i} p**k.

    In one dimension the face is the single site 0 and Y_i is the length of
    the open run starting there, so P(Y_i >= k) = p**k for k <= i and the mean
    is a finite geometric sum. Closed form, no asymptotics, any p -- which is
    what makes dim = 1 worth keeping in the table: it is an exact check on the
    whole stacking/labelling/counting path in a dimension where enumeration
    over 2**(i**dim) configurations is not the only option.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if p == 1.0:
        return float(i)
    return float(p * (1.0 - p ** int(i)) / (1.0 - p))


def crossing_fraction(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    geometry: str = "box",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> float:
    """Fraction of samples in which some cluster spans the box along x_0.

    THE CRITICALITY DIAGNOSTIC, and the reason it is in this file rather than
    in a test: p_c is a literature input here (see P_C_SOURCE), it is a
    different number in every dimension, and a wrong one simulates an
    off-critical system whose bias no estimator in this repo can see. At p_c
    the spanning probability is asymptotically INDEPENDENT of i; below it,
    it decays to 0; above it, it rises to 1. So running this over a ladder of
    i and looking for a flat column is a check on the table -- Experiment H0
    of experiments/05_percolation_highd/README.md -- and it costs one
    labelling per sample, the same work `simulate` already does.

    Not registered as a model observable: it is a probability, not a Y_i, and
    it has no power law for the estimator to fit.
    """
    dim = _check_dim(dim)
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    i = int(i)
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    p = critical_p(dim) if p is None else float(p)
    _check_size(i, dim)

    rng = rng if rng is not None else np.random.default_rng()
    structure = _structure(dim)
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES
                      // (_BYTES_PER_SITE * i ** dim))
    block_n = min(block_n, n)

    spanned = 0
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, i, dim, p), structure)
        roots = _roots(lab, i, dim, geometry)
        near = _anchor_keep(lab, roots)             # touches slab x_0 = 0
        far_root = np.zeros(roots.size, dtype=bool)
        far_root[roots[lab[:, i - 1]]] = True       # touches slab x_0 = i-1
        far = far_root[roots]
        far[0] = False
        both = near & far
        # A spanning cluster is a label kept by both masks; a sample spans iff
        # any of its sites carries one. One gather over the near half is
        # enough -- a spanning cluster necessarily has a site at x_0 = 0.
        spanned += int(both[lab[:, 0]].reshape(rows, -1).any(axis=1).sum())
        offset += rows
        del lab, roots, near, far, both
    return spanned / float(n)
