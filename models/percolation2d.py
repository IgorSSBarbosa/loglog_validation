"""Critical site percolation on the 2-D square lattice -- registered as
MODELS["percolation2d"] in tools/models.py.

The observable, and why it is this one
--------------------------------------
One sample at scale `i` is: fill an `i x i` box with i.i.d. Bernoulli(p) open
sites at p = p_c, then count the open sites reachable from the SOUTH SIDE of
the box by open nearest-neighbour paths that stay inside the box,

    Y_i = #{ x in [0,i)^2 : x open and x <-> row 0 inside the box }.

This is PLAN.md ground rule 7's observable ("side-connected cluster, not
origin cluster"), and this module is the first place it exists in code. The
alternative -- the cluster of the box's centre site, which is what
`presentation18-05-2026/coding` measured as V(r) -- is still reachable here
as `params["anchor"] = "origin"`, but ONLY so the two can be compared head to
head in one experiment (experiments/03_percolation_zd/, Experiment P2). It is
not the default and is not the observable this rung validates.

Both anchors have the SAME leading exponent, so the comparison is about the
noise, not about the answer:

    E Y_i  ~  a0 * i**gamma,   gamma = d_f = 91/48 = 1.8958333...

for "south", because a site at height h above the south side connects to it
with probability of the order of the one-arm probability pi_1(h) ~ h**(-5/48)
(reaching distance h in the bulk and then hitting a full line is a constant-
probability event given the arm, by RSW), so

    E Y_i ~ sum_{h=1}^{i} i * h**(-5/48) ~ i * i**(43/48) = i**(91/48);

and for "origin", because E|C(0) cap B_i| ~ i**2 * pi_1(i) ~ i**(91/48) by the
same exponent. The sum for "south" is dominated by h ~ i, i.e. by sites deep
in the bulk -- the exponent is the bulk d_f, not a surface exponent.

Where they differ is the article's Assumption 6 (eq. 332), sigma_k^2 ->
sigma_inf^2, which the whole CLT (eq. 583) and the Wilson interval (eq. 720)
rest on:

  - "origin": Y_i = 0 unless the centre is connected to distance ~i, which has
    probability pi_1(i) ~ i**(-5/48). Conditionally it is of order
    i**(91/48)/pi_1(i), so

        Var(xi_i) = E[Y_i^2]/E[Y_i]^2 - 1  ~  1/pi_1(i)  ~  i**(5/48)  ->  infinity.

    Assumption 6 FAILS: sigma_k^2 diverges (slowly, as rho**(5k/48)), and
    Assumption 2 (Y_i > 0) fails outright on most draws.
  - "south": Y_i is a sum over ~i boundary columns of contributions that
    decorrelate beyond the correlation length, and is bounded below by the
    open sites of row 0 itself. Its coefficient of variation is expected to be
    O(1) and scale-free, so Assumption 6 holds and Assumption 2 fails only on
    the event that row 0 is entirely closed, which has probability
    (1-p_c)**i -- 1.3e-7 at i = 32, and EXPONENTIALLY small, hence invisible
    to a power-law expansion. (`simulate` does not assert Y > 0 for that
    reason; it reports the rate instead, see `zero_rate`.)

That is the hypothesis this model exists to test, stated as numbers in
experiments/03_percolation_zd/README.md: same gamma, far smaller and far
better-behaved noise, hence a faster-converging gamma-hat at equal budget.

No target_fn -- deliberately, as for models/srw.py
--------------------------------------------------
gamma = 91/48 is known (Stauffer & Aharony; den Nijs / Nienhuis) but is NOT
wired into MODELS["percolation2d"], for the same reason models/srw.py keeps
E|S_k| out of the code path (user's decision 2026-08-20, plans/
three_experiment_ladder.md D1/D2): the estimators must never be handed the
answer they are supposed to measure. 91/48 is written down as an acceptance
criterion in experiments/03_percolation_zd/README.md and checked by hand
against a finished run. The correction exponent omega_1 is NOT known here in
the way it is for srw -- the literature value for 2-D percolation is
Omega = 72/91 ~ 0.791 (Ziff 2011; Aharony & Asikainen 2003), with an analytic
correction of exponent 1 also expected from the box boundary -- so omega_1 is
something this rung MEASURES, not something it checks against a certainty.

Box or cylinder
---------------
`params["geometry"]` chooses the boundary condition in x: "box" (the default,
four walls) or "cylinder" (periodic in x, so only the south and north faces
are boundaries). The observable and the exponent are the same either way --
the argument above never used the side walls -- but the east/west walls are
pure finite-size contamination for a SOUTH-anchored count, and removing them
removes most of the correction-to-scaling term. Measured on two matched
4e9-work-unit runs (experiments/03_percolation_zd/, P4):

    Ybar_i / i**(91/48),  i = 8 .. 512
      box       0.4989 0.4842 0.4742 0.4695 0.4651 0.4660 0.4661   -6.6%
      cylinder  0.5578 0.5611 0.5633 0.5640 0.5671 0.5682 0.5698   +2.1%

The cylinder does NOT remove the correction -- the amplitude still moves, by
2.1% over six doublings against the box's 6.6%, and with the opposite sign.
What it does is make it about 3x smaller, which is enough to change the
character of the estimate: over R = 12 replicates at a common budget the
cylinder's m0 = 2 gamma-hat is the first cell in this project whose bias is
SMALLER than its spread, i.e. variance-limited rather than bias-limited, so
more budget starts helping again. It is also quieter (cv ~ 0.37 against
~ 0.43). A residual +0.004 bias in gamma-hat remains and does not decay with
m0 over 8 <= i <= 512; see the experiment README.

The wrap costs one merge of the labels joined across the x-boundary. That is
done with a vectorized pointer-jumping union-find over LABELS (a few percent
of the number of sites), not over sites, so the extra work is a few passes
over something small -- see `_wrap_roots`.

cost(i) = i**d with d = 2, and it is a fact rather than an assumption
---------------------------------------------------------------------
This is the first model in the repo where article Assumption 7's cost(i)=i**d
is a geometric claim about a real simulation: one sample draws i**2 uniforms
and runs one union-find pass over i**2 sites. `cost_hint` therefore returns
i**2 exactly, and the wall clock has to agree -- see
experiments/03_percolation_zd/README.md's cost-probe criterion. Contrast
models/synthetic.py (d = 0, the scale enters the formula rather than the work)
and models/srw.py (d = 1, Theta(k) steps per sample).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Critical probability for site percolation on the square lattice with
#: nearest-neighbour (4-)connectivity. Jacobsen (2015),
#: p_c = 0.59274605079210(2) -- 14 digits, far beyond what any box size here
#: can resolve (see `_draw_open` on the float32 rounding, which is the
#: coarser of the two by six orders of magnitude).
P_C_SQUARE_SITE = 0.59274605079210

#: 4-connectivity: north/south/east/west only. This is the connectivity
#: P_C_SQUARE_SITE is the threshold FOR -- 8-connectivity (the matching
#: lattice) has p_c = 0.407..., so passing the wrong structure here would
#: silently simulate a supercritical system.
_FOUR_CONNECTED = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)

ANCHORS = ("south", "origin")
GEOMETRIES = ("box", "cylinder")

#: Safety cap on `_wrap_roots`' pointer-jumping loop. The loop provably
#: terminates (every entry of `root` is non-increasing and bounded below by 0,
#: so it reaches a fixed point in finite steps) and doubles its reach each
#: pass, so 64 is astronomically more than any real block needs. It raises
#: rather than breaking out: a merge that has not converged returns silently
#: wrong cluster sizes, which is exactly the class of bug this repo exists to
#: not have.
_MAX_MERGE_PASSES = 64

#: Working-set budget for one (block_n, i) draw, in bytes, so block_n =
#: budget // (bytes_per_site * i**2) bounds the transient arrays regardless of
#: how large n gets -- the same guard models/srw.py uses, and needed for the
#: same reason: an allocation rule hands the smallest scales enormous n.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: Per site, per sample, across the whole pipeline: 4 bytes for the float32
#: uniform, 1 for the boolean padded lattice, 4 for the int32 label image, 1
#: for the boolean keep-mask. Rounded up, and ignoring the (i+1)/i separator
#: overhead, which is under 13% for every i >= 8.
_BYTES_PER_SITE = 10


def _draw_open(rng: np.random.Generator, rows: int, i: int, p: float) -> np.ndarray:
    """`rows` independent i x i boolean lattices, each site open w.p. p.

    float32 uniforms, for the reasons models/srw.py's `_draw_heads` gives:
    one draw per site with no bit packing, so splitting `rows` into sequential
    blocks consumes the RNG stream exactly as one unblocked call would, and
    the output is bit-identical at any block size. `rng.integers(0, 2,
    dtype=np.int8)`-style packing does NOT have that property.

    The cost is a realized open probability differing from `p` by at most
    2**-24 ~ 6e-8. At p_c that is a distance to criticality whose correlation
    length is xi ~ |p - p_c|**(-4/3) ~ 4e9 sites -- seven orders of magnitude
    beyond the largest box this model will ever be run at, so the lattice is
    critical to every scale that matters here.
    """
    return rng.random(size=(rows, i, i), dtype=np.float32) < p


def _label_block(open_grid: np.ndarray) -> np.ndarray:
    """Label each sample's open clusters, vectorized over the whole block.

    The trick: rather than calling ndimage.label once per sample (which for
    small i is dominated by Python/SciPy call overhead -- an allocation rule
    can ask for millions of samples at i = 8), stack the block's lattices
    vertically into ONE tall image with a blank separator row after each, and
    label that image once. No open path can cross a closed row, so a single
    2-D labelling of the stack is exactly the per-sample labelling, and the
    C-level union-find sees one large problem instead of n tiny ones.

    Returns an int32 array of shape (rows, i+1, i) -- a view of the labelled
    stack, with `[:, :i, :]` the sample and `[:, i, :]` its separator row
    (all label 0).
    """
    rows, i, _ = open_grid.shape
    padded = np.zeros((rows * (i + 1), i), dtype=bool)
    padded.reshape(rows, i + 1, i)[:, :i, :] = open_grid
    labels, _ = ndimage.label(padded, structure=_FOUR_CONNECTED)
    return labels.reshape(rows, i + 1, i)


def _wrap_roots(lab: np.ndarray, i: int) -> np.ndarray:
    """label -> canonical label, after joining what the periodic x-boundary joins.

    `ndimage.label` cannot wrap, so a cylinder is labelled as a box and the
    labels of column 0 and column i-1 are merged afterwards wherever both
    sites are open. The merge is a union-find run over LABELS, of which there
    are a few percent as many as there are sites, so it costs a few passes
    over something small rather than another pass over the lattice.

    Vectorized by pointer jumping rather than a Python `find` loop: each pass
    pushes every pair's minimum across the edge in both directions and then
    squares the pointer array (`root[root]`), so a chain of length L resolves
    in O(log L) passes. `root[x] <= x` is an invariant (both operations only
    ever assign a smaller value), which is what makes the loop terminate and
    makes the fixed point the component's smallest label.

    Merges never cross samples: the wrap joins column 0 to column i-1 of the
    SAME sample, so the block-contiguity of labels is preserved.
    """
    nlab = int(lab.max()) + 1
    root = np.arange(nlab, dtype=lab.dtype)
    left, right = lab[:, :i, 0], lab[:, :i, i - 1]
    both = (left > 0) & (right > 0)
    if not both.any():
        return root
    a, b = left[both], right[both]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for _ in range(_MAX_MERGE_PASSES):
        np.minimum.at(root, hi, root[lo])
        np.minimum.at(root, lo, root[hi])
        squared = root[root]
        if np.array_equal(squared, root):
            return root
        root = squared
    raise RuntimeError(
        f"the periodic-boundary label merge did not converge in "
        f"{_MAX_MERGE_PASSES} passes ({nlab} labels, {a.size} wrap edges). "
        f"This should be unreachable -- see _MAX_MERGE_PASSES.")


def _roots(lab: np.ndarray, i: int, geometry: str) -> np.ndarray:
    """label -> canonical label for this geometry (identity for a box)."""
    if geometry == "box":
        return np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(lab, i)


def _south_counts(lab: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """Open sites connected to row 0, per sample. `lab` is (rows, i+1, i).

    The keep-mask is built per ROOT and then mapped back to per LABEL, so the
    only full-size operation is the single `keep[lab]` gather a box already
    paid -- the cylinder adds no second pass over the lattice.
    """
    keep_root = np.zeros(roots.size, dtype=bool)
    keep_root[roots[lab[:, 0, :]]] = True   # every root touching a south row
    keep = keep_root[roots]
    keep[0] = False                   # label 0 is "closed", and the separators
    return keep[lab].sum(axis=(1, 2), dtype=np.int64)


def _origin_counts(lab: np.ndarray, i: int, roots: np.ndarray) -> np.ndarray:
    """Size of the cluster containing the box's centre site, per sample.

    0 when the centre is closed -- which is most of the time, and is the whole
    point of the comparison this anchor exists for (see the module docstring).

    Sizes are accumulated per label and then re-accumulated per root, both
    cheap array operations, rather than by relabelling the lattice.
    """
    by_label = np.bincount(lab.ravel(), minlength=roots.size)
    by_root = np.bincount(roots, weights=by_label, minlength=roots.size)
    centre = roots[lab[:, i // 2, i // 2]]
    return np.where(centre > 0, by_root[centre], 0).astype(np.int64)


def percolation2d(
    i: int,
    n: int = 1,
    p: float = P_C_SQUARE_SITE,
    anchor: str = "south",
    geometry: str = "box",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_i on an i x i critical site-percolation lattice.

    anchor="south" (the default, PLAN.md ground rule 7): open sites connected
    to the south side. anchor="origin": open sites in the cluster of the
    centre site -- for the head-to-head comparison only.

    geometry="box" (the default): four walls. geometry="cylinder": periodic in
    x, so only the south and north faces are boundaries. Same observable and
    same exponent; the cylinder simply removes two walls' worth of
    finite-size contamination (see the module docstring).

    `rng` defaults to a fresh unseeded Generator; pass a seeded one for
    reproducible runs. `block_n` defaults to a size derived from a fixed byte
    budget (_DEFAULT_WORKING_SET_BYTES); results are bit-identical for any
    block size at the same seed, because the blocking is over the leading
    (sample) axis (see `_draw_open`) -- and the geometry does not touch the
    RNG at all, so box and cylinder see the same lattices at the same seed.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; known: {list(ANCHORS)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")

    rng = rng if rng is not None else np.random.default_rng()
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_SITE * i * i))
    block_n = min(block_n, n)

    out = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, i, p))
        roots = _roots(lab, i, geometry)
        out[offset:offset + rows] = (
            _south_counts(lab, roots) if anchor == "south"
            else _origin_counts(lab, i, roots)
        )
        offset += rows
        del lab, roots
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation2d"].simulate.

    params: {"p": float (default p_c), "anchor": "south"|"origin",
             "geometry": "box"|"cylinder"}.

    Deliberately does NOT assert Y_i > 0 the way models/synthetic.py asserts
    Assumption 2. For "south" the violation is real but exponentially rare
    ((1-p_c)**i: 1.3e-7 at i = 32), and for "origin" it is the normal case --
    an assert would either fire once in ten million runs for no actionable
    reason, or fire immediately and make the comparison arm unrunnable. The
    rate is a measurable quantity instead; see `zero_rate`.
    """
    return percolation2d(
        i,
        n=n,
        p=float(params.get("p", P_C_SQUARE_SITE)),
        anchor=params.get("anchor", "south"),
        geometry=params.get("geometry", "box"),
        rng=rng,
    )


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample: i**2, so d = 2 exactly.

    Both halves of the pipeline are linear in the number of sites: i**2
    uniforms are drawn, and one union-find pass visits each site O(1) times
    (ndimage.label, near-linear with path compression). Nothing here depends
    on p, on the anchor, or on the geometry -- the full lattice is labelled
    either way.

    Exact by construction, like models/srw.py's `cost_hint`, which is what
    makes a MEASURED d scoreable against it rather than merely plausible.
    This is the first rung where article Assumption 7's cost(i) = i**d is a
    geometric fact about a real simulator (PLAN.md, ladder step 4).

    The cylinder's wrap-merge is deliberately NOT added here. It is O(number
    of labels) = O(i**2) with a small constant, so it changes cost(i) by a
    factor that is very nearly constant in i (measured 1.14x at i = 32 falling
    to 1.08x at i = 512) -- and only RATIOS across scales reach an allocation,
    so a constant factor cancels exactly. Folding it in would make the
    declared d depend on the geometry while the true d does not.
    """
    return float(i) ** 2


def zero_rate(i: int, p: float = P_C_SQUARE_SITE) -> float:
    """P(Y_i = 0) for anchor="south": exactly (1-p)**i, row 0 entirely closed.

    Independent of the geometry: wrapping x joins clusters but opens no site,
    and an open site of row 0 is connected to the south face either way.

    Exact, not asymptotic: Y_i = 0 iff no site of row 0 is open, since any
    open site of row 0 is itself connected to the south side. Used in
    experiments/03_percolation_zd/README.md's Assumption-2 criterion, and
    checked against Monte Carlo in tools/tests/test_percolation2d.py.
    """
    return float(1.0 - p) ** i
