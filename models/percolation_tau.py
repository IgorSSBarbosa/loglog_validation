"""Cluster-number density of critical site percolation -- the Fisher exponent
tau -- registered as MODELS["percolation_tau"] in tools/models.py.

What is being measured, and why the scale variable is a CLUSTER SIZE
---------------------------------------------------------------------
At p = p_c the number of clusters of size s per lattice site decays as a pure
power (Stauffer & Aharony; Malthe-Sorenssen, "Percolation Theory Using
Python", eq. 4.4 and Fig. 4.2):

    n_s ~ C s**(-tau),      tau = 187/91 = 2.054945...  in d = 2.

The article's estimator (eq. 523-531) fits log E Y_i against log i on a
geometric ladder i = rho**k. To put tau in front of it, the ladder variable
must be the one the power law is IN -- so here the "scale" handed to
`simulate` is the CLUSTER SIZE s, not a box side, and

    Y_s = (number of clusters of size >= s in one box) / L(s)**2      "tail"
    Y_s = (number of clusters of size in [s, b s) in one box) / L(s)**2  "bin"

with L(s) the side of the box that sample is drawn on (below). Both have the
same exponent, because integrating C u**(-tau) from s to either b*s or the
cutoff gives a constant times s**(1-tau):

    E Y_s = a0 * s**gamma * (1 + corrections),      gamma = 1 - tau,

so the pipeline's gamma-hat converts as **tau = 1 - gamma_hat**, and in d = 2
the acceptance value is gamma = 1 - 187/91 = -96/91 = -1.054945...  (A
NEGATIVE gamma, the first in this repo: Y_s decays along the ladder. Nothing
in eq. (523)-(531) cares about the sign -- it is an OLS slope on logs -- but
it is worth knowing before reading a plot.)

Y is a per-site DENSITY, not a raw count: dividing by the exactly-known box
area L(s)**2 is what makes gamma = 1 - tau hold whatever box rule is in
force, so the reporting-time conversion tau = 1 - gamma carries no design
constant. It is the same normalization as the book's eq. (4.4) (which divides
by M, by L**d and, for the histogram, by the bin width ds). The bin width is
deliberately NOT divided out here: ds = (b-1)s is a power of the scale, and
dividing by it would shift the fitted exponent by exactly 1 inside the model
-- moving part of the answer into the simulator. `n(s) ~ s**(-tau)` is
recovered from `Y_s ~ s**(1-tau)` at reporting time instead.

The geometric ladder IS the logarithmic binning
-----------------------------------------------
Fig. 4.2's point is that a linear histogram of cluster sizes is useless in the
tail: large s is rare, so those bins hold 0 or 1 clusters and the log-log plot
scatters. The fix in the book is logarithmic binning, bin edges a**i. This
repo's ladder is already s_k = rho**k, so a ladder of powers of two with
`bin_ratio = 2` reproduces exactly the book's a = 2 binning -- one rung per
bin, tiling the s-axis with no gap and no overlap.

The difference from the book is that each rung is drawn from its OWN,
INDEPENDENT boxes (PLAN.md ground rule 2: no reuse of simulations across
configurations). One lattice does supply every bin at once, and reusing it is
what the book's Fig. 4.2 does, but then the rungs are correlated and the CLT
of eq. (583) -- which is what the error bars come from here -- no longer
applies to them. The samples are cheap; the correlation is not fixable
afterwards.

The finite-box cutoff, and why the box grows with s
---------------------------------------------------
The second failure mode of the naive measurement: in an L x L box no cluster
can have more than L**2 sites, and well before that, n_s(L) is cut off around
s ~ L**d_f. So a fixed box gives a power law that bends down and dies at the
top of the s-range -- the region a fixed-box measurement has to DISCARD (the
"upper half of s ~ L**2" this model was asked for).

Discarding it is exactly the wrong shape for this repo's estimator, which
drops the m0 SMALLEST rungs and keeps the largest. So instead the box is tied
to the rung:

    L(s) = ceil(box_factor * s**box_exponent),      cost(s) = L(s)**2 sites.

Every rung then sits at the same relative distance from its own cutoff, up to
the drift below, and no rung has to be thrown away: the top of the ladder is
as clean as the bottom. With the default box_exponent = 1/2 the cost exponent
is exactly d = 2*box_exponent = 1.

Two consequences worth stating as numbers, both measurable rather than
assumed:

  - **cutoff drift.** s / L**d_f = s**(1 - d_f*box_exponent) / box_factor**d_f,
    which for box_exponent = 1/2 grows as s**(1 - 91/96) = s**(5/96) -- 13%
    per decade of s. Setting box_exponent = 1/d_f = 48/91 = 0.527472... freezes
    it exactly, at the price of importing d_f as a DESIGN constant (the same
    category as `omega1` in an allocation rule: it decides the geometry, never
    reaches an estimator, and a wrong value would change the noise and the
    correction term but not the fitted tau).
  - **Assumption 6.** The mean count per box is
    lambda(s) = L**2 * C s**(1-tau) ~ box_factor**2 * C * s**(2*box_exponent+1-tau),
    and for a near-Poisson count Var(xi_s) ~ 1/lambda. At box_exponent = 1/2
    that drifts as s**(tau-2) = s**(5/91) -- slowly divergent, the same
    character (and nearly the same exponent) as the origin anchor's
    i**(5/48) in models/percolation2d.py. At box_exponent = 48/91 it is
    constant and Assumption 6 holds outright.

`box_factor` sets that constant, and it is nearly FREE: cost and count both
scale as box_factor**2, so cv * sqrt(cost) -- the precision a unit of budget
buys -- barely moves, while the mean count per box (hence the zero fraction)
grows as box_factor**2 and the distance to the cutoff as box_factor**d_f. Measured (bin observable, torus, box_exponent = 1/2):

    box_factor    lambda(64)   cv     zero fraction   cv*L(64)
        8            0.84      1.15       0.45           73.5
       16            3.28      0.58       0.04           73.6
       32           12.91      0.30       0.00           76.3

which is why the default is 16 rather than the 8 that first looked cheap: at
8, nearly half of every run's draws are 0 and carry no information about the
size scale they were drawn for, for no gain in precision per site. Above ~32
the flat trade starts to bend (the box is mostly measuring clusters far below
its own cutoff), so 16-32 is the useful range.

The fixed-box design this model was first asked for is still reachable, and
is the honest comparison arm: `box_exponent = 0` with `box_factor = L` draws every rung on
the same L x L box (then cost(s) is constant, d = 0, which is outside the
allocation formulas -- such a recipe must state its own `n` list), and the
ladder itself must stop well below L**d_f. `shared_sampler` below goes one
step further and lets the rungs share the LATTICES as well as the box, which
is the cheap version -- and, measurably, the more accurate one.

Boundaries: torus by default
----------------------------
A cluster touching the wall of an open box is TRUNCATED, so it is recorded at
a size smaller than its own. That does not merely lose clusters from a bin, it
FEEDS the bin from above -- and since n_s falls steeply, the influx from
larger clusters beats the outflow. Measured, paired on the same lattices (bin
observable, box_factor = 16, box_exponent = 1/2):

    s          16        64       256      1024
    box/torus  1.571     1.561    1.565    1.588

i.e. an open box reports ~57% MORE clusters in every bin than the torus does.
Because L(s) grows with s, the distortion is very nearly an amplitude (it
lands in a0, where it is harmless) rather than an exponent: the residual drift
over those six doublings is +0.0027 in gamma, about half the statistical error
of the pilot run. So the box is usable, but it is 57% of a0 and a correction
term bought for nothing.

`geometry = "torus"` (the default, unlike models/percolation2d.py, whose
SOUTH-anchored observable NEEDS a south wall) removes the boundary entirely:
periodic in both directions, so the only finite-size effect left is the cutoff
above. `geometry = "box"` keeps the four walls so the two can be run head to
head.

Which observable, measured
--------------------------
"bin" and "tail" have the same exponent and, at equal budget on the ladder
s = 8..2048 (4e9 sites), the same bias; they differ in noise and in Assumption
2:

    observable   cv over the ladder   zero fraction   se(gamma) at m0=3
    bin          0.578 - 0.617        0.042 - 0.061   0.0032
    tail         0.420 - 0.446        0.000           0.0023

"tail" is quieter by 1.4x (2x in budget) and has no zeros at all, since some
cluster of size >= s is essentially always present. "bin" is nevertheless the
default: it is the direct estimator of n_s itself -- one bin, one size scale --
whereas the tail integrates every scale above s, so a departure from the pure
power at one size scale is diluted in it rather than visible. It is also the
observable of the book's Fig. 4.2. Pick "tail" when the budget, not the shape
of n_s, is the binding constraint.

No target_fn -- deliberately, as for srw and percolation2d
-----------------------------------------------------------
tau = 187/91 and the hyperscaling relation tau = 1 + d/d_f are known, and are
kept OUT of the code path (user's decision 2026-08-20, plans/
three_experiment_ladder.md D1/D2): an estimator must never be handed the
answer it is measuring. They belong in the experiment README as acceptance
criteria and on the reporting-time `--expect-gamma` / `--truth` flags. The
only exponent this file states is `cost_hint`'s, which is a declared COST and
reaches no estimator of gamma.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

#: Critical probability for site percolation on the square lattice with
#: nearest-neighbour (4-)connectivity, Jacobsen (2015). Same constant, same
#: reason, as models/percolation2d.py -- the two models are deliberately not
#: allowed to import one another (models/README.md), so it is restated here.
P_C_SQUARE_SITE = 0.59274605079210

#: 4-connectivity: north/south/east/west only, the connectivity
#: P_C_SQUARE_SITE is the threshold FOR. The 8-connected lattice has
#: p_c = 0.407..., so the wrong `structure` here silently simulates a
#: supercritical system.
_FOUR_CONNECTED = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)

OBSERVABLES = ("bin", "tail")
GEOMETRIES = ("torus", "box")

#: Defaults for the box rule L(s) = ceil(box_factor * s**box_exponent).
#: box_exponent = 1/2 makes cost(s) = L**2 ~ s exactly (d = 1) and needs no
#: knowledge of d_f; see the module docstring for what 48/91 buys instead, and
#: for the measurement behind box_factor = 16 (it costs nothing in precision
#: per unit budget and takes the zero fraction from 45% to 4%).
DEFAULT_BOX_FACTOR = 16.0
DEFAULT_BOX_EXPONENT = 0.5

#: Bin upper edge, [s, bin_ratio*s), for observable = "bin". 2.0 tiles a
#: powers-of-two ladder exactly -- the book's a = 2 logarithmic binning.
DEFAULT_BIN_RATIO = 2.0

#: Safety cap on the pointer-jumping loop in `_wrap_roots`; it doubles its
#: reach each pass and provably terminates, so this only ever fires on a bug.
_MAX_MERGE_PASSES = 64

#: Working-set budget for one (block_n, L) draw, in bytes: block_n is chosen
#: so the transient arrays stay inside it however large n gets, the same guard
#: (and the same reason) as models/srw.py and models/percolation2d.py -- an
#: allocation rule hands the cheapest rungs enormous n.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: Per site, per sample: 4 bytes for the float32 uniform, 1 for the boolean
#: padded lattice, 4 for the int32 label image. Rounded up, ignoring the
#: (L+1)/L separator overhead (under 13% for every L >= 8).
_BYTES_PER_SITE = 10

#: ceil() with a tolerance, so that box_factor * s**box_exponent landing on an
#: integer from below (8 * 16**0.5 = 32.000000000000004 on some libms) does not
#: cost a whole extra row and column.
_CEIL_TOL = 1e-9


def box_side(s: int, box_factor: float = DEFAULT_BOX_FACTOR,
             box_exponent: float = DEFAULT_BOX_EXPONENT) -> int:
    """Side of the box a rung at cluster size `s` is drawn on.

    L(s) = ceil(box_factor * s**box_exponent), at least 1. Deterministic, and
    the single place the box rule is defined -- `cost_hint` returns its
    square, and `simulate` divides its counts by it.
    """
    if s < 1:
        raise ValueError(f"cluster size s must be >= 1; got {s}")
    if box_factor <= 0:
        raise ValueError(f"box_factor must be > 0; got {box_factor}")
    if box_exponent < 0:
        raise ValueError(f"box_exponent must be >= 0; got {box_exponent}")
    return max(1, int(math.ceil(box_factor * float(s) ** box_exponent - _CEIL_TOL)))


def _draw_open(rng: np.random.Generator, rows: int, L: int, p: float) -> np.ndarray:
    """`rows` independent L x L boolean lattices, each site open w.p. p.

    float32 uniforms, one draw per site, for the reason models/srw.py's
    `_draw_heads` gives: splitting `rows` into sequential blocks then consumes
    the RNG stream exactly as one unblocked call would, so the output is
    bit-identical at any `block_n`. Bit-packed integer draws are not.

    The realized open probability differs from `p` by at most 2**-24 ~ 6e-8,
    a distance to criticality whose correlation length is ~4e9 sites -- far
    beyond any box this model runs at.
    """
    return rng.random(size=(rows, L, L), dtype=np.float32) < p


def _label_block(open_grid: np.ndarray) -> np.ndarray:
    """Label every sample's open clusters in ONE ndimage.label call.

    The block's lattices are stacked vertically into one tall image with a
    blank separator row after each; no open path can cross a closed row, so
    labelling the stack once IS the per-sample labelling, and the C-level
    union-find sees one large problem instead of `rows` tiny ones (an
    allocation rule asks for millions of samples at the cheapest rungs).

    Returns int32 of shape (rows, L+1, L): `[:, :L, :]` is the sample and
    `[:, L, :]` its separator row, all label 0.

    Mirrors models/percolation2d.py's function of the same name. Deliberately
    duplicated rather than imported: models/README.md's rule is that no model
    file imports another, and the y-wrap below needs its own treatment anyway.
    """
    rows, L, _ = open_grid.shape
    padded = np.zeros((rows * (L + 1), L), dtype=bool)
    padded.reshape(rows, L + 1, L)[:, :L, :] = open_grid
    labels, _ = ndimage.label(padded, structure=_FOUR_CONNECTED)
    return labels.reshape(rows, L + 1, L)


def _wrap_roots(lab: np.ndarray, L: int) -> np.ndarray:
    """label -> canonical label after joining what the torus joins.

    `ndimage.label` cannot wrap, so the stack is labelled as a set of boxes
    and the labels joined by the two periodic directions are merged
    afterwards: column 0 with column L-1, and row 0 with row L-1 OF THE SAME
    SAMPLE (the stacked layout keeps each sample's rows contiguous, so the
    y-wrap is `lab[:, 0, :]` against `lab[:, L-1, :]` and never crosses a
    sample boundary).

    The merge is a union-find over LABELS -- a few percent as many as there
    are sites -- vectorized by pointer jumping: each pass pushes every pair's
    minimum across the edge in both directions and then squares the pointer
    array (`root[root]`), so a chain of length C resolves in O(log C) passes.
    `root[x] <= x` is an invariant, which is what makes the loop terminate at
    the component's smallest label.

    The squaring step is not decoration: without it a three-label chain
    reports the size of a piece instead of the size of the cluster.

    The termination test is on the WHOLE pass -- `root` unchanged after the
    unions AND the squaring -- not on the squaring alone. Testing only the
    squaring (`root[root] == root`, which is what models/percolation2d.py
    does) exits as soon as the pointer array is flat, which can happen on a
    pass where the union step still moved something and a further pass would
    move more: it returned 2 clusters where the flood fill finds 1 on the
    32x32 torus at `default_rng(3)`, sample 64 of 300. Pinned by
    `test_matches_an_independent_flood_fill` at L = 32.
    """
    nlab = int(lab.max()) + 1
    root = np.arange(nlab, dtype=lab.dtype)
    edges = ((lab[:, :L, 0], lab[:, :L, L - 1]),      # x-wrap
             (lab[:, 0, :], lab[:, L - 1, :]))         # y-wrap
    a_parts, b_parts = [], []
    for u, v in edges:
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
        f"{_MAX_MERGE_PASSES} passes ({nlab} labels, {a.size} wrap edges). "
        f"This should be unreachable -- see _MAX_MERGE_PASSES.")


def _roots(lab: np.ndarray, L: int, geometry: str) -> np.ndarray:
    """label -> canonical label for this geometry (identity for a box)."""
    if geometry == "box":
        return np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    return _wrap_roots(lab, L)


def _sizes_and_owner(lab: np.ndarray, L: int,
                     roots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-label cluster size and per-label sample index, for a whole block.

    Sizes are accumulated per label and re-accumulated per root -- two cheap
    operations on something label-sized -- rather than by relabelling the
    lattice. Non-root labels come back with size 0, which is what makes them
    invisible to every size window (all windows start at s >= 1).

    The label -> sample map is a single scatter over the sites
    (`owner[flat] = sample index`), which is well defined precisely because a
    cluster never spans two samples: the separator rows block the stack
    direction and the wrap joins a sample only to itself. That scatter is what
    makes the counting vectorized over the whole block; it assumes nothing
    about the ORDER ndimage.label hands out its labels.

    Split out of `_counts_in_range` so that the one-window path (`simulate`)
    and the every-window path (`binned_counts`) share it: the two passes over
    the lattice happen ONCE per block, not once per window, which is the whole
    economy of the shared-lattice sampler.
    """
    rows = lab.shape[0]
    by_label = np.bincount(lab.ravel(), minlength=roots.size)
    size = np.bincount(roots, weights=by_label, minlength=roots.size)
    size[0] = 0.0                      # label 0 is "closed", and the separators

    flat = lab[:, :L, :].reshape(rows, L * L)
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


def _counts_in_range(lab: np.ndarray, L: int, roots: np.ndarray,
                     s_lo: int, s_hi: int | None) -> np.ndarray:
    """Clusters per sample whose size is in [s_lo, s_hi) -- one window."""
    size, owner = _sizes_and_owner(lab, L, roots)
    return _count_window(size, owner, lab.shape[0], s_lo, s_hi)


def percolation_tau(
    s: int,
    n: int = 1,
    p: float = P_C_SQUARE_SITE,
    observable: str = "bin",
    bin_ratio: float = DEFAULT_BIN_RATIO,
    box_factor: float = DEFAULT_BOX_FACTOR,
    box_exponent: float = DEFAULT_BOX_EXPONENT,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_s: clusters at size scale `s`, per lattice site.

    Each sample is one fresh L(s) x L(s) critical lattice, union-found into
    clusters; the sample's value is the number of its clusters whose size is
    in [s, bin_ratio*s) (observable="bin", the logarithmic-histogram bar) or
    >= s (observable="tail", its integrated tail), DIVIDED BY L(s)**2. Both
    have E Y_s ~ a0 * s**(1-tau); see the module docstring for the conversion
    and for what the two switches (`observable`, `geometry`) exist to measure.

    `rng` defaults to a fresh unseeded Generator; pass a seeded one for
    reproducible runs. `block_n` defaults to a size derived from a fixed byte
    budget (_DEFAULT_WORKING_SET_BYTES); results are bit-identical for any
    block size at the same seed, because the blocking is over the leading
    (sample) axis (see `_draw_open`) -- and neither `observable` nor
    `geometry` touches the RNG, so all four combinations see the same
    lattices at the same seed.
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

    rng = rng if rng is not None else np.random.default_rng()
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_SITE * L * L))
    block_n = min(block_n, n)

    out = np.empty(n, dtype=np.float64)
    area = float(L) ** 2
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, L, p))
        roots = _roots(lab, L, geometry)
        out[offset:offset + rows] = _counts_in_range(lab, L, roots, s, s_hi) / area
        offset += rows
        del lab, roots
    return out


def simulate(s: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_tau"].simulate.

    params: {"p": float (default p_c), "observable": "bin"|"tail",
             "bin_ratio": float, "box_factor": float, "box_exponent": float,
             "geometry": "torus"|"box"}.

    Deliberately does NOT assert Assumption 2 (Y_s > 0). Unlike
    models/percolation2d.py's south anchor, where a zero is exponentially
    rare, a zero here is ORDINARY: the count of clusters at one size scale in
    one box is a small near-Poisson number, so a fraction ~exp(-lambda) of
    draws are 0 by design. What the article needs is E Y_s > 0 and a
    convergent Var(xi_s), and both are governed by lambda = L**2 * E n_s --
    which `box_factor` sets, and which a pilot measures (`zero_fraction`
    reports it from a finished sample). An assert would fire on every honest
    run.
    """
    return percolation_tau(
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


def cost_hint(s: int, params: dict | None = None) -> float:
    """Work for one sample: L(s)**2 sites, exactly.

    Both halves of the pipeline are linear in the number of sites -- L**2
    uniforms drawn, one near-linear union-find pass over L**2 sites -- and
    nothing here depends on p, on the observable or on the geometry: the whole
    lattice is labelled either way, and the torus adds one merge over
    something label-sized (a constant factor, which cancels in an allocation
    since only RATIOS across scales reach one).

    So d = 2 * box_exponent, i.e. exactly 1 at the default box_exponent = 1/2,
    and it is a geometric fact about the simulator rather than a stated
    formula -- the same status as models/percolation2d.py's i**2. The `ceil`
    in `box_side` makes cost(s) only ASYMPTOTICALLY a pure power; the wobble
    is a relative 1/L and is absorbed by the OLS-on-logs fit that
    tools/cost_model.py's `declared_exponent` runs over the recipe's ladder.

    params keys read: "box_factor", "box_exponent" (the box rule, nothing else).

    ONE UNIT TRAP, in the same family as models/synthetic.py's work-unit note.
    Only ratios across scales reach an allocation, so the unit of this
    function is free -- but a recipe's `budget` is NOT in these units. The
    allocation rules (tools/allocation.py) charge cost(i) = i**d with the
    SCALE itself, which for srw and percolation2d coincides with the site
    count and here does not: one budget unit is s**d = s, i.e.
    box_factor**2 = 256 lattice sites at the defaults. To spend S sites, ask
    for a budget of S / box_factor**2.

    This model is where that gap was first load-bearing, and it broke the
    planner: src/study/plan.py divided a cost in allocation units by a
    throughput measured in cost_hint units, so it predicted 740 s for a run
    that would have taken 42 hours. The conversion is now explicit and
    measured -- tools/cost_model.cost_unit_ratio, applied in plan.py's
    budget_for_seconds -- and is exactly 1.0 for every model whose cost_hint
    IS i**d. A recipe's hand-written `budget` is still in allocation units.
    """
    params = params or {}
    L = box_side(int(s), float(params.get("box_factor", DEFAULT_BOX_FACTOR)),
                 float(params.get("box_exponent", DEFAULT_BOX_EXPONENT)))
    return float(L) ** 2


# ---------------------------------------------------------------------------
# The shared-lattice sampler: one box, every rung
# ---------------------------------------------------------------------------
#
# `percolation_tau` above draws a fresh box FOR EACH rung, which is what makes
# the rungs independent and the article's CLT apply to them verbatim. It is
# also what makes it expensive: the whole ladder costs sum_k n_k L(s_k)**2.
#
# One L x L lattice already contains clusters of every size below its own
# cutoff, so a single box can serve the entire ladder -- Fig. 4.2's own
# procedure, and the cheap version (user, 2026-09-05). Sizing the box by the
# TOP rung,
#
#     L = ceil( (s_top_edge / cut_fraction) ** (1/df_lower) ),
#
# the ladder costs n * L**2 in total instead, about 3x less than the
# per-rung version at the same top-rung precision -- and every lower rung gets
# MORE data than it had before, since a box that can hold one cluster of size
# s_top holds ~(s_top/s)**(tau-1) times as many of size s.
#
# What is paid for it is independence: the rungs now come from the same
# lattices, so Cov(Ybar_s, Ybar_s') != 0 and the CLT of eq. (583) no longer
# applies as written. That is a deliberate experiment (user, 2026-09-05: "take
# this as an experiment; I still believe the estimator works, with some
# modification, when the correlation decays polynomially or faster"), and the
# correlation is measurable -- see experiments/03_percolation_zd/README.md.
# Every artifact drawn this way is stamped `shared_lattice` in its metadata so
# that no later reader mistakes it for an independent run.
#
# Two constants enter, and BOTH are design inputs in the sense of
# tools/allocation.py's `omega1` -- they decide which rungs are drawn and how
# big the box is, and neither reaches an estimator:
#
#   df_lower      a LOWER bound on d_f, so the trusted region s <= cut*L**df
#                 is conservative. This repo's own measurements are 1.9002
#                 (cylinder), 1.9059 and 1.9161 (south) -- all above 91/48 =
#                 1.8958 -- so DEFAULT_DF_LOWER = 1.85 sits below every one of
#                 them and below the literature value. A d_f that is too small
#                 keeps too few rungs (wasteful, never wrong); one that is too
#                 large keeps rungs the box has already cut off.
#   cut_fraction  the safety factor kappa in s <= kappa * L**df_lower. kappa=1
#                 is NOT safe: s = L**d_f is where the largest cluster in the
#                 box lives, i.e. the middle of the cutoff. The default is
#                 calibrated, not guessed -- see CUT_FRACTION_CALIBRATION.

#: Lower bound on d_f used to decide which rungs a box can serve. See above.
DEFAULT_DF_LOWER = 1.85

#: kappa in s <= kappa * L**df_lower. MEASURED, not guessed: hold one bin
#: fixed, grow L, and watch Ybar_s -- a per-site density, so it must converge.
#: Deviation from the large-L plateau (2026-09-05, 1.2e9 sites per point):
#:
#:     cut_ratio = s_hi/L**1.85   0.43   0.39   0.23   0.21   0.12  <=0.10
#:     bin [64,128)                 --  +13.9%    --   0.00%    --   <0.15%
#:     bin [256,512)            +20.4%     --  +0.39% --     -0.23%  <0.6%
#:
#: The two bins agree in the SCALED variable, which is the thing that had to
#: be true for a single kappa to serve every rung -- so the cutoff really is a
#: function of s/L**d_f alone, and one number bounds the whole ladder. Bias is
#: under the noise floor (0.2%/0.45%) for cut_ratio <~ 0.23 and explodes past
#: ~0.4. The default keeps a 4.6x margin below the last clean point.
#:
#: A guessed 0.01 (the first value here) is safe but 20x too conservative --
#: which costs nothing in sites (cost = n*L**2 is fixed by the top rung's
#: precision, not by L) and only makes each lattice bigger and rarer.
DEFAULT_CUT_FRACTION = 0.05


def shared_box_side(s_top_edge: int, df_lower: float = DEFAULT_DF_LOWER,
                    cut_fraction: float = DEFAULT_CUT_FRACTION) -> int:
    """Smallest box that can serve a ladder whose top bin ends at `s_top_edge`.

    Inverts s <= cut_fraction * L**df_lower. This is the "L ~ s**(1/d_f)" rule:
    the box is sized by the largest cluster size the ladder asks about, with
    `cut_fraction` the margin that keeps that top bin out of the cutoff.
    """
    if s_top_edge < 1:
        raise ValueError(f"s_top_edge must be >= 1; got {s_top_edge}")
    if not 0.0 < cut_fraction <= 1.0:
        raise ValueError(f"cut_fraction must be in (0, 1]; got {cut_fraction}")
    if df_lower <= 0:
        raise ValueError(f"df_lower must be > 0; got {df_lower}")
    return max(1, int(math.ceil((s_top_edge / cut_fraction) ** (1.0 / df_lower)
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
    p: float = P_C_SQUARE_SITE,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
) -> np.ndarray:
    """(n, len(windows)) counts: every window read off the SAME n lattices.

    `windows` is [(s_lo, s_hi or None)], one per rung. Each row is one
    L x L lattice, labelled once; the row's entries are that lattice's cluster
    counts in each window, so entries in one row are correlated by
    construction and rows are i.i.d.

    Identical, window by window, to calling the per-rung path on the same
    lattices: the block draws the same uniforms in the same order, and the
    counting shares `_sizes_and_owner`. That identity is a test
    (`test_binned_counts_matches_the_per_rung_path`), and it is what makes the
    cheap sampler a change of BUDGET rather than a change of observable.
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

    rng = rng if rng is not None else np.random.default_rng()
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_SITE * L * L))
    block_n = min(block_n, n)

    out = np.empty((n, len(windows)), dtype=np.int64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, L, p))
        size, owner = _sizes_and_owner(lab, L, _roots(lab, L, geometry))
        for j, (s_lo, s_hi) in enumerate(windows):
            out[offset:offset + rows, j] = _count_window(size, owner, rows,
                                                         s_lo, s_hi)
        offset += rows
        del lab, size, owner
    return out


def shared_sampler(scales, n: int, params: dict,
                   rng: np.random.Generator) -> tuple[dict, dict]:
    """MODELS["percolation_tau"].shared_sampler -- the whole ladder, one box.

    Returns ({scale: array of n values of Y_s}, info), where `info` records
    the box side, the two design constants behind it, and the site count --
    everything a reader needs to tell this run apart from an independent one.

    Y_s is normalized exactly as `simulate`'s is (count / L**2), so the two
    samplers produce the SAME observable and their gamma-hats are directly
    comparable; only the joint law across rungs differs.

    params: the same keys `simulate` reads (p, bin_ratio, geometry -- the box
    rule keys are ignored, the box comes from the ladder here), plus
    "df_lower" and "cut_fraction".
    """
    scales = [int(s) for s in scales]
    if sorted(scales) != scales or len(set(scales)) != len(scales):
        raise ValueError(f"scales must be strictly increasing; got {scales}")
    windows = bin_edges(scales, float(params.get("bin_ratio", DEFAULT_BIN_RATIO)))
    df_lower = float(params.get("df_lower", DEFAULT_DF_LOWER))
    cut_fraction = float(params.get("cut_fraction", DEFAULT_CUT_FRACTION))
    L = shared_box_side(windows[-1][1], df_lower, cut_fraction)

    counts = binned_counts(L, n, windows,
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


def zero_fraction(draws: np.ndarray) -> float:
    """Fraction of samples with Y_s = 0 -- the Assumption 2 diagnostic.

    Not a closed form (unlike models/percolation2d.py's `zero_rate`, which is
    exactly (1-p)**i): the count of clusters at one size scale has no simple
    exact law. It is reported from a finished sample instead, and is ~
    exp(-lambda) for the mean count lambda per box, i.e. it is `box_factor`
    that controls it. Used by the experiment README's Assumption-2 criterion.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.size == 0:
        raise ValueError("zero_fraction of an empty sample")
    return float((draws == 0.0).mean())
