"""Critical site percolation on Z^dim, swept in slabs -- O(i**(dim-1)) memory
instead of O(i**dim). Registered as MODELS["percolation_zd_stream"].

Same observable, same parameters and same cost_hint as
models/percolation_zd.py; the only difference is that the box is never
materialized. See plans/streaming_percolation.md for the design and for the
decisions behind it -- this file implements §3 (the sequential frontier sweep),
not §4 (the parallel divide and conquer).

Why it exists
-------------
models/percolation_zd.py allocates the whole box: about 10 bytes a site, and
`block_n` bounds how many SAMPLES are in flight but cannot make one sample
smaller. So one sample at dim = 3, i = 2048 is 8.6e9 sites and 80 GiB, and the
ladders stop at i <= 256 (dim 3), 64 (4), 32 (5), 16 (6). Three separate
weaknesses in experiments/05_percolation_highd trace to exactly that: only one
omega_1 estimator above dim = 2 (the bias-decay fit needs 7 rungs), a cost
probe on a short lever arm, and a criticality check with no discriminating
power at dim = 6. It is a speed wall too -- Experiment H8 measured 0.86
Msites/s at dim = 8 against 25.65 at dim = 2, the collapse tracking the working
set leaving cache.

Sweeping the box in slabs along x_0 and carrying only the interface makes the
memory (dim-1)-dimensional: 80 GiB becomes 32 MiB at dim = 3, i = 2048.

Why the obvious version is WRONG, and what this does instead
-------------------------------------------------------------
Connectivity is not causal in the sweep direction. A site can reach the anchor
face only by going DEEPER and coming back, so a sweep that DECIDES
"connected / not connected" as it passes undercounts. Minimal case, anchor =
row 0 (verified against models/percolation_zd.py):

    1 0 0
    1 0 1     <- (1,2) reaches row 0 only via (2,2) -> (2,1) -> (2,0) -> (1,0)
    1 1 1

true count 6, greedy forward sweep 5.

So this does not decide anything as it passes. It carries UNRESOLVED
components -- Hoshen-Kopelman with a frontier:

  - each live component has a size, a "touches the seed set" flag, and (for
    `face_far`) a count of its sites in the far half;
  - a slab is labelled, its labels are unioned across the interface with the
    previous frontier and across any periodic transverse face;
  - a component with no label left in the new frontier is DEAD: it can never
    grow or merge again, so its flag is final. It is banked and freed.

Freeing is what makes the memory bound hold, and it is done by COMPACTION
rather than by refcounting: after each slab the live roots are renumbered into
a dense space and the frontier is rewritten in terms of them, so the label
space is at all times the number of live components, at most i**(dim-1).
Refcounting would be O(1) per label instead of O(live) per slab, but it is the
kind of bookkeeping that fails silently, and this repo has already paid for one
of those (models/percolation2d.py's early-exit merge). Compaction is O(i**dim)
over the whole sweep, the same order as the labelling itself.

`geometry = "torus"` wraps the sweep axis as well, so the first slab's
components must survive to the end to be joined to the last one's: they are
PINNED, never finalized early, at the cost of one extra frontier.

BIT-IDENTICAL to models/percolation_zd.py -- which the design note said was
impossible
-----------------------------------------------------------------------------
plans/streaming_percolation.md §6 argued that a streaming sampler could not
reproduce a materialized run's seed, because "slab-by-slab draws consume the
same stream in a different order", and concluded that the equivalence would
have to be tested on explicit lattices instead. That is true only of a sampler
that VECTORIZES ACROSS SAMPLES, which is what the first draft here did.

numpy fills `rng.random(size=(rows,) + (i,)*dim)` in C order, which is
sample-major and plane-minor: sample 0's planes in order, then sample 1's.
A sweep that processes ONE SAMPLE AT A TIME and draws it plane by plane
consumes exactly that sequence. Verified, not assumed:

    rng.random((rows, i, i))  ==  [[rng.random(i) per plane] per sample]

So this model is bit-identical to models/percolation_zd.py at the same seed,
for every anchor, geometry, i and dim -- and `slab_h` changes the memory and
the number of SciPy calls while changing nothing about the numbers, which is
what models/README.md's rule demands of a working-set knob. Both are tests.

The price is that samples are not batched: at small i the per-sample Python
and SciPy overhead is no longer amortized over a block, so this is SLOWER than
the materialized model there. It is not meant for that regime. Batching would
buy it back and would break both properties above, so it is deliberately not
done -- the memory bound is the product here, and a sampler that silently
stopped reproducing recorded seeds would be a bad trade for it.

It stays a SEPARATE model rather than replacing percolation_zd because the
performance profiles are opposite (see the crossover in
experiments/05_percolation_highd/README.md), and because a model this repo has
already recorded results with should not change its runtime characteristics
underneath them.

`cost_hint` is identical (i**dim), so allocations, budgets and cost_unit_ratio
carry over unchanged and a streaming recipe is comparable to a materialized one
at equal budget.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Site-percolation thresholds on Z^dim, restated rather than imported:
#: models/README.md's rule is that no model file imports another. See
#: models/percolation_zd.py's P_C_SOURCE for the reference behind each entry
#: and `crossing_fraction` there for the diagnostic that checks one.
P_C_SITE_HYPERCUBIC: dict[int, float] = {
    1: 1.0, 2: 0.59274605079210, 3: 0.3116077, 4: 0.19688561,
    5: 0.14079633, 6: 0.109016661, 7: 0.088951121, 8: 0.075210128,
    9: 0.065209595, 10: 0.057592949, 11: 0.051589684, 12: 0.046730976,
    13: 0.042715021,
}

ANCHORS = ("face", "face_far", "origin", "slab")
GEOMETRIES = ("box", "cylinder", "torus")
_ANCHOR_ALIASES = {"south": "face"}
_MAX_DIM = 13


def critical_p(dim: int) -> float:
    """p_c on Z^dim, from the table -- raising, never guessing, if absent."""
    if dim not in P_C_SITE_HYPERCUBIC:
        raise ValueError(
            f"no tabulated p_c for dim = {dim}; known dimensions: "
            f"{sorted(P_C_SITE_HYPERCUBIC)}. Pass params['p'] explicitly.")
    return P_C_SITE_HYPERCUBIC[dim]

#: Working-set budget for one (rows, slab) draw, in bytes. Bounds the FRONTIER
#: now, not the box, which is the whole point.
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: Per site of a slab: 4 bytes for the float32 uniform, 1 for the boolean
#: lattice, 4 for the int32 labels, 4 for the carried frontier ids.
_BYTES_PER_SITE = 13

#: Slab thickness along the sweep axis. 1 is the smallest frontier and the most
#: Python-level calls; larger amortizes the per-slab overhead against more
#: memory. Chosen from the byte budget unless the caller says otherwise, and it
#: never changes the numbers (a test).
_DEFAULT_SLAB_H = None

#: ndimage.label returns int32, so a SLAB of 2**31 sites cannot be labelled.
#: The ceiling is now on i**(dim-1), not i**dim -- which is the entire gain.
_MAX_SITES_PER_SLAB = 2 ** 31 - 1

_MAX_MERGE_PASSES = 64


def _structure(dim: int) -> np.ndarray:
    return ndimage.generate_binary_structure(dim, 1)


def _union(root: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge the pairs (a, b) into `root` by vectorized pointer jumping.

    The same construction as models/percolation_zd.py's `_wrap_roots`, and with
    the same STRICT termination test -- `root` unchanged after the unions AND
    the squaring, not after the squaring alone. Testing only the squaring exits
    on a pass where the union step still moved something, returning a piece of
    a component as if it were the component; the chains here are long (an
    interface plus up to dim-1 periodic faces at once), so it is reachable.
    """
    if a.size == 0:
        return root
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for _ in range(_MAX_MERGE_PASSES):
        before = root.copy()
        np.minimum.at(root, hi, root[lo])
        np.minimum.at(root, lo, root[hi])
        root = root[root]
        if np.array_equal(before, root):
            return root
    raise RuntimeError(
        f"the slab merge did not converge in {_MAX_MERGE_PASSES} passes "
        f"({root.size} ids, {a.size} pairs). This should be unreachable.")


class _Sweep:
    """The frontier sweep, fed one slab at a time.

    Split out from the sampler so the SAME code can be driven from a
    materialized lattice (`count_lattice`, for tests) and from freshly drawn
    slabs (`percolation_zd_stream`, for production). The test that the two
    agree on a given lattice is what replaces bit-identity with
    models/percolation_zd.py.
    """

    def __init__(self, rows: int, i: int, dim: int, anchor: str,
                 anchor_dim: int | None, geometry: str):
        self.rows, self.i, self.dim = rows, i, dim
        self.anchor, self.k, self.geometry = anchor, anchor_dim, geometry
        self.transverse = () if geometry == "box" else tuple(range(1, dim))
        self.wrap_sweep = (geometry == "torus")
        self.depth = 0                      # planes consumed so far
        self.total = np.zeros(rows, dtype=np.int64)
        # Live component state, dense ids 1..k (0 is "nothing").
        self.front = np.zeros((rows,) + (i,) * (dim - 1), dtype=np.int64)
        self.size = np.zeros(1, dtype=np.int64)
        self.far = np.zeros(1, dtype=np.int64)
        self.flag = np.zeros(1, dtype=bool)
        self.owner = np.zeros(1, dtype=np.int64)
        self.pinned = np.zeros(1, dtype=bool)   # first plane, for the torus
        #: The first plane's component ids, kept CURRENT through every
        #: compaction (they are renumbered like the frontier is). Only the
        #: torus needs them, to close the sweep-axis wrap at the end; holding
        #: the ids from slab 0 unchanged would point at the wrong components
        #: two compactions later, which is exactly how this was first wrong.
        self.first = None
        self.struct = _structure(dim)

    # -- the seed set and the counting region -----------------------------
    def _seed_mask(self, sh: int) -> np.ndarray | None:
        """Which sites of this slab are seeds. None when there are none."""
        i, dim, d0 = self.i, self.dim, self.depth
        m = np.zeros((sh,) + (i,) * (dim - 1), dtype=bool)
        if self.anchor in ("face", "face_far"):
            if d0 == 0:
                m[0] = True
                return m
            return None
        c = i // 2
        k = 0 if self.anchor == "origin" else self.k
        idx: list = [slice(None)] * dim
        for a in range(max(k, 1), dim):
            idx[a] = c
        if k == 0:                          # the centre site: axis 0 pinned too
            if not d0 <= c < d0 + sh:
                return None
            idx[0] = c - d0
        m[tuple(idx)] = True
        return m

    def _far_planes(self, sh: int) -> slice | None:
        """Which planes of this slab count, for anchor='face_far'."""
        if self.anchor != "face_far":
            return slice(0, sh)
        lo = max(0, self.i // 2 - self.depth)
        return slice(lo, sh) if lo < sh else None

    # -- one slab ---------------------------------------------------------
    def push(self, slab: np.ndarray) -> None:
        """Consume one slab, shape (rows, sh) + (i,)*(dim-1) of booleans."""
        rows, i, dim = self.rows, self.i, self.dim
        sh = slab.shape[1]
        nprev = self.size.size - 1                    # live ids are 1..nprev

        # Label the slab, samples stacked with a blank separator plane -- the
        # same trick models/percolation_zd.py uses on the whole box, applied to
        # a slab of thickness sh.
        padded = np.zeros((rows * (sh + 1),) + (i,) * (dim - 1), dtype=bool)
        padded.reshape((rows, sh + 1) + (i,) * (dim - 1))[:, :sh] = slab
        lab, nlab = ndimage.label(padded, structure=self.struct)
        lab = lab.reshape((rows, sh + 1) + (i,) * (dim - 1))[:, :sh]

        # One id space: 0 nothing, 1..nprev carried, nprev+1..nprev+nlab new.
        cur = np.where(lab > 0, lab.astype(np.int64) + nprev, 0)
        root = np.arange(nprev + nlab + 1, dtype=np.int64)

        # EVERY pair this slab creates goes into ONE merge -- the interface
        # with the previous frontier, and each periodic transverse face.
        # Merging them in separate passes is wrong and silently so: a later
        # pass can pull an id out from under a root an earlier pass had linked
        # to it, orphaning that root. On a 4x4 torus it lost one site of a
        # six-site cluster. models/percolation_zd.py's `_wrap_roots` collects
        # every axis's edges before its loop for exactly this reason.
        pairs_a, pairs_b = [], []
        if self.depth > 0:
            both = (self.front > 0) & (cur[:, 0] > 0)
            if both.any():
                pairs_a.append(self.front[both])
                pairs_b.append(cur[:, 0][both])
        for axis in self.transverse:
            lo_idx: list = [slice(None)] * cur.ndim
            hi_idx: list = [slice(None)] * cur.ndim
            lo_idx[axis + 1] = 0
            hi_idx[axis + 1] = i - 1
            u, v = cur[tuple(lo_idx)], cur[tuple(hi_idx)]
            both = (u > 0) & (v > 0)
            if both.any():
                pairs_a.append(u[both])
                pairs_b.append(v[both])
        if pairs_a:
            root = _union(root, np.concatenate(pairs_a), np.concatenate(pairs_b))

        # Per-id state for this pass, carried ids first.
        size = np.concatenate([self.size, np.bincount(cur.ravel(),
                                                      minlength=nprev + nlab + 1
                                                      )[nprev + 1:]])
        far_planes = self._far_planes(sh)
        if far_planes is None:
            far_new = np.zeros(nlab, dtype=np.int64)
        else:
            far_new = np.bincount(cur[:, far_planes].ravel(),
                                  minlength=nprev + nlab + 1)[nprev + 1:]
        far = np.concatenate([self.far, far_new])
        flag = np.concatenate([self.flag, np.zeros(nlab, dtype=bool)])
        owner = np.concatenate([self.owner, np.zeros(nlab, dtype=np.int64)])
        pinned = np.concatenate([self.pinned, np.zeros(nlab, dtype=bool)])
        size[0] = far[0] = 0
        who = np.broadcast_to(np.arange(rows, dtype=np.int64)
                              .reshape((rows,) + (1,) * dim), cur.shape)
        owner[cur] = who                       # a component never spans samples

        seeds = self._seed_mask(sh)
        if seeds is not None:
            sel = cur[:, seeds]
            flag[sel[sel > 0]] = True
        if self.wrap_sweep and self.depth == 0:
            pinned[cur[:, 0][cur[:, 0] > 0]] = True

        # Collapse everything onto roots.
        r = root
        size = np.bincount(r, weights=size, minlength=r.size).astype(np.int64)
        far = np.bincount(r, weights=far, minlength=r.size).astype(np.int64)
        f2 = np.zeros(r.size, dtype=bool); np.logical_or.at(f2, r, flag)
        p2 = np.zeros(r.size, dtype=bool); np.logical_or.at(p2, r, pinned)
        o2 = np.zeros(r.size, dtype=np.int64); o2[r] = owner
        size[0] = far[0] = 0; f2[0] = p2[0] = False

        # The new frontier, in root ids; anything not in it (and not pinned) is
        # dead and can be banked.
        new_front = r[cur[:, sh - 1]]
        live = np.zeros(r.size, dtype=bool)
        live[new_front[new_front > 0]] = True
        live |= p2
        dead = (~live) & (np.arange(r.size) == r) & (size > 0)
        if dead.any():
            got = np.where(f2[dead], (far if self.anchor == "face_far" else size)[dead], 0)
            np.add.at(self.total, o2[dead], got)

        # Compact: renumber the live roots densely, rewrite the frontier.
        keep = np.nonzero(live & (np.arange(r.size) == r))[0]
        newid = np.zeros(r.size, dtype=np.int64)
        newid[keep] = np.arange(1, keep.size + 1)
        self.front = newid[new_front]
        self.size = np.concatenate([[0], size[keep]])
        self.far = np.concatenate([[0], far[keep]])
        self.flag = np.concatenate([[False], f2[keep]])
        self.owner = np.concatenate([[0], o2[keep]])
        self.pinned = np.concatenate([[False], p2[keep]])
        if self.wrap_sweep:
            first_ids = r[cur[:, 0]] if self.depth == 0 else r[self.first]
            self.first = newid[first_ids]
        self.depth += sh

    def finish(self) -> np.ndarray:
        """Join the torus's sweep-axis wrap, bank what is left, return counts."""
        n = self.size.size
        root = np.arange(n, dtype=np.int64)
        if self.wrap_sweep and self.first is not None:
            both = (self.front > 0) & (self.first > 0)
            if both.any():
                root = _union(root, self.front[both], self.first[both])
        size = np.bincount(root, weights=self.size, minlength=n).astype(np.int64)
        far = np.bincount(root, weights=self.far, minlength=n).astype(np.int64)
        f2 = np.zeros(n, dtype=bool); np.logical_or.at(f2, root, self.flag)
        o2 = np.zeros(n, dtype=np.int64); o2[root] = self.owner
        size[0] = far[0] = 0; f2[0] = False
        alive = (np.arange(n) == root) & (size > 0)
        got = np.where(f2[alive], (far if self.anchor == "face_far" else size)[alive], 0)
        np.add.at(self.total, o2[alive], got)
        return self.total


# ---------------------------------------------------------------------------
# Drivers: the same sweep, fed from a lattice (tests) or from the RNG (runs)
# ---------------------------------------------------------------------------

def _validate(i, dim, anchor, anchor_dim, geometry):
    anchor = _ANCHOR_ALIASES.get(anchor, anchor)
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; known: {list(ANCHORS)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = int(dim)
    if not 1 <= dim <= _MAX_DIM:
        raise ValueError(f"dim must be in [1, {_MAX_DIM}]; got {dim}")
    i = int(i)
    if i < 1:
        raise ValueError(f"box side must be >= 1; got {i}")
    k = None
    if anchor == "slab":
        if anchor_dim is None:
            raise ValueError('anchor="slab" needs params["anchor_dim"]')
        k = int(anchor_dim)
        if not 0 <= k <= dim:
            raise ValueError(f"anchor_dim must be in [0, {dim}]; got {k}")
    face = i ** (dim - 1)
    if face > _MAX_SITES_PER_SLAB:
        raise ValueError(
            f"one FRONTIER at i = {i}, dim = {dim} is {face:.3g} sites, past "
            f"ndimage.label's int32 label space ({_MAX_SITES_PER_SLAB}). The "
            f"box itself is no longer the constraint -- i**(dim-1) is.")
    return i, dim, anchor, k, geometry


def count_lattice(grid: np.ndarray, anchor: str = "face",
                  anchor_dim: int | None = None, geometry: str = "box",
                  slab_h: int = 1) -> int:
    """The observable of ONE materialized lattice, computed by the sweep.

    Exists so the streaming logic can be checked against
    models/percolation_zd.py on the SAME lattice -- which is what replaces
    bit-identity, since the two consume the RNG differently
    (plans/streaming_percolation.md §5.1, §6). Production never calls this: it
    takes a whole box, which is the thing the sweep exists to avoid.
    """
    dim = grid.ndim
    i = grid.shape[0]
    i, dim, anchor, k, geometry = _validate(i, dim, anchor, anchor_dim, geometry)
    sw = _Sweep(1, i, dim, anchor, k, geometry)
    for h in range(0, i, slab_h):
        sw.push(grid[None, h:min(h + slab_h, i)])
    return int(sw.finish()[0])


def percolation_zd_stream(
    i: int,
    n: int = 1,
    dim: int = 2,
    p: float | None = None,
    anchor: str = "face",
    anchor_dim: int | None = None,
    geometry: str = "box",
    rng: np.random.Generator | None = None,
    block_n: int | None = None,
    slab_h: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y_i, never materializing a whole box.

    BIT-IDENTICAL to models/percolation_zd.py's `percolation_zd` at the same
    seed -- see the module docstring for why that is possible and what it
    costs. `slab_h` trades frontier memory against the number of per-slab
    SciPy calls and does not change the answer; `block_n` is accepted only so
    the signature matches the materialized model, and is ignored, because
    batching samples is what would break both properties.
    """
    i, dim, anchor, k, geometry = _validate(i, dim, anchor, anchor_dim, geometry)
    p = critical_p(dim) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    rng = rng if rng is not None else np.random.default_rng()
    del block_n            # accepted for signature parity; see the docstring

    face = i ** (dim - 1)
    if slab_h is None:
        slab_h = max(1, min(i, _DEFAULT_WORKING_SET_BYTES
                            // (_BYTES_PER_SITE * max(face, 1))))
    slab_h = max(1, min(int(slab_h), i))

    # ONE SAMPLE AT A TIME, planes in order. That is exactly the sequence
    # numpy's C-order fill gives models/percolation_zd.py's single
    # rng.random(size=(rows,) + (i,)*dim) call, which is what makes this
    # bit-identical at the same seed -- and what makes `slab_h` free.
    out = np.empty(n, dtype=np.int64)
    for s in range(n):
        sw = _Sweep(1, i, dim, anchor, k, geometry)
        for h in range(0, i, slab_h):
            sh = min(slab_h, i - h)
            slab = rng.random(size=(1, sh) + (i,) * (dim - 1),
                              dtype=np.float32) < p
            sw.push(slab)
        out[s] = sw.finish()[0]
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_zd_stream"].simulate -- params as percolation_zd's,
    plus an optional "slab_h" that changes the memory and not the numbers."""
    return percolation_zd_stream(
        i, n=n, dim=int(params.get("dim", 2)), p=params.get("p"),
        anchor=params.get("anchor", "face"), anchor_dim=params.get("anchor_dim"),
        geometry=params.get("geometry", "box"), rng=rng,
        slab_h=params.get("slab_h"),
    )


def cost_hint(i: int, params: dict | None = None) -> float:
    """i**dim, identical to models/percolation_zd.py's.

    Streaming changes the MEMORY, not the work: every site is still drawn once
    and visited O(1) times. Keeping the hint identical is what makes a
    streaming recipe comparable to a materialized one at equal budget, and
    keeps cost_unit_ratio at exactly 1.0.
    """
    dim = int((params or {}).get("dim", 2))
    if not 1 <= dim <= _MAX_DIM:
        raise ValueError(f"dim must be in [1, {_MAX_DIM}]; got {dim}")
    return float(i) ** dim


def zero_rate(i: int, dim: int = 2, p: float | None = None) -> float:
    """P(Y_i = 0) for anchor="face": exactly (1-p)**(i**(dim-1))."""
    p = critical_p(int(dim)) if p is None else float(p)
    return float(1.0 - p) ** (int(i) ** (int(dim) - 1))
