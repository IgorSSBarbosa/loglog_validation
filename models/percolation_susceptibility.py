"""Off-critical site percolation on Z^dim: the MEAN CLUSTER SIZE (susceptibility)
as p climbs a geometric ladder toward p_c from below -- registered as
MODELS["percolation_susceptibility"] in tools/models.py.

This is the first experiment in this repo that is NOT at p = p_c. Written for
prompts/gamma_exponent.tex, which asks for

    S(p) ~ |p - p_c|^(-gamma_susc)   for p < p_c

with gamma_susc = 43/18 = 2.38889 in dim = 2 (and 1 exactly for dim >= 6).

The scale IS the distance to criticality
----------------------------------------
Every other model here indexes its ladder by a LENGTH (a box side i, a cluster
size s). Here the ladder is in the distance to the critical point,

    eps(x) = eps0 / x,      p(x) = p_c - eps(x),      x = 1, 2, 4, 8, ...

so a geometric ladder in x is a geometric ladder in 1/eps, and article eq. (232)
reads E Y_x ~ a0 * x^gamma with **gamma = gamma_susc** -- the article's exponent
and the susceptibility exponent are the same number in these coordinates, which
is what lets the existing estimators, the m0 window and the whole allocation
machinery run unmodified. `eps0` is a free amplitude: changing it multiplies a0
and leaves gamma alone.

NOTE ON THE LETTER GAMMA. Everywhere else in this repo gamma is eq. (232)'s
exponent and comes out as d_f (= 91/48 in dim 2). Here the SAME letter in the
SAME equation is the susceptibility exponent 43/18. They are different numbers
because the ladders are different (a length ladder at p_c vs a distance ladder
below it), not because anything is inconsistent. Recipes and READMEs for this
model say gamma_susc.

`eps0` has no dimension-independent default and is not given one: eps0 = 1/2 is
prompts/gamma_exponent.tex's own first rung and is fine in dim = 2 (p_c = 0.593),
but p_c = 0.3116 in dim = 3 and 0.1090 in dim = 6, where p_c - 1/2 is NEGATIVE.
The constructor refuses that rather than clipping, and names p_c/2 in the error.

The observable, and why it is not "the cluster of the origin"
-------------------------------------------------------------
The textbook definition is the mean size of the cluster of a UNIFORMLY CHOSEN
OPEN site, S = sum_s s^2 n_s / sum_s s n_s. Drawing that literally -- one lattice,
one origin, one number -- wastes the whole lattice and is unaffordable in Python.
Instead one sample is one L^dim TORUS and

    Y = (sum over clusters of s^2) / (p * L^dim) = (sum over SITES x of |C(x)|) / (p * L^dim)

which has E Y = S(p) EXACTLY (not approximately, and with no ratio bias): every
site x contributes E|C(x)| = p * S, the denominator is deterministic, and closed
sites contribute 0 to both sides. The 1/p converts "cluster of a site" into
"cluster of an OPEN site"; p is a known input, so dividing by it is exact and it
removes a factor that would otherwise vary along the ladder (p = p_c - eps0/x
carries its own x^-1 correction term, and there is no reason to fit what can be
divided out).

`moment` generalizes it: Y_k = (sum_clusters s^(k+1)) / (p * L^dim) has
E Y_k = E[|C|^k | 0 open]. Only k = 1 and k = 2 are interesting, and k = 2 is
interesting for a specific reason:

    E[|C|]   ~ s_xi^(3-tau)  ->  exponent e1 = (3-tau)/sigma = gamma_susc
    E[|C|^2] ~ s_xi^(4-tau)  ->  exponent e2 = (4-tau)/sigma

    so    1/sigma = e2 - e1    and    tau = 3 - e1/(e2 - e1)

i.e. ONE off-critical ladder, run twice over the same design at two moments,
measures gamma_susc, sigma and tau. With prompts/scaling_relations.tex's
inversion ((tau, sigma) determine everything below d_c) that is the whole
exponent set from this one experiment. In dim = 2: e1 = 43/18 = 2.38889,
e2 = 177/36 = 4.91667, e2 - e1 = 91/36 = 2.52778 = 1/sigma.

Geometry is the TORUS, and the box side is DECLARED, not searched
-----------------------------------------------------------------
Two finite-size decisions, both of which prompts/gamma_exponent.tex gets wrong
in a way worth recording:

(1) The prompt says "increase L until S(p,L) stabilizes". That is a
    data-dependent stopping rule: the L it picks is correlated with the sample,
    so the S reported at that L is biased, and cost-per-sample stops being
    knowable before the draw -- which is what every budget statement in this
    repo depends on. Instead L is DECLARED,

        L(x) = ceil(box_factor * x^nu_box)

    with `nu_box` a design constant, and the saturation is verified once, as a
    calibration, by sweeping box_factor at fixed x (experiments/06_susceptibility/,
    step 1). This is exactly models/percolation_tau.py's DF_LOWER pattern:
    a declared constant, demonstrated safe rather than asserted.

(2) What nu_box has to satisfy is subtler than "big enough". xi ~ eps^-nu ~ x^nu,
    so L/xi ~ x^(nu_box - nu):

        nu_box <  nu   L/xi SHRINKS along the ladder, the finite-size bias grows
                       with x, and it biases the EXPONENT (downward)
        nu_box == nu   L/xi constant -> the bias is a constant FACTOR -> it moves
                       a0 and leaves gamma alone. The cheap, correct choice
        nu_box >  nu   L/xi grows, bias shrinks along the ladder, exponent biased
                       slightly upward but bounded by the bias at the first rung

So nu_box >= nu is the safe side and nu_box = nu is the efficient one. The
default is 1.5, which is >= nu in EVERY dimension (nu = 4/3 in dim 2 and falls
to 1/2 at dim >= 6), i.e. safe without consulting a table of nu. dim = 2 recipes
may lower it to 4/3 once the calibration justifies it -- and must then run two
values and show gamma_hat agrees, the same demonstration DF_LOWER got.

Torus rather than box: a wall truncates every cluster that touches it, and it
does so worst exactly where xi approaches L, i.e. at the rungs that matter. The
torus has no wall. Its own finite-size effect is a cluster that wraps, which is
suppressed in L/xi rather than being O(surface/volume). percolation_zd's
geometry="box" is still reachable as a comparison arm.

Cost
----
cost_hint(x) = L(x)^dim, the sites in one sample, using the ACTUAL ceil'd L so
the hint is exact for the ladder in hand rather than exact only asymptotically.
In the ladder variable that is a power law with exponent nu_box * dim (= 3 at
the dim=2 default), so Assumption cost_is_power_law holds BY CONSTRUCTION and
tools/cost_model.py has a declared exponent to score against, as it does for
srw (d=1) and percolation_zd (d=dim).

No target_fn / true_gamma_key, for the same reason as every other model here:
43/18, 91/36 and 187/91 are acceptance criteria in
experiments/06_susceptibility/README.md and enter only at reporting time.
"""

from __future__ import annotations

import math

import numpy as np

from models.percolation_zd import (
    _BYTES_PER_SITE,
    _DEFAULT_WORKING_SET_BYTES,
    GEOMETRIES,
    _check_dim,
    _check_size,
    _draw_open,
    _label_block,
    _roots,
    _structure,
    critical_p,
)

#: Box-scaling exponent, L ~ x**nu_box. >= nu in every dimension (see docstring).
DEFAULT_NU_BOX = 1.5

#: L = ceil(box_factor * x**nu_box). Calibrated in experiments/06_susceptibility/
#: step 1; this default is the starting point of that sweep, not a measured value.
DEFAULT_BOX_FACTOR = 4.0

#: Highest moment k allowed in Y_k = E[|C|^k | open]. k = 3 already needs s**4,
#: which overflows nothing in float64 but is dominated by single clusters.
_MAX_MOMENT = 3


def epsilon(x: int, eps0: float) -> float:
    """Distance to criticality at ladder position x: eps = eps0 / x."""
    return float(eps0) / float(x)


def p_at(x: int, dim: int = 2, eps0: float = 0.5,
         p_c: float | None = None) -> float:
    """p = p_c - eps0/x, refusing a ladder that leaves the interval.

    Refuses rather than clips: an eps0 that is legal in dim = 2 and negative in
    dim = 3 is precisely the bug prompts/gamma_exponent.tex's original ladder
    (p_c - 1/2) has, and it must not become a silently different experiment.
    """
    pc = critical_p(dim) if p_c is None else float(p_c)
    eps = epsilon(x, eps0)
    p = pc - eps
    if not 0.0 < p < pc:
        raise ValueError(
            f"p = p_c - eps0/x = {p:.6g} is not in (0, p_c = {pc:.6g}) at "
            f"x = {x}, eps0 = {eps0}, dim = {dim}. This model is subcritical by "
            f"construction. In dim >= 3 eps0 = 1/2 is larger than p_c itself; "
            f"use eps0 = p_c/2 = {pc / 2:.6g} (or smaller) for this dimension.")
    return p


def box_side(x: int, box_factor: float = DEFAULT_BOX_FACTOR,
             nu_box: float = DEFAULT_NU_BOX) -> int:
    """L(x) = ceil(box_factor * x**nu_box) -- declared before the draw."""
    if box_factor <= 0:
        raise ValueError(f"box_factor must be > 0; got {box_factor}")
    if nu_box <= 0:
        raise ValueError(f"nu_box must be > 0; got {nu_box}")
    return int(math.ceil(float(box_factor) * float(x) ** float(nu_box)))


def cluster_moment_sum(lab: np.ndarray, L: int, dim: int, roots: np.ndarray,
                       moment: int) -> np.ndarray:
    """sum_clusters s**(moment+1), per sample, for a whole labelled block.

    Sizes are accumulated per label and re-accumulated per root (two operations
    on something label-sized, not another pass over the lattice), and the
    label -> sample map is one scatter over the sites, which is well defined
    because a cluster never spans two samples: the separator slab blocks the
    stack axis and the wrap joins a sample only to itself. Both techniques are
    models/percolation_tau.py's `_sizes_and_owner`, reused here with a weight
    of s**(moment+1) instead of a size window.

    Label 0 (closed sites and separators) is zeroed before weighting, so its
    meaningless owner never contributes.
    """
    rows = lab.shape[0]
    by_label = np.bincount(lab.ravel(), minlength=roots.size)
    size = np.bincount(roots, weights=by_label, minlength=roots.size)
    size[0] = 0.0

    flat = lab[:, :L].reshape(rows, -1)
    owner = np.zeros(roots.size, dtype=np.int64)
    owner[flat] = np.broadcast_to(np.arange(rows, dtype=np.int64)[:, None],
                                  flat.shape)

    weight = size ** (moment + 1)
    weight[0] = 0.0
    return np.bincount(owner, weights=weight, minlength=rows)[:rows]


def percolation_susceptibility(
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
    block_n: int | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y = sum_clusters s^(moment+1) / (p * L^dim).

    E Y = E[|C(0)|^moment | 0 open], exactly, on the infinite lattice; on the
    torus it is that up to clusters large enough to wrap, which is what
    box_factor/nu_box control.

    `p` overrides the ladder (tests and degenerate cases want a specific p);
    otherwise p = p_c - eps0/x. `p_c` overrides the tabulated critical point.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; known: {list(GEOMETRIES)}")
    dim = _check_dim(dim)
    moment = int(moment)
    if not 1 <= moment <= _MAX_MOMENT:
        raise ValueError(f"moment must be in 1..{_MAX_MOMENT}; got {moment}")
    x = int(x)
    if x < 1:
        raise ValueError(f"ladder position x must be >= 1; got {x}")

    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    L = box_side(x, box_factor, nu_box)
    _check_size(L, dim)

    rng = rng if rng is not None else np.random.default_rng()
    structure = _structure(dim)
    if block_n is None:
        block_n = max(1, _DEFAULT_WORKING_SET_BYTES
                      // (_BYTES_PER_SITE * L ** dim))
    block_n = min(block_n, n)

    sites = float(L) ** dim
    out = np.empty(n, dtype=np.float64)
    offset = 0
    while offset < n:
        rows = min(block_n, n - offset)
        lab = _label_block(_draw_open(rng, rows, L, dim, p), structure)
        roots = _roots(lab, L, dim, geometry)
        out[offset:offset + rows] = cluster_moment_sum(
            lab, L, dim, roots, moment) / (p * sites)
        offset += rows
        del lab, roots
    return out


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility"].simulate.

    params: {"dim": int (default 2), "eps0": float (default 0.5 -- dim 2 only,
             see p_at), "p": float (overrides the ladder), "p_c": float,
             "box_factor": float, "nu_box": float, "moment": 1|2|3,
             "geometry": "torus" (default) | "cylinder" | "box"}.
    """
    return percolation_susceptibility(
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


def cost_hint(x: int, params: dict | None = None) -> float:
    """Work for one sample: L(x)**dim, with the ACTUAL ceil'd L.

    A power law in x with exponent nu_box * dim up to the ceil, so Assumption
    cost_is_power_law holds by construction and tools/cost_model.py has a
    declared exponent to score a measured one against. Using the ceil'd L rather
    than box_factor * x**nu_box matters at the bottom of the ladder, where the
    rounding is a large relative cost -- the lesson TODO records from
    SITES_PER_BUDGET_UNIT in high dim.

    params keys read: "dim", "box_factor", "nu_box".
    """
    params = params or {}
    dim = _check_dim(params.get("dim", 2))
    L = box_side(x, float(params.get("box_factor", DEFAULT_BOX_FACTOR)),
                 float(params.get("nu_box", DEFAULT_NU_BOX)))
    return float(L) ** dim


def declared_cost_exponent(params: dict | None = None) -> float:
    """nu_box * dim -- what a measured cost exponent should reproduce."""
    params = params or {}
    return float(params.get("nu_box", DEFAULT_NU_BOX)) * _check_dim(
        params.get("dim", 2))
