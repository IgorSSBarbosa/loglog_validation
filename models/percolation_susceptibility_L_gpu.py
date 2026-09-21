"""The susceptibility ladder indexed by the BOX SIDE L, on a CUDA GPU -- registered
as MODELS["percolation_susceptibility_L_gpu"] in tools/models.py.

Designed with Igor on 2026-09-21 (plans/exact_box_ladder.md); GPU only, by his
decision. The observable, the torus, the draw and the random stream are
models/percolation_susceptibility_gpu.py's, unchanged -- this file only changes
WHICH rung is drawn, i.e. how a scale is turned into (L, p).

Why a second ladder
-------------------
percolation_susceptibility(_gpu) takes the rung x, sets p = p_c - eps0/x and
puts it on a torus of side L = ceil(c * x**nu_box). With L/xi constant the
finite-torus deficit is a constant factor that moves a0 and leaves gamma alone.
The CEILING breaks the constancy: each rung gets its own c_x = L / x**nu_box.
In d = 4 with c = 4, x = 2, 4, 8, ..., 256 give c_x / c - 1 = +8.5%, +5.7%,
+1.2%, +3.3%, +0.7%, +0.7%, +0.2%, +0.3%. Where S(p, L) still grows with L
(G = d log S / d log L = 0.162 +- 0.023 there) rung x is shifted by
G log(c_x / c), which is 70 sigma at x = 2 and not monotone in x: the omega_1
pilot pilot_gsusc_d4_low zig-zagged at 5-31 sigma, and eq. (232) plus one
G log(c_x / c) term fits the pooled runs at chi^2 = 3.7 / 3 where the plain
fit gives 47 / 4 (experiments/09_percolation_susceptibility_gpu/diagnostics/).

The scale IS L
--------------
Index the ladder by the side instead,

    x(L) = (L / c) ** (1 / nu_box),    p(L) = p_c - eps0 / x(L),    cost(L) = L**dim.

On L = 2**k every box is an integer, nothing is rounded and L / x**nu_box = c
EXACTLY on every rung (ratio rho = 2 ** (1 / nu_box) in x). The same three
design constants as before (eps0, c, nu_box), the same relation L = c x**nu_box,
the same estimator; only the choice of rungs changes. Any integer L is an exact
rung, so a grid that is not 2**k is possible -- but tools/allocation.py's
`gamma_closed_form` takes any grid whose rint(log L / log rho) is consecutive
and weights it as if it were exact, so a ROUNDED grid (e.g. round(sqrt(2)**k))
belongs only to the fits that use the actual log L (`gamma_all_points`, the
omega_1 pilot's `fit_correction`), never to eq. (720)'s closed form.

In L units eq. (232) holds exactly with gamma_L = gamma_susc / nu_box and
omega_L = omega_x / nu_box, so gamma_susc = nu_box * gamma_L: nu_box is the
design constant that set p, not an estimate of nu, and the conversion adds no
error. NOTE ON THE LETTER GAMMA, a third meaning: gamma_L is the exponent of
this ladder. It is neither the susceptibility exponent of percolation_susceptibility
(rungs in x) nor gamma / nu = 2 - eta. The tools report gamma_L and omega_L and
never learn nu_box; the conversion is made where the result is written up.

What the design constants still do (plans/exact_box_ladder.md section 3):
none of eps0, c, nu_box is measured, and a "wrong" one only moves L / xi away
from what was intended. eps0 or c wrong: L / xi is a different constant, only
a0 moves. nu_box != nu: L / xi drifts as x**(nu_box - nu) and gamma_hat is
biased by G (nu_box - nu). nu_box stays an UPPER bound on nu (the rule in
models/percolation_susceptibility.py), which fixes the sign.

Cost
----
cost_hint(L) = L**dim exactly: no ceiling, so Assumption 7 (cost is a power law
in the scale) holds with the declared d = dim. That is not a fitted number.

Refusals
--------
Not clipped, as p_at does not clip: p <= 0 (L too small for eps0 and c; the
message names the smallest legal L), a non-integer or L < 2, and `_check_size`
(int32 labels: (L + 1) L**(dim - 1) < 2**31, i.e. L <= 215 in d = 4).
`dim`, `eps0`, `box_factor` and `nu_box` have NO defaults here: they are the
design, and a recipe that forgets one should fail rather than run a different
experiment.
"""

from __future__ import annotations

import numpy as np

from models.percolation_susceptibility_gpu import (
    _check_design,
    _susceptibility_gpu_at,
    block_rows,
)
from models.percolation_susceptibility import (
    GEOMETRIES,
    _check_dim,
    critical_p,
    epsilon,
)

__all__ = ["percolation_susceptibility_L_gpu", "simulate", "cost_hint",
           "declared_cost_exponent", "x_of_L", "p_at_L", "smallest_legal_L",
           "block_rows", "epsilon", "critical_p", "GEOMETRIES"]

#: A torus needs at least 2 sites a side (L = 1 is one site that neighbours itself).
_MIN_SIDE = 2

_REQUIRED = ("dim", "eps0", "box_factor", "nu_box")


def _side(L) -> int:
    """L as an int, refusing a scale that is not an integer >= 2."""
    if int(L) != L:
        raise ValueError(f"the scale is the box side L and must be an integer; got {L!r}")
    L = int(L)
    if L < _MIN_SIDE:
        raise ValueError(f"box side L must be >= {_MIN_SIDE}; got {L}")
    return L


def x_of_L(L: int, box_factor: float, nu_box: float) -> float:
    """x(L) = (L / c) ** (1 / nu_box): the rung of the x-ladder whose box is exactly L."""
    if box_factor <= 0:
        raise ValueError(f"box_factor must be > 0; got {box_factor}")
    if nu_box <= 0:
        raise ValueError(f"nu_box must be > 0; got {nu_box}")
    return (float(L) / float(box_factor)) ** (1.0 / float(nu_box))


def _p(L: int, pc: float, eps0: float, box_factor: float, nu_box: float) -> float:
    return pc - epsilon(x_of_L(L, box_factor, nu_box), eps0)


def smallest_legal_L(dim: int, eps0: float, box_factor: float, nu_box: float,
                     p_c: float | None = None) -> int:
    """Smallest side with 0 < p(L): p = p_c - eps0 / x(L) crosses 0 at x = eps0 / p_c."""
    pc = critical_p(dim) if p_c is None else float(p_c)
    L = _MIN_SIDE
    while not _p(L, pc, eps0, box_factor, nu_box) > 0.0:
        L += 1
    return L


def p_at_L(L: int, dim: int, eps0: float, box_factor: float, nu_box: float,
           p_c: float | None = None) -> float:
    """p(L) = p_c - eps0 / x(L), refusing a rung that leaves (0, p_c).

    Refuses rather than clips, as models/percolation_susceptibility.py's `p_at`:
    a smaller box than the ladder allows is a different experiment, not a
    rounding.
    """
    pc = critical_p(dim) if p_c is None else float(p_c)
    if eps0 <= 0:
        raise ValueError(f"eps0 must be > 0; got {eps0}")
    p = _p(L, pc, eps0, box_factor, nu_box)
    if not 0.0 < p < pc:
        raise ValueError(
            f"p = p_c - eps0/x(L) = {p:.6g} is not in (0, p_c = {pc:.6g}) at "
            f"L = {L}, eps0 = {eps0}, box_factor = {box_factor}, nu_box = {nu_box}, "
            f"dim = {dim}. The smallest legal side on this ladder is "
            f"L = {smallest_legal_L(dim, eps0, box_factor, nu_box, p_c)}.")
    return p


def percolation_susceptibility_L_gpu(
    L: int,
    n: int = 1,
    *,
    dim: int,
    eps0: float,
    box_factor: float,
    nu_box: float,
    p_c: float | None = None,
    moment: int = 1,
    geometry: str = "torus",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n i.i.d. samples of Y = sum_clusters s^(moment+1) / (p * L^dim) at p = p(L).

    Same Y, same torus and same stream as
    models/percolation_susceptibility_gpu.py at the same (L, p); float64, shape (n,).
    """
    dim, moment = _check_design(dim, moment, geometry)
    L = _side(L)
    p = p_at_L(L, dim, float(eps0), float(box_factor), float(nu_box), p_c)
    return _susceptibility_gpu_at(L, p, n, dim, moment, geometry, rng,
                                  name="percolation_susceptibility_L_gpu", cpu=None)


def _design(params: dict) -> dict:
    missing = [k for k in _REQUIRED if k not in params]
    if missing:
        raise ValueError(
            f"percolation_susceptibility_L_gpu needs {list(_REQUIRED)} in params "
            f"(no defaults: they are the design); missing {missing}")
    return {"dim": int(params["dim"]), "eps0": float(params["eps0"]),
            "box_factor": float(params["box_factor"]),
            "nu_box": float(params["nu_box"])}


def simulate(L: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_L_gpu"].simulate.

    params: {"dim": int, "eps0": float, "box_factor": float (c), "nu_box": float}
            all four required; {"p_c": float, "moment": 1|2|3 (default 1),
            "geometry": "torus" (default) | "cylinder" | "box"} optional.
    """
    return percolation_susceptibility_L_gpu(
        L,
        n=n,
        p_c=params.get("p_c"),
        moment=int(params.get("moment", 1)),
        geometry=params.get("geometry", "torus"),
        rng=rng,
        **_design(params),
    )


def _dim(params: dict | None) -> int:
    if not params or "dim" not in params:
        raise ValueError("percolation_susceptibility_L_gpu needs params['dim'] "
                         "(no default: it is the design)")
    return _check_dim(params["dim"])


def cost_hint(L: int, params: dict | None = None) -> float:
    """Work for one sample: L**dim sites, exactly (integer arithmetic, then float)."""
    return float(_side(L) ** _dim(params))


def declared_cost_exponent(params: dict) -> float:
    """dim -- cost is L**dim with no ceiling, so what a measured exponent should reproduce."""
    return float(_dim(params))
