"""Closed-form synthetic model -- registered as MODELS["synthetic"] in tools/models.py.

Realizes Y_i = E[Y_i] * xi_i, with

    E[Y_i] = a0 * i**gamma * exp( sum_j  a_j * i**(-omega_j) )   (article eq. 232)

for an arbitrary number of correction terms (a_j, omega_j), 0 < omega_1 < omega_2
< ..., and xi_i > 0, E[xi_i] = 1, Var(xi_i) = sigma_inf2 (constant-variance
regime, Assumption 6 satisfied exactly by construction rather than only in the
limit). The noise family for xi_i is selected by name through `NOISE_FAMILIES`
so a new family can be added later (e.g. an additive-Gaussian comparison)
without touching the sampling plumbing. Only "lognormal" is implemented so
far, per current sign-off.

A `simulate` call optionally BURNS CPU for `cost_scale * i**cost_d` seconds
per sample, so the model can carry a chosen cost exponent d as well as a
chosen gamma. Off by default (`cost_scale = 0`), and it never touches `rng`,
so switching it on cannot change a single drawn value. See `_burn` for why it
spins rather than sleeps, and calibration/check_no_leakage.py for what it is
for: without it this model has d = 0 exactly, which is outside every
allocation formula and cannot exercise tools/cost_model.py at all.

Ground truth (gamma, a0, ...) is planted and known, so MODELS["synthetic"]
supplies both `target_fn` (the exact E[Y_i] curve) and `true_gamma_key`,
letting src/report/plot_loglog.py overlay the reference curve and run
tools/loglog.py's gamma-hat estimators against it -- this is currently the
only model that does, since it's the only one with a known closed form
(contrast models/srw.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _lognormal_xi(rng: np.random.Generator, size: tuple[int, ...], sigma_inf2: float) -> np.ndarray:
    """xi ~ Lognormal with E[xi] = 1, Var(xi) = sigma_inf2 exactly.

    sigma_inf2 = 0 degenerates to xi == 1 (noiseless power law), which is the
    exact-recovery case checkpoint 0.2 needs.
    """
    tau2 = np.log1p(sigma_inf2)
    z = rng.standard_normal(size)
    return np.exp(np.sqrt(tau2) * z - tau2 / 2.0)


NOISE_FAMILIES: dict[str, Callable[[np.random.Generator, tuple[int, ...], float], np.ndarray]] = {
    "lognormal": _lognormal_xi,
}

#: Refuse a single `simulate` call that would spin longer than this. The burn
#: is n * cost_scale * i**cost_d seconds, which grows in three arguments at
#: once, so a mistyped cost_scale is a wedged machine rather than an error.
#: Raised deliberately high: this is a guard against a typo, not a policy.
MAX_BURN_SECONDS = 300.0


def _burn(seconds: float) -> None:
    """Hold the CPU for `seconds`, consuming no randomness and no memory.

    SPINS rather than sleeps, for four reasons, in decreasing order of
    seriousness (the whole point of the fixture dies with any of them):

    1. `time.sleep` voids the budget model's meaning. `cost_hint` is
       denominated in work units and src/study/pilot.py converts units to
       seconds through a MEASURED throughput for this machine; that conversion
       is only meaningful while elapsed time is proportional to work done. A
       sleeping model burns wall clock doing nothing, so throughput becomes a
       number with no referent and every wall-clock prediction downstream is
       validated against a fiction.
    2. It is the exact failure mode tools/cost_model.py's `compare_cost_models`
       exists to detect ("the machine is no longer compute-bound"). A sleeping
       fixture would permanently trip the diagnostic meant to catch broken
       measurements.
    3. `time.sleep`'s jitter is ADDITIVE (~1 ms granularity on Linux,
       scheduler-dependent overshoot independent of the requested duration),
       which lands in exactly the overhead term `a` of cost(i) = a + b*i**d
       that this fixture is supposed to keep small.
    4. Sleeping N jobs takes the same wall clock as sleeping one, so any future
       concurrent replicate run would show a speedup real work never gives.

    Also deliberately NOT an array burn (`np.sqrt(np.arange(m)).sum()`): for
    large m the array falls out of cache and the cost PER ELEMENT rises, so the
    realized exponent comes out biased above the requested one, as curvature in
    the middle of the probe window that `fit_cost_probe` would misread as a bad
    fit. A spin touches no memory and has no cache cliff.

    `rng` is deliberately not a parameter: the burn must not consume the random
    stream, or seeds stop reproducing and switching the burn on would change
    the answers it is supposed to be invisible to.
    """
    if seconds <= 0:
        return
    if seconds > MAX_BURN_SECONDS:
        raise ValueError(
            f"one simulate() call would spin for {seconds:.1f}s, over "
            f"MAX_BURN_SECONDS={MAX_BURN_SECONDS:g}. The burn is "
            f"n * cost_scale * i**cost_d; check cost_scale against the largest "
            f"(i, n) this model will be called with.")
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@dataclass(frozen=True)
class SyntheticParams:
    """Planted constants for E[Y_i] = a0 * i**gamma * exp(sum_j a_j * i**(-omega_j)).

    `corrections` is a sequence of (a_j, omega_j) pairs, as many as wanted
    (including none, for a pure power law). It is normalized to a tuple of
    float pairs and validated against article eq. (232)'s ordering
    0 < omega_1 < omega_2 < ... on construction, regardless of whether it was
    built from a Python list, a tuple, or JSON (where it round-trips as a list
    of 2-element lists).
    """

    gamma: float
    a0: float = 1.0
    corrections: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    sigma_inf2: float = 0.0
    family: str = "lognormal"
    #: Work burn per sample: `cost_scale` seconds at i = 1, growing as
    #: i**cost_d. cost_scale = 0 (the default) means no burn at all, and then
    #: this model costs the same at every scale, i.e. d = 0.
    cost_d: float = 0.0
    cost_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in NOISE_FAMILIES:
            raise ValueError(f"unknown noise family {self.family!r}; known: {list(NOISE_FAMILIES)}")
        corrections = tuple((float(a), float(omega)) for a, omega in self.corrections)
        omegas = [omega for _, omega in corrections]
        if any(omega <= 0 for omega in omegas):
            raise ValueError("all correction omega_j must be > 0 (article eq. 232)")
        if omegas != sorted(omegas) or len(set(omegas)) != len(omegas):
            raise ValueError(
                "corrections must be ordered by strictly increasing omega_j: "
                "0 < omega_1 < omega_2 < ... (article eq. 232)"
            )
        if self.cost_d < 0:
            raise ValueError(f"cost_d must be >= 0; got {self.cost_d}")
        if self.cost_scale < 0:
            raise ValueError(f"cost_scale must be >= 0 (0 disables the burn); "
                             f"got {self.cost_scale}")
        object.__setattr__(self, "corrections", corrections)

    @property
    def omega1(self) -> float | None:
        """Leading (smallest) correction-to-scaling exponent, or None if there is no correction."""
        return self.corrections[0][1] if self.corrections else None

    @property
    def a1(self) -> float | None:
        """Coefficient of the leading correction term, or None if there is no correction."""
        return self.corrections[0][0] if self.corrections else None


def params_from_dict(d: dict) -> SyntheticParams:
    return SyntheticParams(
        gamma=float(d["gamma"]),
        a0=float(d.get("a0", 1.0)),
        corrections=tuple((float(a), float(omega)) for a, omega in d.get("corrections", [])),
        sigma_inf2=float(d.get("sigma_inf2", 0.0)),
        family=d.get("family", "lognormal"),
        cost_d=float(d.get("cost_d", 0.0)),
        cost_scale=float(d.get("cost_scale", 0.0)),
    )


def mean_Y(i, params: SyntheticParams | dict) -> np.ndarray:
    """E[Y_i], article eq. (232)."""
    if isinstance(params, dict):
        params = params_from_dict(params)
    i = np.asarray(i, dtype=np.float64)
    correction = 0.0
    for a, omega in params.corrections:
        correction = correction + a * i ** (-omega)
    return params.a0 * i**params.gamma * np.exp(correction)


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["synthetic"].simulate: n i.i.d. samples of Y_i (params: SyntheticParams fields, as a dict).

    Draws first, then burns: the burn is `n * cost_scale * i**cost_d` seconds
    of spinning (nothing at all when `cost_scale = 0`, which is the default).
    It comes after the draw and touches neither `rng` nor `y`, so the returned
    array is BIT-IDENTICAL with the burn on and off at the same seed -- the
    same invariance models/srw.py guarantees over `block_n`, and for the same
    reason: a knob that changes the numbers is not a cost knob.

    The burn is computed here from `cost_scale`/`cost_d` rather than by calling
    `cost_hint`, on purpose. `cost_hint` returns 1.0 when there is no burn --
    one WORK UNIT, which is the right answer for an allocation and the wrong
    one for a clock. Feeding it to `_burn` (written that way first) made every
    unburnt call spin for a full second per sample: n * 1.0 read as seconds.
    Two different units, one number, and only the burn path can tell them
    apart -- so it does its own arithmetic.
    """
    p = params_from_dict(params)
    draw_xi = NOISE_FAMILIES[p.family]
    xi = draw_xi(rng, (n,), p.sigma_inf2)
    y = mean_Y(i, p) * xi
    assert np.all(y > 0), f"Assumption 2 (Y_i > 0) violated at scale i={i} for family={p.family!r}"
    _burn(n * p.cost_scale * float(i) ** p.cost_d)
    return y


def target_fn(i, params: dict) -> np.ndarray:
    """MODELS["synthetic"].target_fn: the exact E[Y_i] reference curve (article eq. 232)."""
    return mean_Y(i, params)


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample: `cost_scale * i**cost_d` seconds, or 1 with no burn.

    Without the burn the synthetic generator draws from a closed-form mean plus
    noise: the scale enters the FORMULA, not the amount of work, so d = 0 and
    every scale costs the same. That is correct rather than a defect for a
    statistical testbed -- but it is also unusable as a COST testbed, and
    worse than degenerate: d = 0 is outside the allocation formulas
    (tools/allocation.py's `allocation_constants` raises for d <= 0, and
    `total_cost`'s G divides by rho**d - 1). Which is why the allocation
    experiments all run on srw.

    `cost_scale > 0` turns this model into a cost testbed with ANY exponent,
    exactly known: `simulate` spins for `cost_scale * i**cost_d` seconds per
    sample, so the exponent declared here is the exponent the clock has to
    find. Used by calibration/check_no_leakage.py, whose whole question is
    whether tools/cost_model.py recovers a d it was never told -- which cannot
    be asked of a model with only one possible d, still less of one whose d is
    the constant the estimator would have to guess.

    The unit is seconds here rather than the model's own work unit (srw counts
    steps). That is not an inconsistency: only RATIOS across scales matter to
    an allocation, so the unit is free, and for a spin the natural unit really
    is elapsed time.
    """
    p = params or {}
    scale = float(p.get("cost_scale", 0.0))
    if scale <= 0:
        return 1.0
    return scale * float(i) ** float(p.get("cost_d", 0.0))
