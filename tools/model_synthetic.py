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

Ground truth (gamma, a0, ...) is planted and known, so MODELS["synthetic"]
supplies both `target_fn` (the exact E[Y_i] curve) and `true_gamma_key`,
letting tools/plot_loglog.py overlay the reference curve and run
tools/loglog.py's gamma-hat estimators against it -- this is currently the
only model that does, since it's the only one with a known closed form
(contrast tools/model_srw.py).
"""

from __future__ import annotations

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
    """MODELS["synthetic"].simulate: n i.i.d. samples of Y_i (params: SyntheticParams fields, as a dict)."""
    p = params_from_dict(params)
    draw_xi = NOISE_FAMILIES[p.family]
    xi = draw_xi(rng, (n,), p.sigma_inf2)
    y = mean_Y(i, p) * xi
    assert np.all(y > 0), f"Assumption 2 (Y_i > 0) violated at scale i={i} for family={p.family!r}"
    return y


def target_fn(i, params: dict) -> np.ndarray:
    """MODELS["synthetic"].target_fn: the exact E[Y_i] reference curve (article eq. 232)."""
    return mean_Y(i, params)
