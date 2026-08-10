"""Synthetic data generator for Phase 0 (see ../../PLAN.md).

Realizes Y_i = E[Y_i] * xi_i, with

    E[Y_i] = a0 * i**gamma * exp( sum_j  a_j * i**(-omega_j) )   (article eq. 232)

for an arbitrary number of correction terms (a_j, omega_j), 0 < omega_1 < omega_2
< ..., and xi_i > 0, E[xi_i] = 1, Var(xi_i) = sigma_inf2 (constant-variance
regime, Assumption 6 satisfied exactly by construction rather than only in the
limit).

The noise family for xi_i is selected by name through `NOISE_FAMILIES` so a new
family can be added later (e.g. an additive-Gaussian comparison) without
touching the sampling or reproducibility plumbing below. Only "lognormal" is
implemented so far, per current sign-off.

Reproducibility: every metadata JSON this module writes or reads has the same
shape -- {"params": {...}, "scales": [...], "n": [...], "seed": <int or null>,
"created": "..."} -- and fully determines its samples: same params + scales +
n + seed always regenerates bit-identical data (see `reproduce`). Raw sample
arrays are intentionally never persisted: they're cheap to regenerate exactly
from the JSON, and not saving them avoids data sprawl (PLAN.md ground rule 6).

CLI usage (reads a config, generates, writes the resolved seed back so the
same file becomes a reproducible record -- see example_config.json):

    python3 generator.py -meta example_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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


def mean_Y(i, params: SyntheticParams) -> np.ndarray:
    """E[Y_i], article eq. (232)."""
    i = np.asarray(i, dtype=np.float64)
    correction = 0.0
    for a, omega in params.corrections:
        correction = correction + a * i ** (-omega)
    return params.a0 * i**params.gamma * np.exp(correction)


def generate(
    scales,
    n,
    params: SyntheticParams,
    *,
    seed: int | None = None,
    out_dir: str | Path | None = None,
    tag: str | None = None,
) -> dict[int, np.ndarray]:
    """Draw i.i.d. synthetic samples Y_i at each requested scale.

    Parameters
    ----------
    scales : int or sequence of int
        System size(s) i to sample at.
    n : int or sequence of int
        Number of i.i.d. samples per scale. A scalar applies the same n to
        every scale; a sequence must match `scales` in length.
    params : SyntheticParams
        Planted constants and noise family (see NOISE_FAMILIES).
    seed : int, optional
        RNG seed. If omitted, fresh OS entropy is drawn and recorded in the
        metadata, so the run is still exactly reproducible from the JSON.
    out_dir : path, optional
        If given, a metadata JSON sidecar is written here (see module docstring).
    tag : str, optional
        Filename (without extension) for the metadata file. Defaults to a
        hash of the run's content, so identical reruns overwrite rather than
        accumulate.

    Returns
    -------
    dict[int, np.ndarray]
        Maps each requested scale to its array of `n` i.i.d. samples of Y_i.
    """
    scales_arr = np.atleast_1d(np.asarray(scales, dtype=np.int64))
    if np.ndim(n) == 0:
        n_per_scale = np.full(scales_arr.shape, int(n), dtype=np.int64)
    else:
        n_per_scale = np.asarray(n, dtype=np.int64)
        if n_per_scale.shape != scales_arr.shape:
            raise ValueError("n must be a scalar or match scales in length")

    draw_xi = NOISE_FAMILIES[params.family]

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    samples: dict[int, np.ndarray] = {}
    for i, n_i in zip(scales_arr.tolist(), n_per_scale.tolist()):
        xi = draw_xi(rng, (n_i,), params.sigma_inf2)
        y = mean_Y(i, params) * xi
        assert np.all(y > 0), f"Assumption 2 (Y_i > 0) violated at scale i={i} for family={params.family!r}"
        samples[i] = y

    if out_dir is not None:
        _write_metadata(
            path=Path(out_dir) / f"{tag or _content_id(params, scales_arr.tolist(), n_per_scale.tolist(), seed_seq.entropy)}.json",
            params=params,
            scales=scales_arr.tolist(),
            n=n_per_scale.tolist(),
            seed=seed_seq.entropy,
        )

    return samples


def reproduce(metadata_path: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a metadata JSON written by `generate` or the CLI."""
    meta = json.loads(Path(metadata_path).read_text())
    params = SyntheticParams(**meta["params"])
    return generate(meta["scales"], meta["n"], params, seed=meta["seed"])


def _content_id(params: SyntheticParams, scales: list[int], n: list[int], seed) -> str:
    payload = json.dumps({"params": asdict(params), "scales": scales, "n": n, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _write_metadata(*, path: Path, params: SyntheticParams, scales: list[int], n: list[int], seed) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "params": asdict(params),
        "scales": scales,
        "n": n,
        "seed": seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def params_from_json(d: dict) -> SyntheticParams:
    return SyntheticParams(
        gamma=float(d["gamma"]),
        a0=float(d.get("a0", 1.0)),
        corrections=tuple((float(a), float(omega)) for a, omega in d.get("corrections", [])),
        sigma_inf2=float(d.get("sigma_inf2", 0.0)),
        family=d.get("family", "lognormal"),
    )


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw synthetic Y_i samples from a JSON experiment config, and write the "
            "config back to the same file with the RNG seed resolved -- so re-running "
            "the same file reproduces the exact same data instead of drawing fresh. "
            "See example_config.json for the expected shape."
        )
    )
    parser.add_argument(
        "-meta",
        "--meta",
        dest="meta",
        required=True,
        type=Path,
        help='JSON config: {"params": {"gamma": ..., "a0": ..., "corrections": [[a1, omega1], ...], '
        '"sigma_inf2": ..., "family": "lognormal"}, "scales": [...], "n": ..., "seed": null}',
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    params = params_from_json(cfg["params"])
    scales = cfg["scales"]
    n = cfg["n"]

    # Resolve the seed up front (drawing fresh entropy if the config had none)
    # so we know what to write back before generating.
    seed_seq = np.random.SeedSequence(cfg.get("seed"))
    resolved_seed = seed_seq.entropy
    samples = generate(scales, n, params, seed=resolved_seed)

    scales_arr = np.atleast_1d(np.asarray(scales, dtype=np.int64))
    n_arr = np.full(scales_arr.shape, int(n), dtype=np.int64) if np.ndim(n) == 0 else np.asarray(n, dtype=np.int64)
    _write_metadata(
        path=args.meta,
        params=params,
        scales=scales_arr.tolist(),
        n=n_arr.tolist(),
        seed=resolved_seed,
    )

    print(f"family={params.family!r}  gamma={params.gamma}  corrections={params.corrections}  sigma_inf2={params.sigma_inf2}")
    print(f"{'i':>12} {'n':>8} {'target':>14} {'sample_mean':>14}")
    for i in np.atleast_1d(scales):
        i = int(i)
        y = samples[i]
        print(f"{i:>12} {len(y):>8} {mean_Y(i, params).item():>14.4f} {y.mean():>14.4f}")
    print(f"\nResolved seed written back to {args.meta}")


if __name__ == "__main__":
    _main()
