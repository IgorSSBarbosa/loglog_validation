"""Synthetic data generator for Phase 0 (see ../../PLAN.md, checkpoint 0.1).

Realizes Y_i = E[Y_i] * xi_i, with

    E[Y_i] = a0 * i**gamma * exp(a1 * i**(-omega1))     (article eq. 232, J=1)

and xi_i > 0, E[xi_i] = 1, Var(xi_i) = sigma_inf2 (constant-variance regime,
Assumption 6 satisfied exactly by construction rather than only in the limit).

The noise family for xi_i is selected by name through `NOISE_FAMILIES` so a new
family can be added later (e.g. an additive-Gaussian comparison) without
touching the sampling or reproducibility plumbing below. Only "lognormal" is
implemented so far, per current sign-off.

Reproducibility: `generate(..., out_dir=...)` writes a JSON sidecar recording
every planted parameter, the scales, sample counts, and the exact RNG seed
entropy used. `reproduce(path)` reads such a file back and regenerates
bit-identical samples. Raw sample arrays are intentionally not persisted here:
they're cheap to regenerate exactly from the JSON, and not saving them avoids
data sprawl (PLAN.md ground rule 6). The default output filename is a hash of
the run's content, not a timestamp, so re-running an unchanged configuration
overwrites its old file instead of accumulating a new one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
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
    """Planted constants for E[Y_i] = a0 * i**gamma * exp(a1 * i**(-omega1))."""

    gamma: float
    a0: float = 1.0
    a1: float = 0.0
    omega1: float = 1.0
    sigma_inf2: float = 0.0
    family: str = "lognormal"


def mean_Y(i, params: SyntheticParams) -> np.ndarray:
    """E[Y_i], article eq. (232) with J=1."""
    i = np.asarray(i, dtype=np.float64)
    return params.a0 * i**params.gamma * np.exp(params.a1 * i ** (-params.omega1))


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

    if params.family not in NOISE_FAMILIES:
        raise ValueError(f"unknown noise family {params.family!r}; known: {list(NOISE_FAMILIES)}")
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
            out_dir=Path(out_dir),
            tag=tag,
            params=params,
            scales=scales_arr.tolist(),
            n=n_per_scale.tolist(),
            seed_entropy=seed_seq.entropy,
        )

    return samples


def reproduce(metadata_path: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a metadata JSON written by `generate`."""
    meta = json.loads(Path(metadata_path).read_text())
    params = SyntheticParams(**meta["params"])
    return generate(meta["scales"], meta["n"], params, seed=meta["seed_entropy"])


def _content_id(params: SyntheticParams, scales: list[int], n: list[int], seed_entropy) -> str:
    payload = json.dumps(
        {"params": asdict(params), "scales": scales, "n": n, "seed_entropy": seed_entropy},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _write_metadata(
    *,
    out_dir: Path,
    tag: str | None,
    params: SyntheticParams,
    scales: list[int],
    n: list[int],
    seed_entropy,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or _content_id(params, scales, n, seed_entropy)
    meta = {
        "generator": "experiments/00_synthetic/generator.py:generate",
        "params": asdict(params),
        "scales": scales,
        "n": n,
        "seed_entropy": seed_entropy,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path
