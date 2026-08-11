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

Two kinds of JSON file appear in this module, and they are NOT the same file:

  - a "recipe": authored by hand (or by any tool), holding {"params": {...},
    "scales": [...], "n": ..., "seed": <int or null>}. The CLI's `-meta`
    argument takes a recipe and never modifies it -- rewriting a file the
    caller authored, out from under them, doesn't make sense.
  - "output metadata": written by `generate(..., out_dir=..., tag=...)` (and
    by the CLI) to `<out_dir>/<tag>.json`, alongside the actual samples at
    `<out_dir>/<tag>.npz` (numpy's compressed format -- no new dependency).
    The metadata has the same shape as a recipe but with `seed` always
    resolved to a concrete int, so the pair (.npz, .json) is self-contained:
    the .npz is the data, the .json is everything needed to regenerate it
    from scratch as a correctness check (see `reproduce`), or just to read
    what was run without loading numpy at all.

`load(path)` reads the persisted .npz paired with a metadata/recipe path
(same stem, .npz extension) directly -- fast, no recomputation.
`reproduce(path)` instead regenerates from the recorded params/scales/n/seed
-- slower, but a genuine independent check that the saved data matches what
the recipe actually produces.

CLI usage (recipe is read-only; writes <out_dir>/<tag>.npz + .json, default
out_dir is ./data next to this script):

    python3 generator.py -meta example_config.json
    python3 generator.py -meta example_config.json --tag my_run -o data
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

HERE = Path(__file__).resolve().parent


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


def _normalize_scales_n(scales, n) -> tuple[list[int], list[int]]:
    scales_arr = np.atleast_1d(np.asarray(scales, dtype=np.int64))
    if np.ndim(n) == 0:
        n_arr = np.full(scales_arr.shape, int(n), dtype=np.int64)
    else:
        n_arr = np.asarray(n, dtype=np.int64)
        if n_arr.shape != scales_arr.shape:
            raise ValueError("n must be a scalar or match scales in length")
    return scales_arr.tolist(), n_arr.tolist()


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
        RNG seed. If omitted, fresh OS entropy is drawn (but not recorded
        anywhere unless out_dir is given -- pass an explicit seed, or use the
        CLI / read the returned metadata path, if you need to know it).
    out_dir : path, optional
        If given, the samples are saved to `<out_dir>/<tag>.npz` and their
        metadata to `<out_dir>/<tag>.json` (see module docstring).
    tag : str, optional
        Filename stem for both output files. Defaults to a hash of the run's
        content, so identical reruns overwrite rather than accumulate.

    Returns
    -------
    dict[int, np.ndarray]
        Maps each requested scale to its array of `n` i.i.d. samples of Y_i.
    """
    scales_list, n_list = _normalize_scales_n(scales, n)

    draw_xi = NOISE_FAMILIES[params.family]

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    samples: dict[int, np.ndarray] = {}
    for i, n_i in zip(scales_list, n_list):
        xi = draw_xi(rng, (n_i,), params.sigma_inf2)
        y = mean_Y(i, params) * xi
        assert np.all(y > 0), f"Assumption 2 (Y_i > 0) violated at scale i={i} for family={params.family!r}"
        samples[i] = y

    if out_dir is not None:
        stem = tag or _content_id(params, scales_list, n_list, seed_seq.entropy)
        base = Path(out_dir) / stem
        save_samples(base.with_suffix(".npz"), samples)
        _write_metadata(
            path=base.with_suffix(".json"), params=params, scales=scales_list, n=n_list, seed=seed_seq.entropy
        )

    return samples


def save_samples(path: str | Path, samples: dict[int, np.ndarray]) -> Path:
    """Save {scale: samples} to a compressed .npz (one array per scale, keyed by str(scale))."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(i): arr for i, arr in samples.items()})
    return path


def load_samples(path: str | Path) -> dict[int, np.ndarray]:
    """Load {scale: samples} back from a .npz written by `save_samples`."""
    with np.load(path) as npz:
        return {int(k): npz[k] for k in npz.files}


def load(metadata_or_recipe_path: str | Path) -> dict[int, np.ndarray]:
    """Load the persisted .npz paired with a metadata/recipe JSON (same stem, .npz extension)."""
    path = Path(metadata_or_recipe_path)
    npz_path = path.with_suffix(".npz")
    if not npz_path.exists():
        meta = {}
        if path.exists():
            try:
                meta = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        if "created" in meta:
            reason = f"{path} is generated metadata, but its paired data file is missing (moved or deleted?)."
        else:
            reason = (
                f"{path} looks like a hand-authored recipe, not generated output -- "
                f"recipes don't have data until you run the generator on them."
            )
        raise FileNotFoundError(
            f"{reason}\n"
            f"Expected data at: {npz_path}\n"
            f"Generate it with: python3 generator.py -meta {path} --tag <name>\n"
            f"then point this at the printed 'metadata' path (data/<name>.json), not the recipe."
        )
    return load_samples(npz_path)


def reproduce(metadata_path: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a metadata JSON, from scratch (no I/O on the
    paired .npz) -- an independent check that saved data matches what the recorded recipe
    actually produces. Use `load` instead if you just want the already-computed samples."""
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
            "Draw synthetic Y_i samples from a JSON recipe. The recipe is never modified; "
            "output (samples as .npz + reproducibility metadata as .json, same stem) is "
            "written to --out-dir. See example_config.json for the recipe shape."
        )
    )
    parser.add_argument(
        "-meta",
        "--meta",
        dest="meta",
        required=True,
        type=Path,
        help='Recipe JSON (read-only): {"params": {"gamma": ..., "a0": ..., "corrections": '
        '[[a1, omega1], ...], "sigma_inf2": ..., "family": "lognormal"}, "scales": [...], '
        '"n": ..., "seed": null}',
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory for <tag>.npz + <tag>.json. Defaults to ./data next to this script.",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default=None,
        help="Output filename stem. Defaults to a content hash, so an identical rerun overwrites "
        "rather than accumulating; pass a fixed --tag for a predictable path to chain into other tools.",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    params = params_from_json(cfg["params"])
    scales_list, n_list = _normalize_scales_n(cfg["scales"], cfg["n"])

    seed_seq = np.random.SeedSequence(cfg.get("seed"))
    resolved_seed = seed_seq.entropy

    out_dir = args.out_dir or (HERE / "data")
    samples = generate(scales_list, n_list, params, seed=resolved_seed, out_dir=out_dir, tag=args.tag)

    stem = args.tag or _content_id(params, scales_list, n_list, resolved_seed)

    print(
        f"family={params.family!r}  gamma={params.gamma}  corrections={params.corrections}  "
        f"sigma_inf2={params.sigma_inf2}"
    )
    print(f"{'i':>12} {'n':>8} {'target':>14} {'sample_mean':>14}")
    for i in scales_list:
        y = samples[i]
        print(f"{i:>12} {len(y):>8} {mean_Y(i, params).item():>14.4f} {y.mean():>14.4f}")
    print(f"\nseed     = {resolved_seed}")
    print(f"data     = {out_dir / (stem + '.npz')}")
    print(f"metadata = {out_dir / (stem + '.json')}")
    print(f"(recipe {args.meta} was not modified)")


if __name__ == "__main__":
    _main()
