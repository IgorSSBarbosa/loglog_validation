"""Sample + metadata persistence shared across every model (src/generate.py,
src/measure_cost.py) and consumed by src/plot_loglog.py/src/plot_cost.py.

Each run gets its own directory, named by `tag`: `<out_dir>/<tag>/samples.npz`
(a compressed {scale: array of n i.i.d. samples} archive) alongside
`<out_dir>/<tag>/metadata.json` (model, params, scales, n, seed, timing,
created) -- one run, one self-contained folder, rather than a flat
`<out_dir>/<tag>.npz` + `<out_dir>/<tag>.json` pair. Once there are dozens of
runs (different tags, different models) this keeps `data/` from turning into
an unsorted pile of same-prefixed files.

For runs too large to build entirely in RAM, `open_scale_writer` gives an
alternative: `<out_dir>/<tag>/samples/<scale>.npy`, one real on-disk array per
scale, written into in slices via a memmap instead of assembled in memory and
saved all at once. `load_samples` reads either layout transparently.

Two kinds of JSON file, not to be confused: a hand-authored **recipe**
(read-only, never modified by anything here) vs. **output metadata** written
alongside generated data, with `seed` always resolved to a concrete int so
the run directory is self-contained and independently reproducible (see each
driver's own `reproduce`).

`tools/` may not import from `experiments/` (PLAN.md), so this module knows
nothing about what a "scale" or "params" mean to any particular model --
`params` is just whatever JSON-serializable dict that model's `simulate`
function wants (see tools/models.py), and `model` is just the registry name
used to look it up again on load.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from artifacts import artifact_path, read_artifact


def normalize_scales_n(scales, n) -> tuple[list[int], list[int]]:
    """Scalar-or-sequence `n` -> matched (scales, n) integer lists, same length."""
    scales_arr = np.atleast_1d(np.asarray(scales, dtype=np.int64))
    if np.ndim(n) == 0:
        n_arr = np.full(scales_arr.shape, int(n), dtype=np.int64)
    else:
        n_arr = np.asarray(n, dtype=np.int64)
        if n_arr.shape != scales_arr.shape:
            raise ValueError("n must be a scalar or match scales in length")
    return scales_arr.tolist(), n_arr.tolist()


def run_dir(out_dir: str | Path, tag: str) -> Path:
    """The directory a run's samples.npz + metadata.json live in: <out_dir>/<tag>/."""
    return Path(out_dir) / tag


def save_samples(run_dir: str | Path, samples: dict[int, np.ndarray]) -> Path:
    """Save {scale: samples} to <run_dir>/samples.npz (compressed, one array
    per scale, keyed by str(scale)). Creates run_dir if needed."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "samples.npz"
    np.savez_compressed(path, **{str(i): arr for i, arr in samples.items()})
    return path


def open_scale_writer(run_dir: str | Path, scale: int, n: int, dtype) -> np.memmap:
    """Create <run_dir>/samples/<scale>.npy as a writable on-disk array of
    length n, for streaming a scale's samples in via slices instead of
    holding all n of them in RAM at once. Creates run_dir/samples/ if
    needed. Caller is responsible for filling every slot and, once done,
    dropping the returned memmap (this flushes it to disk)."""
    samples_dir = Path(run_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{scale}.npy"
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n,))


def load_samples(run_dir: str | Path) -> dict[int, np.ndarray]:
    """Load {scale: samples} back from a run directory, whichever of the two
    layouts it was written in: a single <run_dir>/samples.npz (the common
    case, everything fit comfortably in RAM), or <run_dir>/samples/*.npy
    (one memmapped file per scale, for runs streamed via open_scale_writer
    -- loaded with mmap_mode="r" so callers only pull in the pages they
    actually touch).

    Raises a clear FileNotFoundError (not a bare one from inside numpy) if
    neither is present -- e.g. `run_dir` names a recipe's tag that was
    never actually generated.
    """
    run_dir = Path(run_dir)
    npz_path = run_dir / "samples.npz"
    if npz_path.exists():
        with np.load(npz_path) as npz:
            return {int(k): npz[k] for k in npz.files}

    samples_dir = run_dir / "samples"
    if samples_dir.exists():
        npy_paths = sorted(samples_dir.glob("*.npy"))
        if npy_paths:
            return {int(p.stem): np.load(p, mmap_mode="r") for p in npy_paths}

    raise FileNotFoundError(
        f"no data at {npz_path} or {samples_dir}/.\n"
        f"Generate some first (src/generate.py -meta <recipe.json> --tag <name>), "
        f"then pass the printed run directory here."
    )


def load_metadata(run_dir: str | Path) -> dict | None:
    """Load <run_dir>/metadata.json. Returns None (not an error) if there's no
    metadata file -- metadata is optional context (e.g. for a target_fn
    overlay), never required just to plot data."""
    path = artifact_path(Path(run_dir), "samples_meta")
    if not path.exists():
        return None
    return json.loads(path.read_text())


def content_id(params: dict, scales: list[int], n: list[int], seed) -> str:
    """Deterministic tag from a run's content, so an identical rerun
    overwrites rather than accumulating a new run directory."""
    payload = json.dumps({"params": params, "scales": scales, "n": n, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_metadata(
    *,
    run_dir: str | Path,
    model: str,
    params: dict,
    scales: list[int],
    n: list[int],
    seed,
    timing_seconds: dict[int, float] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_path(run_dir, "samples_meta")
    meta = {
        "model": model,
        "params": params,
        "scales": scales,
        "n": n,
        "seed": seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if timing_seconds is not None:
        meta["timing_seconds"] = {str(i): t for i, t in timing_seconds.items()}
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path
