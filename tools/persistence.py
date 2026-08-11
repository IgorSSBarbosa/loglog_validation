"""Sample + metadata persistence shared across experiments.

Extracted from experiments/00_synthetic/generator.py once a second experiment
(SRW, experiments/01_srw/generate.py) needed the identical pattern: save
{scale: array of n i.i.d. samples} to a compressed .npz, alongside a JSON
metadata sidecar (params, scales, n, seed, timing, created) at the same
stem. `tools/` may not import from `experiments/` (PLAN.md), so this module
knows nothing about what a "scale" or "params" mean to any particular
experiment -- `params` is just whatever JSON-serializable dict that
experiment wants recorded (e.g. generator.py's `asdict(SyntheticParams(...))`,
or SRW's `{"model": "srw", "q": 0.5}`).

Two kinds of JSON file, not to be confused (same distinction generator.py
already documented, generalized here): a hand-authored **recipe** (read-only,
never modified) vs. **output metadata** written alongside generated data,
with `seed` always resolved to a concrete int so the pair (.npz, .json) is
self-contained and independently reproducible (see each experiment's own
`reproduce`, which is experiment-specific and stays there).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


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


def save_samples(path: str | Path, samples: dict[int, np.ndarray]) -> Path:
    """Save {scale: samples} to a compressed .npz (one array per scale, keyed by str(scale))."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(i): arr for i, arr in samples.items()})
    return path


def load_samples(path: str | Path) -> dict[int, np.ndarray]:
    """Load {scale: samples} back from an .npz written by `save_samples`.

    Raises a clear FileNotFoundError (not a bare one from inside numpy) if
    `path` doesn't exist -- e.g. because it's a recipe's stem, which never
    has data of its own, rather than the .npz path an actual generated run
    produced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no data at {path}.\n"
            f"Generate some first (see this experiment's own generate.py/generator.py -- "
            f"e.g. `python3 generator.py -meta <recipe.json> --tag <name>`), then pass the "
            f"printed 'data' path (data/<name>.npz) here."
        )
    with np.load(path) as npz:
        return {int(k): npz[k] for k in npz.files}


def load_metadata(data_or_metadata_path: str | Path) -> dict | None:
    """Load the metadata dict paired with a data path (same stem, .json extension), or
    a metadata path directly. Returns None (not an error) if there's no metadata file --
    metadata is optional context (e.g. for a target_fn overlay), never required to plot data."""
    json_path = Path(data_or_metadata_path).with_suffix(".json")
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def content_id(params: dict, scales: list[int], n: list[int], seed) -> str:
    """Deterministic filename stem from a run's content, so an identical rerun
    overwrites rather than accumulating (used as the default `tag`)."""
    payload = json.dumps({"params": params, "scales": scales, "n": n, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_metadata(
    *,
    path: str | Path,
    params: dict,
    scales: list[int],
    n: list[int],
    seed,
    timing_seconds: dict[int, float] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
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
