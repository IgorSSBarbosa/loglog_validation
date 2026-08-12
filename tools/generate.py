"""Single, shared sample generator for every model (tools/models.py) -- the
recipe's "model" field picks which one runs, instead of each experiment
keeping its own copy of this driver.

Two kinds of JSON file appear here, and they are NOT the same file:

  - a "recipe": authored by hand (or by any tool), holding {"model": ...,
    "params": {...}, "scales": [...], "n": ..., "seed": <int or null>}. The
    CLI's `-meta` argument takes a recipe and never modifies it -- rewriting
    a file the caller authored, out from under them, doesn't make sense.
  - "output metadata": written by `generate(..., out_dir=..., tag=...)` (and
    by the CLI) to `<out_dir>/<tag>/metadata.json`, alongside the actual
    samples at `<out_dir>/<tag>/samples.npz` (see tools/persistence.py).
    Same shape as a recipe but `seed` always resolved to a concrete int, and
    "timing_seconds" (wall-clock draw time per scale) added -- the raw
    material for a future meta-log-log plot of cost(i) vs i (article
    Assumption cost_is_power_law, cost(i) = i**d; see tools/cost_model.py).

Downstream tools (tools/plot_loglog.py) take a **run directory**
(`<out_dir>/<tag>/`), not a recipe -- a single recipe run with different
tags/seeds produces many different run directories, so "give me a recipe"
would be ambiguous about which run you mean.

CLI usage (recipe is read-only; writes <out_dir>/<tag>/{samples.npz,
metadata.json}; default out_dir is `data/` next to the recipe file itself,
so each experiment's recipes keep landing in that experiment's own data/):

    python3 generate.py -meta ../experiments/00_synthetic/example_config.json
    python3 generate.py -meta ../experiments/01_srw/example_config.json --tag demo_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this dir, for models/persistence as bare imports too

from models import get_model  # noqa: E402
from persistence import (  # noqa: E402
    content_id,
    load_metadata,
    normalize_scales_n,
    run_dir as _run_dir,
    save_samples,
    write_metadata,
)


def generate(
    model: str,
    scales,
    n,
    params: dict,
    *,
    seed: int | None = None,
    out_dir: str | Path | None = None,
    tag: str | None = None,
    progress: bool = False,
) -> dict[int, np.ndarray]:
    """Draw i.i.d. samples at each requested scale, via MODELS[model].simulate.

    Parameters
    ----------
    model : str
        Registry name in tools/models.py (e.g. "synthetic", "srw").
    scales : int or sequence of int
        System size(s)/step count(s) to sample at.
    n : int or sequence of int
        Number of i.i.d. samples per scale. A scalar applies the same n to
        every scale; a sequence must match `scales` in length.
    params : dict
        Model-specific parameters, passed straight to MODELS[model].simulate.
    seed : int, optional
        RNG seed. If omitted, fresh OS entropy is drawn (but not recorded
        anywhere unless out_dir is given).
    out_dir : path, optional
        If given, the run is saved to `<out_dir>/<tag>/` (see
        tools/persistence.py) -- samples.npz + metadata.json, the latter
        including per-scale wall-clock time under "timing_seconds".
    tag : str, optional
        Run directory name. Defaults to a hash of the run's content, so an
        identical rerun overwrites rather than accumulating a new directory.
    progress : bool, optional
        Print a one-line-per-scale progress update to stderr as sampling
        proceeds. Off by default so library callers (e.g. a Monte Carlo loop
        calling `generate` hundreds of times) aren't spammed; the CLI turns
        it on.

    Returns
    -------
    dict[int, np.ndarray]
        Maps each requested scale to its array of `n` i.i.d. samples.
    """
    spec = get_model(model)
    scales_list, n_list = normalize_scales_n(scales, n)

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    samples: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for idx, (i, n_i) in enumerate(zip(scales_list, n_list), start=1):
        t0 = time.perf_counter()
        samples[i] = spec.simulate(i, n_i, params, rng)
        timings[i] = time.perf_counter() - t0
        if progress:
            print(
                f"\r[{idx}/{len(scales_list)}] scale i={i} n={n_i} ({timings[i] * 1e3:.1f} ms)"
                + " " * 10,
                end="",
                file=sys.stderr,
                flush=True,
            )
    if progress:
        print(file=sys.stderr)

    if out_dir is not None:
        stem = tag or content_id(params, scales_list, n_list, seed_seq.entropy)
        rd = _run_dir(out_dir, stem)
        save_samples(rd, samples)
        write_metadata(
            run_dir=rd,
            model=model,
            params=params,
            scales=scales_list,
            n=n_list,
            seed=seed_seq.entropy,
            timing_seconds=timings,
        )

    return samples


def reproduce(run_dir: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a run directory's metadata,
    from scratch (no I/O on samples.npz) -- an independent check that saved
    data matches what the recorded recipe actually produces."""
    meta = load_metadata(run_dir)
    if meta is None:
        raise FileNotFoundError(f"no metadata.json in {run_dir}")
    return generate(meta["model"], meta["scales"], meta["n"], meta["params"], seed=meta["seed"])


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw samples from a JSON recipe (any model registered in tools/models.py). "
            "The recipe is never modified; output (samples.npz + metadata.json) is written "
            "to --out-dir/<tag>/. See experiments/*/example_config.json for recipe shapes."
        )
    )
    parser.add_argument(
        "-meta", "--meta", dest="meta", required=True, type=Path,
        help='Recipe JSON (read-only): {"model": "synthetic"|"srw", "params": {...}, '
        '"scales": [...], "n": ..., "seed": null}',
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", type=Path, default=None,
        help="Output directory for <tag>/. Defaults to a 'data' directory next to the recipe file.",
    )
    parser.add_argument(
        "--tag", dest="tag", type=str, default=None,
        help="Run directory name. Defaults to a content hash, so an identical rerun overwrites "
        "rather than accumulating; pass a fixed --tag for a predictable path to chain into other tools.",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    model = cfg["model"]
    spec = get_model(model)
    params = cfg["params"]
    scales_list, n_list = normalize_scales_n(cfg["scales"], cfg["n"])

    seed_seq = np.random.SeedSequence(cfg.get("seed"))
    resolved_seed = seed_seq.entropy

    out_dir = args.out_dir or (args.meta.resolve().parent / "data")
    samples = generate(
        model, scales_list, n_list, params, seed=resolved_seed, out_dir=out_dir, tag=args.tag, progress=True
    )

    stem = args.tag or content_id(params, scales_list, n_list, resolved_seed)
    rd = _run_dir(out_dir, stem)
    timing = load_metadata(rd)["timing_seconds"]

    has_target = spec.target_fn is not None
    print(f"model={model!r}  params={params}")
    header = f"{'i':>12} {'n':>8} " + (f"{'target':>14} " if has_target else "") + f"{'sample_mean':>14} {'elapsed_ms':>12}"
    print(header)
    for i in scales_list:
        y = samples[i]
        elapsed_ms = timing[str(i)] * 1e3
        row = f"{i:>12} {len(y):>8} "
        if has_target:
            row += f"{spec.target_fn(i, params).item():>14.4f} "
        row += f"{y.mean():>14.4f} {elapsed_ms:>12.2f}"
        print(row)
    print(f"\nseed     = {resolved_seed}")
    print(f"run_dir  = {rd}")
    print(f"data     = {rd / 'samples.npz'}")
    print(f"metadata = {rd / 'metadata.json'}")
    print(f"(recipe {args.meta} was not modified)")


if __name__ == "__main__":
    _main()
