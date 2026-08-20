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

Downstream tools (src/plot_loglog.py) take a **run directory**
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
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))  # models/persistence live there, as bare imports

from allocation import neyman_allocation, snr_allocation  # noqa: E402
from models import get_model  # noqa: E402
from persistence import (  # noqa: E402
    content_id,
    load_metadata,
    normalize_scales_n,
    open_scale_writer,
    run_dir as _run_dir,
    save_samples,
    write_metadata,
)


def _clear_other_layout(rd: Path, chunked: bool) -> None:
    """Remove whichever of samples.npz / samples/ this run did NOT just write."""
    import shutil

    if chunked:
        stale = rd / "samples.npz"
        if stale.exists():
            stale.unlink()
    else:
        stale_dir = rd / "samples"
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)


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
    max_chunk_bytes: int = 1_000_000_000,
    mem_flush_pct: float = 90.0,
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
    max_chunk_bytes : int, optional
        Working budget (bytes, conservatively assumed 8 bytes/sample) for
        one generation step. Only matters when out_dir is given: if the
        run's total estimated size (sum(n) * 8 bytes) exceeds this budget,
        every scale is streamed straight to its own on-disk array
        (tools.persistence.open_scale_writer, <out_dir>/<tag>/samples/<i>.npy)
        in chunks of this size, instead of being assembled fully in RAM and
        saved once at the end -- avoids OOM on very large `n` (e.g. a naive
        single-shot n=1e8 SRW scale needed hundreds of GiB; see PLAN.md).
        Runs under the budget use the original, simpler in-RAM path
        unchanged, still producing a single samples.npz.
    mem_flush_pct : float, optional
        Only consulted in the chunked path: if system memory usage
        (psutil.virtual_memory().percent) reaches this after a chunk, the
        in-progress scale's memmap is flushed to disk immediately and the
        chunk size is halved (floor 1000) for everything generated after
        that -- a backstop for when max_chunk_bytes alone wasn't
        conservative enough (e.g. other processes competing for RAM).

    Returns
    -------
    dict[int, np.ndarray]
        Maps each requested scale to its array of `n` i.i.d. samples. For a
        chunked run, values are disk-backed memmaps rather than plain
        in-RAM arrays (still valid np.ndarray for callers).
    """
    spec = get_model(model)
    scales_list, n_list = normalize_scales_n(scales, n)

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    chunked = out_dir is not None and sum(n_list) * 8 > max_chunk_bytes
    rd = None
    if out_dir is not None:
        stem = tag or content_id(params, scales_list, n_list, seed_seq.entropy)
        rd = _run_dir(out_dir, stem)

    chunk_state = {"chunk_n": max(1, max_chunk_bytes // 8)}

    def _generate_scale_chunked(i: int, n_i: int) -> np.memmap:
        mm = None
        offset = 0
        while offset < n_i:
            take = min(chunk_state["chunk_n"], n_i - offset)
            chunk = spec.simulate(i, take, params, rng)
            if mm is None:
                mm = open_scale_writer(rd, i, n_i, chunk.dtype)
            mm[offset:offset + take] = chunk
            offset += take
            del chunk
            pct = psutil.virtual_memory().percent
            if pct >= mem_flush_pct:
                mm.flush()
                chunk_state["chunk_n"] = max(1000, chunk_state["chunk_n"] // 2)
                print(
                    f"[generate] memory at {pct:.1f}% -- flushed scale i={i} "
                    f"({offset}/{n_i} samples), shrinking chunk size to "
                    f"{chunk_state['chunk_n']}",
                    file=sys.stderr,
                )
        mm.flush()
        return mm

    samples: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for idx, (i, n_i) in enumerate(zip(scales_list, n_list), start=1):
        t0 = time.perf_counter()
        samples[i] = _generate_scale_chunked(i, n_i) if chunked else spec.simulate(i, n_i, params, rng)
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
        if not chunked:
            save_samples(rd, samples)
        # Drop the *other* layout's leftovers. A run directory is written
        # deterministically and overwritten on rerun (ground rule 6), but the
        # two layouts live at different paths, so a rerun that crosses the
        # chunking threshold in either direction would otherwise leave the
        # previous run's data sitting alongside the new run's. That is not
        # merely untidy: load_samples() resolves a flat samples.npz BEFORE a
        # samples/ directory, so a stale .npz would silently shadow the run
        # that just finished, and every downstream number would describe the
        # old data. Observed for real on experiments/01_srw/data/omega1 when
        # its allocation rule changed (2026-08-20).
        _clear_other_layout(rd, chunked)
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


def _resolve_n(cfg: dict):
    """Recipe `"n"`: a scalar, an explicit per-scale list, or an allocation rule.

    The rule form keeps a recipe reproducible and self-describing -- it records
    *why* those sample counts were chosen, not just the numbers, so the run can
    be regenerated after a cost-model update by re-reading the recipe rather
    than by remembering how the list was produced:

        "n": {"rule": "neyman", "budget": 2e10, "d": 1.0}

    Two rules, both in tools/allocation.py, neither the same as Proposition
    prop:opt (see those docstrings for why they must not be swapped):
    `neyman` (n_i ~ i^(-d/2)) minimizes the variance of Y_bar itself, and
    `snr` (n_i ~ i^(2*omega1), and so needs an extra "omega1" design input)
    equalizes the signal-to-noise ratio of the CORRECTION term -- which is
    what Experiment B actually estimates, and what `neyman` gets wrong.
    """
    n = cfg["n"]
    if not isinstance(n, dict):
        return n

    rule = n.get("rule")
    common = dict(
        budget=float(n["budget"]),
        d=float(n["d"]),
        sigma=n.get("sigma"),
        min_n=int(n.get("min_n", 1)),
    )
    if rule == "neyman":
        alloc = neyman_allocation(cfg["scales"], **common)
    elif rule == "snr":
        alloc = snr_allocation(cfg["scales"], omega1=float(n["omega1"]), **common)
    else:
        raise ValueError(
            f"unknown allocation rule {rule!r} in recipe 'n'; known: 'neyman', 'snr'"
        )
    print(
        f"allocation rule={rule!r} budget={alloc['budget']:.4g} d={n['d']} -> "
        f"cost={alloc['cost']:.4g} ({alloc['exhausted']:.1%} of budget)"
    )
    if alloc["exhausted"] > 1.0:
        print(
            "  warning: min_n clamp binds, so this allocation OVERSPENDS the stated budget "
            "-- raise the budget or drop the smallest scales",
            file=sys.stderr,
        )
    print(f"  n per scale: {dict(zip(alloc['scales'], alloc['n']))}")
    return alloc["n"]


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
    scales_list, n_list = normalize_scales_n(cfg["scales"], _resolve_n(cfg))

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
    data_path = rd / "samples.npz" if (rd / "samples.npz").exists() else rd / "samples"
    print(f"\nseed     = {resolved_seed}")
    print(f"run_dir  = {rd}")
    print(f"data     = {data_path}")
    print(f"metadata = {rd / 'metadata.json'}")
    print(f"(recipe {args.meta} was not modified)")


if __name__ == "__main__":
    _main()
