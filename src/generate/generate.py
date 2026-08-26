"""Single, shared sample generator for every model (tools/models.py) -- the
recipe's "model" field picks which one runs, instead of each experiment
keeping its own copy of this driver.

Two kinds of JSON file appear here, and they are NOT the same file:

  - a "recipe": authored by hand (or by any tool), holding {"model": ...,
    "params": {...}, "scales": [...], "n": ..., "seed": <int or null>}. The
    CLI's `-meta` argument takes a recipe and never modifies it -- rewriting
    a file the caller authored, out from under them, doesn't make sense.
  - "output metadata": written by `generate(..., out_dir=..., tag=...)` (and
    by the CLI) to `<out_dir>/<tag>/samples_meta.json`, alongside the actual
    samples at `<out_dir>/<tag>/samples.npz` (see tools/persistence.py).
    Same shape as a recipe but `seed` always resolved to a concrete int, and
    "timing_seconds" (wall-clock draw time per scale) added -- the raw
    material for a future meta-log-log plot of cost(i) vs i (article
    Assumption cost_is_power_law, cost(i) = i**d; see tools/cost_model.py).

Downstream tools (src/report/plot_loglog.py) take a **run directory**
(`<out_dir>/<tag>/`), not a recipe -- a single recipe run with different
tags/seeds produces many different run directories, so "give me a recipe"
would be ambiguous about which run you mean.

CLI usage (recipe is read-only; writes <out_dir>/<tag>/{samples.npz,
samples_meta.json}; default out_dir is the experiment’s `data/`,
so each experiment's recipes keep landing in that experiment's own data/):

    python3 src/generate/generate.py -meta experiments/00_synthetic/recipes/samples_example.json
    python3 src/generate/generate.py -meta experiments/01_srw/recipes/samples_example.json --tag demo_run
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
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
sys.path.insert(0, str(ROOT / "tools"))      # helper modules, as bare imports

from allocation import neyman_allocation, snr_allocation  # noqa: E402
from artifacts import ARTIFACTS, artifact_path, default_out_dir, load_recipe  # noqa: E402
from cost_model import declared_exponent  # noqa: E402
from rng import as_seed_sequence, seed_record  # noqa: E402
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
    seed=None,
    out_dir: str | Path | None = None,
    tag: str | None = None,
    progress: bool = False,
    reduce=None,
    max_chunk_bytes: int = 1_000_000_000,
    mem_flush_pct: float = 90.0,
) -> dict:
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
    seed : int, SeedSequence, dict, or None
        RNG seed, in any spelling `tools/rng.py`'s `as_seed_sequence` accepts.
        A **SeedSequence is passed through unchanged**, which is what lets a
        caller obeying ground rule 2 hand over one of `SeedSequence.spawn`'s
        children -- `allocation_experiment.sweep` pre-spawns one stream per
        (budget, m0, replicate) cell and calls this function with it. Do NOT
        try to pass a spawned child as an int via `child.entropy`: a child
        carries its PARENT's entropy, so every sibling would collapse onto one
        identical stream (see tools/rng.py's module docstring). If omitted,
        fresh OS entropy is drawn (but not recorded anywhere unless out_dir is
        given). The recorded form is `rng.seed_record`'s, which stays a bare
        int for an un-spawned seed.
    out_dir : path, optional
        If given, the run is saved to `<out_dir>/<tag>/` (see
        tools/persistence.py) -- samples.npz + samples_meta.json, the latter
        including per-scale wall-clock time under "timing_seconds".
    tag : str, optional
        Run directory name. Defaults to a hash of the run's content, so an
        identical rerun overwrites rather than accumulating a new directory.
    progress : bool, optional
        Print a one-line-per-scale progress update to stderr as sampling
        proceeds. Off by default so library callers (e.g. a Monte Carlo loop
        calling `generate` hundreds of times) aren't spammed; the CLI turns
        it on.
    reduce : callable, optional
        Applied to each scale's samples as soon as they are drawn, with the
        raw array dropped immediately after. The point is memory: a caller
        that only wants a statistic (`reduce=np.mean` -- what Experiment C's
        sweep and verify_prediction use) then retains one scale's array at a
        time instead of the whole ladder. Measured at the wide sweep's largest
        cell (n=7,936,507 over m=6 srw scales): retention 381 MB -> 63 MB,
        which moves total peak 725 MB -> 416 MB. The rest of that peak is
        srw's own fixed working set (models/srw.py blocks its draw to a byte
        budget), which `reduce` does not touch. Only valid with out_dir=None:
        reducing and persisting at once would write summaries to a file named
        samples.
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
    dict
        Maps each requested scale to its array of `n` i.i.d. samples -- or,
        when `reduce` is given, to `reduce(samples)`. For a chunked run,
        values are disk-backed memmaps rather than plain in-RAM arrays (still
        valid np.ndarray for callers).
    """
    spec = get_model(model)
    scales_list, n_list = normalize_scales_n(scales, n)

    if reduce is not None and out_dir is not None:
        raise ValueError(
            "reduce= and out_dir= are mutually exclusive: reducing discards the "
            "samples, so persisting the result would write summaries to a file "
            "named samples.npz. Pass one or the other.")

    seed_seq = as_seed_sequence(seed)
    rng = np.random.default_rng(seed_seq)

    chunked = out_dir is not None and sum(n_list) * 8 > max_chunk_bytes
    rd = None
    if out_dir is not None:
        stem = tag or content_id(params, scales_list, n_list, seed_record(seed_seq))
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

    samples: dict = {}
    timings: dict[int, float] = {}
    for idx, (i, n_i) in enumerate(zip(scales_list, n_list), start=1):
        t0 = time.perf_counter()
        drawn = _generate_scale_chunked(i, n_i) if chunked else spec.simulate(i, n_i, params, rng)
        samples[i] = drawn if reduce is None else reduce(drawn)
        del drawn                      # the point of reduce=: drop it now, not at loop end
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
            seed=seed_record(seed_seq),
            timing_seconds=timings,
        )

    return samples


def reproduce(run_dir: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a run directory's metadata,
    from scratch (no I/O on samples.npz) -- an independent check that saved
    data matches what the recorded recipe actually produces."""
    meta = load_metadata(run_dir)
    if meta is None:
        raise FileNotFoundError(
            f"no {ARTIFACTS['samples_meta']} in {run_dir}")
    return generate(meta["model"], meta["scales"], meta["n"], meta["params"], seed=meta["seed"])


#: Why each allocation-rule input exists, printed when a recipe omits one.
#: These are DESIGN inputs: they decide how the budget is split across scales
#: and never enter any fit. The pilot measures omega1 and d from the samples;
#: what these choose is only where the pilot spends them.
DESIGN_INPUTS = {
    "d": ('"d": 1.0',
          "d is Assumption cost_is_power_law's exponent: how the cost of ONE sample\n"
          "  grows with the scale, cost(i) = i**d. It converts the budget into sample\n"
          "  counts and nothing else. A model that declares a `cost_hint` supplies it\n"
          "  automatically -- this model does not, so state it."),
    "omega1": ('"omega1": 1.0',
               "omega1 here is a DESIGN input, not an estimate. It sets the SHAPE of the\n"
               "  allocation, n_i ~ i**(2*omega1), and never reaches an estimator: the fit\n"
               "  sees only the drawn samples. Use 1.0 if you have no idea -- the rule needs\n"
               "  the right sign of the trend, not the right value (see snr_allocation)."),
}


def _design_input(n: dict, key: str, rule: str):
    """One allocation-rule input, or an error that says what it is and why.

    SystemExit, not a bare exception: a recipe missing a key is the author's
    mistake, not a crash, and a traceback buries the one line that would fix
    it. Same convention as constants.require (tools/constants.py).
    """
    if key in n:
        return float(n[key])
    example, why = DESIGN_INPUTS[key]
    raise SystemExit(
        f"the {rule!r} allocation rule needs {key!r} in the recipe's \"n\", and this "
        f"recipe has none.\n"
        f'  add it:  "n": {{"rule": "{rule}", "budget": ..., {example}}}\n'
        f"  {why}")


def resolve_d(cfg: dict, n: dict, rule: str) -> tuple[float, str]:
    """The cost exponent an allocation should spend against, and where it came from.

    Taken from the MODEL when it declares a `cost_hint`, which is exact and
    removes the input from the recipe entirely -- srw's cost_hint says
    Theta(k), so d = 1 by construction rather than by anyone's assertion. Only
    a model that declares nothing has to be told.

    An explicit `"d"` in the recipe still wins: a model can be right about its
    own arithmetic and still not be what the machine is bound by.
    """
    if "d" in cfg.get("n", {}):
        return float(n["d"]), "recipe"
    spec = get_model(cfg["model"])
    if spec.cost_hint is not None:
        return (declared_exponent(cfg["scales"], spec.cost_hint,
                                  cfg.get("params", {})),
                f"declared by {cfg['model']}'s cost_hint")
    return _design_input(n, "d", rule), "recipe"


def resolve_n(cfg: dict):
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

    A rule's inputs are DESIGN inputs and are treated as such: `d` comes from
    the model's own declared cost when it has one, `omega1` must be stated, and
    a missing one is an error naming what it is -- never a silent default. They
    decide how the budget is SPLIT and never reach an estimator.
    """
    n = cfg["n"]
    if not isinstance(n, dict):
        return n

    rule = n.get("rule")
    if rule not in ("neyman", "snr"):
        raise SystemExit(
            f"unknown allocation rule {rule!r} in the recipe's \"n\"; "
            f"known: 'neyman', 'snr'"
        )
    d, d_from = resolve_d(cfg, n, rule)
    common = dict(
        budget=float(n["budget"]),
        d=d,
        sigma=n.get("sigma"),
        min_n=int(n.get("min_n", 1)),
    )
    if rule == "neyman":
        alloc = neyman_allocation(cfg["scales"], **common)
    else:
        alloc = snr_allocation(cfg["scales"],
                               omega1=_design_input(n, "omega1", rule), **common)
    print(
        f"allocation rule={rule!r} budget={alloc['budget']:.4g} d={d:g} ({d_from}) -> "
        f"cost={alloc['cost']:.4g} ({alloc['exhausted']:.1%} of budget)"
    )
    if alloc["exhausted"] > 1.0:
        print(
            "  warning: min_n clamp binds, so this allocation OVERSPENDS the stated budget "
            "-- raise the budget or drop the smallest scales",
            file=sys.stderr,
        )
    print(f"  n per scale: {dict(zip(alloc['scales'], alloc['n']))}")
    starved = [int(i) for i, c in zip(alloc["scales"], alloc["n"]) if c < 2]
    if starved:
        sys.stdout.flush()      # keep the warning below the table it refers to
        # n_i = 1 has no spread, and every consumer of these samples needs one
        # (sigma_log weights the log-log fit). Better said here, where the
        # remedy is obvious, than as a NaN three steps downstream.
        print(
            f"  warning: {len(starved)} scale(s) got n < 2 -- {starved}.\n"
            f"  A single draw has no standard error, so these scales cannot be "
            f"summarized.\n"
            f"  The ladder is too wide for the budget: under {rule!r}, n spans "
            f"{max(alloc['n']) / max(min(alloc['n']), 1):.3g}x across it. "
            f"Raise the budget, drop\n  the smallest scales, or set "
            f'"min_n": 2 (which will overspend the stated budget).',
            file=sys.stderr,
        )
    return alloc["n"]


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw samples from a JSON recipe (any model registered in tools/models.py). "
            "The recipe is never modified; output (samples.npz + samples_meta.json) is written "
            "to --out-dir/<tag>/. See experiments/*/samples_example.json for recipe shapes."
        )
    )
    parser.add_argument(
        "-meta", "--meta", dest="meta", required=True, type=Path,
        help='Recipe JSON (read-only): {"model": "synthetic"|"srw", "params": {...}, '
        '"scales": [...], "n": ..., "seed": null}',
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", type=Path, default=None,
        help="Output directory for <tag>/. Defaults to the experiment's data/ directory (the recipe's grandparent when it sits in recipes/).",
    )
    parser.add_argument(
        "--tag", dest="tag", type=str, default=None,
        help="Run directory name. Defaults to a content hash, so an identical rerun overwrites "
        "rather than accumulating; pass a fixed --tag for a predictable path to chain into other "
        "tools. May contain '/' to nest, which is how replicate groups are laid out: "
        "--tag mygroup/rep0, mygroup/rep1, ... keeps same-config runs together under one folder.",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=None,
        help="Override the recipe's seed. The point of this flag is replicates: the same recipe "
        "run at several seeds is exactly what 'independent replicates of one configuration' means "
        "(ground rule 2), and overriding here avoids editing (or copying) the recipe to vary it. "
        "The resolved seed is always recorded in the run's samples_meta.json.",
    )
    args = parser.parse_args(argv)

    cfg = load_recipe(args.meta, "samples")
    model = cfg["model"]
    spec = get_model(model)
    params = cfg["params"]
    scales_list, n_list = normalize_scales_n(cfg["scales"], resolve_n(cfg))

    seed_seq = np.random.SeedSequence(args.seed if args.seed is not None else cfg.get("seed"))
    resolved_seed = seed_seq.entropy

    out_dir = args.out_dir or default_out_dir(args.meta)
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
    print(f"metadata = {artifact_path(rd, 'samples_meta')}")
    print(f"(recipe {args.meta} was not modified)")


if __name__ == "__main__":
    _main()
