"""SRW sample generator -- the second consumer of tools/persistence.py's
save/load/metadata pattern (the first is experiments/00_synthetic/generator.py).
Mirrors that module's shape deliberately: same recipe/output-metadata
distinction, same CLI flags, same .npz+.json output pair -- only the
sampling step (srw() instead of a closed-form formula + noise family) and
the recipe's "params" (just {"q": ...} instead of SyntheticParams) differ.

NOT Phase 1's gamma-estimation ladder (see experiments/01_srw/README.md):
this produces and persists real SRW sample arrays for their own sake (and
for tools/loglog_plot.py to plot), but there is still no article-sanctioned
closed-form E[Y_i]/gamma/omega1 to validate any gamma_hat computed from them
against -- so this module deliberately does not attempt gamma-estimation.

CLI:
    python3 generate.py -meta example_config.json
    python3 generate.py -meta example_config.json --tag my_run -o data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for tools/
sys.path.insert(0, str(HERE))  # this dir, for srw

from tools.persistence import (  # noqa: E402
    content_id,
    load_metadata,
    load_samples,
    normalize_scales_n,
    save_samples,
    write_metadata,
)

from srw import srw  # noqa: E402


def generate(
    scales,
    n,
    q: float = 0.5,
    *,
    seed: int | None = None,
    out_dir: str | Path | None = None,
    tag: str | None = None,
    progress: bool = False,
) -> dict[int, np.ndarray]:
    """Draw n i.i.d. |S_k| samples at each requested scale k (see srw.py).

    Same parameter/return shape as experiments/00_synthetic/generator.py's
    `generate`, with `q` (the up-step probability) standing in for `params`.
    """
    scales_list, n_list = normalize_scales_n(scales, n)

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    samples: dict[int, np.ndarray] = {}
    timings: dict[int, float] = {}
    for idx, (k, n_k) in enumerate(zip(scales_list, n_list), start=1):
        t0 = time.perf_counter()
        samples[k] = srw(k, n=n_k, q=q, rng=rng)
        timings[k] = time.perf_counter() - t0
        if progress:
            print(
                f"\r[{idx}/{len(scales_list)}] scale k={k} n={n_k} ({timings[k] * 1e3:.1f} ms)"
                + " " * 10,
                end="",
                file=sys.stderr,
                flush=True,
            )
    if progress:
        print(file=sys.stderr)

    if out_dir is not None:
        stem = tag or content_id({"q": q}, scales_list, n_list, seed_seq.entropy)
        base = Path(out_dir) / stem
        save_samples(base.with_suffix(".npz"), samples)
        write_metadata(
            path=base.with_suffix(".json"),
            params={"q": q},
            scales=scales_list,
            n=n_list,
            seed=seed_seq.entropy,
            timing_seconds=timings,
        )

    return samples


def reproduce(metadata_path: str | Path) -> dict[int, np.ndarray]:
    """Regenerate the exact samples described by a metadata JSON, from scratch --
    an independent check that saved data matches what the recorded recipe produces."""
    meta = json.loads(Path(metadata_path).read_text())
    return generate(meta["scales"], meta["n"], meta["params"]["q"], seed=meta["seed"])


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw SRW |S_k| samples from a JSON recipe. The recipe is never modified; "
            "output (samples as .npz + reproducibility metadata as .json, same stem) is "
            "written to --out-dir. See example_config.json for the recipe shape."
        )
    )
    parser.add_argument(
        "-meta", "--meta", dest="meta", required=True, type=Path,
        help='Recipe JSON (read-only): {"params": {"q": 0.5}, "scales": [...], "n": ..., "seed": null}',
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", type=Path, default=None,
        help="Output directory for <tag>.npz + <tag>.json. Defaults to ./data next to this script.",
    )
    parser.add_argument(
        "--tag", dest="tag", type=str, default=None,
        help="Output filename stem. Defaults to a content hash, so an identical rerun overwrites "
        "rather than accumulating; pass a fixed --tag for a predictable path to chain into other tools.",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.meta.read_text())
    q = cfg["params"]["q"]
    scales_list, n_list = normalize_scales_n(cfg["scales"], cfg["n"])

    seed_seq = np.random.SeedSequence(cfg.get("seed"))
    resolved_seed = seed_seq.entropy

    out_dir = args.out_dir or (HERE / "data")
    samples = generate(
        scales_list, n_list, q, seed=resolved_seed, out_dir=out_dir, tag=args.tag, progress=True
    )

    stem = args.tag or content_id({"q": q}, scales_list, n_list, resolved_seed)
    timing = load_metadata(out_dir / stem)["timing_seconds"]

    print(f"q={q}")
    print(f"{'k':>12} {'n':>8} {'sample_mean':>14} {'elapsed_ms':>12}")
    for k in scales_list:
        y = samples[k]
        elapsed_ms = timing[str(k)] * 1e3
        print(f"{k:>12} {len(y):>8} {y.mean():>14.4f} {elapsed_ms:>12.2f}")
    print(f"\nseed     = {resolved_seed}")
    print(f"data     = {out_dir / (stem + '.npz')}")
    print(f"metadata = {out_dir / (stem + '.json')}")
    print(f"(recipe {args.meta} was not modified)")


if __name__ == "__main__":
    _main()
