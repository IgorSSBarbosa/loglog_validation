"""Shared-lattice sample generator: one set of realizations, every rung.

The cheap counterpart of src/generate/generate.py, for models that declare a
`shared_sampler` in tools/models.py (today: percolation_tau only).

WHY IT EXISTS. `generate.py` draws each rung from its own independent stream,
which is PLAN.md ground rule 2 and is what makes the article's CLT (eq. 583)
apply to the ladder verbatim. For a cluster-size ladder that is also
wasteful: one critical lattice already contains clusters of EVERY size below
its own finite-size cutoff, so a box sized for the top rung serves the whole
ladder. Sizing it that way costs n*L**2 for the ladder instead of
sum_k n_k L(s_k)**2 -- about 3x less at equal top-rung precision, with the
lower rungs getting far MORE data than the independent version gave them.

WHAT IS PAID. The rungs stop being independent. Cov(Ybar_s, Ybar_s') != 0, so
the CLT's variance is no longer the sum of per-rung variances, and any error
bar computed as if they were independent is a claim to be checked rather than
a fact. This driver therefore:

  - refuses any model that does not declare a `shared_sampler`, with a message
    saying to use generate.py instead;
  - stamps `shared_lattice: true` (plus L, the two design constants, the site
    count and each rung's cut ratio) into params in samples_meta.json, so a
    reader of the artifact cannot mistake it for an independent run;
  - prints the same, loudly, at the end of the run.

It writes the SAME artifacts as generate.py (<out_dir>/<tag>/samples.npz +
samples_meta.json), so every downstream tool -- estimate_omega1.py,
plot_loglog.py, report.py -- reads it unchanged. That is deliberate: the point
of the experiment is to compare the two samplers through the SAME estimators.

The recipe (kind "samples_shared") differs from a "samples" one in exactly one
way: `n_lattices` (a single integer) replaces `n`, because there is nothing to
allocate across scales -- every rung is read off the same lattices, so a
per-scale n has no meaning here. There is no allocation rule for the same
reason.

    {
      "kind": "samples_shared",
      "model": "percolation_tau",
      "params": {"p": 0.5927460507921, "bin_ratio": 2.0, "geometry": "torus",
                 "df_lower": 1.85, "cut_fraction": 0.01},
      "scales": [8, 16, 32, 64, 128, 256, 512, 1024],
      "n_lattices": 2400,
      "seed": 20260905
    }

CLI:

    python3 src/generate/generate_shared.py -meta <recipe> --tag <run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import default_out_dir, load_recipe  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools.persistence import (  # noqa: E402
    content_id,
    run_dir as _run_dir,
    save_samples,
    write_metadata,
)
from tools.rng import as_seed_sequence, seed_record  # noqa: E402


def generate_shared(
    model: str,
    scales,
    n_lattices: int,
    params: dict | None = None,
    seed=None,
    out_dir=None,
    tag: str | None = None,
) -> dict:
    """Draw the whole ladder from one set of `n_lattices` realizations.

    Returns {"samples", "info", "seed", "run_dir", "elapsed_seconds"}; writes
    the run directory only when `out_dir` is given.
    """
    spec = get_model(model)
    if spec.shared_sampler is None:
        raise ValueError(
            f"model {model!r} declares no shared_sampler, so its rungs cannot "
            f"share realizations -- use src/generate/generate.py, which draws "
            f"each scale from its own independent stream")
    params = dict(params or {})
    scales = [int(s) for s in scales]
    n_lattices = int(n_lattices)
    if n_lattices < 1:
        raise ValueError(f"n_lattices must be >= 1; got {n_lattices}")

    ss = as_seed_sequence(seed)
    rng = np.random.default_rng(ss)
    t0 = time.perf_counter()
    samples, info = spec.shared_sampler(scales, n_lattices, params, rng)
    elapsed = time.perf_counter() - t0

    result = {"samples": samples, "info": info, "seed": seed_record(ss),
              "elapsed_seconds": elapsed, "run_dir": None}
    if out_dir is None:
        return result

    # Provenance goes INSIDE the artifact (tools/artifacts.py's rule): the
    # shared-lattice facts ride in params, which is the field that says how
    # these numbers were produced. percolation_tau.simulate ignores keys it
    # does not know, so a re-simulation from this metadata still runs.
    stamped = dict(params)
    stamped.update(info)
    n_list = [n_lattices] * len(scales)
    tag = tag or content_id(stamped, scales, n_list, result["seed"])
    rd = _run_dir(out_dir, tag)
    save_samples(rd, samples)
    write_metadata(run_dir=rd, model=model, params=stamped, scales=scales,
                   n=n_list, seed=result["seed"],
                   timing_seconds={s: elapsed / len(scales) for s in scales})
    result["run_dir"] = rd
    return result


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-meta", required=True, type=Path,
                    help="a 'samples_shared' recipe")
    ap.add_argument("--tag", default=None, help="run directory name")
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="default: the experiment's data/")
    ap.add_argument("--seed", default=None, type=int,
                    help="override the recipe's seed (how replicates are drawn)")
    a = ap.parse_args(argv)

    cfg = load_recipe(a.meta, "samples_shared")
    seed = a.seed if a.seed is not None else cfg.get("seed")
    out_dir = a.out_dir or default_out_dir(a.meta)

    res = generate_shared(cfg["model"], cfg["scales"], cfg["n_lattices"],
                          params=cfg.get("params"), seed=seed,
                          out_dir=out_dir, tag=a.tag)
    info = res["info"]

    print(f"model={cfg['model']!r}  params={cfg.get('params', {})}")
    print(f"SHARED LATTICE: all {len(cfg['scales'])} rungs read off the SAME "
          f"{info['n_lattices']} realizations of one {info['L']}x{info['L']} box")
    print(f"  box rule: L = ceil((s_top_edge/{info['cut_fraction']})"
          f"**(1/{info['df_lower']})) = {info['L']}   "
          f"[{info['sites']:.3g} sites total, {res['elapsed_seconds']:.1f} s]")
    print(f"{'i':>12} {'n':>8} {'sample_mean':>14} {'cut_ratio':>11}")
    for s, z in zip(cfg["scales"], info["cut_ratio"]):
        print(f"{s:12d} {info['n_lattices']:8d} "
              f"{res['samples'][s].mean():14.6e} {z:11.5f}")
    print(f"\nseed     = {res['seed']}")
    print(f"run_dir  = {res['run_dir']}")
    print("\nNOTE: the rungs of this run are CORRELATED -- they come from the "
          "same lattices.\n      Error bars computed as if they were "
          "independent (estimate_omega1.py, report.py)\n      are claims to be "
          "checked, not facts. The metadata records shared_lattice=true.")


if __name__ == "__main__":
    _main()
