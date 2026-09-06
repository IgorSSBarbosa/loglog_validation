"""Is the p in a recipe actually the critical point? -- the diagnostic that
guards models/percolation_zd.py's p_c table.

Why this driver exists
----------------------
In two dimensions p_c = 0.59274605079210 is known to fourteen digits and
nothing a simulation here can build resolves the uncertainty, so
experiments/03_percolation_zd never had to check it. Above two dimensions the
threshold is a DIFFERENT NUMBER in every dimension, taken from the literature
(models/percolation_zd.py's P_C_SOURCE records which paper each one comes
from), and a wrong entry -- a transposed digit, a value read off the bond
rather than the site table, the wrong lattice -- simulates a systematically
off-critical system. That is a bias no estimator in this repo can see: a
slightly supercritical lattice still gives a clean power law over a short
ladder, with the wrong exponent.

So the table gets a numeric check before anything is measured on it
(PLAN.md ground rule 1). At p_c the probability that some open cluster spans
the box along x_0 is asymptotically INDEPENDENT of the box side; below p_c it
decays to 0 and above it rises to 1. This script measures that probability
over a ladder of box sides and reports the trend, plus the same ladder at
p_c(1 +/- delta) as controls -- because "flat" only means something next to a
"not flat".

What it does NOT do: measure p_c. Locating a threshold properly needs a
finite-size-scaling crossing analysis over several sizes and a fit; this is a
falsification test, and it is deliberately the cheap one. A PASS says the
tabulated value is consistent with criticality at the sizes affordable here; a
FAIL says stop and go back to the reference.

The recipe is an ordinary `samples` recipe -- the same one the run being
guarded uses -- so the check is run on exactly the model, params, dim,
geometry and ladder that will be measured, not on a paraphrase of them. `n`
is read from `--n` rather than from the recipe's allocation rule: this is a
probability, so the sample size wanted is flat across scales, not
budget-allocated.

CLI:
    python3 src/estimate/check_criticality.py \
        -meta experiments/05_percolation_highd/recipes/samples_df_d3_facefar.json \
        --n 400 --tag criticality_d3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                    # repo root; src/<layer>/ -> ../../
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import default_out_dir, load_recipe, write_artifact  # noqa: E402
from tools.rng import as_seed_sequence, seed_record, spawn  # noqa: E402

#: Models that can answer the question, and the function that answers it.
#: Keyed rather than dispatched on `MODELS` because spanning is a property of a
#: box-side ladder: models/percolation_tau_zd.py's scale is a CLUSTER SIZE and
#: has no box side of its own to span, so it is deliberately absent -- check
#: its dimension with a percolation_zd recipe at the same dim and p.
SPANNING = {"percolation_zd": "models.percolation_zd"}

#: How far off p_c the two control arms sit. Large enough that the drift is
#: unmistakable at the box sides this can afford, small enough to stay in the
#: critical window's neighbourhood rather than testing p = 0 against p = 1.
DEFAULT_DELTA = 0.08

#: A PASS wants the spanning probability at p_c to move by less than this
#: across the ladder, and each control arm to move by more, in the right
#: direction. Not a theory-derived number: it is the size of the finite-size
#: correction this repo can afford to leave unresolved, and the driver reports
#: the measured drifts so a reader can apply their own.
FLAT_TOL = 0.12


def _spanner(model: str):
    if model not in SPANNING:
        raise ValueError(
            f"model {model!r} has no spanning diagnostic; known: "
            f"{sorted(SPANNING)}. A cluster-size ladder (percolation_tau_zd) "
            f"has no box side to span -- check that dimension with a "
            f"percolation_zd recipe at the same dim, p and geometry.")
    import importlib
    return importlib.import_module(SPANNING[model]).crossing_fraction


def _binom_se(f: float, n: int) -> float:
    return float(np.sqrt(max(f * (1.0 - f), 1e-12) / n))


def measure(model: str, scales, n: int, params: dict, *, delta: float,
            seed=None) -> dict:
    """Spanning probability over `scales` at p_c and at p_c*(1 -/+ delta).

    Every (arm, scale) cell draws its own independent stream, spawned from one
    root seed -- PLAN.md ground rule 2, and the reason two arms at the same
    scale are not paired: a paired comparison would make the control look
    flatter than it is by sharing the lattices that decide it.
    """
    crossing = _spanner(model)
    dim = int(params.get("dim", 2))
    geometry = params.get("geometry", "box")
    p_c = params.get("p")
    if p_c is None:
        import importlib
        p_c = importlib.import_module(SPANNING[model]).critical_p(dim)
    p_c = float(p_c)

    arms = {"p_c": p_c, "below": p_c * (1.0 - delta), "above": p_c * (1.0 + delta)}
    root = as_seed_sequence(seed)
    streams = spawn(root, len(arms) * len(scales))

    spanning: dict[str, list[float]] = {}
    stderr: dict[str, list[float]] = {}
    started = time.perf_counter()
    for a, (name, p) in enumerate(arms.items()):
        row, err = [], []
        for k, i in enumerate(scales):
            rng = np.random.default_rng(streams[a * len(scales) + k])
            f = crossing(int(i), n=n, dim=dim, p=p, geometry=geometry, rng=rng)
            row.append(float(f))
            err.append(_binom_se(f, n))
        spanning[name] = row
        stderr[name] = err
    elapsed = time.perf_counter() - started

    drift = {name: row[-1] - row[0] for name, row in spanning.items()}
    verdict = (abs(drift["p_c"]) < FLAT_TOL
               and drift["below"] < -FLAT_TOL
               and drift["above"] > FLAT_TOL)
    return {
        "model": model, "params": params, "dim": dim, "geometry": geometry,
        "scales": [int(i) for i in scales], "n": int(n),
        "p": arms, "delta": delta,
        "spanning": spanning, "stderr": stderr, "drift": drift,
        "flat_tol": FLAT_TOL, "pass": bool(verdict),
        "seed": seed_record(root), "elapsed_s": elapsed,
    }


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-meta", "--meta", dest="meta", required=True, type=Path,
                        help="A `samples` recipe: the check runs on its model, "
                             "params, dim, geometry and ladder.")
    parser.add_argument("--n", type=int, default=400,
                        help="Samples per (arm, scale). Flat across scales: "
                             "this is a probability, not a budget-allocated mean.")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help="Relative offset of the two control arms from p_c.")
    parser.add_argument("--scales", type=int, nargs="+", default=None,
                        help="Override the recipe's ladder (spanning needs only "
                             "a few sizes, and the top of a sampling ladder is "
                             "usually too expensive to repeat here).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="criticality")
    args = parser.parse_args(argv)

    cfg = load_recipe(args.meta, "samples")
    scales = args.scales if args.scales is not None else cfg["scales"]
    seed = args.seed if args.seed is not None else cfg.get("seed")
    result = measure(cfg["model"], scales, args.n, cfg.get("params", {}),
                     delta=args.delta, seed=seed)

    out_dir = args.out_dir or default_out_dir(args.meta)
    run_dir = out_dir / args.tag
    write_artifact(run_dir, "criticality", result,
                   produced_by="src/estimate/check_criticality.py",
                   recipe=args.meta)

    print(f"model={result['model']!r}  dim={result['dim']}  "
          f"geometry={result['geometry']!r}  n={result['n']}  "
          f"seed={result['seed']}  {result['elapsed_s']:.1f}s")
    print(f"p_c = {result['p']['p_c']:.9g}  "
          f"controls at +/-{result['delta']:.0%}\n")
    head = "".join(f"{('i=' + str(i)):>15}" for i in result["scales"])
    print(f"{'arm':>8}{head}{'drift':>10}")
    for name in ("below", "p_c", "above"):
        row = "".join(f"{f:>9.3f}+-{e:.3f}"
                      for f, e in zip(result["spanning"][name],
                                      result["stderr"][name]))
        print(f"{name:>8}{row}{result['drift'][name]:>10.3f}")
    print(f"\n{'PASS' if result['pass'] else 'FAIL'}: at p_c the spanning "
          f"probability moves {result['drift']['p_c']:+.3f} across the ladder "
          f"(want |.| < {FLAT_TOL}), while the controls move "
          f"{result['drift']['below']:+.3f} and {result['drift']['above']:+.3f} "
          f"(want past -/+{FLAT_TOL}).")
    if not result["pass"]:
        print("A FAIL is not necessarily a wrong p_c: at the small box sides "
              "high dimensions can afford, the finite-size correction to the "
              "spanning probability is itself large. Read the numbers, and "
              "compare the drift AT p_c against the controls' before "
              "concluding anything about the table.")


if __name__ == "__main__":
    _main()
