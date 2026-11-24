"""Build experiment 52's reused pilot at one p: 19's n4e6 ladder, with the constants the user chose.

Experiment 52 (user, 2026-11-04): no pilot is drawn and omega1, a1 are NOT refitted per p.
The user's reason: with omega1 fixed, a per-p refit of a1 does not work, because the per-p
pilots do not have the single-correction shape (19's hump). So at every p:

  omega1, a1: BORROWED from experiment 50's p = 0.04 tail fit,
    experiments/50_rwre_lazy_long_omega/data/pilot_p0.04_pool/constants.json
    (eq. (pilot-fit) on 48's local-slope ln Ybar, gamma free, 1024..2^21:
    omega1 = 0.2552 +/- 0.0232, a1 = -1.0378 +/- 0.0497). Carried with their se and
    source, stamped "borrowed from 50 p = 0.04".
  cv: measured at this p from the reused ladder, 19's omega_p<p>_n4e6 (n = 800,000 x 5):
    the mean per-scale cv over its TOP_CV top scales (2^14..2^18 on the 4..2^18 ladders,
    2^16..2^20 on the 4..2^20 ones), as 50's build_pilots.py did at p = 0.3. A moment,
    not a fit.
  d, throughput: time_window.py at this p, on the host that draws the run
    (data/cost_window_p<p>_<tag>/cost_probe.json). The only per-p refit.

THE CAP AND KERNEL NOTE (as 50). Reuse runs the reused pilot's own recipe, so the built
pilot.json carries the run's recipe -- kernel "pool", cap 393216 -- although 19's draws were
made with the warp kernel at the default cap 2^15. Neither changes a draw: cap only moves
the overflow (experiments 41, 48), and the pooled kernel is bit-identical to the warp
kernel (50, 2026-11-02). Recorded in pilot.json (`provenance_note`) and in the sources.

THE COST BLOCK. pilot.json's `cost` is 19's own batched probe at this p (4..2^16, warp
kernel), kept for plot_pipeline.py's panels only; the plan reads d and the throughput.
19's pilots predate the per-probe list `probes` that plot_pipeline reads, so it is given
one entry: 19's aggregated (median) per-scale timing and its affine fit.

Writes data/pilot_p<p>_<tag>/{pilot.json, constants.json}. Draws nothing.

    python3 experiments/52_rwre_lazy_fixed_omega_sweep/build_constants.py --p 0.3 --tag 3h
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from tools import constants as C  # noqa: E402
from tools.artifacts import read_artifact, write_artifact  # noqa: E402

SRC_19 = ROOT / "experiments/19_rwre_lazy_omega/data"
SRC_OMEGA = ROOT / "experiments/50_rwre_lazy_long_omega/data/pilot_p0.04_pool"
TOP_CV = 5
CAP = 393216


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def build(p: float, tag: str, out_root: Path) -> Path:
    source = SRC_19 / f"omega_p{p:g}_n4e6"
    pilot = read_artifact(source, "pilot")
    timing_dir = out_root / f"cost_window_p{p:g}_{tag}"
    timing = read_artifact(timing_dir, "cost_probe")
    fit = timing["fit"]

    recipe = dict(pilot["recipe"])
    recipe["params"] = {**recipe.get("params", {}), "p": p, "cap": CAP, "kernel": "pool"}
    if timing["params"] != recipe["params"]:
        raise SystemExit(f"{timing_dir} timed {timing['params']}, the run is {recipe['params']}")
    if not math.isclose(pilot["recipe"]["params"]["p"], p):
        raise SystemExit(f"{source} was drawn at p = {pilot['recipe']['params']['p']}, not {p}")
    recipe.pop("seed", None)              # the run's seed is passed on the command line
    drawn_cap = pilot["recipe"].get("params", {}).get("cap", 32768)

    borrowed = C.load(SRC_OMEGA)
    consts = {}
    for k in ("omega1", "a1"):
        b = C.require(borrowed, k)
        consts[k] = C.measured(b.value, b.se, f"borrowed from 50 p = 0.04 (user, 2026-11-04: "
                               f"not refitted per p): {b.source}", b.origin)

    scales = np.asarray(pilot["scales"], float)
    cv = np.asarray(pilot["cv_per_scale"], float)[-TOP_CV:]
    top = scales[-TOP_CV:]
    base = f"19 omega_p{p:g}_n4e6 (draws at cap {drawn_cap}; cap changes no draw)"
    consts["cv"] = C.measured(float(cv.mean()), float(cv.std(ddof=1) / math.sqrt(cv.size)),
                              f"{base}, mean over its top {cv.size} scales "
                              f"{int(top[0])}..{int(top[-1])}", _rel(source))
    consts["d"] = C.measured(fit["d"], fit["d_se"],
                             f"time_window.py, the pooled kernel at {timing['scales'][0]}.."
                             f"{timing['scales'][-1]}, n = {timing['n']} per scale, host "
                             f"{timing['host']}, seed {timing['seed']}; se from the OLS residuals",
                             _rel(timing_dir))

    cost = dict(pilot["cost"])
    cost.setdefault("probes", [{"scales": cost["scales"], "elapsed": cost["elapsed"],
                                "affine": cost["affine"]}])
    note = [f"Built by experiments/52_rwre_lazy_fixed_omega_sweep/build_constants.py "
            f"(user, 2026-11-04).",
            f"Draws: {_rel(source)}, made with the warp kernel at cap {drawn_cap}. This "
            f"recipe says kernel 'pool', cap {CAP}: neither changes a draw (cap only moves "
            f"the overflow, experiments 41, 48; the pooled kernel is bit-identical, 50).",
            f"omega1, a1: borrowed from {_rel(SRC_OMEGA)} (not refitted per p). cv: this "
            f"ladder's top {TOP_CV} scales. d, throughput: {_rel(timing_dir)} (the run's "
            f"kernel on the run's window, on the run's host).",
            "`cost` is 19's own pilot probe at this p (4..2^16, warp kernel), kept for the "
            "figures only; `probes` holds one entry, 19's aggregated per-scale timing."]
    body = {k: v for k, v in pilot.items() if k not in ("artifact", "produced_by")}
    body.update(recipe=recipe, cost=cost, throughput=fit["throughput"],
                throughput_source=f"{_rel(timing_dir)}: {timing['method']}, n = {timing['n']}, "
                                  f"host {timing['host']}",
                provenance_note=note, built_from=_rel(source))
    sd = out_root / f"pilot_p{p:g}_{tag}"
    sd.mkdir(parents=True, exist_ok=True)
    write_artifact(sd, "pilot", body,
                   produced_by="experiments/52_rwre_lazy_fixed_omega_sweep/build_constants.py")
    C.save(sd, consts)
    return sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--tag", required=True, help="the run's tag, as given to time_window.py")
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()
    sd = build(a.p, a.tag, a.data_root)
    print(f"{sd}:\n{C.format_table(C.load(sd))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
