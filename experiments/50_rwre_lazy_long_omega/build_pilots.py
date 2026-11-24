"""Build the two reused pilots of experiment 50: an existing study's pilot, with tail constants.

Experiment 50 (user, 2026-11-02): no pilot is drawn. Each long run reuses data already
drawn (`autopilot.py --reuse-pilot`), and the pilot's omega1, a1 and cv are replaced by a
fit on the TAIL of a long ladder, as experiments/41's tail_constants.py did. The pilots'
own omega1 are short-ladder values (48's p = 0.04: 0.401 on 4..1024, against about 0.2
from its long run; 19's p = 0.3: 0.009, the hump).

  p = 0.04, source: 48's `sieve_p0.04_3h_cap131072` (pilot 4..1024, cap 131072).
    omega1, a1: eq. (pilot-fit), ln Ybar_i = ln a0 + gamma ln i + a1 i^-omega1, by
      weighted least squares with gamma, a0, a1, omega1 all free, on the ln Ybar of that
      study's local_slope.json (every scale 4..2^21 at the long run's n, 5 replicates),
      scales >= TAIL_004. gamma is free, so the omega1 borrowed below is not fitted
      under gamma = 1/2.
    cv: the mean per-scale cv of its long run (final.json, 2^16..2^21).
  p = 0.3, source: 19's `omega_p0.3_n4e6` (4..2^18, n = 800,000 x 5, cap 2^15).
    omega1: borrowed from p = 0.04 (the user's choice, 2026-11-02: past the hump the tail
      exponent is taken as p-independent, as 19's family fit has it).
    a1: the same eq. (pilot-fit) with omega1 held at the borrowed value, linear WLS on
      scales >= TAIL_03. Its se adds the conditional WLS se and the borrowed omega1's
      se carried through d a1 / d omega1, in quadrature.
    cv: the mean per-scale cv over the same scales.
  d, throughput (both p): time_window.py's timing of the run's own kernel on the run's
    window, data/cost_window_p<p>/cost_probe.json (user, 2026-11-02). The pilots' own
    were timed on 4..1024 with the warp kernel, d = 0.97 against 1.2-1.4 on the window,
    and every plan built on them under-ran by 2-5x. pilot.json's `cost` block stays the
    source pilot's probe, for the figures; the plan reads d and throughput only.

THE CAP AND KERNEL NOTE (user, 2026-11-02). The runs use the pooled kernel
(`kernel` "pool", models/rwre_gpu_lazy.py) with cap 393216, where cap costs only page-table
bytes. Reuse runs the reused pilot's own recipe, so each built pilot.json carries the run's
recipe -- kernel "pool", cap 393216 -- although its draws were made with the warp kernel
at another cap (48: 131072; 19: 2^15). Neither changes a draw: cap only moves the
overflow (experiments 41, 48), and the pooled kernel is bit-identical to the warp kernel
(checked 2026-11-02, every column, with and without evictions). Recorded in the built
pilot.json (`provenance_note`) and in every replaced constant's source.

Writes data/pilot_p<p>_pool/{pilot.json, constants.json}. Draws nothing.

    python3 experiments/50_rwre_lazy_long_omega/build_pilots.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.stats import chi2

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from tools import constants as C  # noqa: E402
from tools.artifacts import read_artifact, write_artifact  # noqa: E402

SRC_004 = ROOT / "experiments/48_rwre_lazy_psmall_sieve/data/sieve_p0.04_3h_cap131072"
SRC_03 = ROOT / "experiments/19_rwre_lazy_omega/data/omega_p0.3_n4e6"
TAIL_004 = 1024        # the first start from which the free fit is accepted (chi2 p = 0.18)
TAIL_03 = 16384        # past 19's hump (peak near 2^8-2^10), where a fixed-omega fit holds
CAP = 393216


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def fit_free(scales, log_y, se) -> dict:
    """eq. (pilot-fit) in ln Ybar, all four parameters free. Best of several omega starts."""
    j = np.log2(scales)

    def resid(th):
        A, g, a1, w = th
        return (A + g * j * math.log(2) + a1 * 2.0 ** (-w * j) - log_y) / se

    best = None
    for w0 in (0.05, 0.1, 0.2, 0.4, 0.8):
        r = optimize.least_squares(resid, [log_y[0] - 0.5 * math.log(scales[0]), 0.5, -0.5, w0],
                                   bounds=([-np.inf, -np.inf, -np.inf, 1e-3], np.inf))
        if best is None or r.cost < best.cost:
            best = r
    cov = np.linalg.inv(best.jac.T @ best.jac)
    chi = 2 * best.cost
    dof = len(j) - 4
    return {"gamma": best.x[1], "a1": best.x[2], "omega1": best.x[3], "cov": cov,
            "chi2": chi, "dof": dof, "p": float(chi2.sf(chi, dof))}


def fit_fixed_omega(scales, log_y, se, w: float) -> dict:
    """eq. (pilot-fit) with omega1 = w: linear in (ln a0, gamma, a1)."""
    i = np.asarray(scales, float)
    X = np.column_stack([np.ones_like(i), np.log(i), i ** -w]) / se[:, None]
    beta, *_ = np.linalg.lstsq(X, log_y / se, rcond=None)
    r = log_y / se - X @ beta
    cov = np.linalg.inv(X.T @ X)
    dof = len(i) - 3
    return {"gamma": beta[1], "a1": beta[2], "a1_se": math.sqrt(cov[2, 2]),
            "chi2": float(r @ r), "dof": dof, "p": float(chi2.sf(r @ r, dof))}


def constants_004() -> tuple[dict, str]:
    ls = json.loads((SRC_004 / "local_slope.json").read_text())["analysis"]
    scales = np.asarray(ls["scales"], float)
    t = scales >= TAIL_004
    f = fit_free(scales[t], np.asarray(ls["log_y"])[t], np.asarray(ls["se_log_y"])[t])
    final = read_artifact(SRC_004, "final")
    cv = np.mean([r["cv"] for r in final["per_replicate"]], axis=0)
    prov = (f"48 sieve_p0.04_3h_cap131072 local slopes, eq. (pilot-fit) on ln Ybar, "
            f"gamma free, scales {TAIL_004}..{int(scales[-1])} ({t.sum()} points), "
            f"chi2 {f['chi2']:.1f}/{f['dof']}, p = {f['p']:.2g}")
    out = {
        "omega1": C.measured(f["omega1"], math.sqrt(f["cov"][3, 3]), prov, _rel(SRC_004)),
        "a1": C.measured(f["a1"], math.sqrt(f["cov"][2, 2]), prov, _rel(SRC_004)),
        "cv": C.measured(float(cv.mean()), float(cv.std(ddof=1) / math.sqrt(cv.size)),
                         f"48 sieve_p0.04_3h_cap131072 long run, mean over {cv.size} scales "
                         f"{final['scales'][0]}..{final['scales'][-1]}", _rel(SRC_004)),
    }
    summary = (f"p = 0.04: gamma = {f['gamma']:.4f} (free), omega1 = {f['omega1']:.4f} "
               f"+/- {math.sqrt(f['cov'][3, 3]):.4f}, a1 = {f['a1']:+.4f} +/- "
               f"{math.sqrt(f['cov'][2, 2]):.4f}, chi2 {f['chi2']:.1f}/{f['dof']} (p = {f['p']:.2g})")
    return out, summary


def constants_03(omega1: C.Constant) -> tuple[dict, str]:
    pilot = read_artifact(SRC_03, "pilot")
    scales = np.asarray(pilot["scales"], float)
    t = scales >= TAIL_03
    ly = np.log(np.asarray(pilot["y_bar"], float))[t]
    se = np.asarray(pilot["sigma_log"], float)[t]
    w, w_se = omega1.value, omega1.se
    f = fit_fixed_omega(scales[t], ly, se, w)
    h = 1e-4
    da1 = (fit_fixed_omega(scales[t], ly, se, w + h)["a1"]
           - fit_fixed_omega(scales[t], ly, se, w - h)["a1"]) / (2 * h)
    a1_se = math.hypot(f["a1_se"], da1 * w_se)
    cv = np.asarray(pilot["cv_per_scale"], float)[t]
    base = f"19 omega_p0.3_n4e6 (draws at cap 2^15; cap changes no draw)"
    out = {
        "omega1": C.measured(w, w_se, f"borrowed from p = 0.04: {omega1.source}", omega1.origin),
        "a1": C.measured(f["a1"], a1_se,
                         f"{base}, eq. (pilot-fit) on ln Ybar with omega1 fixed at the "
                         f"borrowed {w:.4f}, scales {TAIL_03}..{int(scales[-1])} "
                         f"({t.sum()} points), chi2 {f['chi2']:.1f}/{f['dof']}; se adds "
                         f"omega1's through da1/domega1 = {da1:.3g}", _rel(SRC_03)),
        "cv": C.measured(float(cv.mean()), float(cv.std(ddof=1) / math.sqrt(cv.size)),
                         f"{base}, mean over {cv.size} scales {TAIL_03}..{int(scales[-1])}",
                         _rel(SRC_03)),
    }
    summary = (f"p = 0.3: omega1 = {w:.4f} (borrowed), gamma = {f['gamma']:.4f}, "
               f"a1 = {f['a1']:+.4f} +/- {a1_se:.4f}, chi2 {f['chi2']:.1f}/{f['dof']} "
               f"(p = {f['p']:.2g})")
    return out, summary


def build(source: Path, p: float, replace: dict, machine: Path, out_root: Path) -> Path:
    """data/pilot_p<p>_pool: the source's pilot.json under the run's recipe, new constants."""
    pilot = read_artifact(source, "pilot")
    mach = read_artifact(machine, "pilot")
    consts = C.load(source)
    timing_dir = out_root / f"cost_window_p{p:g}"
    timing = read_artifact(timing_dir, "cost_probe")
    fit = timing["fit"]
    recipe = dict(pilot["recipe"])
    recipe["params"] = {**recipe.get("params", {}), "p": p, "cap": CAP, "kernel": "pool"}
    if timing["params"] != recipe["params"]:
        raise SystemExit(f"{timing_dir} timed {timing['params']}, the run is {recipe['params']}")
    recipe.pop("seed", None)              # the run's seed is passed on the command line
    drawn_cap = pilot["recipe"].get("params", {}).get("cap", 32768)
    note = [f"Built by experiments/50_rwre_lazy_long_omega/build_pilots.py (user, 2026-11-02).",
            f"Draws: {_rel(source)}, made with the warp kernel at cap {drawn_cap}. This "
            f"recipe says kernel 'pool', cap {CAP}: neither changes a draw (cap only moves "
            f"the overflow, experiments 41, 48; the pooled kernel is bit-identical).",
            f"omega1, a1, cv: tail constants, see constants.json. d, throughput: "
            f"{_rel(timing_dir)} (the run's kernel on the run's window). `cost` is "
            f"{_rel(machine)}'s pilot probe on 4..1024, kept for the figures only."]
    body = {k: v for k, v in pilot.items() if k not in ("artifact", "produced_by")}
    # The cost probe travels with d and the throughput (19's pilot.json predates `probes`).
    body.update(recipe=recipe, cost=mach["cost"], throughput=fit["throughput"],
                throughput_source=f"{_rel(timing_dir)}: {timing['method']}, n = {timing['n']}, "
                                  f"host {timing['host']}",
                provenance_note=note, built_from=_rel(source))
    consts["d"] = C.measured(fit["d"], fit["d_se"],
                             f"time_window.py, the pooled kernel at {timing['scales'][0]}.."
                             f"{timing['scales'][-1]}, n = {timing['n']} per scale, host "
                             f"{timing['host']}, seed {timing['seed']}; se from the OLS residuals",
                             _rel(timing_dir))
    consts.update(replace)
    sd = out_root / f"pilot_p{p:g}_pool"
    sd.mkdir(parents=True, exist_ok=True)
    write_artifact(sd, "pilot", body,
                   produced_by="experiments/50_rwre_lazy_long_omega/build_pilots.py")
    C.save(sd, consts)
    return sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()
    c004, s004 = constants_004()
    c03, s03 = constants_03(c004["omega1"])
    print(s004)
    print(s03)
    for p, src, rep in ((0.04, SRC_004, c004), (0.3, SRC_03, c03)):
        sd = build(src, p, rep, SRC_004, a.data_root)
        print(f"\n{sd}:\n{C.format_table(C.load(sd))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
