"""Time rwre_gpu_lazy's pooled kernel on experiment 52's window at one p, for d and throughput.

Experiment 50's time_window.py, unchanged in method (user, 2026-11-04: d and the
throughput are the only per-p refit of 52). Why not the reused pilot's cost probe: 19's
probe was timed on 4..2^16 with the warp kernel, where a sample's work is mostly the
per-step cost; on the window the deque width W(k) enters (work ~ k W(k)) and the local
exponent is 1.25-1.8 (50). So this times one call simulate(i, n) per window scale, at an
n of several device waves, as the run will call it, and fits eq. (cost), seconds per
sample = i^d / throughput, by least squares in logs.

run_p.sh calls it on the host that then draws the run, so the throughput is that host's.
Draws are discarded; the seed is recorded. Writes data/cost_window_p<p>_<tag>/cost_probe.json.

    python3 experiments/52_rwre_lazy_fixed_omega_sweep/time_window.py --p 0.3 --tag 3h --seed 202611045226
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from tools.artifacts import write_artifact  # noqa: E402
from tools.models import get_model  # noqa: E402

#: The run's recipe params, but p (build_constants.py writes the same into the pilots).
PARAMS = {"alpha": 0.5, "cap": 393216, "kernel": "pool"}
SCALES = [2 ** j for j in range(18, 24)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--tag", required=True, help="the run's tag (run_p.sh: the budget, or 'smoke')")
    ap.add_argument("--n", type=int, default=16384, help="samples per call (default 16384, ~5 waves)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()

    spec = get_model("rwre_gpu_lazy")
    params = {**PARAMS, "p": a.p}
    rng = np.random.default_rng(np.random.SeedSequence(a.seed))
    t_start = time.perf_counter()
    spec.simulate(1024, 64, params, rng)                  # compile, warm up
    secs = []
    for i in SCALES:
        t0 = time.perf_counter()
        spec.simulate(i, a.n, params, rng)
        secs.append((time.perf_counter() - t0) / a.n)
        print(f"  i = 2^{int(math.log2(i))}: {secs[-1] * 1e3:.3f} ms/sample", flush=True)

    x, y = np.log(SCALES), np.log(secs)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    cov = (r @ r) / (len(x) - 2) * np.linalg.inv(X.T @ X)
    d, d_se, tp = float(beta[1]), float(math.sqrt(cov[1, 1])), float(math.exp(-beta[0]))
    local = np.diff(y) / np.diff(x)
    print(f"d = {d:.4f} +/- {d_se:.4f}, throughput {tp:.4g} /s; local exponents "
          + " ".join(f"{v:.3f}" for v in local))
    sd = a.data_root / f"cost_window_p{a.p:g}_{a.tag}"
    sd.mkdir(parents=True, exist_ok=True)
    write_artifact(sd, "cost_probe", {
        "model": "rwre_gpu_lazy", "params": params, "scales": SCALES, "n": a.n,
        "seconds_per_sample": secs, "seed": a.seed, "host": socket.gethostname(),
        "elapsed": time.perf_counter() - t_start,
        "fit": {"d": d, "d_se": d_se, "throughput": tp, "rel_rmse": float(np.sqrt(np.mean(r ** 2))),
                "local_exponents": local.tolist()},
        "method": "one call simulate(i, n) per scale; ln(s/sample) = d ln i - ln throughput, OLS",
    }, produced_by="experiments/52_rwre_lazy_fixed_omega_sweep/time_window.py")
    print(f"wrote {sd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
