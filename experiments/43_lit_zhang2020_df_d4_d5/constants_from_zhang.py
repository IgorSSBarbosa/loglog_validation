"""Experiment 43: seed a study with constants measured on the Zhang et al. replicate,
with omega1 fixed at the literature lower bound instead of fitted.

    .venv/bin/python experiments/43_lit_zhang2020_df_d4_d5/constants_from_zhang.py \
        experiments/43_lit_zhang2020_df_d4_d5/data/zhang2020_d5 --study zhang_w070_8h --omega1 0.70

Writes <data-root>/<study>/constants.json and pilot.json, so plan.py and run.py treat
the study like one pilot.py produced. Why not pilot.py: its pilot (2..32) fitted
omega1 = 1.43 from the fast early curvature, and timed d = 5.03 on rungs below the
~10^8-site slowdown (README, Assumption 7 row). The Zhang replicate reaches L = 60.

  - omega1: user override (user, 2026-10-30: literature lower bound 0.70).
  - a1: eq. (232), J = 1, omega1 held at the override, weighted LS on log Ybar at the
    first L_m whose chi2 passes tools/lmin_rule.acceptable.
  - d: OLS of log(seconds per sample) on log L over L >= 8, from the run log.
  - throughput: sites per second at the largest size, since the top rung dominates
    every ladder's cost and is the slowest per site.
  - cv: mean over the 13 sizes, the pilot's convention.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts import write_artifact  # noqa: E402
from tools.constants import Constant, format_table, measured, save  # noqa: E402
from tools.lmin_rule import acceptable  # noqa: E402

D_FROM = 8  # smallest L in the cost fit: below it the per-sample overhead dominates


def main(data: Path, study: str, omega1: float) -> None:
    arm = json.loads((data / "protocol_arm.json").read_text())
    r = arm["per_replicate"][0]
    L = np.array(arm["scales"], float)
    n = np.array(arm["n"], float)
    y = np.array(r["y_bar"])
    s = np.array(r["sigma_log"])
    src = f"{data.relative_to(ROOT)}/protocol_arm.json"

    f = lambda x, g, la0, a1: la0 + g * np.log(x) + a1 * x ** (-omega1)  # noqa: E731
    for lm in L[:-3]:
        k = L >= lm
        q, c = curve_fit(f, L[k], np.log(y[k]), p0=[3.5, 0.0, -1.0], sigma=s[k], absolute_sigma=True)
        chi2 = float(np.sum(((np.log(y[k]) - f(L[k], *q)) / s[k]) ** 2))
        dof = int(k.sum()) - 3
        if acceptable(chi2, dof):
            break
    else:
        raise SystemExit(f"no L_m passes the chi2 rule with omega1 = {omega1}")
    se = np.sqrt(np.diag(c))
    print(f"eq. (232) at omega1 = {omega1}: L_m = {lm:.0f}, gamma = {q[0]:.4f}({se[0]:.4f}), "
          f"a1 = {q[2]:.4f}({se[2]:.4f}), chi2/DF = {chi2:.1f}/{dof}")

    ms = np.array([float(m) for m in re.findall(r"\(([\d.]+) ms\)", data.with_suffix(".log").read_text())])
    per_sample = ms / 1e3 / n
    k = L >= D_FROM
    (d, _), cov = np.polyfit(np.log(L[k]), np.log(per_sample[k]), 1, cov=True)
    throughput = L[-1] ** 5 / per_sample[-1]
    cv = np.sqrt(r["cv2"])

    consts = {
        "omega1": Constant(omega1, None, "user override (--omega1): literature lower bound, user 2026-10-30"),
        "a1": measured(q[2], se[2], f"Zhang replicate, eq. (232) J=1 with omega1 fixed at {omega1}, "
                                    f"L_m={lm:.0f}, chi2/DF={chi2:.1f}/{dof}", src),
        "d": measured(d, np.sqrt(cov[0, 0]), f"Zhang replicate run log, OLS of log s/sample on log L, "
                                             f"L>={D_FROM}", src),
        "cv": measured(cv.mean(), None, f"Zhang replicate, mean over {L.size} scales, "
                                        f"spread {cv.min():.4f}-{cv.max():.4f}", src),
    }
    sd = data.parent / study
    save(sd, consts)
    write_artifact(sd, "pilot", {
        "recipe": {"kind": "samples", "model": arm["model"], "params": arm["params"]},
        "scales": L.tolist(), "n": n.tolist(), "y_bar": y.tolist(), "sigma_log": s.tolist(),
        "direct_fit": {"gamma": q[0], "a0": float(np.exp(q[1])), "a1": q[2], "omega1": omega1},
        "throughput": throughput,
        "throughput_source": f"the Zhang replicate's clock at L={L[-1]:.0f} ({src})",
        "replicates": 1,
        "seeded_from": src,
    }, produced_by="experiments/43_lit_zhang2020_df_d4_d5/constants_from_zhang.py")
    print(format_table(consts))
    print(f"  throughput {throughput:.4g} sites/s\nwrote {sd}/constants.json, pilot.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path, help="the protocol_arm data dir, e.g. data/zhang2020_d5")
    p.add_argument("--study", required=True)
    p.add_argument("--omega1", type=float, required=True)
    a = p.parse_args()
    main(a.data.resolve(), a.study, a.omega1)
