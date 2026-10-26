"""Replace a pilot's omega1, a1 and cv with values from 19's tail fit at p = 0. Keep its d.

Experiment 41 (user, 2026-10-15). 19's autopilot planned p = 0 with the pilot's omega1 =
0.450, fitted on 4..1024. 19's `n4e6` sweep (n = 800,000 x 5, 4..2^18) shows that this is a
short-ladder value. The (P) fit gamma + c k^-omega on the local slopes s_k with k >= 256 gives
omega = 0.153 +/- 0.024 (chi2 p = 0.54). The user chose that fit. They also chose to take a1 and cv
from the same data and to keep only d from the pilot.

a1 comes from c through eq. (232), ln Ybar_k = ln a0 + gamma ln k + a1 k^-omega1. The local
slope at the lower scale k is s_k = (ln Ybar_2k - ln Ybar_k) / ln 2, which gives
    s_k = gamma + a1 (2^-omega - 1) / ln 2 * k^-omega,  so  a1 = c ln 2 / (2^-omega - 1).
Its se is propagated from the (c, omega) covariance. `local_slope.fit_power` reports only
the diagonal of that covariance, so the fit is redone here from the stored s_k and se and
checked against the stored result.

cv is the mean over the same k >= 256 scales of the per-scale cv. Its se is that mean's
standard error across those scales.

    python3 experiments/41_rwre_p0_tail_omega/tail_constants.py --study pilot_p0 \\
        --data-root experiments/41_rwre_p0_tail_omega/data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from tools import constants as C  # noqa: E402

SOURCE = ROOT / "experiments/19_rwre_lazy_omega/data/omega_n4e6.json"
P_KEY = "0"
TAIL = 256


def tail_constants(source: Path) -> dict[str, C.Constant]:
    data = json.loads(source.read_text())
    if data["args"]["tail"] != TAIL:
        raise ValueError(f"{source} was fitted with tail {data['args']['tail']}, not {TAIL}")
    r = data["results"][P_KEY]
    stored = r["fits"]["P_tail"]
    lower = np.asarray(r["scales"][:-1], float)
    t = lower >= TAIL
    k, s, se = lower[t], np.asarray(r["s"])[t], np.asarray(r["se"])[t]

    popt, pcov = optimize.curve_fit(lambda x, g, c, o: g + c * x ** -o, k, s,
                                    p0=[stored["gamma"], stored["c"], stored["omega"]],
                                    sigma=se, absolute_sigma=True, maxfev=20000)
    for j, name in enumerate(("gamma", "c", "omega")):
        if not math.isclose(popt[j], stored[name], rel_tol=1e-6):
            raise ValueError(f"refit {name} = {popt[j]} does not reproduce the stored {stored[name]}")
    _, c, om = popt

    q = 2.0 ** -om - 1
    a1 = c * math.log(2) / q
    # d a1 / d c and d a1 / d omega, the latter through q' = -ln 2 * 2^-omega.
    grad = np.array([math.log(2) / q, c * math.log(2) ** 2 * 2.0 ** -om / q ** 2])
    a1_se = float(np.sqrt(grad @ pcov[1:, 1:] @ grad))

    cv = np.asarray(r["cv"])[np.asarray(r["scales"]) >= TAIL]
    rel = SOURCE.relative_to(ROOT)
    prov = f"19 n4e6, p=0, (P) fit on s_k, k >= {TAIL} ({t.sum()} slopes)"
    return {
        "omega1": C.measured(om, math.sqrt(pcov[2, 2]), prov, rel),
        "a1": C.measured(a1, a1_se, f"{prov}, a1 = c ln2 / (2^-omega - 1)", rel),
        "cv": C.measured(float(cv.mean()), float(cv.std(ddof=1) / math.sqrt(cv.size)),
                         f"19 n4e6, p=0, mean over {cv.size} scales {TAIL}..{int(r['scales'][-1])}, "
                         f"spread {cv.min():.4f}-{cv.max():.4f}", rel),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True, help="study whose constants.json is rewritten")
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args()

    sd = a.data_root / a.study
    consts = C.load(sd)
    C.require(consts, "d")
    before = C.format_table(consts)
    consts.update(tail_constants(SOURCE))
    C.save(sd, consts)
    print(f"before:\n{before}\n\nafter ({sd / C.CONSTANTS_FILE}):\n{C.format_table(consts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
