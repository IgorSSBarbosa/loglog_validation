"""Test a long run's local slopes against the pre-registered predictions (predict.py).

The drawn slopes s_k = log2(Ybar_2k / Ybar_k), k = 2^18..2^22, come from the run's
final.json: Ybar pooled over replicates (equal n), the se of ln Ybar from each
replicate's sigma_log, sigma^2 / R pooled. Consecutive s_k share a Ybar, so their
covariance is the bidiagonal D diag(sigma^2) D^T / ln2^2, and the predictions share
one fit, so theirs is full (predict.predict). Each model is scored by

    chi2 = r^T (C_drawn + C_pred)^-1 r,   r = s_drawn - s_pred,   5 dof,

which is the pre-registered test: all of the window's slopes together.

    python3 experiments/50_rwre_lazy_long_omega/compare.py --p 0.04 --study long_p0.04_24h
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from predict import LN2, models, predict  # noqa: E402


def drawn_slopes(final: dict) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """(lower scales k, s_k, their covariance, per-replicate s_k)."""
    reps = final["per_replicate"]
    y = np.array([r["y_bar"] for r in reps])
    sig = np.array([r["sigma_log"] for r in reps])
    R = len(reps)
    ly = np.log(y.mean(axis=0))
    var = (sig ** 2).mean(axis=0) / R
    D = np.diff(np.eye(len(ly)), axis=0)
    per_rep = np.diff(np.log(y), axis=1) / LN2
    return final["scales"][:-1], D @ ly / LN2, D @ np.diag(var) @ D.T / LN2 ** 2, per_rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p", required=True, choices=["0.04", "0.3"])
    ap.add_argument("--study", required=True)
    ap.add_argument("--data-root", type=Path, default=EXP / "data")
    a = ap.parse_args()

    final = json.loads((a.data_root / a.study / "final.json").read_text())
    ks, s, cs, per_rep = drawn_slopes(final)
    print(f"{a.study}: {final['replicates']} replicate(s) x n = {final['n']:,}"
          + (f" (closed early, {final['replicates_planned']} planned)"
             if final.get("closed_partial") else ""))
    print("  k      " + "  ".join(f"2^{int(math.log2(k)):<9d}" for k in ks))
    print("  drawn  " + "  ".join(f"{v:.4f}±{e:.4f}" for v, e in zip(s, np.sqrt(np.diag(cs)))))
    if len(per_rep) > 1:
        print("  rep sd " + "  ".join(f"{e:.4f}     " for e in per_rep.std(axis=0, ddof=1)
                                     / math.sqrt(len(per_rep)))
              + "  (scatter of the replicates' s_k / sqrt R, the check on the se above)")
    for p, lab, (names, x, cov, _, _, full) in models():
        if p != a.p:
            continue
        v, cp = predict(names, x, cov, full, ks)
        r = s - v
        c2 = float(r @ np.linalg.solve(cs + cp, r))
        print(f"  {lab:24s} " + "  ".join(f"{d:+.4f}     " for d in r)
              + f"  chi2 = {c2:.1f}/{len(r)}, p = {chi2.sf(c2, len(r)):.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
