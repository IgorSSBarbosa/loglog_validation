"""Experiment 43: fit cost(L) = k0 + k1 * L**d, L = 2**k, to the per-sample draw time
of the zhang_w070_8h ladder, and plot it.

    .venv/bin/python experiments/43_lit_zhang2020_df_d4_d5/plot_cost_fit.py \
        experiments/43_lit_zhang2020_df_d4_d5/data/cost_check_2_64 \
        --plot experiments/43_lit_zhang2020_df_d4_d5/images/cost_fit_2_64.png

Reads generate.py's timing_seconds. L = 2 and 4 are left out of the fit: L = 2 is the
first call and carries the CUDA start-up, and L = 4 is slower per sample than L = 8, which
no k0 + k1 L^d can follow. The fit is least squares on log t, so each size counts by its
relative error, since the times span four decades. The Zhang replicate (purus, same GPU
model) is drawn for reference, not fitted.
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

from tools.loglog_plot import _BLUE, _GRIDLINE, _INK, _MUTED  # noqa: E402

FIT_FROM = 8
ZHANG = ROOT / "experiments/43_lit_zhang2020_df_d4_d5/data/zhang2020_d5"


def log_affine(L: np.ndarray, lk0: float, lk1: float, d: float) -> np.ndarray:
    """log(k0 + k1 L^d), with k0 = e^lk0 and k1 = e^lk1 so both stay positive."""
    return np.logaddexp(lk0, lk1 + d * np.log(L))


def main(data: Path, out: Path) -> None:
    meta = json.loads((data / "samples_meta.json").read_text())
    L = np.array(meta["scales"], float)
    n = np.broadcast_to(np.asarray(meta["n"], float), L.shape)
    t = np.array([meta["timing_seconds"][str(int(i))] for i in L]) / n
    k = L >= FIT_FROM
    q, c = curve_fit(log_affine, L[k], np.log(t[k]), p0=[np.log(t[k][0]), -25.0, 5.5])
    se = np.sqrt(np.diag(c))
    k0, k1, d = np.exp(q[0]), np.exp(q[1]), q[2]
    resid = np.log(t[k]) - log_affine(L[k], *q)
    print(f"fit on L >= {FIT_FROM}: k0 = {k0 * 1e3:.4f} ms, k1 = {k1:.4g} s, d = {d:.3f}({se[2]:.3f}), "
          f"rms log residual {np.sqrt(np.mean(resid ** 2)):.3f}")
    for i, ti in zip(L, t):
        print(f"  L={i:>3.0f} measured {ti * 1e3:9.4f} ms  fit {np.exp(log_affine(i, *q)) * 1e3:9.4f} ms")

    arm = json.loads((ZHANG / "protocol_arm.json").read_text())
    ms = np.array([float(m) for m in re.findall(r"\(([\d.]+) ms\)", ZHANG.with_suffix(".log").read_text())])
    Lz, tz = np.array(arm["scales"], float), ms / 1e3 / np.array(arm["n"], float)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.geomspace(2, 64, 200)
    ax.plot(x, np.exp(log_affine(x, *q)), lw=2, color=_BLUE, zorder=2,
            label=rf"$k_0+k_1L^d$ on $L\geq{FIT_FROM}$: $d$={d:.2f}$\pm${se[2]:.2f}, "
                  rf"$k_0$={k0 * 1e3:.3f} ms")
    ax.plot(Lz, tz, "s", ms=6, color=_MUTED, mec="white", mew=1, zorder=3,
            label=r"Zhang replicate, $n=10^5$–$10^6$ (purus), not fitted")
    ax.plot(L[k], t[k], "o", ms=8, color=_INK, mec="white", mew=1.5, zorder=4,
            label=f"cost check, n = {n[0]:.0f} per size (solimoes), fitted")
    ax.plot(L[~k], t[~k], "o", ms=8, mfc="white", mec=_INK, mew=1.5, zorder=4,
            label="cost check, start-up dominated, not fitted")
    ax.set(xscale="log", yscale="log", xlabel="$L$", ylabel="seconds per sample",
           title=r"5D percolation_zd_gpu, face_far cylinder: draw cost per sample")
    ax.set_xticks(L, [f"{v:.0f}" for v in L])
    ax.minorticks_off()
    ax.grid(color=_GRIDLINE, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path, help="generate.py run dir with samples_meta.json")
    p.add_argument("--plot", type=Path, required=True)
    a = p.parse_args()
    main(a.data, a.plot)
