"""Experiment 24: Assumption 7's cost exponent across dimension, from the pilots' own probes.

Every pilot in experiments 20-31 already ran five batched cost probes (`pilot.json`'s
`cost`), so this draws nothing: it collects them per dimension and asks how far the
clock's d sits from the declared i**dim, and where along the ladder the gap opens.
The per-site time elapsed(i) / i**dim is flat if the declared cost is exact; its local
slope between consecutive rungs is the drift d_local - dim.

CLI:
    python3 experiments/24_perc_cost_d2_d5/cost_from_pilots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: One `face_far`/cylinder pilot per dimension (d = 2 has no face_far; its south
#: anchor on the cylinder is the same count). Chosen as the widest ladder per dim.
PILOTS: dict[int, str] = {
    2: "20_perc_df_d2_clt",
    3: "27_perc_df_d3_facefar",
    4: "29_perc_df_d4_facefar",
    5: "30_perc_df_d5_facefar",
    6: "31_perc_df_d6_facefar",
}


def collect() -> list[dict]:
    rows = []
    for dim, exp in PILOTS.items():
        p = json.loads((ROOT / "experiments" / exp / "data" / "pilot" / "pilot.json").read_text())
        c = p["cost"]
        scales = np.array(c["scales"], dtype=float)
        per_site = np.array(c["elapsed"]) / scales**dim
        probes = np.array(c["d_probes"])
        rows.append({
            "dim": dim, "exp": exp, "model": p["recipe"]["model"],
            "d_hat": float(probes.mean()), "d_sd": float(probes.std(ddof=1)),
            "scales": scales.tolist(), "elapsed": c["elapsed"], "ns_per_site": (per_site * 1e9).tolist(),
            "local_drift": (np.diff(np.log(per_site)) / np.diff(np.log(scales))).tolist(),
        })
    return rows


def plot(rows: list[dict], out: Path) -> None:
    from matplotlib.figure import Figure

    from tools.loglog_plot import _AQUA, _BLUE, _GRIDLINE, _INK, _MUTED, _ORANGE

    colors = [_BLUE, _ORANGE, _AQUA, _INK, _MUTED]
    fig = Figure(figsize=(11, 4.2), constrained_layout=True)
    a, b = fig.subplots(1, 2)
    for r, col in zip(rows, colors):
        a.loglog(r["scales"], r["elapsed"], "o-", color=col, label=f"dim {r['dim']}: d̂ = {r['d_hat']:.3f}")
        b.semilogx(r["scales"], r["ns_per_site"], "o-", color=col, label=f"dim {r['dim']}")
    a.set(xlabel="box side i", ylabel="seconds per sample (batched probe)", title="(a) cost per sample")
    b.set(xlabel="box side i", ylabel="ns per site  = elapsed / i^dim", title="(b) per-site cost: flat iff cost = i^dim")
    for ax in (a, b):
        ax.grid(True, which="both", color=_GRIDLINE, lw=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(out, dpi=130)


def main() -> None:
    rows = collect()
    print("| dim | pilot | d̂ (5 probes) | d̂ / dim | ns/site, smallest → largest i | local drift d_local − dim per doubling |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ns = r["ns_per_site"]
        drift = ", ".join(f"{x:+.2f}" for x in r["local_drift"])
        print(f"| {r['dim']} | {r['exp']} | {r['d_hat']:.3f} ± {r['d_sd']:.3f} | {r['d_hat'] / r['dim']:.3f} "
              f"| {ns[0]:.3g} → {ns[-1]:.3g} | {drift} |")
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    (out / "cost_from_pilots.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out / "cost_from_pilots.png")
    print(f"\noutput = {out / 'cost_from_pilots.json'}\nfigure = {out / 'cost_from_pilots.png'}")


if __name__ == "__main__":
    main()
