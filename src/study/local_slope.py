"""Consecutive-point slopes below a finished study (plans/local_slope.md).

Takes any finished study (autopilot, or pilot -> plan -> run) and draws the slope
s_k between each pair of consecutive scales, from the smallest pilot scale up to the
top of the final ladder, then asks how s_k approaches its limit: a power law in i
(Assumption 1, eq. 232), a logarithm, or a crossover. The analysis is
`tools/local_slope.py`'s; this driver only assembles the replicate means it needs.

Samples: reuse first, draw only the gaps, one n everywhere
----------------------------------------------------------
Per-replicate n is the final's n (n_f) at EVERY scale (Igor, 2026-09-28): log Ybar is
biased by -cv^2/(2n), so unequal n on the two scales of a pair adds a spurious
cv^2/(2 log rho) (1/n_k - 1/n_{k+1}) to s_k. Each stored mean is brought to n_f:

    stored n = n_f   reused as-is
    stored n < n_f   topped up: n_f - n fresh draws at that scale, merged exactly
    stored n > n_f   dropped and redrawn at n_f (a mean cannot be shrunk)
    no stored mean   a gap: drawn fresh at n_f, with the final's replicate count

Nothing is drawn above the top of the final ladder. Every draw goes through
`generate()` with its own `spawn` child of --seed, one per (scale, replicate), and
lands in `local_slope.json` as a cell with its n, source and seed, so a rerun reuses
it. Before drawing, the predicted wall clock is printed (the pilot's throughput x the
steps to draw); there is no confirmation prompt and no cost cap (decided 2026-09-28).

--truth is reporting only: it adds panel (c), |s_k - gamma_truth| on log-log, and is
recorded as user-supplied. It never reaches a fit (ground rule 4).

CLI:
    python3 src/study/local_slope.py --study <s> --data-root <D> --seed <int>
            [--min-scale I] [--truth GAMMA] [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:    # run as a script: `tools.*`/`src.*`/`models.*`
    sys.path.insert(0, str(ROOT))   # resolve from the repo root, nowhere else

from tools.artifacts import artifact_path, read_artifact, write_artifact  # noqa: E402
from tools.constants import load, require  # noqa: E402
from tools import local_slope as LS  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools import progress as P  # noqa: E402
from tools.rng import as_seed_sequence, seed_record, spawn  # noqa: E402
from tools.summary import summarize_scale  # noqa: E402

from src.budget.allocation_table import human_time  # noqa: E402
from src.generate.generate import generate  # noqa: E402

PRODUCED_BY = "src/study/local_slope.py"

# Same palette as tools/loglog_plot.py (dataviz skill's validated default, light mode).
_BLUE, _ORANGE, _AQUA, _MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


def _entropy(record) -> int | None:
    if record is None:
        return None
    return int(as_seed_sequence(record).entropy)


def _k_of(scale: float, rho: float) -> int | None:
    """k with rho^k == scale, or None when the scale is off the rho-grid."""
    k = int(round(math.log(scale) / math.log(rho)))
    return k if np.isclose(rho ** k, scale, rtol=1e-9) else None


def stored_cells(pilot: dict, final: dict) -> list[dict]:
    """Every replicate mean the study already holds, one cell per (scale, replicate)."""
    cells = []
    for r, rep in enumerate(pilot["per_replicate"]):
        for j, i in enumerate(pilot["scales"]):
            cells.append({"scale": int(i), "key": f"pilot:{r}", "n": int(pilot["n"][j]),
                          "y_bar": rep["y_bar"][j], "cv": rep["cv"][j],
                          "source": f"pilot replicate {r}", "seeds": []})
    for r, rep in enumerate(final["per_replicate"]):
        for j, i in enumerate(final["scales"]):
            cells.append({"scale": int(i), "key": f"final:{r}", "n": int(final["n"]),
                          "y_bar": rep["y_bar"][j], "cv": rep["cv"][j],
                          "source": f"final replicate {r}",
                          "seeds": [final["seeds"][r]]})
    return cells


def make_plan(pilot: dict, final: dict, prev: dict | None, min_scale: int) -> dict:
    """Per ladder scale, what is reused and what is drawn. Draws nothing.

    Returns the ladder, the cells kept as they are, the actions (top-up, redraw, gap)
    and the stored means left out (off the ladder). A cell a previous run of this
    driver already brought to n_f is found by its (scale, key) and reused.
    """
    rho, n_f = float(final["rho"]), final["n"]
    if not isinstance(n_f, int):
        raise SystemExit(f"final.json's n is {n_f!r}: this tool needs one n per replicate "
                         f"at every scale, and a final with per-scale n has none to fix")
    k_lo, k_hi = _k_of(min_scale, rho), _k_of(max(final["scales"]), rho)
    if k_lo is None or k_hi is None:
        raise SystemExit(f"--min-scale {min_scale} and the final's top scale "
                         f"{max(final['scales'])} must both be powers of rho = {rho}")
    ladder = [int(round(rho ** k)) for k in range(k_lo, k_hi + 1)]
    if not set(final["scales"]) <= set(ladder):
        raise SystemExit(f"the final's scales {final['scales']} are not on the ladder {ladder}")

    redone = {}
    if prev is not None:
        for c in prev.get("cells", []):
            if c["n"] == n_f:
                redone[(c["scale"], c["key"])] = c

    keep, actions, ignored = [], [], []
    by_scale: dict[int, list[dict]] = {i: [] for i in ladder}
    for c in stored_cells(pilot, final):
        (by_scale[c["scale"]] if c["scale"] in by_scale else ignored).append(c)
    R_final = len(final["per_replicate"])
    for i in ladder:
        cells = by_scale[i]
        if not cells:
            cells = [{"scale": i, "key": f"gap:{r}", "n": 0} for r in range(R_final)]
        for c in cells:
            done = redone.get((i, c["key"]))
            if c["n"] == n_f:
                keep.append(c)
            elif done is not None:
                keep.append(done)
            elif c["n"] == 0:
                actions.append({"scale": i, "key": c["key"], "action": "gap",
                                "draw": n_f, "base": None})
            elif c["n"] < n_f and n_f - c["n"] >= 2:
                actions.append({"scale": i, "key": c["key"], "action": "top-up",
                                "draw": n_f - c["n"], "base": c})
            else:
                # n > n_f cannot be shrunk; a top-up of one draw has no summary
                # (tools/summary.py needs 2), so it is redrawn whole as well.
                actions.append({"scale": i, "key": c["key"], "action": "redraw",
                                "draw": n_f, "base": c})
    return {"ladder": ladder, "rho": rho, "n": n_f, "keep": keep, "actions": actions,
            "ignored": ignored}


def predicted_seconds(actions, model: str, params: dict, throughput) -> float | None:
    spec = get_model(model)
    steps = sum(a["draw"] * (spec.cost_hint(a["scale"], params) if spec.cost_hint else 1.0)
                for a in actions)
    return steps / throughput if throughput else None


def print_plan(plan: dict, seconds) -> None:
    print(f"ladder     = {plan['ladder'][0]} .. {plan['ladder'][-1]}  "
          f"(rho = {plan['rho']:g}, {len(plan['ladder'])} scales), n = {plan['n']:,} "
          f"per replicate at every scale")
    print(f"\n{'scale':>10} {'R':>3}  {'reuse':>5} {'top-up':>6} {'redraw':>6} {'gap':>4}"
          f"  {'draws':>14}")
    for i in plan["ladder"]:
        acts = [a for a in plan["actions"] if a["scale"] == i]
        kept = sum(c["scale"] == i for c in plan["keep"])
        count = {k: sum(a["action"] == k for a in acts) for k in ("top-up", "redraw", "gap")}
        print(f"{i:>10} {kept + len(acts):>3}  {kept:>5} {count['top-up']:>6} "
              f"{count['redraw']:>6} {count['gap']:>4}  {sum(a['draw'] for a in acts):>14,}")
    dropped: dict[tuple, list[int]] = {}
    for a in plan["actions"]:
        if a["action"] == "redraw":
            dropped.setdefault((a["base"]["key"].split(":")[0], a["base"]["n"]),
                               []).append(a["scale"])
    for (src, n_old), scales in dropped.items():
        print(f"  dropped: {src} means at n = {n_old:,} (cannot be shrunk to "
              f"{plan['n']:,}), redrawn at scales {sorted(set(scales))}")
    off = sorted({c["scale"] for c in plan["ignored"]})
    if off:
        print(f"  stored means off the ladder, not used: scales {off}")
    print(f"\npredicted  = {human_time(seconds) if seconds is not None else 'unknown'}"
          f" to draw {sum(a['draw'] for a in plan['actions']):,} samples "
          f"(pilot throughput x steps)")


def draw(plan: dict, model: str, params: dict, seed, on_cell) -> list[dict]:
    """Execute the plan's actions, one generate() call and one spawn child per cell."""
    out = []
    for a, ss in zip(plan["actions"], spawn(seed, len(plan["actions"]))):
        t = time.perf_counter()
        (i, stat), = generate(model, [a["scale"]], a["draw"], params, seed=ss,
                              reduce=summarize_scale).items()
        y_new, _, cv_new, _ = stat
        rec = seed_record(ss)
        if a["action"] == "top-up":
            b = a["base"]
            n, y, cv = LS.merge_means(b["n"], b["y_bar"], b["cv"], a["draw"], y_new, cv_new)
            cell = {"scale": i, "key": a["key"], "n": n, "y_bar": y, "cv": cv,
                    "source": f"{b['source']} (n = {b['n']:,}) + top-up of {a['draw']:,}",
                    "seeds": b["seeds"] + [rec]}
        else:
            what = (f"gap draw {a['key'].split(':')[1]}" if a["action"] == "gap"
                    else f"redraw of {a['base']['source']} (had n = {a['base']['n']:,})")
            cell = {"scale": i, "key": a["key"], "n": a["draw"], "y_bar": y_new,
                    "cv": cv_new, "source": what, "seeds": [rec]}
        cell["drawn_seconds"] = time.perf_counter() - t
        out.append(cell)
        on_cell(out)
    return out


def _fmt(v, se, w=8, p=5) -> str:
    return " " * (w + p + 5) if v is None else f"{v:+{w}.{p}f} +- {se:<{p + 2}.{p}f}"


def print_analysis(res: dict, d: float, n: int) -> None:
    sc = res["scales"]
    print(f"\n{'i':>10} {'R':>3} {'s_k (to rho i)':>24} {'se CLT':>9} "
          f"{'Delta s_k':>24} {'w_k':>22}")
    for k, i in enumerate(sc[:-1]):
        s = _fmt(res["s"][k], res["se_s"][k])
        clt = f"{res['se_s_clt'][k]:9.2e}"
        ds = w = ""
        if k < len(res["ds"]):
            ds = _fmt(res["ds"][k], res["se_ds"][k], 9, 6) + \
                (" " if res["ds_resolved"][k] else "?")
        if k < len(res["w"]) and res["w"][k] is not None:
            w = f"{res['w'][k]:+7.3f} +- {res['se_w'][k]:.3f}" + \
                (" " if res["w_resolved"][k] else "?")
        print(f"{i:>10} {res['R'][k]:>3} {s:>24} {clt} {ds:>24} {w:>22}")
    print(f"  ? = unresolved (|Delta s| <= {LS.RESOLVED_Z:g} se): not read")
    print(f"  n-mismatch term, max |.| = {np.max(np.abs(res['n_mismatch'])):.2e} "
          f"(0 by construction: n = {n:,} at every scale)")

    fp, fl = res["fits"]["power"], res["fits"]["log"]
    print("\nfits to s_k, weighted by se (compared by chi2 and p only):")
    if fp is not None:
        om = f"{fp['omega']:.4f}" + (f" +- {fp['se_omega']:.4f}" if "se_omega" in fp else "")
        p = "n/a" if fp["p_value"] is None else f"{fp['p_value']:.3g}"
        print(f"  (P) gamma + c i^-omega : gamma = {fp['gamma']:.6f}, omega = {om}"
              f"{' [ON BOUND]' if fp['omega_on_bound'] else ''}, c = {fp['c']:+.4g};"
              f"  chi2 = {fp['chi2']:.2f} / {fp['dof']} dof, p = {p}")
    if fl is not None:
        p = "n/a" if fl["p_value"] is None else f"{fl['p_value']:.3g}"
        print(f"  (L) gamma + c / ln i   : gamma = {fl['gamma']:.6f} +- {fl['se_gamma']:.6f}, "
              f"c = {fl['c']:+.4g};  chi2 = {fl['chi2']:.2f} / {fl['dof']} dof, p = {p}")
    if fp is not None:
        om = fp["omega"]
        print(f"\nbudget (d = {d:.4f}): a pair costs B_k ~ n i^d, so at this fixed n the "
              f"bias falls like B^-(omega/d) = B^-{om / d:.4f};\n  with n grown as "
              f"eqs. (945)-(946), eq. (941)/(966) give B^-(omega/(d + 2 omega)) = "
              f"B^-{om / (d + 2 * om):.4f}  (omega from (P))")


def plot(sd: Path, res: dict, *, model: str, params: dict, n: int, d: float,
         truth: float | None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = res["rho"]
    sc = np.asarray(res["scales"], float)
    lower = sc[:-1]
    x = np.log2(lower)
    panels = 3 if truth is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.4))

    ax = axes[0]
    ax.errorbar(x, res["s"], yerr=res["se_s"], fmt="o", color=_BLUE, ms=5, capsize=2,
                label=r"$s_k$ $\pm$ se")
    xx = np.linspace(x.min(), x.max(), 200)
    fp, fl = res["fits"]["power"], res["fits"]["log"]
    if fp is not None:
        ax.plot(xx, fp["gamma"] + fp["c"] * (2.0 ** xx) ** -fp["omega"], "--", color=_ORANGE,
                lw=1.5, label=rf"(P) $\omega$={fp['omega']:.3g}, $\chi^2$={fp['chi2']:.1f}/{fp['dof']}")
    if fl is not None:
        ax.plot(xx, fl["gamma"] + fl["c"] / (xx * np.log(2)), ":", color=_AQUA, lw=1.8,
                label=rf"(L) $c/\ln i$, $\chi^2$={fl['chi2']:.1f}/{fl['dof']}")
    ax.set_xlabel(r"$\log_2 i$  ($s_k$ between $i$ and $\rho i$)")
    ax.set_ylabel(r"$s_k$")
    ax.set_title("(a) consecutive-point slopes")
    ax.legend(fontsize=8, frameon=False)
    # B_k = n (i^d + (rho i)^d) per replicate: one sample of each scale of the pair.
    off = math.log10(n) + math.log10(1 + rho ** d)
    sec = ax.secondary_xaxis("top", functions=(lambda v: off + d * v * math.log10(2),
                                               lambda b: (b - off) / (d * math.log10(2))))
    sec.set_xlabel(rf"$\log_{{10}} B_k$ ($d$ = {d:.3g})", fontsize=8)

    ax = axes[1]
    w = np.array([np.nan if v is None else v for v in res["w"]], float)
    se_w = np.array([np.nan if v is None else v for v in res["se_w"]], float)
    ok = np.array(res["w_resolved"], bool)
    xw = x[:w.size]
    ax.errorbar(xw[ok], w[ok], yerr=se_w[ok], fmt="o", color=_BLUE, ms=5, capsize=2,
                label="resolved")
    ax.errorbar(xw[~ok], w[~ok], yerr=se_w[~ok], fmt="o", color=_BLUE, mfc="none", ms=5,
                capsize=2, alpha=0.6, label="unresolved (not read)")
    if fp is not None:
        ax.axhline(fp["omega"], color=_MUTED, lw=1, ls="--", label=r"$\omega$ from (P)")
    ax.axhline(0, color=_MUTED, lw=0.6)
    ax.set_xlabel(r"$\log_2 i$")
    ax.set_ylabel(r"$w_k = -\log(\Delta s_{k+1}/\Delta s_k)/\log\rho$")
    ax.set_title("(b) local exponent of the differences")
    ax.legend(fontsize=8, frameon=False)

    if truth is not None:
        ax = axes[2]
        dev = np.abs(np.asarray(res["s"]) - truth)
        ax.errorbar(lower, dev, yerr=res["se_s"], fmt="o", color=_BLUE, ms=5, capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$i$")
        ax.set_ylabel(r"$|s_k - \gamma_{\rm truth}|$")
        ax.set_title(rf"(c) against user-supplied $\gamma$ = {truth:g}")

    for a in axes:
        a.grid(True, color="#e1e0d9", lw=0.6)
    fig.suptitle(f"{model} {params}  --  n = {n:,} per replicate at every scale", fontsize=10)
    fig.tight_layout()
    out = sd / "local_slope.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="root seed for every top-up, redraw and gap draw; recorded, and "
                        "refused if the study or an earlier run here already used it")
    p.add_argument("--min-scale", type=int, default=None,
                   help="bottom of the ladder (default: the pilot's smallest scale)")
    p.add_argument("--truth", type=float, default=None,
                   help="reporting only: adds panel (c), |s_k - GAMMA|; never fitted")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be reused and drawn, and the predicted time")
    a = p.parse_args(argv)

    sd = Path(a.data_root) / a.study
    pilot = read_artifact(sd, "pilot")
    final = read_artifact(sd, "final")
    prev = read_artifact(sd, "local_slope", required=False)
    model, params = final["model"], final["params"]
    if (pilot["recipe"]["model"], pilot["recipe"].get("params", {})) != (model, params):
        raise SystemExit(f"pilot ({pilot['recipe']['model']} {pilot['recipe'].get('params')}) "
                         f"and final ({model} {params}) are not the same configuration")
    if prev is not None and (prev["model"], prev["params"]) != (model, params):
        raise SystemExit(f"{artifact_path(sd, 'local_slope')} is for "
                         f"{prev['model']} {prev['params']}, not {model} {params}")
    d = require(load(sd), "d").value

    min_scale = a.min_scale if a.min_scale is not None else min(pilot["scales"])
    plan = make_plan(pilot, final, prev, min_scale)
    seconds = predicted_seconds(plan["actions"], model, params, pilot.get("throughput"))
    print(f"study      = {sd}")
    print(f"model      = {model}  params={params}")
    print_plan(plan, seconds)

    used = {_entropy(pilot["recipe"].get("seed"))} | {_entropy(s) for s in final["seeds"]}
    auto = read_artifact(sd, "autopilot", required=False)
    if auto is not None:
        used.add(_entropy(auto.get("seed")))
    used |= set((prev or {}).get("seeds_used", []))
    used.discard(None)
    if plan["actions"] and a.seed in used:
        raise SystemExit(f"--seed {a.seed} was already used by this study (seeds used: "
                         f"{sorted(used)}); its spawn children would repeat those streams. "
                         f"Pass another --seed.")
    if a.dry_run:
        print("\n--dry-run: nothing drawn.")
        return

    t0 = time.perf_counter()
    seeds_used = sorted(set((prev or {}).get("seeds_used", []))
                        | ({a.seed} if plan["actions"] else set()))
    base = {"model": model, "params": params, "rho": plan["rho"], "n": plan["n"],
            "ladder": plan["ladder"], "min_scale": min_scale, "seed": a.seed,
            "seeds_used": seeds_used, "predicted_seconds": seconds,
            "truth": None if a.truth is None else
            {"value": a.truth, "source": "user-supplied (--truth), reporting only"}}

    def checkpoint(new_cells):
        write_artifact(sd, "local_slope", {
            **base, "complete": False, "cells": plan["keep"] + new_cells,
            "elapsed_seconds": time.perf_counter() - t0}, produced_by=PRODUCED_BY)

    if plan["actions"]:
        P.say(f"drawing {len(plan['actions'])} cell(s) ...")
    new = draw(plan, model, params, a.seed, checkpoint)
    cells = plan["keep"] + new

    by_scale = {i: [c for c in cells if c["scale"] == i] for i in plan["ladder"]}
    n_per = [sorted({c["n"] for c in by_scale[i]}) for i in plan["ladder"]]
    assert all(v == [plan["n"]] for v in n_per), n_per
    res = LS.analyse(plan["ladder"], [[c["y_bar"] for c in by_scale[i]] for i in plan["ladder"]],
                     plan["rho"], cv=[np.mean([c["cv"] for c in by_scale[i]])
                                      for i in plan["ladder"]],
                     n=[plan["n"]] * len(plan["ladder"]))
    elapsed = time.perf_counter() - t0
    out = write_artifact(sd, "local_slope", {
        **base, "complete": True, "cells": cells, "d": d, "analysis": res,
        "per_scale": [{"scale": i, "n": plan["n"], "R": len(by_scale[i]),
                       "sources": [c["source"] for c in by_scale[i]]}
                      for i in plan["ladder"]],
        "drawn_cells": len(new), "elapsed_seconds": elapsed}, produced_by=PRODUCED_BY)
    print_analysis(res, d, plan["n"])
    png = plot(sd, res, model=model, params=params, n=plan["n"], d=d, truth=a.truth)
    print(f"\ndrew {len(new)} cell(s) in {human_time(elapsed)}"
          + (f" (predicted {human_time(seconds)})" if seconds and new else ""))
    print(f"output = {out}\n         {png}")


if __name__ == "__main__":
    _main()
