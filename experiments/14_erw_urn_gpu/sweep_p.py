"""Run `autopilot.py` on `erw_urn_gpu` over a sieve of p, one study per p, with every
final run's n kept at or above a full device.

This is experiments/12_erw_gpu/sweep_p.py (same p grid, same summary tables, same
resume rule) for the urn sampler, plus one thing 12 never needed: a ceiling on the
ladder so that the plan cannot starve the GPU of samples.

Why a ceiling (user, 2026-09-28: "be careful with the number of samples to use the
GPU efficiently"). The urn runs one sample per thread, so a call of n samples costs
one sample's latency until n fills the device (models/erw_urn_gpu.py,
`resident_threads`: 261,120 on the RTX 5090). plan.py draws the same n at every
scale and, when omega1 is small, spends a growing budget by sliding the ladder up
rather than raising n: 12's 1h sweep planned n = 131,286 on 262144..8388608 at
p = 0.775. The urn is ~240x faster, so the same plan would slide further and drop n
well under a full device, where the time plan.py predicts is wrong and most of the
card idles. So before each study this script finds the highest ladder on which the
final run can still afford `--min-n` samples per scale per replicate, and hands its
top as autopilot's `--max-scale`. Every lower ladder then affords more samples, so
whatever m0 the plan picks, n >= --min-n.

The ceiling, per study:

    K_max = ladder(m0, m, rho)[-1]  for the largest m0 with
            R * min_n * sum(ladder(m0, m, rho)) <= (1 - pilot_cap) * T * throughput

T is --time, (1 - pilot_cap) T is the least the final run gets (autopilot's pilot
may use at most pilot_cap of it), and R, m, rho and pilot_cap are autopilot's (read
from the arguments after `--`, else its defaults). A --max-scale after `--` is kept
if lower.

`throughput` must be the rate plan.py will SPEND by, not the device's: the plan's n
on a given ladder is (seconds x throughput) / sum(k), so a ceiling sized on a faster
rate than the plan's lets n land under min_n. plan.py's rate is pilot.json's, which
for a batched model is sum(k) / sum(a + b k^d) from the pilot's cost probe on the
pilot's own scales (src/study/pilot.py; the pilot draws the same n at every scale, so
its counts cancel). This script runs that same probe, `pilot.resolve_cost`, on the
template's scales -- once per sweep, since p does not enter the work -- and divides
by THROUGHPUT_MARGIN for its run-to-run spread. Measured 2026-09-28 on the RTX 5090:
four probes gave 1.80-2.15e11 steps/s, while the device runs a full final at ~4.1e11
(README, "The pilot's clock"): the pilot's 4..4096 ladder sees mostly per-sample
fixed cost, so plans are conservative and finals take about half their budget.

    python3 experiments/14_erw_urn_gpu/sweep_p.py --tag 1h --time 1h
    python3 experiments/14_erw_urn_gpu/sweep_p.py --tag 1h --time 1h --p 0.75 -- --replicates 5

Anything after `--` is passed to every `autopilot.py` call. The summary adds a
`full device` column: the final run's n against --min-n.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import erw_urn_gpu  # noqa: E402
from src.study import pilot as pilot_mod  # noqa: E402
from src.study.autopilot import PILOT_CAP  # noqa: E402
from src.study.plan import parse_duration  # noqa: E402
from tools.allocation import ladder  # noqa: E402
from tools.models import get_model  # noqa: E402

TEMPLATE = EXP / "recipes" / "samples_super_p0.9.json"

P_GRID = [0.75, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.725, 0.775, 0.8, 0.9]

#: What the planner's throughput is divided by before sizing the ceiling: four
#: re-runs of the pilot's probe spread 1.80-2.15e11 steps/s (2026-09-28), so the
#: pilot's own run may read up to ~20% below this script's.
THROUGHPUT_MARGIN = 1.25


def autopilot_settings(extra: list[str]) -> argparse.Namespace:
    """The autopilot arguments the ceiling depends on, as autopilot would read them."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--pilot-cap", type=float, default=PILOT_CAP, dest="pilot_cap")
    ap.add_argument("--max-scale", type=float, default=None, dest="max_scale")
    return ap.parse_known_args(extra)[0]


def without_max_scale(extra: list[str]) -> list[str]:
    """`extra` less any --max-scale, which this script replaces by the lower ceiling."""
    out, skip = [], False
    for a in extra:
        if skip:
            skip = False
        elif a == "--max-scale":
            skip = True
        elif not a.startswith("--max-scale="):
            out.append(a)
    return out


def planner_throughput(template: dict) -> float:
    """The throughput pilot.py will hand plan.py for this template, re-measured here.

    pilot.py's formula (the batched branch after `resolve_cost`), with equal counts.
    """
    scales = [int(k) for k in template["scales"]]
    cost = pilot_mod.resolve_cost(template["model"], template["params"], scales)
    aff = cost["affine"]
    spec = get_model(template["model"])
    work = sum(spec.cost_hint(k, template["params"]) for k in scales)
    return work / sum(aff["a"] + aff["b"] * float(k) ** aff["d"] for k in scales)


def ceiling(seconds: float, throughput: float, min_n: int, s: argparse.Namespace) -> dict:
    """The highest ladder whose final run affords min_n samples per scale per replicate."""
    steps = (1 - s.pilot_cap) * seconds * throughput
    best = None
    m0 = 0
    while s.replicates * min_n * sum(ladder(m0, s.m, s.rho)) <= steps:
        best = m0
        m0 += 1
    if best is None:
        raise SystemExit(
            f"[sweep] even the lowest ladder {ladder(0, s.m, s.rho)} cannot draw "
            f"{min_n:,} samples per scale x {s.replicates} replicates in "
            f"(1 - {s.pilot_cap:g}) x {seconds:g} s at {throughput:.3g} steps/s. "
            f"Give more --time, fewer replicates, or a lower --min-n.")
    top = ladder(best, s.m, s.rho)[-1]
    if s.max_scale is not None:
        top = min(top, int(s.max_scale))
    return {"max_scale": top, "m0_max": best}


def write_recipe(p: float) -> Path:
    recipe = json.loads(TEMPLATE.read_text())
    recipe["params"]["p"] = p
    recipe["_note"] = [
        f"GENERATED by experiments/14_erw_urn_gpu/sweep_p.py: {TEMPLATE.name} with only",
        "params.p changed. Edit the template or the script, not this file.",
    ]
    path = EXP / "recipes" / f"samples_sweep_p{p}.json"
    path.write_text(json.dumps(recipe, indent=2) + "\n")
    return path


def gamma_true(p: float) -> float:
    """ERW: diffusive (1/2) up to the critical p = 3/4, superdiffusive 2p - 1 above."""
    return 0.5 if p <= 0.75 else 2 * p - 1


def ladder_str(scales: list) -> str:
    return f"{scales[0]}..{scales[-1]} ({len(scales)})"


def draws(n: int, replicates: int) -> str:
    return f"{n:,} × {replicates}"


def summary_rows(p: float, study: str, data: Path, code: int | None,
                 min_n: int) -> tuple[str, str]:
    """(gamma row, omega1 row) for one study."""
    ap_path = data / study / "autopilot.json"
    if not ap_path.exists():
        return (f"| {p} | `{study}` | exit {code} | {gamma_true(p):.3f} | - | - | - | - | - | - |",
                f"| {p} | - | - | - | - | - | - | - | - |")
    ap = json.loads(ap_path.read_text())
    pilot = json.loads((data / study / "pilot.json").read_text())
    om = json.loads((data / study / "constants.json").read_text())["omega1"]
    assert len(set(pilot["n"])) == 1, f"{study}: pilot n varies by scale: {pilot['n']}"
    assert pilot["direct_fit"]["omega1"] == om["value"], f"{study}: omega1 not from direct_fit"
    om_pilot = (f"| {p} | {om['value']:.3f} ± {om['se']:.3f} | {pilot['direct_fit']['converged']} "
                f"| {ladder_str(pilot['scales'])} | {draws(pilot['n'][0], pilot['replicates'])}")
    if not ap.get("drawn"):
        return (f"| {p} | `{study}` | not drawn (gate) | {gamma_true(p):.3f} | - | - | - | - | - "
                f"| {ap['pilot_seconds'] / 60:.1f} |",
                f"{om_pilot} | - | - | - | - |")
    answer = json.loads((data / study / "answer.json").read_text())
    final_s = json.loads((data / study / "final.json").read_text())["elapsed_seconds"]
    wall = f"{(ap['pilot_seconds'] + final_s) / 60:.1f}"
    lo, hi = answer["wilson"]["interval"]
    forced = " (forced)" if ap.get("forced") else ""
    final = draws(answer["n"], answer["replicates"])
    full = "yes" if answer["n"] >= min_n else f"**no** ({answer['n'] / min_n:.2f})"
    fit = answer["fit"]
    return ((f"| {p} | `{study}` | drawn{forced} | {gamma_true(p):.3f} "
             f"| {ap['gamma']:.5f} ± {answer['se']:.5f} | [{lo:.4f}, {hi:.4f}] "
             f"| {answer['scales'][0]}..{answer['scales'][-1]} | {final} | {full} | {wall} |"),
            (f"{om_pilot} | {fit['omega1']:.3f} | {fit['a1']:.3g} "
             f"| {ladder_str(answer['scales'])} | {final} |"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="study name suffix: sweep_p<p>_<tag>")
    ap.add_argument("--time", required=True, help="autopilot --time for EACH p, e.g. 1h")
    ap.add_argument("--p", type=float, nargs="+", default=P_GRID, help="p values, run in this order")
    ap.add_argument("--min-n", type=int, default=None, dest="min_n",
                    help="least samples per scale per replicate in the final run "
                         "(default: the device's resident threads)")
    ap.add_argument("--data-root", type=Path, default=EXP / "data",
                    help="where the studies live (default: this experiment's data/)")
    ap.add_argument("autopilot_args", nargs=argparse.REMAINDER,
                    help="after `--`: extra arguments for every autopilot.py call")
    args = ap.parse_args()
    extra = [a for a in args.autopilot_args if a != "--"]
    settings = autopilot_settings(extra)
    extra = without_max_scale(extra)

    min_n = args.min_n if args.min_n is not None else erw_urn_gpu.resident_threads()
    throughput = planner_throughput(json.loads(TEMPLATE.read_text())) / THROUGHPUT_MARGIN
    cap = ceiling(parse_duration(args.time), throughput, min_n, settings)
    print(f"[sweep] min_n = {min_n:,}, planner throughput / {THROUGHPUT_MARGIN:g} = {throughput:.3g} steps/s, "
          f"R = {settings.replicates}, m = {settings.m}, rho = {settings.rho:g}, "
          f"pilot_cap = {settings.pilot_cap:g} -> --max-scale {cap['max_scale']} "
          f"(ladder {ladder(cap['m0_max'], settings.m, settings.rho)})", flush=True)

    data = args.data_root.resolve()
    rows = []
    failed = []
    for p in args.p:
        study = f"sweep_p{p}_{args.tag}"
        if (data / study / "autopilot.json").exists():
            print(f"[sweep] p={p}: {study} already finished, skipping", flush=True)
            rows.append((p, summary_rows(p, study, data, None, min_n)))
            continue
        recipe = write_recipe(p)
        data.mkdir(exist_ok=True)
        log = data / f"{study}.log"
        # -u: see 12_erw_gpu/sweep_p.py -- piped stdout is block-buffered.
        cmd = [sys.executable, "-u", str(ROOT / "src/study/autopilot.py"), "-meta", str(recipe),
               "--study", study, "--data-root", str(data), "--time", args.time,
               "--max-scale", str(cap["max_scale"]), "--no-progress", *extra]
        print(f"[sweep] p={p}: {' '.join(cmd)}\n[sweep]   log -> {log}", flush=True)
        t0 = time.monotonic()
        with log.open("w") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
                fh.flush()
            code = proc.wait()
        minutes = (time.monotonic() - t0) / 60
        print(f"[sweep] p={p}: exit {code} after {minutes:.1f} min", flush=True)
        rows.append((p, summary_rows(p, study, data, code, min_n)))
        if code != 0:
            failed.append(p)

    rows = [r for _, r in sorted(rows, key=lambda t: t[0])]
    table = "\n".join([
        f"# erw_urn_gpu p sweep, tag `{args.tag}`, --time {args.time} per p",
        "",
        "True gamma: 1/2 for p <= 3/4, 2p - 1 above (the two agree at p = 3/4).",
        "",
        f"Ceiling: --max-scale {cap['max_scale']}, so that the final run draws at least "
        f"{min_n:,} samples per scale per replicate (the pilot probe's throughput "
        f"/ {THROUGHPUT_MARGIN:g} = {throughput:.3g} steps/s).",
        "",
        "## gamma_hat",
        "",
        "| p | study | status | gamma true | gamma_hat ± replicate se | Wilson 95% "
        "| final ladder | n/scale × reps | full device | min (pilot + draw) |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(g for g, _ in rows),
        "",
        "`n/scale × reps` = draws per scale in each replicate × independent replicates.",
        f"`full device` = n/scale >= {min_n:,}; if not, the ratio.",
        "",
        "## omega1",
        "",
        "As in 12_erw_gpu: **pilot** is `constants.json`'s eq. (232) fit, which sized the",
        "plan; **final fit** is `answer.json[\"fit\"]`, mostly noise. Read omega1 from the pilot.",
        "",
        "| p | omega1 pilot ± se | pilot converged | pilot scales | pilot n/scale × reps "
        "| omega1 final fit | a1 final fit | final scales | final n/scale × reps |",
        "|---|---|---|---|---|---|---|---|---|",
        *(o for _, o in rows),
        "",
        "Scales are dyadic; `lo..hi (k)` = k scales from lo to hi.",
    ])
    out = data / f"sweep_p_{args.tag}.md"
    out.write_text(table + "\n")
    print(f"\n{table}\n\n[sweep] summary -> {out}", flush=True)
    if failed:
        print(f"[sweep] nonzero exit for p = {failed}; see their logs", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
