"""How long would the exact box ladder take to match the literature's precision?

The method of experiments/exponents_report.md section 4, for the one model it
could not plan: gamma_susc in d = 4, which the x-ladder pilot left unidentified
(omega_1 = 0.025 +/- 1.7) and the exact box ladder (`percolation_susceptibility_L_gpu`)
identifies (omega_L = 2.04 +/- 0.07).

    python3 experiments/09_percolation_susceptibility_gpu/time_to_precision.py \\
        --data-root experiments/09_percolation_susceptibility_gpu/data \\
        --run autopilot_gsusc_L_d4 --sigma 0.006 --reference 1.430

What is computed, and what is not new:

- Times are `src/study/plan.py`'s own `budget_for_target` on the pilot's constants,
  the smallest budget whose predicted error on the MEAN of R replicates,
  sqrt(bias**2 + sd**2/R), reaches the target. Only the budget ceiling is raised
  (1e18 -> 1e40) so astronomical answers print as numbers, not "unreachable".
- The tools work in gamma_L, the exponent against the box side L. The literature
  quotes gamma_susc = nu_box * gamma_L, so a target sigma on gamma_susc is
  sigma / nu_box on gamma_L. Forgetting that factor understates the cost.
- "X" marks a top rung whose box has more sites than an int32 label can address
  (2**31 - 1): such a plan cannot run on the current sampler however long it is
  given. Sites per sample are L**dim exactly on this ladder.
- The last column ignores bias: the pilot's own wall clock scaled by
  (se/target)**2 at its deepest four-rung window. A floor for the ladder, not a plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study.plan import _cost_ratio, budget_for_target  # noqa: E402
from tools.allocation import allocation_constants, max_m0_for_scale  # noqa: E402
from tools.artifacts import artifact_path  # noqa: E402
from tools.constants import load, require  # noqa: E402
from tools.loglog import closed_form_weights, gamma_closed_form  # noqa: E402
from tools.models import get_model  # noqa: E402
from tools.summary import LOG_MOMENT_DELTA  # noqa: E402
from tools.wilson import sigma_se_per_scale, wilson_interval  # noqa: E402

INT32_SITES = 2**31 - 1


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    d = s / 86400
    if d < 365.25:
        return f"{d:.0f} d" if d >= 10 else f"{d:.1f} d"
    y = d / 365.25
    if y < 1e4:
        return f"{y:,.0f} yr" if y >= 10 else f"{y:.1f} yr"
    e = int(math.floor(math.log10(y)))
    return f"{y / 10**e:.1f}·10^{e} yr"


def predicted_bias(m: int, m0: int, rho: float, a1: float, omega1: float) -> float:
    """Signed E[gamma_hat_L] - gamma_L of the eq. (526) estimator under one correction.

    The magnitude is `allocation_constants`' Cb * rho**(-m0*omega1), which is
    what plan.py plans on; the sign is a1's times the weights', which the
    planner discards.
    """
    j = np.arange(1, m + 1, dtype=float)
    w = closed_form_weights(m)
    return float(a1 * np.sum(w * rho ** (-(m0 + j) * omega1)) / math.log(rho))


class Ladder:
    """Constants and cost model for one pilot, and the plan for a target."""

    def __init__(self, sd: Path, rho: float, m: int, replicates: int):
        self.sd, self.rho, self.m, self.R = sd, rho, m, replicates
        self.consts = load(sd)
        self.pilot = json.loads(artifact_path(sd, "pilot").read_text())
        recipe = self.pilot["recipe"]
        if "nu_box" not in recipe.get("params", {}):
            raise SystemExit(f"{sd.name} is not an exact-box-ladder pilot (no nu_box in its "
                             f"params); the gamma_susc = nu_box * gamma_L conversion needs it")
        self.nu = float(recipe["params"]["nu_box"])
        self.params = recipe["params"]
        self.spec = get_model(recipe["model"])
        self.d = require(self.consts, "d").value
        self.omega1 = require(self.consts, "omega1")
        self.a1 = require(self.consts, "a1").value
        self.cv = require(self.consts, "cv").value
        self.throughput = self.pilot["throughput"]
        self.ratio, _ = _cost_ratio(self.pilot, self.d)

    def plan(self, target_L: float, *, omega1: float | None = None, a1: float | None = None,
             m: int | None = None, max_scale: float | None = None) -> dict:
        return budget_for_target(
            target_L, replicates=self.R, hi=1e40, d=self.d,
            omega1=self.omega1.value if omega1 is None else omega1, rho=self.rho,
            m=self.m if m is None else m, a1=self.a1 if a1 is None else a1, cv=self.cv,
            throughput=self.throughput, cost_ratio=self.ratio, max_scale=max_scale)

    def sites(self, L: float) -> float:
        return float(self.spec.cost_hint(L, self.params))

    def describe(self, p: dict) -> str:
        if not p.get("feasible"):
            return f"unreachable: {p['why']}"
        top = p["scales"][-1]
        big = self.sites(top) > INT32_SITES
        return (f"{fmt_time(p['seconds'] * self.R)}"
                f" (m0={p['m0']}, L={p['scales'][0]}..{top}, {self.sites(top):.2g} sites/sample"
                f"{' ✗' if big else ''})")

    def stat_only_seconds(self, target_L: float) -> tuple[float, float]:
        """(seconds, se of the pilot's deepest 4-rung window) with the bias ignored."""
        pl = self.pilot
        n = np.array(pl["n"], float)[-4:] * pl["replicates"]
        cv2 = np.array(pl["cv_per_scale"], float)[-4:] ** 2
        se = sigma_se_per_scale(n, 4, self.rho, cv2)
        return pl["drawn_seconds"] * (se / target_L) ** 2, se


class Source:
    """Pooled per-scale summaries of one draw (a pilot or a run) on the ladder."""

    def __init__(self, name: str, replicates: int, scales, y_bar, n_pool, cv, log_moment_by_rep):
        self.name, self.R = name, replicates
        self.scales = [int(s) for s in scales]
        self.y = np.array(y_bar, float)
        self.n = np.array(n_pool, float)
        self.cv = np.array(cv, float)
        self.lm = np.mean(np.array(log_moment_by_rep, float), axis=0)

    @classmethod
    def from_pilot(cls, L: Ladder) -> "Source":
        pl = L.pilot
        return cls(f"pilot ({pl['replicates']} reps, Neyman n)", pl["replicates"],
                   pl["scales"], pl["y_bar"],
                   np.array(pl["n"], float) * pl["replicates"], pl["cv_per_scale"],
                   [r["log_moment"] for r in pl["per_replicate"]])

    @classmethod
    def from_run(cls, run_sd: Path) -> "Source":
        a = json.loads(artifact_path(run_sd, "answer").read_text())
        f = json.loads(artifact_path(run_sd, "final").read_text())
        return cls(f"run ({a['replicates']} reps, n = {a['n']:,})", a["replicates"],
                   a["scales"], a["y_bar"],
                   np.full(len(a["scales"]), a["n"] * a["replicates"], float), a["cv"],
                   [r["log_moment"] for r in f["per_replicate"]])


def window_row(L: Ladder, src: Source, m: int, reference: float, start: int | None = None) -> dict:
    """gamma_susc on the deepest m rungs of `src`, against the literature, and what the
    pilot's own (omega_L, a1) predicted that gap would be. eq. (720) as
    exponents_report.md extends it: per-scale variance, harmonic-mean n, and the mean
    of R replicates shrinks only the statistical term -- B_good and B_bad keep the
    per-replicate n. The window is the deepest m rungs unless `start` says where it begins."""
    rho = L.rho
    first = len(src.scales) - m if start is None else start
    sl = slice(first, first + m)
    scales, y, n, cv2 = src.scales[sl], src.y[sl], src.n[sl], src.cv[sl] ** 2
    m0 = int(round(math.log(scales[0], rho))) - 1
    g = float(gamma_closed_form(scales, y, rho, m0))
    se = sigma_se_per_scale(n, m, rho, cv2)     # n is pooled over replicates: the se of the mean
    n_rep = n / src.R
    w = wilson_interval(g, float(len(n_rep) / np.sum(1.0 / n_rep)), m, m0, rho,
                        sigma_inf2=float(cv2[-1]), sigma_max2=float(cv2.max()),
                        a1=L.a1, omega1=L.omega1.value, Lambda=float(src.lm[sl].max()),
                        delta=LOG_MOMENT_DELTA, se_override=se)
    return {"scales": scales, "m0": m0, "gamma_susc": L.nu * g, "se": L.nu * se,
            "half": L.nu * w["half_width"], "gap": L.nu * g - reference,
            "predicted": L.nu * predicted_bias(m, m0, rho, L.a1, L.omega1.value)}


def floor_susc(L: Ladder, m: int, m0: int, omega1: float) -> float:
    """The bias no budget removes on this ladder, on gamma_susc."""
    return L.nu * allocation_constants(L.d, omega1, L.rho, m, L.a1, L.cv)["Cb"] \
        * L.rho ** (-m0 * omega1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--study", default="pilot_gsusc_L_d4", help="the pilot whose constants to plan on")
    p.add_argument("--run", default=None, help="a finished study on the same ladder, for the "
                   "predicted-vs-observed bias check")
    p.add_argument("--sigma", type=float, default=0.006,
                   help="the literature's uncertainty on gamma_susc: 1.430(6) -> 0.006")
    p.add_argument("--reference", type=float, default=1.430, help="the literature's gamma_susc")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--m", type=int, default=5, help="rungs per ladder; the last study used 5")
    a = p.parse_args(argv)

    L = Ladder(a.data_root / a.study, a.rho, a.m, a.replicates)
    w = L.omega1
    print(f"study {a.study}: d = {L.d:.4f}, omega_L = {w.value:.4f} +/- {w.se:.4f}, "
          f"a1 = {L.a1:.4f}, cv = {L.cv:.4f}, throughput = {L.throughput:.3g} sites/s, "
          f"nu_box = {L.nu}; R = {L.R}, rho = {L.rho:g}, m = {L.m}\n")

    print(f"## Extrapolation check: gamma_susc against the literature {a.reference} +/- {a.sigma}\n")
    print("| draw | window | gamma_susc | stat se | eq. (720) +/- | observed - literature "
          "| predicted by the pilot's (omega_L, a1) |")
    print("|---|---|---|---|---|---|---|")
    sources = [Source.from_pilot(L)] + ([Source.from_run(a.data_root / a.run)] if a.run else [])
    for src in sources:
        for m in sorted({len(src.scales), 4, 3}, reverse=True):
            r = window_row(L, src, m, a.reference)
            print(f"| {src.name} | L = {r['scales'][0]}..{r['scales'][-1]} (m={m}, m0={r['m0']}) "
                  f"| {r['gamma_susc']:.4f} | {r['se']:.4f} | {r['half']:.4f} | {r['gap']:+.4f} "
                  f"| {r['predicted']:+.4f} |")
    if a.run:
        fit = json.loads(artifact_path(a.data_root / a.run, "answer").read_text())["fit"]
        print(f"\nThe run's own eq. (232) refit: omega_L = {fit['omega1']:.3f}, a1 = {fit['a1']:.2f} "
              f"(the pilot's: {w.value:.3f}, {L.a1:.2f}).")

    # No fit here: three-rung windows slid up the ladder. If the correction is a geometric
    # decay, the step between successive windows shrinks by rho**(-omega_L) each time.
    print("\n## Three-rung windows sliding up the ladder (no fit involved)\n")
    print("| draw | window | gamma_susc | step from the window below | ratio | implied omega_L |")
    print("|---|---|---|---|---|---|")
    for src in sources:
        prev, prev_step = None, None
        for start in range(len(src.scales) - 2):
            r = window_row(L, src, 3, a.reference, start=start)
            step = None if prev is None else r["gamma_susc"] - prev
            ratio = None if prev_step in (None, 0) or step is None else step / prev_step
            print(f"| {src.name} | L = {r['scales'][0]}..{r['scales'][-1]} | {r['gamma_susc']:.4f} | "
                  f"{'' if step is None else f'{step:+.4f}'} | "
                  f"{'' if ratio is None else f'{ratio:.3f}'} | "
                  f"{'' if not ratio or ratio <= 0 else f'{-math.log(ratio, L.rho):.2f}'} |")
            prev, prev_step = r["gamma_susc"], step

    print(f"\n## Time on one GPU, R = {L.R}, error on the answer <= target (gamma_susc units)\n")
    print("| target | on gamma_L | time (omega_L +/- 1 se) | stat term only |")
    print("|---|---|---|---|")
    for label, t in (("sigma", a.sigma), ("sigma/10", a.sigma / 10), ("sigma/100", a.sigma / 100)):
        tL = t / L.nu
        stat, _ = L.stat_only_seconds(tL)
        print(f"| {label} = {t:g} | {tL:.3g} | **{L.describe(L.plan(tL))}**<br>"
              f"omega+1se: {L.describe(L.plan(tL, omega1=w.value + w.se))}<br>"
              f"omega-1se: {L.describe(L.plan(tL, omega1=max(1e-3, w.value - w.se)))} "
              f"| {fmt_time(stat)} |")

    if a.run:
        # The pilot and the run's own refit disagree on omega_L by 8 pilot se: a
        # one-term model fitted with different weights gives a different exponent,
        # which is misspecification, not noise. Price the same targets both ways.
        print(f"\nSame targets on the run's own refit constants (omega_L = {fit['omega1']:.3f}, "
              f"a1 = {fit['a1']:.2f}):\n")
        print("| target | time |")
        print("|---|---|")
        for label, t in (("sigma", a.sigma), ("sigma/10", a.sigma / 10), ("sigma/100", a.sigma / 100)):
            pl = L.plan(t / L.nu, omega1=fit["omega1"], a1=fit["a1"])
            print(f"| {label} = {t:g} | {L.describe(pl)} |")

    top = int(INT32_SITES ** (1 / L.params["dim"]))
    print(f"\n## Only ladders that fit an int32 label (top rung L <= {top}, i.e. 128), "
          f"for sigma = {a.sigma:g}\n")
    print("| m | ladder | bias floor (gamma_susc) at omega_L | time at omega_L | omega+1se | omega-1se |")
    print("|---|---|---|---|---|---|")
    floors = []
    for m in (2, 3, 4, 5):
        m0 = max_m0_for_scale(top, m, L.rho)
        floors.append(floor_susc(L, m, m0, w.value))
        cells = []
        for om in (w.value, w.value + w.se, max(1e-3, w.value - w.se)):
            pl = L.plan(a.sigma / L.nu, m=m, max_scale=top, omega1=om)
            cells.append(L.describe(pl).split(" (")[0] if pl.get("feasible")
                         else f"never (floor {floor_susc(L, m, m0, om):.2g})")
        print(f"| {m} | L = {int(L.rho ** (m0 + 1))}..{int(L.rho ** (m0 + m))} "
              f"| {floors[-1]:.4g} | " + " | ".join(cells) + " |")
    lo_t = a.sigma / 10
    print(f"\nThe lowest floor of any legal ladder is {min(floors):.3g}; sigma/10 = {lo_t:g} is "
          f"{'below' if lo_t < min(floors) else 'above'} it, so "
          f"{'no legal ladder reaches it' if lo_t < min(floors) else 'some legal ladder can'}.")

    print(f"\n## Sensitivity to the ladder width m (omega_L, no ceiling; each cell is the time "
          f"and the top rung)\n")
    print("| m | " + " | ".join(f"sigma/{k}" if k > 1 else "sigma" for k in (1, 10, 100)) + " |")
    print("|---|---|---|---|")
    for m in range(2, 8):
        cells = []
        for k in (1, 10, 100):
            pl = L.plan(a.sigma / k / L.nu, m=m)
            cells.append(f"{fmt_time(pl['seconds'] * L.R)}, L={pl['scales'][-1]}"
                         if pl.get("feasible") else "unreachable")
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
