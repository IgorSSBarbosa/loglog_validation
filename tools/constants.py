"""Meta-constants with their errors and where they came from.

Every number that reaches an allocation decision -- d, omega1, a1, cv,
throughput -- is an ESTIMATE, and this module refuses to let one travel
anonymously.

The bug this closes. `src/budget/allocation_table.py` used to define

    FALLBACK_D      = 1.0                  # srw is Theta(k) by construction
    FALLBACK_CV     = sqrt(pi/2 - 1)       # half-normal limit for |S_k|
    FALLBACK_OMEGA1 = 1.0155

and, worse, `--d` and `--omega1` DEFAULTED to 1.0 with no marker at all. Two
of those are srw's exact truths, so on srw the table printed the right answer
whether or not anything had been measured -- and `omega1 = 1.0` appeared in
the input block looking exactly like a measurement sitting beside a real one.
Every published table read omega1 = 1 while Experiment B's own runs measured
0.907, 0.986, 1.198, 0.486... The good agreement was partly the default.

The rule now: a constant is MEASURED (with a value, an error where one exists,
and a provenance string naming the file it came from) or it is a deliberate
USER OVERRIDE (stamped as such) or it is ABSENT -- and absent is an error that
names the flag you can use to supply it, never a silent substitution.

`se` may legitimately be None (one replicate gives no spread); that is not the
same as absent, and `format_table` renders the two differently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

#: Filename inside a study directory. Named for what it holds, per artifacts.py.
CONSTANTS_FILE = "constants.json"

#: Which flag supplies each constant by hand, for the error message.
FLAGS = {"d": "--d", "omega1": "--omega1", "a1": "--a1",
         "cv": "--cv", "throughput": "--throughput"}

#: One line each, for the error message when a constant is missing.
HOW_TO_MEASURE = {
    "d":          "python3 src/estimate/measure_cost.py -meta <cost recipe>",
    "omega1":     "python3 src/estimate/estimate_omega1.py -data <run dir>",
    "a1":         "python3 src/estimate/estimate_omega1.py -data <run dir>",
    "cv":         "any run's samples.npz (generate.py) -- computed from them",
    "throughput": "python3 src/budget/allocation_experiment.py -meta <sweep recipe>",
}


@dataclass
class Constant:
    """One estimated constant. `se=None` means 'no spread available', not 'exact'."""

    value: float
    se: float | None = None
    source: str = ""          # human-readable provenance, e.g. "6 replicates, pooled"
    origin: str | None = None  # the file it was read from, repo-relative

    @property
    def is_override(self) -> bool:
        return self.source.startswith("user override")


def measured(value, se, source, origin=None) -> Constant:
    """A constant read off a real run."""
    return Constant(float(value), None if se is None else float(se),
                    source, str(origin) if origin else None)


def override(value, name: str) -> Constant:
    """A constant the user supplied by hand. Stamped, so it can never pass as measured."""
    flag = FLAGS.get(name, f"--{name}")
    return Constant(float(value), None, f"user override ({flag})", None)


def require(constants: dict, name: str) -> Constant:
    """Fetch `name` or exit with instructions. The whole point of this module.

    Deliberately a hard failure rather than a warning: a warning scrolls past,
    and the number it warns about goes on to size a run that may take hours.
    """
    c = constants.get(name)
    if c is not None:
        return c
    flag = FLAGS.get(name, f"--{name}")
    how = HOW_TO_MEASURE.get(name, "")
    raise SystemExit(
        f"\nno measured value for {name!r}, and there is no default.\n"
        f"  This used to fall back to a hardcoded constant, which on srw was the\n"
        f"  exact truth -- so the table looked right whether or not anything had\n"
        f"  been measured. It no longer does that.\n\n"
        f"  Measure it:  {how}\n"
        f"  Or state it: {flag} <value>   (recorded as a user override, not a measurement)\n")


def load(study_dir) -> dict[str, Constant]:
    """Read a study's constants.json. Missing file means no constants, not an error."""
    p = Path(study_dir) / CONSTANTS_FILE
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: Constant(**v) for k, v in raw.items() if isinstance(v, dict)}


def save(study_dir, constants: dict) -> Path:
    """Write constants.json, sorted so a diff between two studies is readable."""
    p = Path(study_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / CONSTANTS_FILE
    out.write_text(json.dumps({k: asdict(v) for k, v in sorted(constants.items())},
                              indent=2))
    return out


def format_value(c: Constant, width: int = 10) -> str:
    """`-0.2435 +/- 0.0432`, or `-0.2435` padded when no spread exists."""
    v = f"{c.value:+.4f}" if abs(c.value) < 1e4 else f"{c.value:.4g}"
    return f"{v:>{width}}" + (f" +/- {c.se:.4f}" if c.se is not None else " " * 12)


def format_table(constants: dict, order=("d", "omega1", "a1", "cv", "throughput")) -> str:
    """The input block: one line per constant, value, error, provenance.

    Overrides are marked `<-- NOT MEASURED` so a hand-supplied number can never
    be mistaken for one that came off a run.
    """
    names = [n for n in order if n in constants] + \
            [n for n in sorted(constants) if n not in order]
    lines = []
    for n in names:
        c = constants[n]
        mark = "   <-- NOT MEASURED" if c.is_override else ""
        lines.append(f"  {n:<11}{format_value(c)}  ({c.source}){mark}")
    return "\n".join(lines)
