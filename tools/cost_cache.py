"""The cost probe's answer, kept per (model, params, machine).

WHY THIS IS A SEPARATE FILE FROM A STUDY

`omega1`, `a1` and `cv` are properties of the model and the LADDER, and they
are measured from draws -- so they belong to the study that drew them. `d` and
`throughput` are not: they are properties of the model and the MACHINE, the
probe that measures `d` is deliberately fixed-seeded (COST_PROBE_SEED, so that
re-running a pilot cannot move `d` for reasons unrelated to the machine), and
`throughput` is a rate this computer sustains. Two studies of the same model on
the same machine ask the clock the same question and get the same answer.

So this cache is what survives a change of LADDER, which is exactly what
`pilot.py --reuse-pilot` cannot carry: that flag requires the scales to match,
because the constants it carries were fitted on them. Re-plan the same model
over a different ladder and the statistical constants must be re-measured while
the machine ones need not be.

Reuse is OPT-IN (`--reuse-cost`), never automatic. A cached `d` that is silently
out of date makes every wall-clock prediction downstream wrong in a way nothing
in the study can detect -- the same class of silent substitution
tools/constants.py exists to prevent -- so the flag is the user saying "this
machine has not changed", and the age of the entry is printed wherever the
constant it produced appears.

Writing is NOT opt-in: a fresh probe always lands here, so `--reuse-cost` has
something to find the next time. The file lives under `local/`, which is
gitignored: these numbers describe one computer and are not part of the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Machine-local, gitignored (see .gitignore's `local/`). Not an artifact in
#: tools/artifacts.py's sense: it is not produced BY a run and is not part of
#: any study's record -- it is this computer's own notebook.
CACHE_PATH = ROOT / "local" / "cost_cache.json"

#: Environment override for that path. Exists because writing is NOT opt-in:
#: every pilot saves its probe here, so any process that runs a pilot touches
#: state outside its own output directory -- and calibration/exercise_all.py
#: runs several, from a throwaway scratch tree it is the whole point of that
#: file never to escape. The audit points this at its scratch directory; a user
#: with two checkouts sharing a machine can point it at one shared file.
CACHE_ENV = "LOGLOG_COST_CACHE"


def cache_path() -> Path:
    """Where the cache lives for this process."""
    return Path(os.environ.get(CACHE_ENV) or CACHE_PATH)

#: Age past which `describe` calls an entry stale. A month is long enough that
#: an ordinary week of work reuses freely, and short enough that a machine
#: rebuilt, a numpy upgraded or a laptop that has started thermal-throttling
#: does not go unnoticed for a year. A note, never a refusal: the user asked
#: for the reuse, and the staleness they are being warned about is the thing
#: they asserted was fine.
STALE_DAYS = 30.0


def machine_id() -> str:
    """What counts as "the same machine" for a timing measurement."""
    return f"{platform.node()}/{platform.machine()}/py{platform.python_version()}"


def key(model: str, params: dict | None = None) -> str:
    """One entry per (model, params). Params are canonicalised, not stringified.

    `{"dim": 3, "anchor": "slab"}` and `{"anchor": "slab", "dim": 3}` are the
    same process and must hit the same entry, so the digest is taken over
    sort_keys JSON rather than over whatever order the recipe happened to use.
    """
    blob = json.dumps(params or {}, sort_keys=True, default=str)
    return f"{model}:{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # A corrupt cache is a missing cache: it holds nothing that cannot be
        # re-measured in seconds, and refusing to run over it would be absurd.
        return {}


def load(model: str, params: dict | None = None, *, path: Path | None = None,
         machine: str | None = None) -> dict | None:
    """The cached entry for this (model, params, machine), or None.

    Keyed by machine as well as by model: the same repo on a laptop and on a
    workstation holds two entries, and neither can be handed the other's `d`.
    """
    entry = _read(Path(path) if path else cache_path()).get(key(model, params))
    if entry is None:
        return None
    return entry if entry.get("machine") == (machine or machine_id()) else None


def save(model: str, params: dict | None, probe: dict, *,
         throughput: float | None = None, path: Path | None = None,
         machine: str | None = None) -> Path:
    """Record a fresh probe. Overwrites the entry for this key and machine."""
    p = Path(path) if path else cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cache = _read(p)
    cache[key(model, params)] = {
        "model": model,
        "params": params or {},
        "machine": machine or machine_id(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe": probe,
        "throughput": None if throughput is None else float(throughput),
    }
    p.write_text(json.dumps(cache, indent=2, sort_keys=True, default=str))
    return p


def age_days(entry: dict) -> float | None:
    """How old the entry is, or None if its timestamp is unreadable."""
    try:
        t = datetime.fromisoformat(str(entry["created"]))
    except (KeyError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0


def describe(entry: dict) -> str:
    """One line naming when and where it was measured -- for a provenance string."""
    age = age_days(entry)
    when = str(entry.get("created", "?"))[:10]
    host = str(entry.get("machine", "?")).split("/")[0]
    stale = ", STALE" if age is not None and age > STALE_DAYS else ""
    old = f", {age:.0f} d old{stale}" if age is not None else ""
    try:
        where = cache_path().relative_to(ROOT)
    except ValueError:
        where = cache_path()
    return f"reused from {where} (measured {when} on {host}{old})"
