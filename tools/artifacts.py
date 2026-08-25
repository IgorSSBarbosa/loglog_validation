"""One place that decides what a run directory's files are called.

The problem this solves. `result.json` was written by three different producers
-- measure_cost, allocation_experiment and verify_prediction -- with three
different schemas, and `results.json` by a fourth. Thirteen files on disk shared
the first name. Telling them apart meant duck-typing on a schema key
(`allocation_table.measured_cost_exponent` globbed result.json and kept only
files with an "affine" key) or hardcoding a directory name
(`measured_throughput` looked in `allocation/` specifically). Both work until
they don't: allocation_sweep and prediction_check BOTH carry a "cells" key, so
schema alone cannot separate them, and a renamed folder breaks the other.

The rule: a file is named for WHAT IT IS, not for what wrote it. `omega1.json`
was already the good pattern and is unchanged. Naming by producer instead --
`metadata_generate.json` -- couples data to a script name, and scripts move:
`src/generate.py` became `src/generate/generate.py` on 2026-08-25 while the data
it wrote in March stayed put. Data outlives code.

Provenance still matters, so `write_artifact` stamps it INSIDE the file
(`produced_by`, `created`), where it can carry the script, the recipe and the
time rather than just a name.

Legacy files are still readable: `read_artifact` falls back to the old name and
says so. `python3 tools/artifacts.py --migrate <root>` renames them in place.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

#: kind -> filename. The single source of truth; nothing else hardcodes a name.
ARTIFACTS: dict[str, str] = {
    "samples_meta":     "samples_meta.json",      # was metadata.json
    "cost_probe":       "cost_probe.json",        # was result.json
    "allocation_sweep": "allocation_sweep.json",  # was result.json
    "prediction_check": "prediction_check.json",  # was result.json
    "gamma_estimates":  "gamma_estimates.json",   # was results.json
    "omega1":           "omega1.json",            # unchanged, already correct
    "coverage":         "coverage.json",          # check_coverage --json
}

#: old filename -> the kinds that ever used it, most specific first.
LEGACY: dict[str, tuple[str, ...]] = {
    "metadata.json": ("samples_meta",),
    "results.json":  ("gamma_estimates",),
    "result.json":   ("cost_probe", "prediction_check", "allocation_sweep"),
}


def _looks_like(kind: str, payload: dict) -> bool:
    """Schema signature, used ONLY to classify legacy files during migration.

    Deliberately not used at read time: once a file has its proper name, the
    name is the answer. These predicates exist because three producers shared
    `result.json`, and two of them (allocation_sweep, prediction_check) both
    carry "cells" -- so the discriminator has to be a key only one of them has.
    """
    keys = set(payload)
    if kind == "cost_probe":
        # both the current affine-fit format and the older d_hat-only one
        return "affine" in keys or {"elapsed_min", "d_hat"} <= keys
    if kind == "prediction_check":
        return "cells" in keys and "throughput" in keys
    if kind == "allocation_sweep":
        return "cells" in keys and "budgets" in keys
    if kind == "gamma_estimates":
        return "all_points" in keys or "two_point" in keys
    if kind == "samples_meta":
        return "scales" in keys and "n" in keys and "cells" not in keys
    if kind == "omega1":
        return "direct_fit" in keys
    return False


def classify(path) -> str | None:
    """Which kind a legacy file holds, or None if it cannot be told."""
    path = Path(path)
    candidates = LEGACY.get(path.name)
    if candidates is None:
        for kind, name in ARTIFACTS.items():
            if path.name == name:
                return kind
        return None
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if len(candidates) == 1:
        return candidates[0]
    for kind in candidates:
        if _looks_like(kind, payload):
            return kind
    return None


def artifact_path(run_dir, kind: str) -> Path:
    """Where `kind` lives in `run_dir`. Raises on an unknown kind."""
    if kind not in ARTIFACTS:
        raise ValueError(f"unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}")
    return Path(run_dir) / ARTIFACTS[kind]


def write_artifact(run_dir, kind: str, payload: dict, *,
                   produced_by: str | None = None, recipe=None) -> Path:
    """Write `payload` under the canonical name, stamping provenance inside it.

    `produced_by` should be the script's path relative to the repo root. It
    goes in the file rather than the filename precisely so that renaming the
    script later cannot make old data misleading.
    """
    path = artifact_path(run_dir, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("artifact", kind)
    body.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    if produced_by:
        body.setdefault("produced_by", produced_by)
    if recipe is not None:
        body.setdefault("recipe", str(recipe))
    path.write_text(json.dumps(body, indent=2, sort_keys=True))
    return path


def read_artifact(run_dir, kind: str, *, required: bool = True) -> dict | None:
    """Read `kind`, falling back to its legacy name with a warning.

    The warning is the point: a silent fallback would let a half-migrated
    `data/` look healthy right up until two producers' files collide again.
    """
    path = artifact_path(run_dir, kind)
    if path.exists():
        return json.loads(path.read_text())

    for old, kinds in LEGACY.items():
        if kind not in kinds:
            continue
        legacy = Path(run_dir) / old
        if not legacy.exists():
            continue
        if len(kinds) > 1 and classify(legacy) != kind:
            continue          # a different producer's file wearing the old name
        warnings.warn(
            f"{legacy} uses the legacy name {old!r}; expected "
            f"{ARTIFACTS[kind]!r}. Run: python3 tools/artifacts.py --migrate "
            f"<data root>", DeprecationWarning, stacklevel=2)
        return json.loads(legacy.read_text())

    if required:
        raise FileNotFoundError(
            f"no {kind} artifact in {run_dir} (looked for {ARTIFACTS[kind]!r}"
            + (f" and legacy {[k for k, v in LEGACY.items() if kind in v]}"
               if any(kind in v for v in LEGACY.values()) else "") + ")")
    return None


def find_artifacts(root, kind: str) -> list[Path]:
    """Every `kind` under `root`, by exact name -- no globbing plus guessing.

    This replaces `root.rglob("result.json")` followed by a schema test, which
    could not distinguish allocation_sweep from prediction_check at all.
    """
    root = Path(root)
    found = sorted(root.rglob(ARTIFACTS[kind]))
    if found:
        return found
    return [p for old, kinds in LEGACY.items() if kind in kinds
            for p in sorted(root.rglob(old)) if classify(p) == kind]


def migrate(root, *, dry_run: bool = False) -> list[tuple[Path, Path, str]]:
    """Rename every legacy artifact under `root` to its canonical name.

    Returns (old, new, status) per file. Never overwrites: a collision is
    reported and skipped, since two files claiming one name is the exact
    problem this module exists to end.
    """
    out = []
    root = Path(root)
    for old_name in LEGACY:
        for path in sorted(root.rglob(old_name)):
            kind = classify(path)
            if kind is None:
                out.append((path, path, "SKIP unrecognized schema"))
                continue
            target = path.with_name(ARTIFACTS[kind])
            if target.exists():
                out.append((path, target, "SKIP target exists"))
                continue
            if not dry_run:
                path.rename(target)
            out.append((path, target, "renamed" if not dry_run else "would rename"))
    return out


def _main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--migrate", type=Path, metavar="ROOT",
                   help="rename legacy artifacts under ROOT")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="show the naming table")
    a = p.parse_args(argv)

    if a.list or not a.migrate:
        print(f"  {'kind':<18} {'filename':<26} was")
        was = {k: [o for o, ks in LEGACY.items() if k in ks] for k in ARTIFACTS}
        for kind, name in ARTIFACTS.items():
            print(f"  {kind:<18} {name:<26} {', '.join(was[kind]) or '(new)'}")
        if not a.migrate:
            return

    rows = migrate(a.migrate, dry_run=a.dry_run)
    print(f"\n{len(rows)} legacy file(s) under {a.migrate}:")
    for old, new, status in rows:
        arrow = "->" if old != new else "  "
        print(f"  {status:<24} {old} {arrow} {new.name if old != new else ''}")


if __name__ == "__main__":
    _main()
