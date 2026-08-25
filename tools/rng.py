"""Seeding, and the one way to record a seed so a run can be regenerated.

Ground rule 2 says every (experiment, configuration, replicate) draws fresh,
independent randomness. numpy's answer is `SeedSequence.spawn`, and this module
exists because a spawned stream cannot be round-tripped through the plain
integer seed that every driver's API used to take.

The trap this module exists to close
------------------------------------
A spawned child does NOT carry its own entropy. It carries its PARENT's:

    ss   = SeedSequence(12345)
    kid  = ss.spawn(1)[0]
    kid.entropy == ss.entropy          # True
    kid.spawn_key                      # (0,)   <- the only distinguishing part

So the obvious way to pass a child through an `int`-typed seed parameter --
hand over `kid.entropy` -- silently collapses every replicate onto the SAME
stream. That is exactly the pool-reuse bug in `presentation18-05-2026`
(PLAN.md ground rule 2) with a different disguise: it produces plausible
numbers, no error, and comparisons between configurations that are quietly
invalid because they share randomness.

`SeedSequence(kid)` does not work either -- it raises TypeError, because the
first positional parameter is `entropy`, which must be ints.

Hence: seeds travel as `SeedSequence` objects, and are RECORDED via
`seed_record`, which keeps both halves.

Backward compatibility
----------------------
`seed_record` returns a bare int whenever `spawn_key` is empty, so an
un-spawned run's metadata is byte-identical to what this repo has always
written, and every `samples_meta.json` on disk still loads. Only a spawned
seed produces the richer `{"entropy": ..., "spawn_key": [...]}` form.
"""

from __future__ import annotations

import numpy as np

SeedSequence = np.random.SeedSequence


def as_seed_sequence(seed=None) -> np.random.SeedSequence:
    """Coerce any accepted seed spelling to a SeedSequence.

    Accepts: None (fresh OS entropy), an int, a SeedSequence (returned
    unchanged, so a spawned stream keeps its spawn_key), or the dict produced
    by `seed_record`.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, dict):
        try:
            entropy, spawn_key = seed["entropy"], seed.get("spawn_key", ())
        except KeyError:
            raise ValueError(
                f"a dict seed must carry 'entropy' (and optionally 'spawn_key'); "
                f"got keys {sorted(seed)}") from None
        return np.random.SeedSequence(entropy, spawn_key=tuple(spawn_key))
    return np.random.SeedSequence(seed)


def seed_record(seed) -> int | dict:
    """A JSON-safe record of `seed`, complete enough to rebuild the stream.

    A bare int for an un-spawned seed (what this repo has always written), and
    `{"entropy": ..., "spawn_key": [...]}` for a spawned one -- because for a
    spawned stream the entropy alone is the parent's and would rebuild the
    WRONG stream, identically for every sibling. Round-trips through
    `as_seed_sequence`.
    """
    ss = as_seed_sequence(seed)
    if not ss.spawn_key:
        return ss.entropy
    return {"entropy": ss.entropy, "spawn_key": list(ss.spawn_key)}


def spawn(seed, n: int) -> list[np.random.SeedSequence]:
    """`n` independent child streams of `seed` -- ground rule 2's primitive.

    Thin wrapper over SeedSequence.spawn; it exists so call sites name this
    module rather than reaching for numpy directly, and so `seed` may be any
    spelling `as_seed_sequence` accepts.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return as_seed_sequence(seed).spawn(n)
