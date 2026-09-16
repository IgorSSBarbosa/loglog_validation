"""Saying what the program is doing, cheaply.

WHY THIS EXISTS (Igor, 2026-09-14). A study spends minutes inside three
phases that print nothing at all -- the cost probe, the per-replicate
correction fits, and the report -- so the console sits blank and the only
honest reading from outside is "it may have crashed". The fix asked for is
not a profiler: it is a print. "running something ... please wait, your
computer did not crash."

WHAT IT MAY COST. Nothing measurable. Every call here is one `print` to
stderr at a phase boundary, and a phase boundary is by construction a place
where the program is about to spend seconds. There is no per-sample, per-row
or per-call instrumentation in this module, and none may be added: timing
that runs inside the hot loop is a separate concern and lives outside the
shipped tree (local/profiling/), precisely so it cannot slow a real study
down.

TWO LEVELS.
  `say`/`phase`   always print. They are the "not crashed" signal, one line
                  per phase, and are not opt-in: a silent run is the bug.
  `detail`        prints only under --verbose. Per-rung probe timings,
                  per-scale draw sizes -- useful when a phase is behaving
                  oddly, noise when it is not.

STDERR, not stdout, everywhere. The reports these drivers print are data a
user pipes to a file; progress chatter is not, and mixing them would put
carriage returns in the middle of a results table.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

#: --verbose. Set once by a CLI via `set_verbose`; read by `detail`.
VERBOSE = False

#: True while a `phase` line is open (printed without its newline), so the
#: next thing to write can close it first rather than landing mid-line.
_OPEN = [False]


def set_verbose(flag: bool) -> None:
    global VERBOSE
    VERBOSE = bool(flag)


def _close_line() -> None:
    if _OPEN[0]:
        print(file=sys.stderr)
        _OPEN[0] = False


def say(msg: str = "") -> None:
    """One line of progress. Always printed."""
    _close_line()
    print(msg, file=sys.stderr, flush=True)


def detail(msg: str) -> None:
    """One line of progress, only under --verbose."""
    if VERBOSE:
        say(msg)


@contextmanager
def phase(msg: str, *, verbose_only: bool = False):
    """Announce a phase, then report how long it took.

        with phase("measuring the cost exponent d"):
            ...
        -> "  measuring the cost exponent d ... 12.4s"

    The opening half is written WITHOUT its newline so that a quick phase is
    one line rather than two. Anything printed in between closes that line
    first (`_close_line`), so nested output never lands mid-sentence -- the
    timing then arrives on its own line, correctly attributed.
    """
    if verbose_only and not VERBOSE:
        yield
        return
    _close_line()
    print(f"  {msg} ... ", end="", file=sys.stderr, flush=True)
    _OPEN[0] = True
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if _OPEN[0]:
            print(f"{dt:.1f}s", file=sys.stderr, flush=True)
            _OPEN[0] = False
        else:
            print(f"  ({msg}: {dt:.1f}s)", file=sys.stderr, flush=True)


class _Null:
    """What `bar` returns when there is nothing to draw a bar on.

    A no-op with tqdm's surface, so callers never branch on whether a bar
    exists. `write` still goes to stderr: a message the caller wanted printed
    is not progress decoration, and losing it because tqdm is absent would be
    the opposite of this module's point.
    """

    def update(self, _n=1): pass
    def set_postfix_str(self, _s, refresh=True): pass
    def set_description(self, _s, refresh=True): pass
    def write(self, msg): print(msg, file=sys.stderr)
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def bar(total, desc: str, *, unit: str = "step", enabled: bool = True,
        unit_scale: bool = True, leave: bool = True):
    """A tqdm over `total` units, or a silent stand-in.

    Disabled off a TTY -- a piped or logged run would otherwise fill with
    carriage returns -- and degraded to `_Null` if tqdm is not installed, so a
    missing optional dependency never costs someone a two-hour draw.
    """
    if not enabled or not total or not sys.stderr.isatty():
        return _Null()
    try:
        from tqdm import tqdm
    except ImportError:
        return _Null()
    _close_line()
    return tqdm(total=total, unit=unit, unit_scale=unit_scale, leave=leave,
                dynamic_ncols=True, file=sys.stderr, desc=desc,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")


#: Seconds above which a step is worth a line of its own (Igor, 2026-09-14:
#: "profile what is taking more than 2s"). Below it a step is noise: a
#: 10-rung synthetic ladder finishes every rung in under a millisecond, and
#: announcing each one buries the phases that actually made the user wait.
ANNOUNCE_SECONDS = 2.0


class ScaleNarrator:
    """Narrates a ladder of draws, saying only what was worth waiting for.

    THE PROBLEM IT SOLVES. A rung is one atomic `spec.simulate` call, so from
    outside it is invisible until it returns -- and the top rung of a ladder is
    routinely most of the run (measured: 51% of a 248 s srw study was the
    single i=8192 call). Announcing rungs only on COMPLETION therefore prints
    the explanation after the silence it was meant to explain. Announcing all
    of them on ENTRY fixes that and creates the opposite problem: on synthetic,
    where every rung is sub-millisecond, three replicates over six doubling
    rounds is 180 lines saying nothing happened.

    THE RULE. A rung announces itself on the way in only when it is PREDICTED
    to cost more than `ANNOUNCE_SECONDS`, and reports on the way out only when
    it actually did. The prediction is the pace measured on the rungs already
    drawn in this ladder -- seconds per unit of the model's own `cost_hint`,
    which is exact by construction for every model that declares one. Ladders
    ascend, so by the time an expensive rung arrives there is always pace data
    from the cheap ones below it; before any exists nothing is claimed.

    No clock of its own beyond the two `perf_counter` calls the caller already
    makes, and no state per sample -- one float per rung.
    """

    def __init__(self, scales, counts, cost_hint=None, params=None,
                 prefix: str = "    ", threshold: float = ANNOUNCE_SECONDS,
                 predicted_seconds: float | None = None):
        self.prefix, self.threshold = prefix, threshold
        self.total = len(scales)
        #: What the PLAN says this ladder costs, when a plan exists. Seeds the
        #: pace before anything has been drawn, so the first rung -- the one
        #: nothing can have learned a pace from -- is not the one rung that
        #: stays silent. The run phase has this; a pilot round does not.
        self.predicted_seconds = predicted_seconds
        self.work = {}
        if cost_hint is not None:
            for i, c in zip(scales, counts):
                try:
                    self.work[int(i)] = float(c) * float(cost_hint(int(i),
                                                                   params or {}))
                except Exception:          # a hint that refuses a scale is not
                    self.work[int(i)] = 0.0   # a reason to lose the progress line
        self._done_work = 0.0
        self._done_secs = 0.0

    def _predict(self, i) -> float | None:
        w = self.work.get(int(i), 0.0)
        if not w:
            return None
        if self._done_work > 0 and self._done_secs > 0:
            return w * (self._done_secs / self._done_work)     # measured pace
        total_work = sum(self.work.values())
        if self.predicted_seconds and total_work > 0:
            return w * self.predicted_seconds / total_work     # the plan's
        return None

    def start(self, i, n_i, idx, _total=None) -> None:
        eta = self._predict(i)
        line = (f"{self.prefix}[{idx}/{self.total}] drawing scale "
                f"i={int(i)}, n={int(n_i):,}")
        if eta is not None and eta >= self.threshold:
            say(f"{line} -- about {eta:.0f}s, please wait")
        elif eta is None and idx == 1:
            # Nothing has been timed and no plan says what this costs, so the
            # honest message is that the wait is unknown -- which is still
            # better than the silence it replaces. Only the FIRST rung gets
            # this; from the second on there is always a measured pace.
            say(f"{line} (first rung -- no timing yet, please wait)")
        else:
            detail(line + (f" (~{eta:.1f}s)" if eta is not None else ""))

    def done(self, i, n_i, secs) -> None:
        self._done_work += self.work.get(int(i), 0.0)
        self._done_secs += float(secs)
        msg = f"{self.prefix}    scale i={int(i)} done in {secs:.1f}s"
        say(msg) if secs >= self.threshold else detail(msg)
