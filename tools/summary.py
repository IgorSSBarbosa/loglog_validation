"""What a replicate of a study *is*, once its draws are thrown away.

Every step of the workflow (src/study/) draws the same thing -- one replicate
over a ladder of scales -- and keeps the same four numbers per scale:

    y_bar      the sample mean of Y_i, the quantity eq. (232) models
    sigma_log  its standard error ON THE LOG SCALE, sd/(sqrt(n)*y_bar),
               which is what the log-log fit weights by
    cv         the coefficient of variation sd/y_bar, which is what an
               allocation needs (tools/allocation.py) and what the next
               pilot's `cv` constant is measured from
    log_moment E|log xi|^(2+delta) with xi = Y_i/E Y_i. NO LONGER USED:
               it fed the Lambda of the old Wilson bound's B_bad term, and
               the 2026-11-04 article revision dropped that assumption and
               term (user). Still computed so every stored summary keeps one
               schema; nothing downstream reads it.
    abs_moment E|xi|^(2+delta), the per-scale ingredient of the moment
               assumption (A4). Here because it is an expectation over
               INDIVIDUAL draws, not a function of the sample mean, so it
               cannot be recovered afterwards from a summary (user, 2026-10-28).
    n_nonpos   how many draws were <= 0, the positivity check (A2). Those
               draws are the ones log_moment drops.

Those four are the entire interface between drawing and fitting: nothing
downstream -- correction.fit_correction, report.analyse, plan.py -- ever looks
at a raw draw. That is what makes it safe for run.py to summarize on the fly
and never write the samples, and it is why `abs_moment` had to be added here
rather than computed later: the first three are functions of the sample mean
and could be recovered from stored summaries, but a moment of xi is an
expectation over individual draws and is gone the moment they are freed. And it
is why this definition lives in one place
instead of being retyped in pilot.py and run.py (where it was, verbatim, twice).

Deliberately free of any dependency on src/: a summary is a fact about an
array, not about how the array was obtained. Callers pass `summarize_scale`
straight to generate(..., reduce=) so the draws are collapsed and freed one
scale at a time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: The per-scale summary, in order. `replicate_summary` transposes a
#: {scale: tuple} mapping into one list per name, which is the shape the
#: pilot/final artifacts store and the fitters read.
SUMMARY_FIELDS = ("y_bar", "sigma_log", "cv", "log_moment", "abs_moment", "n_nonpos")

#: The delta of the (2+delta)-moments, Assumption 2_plus_delta_moments (A4),
#: the article's usual delta = 1. Fixed rather than exposed: a summary on disk
#: carries the number, not the exponent that produced it.
LOG_MOMENT_DELTA = 1.0


def summarize_scale(draws) -> tuple[float, float, float, float, float, int]:
    """SUMMARY_FIELDS for one scale's draws, in a single pass.

    `ddof=1` because these are estimates from a sample, not a population --
    which is also why a single draw is refused rather than summarized: two of
    the four numbers are spreads, and a spread of one point does not exist.
    An allocation that puts n=1 on a scale has starved it, and the log-log fit
    weights by sigma_log, so a NaN there silently poisons the whole fit.
    """
    s = np.asarray(draws, dtype=float)
    if s.size < 2:
        raise ValueError(
            f"a summary needs at least 2 draws; got {s.size}. The allocation "
            f"starved this scale -- raise the budget, drop the smallest scales, "
            f'or set "min_n": 2 in the recipe\'s "n".')
    mean = float(s.mean())
    if mean == 0.0:
        raise ValueError(
            f"all {s.size} draws at this scale are zero, so y_bar = 0 and the "
            f"log-log fit has nothing to take a logarithm of. Almost always the "
            f"same starved-scale problem: too few draws at too small a scale.")
    sd = float(s.std(ddof=1))

    # xi = Y/E Y, with the sample mean standing in for E Y -- the same
    # definition tools/wilson.py's `moment_bounds` uses. The zeros are dropped
    # from log_moment (|S_k| = 0 happens, and log 0 is not a number).
    xi = s / mean
    pos = xi > 0
    log_moment = (float(np.mean(np.abs(np.log(xi[pos])) ** (2 + LOG_MOMENT_DELTA)))
                  if pos.any() else float("nan"))
    abs_moment = float(np.mean(np.abs(xi) ** (2 + LOG_MOMENT_DELTA)))
    return (mean, float(sd / (np.sqrt(s.size) * mean)), float(sd / mean), log_moment,
            abs_moment, int(np.count_nonzero(~pos)))


def replicate_summary(stats: dict, scales: Sequence[int]) -> dict:
    """{scale: tuple} -> {"y_bar": [...], "sigma_log": [...], ...}, column-major.

    Column-major, ordered by `scales` rather than by the mapping's own key
    order, so a replicate lines up positionally with the ladder it was drawn
    on however the generator happened to iterate.
    """
    return {name: [(int if name == "n_nonpos" else float)(stats[i][j]) for i in scales]
            for j, name in enumerate(SUMMARY_FIELDS)}
