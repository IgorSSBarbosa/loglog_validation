"""What a replicate of a study *is*, once its draws are thrown away.

Every step of the workflow (src/study/) draws the same thing -- one replicate
over a ladder of scales -- and keeps the same three numbers per scale:

    y_bar      the sample mean of Y_i, the quantity eq. (232) models
    sigma_log  its standard error ON THE LOG SCALE, sd/(sqrt(n)*y_bar),
               which is what the log-log fit weights by
    cv         the coefficient of variation sd/y_bar, which is what an
               allocation needs (tools/allocation.py) and what the next
               pilot's `cv` constant is measured from

Those three are the entire interface between drawing and fitting: nothing
downstream -- correction.fit_correction, report.analyse, plan.py -- ever looks
at a raw draw. That is what makes it safe for run.py to summarize on the fly
and never write the samples, and it is why this definition lives in one place
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
#: {scale: triple} mapping into one list per name, which is the shape the
#: pilot/final artifacts store and the fitters read.
SUMMARY_FIELDS = ("y_bar", "sigma_log", "cv")


def summarize_scale(draws) -> tuple[float, float, float]:
    """(y_bar, sigma_log, cv) for one scale's draws, in a single pass.

    `ddof=1` because these are estimates from a sample, not a population.
    """
    s = np.asarray(draws, dtype=float)
    mean = float(s.mean())
    sd = float(s.std(ddof=1))
    return mean, float(sd / (np.sqrt(s.size) * mean)), float(sd / mean)


def replicate_summary(stats: dict, scales: Sequence[int]) -> dict:
    """{scale: (y_bar, sigma_log, cv)} -> {"y_bar": [...], "sigma_log": [...], ...}.

    Column-major, ordered by `scales` rather than by the mapping's own key
    order, so a replicate lines up positionally with the ladder it was drawn
    on however the generator happened to iterate.
    """
    return {name: [float(stats[i][j]) for i in scales]
            for j, name in enumerate(SUMMARY_FIELDS)}
