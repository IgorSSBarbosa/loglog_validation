"""Generic log-log plot of sample means vs scale.

Per PLAN.md ("tools/ may not import from experiments/"), this module only
knows about `{scale: samples}` dicts -- plain Python/NumPy, nothing about
where the samples came from. Any experiment (synthetic, SRW, percolation,
...) can hand it a dict shaped like what `generator.py`'s `generate()`
returns and get the same plot.

The article's log-log plot technique (Section 2, eq. 232 and around line 275)
plots E[Y_i] (or its empirical estimate Y_bar_i) against i on log-log axes,
expecting an approximately straight line of slope gamma once finite-size
effects are small. This draws exactly that: one point per scale, at
(i, Y_bar_i), with +-1 standard error bars, on log-log axes.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def loglog_points(samples: Mapping[int, Sequence[float]]):
    """Reduce {i: samples} to sorted arrays (scales, y_bar, se).

    se is the standard error of the mean, std(samples[i], ddof=1) / sqrt(n_i).
    """
    scales = np.array(sorted(samples))
    y_bar = np.array([np.mean(samples[i]) for i in scales], dtype=np.float64)
    se = np.array(
        [np.std(samples[i], ddof=1) / np.sqrt(len(samples[i])) for i in scales],
        dtype=np.float64,
    )
    return scales, y_bar, se


def loglog_plot(
    samples: Mapping[int, Sequence[float]],
    *,
    ax=None,
    target_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    label: str | None = None,
):
    """Plot Y_bar_i vs i on log-log axes, with +-1 SE error bars.

    Parameters
    ----------
    samples : {i: array of n_i i.i.d. draws of Y_i}
    ax : matplotlib Axes, optional. A new figure/axes is created if omitted.
    target_fn : optional callable i (array) -> E[Y_i], overlaid as a dashed
        reference curve when ground truth is known (e.g. synthetic data).
    label : optional legend label for the sample-mean series.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt  # deferred: keep this module importable without matplotlib

    if ax is None:
        _, ax = plt.subplots()

    scales, y_bar, se = loglog_points(samples)

    ax.errorbar(scales, y_bar, yerr=se, fmt="o", capsize=3, label=label or r"sample mean $\pm$ 1 SE")

    if target_fn is not None:
        fine = np.geomspace(scales.min(), scales.max(), 200)
        ax.plot(fine, target_fn(fine), "--", color="gray", label=r"$\mathbb{E}\,Y_i$ (known)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$i$")
    ax.set_ylabel(r"$\overline{Y}_i$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    return ax
