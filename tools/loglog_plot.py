"""Generic log-log plot of sample means vs scale.

Per PLAN.md ("tools/ may not import from experiments/"), this module only
knows about `{scale: samples}` dicts -- plain Python/NumPy, nothing about
where the samples came from. Any model (synthetic, SRW, percolation, ...
registered in tools/models.py) can hand it a dict shaped like what
src/generate.py's `generate()` returns and get the same plot.

The article's log-log plot technique (Section 2, eq. 232 and around line 275)
plots E[Y_i] (or its empirical estimate Y_bar_i) against i on log-log axes,
expecting an approximately straight line of slope gamma once finite-size
effects are small. This draws exactly that: one point per scale, at
(i, Y_bar_i), with +-1 standard error bars, on log-log axes.

`estimates_plot` is a second, separate chart for `tools.loglog.compare_methods`'s
output -- comparing the gamma-hat estimates themselves, not the raw data.

Colors follow the dataviz skill's validated default palette (categorical slots
1-3; see references/palette.md), status colors reserved for the mle estimator's
trustworthy/not state rather than spent as a 4th competing hue (the palette's
series-count ladder caps an all-pairs-visible chart -- points from two series
plus two reference lines, all visible at once -- at three hues).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

# Colors, dataviz skill's validated default palette (references/palette.md),
# light mode. Categorical slots 1-3, fixed order, not cycled. Status color
# reserved for mle's untrustworthy state, not spent as a 4th competing hue.
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_INK = "#0b0b0b"  # primary ink
_MUTED = "#898781"  # muted ink, for the true_gamma target reference
_GRIDLINE = "#e1e0d9"  # hairline gridline
_CRITICAL = "#d03b3b"  # status palette, reserved for mle's untrustworthy state


def loglog_points(samples: Mapping[int, Sequence[float]]):
    """Reduce {i: samples} to sorted arrays (scales, y_bar, se, n).

    se is the standard error of the mean, std(samples[i], ddof=1) / sqrt(n_i).
    n is the per-scale sample count -- needed by tools.loglog.gamma_mle, which
    (unlike the OLS-based estimators) uses it directly in the model's variance.
    """
    scales = np.array(sorted(samples))
    y_bar = np.array([np.mean(samples[i]) for i in scales], dtype=np.float64)
    n = np.array([len(samples[i]) for i in scales], dtype=np.int64)
    se = np.array(
        [np.std(samples[i], ddof=1) / np.sqrt(len(samples[i])) for i in scales],
        dtype=np.float64,
    )
    return scales, y_bar, se, n


def loglog_plot(
    samples: Mapping[int, Sequence[float]],
    *,
    ax=None,
    target_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    fit_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    fit_label: str | None = None,
    label: str | None = None,
):
    """Plot Y_bar_i vs i on log-log axes, with +-1 SE error bars.

    Parameters
    ----------
    samples : {i: array of n_i i.i.d. draws of Y_i}
    ax : matplotlib Axes, optional. A new figure/axes is created if omitted.
    target_fn : optional callable i (array) -> E[Y_i], overlaid as a dashed
        gray reference curve when ground truth is known (e.g. synthetic data).
    fit_fn : optional callable i (array) -> fitted value, overlaid as a solid
        colored line -- e.g. an OLS fit computed from this same data, distinct
        from target_fn (a known truth) even when both are present at once.
    fit_label : legend label for fit_fn; ignored if fit_fn is None.
    label : optional legend label for the sample-mean series.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt  # deferred: keep this module importable without matplotlib

    if ax is None:
        _, ax = plt.subplots()

    scales, y_bar, se, _n = loglog_points(samples)

    ax.errorbar(scales, y_bar, yerr=se, fmt="o", capsize=3, label=label or r"sample mean $\pm$ 1 SE")

    if target_fn is not None:
        fine = np.geomspace(scales.min(), scales.max(), 200)
        ax.plot(fine, target_fn(fine), "--", color="gray", label=r"$\mathbb{E}\,Y_i$ (known)")

    if fit_fn is not None:
        fine = np.geomspace(scales.min(), scales.max(), 200)
        ax.plot(fine, fit_fn(fine), "-", color=_ORANGE, linewidth=2, label=fit_label or "OLS fit")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$i$")
    ax.set_ylabel(r"$\overline{Y}_i$")
    ax.grid(True, which="both", ls="-", lw=0.6, color=_GRIDLINE, alpha=0.8)
    ax.legend()
    return ax


def estimates_plot(results: dict, *, ax=None):
    """Plot the gamma-hat estimates from `tools.loglog.compare_methods`'s output.

    `two_point` and `drop_leading` each produce a *sequence* of estimates, one
    per window -- plotted against the smallest scale in each window (log x-axis),
    so the chart shows directly whether the estimate is converging as small,
    more finite-size-biased scales are dropped. `all_points` and `mle` are
    single estimates with no window to vary over, so they're drawn as
    horizontal reference lines across the full width instead of forced onto
    the same x-axis dishonestly. `mle` is drawn in primary ink when its
    `trustworthy` diagnostic is True, and in the status-critical color
    (with an explicit warning in its legend label -- never color alone) when
    it's False. `true_gamma`, if present in `results`, is a muted dashed
    reference line.

    Parameters
    ----------
    results : dict, as returned by `tools.loglog.compare_methods`.
    ax : matplotlib Axes, optional. A new figure/axes is created if omitted.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt  # deferred: keep this module importable without matplotlib

    if ax is None:
        _, ax = plt.subplots()

    m = results["methods"]
    scales = results["scales"]
    x_min, x_max = min(scales), max(scales)

    two_point = m["two_point"]["estimates"]
    if two_point:
        xs = [e["scales"][0] for e in two_point]
        ys = [e["gamma_hat"] for e in two_point]
        ax.plot(
            xs, ys, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            linewidth=2, color=_BLUE, label="two_point ($m=2$)",
        )

    drop_leading = m["drop_leading"]["estimates"]
    if drop_leading:
        xs = [e["scales_used"][0] for e in drop_leading]
        ys = [e["gamma_hat"] for e in drop_leading]
        ax.plot(
            xs, ys, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            linewidth=2, color=_ORANGE, label="drop_leading",
        )

    all_points_gamma = m["all_points"]["gamma_hat"]
    ax.hlines(all_points_gamma, x_min, x_max, color=_AQUA, linewidth=2, label=f"all_points ({all_points_gamma:.4f})")

    mle = m["mle"]
    if mle["trustworthy"]:
        ax.hlines(mle["gamma_hat"], x_min, x_max, color=_INK, linewidth=2, label=f"mle ({mle['gamma_hat']:.4f})")
    else:
        ax.hlines(
            mle["gamma_hat"], x_min, x_max, color=_CRITICAL, linewidth=2, linestyle="--",
            label=f"mle ({mle['gamma_hat']:.4f})  ⚠ not trustworthy",
        )

    true_gamma = results.get("true_gamma")
    if true_gamma is not None:
        ax.axhline(true_gamma, color=_MUTED, linewidth=1.5, linestyle="--", label=rf"true $\gamma$ ({true_gamma:.4f})")

    ax.set_xscale("log")
    ax.set_xlabel("smallest scale $i$ used in window")
    ax.set_ylabel(r"$\hat\gamma$")
    ax.grid(True, which="both", ls="-", lw=0.6, color=_GRIDLINE, alpha=0.8)
    ax.legend(fontsize=8, loc="best")
    return ax
