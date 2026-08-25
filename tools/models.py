"""Model registry: names a model to src/generate/generate.py, src/estimate/measure_cost.py, and
src/report/plot_loglog.py via a recipe's "model" field, so those scripts are single,
shared drivers rather than one copy per experiment. This module is purely an
importer/registry -- each model's actual sampling logic lives in its own
models/<name>.py (a sibling of tools/, not a submodule of it).

Adding a model means writing a models/<name>.py with a `simulate(i, n,
params, rng) -> array of n samples` function (and, only if the article gives
a known closed form for it, a `target_fn(i, params) -> E[Y_i]` and a
`true_gamma_key` naming which params key holds the true gamma) and adding
one entry here -- no changes to the driver scripts themselves.

Import note: the models/ directory is deliberately never imported by its own
name ("models") from here -- this file is itself named tools/models.py, and
since it's usually already bound to `sys.modules["models"]` by the time this
line runs (callers reach it via a bare `from models import get_model`, tools/
itself being on sys.path), `from models.srw import ...` would self-referentially
resolve back to *this* module instead of the models/ directory. Instead,
models/ itself is added to sys.path and its contents imported as bare
top-level names (`srw`, `synthetic`) -- never through the word "models".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

import srw as model_srw  # noqa: E402
import synthetic as model_synthetic  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    """What the rest of the repo needs to know about a model.

    `simulate` is the only required field. `cost_hint` is what makes a model
    plannable: see the note below on why it exists and what unit it is in.
    """

    simulate: Callable[[int, int, dict, np.random.Generator], np.ndarray]
    target_fn: Callable[[np.ndarray, dict], np.ndarray] | None = None
    true_gamma_key: str | None = None
    #: cost_hint(i, params) -> expected work for ONE sample at scale i, in the
    #: model's own natural unit (walk steps, sites explored, lattice updates --
    #: whatever the model actually counts). Only RATIOS across scales matter,
    #: so the unit is free; a constant factor cancels in the allocation.
    #:
    #: Why declared rather than timed. Assumption cost_is_power_law wants
    #: cost(i) = i**d, and wall clock does not satisfy it: a fixed per-call
    #: overhead makes measured time affine, a + b*i**d. Measured on srw, which
    #: is Theta(k) by construction so d = 1 exactly:
    #:
    #:     time, pure power law      d_hat = 0.771   (23% low)
    #:     time, affine a + b*i**d   d_hat = 1.006
    #:     declared step count       d     = 1.000   exactly, no fit
    #:
    #: and restricted to k <= 16384, the regime where omega_1 has to be
    #: measured, the pure-time fit gives 0.506 -- 49% low, because the overhead
    #: is 88% of the measurement at k = 256. Timing still runs, as a
    #: cross-check (tools/cost_model.py's `compare_cost_models`); a declared
    #: hint that disagrees with the clock is a warning, not a silent choice.
    cost_hint: Callable[[int, dict], float] | None = None


MODELS: dict[str, ModelSpec] = {
    "synthetic": ModelSpec(
        simulate=model_synthetic.simulate,
        cost_hint=model_synthetic.cost_hint,
        target_fn=model_synthetic.target_fn,
        true_gamma_key="gamma",
    ),
    "srw": ModelSpec(
        simulate=model_srw.simulate,
        cost_hint=model_srw.cost_hint,
        # No target_fn/true_gamma_key: no article-sanctioned closed form for
        # SRW yet (see models/srw.py, experiments/01_srw/README.md). This is
        # what keeps src/report/plot_loglog.py from overlaying a reference curve or
        # reporting a true_gamma for this model -- not a special case in the
        # driver, just the absence of a target_fn here. The gamma-hat
        # estimators themselves still run (comparing estimators against each
        # other doesn't need a known truth), flagged as exploratory instead.
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {list(MODELS)}")
    return MODELS[name]
