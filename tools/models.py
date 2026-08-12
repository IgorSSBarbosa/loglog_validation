"""Model registry: names a model to src/generate.py, src/measure_cost.py, and
src/plot_loglog.py via a recipe's "model" field, so those scripts are single,
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
    simulate: Callable[[int, int, dict, np.random.Generator], np.ndarray]
    target_fn: Callable[[np.ndarray, dict], np.ndarray] | None = None
    true_gamma_key: str | None = None


MODELS: dict[str, ModelSpec] = {
    "synthetic": ModelSpec(
        simulate=model_synthetic.simulate,
        target_fn=model_synthetic.target_fn,
        true_gamma_key="gamma",
    ),
    "srw": ModelSpec(
        simulate=model_srw.simulate,
        # No target_fn/true_gamma_key: no article-sanctioned closed form for
        # SRW yet (see models/srw.py, experiments/01_srw/README.md). This is
        # what keeps src/plot_loglog.py from running gamma-hat estimators
        # against this model -- not a special case in the driver, just the
        # absence of a target_fn here.
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {list(MODELS)}")
    return MODELS[name]
