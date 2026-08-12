"""Model registry: names a model to tools/generate.py, tools/measure_cost.py,
and tools/plot_loglog.py via a recipe's "model" field, so those scripts are
single, shared drivers rather than one copy per experiment (tools/model_*.py
holds each model's actual sampling logic).

Adding a model means writing a tools/model_<name>.py with a `simulate(i, n,
params, rng) -> array of n samples` function (and, only if the article gives
a known closed form for it, a `target_fn(i, params) -> E[Y_i]` and a
`true_gamma_key` naming which params key holds the true gamma) and adding
one entry here -- no changes to the driver scripts themselves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Self-contained: works whether this module is reached as `tools.models`
# (repo root on sys.path) or as a bare `models` (tools/ itself on sys.path) --
# either way, the model_*.py siblings need their own directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_srw  # noqa: E402
import model_synthetic  # noqa: E402


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
        # SRW yet (see model_srw.py, experiments/01_srw/README.md). This is
        # what keeps tools/plot_loglog.py from running gamma-hat estimators
        # against this model -- not a special case in the driver, just the
        # absence of a target_fn here.
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {list(MODELS)}")
    return MODELS[name]
