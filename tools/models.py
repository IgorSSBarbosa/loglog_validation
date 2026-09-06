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

Import note: this file is `tools.models` and the simulator package is
`models` -- two different names for two different things, which is exactly why
every import in this repo is written out in full. `from models import srw`
below reaches models/srw.py, never this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from models import percolation2d as model_percolation2d
from models import percolation_tau as model_percolation_tau
from models import percolation_tau_zd as model_percolation_tau_zd
from models import percolation_zd as model_percolation_zd
from models import srw as model_srw
from models import synthetic as model_synthetic


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
    #: shared_sampler(scales, n, params, rng) -> ({scale: n samples}, info)
    #:
    #: The whole ladder drawn from ONE set of n underlying realizations,
    #: instead of `simulate` being called once per scale with an independent
    #: stream. Declared only by models where one realization contains every
    #: rung at once -- percolation_tau, where a single lattice holds clusters
    #: of every size below its cutoff -- and it exists to make the cheap
    #: version of that measurement runnable (user, 2026-09-05).
    #:
    #: It deliberately BREAKS PLAN.md ground rule 2 across scales: the rungs
    #: share randomness, so Cov(Ybar_s, Ybar_s') != 0 and the CLT of eq. (583)
    #: does not apply to them as written. That is the experiment. The rule
    #: still holds where it is load-bearing -- rows are i.i.d., and replicates
    #: draw independent streams -- and every run drawn this way is stamped
    #: `shared_lattice` in its metadata (src/generate/generate_shared.py) so
    #: no reader can mistake one for the other.
    shared_sampler: Callable[..., tuple[dict, dict]] | None = None


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
    "percolation2d": ModelSpec(
        simulate=model_percolation2d.simulate,
        cost_hint=model_percolation2d.cost_hint,
        # No target_fn/true_gamma_key, for the same reason as srw above:
        # gamma = d_f = 91/48 is known from the literature but is kept OUT of
        # the code path so no estimator is handed the answer it is measuring.
        # It lives as a written acceptance criterion in
        # experiments/03_percolation_zd/README.md. See models/percolation2d.py.
    ),
    "percolation_tau": ModelSpec(
        simulate=model_percolation_tau.simulate,
        cost_hint=model_percolation_tau.cost_hint,
        # No target_fn/true_gamma_key, for the same reason as the two above.
        # The scale handed to simulate() here is a CLUSTER SIZE s, not a box
        # side, and gamma = 1 - tau: the Fisher exponent tau = 187/91 and the
        # hyperscaling relation tau = 1 + d/d_f stay OUT of the code and enter
        # only as --expect-gamma / --truth at reporting time. See
        # models/percolation_tau.py.
        shared_sampler=model_percolation_tau.shared_sampler,
    ),
    "percolation_zd": ModelSpec(
        simulate=model_percolation_zd.simulate,
        cost_hint=model_percolation_zd.cost_hint,
        # models/percolation2d.py generalized to a spatial dimension that is a
        # PARAMETER, so Assumption 7's cost exponent d IS params["dim"]:
        # cost_hint(i) = i**dim, exactly, and one model exercises the
        # allocation theory at d = 2..6 (PLAN.md ladder step 4). At dim = 2 it
        # reproduces MODELS["percolation2d"] bit for bit.
        #
        # No target_fn/true_gamma_key, for the same reason as every model
        # above: d_f(dim), and the exact mean-field d_f = 4 for dim >= 6, are
        # acceptance criteria in experiments/05_percolation_highd/README.md and
        # reach the code only as --expect-gamma / --truth at reporting time.
    ),
    "percolation_tau_zd": ModelSpec(
        simulate=model_percolation_tau_zd.simulate,
        cost_hint=model_percolation_tau_zd.cost_hint,
        # models/percolation_tau.py generalized the same way. The scale is a
        # CLUSTER SIZE s and gamma = 1 - tau with tau = 1 + dim/d_f; at
        # dim >= 6 that is EXACTLY 5/2, the first real (unplanted) process in
        # this repo with a rational target. Kept out of the code path all the
        # same.
        shared_sampler=model_percolation_tau_zd.shared_sampler,
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {list(MODELS)}")
    return MODELS[name]
