"""Random walk on the simple symmetric exclusion process -- MODELS["rwre"].

PLAN.md's ladder step 3. A walker reads a *dynamic* disordered environment:
a 1-D SSEP at density alpha, in which the walker's local drift depends on
whether it is standing on a particle or on a hole,

    P(left  | particle) = p        P(left  | hole) = 1 - p
    P(right | particle) = 1 - p    P(right | hole) = p

(Igor's convention, prompts/planned_tasks/RWRE_prompt.tex; Avena-Thomann's
arXiv:1201.2890 eq. (1.5) is the same rule with their p = P(right | particle),
so their p = 2/3 is this module's p = 1/3.) At p = 1/2 the environment is
invisible and the walk IS a simple random walk -- which is what makes
MODELS["srw"] a literal control arm rather than a loose analogy.

The observable is Y_k = |X_k| after k walker steps, the same observable as
experiments/01_srw. `alpha` is the environment density: rho is the geometric
scale ratio everywhere in this repo, so the density gets its own letter.

No target_fn and no true_gamma_key, for the reason every model here since srw
has had neither: nothing known is handed to an estimator. The difference is
that here almost nothing IS known -- see experiments/02_rwre/README.md.

What is specified, and why each choice is a choice
--------------------------------------------------
"Particles move to adjacent empty sites" is not a specification; these four
decisions all move the answer, and all four were taken explicitly (2026-09-15).

(a) ENVIRONMENT: a brick-wall parity sweep. One sweep picks a parity b in
    {0, 1} -- independently for each sample -- and every bond of that parity
    swaps its two sites independently with probability `swap_prob`. Because
    the swap is applied to the SITES and not to the particles, this is the
    stirring construction: the update is a random permutation of coordinates
    chosen independently of the configuration, so

        the marginal law of the environment is EXACTLY product Bernoulli(alpha)
        at every time t,

    for every alpha, which is a closed-form property a test can check rather
    than a convergence a test has to hope for (test_rwre.py::
    test_environment_marginal_is_invariant). Observing only occupancies, the
    stirring process IS the SSEP.

    The parity is drawn PER SAMPLE, not once per sweep for the whole block.
    Sharing it would correlate the rows, which ground rule 2 forbids (the n
    samples at a scale must be i.i.d.), and making it deterministic
    (b = t mod 2) would be worse than either: the walker's index parity is
    also determined by t, so the walker's site would be the LEFT endpoint of
    its active bond at every single step -- a systematic coupling between the
    walker and the update it is reading. The cost is that the sweep is a
    masked update over all W columns rather than a slice over W/2; a shared
    parity would be ~2-3x faster and is the first thing to try if this model
    ever needs to be optimized. Correctness first (the prompt's own
    instruction: do the honest version, optimize after it agrees with itself).

(b) STEP ORDER: read, jump, then stir. At time t the walker reads eta_t(X_t) --
    the site it has been standing on -- jumps, and only then does the
    environment move, so no swap happens "during" the walker's own jump.

(c) RATE: `env_sweeps` sweeps per walker step. Expected swap attempts per bond
    per unit time is env_sweeps * swap_prob / 2 (a bond is picked when its
    parity is, w.p. 1/2), so in the units of Avena-Thomann -- walk at rate 1,
    SSE generator at rate gamma per bond --

        gamma_eff = env_sweeps * swap_prob / 2,

    and the default (4, 1/2) puts the walk and the environment at the same
    rate, gamma_eff = 1. This is a real physical parameter, not a tuning knob:
    their Conjecture 3.5 is a phase diagram IN it (sub-diffusive below
    gamma_1, super-diffusive between, diffusive above gamma_2, both boundaries
    increasing in p). env_sweeps = 0 freezes the environment and gives Sinai's
    RWRE, whose displacement grows like (log k)^2.

(d) WINDOW: W(k) = window_c * ceil(sqrt k), rounded up to even, periodic, with
    the walker starting at the centre. The walker moves ~sqrt(k) and the SSEP
    transports information diffusively, so the probability that anything wraps
    is bounded by 2 exp(-window_c^2 / 8) ~ 1.5e-8 at the default 12 -- small
    enough that no log factor is needed, which matters because it leaves

        cost(k) = k * W(k) = window_c * k^(3/2)

    an EXACT power law: Assumption 7 (eq. 353) holds here with d = 3/2 on the
    nose. The bound is stated so it can be tested rather than trusted: doubling
    window_c must not move a fitted constant by more than its own error
    (experiments/02_rwre/README.md, criterion A4). W is kept even so that both
    parities are perfect matchings of the periodic window -- with W odd, the
    wrap bond would share a site with another bond of the same parity and the
    "swap" would stop being a permutation.

    `window_exponent` (default 1/2) replaces the sqrt: W(k) = window_c *
    ceil(k^window_exponent). The bound above assumed DIFFUSIVE spread, and
    experiments/02_rwre measured |X_k| ~ k^0.58 away from p = 1/2, which
    reaches |X|/(W/2) = 0.53 at k = 32768. At 3/4 (user, 2026-09-24, for
    experiments/13_rwre_gpu's longer ladders: "just to be safe") the window
    outgrows any exponent below 3/4 and cost(k) = window_c * k^(7/4), d = 7/4.
    At the default every draw is bit-identical to before the parameter existed.

Blocking, and a deliberate deviation from models/README.md's contract
---------------------------------------------------------------------
That contract asks a model to block over n and stay bit-identical at any block
size, because src/generate/generate.py's chunked path re-enters simulate()
with a smaller n. srw satisfies it by drawing one (block_n, k) array per block,
so row-blocking consumes numpy's row-major stream in the same order an
unblocked call would; percolation_zd_stream satisfies it by drawing one sample
at a time.

Neither is available here: one sample is a k-step LOOP, so the stream is
interleaved across steps, and row-blocking necessarily reorders it. What holds
instead, and what the tests check, is:

  - the output is a pure function of (seed, i, n, params) -- `rows` is derived
    from a fixed byte budget and W(i), both pure functions of (i, params), so
    there is no hidden knob and a rerun reproduces a run bit for bit;
  - two different byte budgets agree DISTRIBUTIONALLY (two-sample KS), which
    is the property the statistics actually rest on.

The alternative -- per-sample draws, contract kept verbatim -- costs about 3x
at the top rung, which is where the budget goes. That was weighed and declined
(user, 2026-09-15). generate.py's chunked path would need >1.25e8 samples at a
single scale to trigger here and never will; if it ever does, the run is still
valid i.i.d. sampling, it just will not be bit-comparable to an unchunked one.
"""

from __future__ import annotations

import numpy as np

#: Working-set budget for one row-block, in bytes. The per-sweep temporaries
#: (a float32 uniform, the parity mask, the two rolled copies) dominate; ~10
#: bytes per site is the measured working set per row, so rows is set from
#: this budget and W. Same role as srw's _DEFAULT_WORKING_SET_BYTES: it bounds
#: peak transient memory regardless of how large n gets, and -- unlike srw's --
#: it is NOT a free knob, because changing it changes the draws (see the
#: module docstring).
_DEFAULT_WORKING_SET_BYTES = 256 * 1024 * 1024

#: env (1 byte/site) + one float32 uniform (4) + the rolled/masked temporaries
#: the sweep allocates (~5). Only the ORDER of magnitude matters: it decides
#: how many rows fit, not what they contain.
_BYTES_PER_SITE = 10

_DEFAULTS = {
    "p": 0.5,
    "alpha": 0.5,
    "env_sweeps": 4,
    "swap_prob": 0.5,
    "window_c": 12.0,
    "window_exponent": 0.5,
}


def window_width(k: int, window_c: float = 12.0, window_exponent: float = 0.5) -> int:
    """Width of the periodic environment window for a k-step walk.

    window_c * ceil(k**window_exponent), rounded up to an even number (both
    parities must be perfect matchings of the periodic window -- see the
    module docstring), and never below 4 so that there are at least two bonds
    of each parity.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if window_c <= 0:
        raise ValueError(f"window_c must be > 0; got {window_c}")
    if window_exponent == 0.5:
        root = np.ceil(np.sqrt(float(k)))      # the original rule, bit for bit
    else:
        # pow is not exact at perfect powers (16**0.75 may be 8.000000001), and a
        # ceil there would add a whole site; snap within 1e-9 relative first.
        r = float(k) ** window_exponent
        root = float(round(r)) if abs(r - round(r)) <= 1e-9 * r else float(np.ceil(r))
    w = int(np.ceil(window_c * root))
    w += w % 2
    return max(w, 4)


def _check(params: dict) -> dict:
    """Merge with defaults and reject anything outside its range."""
    q = dict(_DEFAULTS)
    unknown = set(params) - set(_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown rwre params {sorted(unknown)}; known: {sorted(_DEFAULTS)}")
    q.update(params)
    if not 0.0 <= q["p"] <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {q['p']}")
    if not 0.0 <= q["alpha"] <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {q['alpha']}")
    if not 0.0 <= q["swap_prob"] <= 1.0:
        raise ValueError(f"swap_prob must be in [0, 1]; got {q['swap_prob']}")
    if int(q["env_sweeps"]) != q["env_sweeps"] or q["env_sweeps"] < 0:
        raise ValueError(f"env_sweeps must be a non-negative integer; "
                         f"got {q['env_sweeps']}")
    q["env_sweeps"] = int(q["env_sweeps"])
    if not 0.5 <= q["window_exponent"] <= 1.0:
        raise ValueError(f"window_exponent must be in [1/2, 1] (the walk spreads at "
                         f"least diffusively, at most ballistically); "
                         f"got {q['window_exponent']}")
    return q


def gamma_eff(params: dict | None = None) -> float:
    """Environment rate in Avena-Thomann units: walk at rate 1, SSE at gamma.

    A bond is picked when its parity is (probability 1/2) and then swaps with
    probability swap_prob, env_sweeps times per walker step, so the expected
    number of swap attempts per bond per unit time is
    env_sweeps * swap_prob / 2. Reporting-only: nothing in the simulation
    reads it.
    """
    q = _check(dict(params or {}))
    return q["env_sweeps"] * q["swap_prob"] / 2.0


def _stir(env: np.ndarray, col_parity: np.ndarray, swap_prob: float,
          rng: np.random.Generator) -> np.ndarray:
    """One brick-wall sweep, in place of a loop over bonds.

    `active[r, j]` flags the bond (j, j+1 mod W) of row r, stored at its LEFT
    endpoint. A bond is active when its parity matches the row's parity for
    this sweep and its own uniform falls under swap_prob. Both parities are
    perfect matchings (W is even), so no site belongs to two active bonds and
    the whole sweep is one permutation, applied with two rolls:

        site j takes env[j+1] if bond j is active,
        site j takes env[j-1] if bond j-1 is active,
        otherwise it keeps what it had.
    """
    rows, w = env.shape
    par = (rng.random(rows, dtype=np.float32) < 0.5).astype(np.int8)
    u = rng.random((rows, w), dtype=np.float32)
    active = (col_parity[None, :] == par[:, None]) & (u < swap_prob)
    return np.where(active, np.roll(env, -1, axis=1),
                    np.where(np.roll(active, 1, axis=1),
                             np.roll(env, 1, axis=1), env))


def walk(k: int, n: int = 1, params: dict | None = None,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """n i.i.d. realizations of the SIGNED displacement X_k. int64, shape (n,).

    Annealed, as ground rule 2 requires: every sample draws its own fresh
    environment and its own walk. One environment carrying many walks is the
    quenched measure -- a different experiment, and not one this function can
    be talked into by any choice of arguments.

    `rng` defaults to a fresh unseeded Generator; pass a seeded one for
    reproducible runs. simulate() below returns |X_k|; this function exists
    because the SIGN is the environment's zero-check (E X_k = 0 exactly, by
    particle-hole symmetry composed with reflection) and the tests need it.
    """
    q = _check(dict(params or {}))
    rng = rng if rng is not None else np.random.default_rng()
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")

    w = window_width(k, q["window_c"], q["window_exponent"])
    half = w // 2
    col_parity = (np.arange(w) % 2).astype(np.int8)
    p, alpha, sweeps, swap_prob = q["p"], q["alpha"], q["env_sweeps"], q["swap_prob"]

    block = max(1, _DEFAULT_WORKING_SET_BYTES // (_BYTES_PER_SITE * w))
    out = np.empty(n, dtype=np.int64)

    offset = 0
    while offset < n:
        rows = min(block, n - offset)
        env = rng.random((rows, w), dtype=np.float32) < alpha
        x = np.zeros(rows, dtype=np.int64)
        who = np.arange(rows)
        for _ in range(k):
            # read: is the walker standing on a particle?
            on_particle = env[who, (x + half) % w]
            # jump: P(left) is p on a particle and 1 - p on a hole.
            p_left = np.where(on_particle, p, 1.0 - p)
            x += np.where(rng.random(rows) < p_left, -1, 1)
            # stir: the environment moves only after the walker has read it.
            for _ in range(sweeps):
                env = _stir(env, col_parity, swap_prob, rng)
        out[offset:offset + rows] = x
        offset += rows
    return out


def simulate(i: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["rwre"].simulate: n i.i.d. samples of Y_i = |X_i| at scale i.

    The scale i IS the number of walker steps k.
    """
    return np.abs(walk(i, n=n, params=params, rng=rng))


def cost_hint(i: int, params: dict | None = None) -> float:
    """Work for one sample of |X_i|: i steps over a window of W(i) sites.

    Exact, not an estimate -- every step updates the whole window, `env_sweeps`
    times, and env_sweeps is constant across scales so it cancels in every
    ratio the allocation takes (tools/models.py's ModelSpec.cost_hint). Hence

        cost(i) = i * W(i) = window_c * i^(1 + window_exponent),

    and Assumption 7's exponent is d = 3/2 exactly at the default window
    (7/4 at window_exponent = 3/4) -- known rather than fitted,
    as it is for srw (d = 1) and percolation_zd (d = dim), so a measured d can
    be scored against it.
    """
    q = _check(dict(params or {}))
    return float(i) * float(window_width(i, q["window_c"], q["window_exponent"]))
