# Log-Log Plot Technique — Simulation Validation

Companion simulation project for the paper **"The Log-Log Plot Technique"**
(`../article_writting/article.tex`). Purpose: **validate or refute** the paper's
assumptions and theorems by simulation, across a ladder of testbeds of increasing
complexity. Writing/review of the paper itself happens in `article_writting/`, not here.

This is a fresh restart of `../loglog_experiments/` (kept, untouched, as a historical
reference — do not port its code). That attempt moved fast enough that it became hard
to trust which numbers were actually checked against anything. This project inverts the
priority: **correctness of each piece, checked numerically, comes before any figure.**

## Ground rules

1. **Verify before visualize.** A tool or model is not "done" when it produces a plot;
   it is done when it has passed a stated numeric check (a closed-form identity, a
   degenerate/noiseless case, or a calibration test with an explicit tolerance). Every
   `experiments/*/README.md` states its acceptance criteria as numbers, not adjectives.
   A figure is a supplement to a passing numeric check, never a replacement for one.
2. **Independence rule — no reuse of simulations across configurations.** Every
   `(experiment, configuration, replicate)` draws fresh, independent randomness. Raw
   samples generated for one scale, one budget point, or one estimator window must
   never be sliced or resampled into another. This is a direct fix for the bug in
   `presentation18-05-2026/coding/analysis/*.py`, which sliced overlapping chunks of one
   shared `V_pool` across different `(n, m, m0)` regimes and different budgets — breaking
   the i.i.d.-across-replicates assumption the theory depends on, and making the
   resulting comparisons between regimes silently invalid. See `tools/rng.py`.
3. **One experiment at a time**, and within it, one checkpoint at a time. Design is
   agreed before coding, especially wherever the article itself is still incomplete
   (e.g. the SRW and Bethe-lattice appendices are currently empty stubs — those
   testbeds need an explicit closed-form target before any code is written).
4. **No guessed formulas.** Every statistical object used is taken from the article by
   equation/theorem number (see the reference table below). If something is ambiguous,
   stop and ask.
5. **Reproducibility.** Every stochastic run seeds its RNG and records the seed,
   config, and elapsed time in a metadata sidecar next to its output.
6. **No output sprawl.** Each `(experiment, configuration)` writes to a fixed,
   deterministic path and overwrites on rerun. Timestamped one-off files are only for
   explicitly archived milestone results, not the default — this is a direct fix for
   the timestamp-per-run clutter in `loglog_experiments/data/` and `figures/`.
7. **Percolation observable: side-connected cluster, not origin cluster.** For all
   percolation testbeds, $Y_i$ is the number of open sites in a box of side $i$
   connected to a full face (side) of the box — not the cluster containing the origin.
   This was the concrete methodological fix requested for this project, correcting
   `presentation18-05-2026`'s origin-anchored $V(r)$.

## Paper objects (source of truth — cite by equation/theorem, don't re-derive)

| Object | Article location | Form |
|---|---|---|
| $J$-order expansion (Assumption 1) | eq. (232) | $\mathbb{E} Y_i = a_0 i^\gamma \exp(a_1 i^{-\omega_1} + \cdots + \phi_J(i) i^{-\omega_J})$ |
| Positivity (Assumption 2) | line 279 | $Y_i > 0$ |
| $\phi_j$ uniformly bounded (Assumption 3) | eq. (289) | $\max_j \sup_i \lvert\phi_j(i)\rvert \le \phi^+$ |
| $(2+\delta)$-moment of $\xi_k$ (Assumption 4) | eq. (305) | $\mathbb{E}\lvert\xi_k - 1\rvert^{2+\delta} \le M$ |
| $(2+\delta)$-moment of $\log\xi_k$ (Assumption 5) | eq. (319) | $\mathbb{E}\lvert\log\xi_k\rvert^{2+\delta} \le \Lambda$ |
| $\sigma_k^2 \to \sigma_\infty^2$ (Assumption 6) | eq. (332) | convergence; polynomial-rate variant eq. (342) |
| $cost(i) = i^d$ (Assumption 7) | eq. (353) | budget cost model |
| Weighted estimator $\hat\beta,\hat\gamma$ | eq. (523)–(531) | $\hat\beta = \sum_k w_{k,m}\log\overline{Y}_{\rho^k}$ |
| Weights | eq. (526) | $w_{k,m} = \dfrac{12(k - m_0 - (m+1)/2)}{m(m^2-1)}$ |
| Weight identities (Lemma linearization) | eq. (542) | $\sum w_{k,m}=0$, $\sum w_{k,m}k = 1$ |
| CLT for $\hat\gamma$ | Theorem, eq. (583) | $\sqrt{nm^3}(\hat\gamma - \mathbb{E}\hat\gamma) \dto \mathcal N(0, 12\sigma_\infty^2/\log^2\rho)$ |
| Wilson interval | Theorem, eq. (720) | 4-term bound: finite-size + good-event + bad-event bias + $\Phi(\alpha)\sigma_{\mathrm{se}}$ |
| Finite-size bias order | Prop. (820) | $\mathcal B_{\mathrm{fs}} \asymp \rho^{-\omega_1 m_0}/m^2$ |
| Optimal allocation | Prop. (932), eq. (945)–(946) | $\theta_1 = \frac{2\omega_1}{d+2\omega_1}$, $\theta_2 = \frac{1}{d+2\omega_1}$; $n = \kappa B^{\theta_1}$, $m_0 = \theta_2 \log_\rho B$ |
| Error decay | eq. (941)/(966) | $\lvert\hat\beta - \beta\rvert \lesssim B^{-\omega_1/(d+2\omega_1)}$, MSE $\lesssim B^{-2\omega_1/(d+2\omega_1)}$ |
| Minimax lower bound | Theorem (1011) | matches the rate up to $\log^2 B$ |

$\sigma_k^2 = Var(\xi_k)$, $\xi_k = Y_{\rho^k}/\mathbb{E} Y_{\rho^k}$ (notation summary, line 2352).

## Repository layout

```
loglog_validation/
  PLAN.md  TODO.md  README.md  CATALOG.md  requirements.txt
  src/                          <- the scripts a human runs directly (user's own
                                    framing, 2026-08-12): `python3 src/<layer>/x.py`.
                                    tools/ holds what these call, not the other way
                                    around -- src/ may import tools/, tools/ may not
                                    import src/. Split into four layers by the
                                    question each answers (2026-08-25).
    generate/generate.py          single shared sample-generator CLI/API, dispatches
                                  on a recipe's "model" field into tools/models.py's
                                  MODELS registry instead of each experiment
                                  keeping its own copy
    estimate/measure_cost.py      cost-model-exponent probe, same MODELS dispatch;
                                  cross-checks the wall clock against the model's
                                  own declared cost_hint
    estimate/estimate_omega1.py   Experiment B's analysis driver (omega_1, a_1)
    estimate/check_coverage.py    checkpoint 0.4: are the error bars calibrated?
                                  arms planted / planting / rate / wilson
    budget/allocation_experiment.py  Experiment C: sweep (budget x m0), paired arms
    budget/allocation_table.py    precision vs wall clock, from measured constants
    budget/verify_prediction.py   run the tuned ladders for real, compare
    report/plot_loglog.py         single shared log-log plotter; overlays a known
                                  E Y_i curve only when the model has a target_fn
                                  (currently only "synthetic")
    report/plot_cost.py           plot of measure_cost.py's output
    report/plot_allocation.py     Experiment C's two panels
  tools/                        <- shared, experiment-agnostic *helper* functions:
                                    called by src/'s scripts, or by each other, or
                                    by tests -- never run directly. May not import
                                    from experiments/ or src/. Every function here
                                    has a unit test in tools/tests/ before it is
                                    used by any experiment. tools/tests/ is
                                    gitignored (local verification only, not
                                    tracked -- user request 2026-08-12; kept on
                                    disk, run via `python3 -m pytest`, but absent
                                    from a fresh checkout/worktree).
    loglog.py                    weighted OLS estimator (eq. 523-531) + the
                                  closed-form weights (eq. 526) -- the canonical
                                  definition, imported by allocation.py and wilson.py
    correction.py                omega_1 and a_1: direct fit of eq. 232, bias decay
    wilson.py                    Wilson CI (eq. 720), for gamma only
    allocation.py                budget allocation rule (eq. 945-946) + cost
                                  accounting + the tuned constant kappa the rate
                                  theorem drops
    cost_model.py                cost exponent d (pure + affine fits), timing
                                  aggregators, declared-vs-measured cross-check
    coverage.py                  calibration harness for error bars (checkpoint 0.4)
    artifacts.py                 the naming registry: what every file on disk is
                                  called, in (recipes, by kind) and out (run
                                  artifacts, by content). Provenance goes INSIDE
                                  each file, not in its name
    bootstrap.py                 resampling helpers for constants (sigma_inf^2, omega1, a1)
                                  -- NOT WRITTEN YET
    rng.py                       independent-stream seeding (SeedSequence.spawn) --
                                  enforces ground rule 2. as_seed_sequence/seed_record
                                  exist because a spawned child carries its PARENT's
                                  entropy, so passing one as an int collapses every
                                  replicate onto one stream (see its docstring)
    persistence.py                sample+metadata save/load, shared by every model: one
                                  run = one `<out_dir>/<tag>/` folder holding samples.npz
                                  + samples_meta.json (not `io.py`: that name would shadow
                                  the stdlib `io` module once tools/ is on sys.path)
    models.py                    MODELS registry (name -> ModelSpec: simulate, optional
                                  cost_hint/target_fn/true_gamma_key) -- purely an
                                  importer; one entry per models/<name>.py (see below),
                                  which it reaches by adding models/ to sys.path and
                                  importing "srw"/"synthetic" as bare names, never
                                  through the literal name "models" (self-collision
                                  with this file's own identity as tools/models.py
                                  -- see its docstring)
    loglog_plot.py               the shared charts
    tests/                       pytest unit tests, run against closed forms (see note
                                  above -- gitignored)
  models/                       <- per-model simulation logic, one file per
                                    registered model (user's own framing,
                                    2026-08-12), kept separate from both tools/ and
                                    src/. Each exposes simulate(i, n, params, rng),
                                    optionally cost_hint(i, params), and -- only when
                                    the article gives a known closed form --
                                    target_fn(i, params).
    synthetic.py                  the closed-form model (SyntheticParams,
                                  NOISE_FAMILIES, mean_Y, eq. 232) -- has a
                                  target_fn, currently the only one that does;
                                  cost_hint = 1, so d = 0 and it cannot exercise
                                  the budget machinery
    srw.py                        srw(k, n, q, rng) -- cost_hint(i) = i, exact, which
                                  is what makes it the budget testbed. No target_fn:
                                  E|S_k| is known exactly but is deliberately kept out
                                  of the code path (user, 2026-08-20), stated as a
                                  README acceptance criterion instead
  derivations/                  <- standalone write-ups too long for a docstring:
                                  the gamma MLE; the dropped allocation constant
                                  and the 88%-coverage defect
  experiments/
    00_synthetic/                planted-truth model, cheap -- start here
    01_srw/                      simple random walk: the cost probe, Experiment B
                                  (omega_1), Experiment C (budget allocation) and
                                  checkpoint 0.4 all live here
    02_rwre/                     random walk in random environment
    03_percolation_zd/           site percolation, Z^d, d = 2..6/7
    04_percolation_hierarchical/ Bethe lattice / hierarchical graphs
    each experiment/:
      README.md                  what this validates + numeric acceptance criteria +
                                  current status (not started / in progress / passing)
      recipes/<kind>_<name>.json  inputs, named for their kind and validated on load
                                  (samples_*, cost_*, sweep_* -- see
                                  tools/artifacts.py's RECIPES). The model-specific
                                  simulator itself lives in models/<name>.py, not here
      data/                      this experiment's runs (gitignored), one `<tag>/`
                                  folder per run -- never shared with another
                                  experiment. Files are named for what they hold
                                  (samples_meta.json, cost_probe.json, omega1.json,
                                  allocation_sweep.json, prediction_check.json,
                                  gamma_estimates.json, coverage.json), never
                                  `result.json`
      images/                    this experiment's figures only (committed evidence,
                                  ground rule 1), never shared
```

## Experiment ladder

Each rung only starts once the previous rung's acceptance criteria pass. Rungs get
harder along two axes at once: **statistical** (is $Y_i$ well-modeled by the $J$-order
expansion, are the moment assumptions plausible) and **computational** (is $cost(i)
\approx i^d$ actually true here, and what does that do to feasible box sizes).

1. **Synthetic** — ground truth is planted, cost is a stated formula, no model-fidelity
   question at all. Purely tests the *statistical* machinery (estimator, CLT, Wilson CI,
   allocation rule, $\omega_1$-bootstrap) in isolation. See detailed checkpoints below.
2. **SRW** — first real stochastic process with an exactly known asymptotic (the article
   reserves Appendix `appendix-SimpleRandomWalk` for this but it's currently empty), so
   ground truth for $\gamma,\omega_1$ can be checked by hand, not just by planting it.
   Cheap to simulate ($cost(i)$ linear-ish), so still mostly a statistics-focused rung.
3. **RWRE** — same idea as SRW but with a disordered environment; introduces genuine
   model uncertainty (the relevant exponent isn't classical) and connects to
   `critical_exponents/` (whose `estimators/log_log_plot.py` implements the same
   estimator this project validates — worth cross-checking against, not importing).
4. **Percolation, $\mathbb Z^d$** — first rung where $cost(i)=i^d$ is a real geometric
   fact to verify (BFS/union-find over $i^d$ sites), not an assumption. $d=2$ has known
   $d_f = 91/48$; higher $d$ up to the mean-field threshold ($d\ge 6$) are progressively
   more expensive and are where the budget-allocation theory should matter most in
   practice. Side-connected cluster (ground rule 7), not origin cluster.
5. **Hierarchical / Bethe lattice** — exactly solvable via branching-process recursion
   (also currently an empty article appendix), so simulation can be checked against an
   exact generating-function computation rather than only Monte Carlo — the strongest
   correctness check available in this project.

---

## Phase 0 — Synthetic data: detailed checkpoints

This is the immediate next work. Model: $\mathbb{E} Y_i = a_0\, i^\gamma \exp(a_1 i^{-\omega_1})$
(the $J=1$ case of eq. 232), realized multiplicatively as $Y_i = \mathbb{E} Y_i \cdot \xi_i$ with
$\xi_i > 0$, $\mathbb{E}\xi_i = 1$, chosen (e.g. lognormal) so $Var(\xi_i) = \sigma_i^2 \to
\sigma_\infty^2$ by construction — letting us plant $\sigma_\infty^2$ directly and check
Assumption 6 is *exactly* satisfied by design, before ever touching a model where it's
only approximately true.

| # | Checkpoint | Acceptance criterion (numeric) |
|---|---|---|
| 0.1 | Planted generator matches its own formula | At large $i$ (e.g. $i=2^{20}$), empirical mean over many i.i.d. draws matches $a_0 i^\gamma\exp(a_1 i^{-\omega_1})$ within 3 Monte-Carlo standard errors |
| 0.2 | Estimator is algebraically correct | `tools/loglog.py` weights satisfy $\sum w_{k,m}=0$, $\sum w_{k,m}k=1$ to float precision, for a spread of $(m,m_0)$; on **noiseless** data ($a_1=0$, pure power law) recovers $\gamma$ to float precision for any $(m,m_0)$ |
| 0.3 | CLT holds empirically | Over many independent fresh replicates (ground rule 2) at fixed $(n,m,m_0)$: $\mathbb{E}\hat\gamma \approx \gamma$ (small bias) and $Var(\hat\gamma)$ matches $12\sigma_\infty^2/(nm^3\log^2\rho)$ (eq. 583) within a bootstrap CI |
| 0.4 | $\omega_1$-bootstrap recovers the planted constant | A resampling/estimation procedure for $(\omega_1, \sigma_\infty^2, a_1)$ — candidates: nonlinear fit of the expansion directly, or local-slope-drift regression — is **calibration-tested**: over $\gtrsim 200$ independent full synthetic experiments, the bootstrap 95% CI for $\omega_1$ covers the true value in $\approx 93$–$97\%$ of them |
| 0.5 | Error-decay law | Under the optimal allocation (eq. 945-946), for a geometric grid of budgets $B$, empirical RMSE$(B)$ over many fresh independent replicates per $B$ has $\log$-$\log$ slope matching $-\omega_1/(d+2\omega_1)$ within a stated CI; Wilson CI (eq. 720) empirical coverage checked against nominal level at the same time |

Checkpoint 0.4 is exactly the "bootstrap strategy to calculate constants needed for a
scalable use of the computational budget" step requested — it's what makes 0.5 possible
without cheating by looking at the planted truth.

## Open questions before Phase 1 (SRW)

- The article's `appendix-SimpleRandomWalk` and `appendix-BetheLattice` are empty
  section headers — no closed form is written down yet. Before writing any SRW code we
  need to agree on: which observable $Y_i$ (return probability at step $i$? range after
  $i$ steps? something else), its known $\gamma$, and its known/conjectured $\omega_1$.
  This should probably be worked out on paper (and could become the content that fills
  those appendices) before Phase 1 starts.
