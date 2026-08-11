# TODO

Checked items are struck through. One checkpoint at a time; each is "done" only when
its numeric acceptance criterion (see PLAN.md) passes, not when it runs without error.

## Setup
- [x] ~~Read article theorems relevant to simulation (Sections 2-4, Appendix technical
      proofs, notation summary)~~
- [x] ~~Survey `presentation18-05-2026/` for reusable lessons; identify the pool-reuse
      bug (`analysis/*.py` slicing a shared `V_pool` across regimes/budgets)~~
- [x] ~~Survey `loglog_experiments/` (prior attempt); decide fresh restart, no code
      ported~~
- [x] ~~Write `PLAN.md`~~
- [x] ~~Pin `requirements.txt` (numpy, scipy for stats/bootstrap, matplotlib, pytest)~~
- [x] ~~`git init`, first commit of scaffolding~~

## Phase 0 — Synthetic (see PLAN.md for full checkpoint table)
- [x] ~~Planted generator (`generator.py`): pluggable noise family, arbitrary-length
      $(a_j,\omega_j)$ corrections, CLI (`-meta config.json`) + programmatic
      (`generate(out_dir=...)`) entry points, JSON reproducibility~~
- [x] ~~Persist actual samples (`.npz`, not just metadata); stop rewriting the input
      recipe -- output (data + metadata, same stem) goes to `data/` instead.
      `load_samples()` reads persisted data directly; `reproduce()` regenerates from
      the recipe as a separate correctness check~~
- [x] ~~`plot_loglog.py` takes a data path (`-data`, the `.npz`) instead of a JSON --
      one recipe can produce many runs, so a JSON was ambiguous about which data was
      meant; metadata for the reference-curve overlay is now optional (`load_metadata`
      returns None rather than erroring if missing), never required just to plot~~
- [x] ~~`tools/loglog_plot.py`: generic log-log plot of $\overline Y_i$ vs $i$ (any
      experiment's `{scale: samples}`), $\pm 1$ SE bars, optional known-$\mathbb{E} Y_i$
      overlay; `experiments/00_synthetic/plot_loglog.py` wires it to `generator.py`~~
- [x] ~~`tools/loglog.py`: three $\hat\gamma$ estimators (`gamma_all_points`,
      `gamma_two_point`, `gamma_drop_leading`) as the general OLS slope of
      $\log\overline Y_i$ vs $\log i$; `compare_methods` bundles them.
      `plot_loglog.py` writes `data/<stem>_results.json`. Verified: exact recovery
      on a noiseless planted power law, unsorted-input handling~~
- [x] ~~4th estimator, `gamma_mle`: MLE under $\overline Y_k\sim\mathcal N(\mu_k,
      \sigma^2\mu_k^2/n_k)$ (not the Hill estimator originally requested — see
      below). Full derivation + second-order/concavity analysis in
      `derivations/mle_gamma_estimator.tex`; direct joint optimization over
      $(\gamma,\log a_0,\log\sigma^2)$, with `converged`/`region_ok`/`hessian_pd`/
      `trustworthy` diagnostics that must be checked before trusting `gamma_hat`.
      Verified: 200-replicate Monte Carlo matches the derivation's numbers exactly
      (5/200 not-trustworthy, always via optimizer non-convergence rather than a
      bad estimate); `compare_methods`/`loglog_points` updated to thread `n`
      through~~
- [x] ~~`tools/loglog_plot.py`'s `estimates_plot`: visualizes `compare_methods`'
      four estimators (dataviz-skill palette; two_point/drop_leading vs smallest
      scale in window, all_points/mle as reference lines, mle styled by its
      `trustworthy` status color rather than a 4th hue); `plot_loglog.py` writes
      `images/<stem>_estimates.png`. Verified against both a 3-scale and a dense
      10-scale synthetic run, and the untrustworthy-MLE styling path~~
- [x] ~~`generator.py`: record per-scale wall-clock draw time (`timing_seconds` in
      the output metadata JSON -- raw material for a future meta-log-log plot of
      cost(i) vs i, to estimate the article's cost-model exponent d) and an
      opt-in `progress` flag (stderr, one line per scale; off by default so
      library/Monte-Carlo callers aren't spammed, on for the CLI, which also
      gains an `elapsed_ms` column in its summary table)~~
- [ ] Decide how checkpoint acceptance criteria actually get verified going forward
      (the first attempt, a standalone `run_checkpoint_0_1.py` script, was removed —
      unclear value, not the right shape; alternative not yet agreed)
- [ ] 0.1 fidelity check — reopen once verification approach is settled
- [x] ~~0.2 the article's exact closed-form $w_{k,m}$ weighted estimator (eq. 523-526)
      + algebraic-identity unit tests (eq. 542) — `tools/loglog.py`'s general OLS form
      is mathematically equivalent on a consecutive grid but the closed form itself
      isn't implemented/tested yet~~ — `closed_form_weights`/`gamma_closed_form` added
      to `tools/loglog.py`; `tools/tests/test_loglog.py` (first test file in the repo)
      checks all five weight identities (a)-(e) of Lemma "Elementary identities" for a
      spread of $(m,m_0)$, exact noiseless recovery, agreement with `gamma_all_points`
      on a consecutive grid, and rejection of a non-consecutive one. Verified via
      `python3 -m pytest tools/tests/`, all passing
- [ ] Classical Hill estimator (tail-index from one heavy-tailed sample) -- the
      thing originally requested under this name turned out to target a different
      problem than ours; resolved instead via the MLE above. True Hill estimator
      still not implemented/discussed further
- [ ] 0.3 CLT empirical check (fresh replicates only, per `tools/rng.py`)
- [ ] 0.4 $\omega_1$/$\sigma_\infty^2$/$a_1$ bootstrap + coverage calibration
- [ ] 0.5 Error-decay law under optimal allocation + Wilson CI coverage

## Shared tools (built alongside Phase 0, as each is first needed)
- [ ] `tools/rng.py` — independent-stream seeding (ground rule 2); not yet needed since
      0.1 only draws one replicate per call — will extract once 0.3 needs many fresh
      independent replicates
- [ ] `tools/io.py` — metadata sidecar (seed, config, timing), fixed deterministic paths;
      kept local to `experiments/00_synthetic/generator.py` for now (one consumer so
      far) — extract to `tools/` once a second experiment needs the same pattern
- [ ] `tools/allocation.py` — budget allocation rule + cost accounting
- [ ] `tools/wilson.py` — Wilson CI
- [ ] `tools/bootstrap.py` — resampling for constants

## Later phases (not started, not designed yet)
- [ ] Resolve open question: SRW/Bethe-lattice closed forms (article appendices are
      empty stubs) — needed before Phase 1 can be designed
- [ ] Phase 1 — SRW
- [ ] Phase 2 — RWRE (cross-check against `critical_exponents/estimators/log_log_plot.py`)
- [ ] Phase 3 — Percolation $\mathbb Z^d$, $d=2..6/7$, side-connected cluster
- [ ] Phase 4 — Percolation on hierarchical/Bethe graphs, exact recursion cross-check
