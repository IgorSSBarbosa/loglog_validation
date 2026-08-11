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
      recipe -- output (data + metadata, same stem) goes to `data/` instead. `load()`
      reads persisted data directly; `reproduce()` regenerates from the recipe as a
      separate correctness check~~
- [x] ~~`tools/loglog_plot.py`: generic log-log plot of $\overline Y_i$ vs $i$ (any
      experiment's `{scale: samples}`), $\pm 1$ SE bars, optional known-$\mathbb{E} Y_i$
      overlay; `experiments/00_synthetic/plot_loglog.py` wires it to `generator.py`~~
- [ ] Decide how checkpoint acceptance criteria actually get verified going forward
      (the first attempt, a standalone `run_checkpoint_0_1.py` script, was removed —
      unclear value, not the right shape; alternative not yet agreed)
- [ ] 0.1 fidelity check — reopen once verification approach is settled
- [ ] 0.2 `tools/loglog.py` estimator + identity/noiseless unit tests
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
- [ ] `tools/loglog.py` — weighted estimator
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
