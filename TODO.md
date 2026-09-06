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
- [x] ~~0.4 coverage calibration of the $\omega_1$/$a_1$/$\gamma$ error bars (done
      2026-08-22). **FOUND A REAL DEFECT**: every "95%" interval this repo had
      published was an 88% interval. `calibration/check_coverage.py` replays Experiment B's
      exact configuration (scales $8..256$, the real per-scale $n$, $R=5$) 2000 times
      against srw's known truth and counts interval hits; all six quantity/centre
      combinations came in at $0.877$--$0.891$ against a nominal $0.95$, and all six
      are calibrated ($0.946$--$0.955$) once the quantile is $t(R-1)=2.776$ instead of
      the normal's $1.960$. The cause is *only* the quantile: `se_ratio` (stated se
      over the estimates' actual scatter) is $0.94$--$1.03$, so the bar is the right
      size -- an sd from 5 points is noisy and downward-biased ($c_4(5)=0.940$) and
      $t(4)$ is exactly that correction. New `tools/coverage.py` (`coverage_test`,
      `coverage_multi` for scoring many quantities from one pass, `rescore` for
      re-asking at another level without re-running, `interval`,
      `wilson_score_interval`, `se_ratio`, `combine_se`, `consistency_threshold`).
      Three side findings: (a) the pooling-vs-averaging decision is confirmed at 2000
      trials -- $\hat a_1$ bias $-0.0155$ mean-of-fits vs $-0.0019$ pooled, 8x smaller;
      (b) the "mismatched pair" worry (centre from the pooled refit, width from the
      unpooled spread) is backwards -- pooling shrinks the centre's scatter, moving it
      *closer* to the stated se, `se_ratio` $0.870\to0.974$; (c) calibration is
      $n$-dependent and does not transfer -- below $n$-scale $\approx0.1$ the estimator
      itself fails ($(a_1,\omega_1)$ unidentified, $\hat a_1\approx-2\times10^{10}$ at
      $0.003$), so coverage must be re-checked before trusting an error bar at a
      smaller budget. The full-pipeline srw arm was dropped as unaffordable in the
      regime that works ($\approx160$ s/trial); the Gaussian planting is instead
      validated directly by `--arm planting` (one-sample KS of real srw
      $\overline Y_i$ against the planted normal, run at deliberately small $n$ since
      normality of a sample mean only improves with $n$ -- all six scales pass,
      observed sd within 2% of exact). Fix applied where an se becomes a *decision*:
      `src/report/plot_allocation.py`'s $|z|<2$ rule now uses `combine_se`'s
      Welch--Satterthwaite effective dof and cuts at $t(\mathrm{dof_{eff}})=2.111$;
      both existing verdicts are unchanged. Verified: `tools/tests/test_coverage.py`
      (33 cases -- an exactly-calibrated interval must measure 95%, a halved one and a
      biased one must each be caught and *distinguished*, `coverage_multi` must agree
      with `coverage_test` bit-for-bit on one seed, `rescore` must equal a fresh run at
      the new setting) and `tools/tests/test_check_coverage.py` (14 cases -- exact
      $\mathbb E|S_k|$ against brute-force enumeration, the parity staircase, the
      replay pooling the same way `allocation_table.py` does, the KS test shown to have
      teeth against a one-se shift). Still open from the original 0.4 wording: a
      *bootstrap* (as opposed to replicate-spread) estimator for these constants, and
      $\sigma_\infty^2$~~
- [ ] 0.5 Error-decay law under optimal allocation + Wilson CI coverage
- [x] ~~**No-leakage check** (`calibration/check_no_leakage.py`, done 2026-09-04) --
      the falsification test for the question this repo keeps having to answer: is any
      constant secretly hardcoded? On srw the estimates are *right*, which is exactly
      the problem, since $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$, $d=1$ are numbers a
      default can sit on (`tools/constants.py` documents the version of that bug this
      project shipped). So it PLANTS truths from a seeded generator that appears in no
      recipe or default, runs the real pipeline, and regresses recovered on planted.
      Two criteria: unbiasedness (within a multiplicity-corrected $t$ threshold, **or**
      to better than 1% of the truth -- `se` is sampling error only and a nonlinear fit
      has a systematic floor below it), and **responsiveness** (slope $=1\pm0.15$,
      $R^2\ge0.9$), which is the one that catches a leak: a hardcoded constant gives a
      flat line, while criterion 1 alone would be *fooled on srw*. Four arms -- `gamma`
      (the article's own eq. 523-526 estimator), `correction` (the real pilot ->
      `fit_correction`), `cost` ($d\in\{0.5,0.75,1,1.5,2\}$ against a new tunable spin
      burn in `models/synthetic.py`, which without it has $d=0$ and cannot exercise the
      cost machinery at all), and `srw` at $q\ne1/2$, scored against the **exact**
      $\mathbb E|S_k|$ so the estimator's own bias is subtracted rather than assumed
      small. **40 cells, all seven parameter checks pass**, slopes $1.000\pm0.015$,
      every $R^2\ge0.996$. And the negative control that makes that mean something:
      `--inject-leak omega1=1.0155` puts the literal old `FALLBACK_OMEGA1` back into
      the real fit and the check catches it -- slope $-0.000$, no se at all, 39-71%
      relative error -- while $\gamma$, $a_0$, $a_1$ stay green, so it *localizes* the
      leak. A failure is diagnosed as `flat` (a leak) or `unidentified at this budget`
      (a budget problem that says nothing either way -- stage 3.0's vacuous-test lesson
      made printable). Side finding, measured and **not** acted on: on a model whose declared cost
      is exact by construction, `_resolve_d` called `MISMATCH` on **13 of 40 probes
      (32%), and 17 of 50 at ten probes per cell** -- every one a false alarm. Cause: the se a single probe states for its
      own $\hat d$ is 2.4-4.4x smaller than the probe-to-probe spread of $\hat d$
      (both sources measure *within*-probe jitter), so `D_MISMATCH_Z = 3` really tests
      at $|z|\approx1.1$. The point estimate is fine -- mean $\hat d$ within 0.004 of
      truth at every exponent. Fix proposed in `plans/saverepo.md` stage 4: take
      `se(d)` from a few repeated probes, ~0.2 s each. The arm re-measures
      `mismatch_rate` on every run, so the fix is checkable.
      Verified: `tools/tests/test_check_no_leakage.py` (28 cases -- a hardcoded
      constant must fail, a hardcoded constant near the truth must still fail
      criterion 2 while passing 1, a constant offset must fail 1 and pass 2, the srw
      reference against the closed form) and `tools/tests/test_synthetic.py` (15 --
      the burn consumes no randomness, realizes its declared exponent, and stays under
      the overhead ceiling)~~

## Shared tools (built alongside Phase 0, as each is first needed)
- [ ] `tools/rng.py` — independent-stream seeding (ground rule 2); not yet needed since
      0.1 only draws one replicate per call — will extract once 0.3 needs many fresh
      independent replicates
- [x] ~~`tools/io.py` — metadata sidecar (seed, config, timing), fixed deterministic paths;
      kept local to `experiments/00_synthetic/generator.py` for now (one consumer so
      far) — extract to `tools/` once a second experiment needs the same pattern~~ —
      extracted as `tools/persistence.py` (not `tools/io.py`: that name would shadow
      the stdlib `io` module once `tools/` is on `sys.path`, breaking anything else
      imported afterward in the same process). `save_samples`/`load_samples`/
      `load_metadata`/`write_metadata`/`content_id`/`normalize_scales_n`, generic over
      any JSON-serializable `params` dict. `generator.py` refactored to import these
      instead of its own local copies (re-exports the names, so `plot_loglog.py`'s
      `from generator import load_metadata, load_samples, ...` needed no changes;
      content-hash payload shape kept byte-identical, so existing hash-named committed
      images stay valid). Second consumer: `experiments/01_srw/generate.py` (new) draws
      $n$ i.i.d. $|S_k|$ samples per scale via `srw()`, same recipe/output shape as
      `generator.py`; its own `experiments/01_srw/plot_loglog.py` (new) reuses
      `tools/loglog_plot.py` directly, deliberately *without* running
      `tools/loglog.py`'s $\hat\gamma$ estimators or a reference-curve overlay (no
      article-sanctioned closed form for SRW yet -- see `experiments/01_srw/README.md`
      "Sample generation" section). Verified: `tools/tests/test_persistence.py` (8
      cases), `experiments/01_srw/test_generate.py` (shape + `reproduce()` exact-match),
      full suite + both experiments' CLIs run end-to-end
- [x] ~~`tools/cost_model.py` — cost-model exponent $d$ estimator (article Assumption
      `cost_is_power_law`, $\mathrm{cost}(i)=i^d$); reuses `tools/loglog.py`'s OLS-slope
      machinery (same log-log-linear form as $\gamma$) behind a name-keyed registry
      (`COST_ESTIMATORS`) so alternative approaches can be added later. Validated two
      ways: `tools/tests/test_cost_model.py` on synthetic noiseless cost curves for a
      spread of $d$, and `experiments/01_srw/` (new `srw.py` + `measure_cost.py` +
      `test_cost_probe.py`) timing a genuinely $\Theta(k)$ simple-random-walk simulator
      -- recovers $\hat d\approx 0.90$--$1.09$ depending on how many small, overhead-
      dominated scales are dropped, comfortably inside the $[0.8,1.2]$ acceptance band
      around the known ground truth $d=1$. This SRW use is separate from -- and does not
      unblock -- Phase 1's still-blocked gamma-estimation ladder (see `experiments/01_srw/README.md`)~~
- [x] ~~`tools/allocation.py` — budget allocation rule + cost accounting (now unblocked
      from a "how do we get $d$" standpoint, but not started)~~ — `optimal_allocation`
      (Proposition `prop:opt`, eq. 945-946) and `total_cost` (Lemma `lem:budget`'s
      closed-form geometric-sum cost). Discretization (the theorem treats $n$, $m_0$ as
      continuous; an experiment needs integers) is not addressed by the article, so this
      was a real design decision, not a formula lookup: flooring both is provably safe
      (`total_cost` increasing in both $\Rightarrow$ cost $\le B$) whenever the continuous
      $n_{\mathrm{exact}}\ge1$; testing that invariant caught a real edge case -- at small
      $B$, $n_{\mathrm{exact}}<1$ and forcing $n=1$ would silently overspend the budget
      (e.g. $B=10$ costing 28). Fixed via an `integer_feasible` diagnostic flag
      (`n`/`m0`/`cost` are `None` when `False`), the same "diagnostic the caller must
      check, not a raise" pattern `gamma_mle`'s `trustworthy` already uses -- the
      continuous quantities ($\theta_1,\theta_2,\kappa$, $n_{\mathrm{exact}}$,
      $m_{0,\mathrm{exact}}$) stay well-defined and returned either way. Verified:
      `tools/tests/test_allocation.py` (15 cases) -- $\theta_1+d\theta_2=1$ exactly at the
      optimum, continuous allocation costs exactly $B$, discretized allocation never
      exceeds $B$ when feasible, the small-$B$ infeasibility case itself, parameter
      validation ($d,\omega_1>0$, $\rho>1$, $m\ge1$, $B\ge1$)
- [x] ~~Consolidate per-experiment scripts (user request, 2026-08-12): each experiment
      had grown its own `generate.py`/`measure_cost.py`/`plot_cost.py`/`plot_loglog.py`
      -- replaced with one shared copy of each in `tools/`, dispatching on a recipe's
      new `"model"` field via a `tools/models.py` registry (`ModelSpec`: `simulate`,
      optional `target_fn`/`true_gamma_key`). Model-specific code moved out of
      `experiments/*/`: `tools/model_synthetic.py` (was `generator.py`'s
      `SyntheticParams`/`NOISE_FAMILIES`/`mean_Y`), `tools/model_srw.py` (was
      `experiments/01_srw/srw.py`). `tools/plot_loglog.py` only runs `loglog.py`'s
      gamma-hat estimators when the dispatched model has a `target_fn` -- currently
      only `"synthetic"` -- preserving the deliberate SRW behavior without a special
      case in the driver. `tools/persistence.py` now nests each run under
      `<out_dir>/<tag>/{samples.npz,metadata.json}` (was flat `<tag>.npz`+`<tag>.json`)
      so `data/` stays navigable with dozens of runs; `write_metadata` also records
      `model`. Default `out_dir` for both `generate.py` and `measure_cost.py` is derived
      from the recipe file's own location (`<meta>.parent/data`), not the (no longer
      experiment-specific) script's location, so each experiment's runs still land
      under that experiment's own `data/`. Also (same request): `tools/tests/` is now
      gitignored -- kept on disk, run locally via `python3 -m pytest`, but not tracked
      in git and therefore absent from a fresh checkout or `EnterWorktree` worktree; all
      test files were consolidated there (`experiments/01_srw/test_*.py` deleted,
      content ported to `tools/tests/test_model_srw.py`/`test_generate.py`/
      `test_measure_cost.py`, plus new `test_models.py`). Verified: full local suite (49
      cases) passing, both experiments' generate/plot/measure_cost/plot_cost CLIs run
      end-to-end producing identical numbers to before the consolidation
- [x] ~~Three-way split, `tools`/`src`/`models` (user request, 2026-08-12): pulled the
      CLI drivers (`generate.py`, `plot_loglog.py`, `measure_cost.py`, `plot_cost.py`)
      out of `tools/` into a new `src/` -- "code called by other code" (`tools/`) vs.
      "code called directly" (`src/`), the user's own framing. Then pulled
      `tools/model_synthetic.py`/`tools/model_srw.py` out into their own new `models/`
      (`models/synthetic.py`/`models/srw.py`), with `tools/models.py` left behind as a
      pure importer/registry. That last move had a real self-collision risk: the new
      top-level package is named `models`, the same name as the file `tools/models.py`
      that needs to import from it -- `from models.srw import ...` from inside
      `tools/models.py` would resolve back to itself (already bound in
      `sys.modules["models"]` by whichever caller reached it via a bare `from models
      import ...`), not the sibling directory. Fixed by never importing the literal
      name `models` from within `tools/models.py`: it adds `models/` itself to
      `sys.path` and imports `srw`/`synthetic` as bare top-level names instead. Also
      caught and fixed a real pre-existing bug while touching this: `test_measure_cost.py`
      inserted the wrong directory onto `sys.path` (`Path(__file__).resolve().parent`,
      i.e. `tools/tests/` itself) and only passed when run as part of the full suite,
      because an earlier-collected test file happened to have already fixed `sys.path`
      as a side effect -- failed when run in isolation. Verified: full local suite (49
      cases) passing both together and with `test_measure_cost.py`/`test_srw.py` run in
      isolation; all four `src/` CLIs re-run end-to-end producing identical numbers
- [x] ~~Generalize the estimator comparison to every model, not just `"synthetic"`
      (user request, 2026-08-12): `src/report/plot_loglog.py`'s raw-data `plot.png` now always
      overlays the all-points OLS fit (solid line, $\hat\gamma$ in the legend) — needs
      no known ground truth, just the data itself — in addition to the known
      $\mathbb{E} Y_i$ curve (dashed) when a `target_fn` exists. `tools/loglog.py`'s
      `compare_methods` (all four estimators) now runs and writes `results.json`
      unconditionally, for every model, not gated on `target_fn` — comparing estimators
      against each other doesn't need a known truth, only comparing against one does;
      when `true_gamma` is unknown, an explicit "exploratory, not validated" note is
      printed instead of skipping the computation. The four-estimator comparison
      *chart* (`estimates.png`) is opt-in via a new `--estimates` flag, since unlike
      `results.json` it's a supplementary figure (ground rule 1). Also added
      `all_points.a0_hat` to `compare_methods`'s output (the OLS fit's intercept,
      exp'd) — needed to actually draw the fitted line, previously discarded. Verified:
      `tools/tests/test_loglog.py` (+2 cases — `a0_hat` recovery, `true_gamma`
      genuinely optional), both models' `plot_loglog.py --estimates` re-run end-to-end
      (SRW's four estimators agree closely around $\hat\gamma\approx0.5$, consistent
      with the classical $\sqrt{2k/\pi}$ asymptotic, without any `target_fn` being
      registered for it)
- [x] ~~Fix OOM on large-`n` SRW runs (user request, 2026-08-19): `experiments/01_srw/recipes/samples_huge.json`
      ($n=10^8$, scales up to $1024$) had to be killed for exhausting memory. Root cause: a
      single unblocked `srw(k, n, ...)` call drew one $(n,k)$ matrix (819 GiB at
      $k=1024,n=10^8$ with the old `int64` dtype) before `generate.py`'s loop ever regained
      control -- flushing already-*finished* scales at some memory threshold wouldn't have
      helped, since the very first over-budget scale never finishes. Fixed at two
      independent, complementary layers: `models/srw.py` now draws int8 steps in
      `(block_n, k)` blocks over the $n$ axis (not $k$ -- splitting the leading axis
      preserves numpy's row-major RNG draw order, so results are bit-identical to the
      unblocked path for the same seed at any block size; splitting $k$ would not have this
      property, see the module's docstring) bounded to a fixed byte budget regardless of how
      large $n$ or $k$ get; `src/generate/generate.py` streams any run whose total estimated size
      exceeds a byte budget straight to on-disk per-scale arrays
      (`tools/persistence.py`'s new `open_scale_writer`/`load_samples` fallback,
      `<tag>/samples/<scale>.npy` instead of one `<tag>/samples.npz`) in chunks, with a
      `psutil`-based backstop (new dependency) that shrinks the chunk size further if
      system memory hits 90% mid-run. Ordinary-sized runs are completely unaffected (same
      `samples.npz` output, unchanged code path). Verified: `tools/tests/test_srw.py` (+2
      cases -- `block_n` exact-equivalence with the unblocked path, a large-$(n,k)$ case
      that would be gigabytes unblocked), `tools/tests/test_persistence.py` (+2 cases --
      `open_scale_writer` round-trip, flat `samples.npz` takes precedence over a stray
      `samples/` dir), `tools/tests/test_generate.py` (+1 case -- chunked path matches the
      in-RAM path exactly for the same seed); full suite (56 cases) passing. Smoke-tested
      end-to-end at $n=2\times10^7$ (20x the already-working `larger_test.json`) -- RSS
      stayed a few GiB throughout, no flush warnings triggered, chunked output matched the
      in-RAM path exactly. `Huge_test.json` itself not re-run end-to-end this session (would
      take a long time) but is expected to complete without OOM now~~
- [x] ~~Make the cost probe measure cost rather than overhead, and speed up `srw` without
      flattening it (user request, 2026-08-20; `plans/three_experiment_ladder.md` §1).
      Context: the user wants enough samples for `estimates.png` to show a smooth
      $\hat\gamma_i\approx\gamma+a\,i^{-\omega_1}$ decay, which forces small scales -- but at
      small scales `cost_probe`/`time_measure` showed no power law, so the budget-allocation
      theorem couldn't be applied there. Diagnosed as **model misspecification, not timing
      noise**: over $k=2\dots1024$ the measured cost is affine, $\approx22\,\mu s +
      0.025\,\mu s\cdot k$, i.e. a fixed Python/NumPy dispatch overhead that doesn't scale
      with $k$ at all (100% of the measurement at $k=2$, 88% at $k=256$). Also confirmed the
      timing *aggregator* was never the problem -- min 0.884 / median 0.888 / iqmean 0.887 /
      mean 0.850 / q95 0.811 on the same data. Three changes: (1) `tools/cost_model.py` gains
      `estimate_cost_affine`, fitting $\mathrm{cost}(i)=a+b\,i^d$ in log space and reporting
      $a$ as a diagnostic -- this rescues the small-scale regime completely, taking
      `time_measure` from $\hat d=0.103$ to $\hat d=0.948$ and `cost_probe` to
      $\hat d=1.006$ against ground truth $d=1$; (2) an `AGGREGATORS` registry
      (min/median/mean/q95/iqmean, per-recipe, default `median` -- user's choice) plus
      `median_ci`, a distribution-free interval from the order statistics, which is the
      actual reason to prefer `median` over `min` (a minimum has no comparable interval);
      (3) `models/srw.py` swaps `rng.choice(..., p=)` for a float32-uniform draw, 4.4x
      faster (28.3 -> 6.6 µs/sample at $k=1024$; full suite 59s -> 15s). Two faster draws
      were tried and rejected: `rng.integers(0,2,dtype=int8)` (fastest, but numpy's
      bit-packing discards leftover bits per call, so row-blocking stops being exact --
      caught by the existing `block_n` invariance test, and it would have silently broken
      `generate.py`'s chunked path too), and `rng.binomial(k,q,size=n)` (~375x faster,
      distributionally identical, but constant-time per scale -- it would destroy the
      $\Theta(k)$ cost that makes `srw` a percolation stand-in and void Experiments A/C;
      user's call). Note the speedup made the *pure* $\hat d$ worse (0.88 -> 0.77): faster
      work against unchanged overhead, which is exactly the misspecification, and is why
      acceptance moved to the affine fit. Verified: `tools/tests/test_cost_model.py` (+8
      cases -- exact $(a,b,d)$ recovery on planted affine curves, the pure fit shown
      genuinely biased low on the same data, aggregator behaviour under a planted outlier,
      `median_ci` coverage measured empirically at 95.6% vs nominal 95% and its
      $1/\sqrt N$ shrinkage), `tools/tests/test_measure_cost.py` (+2 cases, acceptance
      moved to the affine $\hat d$); full suite 67 cases passing. Cross-check: the
      independent `drop_leading` local-slope diagnostic converges to 0.998, agreeing with
      the affine fit's 1.006 and the $\Theta(k)$ ground truth~~
- [x] ~~Record the exact ground truth for $Y_k=\lvert S_k\rvert$ (2026-08-20). While
      designing the $\omega_1$ experiment, found that
      $\mathbb{E}\lvert S_k\rvert=k\binom{k-1}{\lfloor(k-1)/2\rfloor}2^{-(k-1)}$ exactly --
      verified exhaustively against $2^{-k}\sum_j\binom kj\lvert2j-k\rvert$ for every
      $k=1..200$ and by Monte Carlo at $2\times10^6$ samples. Its expansion is
      $\sqrt{2/\pi}\,k^{1/2}\exp(-\frac14 k^{-1}+\frac1{24}k^{-3}+O(k^{-5}))$, i.e. exactly
      article eq. (232) with $a_0=\sqrt{2/\pi}$, $\gamma=1/2$, $\omega_1=1$, $a_1=-1/4$ --
      and the $k^{-2}$ term cancels identically, putting $\omega_2=3$, three orders below
      $\omega_1$, which makes this an unusually clean testbed for measuring $\omega_1$.
      Deliberately NOT wired into `tools/models.py` as a `target_fn`/`true_gamma_key`
      (user's decision D1/D2, 2026-08-20): stated as hand-checked acceptance criteria in
      `experiments/01_srw/README.md` instead, so the estimators are never handed the answer
      they are supposed to be measuring; and the article's `appendix-SimpleRandomWalk` is
      left untouched, since the user already has these derivations elsewhere~~
- [x] ~~Experiment A — the cost exponent $d$ (`plans/three_experiment_ladder.md` §2).
      **CLOSED 2026-09-04 (user's decision), with one of its two cross-checks run and
      the other deliberately dropped.** What A was for is a $d$ that can be trusted by
      the allocation, and $d$ now has two independent determinations that agree:
      the wall clock, via `estimate_cost_affine`'s $a + b\,i^d$ fit, and each model's
      own declared `cost_hint`, which for srw is exact by construction ($i$ steps per
      sample, no early exit). `src/estimate/measure_cost.py` scores one against the
      other on every run and `src/study/pilot.py` does the same inside a study
      (`_resolve_d`, `d_check` in `pilot.json`), so a disagreement is now a standing
      check rather than an experiment someone has to remember to run — measured
      2026-09-04 on a pilot ladder: $d = 0.9969 \pm 0.0586$ against a declared $1$.
      The **amortized/batched timing comparison is not run and will not be**: it was
      designed to separate per-call overhead from real work, and the affine fit
      already does that by fitting the overhead as a parameter ($a \approx 22\,\mu s$
      on this machine, reported as `overhead_share`). Batching would answer the same
      question by a second route at the cost of a second timing harness, and the two
      determinations already in place disagree by less than the clock's own noise~~
- [x] ~~Experiment B — measure $\omega_1$ (`plans/three_experiment_ladder.md` §3, done
      2026-08-20). **PASSED**: $\omega_1 = 1.0155 \pm 0.1050$ against the known $1$, alongside
      $\gamma = 0.5000 \pm 0.0003$, $a_1 = -0.2748 \pm 0.0597$, $a_0 = 0.7979 \pm 0.0017$ --
      all four within half a standard error -- over 5 independent replicates ($B=4\times10^{10}$
      each, scales $8..256$). New: `tools/correction.py` (two estimators: a direct fit of
      eq. (232)'s one-correction truncation, and a fit of how `gamma_drop_leading`'s bias
      decays), `src/estimate/estimate_omega1.py` (the driver, writes `<run_dir>/omega1.json`), and
      recipe-level allocation rules in `generate.py` (`"n": {"rule": ..., "budget": ...}`).
      Three findings changed the design mid-flight, each documented where it bites:
      (a) **the planned Neyman allocation was wrong.** It minimizes the variance of
      $\overline Y_i$, but $\omega_1$ lives in the *correction term*, whose size shrinks with
      $i$ -- so Neyman over-samples where the correction is already resolved and starves
      where it is buried (measured SNR 460 at $k=2$ down to 0.25 at $k=1024$). Replaced by
      `snr_allocation`, $n_i \propto i^{2\omega_1}$, i.e. INCREASING in $i$ -- the opposite
      trend. `neyman_allocation` is kept and tested, with a test asserting the two trend
      oppositely so they can never be silently conflated. (b) **the scale grid must not mix
      parities**: $\mathbb E|S_{2m-1}| = \mathbb E|S_{2m}|$ exactly, so the mean is a
      staircase; a $\rho=\sqrt2$ grid returns $\hat\omega_1 \approx 17.8$ on *exact*
      means, and the failure is silent (converges, small residual). Powers of 2 are safe.
      (c) **the window is a bias-variance tradeoff**: including $k=2$ caps $\hat\omega_1$
      at 0.959 no matter the sample size (the $\omega_2=3$ term is 4.2% of the $\omega_1$
      term there), while dropping small scales shrinks the signal -- hence $8..256$
      (ceiling 0.995). Also fixed a latent data-corruption bug found while running this:
      a rerun crossing the chunking threshold left the old layout in place, and
      `load_samples` prefers `samples.npz` over `samples/`, so a stale file silently
      shadowed the fresh run~~
- [x] ~~Experiment C — test Proposition `prop:opt`'s allocation (`plans/three_experiment_ladder.md`
      §4, done 2026-08-20). `src/budget/allocation_experiment.py` +
      `experiments/01_srw/recipes/sweep_allocation.json`. The planned single flat-$n$ control arm was
      generalized: `prop:opt` already fixes $n$ uniform, so what it really chooses is $m_0$, and
      the honest control is **every other $m_0$ at the same budget**. That also separates two
      claims a single-point measurement cannot distinguish. **RATE passes**: measured
      $\mathrm d\log\mathrm{RMSE}/\mathrm d\log B = -0.364$ at `prop:opt`'s own $m_0$
      ($-0.384$ at the best $m_0$) against the predicted $-\omega_1/(d+2\omega_1)=-1/3$.
      **POINT fails by a constant**: `prop:opt`'s $m_0$ is 3-4 too high at every budget
      (7 vs 3, 8 vs 5, 9 vs 6 at $B=10^7,10^8,10^9$), costing $2.18$, $2.18$, $2.39\times$ in
      RMSE. It is an offset, not a wrong trend -- the empirical argmin tracks
      $\theta_2\log_\rho B$ in slope (0.30 across five decades analytically, vs
      $\theta_2=1/3$) but sits $\approx3.3$ lower. Mechanism: `prop:opt` **over-corrects for
      bias** -- at the empirical optimum $|\mathrm{bias}|/\mathrm{sd} = 1.68/0.77/0.60$, the
      balance its own derivation argues for, but at the $m_0$ it names, $0.09/0.12/0.30$, i.e.
      bias driven far below the noise floor at the cost of samples. Confirmed independently by
      an exact calculation (article weights on the exact $\mathbb E|S_k|$, analytic sd from the
      half-normal CV) that draws no samples and predicts the same argmins and penalties. None of
      this contradicts the theorem, which is a rate result correct up to constants -- but the
      dropped constant costs a factor $\approx2.2$-$2.4$ in RMSE, i.e. $\approx10\times$ in
      budget, for anyone following the formula literally. The offset **is** derivable in
      closed form from $a_1$, the CV and $\|w\|$, and was: $\theta_2\log_\rho\kappa=-3.94$
      here, implemented as `allocation_constants`/`tuned_allocation`. Re-scored on the wide
      sweep ($10^4$--$10^9$, $R=40$, tag `allocation_wide`): the gap to the measured argmin is
      flat in $B$ (slope $-0.05$ per decade of $\log_\rho B$, mean gap $3.67$), confirming a
      pure constant rather than a wrong $\theta_2$; the tuned $m_0$ lands within one step of
      the argmin at all six budgets, for an RMSE penalty of $1.00$--$1.02\times$ against
      `prop:opt`'s $2.02$--$3.35\times$. Verified:
      `tools/tests/test_allocation_experiment.py` (13 cases -- ladder shape, budget arithmetic
      never overspends, replicate streams reproduce and are independent across cells,
      unaffordable cells marked skipped rather than faked, summary/rate machinery on planted
      inputs)~~
- [x] ~~`tools/wilson.py` — the article's Wilson interval, Theorem `thm:wilson`
      (eq. 720): the four-term bound $\mathcal B_{\mathrm{fs}}+\mathcal
      B_{\mathrm{good}}+\mathcal B_{\mathrm{bad}}+\Phi(\alpha)\sigma_{\mathrm{se}}$
      on $|\hat\beta-\beta|$, done 2026-08-24 **for $\gamma$ only** (user's scope call).
      That restriction is the theorem's, not ours: eq. (720) is a statement about
      $\hat\beta=\sum_k w_{k,m}\log\overline Y_{\rho^k}$ and says nothing about
      $\omega_1$ or $a_1$, which come from `correction.py`'s nonlinear fit that the
      article does not analyse. Why it was worth building rather than keeping the
      replicate interval: its $\sigma_{\mathrm{se}}=\sqrt{12\sigma_\infty^2/(nm^3)}$
      is a **closed form** in $\sigma_\infty^2$, estimable from the raw samples
      ($1.7\times10^8$ of them at $k=256$, relative error $\sim5\times10^{-5}$), so it
      needs no replicates and $\Phi(\alpha)=1.960$ is legitimate where the 5-replicate
      interval is forced to $t_4=2.776$. Verified against measurement: $\sigma_{\mathrm{se}}$
      predicts `gamma_closed_form`'s scatter to 1.1% ($4.313$ vs $4.267\times10^{-4}$),
      and $\mathcal B_{\mathrm{fs}}$ bounds its true bias by $1.60\times$ — correctly
      conservative. **The decisive comparison** (1500 trials, equal total budget, on the
      article's own estimator): the replicate interval covers $0.000/0.021/0.835/0.945/
      0.955$ at $m_0=2/4/6/8/10$ — it collapses on shallow ladders because it has no bias
      term at all and sits 8 half-widths off truth — while the Wilson bound covers
      $1.000/1.000/0.996/0.982/0.966$ everywhere and, once bias is negligible, is 20-29%
      **narrower** at the same budget. Also: `moment_bounds` estimates
      $\sigma_\infty^2$/$\sigma_{\max}^2$/$\Lambda$ from real samples, and
      `sigma_se_per_scale` generalises the fourth term to the non-uniform $n$ that
      Experiment B's snr allocation produces (eq. 720 assumes uniform $n$; substituting a
      mean $n$ there is wrong by a factor of ten). Terms whose constants we have not
      measured ($\mathcal B_{\mathrm{bad}}$ needs $\Lambda,\delta$; the $\omega_2$ piece
      needs $\phi^+$) are omitted only with a loud `complete=False` flag — a bound missing
      a term is not a bound. Verified: `tools/tests/test_wilson.py` (23 cases — each term
      against the equation as written, $\mathcal B_{\mathrm{fs}}$ shown to bound the exact
      bias, the bias/variance crossover in $m_0$, monotonicity, and that omitting a term
      is impossible to miss). Not covered: $\omega_1$/$a_1$ intervals, which still need
      replicates or `tools/bootstrap.py`~~ — note `tools/coverage.py`'s
      `wilson_score_interval` remains a **different object** (binomial score interval, for
      putting a CI on a measured coverage proportion)
- [ ] `tools/bootstrap.py` — resampling for constants

## Later phases (not started, not designed yet)
- [ ] Resolve open question: Bethe-lattice closed form (article appendix is an empty
      stub) — needed before Phase 4 can be designed. The **SRW** half of this is now
      settled for $Y_k=\lvert S_k\rvert$: exact $\mathbb{E} Y_k$, $\gamma=1/2$,
      $\omega_1=1$, $a_1=-1/4$, $\omega_2=3$, recorded as acceptance criteria in
      `experiments/01_srw/README.md` (article deliberately left untouched — user's D1)
- [ ] Phase 1 — SRW
- [ ] Phase 2 — RWRE (cross-check against `critical_exponents/estimators/log_log_plot.py`)
- [ ] Phase 3 — Percolation $\mathbb Z^d$, $d=2..6/7$, side-connected cluster
  - [x] ~~$d=2$ model: `models/percolation2d.py`, `MODELS["percolation2d"]`. $Y_i$ =
        open sites of an $i\times i$ box connected to the SOUTH side, at
        $p_c=0.59274605079210$, 4-connected (ground rule 7). No `target_fn`: $91/48$
        is an acceptance criterion, not an input. `anchor="origin"` exists only as the
        comparison arm. Verified: `tools/tests/test_percolation2d.py` — exact
        enumeration of $\mathbb{E}Y_i$ over all $2^{i^2}$ configurations at $i=2,3$, an
        independent flood fill on random critical lattices, $p\in\{0,1\}$, $i=1$,
        block-invariance, and $P(Y_i=0)=(1-p)^i$ exactly~~
  - [x] ~~$d=2$ cost model: affine $\hat d = 2.0285\pm0.0175$ vs the declared $2$
        ($+1.63\sigma$). **First rung where Assumption 7 is a measured geometric fact**,
        not a stated formula. See `experiments/03_percolation_zd/README.md`, P1~~
  - [x] ~~$d=2$ first $\hat d_f$: $1.9161$ (direct fit of eq. 232) / $1.9059$
        (bias-decay), vs $91/48=1.89583$. cv$(Y_i)$ flat at $\approx0.43$ across
        $i=8..512$, so Assumption 6 looks satisfied. P2~~
  - [x] ~~Side-vs-origin head to head at equal budget
        (`src/estimate/compare_observables.py`, P3)~~
  - [x] ~~Cylinder geometry (periodic in $x$), P4. Removes the two side walls, which
        are pure contamination for a south-anchored count: RMSE $3.7\times$ better at
        all points and $2.2\times$ at $m_0=2$, at $1.04\times$ the cost, and the
        $m_0=2$ cell becomes the **first in this project that is variance-dominated
        rather than bias-dominated**. Cost exponent unchanged ($\hat d = 1.978\pm0.039$
        against the declared 2), so the wrap-merge lands in the affine overhead~~
  - [ ] The cylinder's residual $+0.004$ bias in $\hat\gamma$ ($\approx7\sigma$) is now
        the leading error. It does **not** decay with $m_0$ over $8\le i\le512$ — the
        drop-leading ladder is flat at $\approx1.900$ — so a wider ladder is needed to
        tell a very slowly decaying correction from an amplitude effect the
        one-correction model cannot express
  - [ ] $\omega_1$ for $d=2$ is **unsettled**: on the box the two estimators give $0.33$
        and $0.38$ over $8\le i\le512$, against a literature $\Omega=72/91\approx0.79$;
        on the cylinder the bias-decay estimator does not converge at all, correctly,
        because the $\hat\gamma(m_0)$ sequence it fits is flat. Needs a wider ladder
        (to $i\sim4096$) and probably a two-correction fit before anything is claimed
  - [ ] Parallel generation: `generate.py` is single-threaded and this machine has 12
        threads; ground rule 2's spawned streams already make replicate-level fan-out
        safe and reproducible. Prerequisite for the wider ladder above ($31.7$ ms per
        sample at $i=1024$)
  - [ ] Wilson interval (eq. 720) on $\hat d_f$: needs $\omega_1$ first
  - [x] ~~$d\ge3$: a general-$d$ simulator. **Done, 2026-09-06**:
        `models/percolation_zd.py` and `models/percolation_tau_zd.py` are
        `percolation2d.py` / `percolation_tau.py` with the spatial dimension as a
        `params["dim"]`, bit-identical to their 2-D originals at `dim = 2`
        (`test_matches_percolation2d_bit_for_bit`,
        `test_matches_percolation_tau_bit_for_bit`). `cost_hint(i) = i**dim`, so
        Assumption 7's exponent $d$ *is* the dimension and one recipe field sweeps
        $d = 2\ldots6$. Experiments in `experiments/05_percolation_highd/`~~
  - [x] ~~**Ground rule 7's observable is a $d\le4$ statement.** Generalizing the 2-D
        derivation gives $\gamma_{\text{face}} = \max(d_f, d-1)$, because
        $\sum_h h^{-\beta/\nu}$ stops being dominated by $h\sim i$ once
        $\beta/\nu = d - d_f > 1$, i.e. from $d = 5$ on — so the face-connected count
        measures the trivial surface exponent $d-1$ there, not $d_f$.
        `anchor="face_far"` (the far half of the box only) restores $\gamma = d_f$ in
        every dimension and keeps Assumption 6. Measured at $4\times10^9$ sites:
        $\hat\gamma = 2.5295$ vs $d_f(3)=2.523$ and $3.0501$ vs $d_f(4)=3.045$
        (`face_far`), against $2.5591$ and $3.2102$ (`face`)~~
  - [x] ~~$\tau$ in $d=3$ (`models/percolation_tau_zd.py`): $\hat\tau = 2.1860$ / $2.1946$
        at $m_0 = 5,6$ against $\tau(3) = 1 + 3/d_f = 2.18906$, on an 8-rung ladder
        $s = 8\ldots1024$ at $4.3\times10^9$ sites. cv flat at $0.34$–$0.38$ (Assumption 6
        clean, and *quieter* than the 2-D run's $0.58$–$0.62$), both $\omega_1$ estimators
        available and agreeing to $0.03$. **The cluster-size ladder does not have the
        memory problem the $d_f$ ladder has**: $L(s)\propto s^{1/d_f}$, so eight rungs fit
        in every dimension up to 6~~
  - [x] ~~`DF_LOWER` is a design constant, demonstrated not asserted: two runs differing
        only in `box_exponent` ($1/d_f^- = 0.4049$ vs the 2-D model's $0.5$) agree on
        $\hat\tau$ to $0.003$ at $m_0\le3$, while the cv goes from flat ($0.384\to0.340$)
        to falling ($0.290\to0.148$) and the cost exponent from $1.21$ to $1.50$~~
  - [x] ~~**The origin anchor does not measure $d_f$** — a correction to
        `models/percolation2d.py`, `experiments/03_percolation_zd/README.md` and
        `PLAN.md` ground rule 7 (2026-09-06). $\mathbb E|C(0)\cap B_i|$ is the
        box-restricted susceptibility $i^{\gamma/\nu}=i^{d-2\beta/\nu}$ ($43/24$ in
        $d=2$, not $91/48$): the slip was writing $i^2\pi_1(i)$ as $i^{91/48}$ when it is
        $i^{43/24}$. P3's own origin arm measured $1.7593\ldots1.7955$, converging on
        $43/24$ — recorded at the time as an observable failing to converge. Confirmed at
        $d=3$: $1.995, 2.120, 1.865$ against $\gamma/\nu(3)=2.045$. No code changes; P3's
        conclusion (use the side anchor) is unchanged and strengthened~~
  - [x] ~~**`anchor="slab"` with `anchor_dim` $=k$** (Igor's proposal, 2026-09-06):
        the seed set's dimension as one knob, $k=0$ the centre site through
        $k=\mathrm{dim}$ every site. $\gamma(k)=k+\gamma/\nu$ below $\beta/\nu$, $d_f$
        on the plateau $\beta/\nu\le k\le d_f$, $k$ above it — so the plateau *measures*
        $d_f$ and its edges give $\beta/\nu$ and $d_f$. Verified: $k=0$ reproduces
        `origin` sample-for-sample by a different code path, $k=\mathrm{dim}$ gives
        $\gamma=\mathrm{dim}$ exactly, monotone in $k$, flood-fill agreement at
        $\mathrm{dim}=3$ for all geometries~~
  - [ ] **H7's full-budget sweep.** The prototype settles $\mathrm{dim}=3$ (the axis
        wins: bias $0.017$ vs the $k=2$ slab's $0.127$, and the zero fraction goes
        $0.698\to0.000$) and leaves $\mathrm{dim}=5$ open in two places — whether $k=1$
        really falls out of the plateau (errors are $\pm0.2$–$1.8$ there), and why $k=3$
        reads $\approx4.0$ where the derivation says $3.54$
  - [ ] **`SITES_PER_BUDGET_UNIT = 256` does not hold in high $\mathrm{dim}$.** The
        intent was that one allocation budget unit is $256$ lattice sites in every
        dimension (`box_factor` $=256^{1/\mathrm{dim}}$), so the recipe-writing rule needs
        no per-dimension footnote. `box_side`'s `ceil` defeats it: `box_factor` shrinks
        with $\mathrm{dim}$, so $L$ is small ($5$–$15$ at $\mathrm{dim}=6$) and the
        rounding is a large relative cost. Measured `cost_unit_ratio` on
        $s = 8\ldots1024$: $264, 277, 434, 413, \mathbf{1031}, 675$ at
        $\mathrm{dim} = 2\ldots7$ — good to $8\%$ at $\mathrm{dim}\le3$, a $4\times$
        under-estimate at $6$. Nothing downstream is wrong (`cost_unit_ratio` is exact for
        the ladder in hand and `plan.py` bisects on seconds), but a **hand-written
        `budget`** is: `samples_tau_d6.json` asks for what looks like $4\times10^{9}$
        sites and is $1.61\times10^{10}$. Either document "call `cost_unit_ratio` first"
        as the rule, or make `box_side` return a value the unit identity survives
  - [ ] $\tau$ at $\mathrm{dim}\ge6$, where $\tau = 5/2$ and $d_f = 4$ are **exact** --
        the first real (unplanted) process in this repo with a rational target.
        `samples_tau_d6.json` / `samples_tau_d7.json`; the $d=6$ run is ~$5.6\times10^9$
        sites and takes tens of minutes single-threaded
  - [ ] $\omega_1$ in $d\ge3$ has **only one estimator available**: a ladder short
        enough to fit in memory ($6$ rungs at $d=3$, $5$ at $d=4$, $4$ at $d=5$) leaves
        fewer than the 4 drop-leading windows `estimate_omega1.py`'s bias-decay fit
        needs, so it errors out and only the direct fit of eq. (232) reports. Two
        independent estimators disagreeing is what the 2-D rung used to catch a bad
        $\omega_1$; that check is gone above $d=2$ until the ladder can be widened
  - [ ] **The $p_c$ table above $d=5$ is not independently checked.**
        `src/estimate/check_criticality.py` PASSes at $d=3,4,5$ on the box (spanning
        drift $-0.030$, $-0.035$, $+0.065$ against controls at $\mp0.12$ or more) and
        is **inconclusive at $d=6$** ($+0.157$ against controls $-0.277/+0.233$): the
        spanning probability's own finite-size correction is large at the box sides
        $d=6$ can afford, and at and above the upper critical dimension flatness at
        $p_c$ stops being the right criterion. $p_c(6)$ and above rest on Mertens &
        Moore (2018) alone
  - [ ] **The cylinder is the wrong geometry for the criticality check in high $d$**,
        though it stays the right one for sampling: with $d-1$ periodic transverse
        directions the spanning probability at $p_c$ rises with $i$ (drift $+0.04$ at
        $d=3$, $+0.113$ at $d=4$, $+0.323$ at $d=5$) because more transverse channels
        get more chances. Run H0 on the box
  - [ ] The measured cost exponent falls increasingly short of the declared one as $d$
        grows — $2.934\pm0.016$ vs $3$, $3.809\pm0.030$ vs $4$, $4.620\pm0.042$ vs $5$
        (all PASS at the driver's 20% tolerance, the last flagged DISAGREE at
        $9.1\sigma$). The affordable probe ladder shrinks with $d$ while the fixed
        $\approx220\,\mu s$ dispatch overhead does not — it is 95% of the measurement
        at the bottom rung in $d=4$ — so the affine fit has a short lever arm. Needs a
        probe that reaches higher, which needs the memory budget raised
- [ ] Phase 4 — Percolation on hierarchical/Bethe graphs, exact recursion cross-check
