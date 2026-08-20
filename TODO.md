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
      (user request, 2026-08-12): `src/plot_loglog.py`'s raw-data `plot.png` now always
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
- [x] ~~Fix OOM on large-`n` SRW runs (user request, 2026-08-19): `experiments/01_srw/Huge_test.json`
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
      large $n$ or $k$ get; `src/generate.py` streams any run whose total estimated size
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
- [ ] Experiment A — amortized/batched cost measurement (`plans/three_experiment_ladder.md` §2).
      May now be redundant: the affine fit already recovers $d$ correctly from $n=1$ timings.
      Worth running as an independent cross-check rather than assuming either way
- [x] ~~Experiment B — measure $\omega_1$ (`plans/three_experiment_ladder.md` §3, done
      2026-08-20). **PASSED**: $\omega_1 = 1.0155 \pm 0.1050$ against the known $1$, alongside
      $\gamma = 0.5000 \pm 0.0003$, $a_1 = -0.2748 \pm 0.0597$, $a_0 = 0.7979 \pm 0.0017$ --
      all four within half a standard error -- over 5 independent replicates ($B=4\times10^{10}$
      each, scales $8..256$). New: `tools/correction.py` (two estimators: a direct fit of
      eq. (232)'s one-correction truncation, and a fit of how `gamma_drop_leading`'s bias
      decays), `src/estimate_omega1.py` (the driver, writes `<run_dir>/omega1.json`), and
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
- [ ] Experiment C — $\gamma$ under `tools/allocation.py`'s optimal budget, **with a flat-$n$
      control arm at equal budget** (`plans/three_experiment_ladder.md` §4). Check first
      whether `prop:opt` degenerates at $(d,\omega_1)=(1,1)$
- [ ] `tools/wilson.py` — Wilson CI
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
- [ ] Phase 4 — Percolation on hierarchical/Bethe graphs, exact recursion cross-check
