# derivations

Standalone mathematical write-ups: derivations that are too long for a docstring and
not part of the article itself. LaTeX sources, compiled with your own toolchain
(this machine has none) — `pdflatex <file>.tex` twice where a table of contents is
present.

- `mle_gamma_estimator.tex` — the Gaussian MLE for $\gamma$ from per-scale sample
  means (`tools/loglog.py`'s `gamma_mle`): derivation, second-order/concavity
  analysis, and the diagnostics a caller must check before trusting `gamma_hat`.

- `allocation_constant_and_coverage.tex` (+ a `.md` copy of the same content, for
  reading without LaTeX) — two constants that were being dropped. Part I derives in
  closed form the multiplicative constant $\kappa$ that Proposition `prop:opt`
  discards from its budget allocation, worth 8–38x in compute, from measured
  quantities only. Part II derives why every "95%" interval this repo published was
  an 88% interval — $\Pr(|t_4| < 1.96) = 0.8784$ — and what changed as a result.
  Companion to `experiments/01_srw/README.md`, which carries the narrative.

Keep the `.tex` canonical when both formats exist; regenerate the `.md` rather than
editing the two separately.
