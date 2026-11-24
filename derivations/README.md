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

- `heteroscedastic_allocation.tex` — the long-run planner (`m0-star`) when
  $\sigma_i^2$ grows with the scale instead of converging: $\mathrm{Var}\,\hat\beta$
  and the $O(1/n)$ log bias for scale-dependent $\sigma_i$, the adapted $m_0^\star$,
  its rate ($d \to d+2\nu$ in `def:opt`), estimating $\nu$ from the pilot, and a
  guard on $\sigma/\sqrt n$ where the Gaussian model fails. Worked on experiment 60's
  $d=5$ study. An extension, not from the article: awaiting sign-off before it
  enters `plan.py`.

- `batched_observable.tex` — the same problem solved by changing the observable:
  average $K_i=\lceil c\,i^{2\nu}\rceil$ copies of $Y_i$ per sample. Same mean (same
  $\gamma$, $a_1$, $\omega_1$), converging variance, and, via Rosenthal's inequality,
  the uniform $2+\delta$ moment when $\mathbb E|\xi_i-1|^{2+\delta}/\sigma_i^{2+\delta}
  = O(i^{\nu\delta})$, which holds at equality for Leath's indicator, under the
  cluster-size scaling ansatz, and in the measured $\gamma_{\rm susc}$ data. Then the
  article's `thm:CLT`, Wilson bound and planner apply unchanged with $d\to d+2\nu$.
  Fixed-budget price against equal $n$: 12–14% in sd at $d=5$.

Keep the `.tex` canonical when both formats exist; regenerate the `.md` rather than
editing the two separately.
