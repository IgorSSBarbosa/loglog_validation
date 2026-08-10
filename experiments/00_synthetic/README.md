# 00_synthetic

**Status: not started.** See `PLAN.md` (repo root) "Phase 0" for the full checkpoint
table with acceptance criteria — reproduced here for convenience, kept in sync manually.

Model: $\E Y_i = a_0\, i^\gamma \exp(a_1 i^{-\omega_1})$ (eq. 232, $J=1$), realized
multiplicatively as $Y_i = \E Y_i \cdot \xi_i$, $\xi_i>0$, $\E\xi_i=1$,
$\Var(\xi_i)=\sigma_i^2 \to \sigma_\infty^2$ by construction (e.g. lognormal noise).
Ground truth $(\gamma, a_0, a_1, \omega_1, \sigma_\infty^2)$ is planted and known, so this
rung tests only the *statistical* machinery, not model fidelity.

| # | Checkpoint | Acceptance criterion |
|---|---|---|
| 0.1 | Generator matches its own formula | empirical mean at large $i$ within 3 MC standard errors of $a_0 i^\gamma\exp(a_1 i^{-\omega_1})$ |
| 0.2 | Estimator is algebraically correct | weight identities exact to float precision; noiseless data recovers $\gamma$ exactly |
| 0.3 | CLT holds empirically | $\E\hat\gamma\approx\gamma$, $\Var(\hat\gamma)$ matches eq. (583) within bootstrap CI, over fresh independent replicates |
| 0.4 | $\omega_1$-bootstrap calibrated | 95% CI for $\omega_1$ covers truth in $\approx$93-97% of $\gtrsim$200 independent synthetic experiments |
| 0.5 | Error-decay law | RMSE$(B)$ log-log slope matches $-\omega_1/(d+2\omega_1)$ under optimal allocation (eq. 945-946); Wilson CI (eq. 720) coverage checked |

No code here yet — checkpoint 0.1 is next, pending sign-off on the generator design
(noise family for $\xi_i$, specific planted constants to start with).
