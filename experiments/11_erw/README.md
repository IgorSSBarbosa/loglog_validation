# 11_erw — Elephant Random Walk

**Status: recipes only (2026-09-22).** The model is built and verified
(`models/erw.py`, `tools/tests/test_erw.py`); the recipes below are the first examples,
and the $p$ grid, the ladder and the time per arm are still to be decided. Nothing here
is a result yet except E1 and E3, which are properties of the model rather than of a run.

**The point of the experiment:** the ERW has a phase transition in its memory
parameter, and the exponent of $\mathbb E\lvert S_k\rvert$ is **proven** on both sides
of it — so it is the first testbed in this repo where $\gamma\ne1/2$ is known exactly
(for `rwre` it was measured, not known), and it has a critical point where the
article's eq. (232) is false.

## The model

Bercu, *A martingale approach for the elephant random walk*, J. Phys. A **51** (2018)
015201, doi:10.1088/1751-8121/aa95a6
([pdf](https://bernardbercu.wordpress.com/wp-content/uploads/2020/04/b-jpa-2018-3.pdf)),
§2, eq. (2.1): $X_1=\pm1$ with probability $1/2$ each; for $n\ge1$, draw $k$ uniformly
from $\{1,\dots,n\}$ and set $X_{n+1}=+X_k$ with probability $p$, $-X_k$ with
probability $1-p$; $S_{n+1}=S_n+X_{n+1}$. The observable is $Y_k=\lvert S_k\rvert$ after
$k$ steps — the same as `01_srw` and `02_rwre`.

`models/erw.py`, registered as `MODELS["erw"]`, one parameter `p` (required). CPU only.
How it samples, why, and what it costs is in `models/README.md`; in one line: the
literal rule, drawn up front as a random recursive tree, $\mathrm{cost}(k)=k$, $d=1$.

## What is known — the acceptance criteria come from here

Every entry cites Bercu (2018). With $q=1/2$ (the fair first step) the paper's $\mathbb
E[L]$ and every odd moment vanish.

| regime | Bercu | $\gamma$ | $a_0$ |
|---|---|---|---|
| $p<3/4$ | $S_k/\sqrt k\to\mathcal N(0,\frac1{3-4p})$, eq. (3.5) | $1/2$ | $\sqrt{2/(\pi(3-4p))}$ |
| $p=3/4$ | $S_k/\sqrt{k\log k}\to\mathcal N(0,1)$, eq. (3.10) | none of eq. (232)'s form | — |
| $p>3/4$ | $S_k/k^{2p-1}\to L$ a.s. and in $\mathbb L^4$, eqs. (3.11)–(3.12) | $2p-1$ | $\mathbb E\lvert L\rvert$, no closed form |

- **From a limit law to $\mathbb E\lvert S_k\rvert$.** Convergence in law does not by
  itself move the mean; uniform integrability does. Below $3/4$ it comes from the exact
  second moment, eq. (A.3) in expectation:
  $\mathbb E S_{k+1}^2 = 1+\bigl(1+\tfrac{2(2p-1)}k\bigr)\mathbb E S_k^2$, which is
  $O(k)$; above $3/4$ it is the $\mathbb L^4$ convergence of eq. (3.12). Hence the $a_0$
  column. Checked against the exact $\mathbb E\lvert S_k\rvert$ (dynamic programming over
  the $(n,S_n)$ chain of eq. 2.2): at $p=0.6$, $\mathbb E\lvert S_k\rvert/\sqrt k =
  1.02852$ at $k=4096$ and $1.02939$ at $16384$, against the limit $1.03006$.
- **At $p=3/4$**, (A.3) solves to $\mathbb E S_k^2 = k\,H_k$ exactly ($H_k$ the harmonic
  number; checked against the recursion to $2\times10^{-15}$ relative at $k=1000$). The $\log k$ is not of the form
  $a_0k^\gamma\exp(\sum_ja_jk^{-\omega_j})$, so **Assumption 1 fails** here. The local
  slope of $\sqrt{k\log k}$ is $\tfrac12+\tfrac1{2\ln k}$ — $0.56$ at $k=4096$ — and
  it reaches $1/2$ only logarithmically.
- **Above $3/4$**, eq. (3.14) gives $\mathbb E L^2 = 1/\bigl((4p-3)\Gamma(2(2p-1))\bigr)$
  — $1.865$ at $p=0.9$ — so $a_0\le1.366$ by Jensen, the only check on $a_0$ there. $L$
  is not Gaussian (Remark 3.2).
- **$p=1/2$ is `srw` in law**, so `01_srw`'s four exact constants hold: $\gamma=1/2$,
  $a_0=\sqrt{2/\pi}$, $\omega_1=1$, $a_1=-1/4$.
- **$\mathbb E S_k=0$ exactly**, at every $k$ and $p$ (eq. 2.3 with $\mathbb E S_1=0$).
- **$\omega_1$ is not known** for $\lvert S_k\rvert$ at $p\ne1/2$, so it is not an
  acceptance criterion. On the exact means above, the relative gap to the $p=0.6$ limit
  shrinks by $2.28\times$ from $k=4096$ to $16384$, i.e. like $k^{-0.59}$ — consistent
  with $3-4p=0.6$, the exponent of the $\mathbb E S_k^2$ correction, but an
  observation, not a theorem.

These stay **out of the code path** (PLAN.md ground rule 4): `MODELS["erw"]` has no
`target_fn` and no `true_gamma_key`, so the report prints γ̂ as exploratory and nothing
upstream of it is told the answer.

## Acceptance criteria — proposed

Numbers, not adjectives (ground rule 1). E2 must pass before E4–E6 are quoted, as in
`02_rwre`. To be confirmed with the user together with the $p$ grid.

| | criterion | status |
|---|---|---|
| **E1** | cost: the clock recovers the declared $d=1$ | **PASS** (below) |
| **E2** | calibration at $p=1/2$: $\hat\gamma=1/2$, $a_0=\sqrt{2/\pi}$, $\omega_1=1$, $a_1=-1/4$, each inside its se | not run |
| **E3** | zero-check: $\lvert\mathbb E S_k\rvert<4$ se | **PASS** in `tools/tests/test_erw.py` ($k=256$, $p\in\{0.3,0.9\}$) and `sec_erw` ($k=64$, $p=0.8$) — the pipeline sees only $\lvert S_k\rvert$ |
| **E4** | diffusive, $p=0.6$: $\hat\gamma$ consistent with $1/2$; $a_0\to1.030$ reported secondary and flagged, as `02_rwre` does | not run |
| **E5** | critical, $p=3/4$: no $\gamma$ to recover. What the pipeline says — the pilot's gate, the $\omega_1$ fit, the drop-leading ladder against $\tfrac12+\tfrac1{2\ln k}$ — and whether it certifies an exponent above $1/2$ | not run; criterion to agree |
| **E6** | superdiffusive, $p=0.9$: $\hat\gamma$ consistent with $2p-1=0.8$ | not run |

### E1 — cost: PASS, with a caveat about the probe

`recipes/cost_probe.json`, $k=2^{10}..2^{18}$, run twice on the same idle host
(2026-09-22):

| run | affine $\hat d$ | small-$k$ overhead | verdict |
|---|---|---|---|
| 1 | $1.093\pm0.029$ | 45 µs | PASS (20% tolerance), $+3.2\sigma$ warning |
| 2 | $0.924\pm0.031$ | 122 µs | PASS (20% tolerance), $-2.5\sigma$ |

Opposite sides of 1: the $n=1$ per-call overhead did not reproduce between runs, while
the top three rungs did, to 4%, each doubling exactly with $k$. The pilot's own probe
(512..4096, five windows) read $1.085\pm0.132$. The allocation uses the declared $d=1$
either way.

## The recipes

| recipe | for | what it is |
|---|---|---|
| `samples_example.json` | `generate.py` | a few-second demo at $p=0.9$, $k=64..8192$, $n=4000$, fixed seed |
| `cost_probe.json` | `measure_cost.py` | E1 |
| `samples_calib_p0.5.json` | `autopilot.py` | E2, the control arm |
| `samples_diffusive_p0.6.json` | `autopilot.py` | E4 |
| `samples_critical_p0.75.json` | `autopilot.py` | E5, the misspecification probe |
| `samples_super_p0.9.json` | `autopilot.py` | E6 |

The four pilot recipes are identical but for $p$: ladder $4..4096$ in powers of two
(`|S_k|` is a parity staircase — see `01_srw`), `"snr"` with no $\omega_1$ (flat
$n$, no design constant stated), budget $10^9$, `seed: null`. Each `_note` says why.
In `tools/allocation.py`'s units a flat replicate is $10^9/8188 = 122{,}129$ samples per
rung, and it drew in **36.8 s** on this machine ($2.7\times10^7$ steps/s, the pilot's
own clock). That is the pilot's *starting* budget; autopilot doubles it until its gates
pass.

```bash
# the demo -- seconds
python3 src/generate/generate.py -meta experiments/11_erw/recipes/samples_example.json --tag demo
python3 src/report/plot_loglog.py -data experiments/11_erw/data/demo --estimates

# E1
python3 src/estimate/measure_cost.py -meta experiments/11_erw/recipes/cost_probe.json --tag cost

# one study per arm: pilot -> plan -> run -> report. --time is the TOTAL
python3 src/study/autopilot.py -meta experiments/11_erw/recipes/samples_calib_p0.5.json      --study calib_p0.5  --time 20m
python3 src/study/autopilot.py -meta experiments/11_erw/recipes/samples_diffusive_p0.6.json  --study diff_p0.6   --time 20m
python3 src/study/autopilot.py -meta experiments/11_erw/recipes/samples_critical_p0.75.json  --study crit_p0.75  --time 20m
python3 src/study/autopilot.py -meta experiments/11_erw/recipes/samples_super_p0.9.json      --study super_p0.9  --time 20m
```

`20m` is a placeholder, not a decision.

### Smoke runs — checks that the path works, not evidence

- The demo returns $\hat\gamma=0.8024$ (all points, `mle` the same) against $2p-1=0.8$;
  the report says, correctly, that it has no truth to score it against.
- One pilot replicate of `samples_super_p0.9.json`, into a scratch directory:
  $\gamma=0.8011$, $\omega_1=0.68$, $a_1=-0.246$, cv $0.466$ — one replicate, so no
  standard errors, and the pilot says so.

## Open

- **The $p$ grid, the ladder, the time per arm** — the user's call, deliberately left
  open. $p\le1/2$ is not redundant here: unlike `rwre` there is no $p\leftrightarrow1-p$
  symmetry, and $p<1/2$ is anti-correlated.
- **An exact bias reference.** $\mathbb E\lvert S_k\rvert$ is computable exactly on any
  ladder (the DP in `tools/tests/test_erw.py`, $O(k^2)$), so eq. (523)–(526)'s estimator
  can be evaluated at $n=\infty$: its finite-size bias alone, with no noise. That
  separates the two error sources the Wilson interval (eq. 720) adds together, and
  would score $\mathcal B_{\rm fs}$ directly — at $p=3/4$ as well, where there is no
  $\omega_1$ for the interval to use.
