# 02_rwre — Random Walk in Random Environment

**Status: model built and unit-tested; measurements in progress.** PLAN.md's ladder
step 3 — the first rung where the exponent being measured is *not known*, by anyone.

## The model

The environment is a one-dimensional simple symmetric exclusion process at density
$\alpha$ (the density gets its own letter because $\rho$ is the geometric scale ratio
everywhere in this repo). Above it walks a walker whose local drift depends on what it
is standing on:

$$P(\text{left}\mid\text{particle}) = p,\qquad P(\text{left}\mid\text{hole}) = 1-p,$$

and the observable is $Y_k=\lvert X_k\rvert$ after $k$ walker steps — the same
observable as `01_srw`. At $p=1/2$ the two rules coincide, the environment cannot be
read at all, and the walk **is** a simple random walk; that is what makes `01_srw` a
literal control arm rather than a loose analogy.

`models/rwre.py`, registered as `MODELS["rwre"]`. Parameters: `p`, `alpha` (0.5),
`env_sweeps` (4), `swap_prob` (0.5), `window_c` (12.0).

### Four things the phrase "particles move to adjacent empty sites" does not say

Each of these moves the answer, so each is a stated decision (2026-09-15) rather than an
implementation detail.

**(a) The environment update.** A brick-wall parity sweep: pick a parity $b\in\{0,1\}$,
and every bond of that parity swaps its two sites independently with probability
`swap_prob`. Because the swap is applied to the *sites* and not to the particles, this
is the **stirring construction** — a random permutation of coordinates chosen
independently of the configuration — from which two closed forms follow:

- product Bernoulli$(\alpha)$ is **exactly** invariant, at every time and for every
  $\alpha$, so the environment's law is a thing a test can assert rather than a
  convergence a test has to hope for;
- the particle count is conserved **per sample**, which a broken wrap bond would break
  while still conserving the total.

Observing only occupancies, stirring *is* the SSEP. The parity is drawn per sample:
sharing it across the block would correlate rows that ground rule 2 requires to be
i.i.d., and making it deterministic ($b=t \bmod 2$) would be worse than either, since
the walker's index parity is also fixed by $t$ and the walker would sit on the left
endpoint of its own active bond at every step.

**(b) The order within a step.** Read $\eta_t(X_t)$, jump, *then* stir. No swap happens
"during" the walker's own jump.

**(c) The rate.** `env_sweeps` sweeps per walker step. A bond is picked when its parity
is, so the expected swap attempts per bond per unit time is

$$\gamma_{\rm eff} = \frac{\texttt{env\_sweeps}\cdot\texttt{swap\_prob}}{2},$$

which is exactly the $\gamma$ of the literature below (walk at rate 1, SSE generator at
rate $\gamma$ per bond). The default $(4,\,1/2)$ gives $\gamma_{\rm eff}=1$: walk and
environment at the same rate. `env_sweeps = 0` freezes the environment and gives Sinai's
RWRE.

**(d) The window.** Periodic, of width $W(k)=\texttt{window\_c}\cdot\lceil\sqrt k\rceil$
rounded up to even, with the walker starting at the centre. The walk moves $\sim\sqrt k$
and the SSEP transports information diffusively, so the probability that anything wraps
is bounded by $2e^{-c^2/8}\approx1.5\times10^{-8}$ at the default $c=12$ — small enough
that no $\sqrt{\log k}$ factor is needed, which matters because it leaves

$$\mathrm{cost}(k) = k\,W(k) = c\,k^{3/2}$$

an **exact** power law: Assumption 7 (eq. 353) holds here with $d=3/2$ on the nose, the
first non-integer $d$ in this repo and the third that is known rather than fitted. $W$ is
even so that both parities are perfect matchings of the periodic window.

Every sample is **annealed**: a fresh environment and a fresh walk, as ground rule 2
requires. One environment carrying many walks is the quenched measure — a different
experiment, and not one `models/rwre.py` can be talked into.

## Two exact symmetries, derived rather than assumed

Particle–hole ($\eta\to1-\eta$) preserves Bernoulli$(1/2)$ and the dynamics, and maps
rule $p$ to rule $1-p$ at fixed $X$. Reflection ($x\to-x$) also maps rule $p$ to rule
$1-p$, but with $X\to-X$. Composing them:

$$X_k \overset{d}{=} -X_k \quad\text{for every } p \qquad\Longrightarrow\qquad \mathbb E X_k = 0 \text{ exactly},$$

$$\lvert X_k\rvert \text{ has the same law at } p \text{ and at } 1-p.$$

The first is the environment's zero-check: a drift at several standard errors is a bug,
not physics. The second halves the sieve — $p\in[0,1/2]$ carries all the information.
Both are tested (`tools/tests/test_rwre.py`).

## What is known, and what is not

- **Hilário, Kious, Teixeira**, *Random walk on the simple symmetric exclusion process*,
  [arXiv:1906.03167](https://arxiv.org/abs/1906.03167), Comm. Math. Phys. 379 (2020).
  This model exactly. They prove an LLN for all densities except at most two, and a
  functional CLT **only where the speed is non-zero**. For the special case of density
  $1/2$ with the jump distributions on a particle and on a hole symmetric to each other
  — *our* case — they prove an LLN with **zero** limiting speed. No CLT there. **The
  fluctuation exponent at $\alpha=1/2$ is open.**
- **Avena, Thomann**, *Continuity and anomalous fluctuations in random walks in dynamic
  random environments: numerics, phase diagrams and conjectures*,
  [arXiv:1201.2890](https://arxiv.org/abs/1201.2890), J. Stat. Phys. 147 (2012)
  1041–1067. Simulations of the same model; their eq. (1.5) is the rule above with their
  $p=P(\text{right}\mid\text{particle})$, so **our $p=1/3$ is their $p=2/3$**. Their
  Conjecture 3.5: for the SSE there exist $\tilde\gamma_1<\tilde\gamma_2$, both
  increasing in $p$, with the walk sub-diffusive for $\gamma<\tilde\gamma_1$,
  super-diffusive for $\tilde\gamma_1<\gamma<\tilde\gamma_2$ and diffusive above. This is
  the conjectured critical parameter; at fixed environment rate it becomes a critical
  $p$. Their Figure 17 tabulates measured exponents at $\rho=0.5$ over a $(p,\gamma)$
  grid.
- **Their estimator is $\alpha(n)=\log(\mathrm{SD}_n)/\log n$ at the largest $n$.** They
  state its $\log c/\log n$ bias themselves and do not remove it — which is precisely
  what the weighted estimator (eq. 523–526) plus a fitted correction-to-scaling exists to
  do. Our ladder, budget and sieve grid were fixed **without** reference to their grid
  (user's decision: measure cold, compare afterwards), and their numbers are read only
  once ours exist.
- **Not** Kesten–Spitzer. A static i.i.d. environment of this kind is **Sinai's RWRE** —
  recurrent at density $1/2$ with $X_t\sim(\log t)^2$ (Avena–Thomann §2.1–2.2). Kesten–
  Spitzer $k^{3/4}$ is random *scenery*, a different model. This correction matters
  because it is the `env_sweeps = 0` limit of our own parameter.

## Acceptance criteria

Numbers, not adjectives (ground rule 1). A2 must pass before A5 is quoted, and A5 before
the sieve is spent.

| | criterion | status |
|---|---|---|
| **A1** | cost: the measured exponent recovers the declared $d=3/2$ | **PASS** (below) |
| **A2** | calibration at $p=1/2$: $\hat\gamma=1/2$, $a_0=\sqrt{2/\pi}$, $\omega_1=1$, $a_1=-1/4$ each inside their se, plus a two-sample KS against `srw` draws at matched $k$ | in progress |
| **A3** | zero-check: $\lvert\mathbb E X_k\rvert<3$ se at every scale | **PASS** (unit tests) |
| **A4** | window: doubling `window_c` from 12 to 24 moves $\hat\gamma$ by less than 1 se | pending |
| **A5** | headline at $p=1/3$: $\hat\gamma$ with the eq. (720) Wilson interval and the replicate interval, and a verdict on $\gamma=1/2$ against $\gamma>1/2$ | in progress |
| **A6** | sieve: $\hat\gamma(p)$ over $p\in\{0,0.1,0.2,0.3,1/3,0.4,0.45,0.5\}$, one independent study per point | pending |

$D$ is reported only where $\hat\gamma$ is consistent with $1/2$, and then only as a
**secondary, flagged** number from the $a_0$ of the eq. (232) fit: on `01_srw` that same
fit returns $a_0=0.6703$ against the exact $\sqrt{2/\pi}=0.7979$, the missing factor
hiding in $\exp(a_1 i^{-\omega_1})$, so no decimals are claimed. An amplitude estimator
with an honest standard error would mean inverting `gamma_mle`'s Hessian in
`tools/loglog.py`; that is a separate task, deliberately not done here.

## A1 — cost model: is $\mathrm{cost}(k)=k^d$ with $d=3/2$?

`recipes/cost_probe.json`, run into `data/cost/`. Two regimes, and the difference between
them is the point:

| measurement | $\hat d$ |
|---|---|
| probe, pure power law, $k=128..16384$ | 1.1315 |
| probe, affine $a+b\,k^d$ | $1.1950\pm0.0301$ |
| **amortized, $n=64$ per scale, $k=512..4096$** | **1.4926** |
| declared `cost_hint` | 1.4928 |

The probe times `simulate(i, n=1)`, and Assumption 7 does define $\mathrm{cost}(i)$ as
the cost of *one* sample — but `rwre` is vectorized over samples, so at $n=1$ every numpy
call runs on a $(1,W)$ array and per-call overhead, which grows only weakly in $W$, is
most of the measurement. The probe's own drop-leading ladder says so: $\hat d$ climbs
monotonically $1.13\to1.28$ as the smallest scales are dropped, instead of sitting still.
Re-timed at $n=64$ — the regime a run actually uses — and dropping the two smallest $k$
where the same overhead still bites, the exponent is $1.4926$ against the declared
$1.4928$, i.e. **exact to four digits**. Measured throughput: $3.0\times10^7$ cost units
per second on this machine.

This is a real caveat for the *planner*, not for the estimator: the allocation uses the
declared value (which is right), and `measure_cost.py`'s cross-check is warning about a
regime the run never enters.

## Results

*(A2, A4, A5, A6 land here as they are measured.)*

## Open

- The amplitude estimator with a standard error (the prompt's second checkpoint): needs
  `tools/loglog.py` to invert the Hessian `gamma_mle` already builds, and `01_srw` — where
  both $a_0$ and $\gamma$ are known exactly — as the fixture. Its own prompt.
- The sieve in $\gamma_{\rm eff}$ rather than in $p$, which is the axis Conjecture 3.5 is
  actually stated in. `env_sweeps` is a recipe parameter precisely so that this costs no
  new code.
- A lazy environment update (only near the walker) would drop $d$ below $3/2$, but
  correctness needs an argument. The honest $k^{3/2}$ version comes first.
- Cross-check against `../../critical_exponents/estimators/log_log_plot.py`, which
  implements the same article estimator. Compare, do not import.
