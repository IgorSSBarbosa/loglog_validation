# 03_percolation_zd — Site percolation on $\mathbb Z^d$

**Status: $d=2$ in progress — P1 (cost model), P3 (side vs origin) and P4 (cylinder vs
box) pass; P2's $\hat d_f$ is measured but its $\omega_1$ is open. $d\ge3$ not started.**

Simulator: `models/percolation2d.py`, registered as `MODELS["percolation2d"]`.
Driven by the shared scripts (`src/generate/generate.py`,
`src/estimate/measure_cost.py`, `src/estimate/estimate_omega1.py`,
`src/estimate/compare_observables.py`, `src/report/plot_loglog.py`), so this folder holds
only recipes, this README, `data/` (gitignored) and `images/`.

## The observable

One sample at scale $i$: fill an $i\times i$ box with i.i.d. Bernoulli($p_c$) open sites
and count the open sites reachable from the **south side** of the box by open
nearest-neighbour ($4$-connected) paths that stay inside the box,

$$Y_i \;=\; \#\{x\in[0,i)^2 : x \text{ open and } x \leftrightarrow \text{row }0 \text{ inside the box}\}.$$

$p_c = 0.59274605079210$ (Jacobsen 2015) for site percolation on the square lattice with
$4$-connectivity — **not** the $8$-connected value $0.4073$, which the same code with the
wrong `structure` would silently simulate.

This is `PLAN.md` ground rule 7's observable, and this is the first place it exists in
code. The previous attempt (`presentation18-05-2026/coding`) anchored $V(r)$ at the
**origin** instead; that variant is still reachable here as `params["anchor"] = "origin"`,
but only so the two can be run head to head (Experiment P3 below). It is not the default.

### Why the exponent is the same for both anchors

A site at height $h$ above the south side is connected to it with probability of the
order of the bulk one-arm probability $\pi_1(h)\asymp h^{-5/48}$ (reach distance $h$, then
hit a full line with conditional probability bounded below, by RSW). So

$$\mathbb{E}Y_i \;\asymp\; \sum_{h=1}^{i} i\cdot h^{-5/48} \;\asymp\; i\cdot i^{43/48} \;=\; i^{91/48},$$

and the sum is dominated by $h\sim i$ — deep in the bulk, so the exponent is the bulk
$d_f$, not a surface exponent. The origin anchor gives $\mathbb{E}|C(0)\cap B_i| \asymp i^2\pi_1(i) = i^{91/48}$
by the same exponent. **Both anchors estimate the same $\gamma$; the difference is
entirely in the bias and the noise.**

### Why the side anchor should converge faster (the hypothesis)

Article Assumption 6 (eq. 332) requires $\sigma_k^2 = \mathrm{Var}(\xi_k)\to\sigma_\infty^2$,
and the CLT (eq. 583) and Wilson interval (eq. 720) rest on it.

- **origin**: $Y_i = 0$ unless the centre reaches distance $\sim i$, probability $\pi_1(i)\asymp i^{-5/48}$;
  conditionally $Y_i$ is of order $i^{91/48}/\pi_1(i)$, so
  $\mathrm{Var}(\xi_i)\asymp 1/\pi_1(i) \asymp i^{5/48}\to\infty$.
  Assumption 6 **fails** (slowly, as $\rho^{5k/48}$), and Assumption 2 ($Y_i>0$) fails on
  most draws.
- **south**: $Y_i$ sums contributions across $\sim i$ boundary columns that decorrelate
  beyond the correlation length, and is bounded below by the open sites of row $0$. Its
  coefficient of variation should be $O(1)$ and scale-free, so Assumption 6 holds; and
  $Y_i = 0$ **iff row 0 is entirely closed**, exactly probability $(1-p_c)^i$
  ($1.3\times10^{-7}$ at $i=32$) — exponentially small, hence invisible to a power-law
  expansion. `models/percolation2d.py`'s `zero_rate` is that exact formula.

## Known values — acceptance criteria, not inputs

| object | value | source |
|---|---|---|
| $\gamma = d_f$ | $91/48 = 1.8958333\ldots$ | den Nijs / Nienhuis; Stauffer & Aharony |
| $p_c$ | $0.59274605079210(2)$ | Jacobsen (2015) |
| $d$ (cost exponent) | $2$ exactly | $i^2$ sites drawn, one union-find pass |
| $\omega_1$ | **not assumed** | literature $\Omega = 72/91\approx0.791$ (Ziff 2011; Aharony & Asikainen 2003), plus an analytic correction of exponent $1$ expected from the box boundary — this rung *measures* $\omega_1$ rather than checking it |

Following the decision made for `srw` (user, 2026-08-20; `plans/three_experiment_ladder.md`
D1/D2), **none of these is wired into the code**: `MODELS["percolation2d"]` has no
`target_fn` and no `true_gamma_key`, so `src/report/plot_loglog.py` overlays no reference
curve and reports no `true_gamma`. $91/48$ enters only as a `--expect-gamma` /
`--truth` argument at *reporting* time. Do not add a `target_fn` without revisiting that.

`cost_hint(i) = i**2` is the one exception, and is not the same kind of thing: it is a
declared *cost*, an input to allocation, and never reaches an estimator of $\gamma$.

---

## Experiment P1 — cost model: is $\mathrm{cost}(i)=i^d$ with $d=2$?

This is the first rung where article Assumption 7 is a geometric claim about a real
simulator rather than a stated formula (`PLAN.md`, ladder step 4).

```bash
python3 src/estimate/measure_cost.py -meta experiments/03_percolation_zd/recipes/cost_percolation2d.json --tag cost_south
```

**Acceptance criterion:** affine $\hat d\in[1.8,2.2]$ over $i=16\ldots1024$, and the
declared-vs-measured gap under the driver's 20% tolerance.

**Result (2026-09-04, PASS).**

| fit | $\hat d$ | note |
|---|---|---|
| pure power $c\,i^d$ | $1.574$ | biased low — overhead is 89% of the measurement at $i=16$ |
| **affine $a+b\,i^d$**, box | $\mathbf{2.0285 \pm 0.0175}$ | $a = 57.0\,\mu s$, rel_rmse $2.3\times10^{-3}$ |
| **affine $a+b\,i^d$**, cylinder | $\mathbf{1.9780 \pm 0.0388}$ | $a = 103.7\,\mu s$ — the wrap-merge is in the *overhead*, not the exponent |
| declared `cost_hint` | $2.0000$ | gap $+1.63\sigma$ (box), $-0.57\sigma$ (cylinder) |

The drop-leading ladder agrees and converges from below, exactly as on `srw`:
$m_0=0\!:\!1.574 \to m_0=2\!:\!1.907 \to m_0=3\!:\!1.993 \to m_0=4\!:\!2.035$.

So the same lesson as `experiments/01_srw` transfers: **the pure-power fit is the wrong
one at small $i$**, and the affine fit recovers $d$. Here it recovers a $d$ that is a
fact about the lattice, not about a `sleep`.

---

## Experiment P2 — $\hat d_f$ from the side-connected cluster

```bash
python3 src/generate/generate.py -meta experiments/03_percolation_zd/recipes/samples_df_south.json --tag df_south
python3 src/estimate/estimate_omega1.py -data experiments/03_percolation_zd/data/df_south --expect-gamma 1.8958333333333333
python3 src/report/plot_loglog.py -data experiments/03_percolation_zd/data/df_south -o experiments/03_percolation_zd/images/df_south_loglog.png
```

Budget $4\times10^{9}$ work units (= sites), split by the `neyman` rule — which needs no
design constant at all here, since $d=2$ comes from the model's own `cost_hint`. Scales
$8,16,\dots,512$ (powers of two, one ladder, $\rho=2$); $n$ from $492\,125$ down to
$7\,689$. Wall clock 2m18s.

**Acceptance criterion:** $\hat\gamma$ within 15% of $91/48$ (the driver's default
tolerance), and the two independent $\omega_1$ estimators agreeing.

**Result (2026-09-04, PASS).**

| quantity | direct fit of eq. (232) | bias-decay fit |
|---|---|---|
| $\hat\gamma$ | $1.9161$ (+1.07% of $91/48$) | $\hat\gamma_\infty = 1.9059$ (+0.53%) |
| $\hat\omega_1$ | $0.380$ | $0.329$ |
| $\hat a_0$ | $0.395$ | — |
| $\hat a_1$ | $0.422$ | $-0.051$ (of $\hat\gamma(i)$, a different functional) |
| rel_rmse | $1.11\times10^{-3}$ | $7.66\times10^{-5}$ |

$\hat\omega_1\approx0.33$–$0.38$ is well below the literature $\Omega=72/91\approx0.79$.
That is **not** recorded here as a refutation: over a window as short as $8\le i\le512$
the fitted $\omega_1$ is an *effective* exponent absorbing whatever else varies slowly
(the analytic $i^{-1}$ boundary term above all), and the two estimators, which are
different functionals of the same means, land $0.05$ apart rather than agreeing sharply.
Settling $\omega_1$ needs a wider ladder and replicates with error bars — **open, see
`TODO.md`**.

The article's own estimator (the weighted OLS of eq. 523–531, here as
`gamma_drop_leading`) converges cleanly from below as the contaminated small scales are
dropped — this is the plainest evidence in the run:

| $m_0$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $\hat\gamma$ | 1.8804 | 1.8855 | 1.8898 | 1.8930 | **1.8974** | 1.8962 |

against $91/48 = 1.89583$; $m_0=4$ is $+0.08\%$ off. (`images/df_south_loglog.png`,
`data/df_south/gamma_estimates.json`.)

The per-scale coefficient of variation is flat, which is the Assumption 6 check, and the
amplitude $\overline Y_i/i^{91/48}$ barely drifts, which is the correction-to-scaling one:

| $i$ | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| $\mathrm{cv}(Y_i)$ | 0.476 | 0.453 | 0.437 | 0.433 | 0.436 | 0.424 | 0.429 |
| $\overline Y_i/i^{91/48}$ | 0.4989 | 0.4842 | 0.4742 | 0.4695 | 0.4651 | 0.4660 | 0.4661 |

The amplitude falls by only $7.0\%$ over six doublings and has flattened by $i=128$.

---

## Experiment P3 — side vs origin at equal budget

The question that motivated the new observable (Igor, 2026-09-04): does the change of
anchor make $\hat d_f$ converge *faster*? Two arms, identical scales, identical $n$,
identical cost per replicate, differing only in `params["anchor"]`; $R$ independent full
experiments per arm (ground rule 2 — nothing is shared between arms, replicates or
scales), one $\hat\gamma$ from each, scored as bias / sd / RMSE against $91/48$.

```bash
python3 src/estimate/compare_observables.py \
  --arm south=experiments/03_percolation_zd/recipes/samples_compare_south.json \
  --arm origin=experiments/03_percolation_zd/recipes/samples_compare_origin.json \
  --replicates 12 --truth 1.8958333333333333 --seed 20260904 --tag compare_anchors
```

**Acceptance criteria.** (a) The south arm's cv is flat in $i$ to within 15% across the
ladder, the origin arm's is not; (b) the south arm's RMSE is smaller; (c) the report
states *why* — bias or variance — rather than only that it is smaller.

**Result (2026-09-04, PASS on all three).** $R=12$ independent full experiments per arm,
budget $4\times10^{8}$ work units each ($13.6$ s measured, identical for both arms),
scales $8..512$, seed `20260904`.

| | south | origin |
|---|---|---|
| $\mathrm{cv}(Y_i)$, $i=8\to512$ | $0.477\to0.425$ | $1.006\to1.411$ |
| log-log slope of $\mathrm{cv}$ | $\mathbf{-0.024}$ | $\mathbf{+0.081}$ |
| $\hat\gamma$ (all points) | $1.8806\pm0.0024$ | $1.7599\pm0.0071$ |
| bias vs $91/48$ | $-0.0152$ | $-0.1360$ |
| RMSE | $0.0154$ | $0.1361$ |
| $\hat\gamma$ ($m_0=2$) | $\mathbf{1.8895\pm0.0040}$ | $1.7804\pm0.0124$ |
| bias, RMSE ($m_0=2$) | $-0.0063$, $0.0074$ | $-0.1154$, $0.1160$ |

**The change of observable is worth $\mathbf{8.9\times}$ in RMSE at all points and
$\mathbf{15.7\times}$ at $m_0=2$**, at identical cost. And it is worth *more* than that,
because both arms are **bias-dominated** at this budget ($|{\rm bias}|>{\rm sd}$ in
every cell) — the origin arm cannot buy its way level at any budget, so the nominal
"$78\times$ / $246\times$ the budget" figures are lower bounds on the gap, not estimates
of it. The driver prints that caveat itself.

Assumption 6, checked directly: the south arm's cv is **flat, in fact very slightly
decreasing** ($-0.024$ in log-log slope), while the origin arm's **grows** ($+0.081$).
The predicted origin growth from $\mathrm{Var}(\xi_i)\asymp1/\pi_1(i)\asymp i^{5/48}$ is
a cv slope of $5/96 = 0.052$: the sign and the order of magnitude match, the value does
not, which over a window as short as $8\le i\le512$ is expected and is not tested here.
The qualitative claim — *Assumption 6 holds for the side anchor and fails for the origin
anchor* — is what the numbers support.

Where the origin arm's bias comes from is visible directly in the matched deep runs
(`data/df_south`, `data/df_origin`, both at the full $4\times10^{9}$ budget, same seed,
same scales, same $n$). The amplitude $\overline Y_i/i^{91/48}$:

| $i$ | 8 | 16 | 32 | 64 | 128 | 256 | 512 | drop |
|---|---|---|---|---|---|---|---|---|
| south | 0.4989 | 0.4842 | 0.4742 | 0.4695 | 0.4651 | 0.4660 | 0.4661 | $\times1.070$ |
| origin | 0.2984 | 0.2595 | 0.2292 | 0.2093 | 0.1916 | 0.1806 | 0.1667 | $\times1.790$ |

The south amplitude has flattened by $i\approx128$; the origin amplitude is still falling
at $i=512$. Its drop-leading ladder shows the same thing —
$1.7593,1.7716,1.7828,1.7889,1.7955,1.7802$ — climbing but nowhere near $91/48$ by
$m_0=4$, where south is already at $1.8974$. **The origin observable simply carries a far
larger correction-to-scaling amplitude at these box sizes**: that is a statement about
the observable, not about the estimator.

### One caveat, stated because it cuts the other way

At the full budget, the *four-parameter nonlinear* fit of eq. (232) (which is not the
article's $\hat\gamma$ estimator — that is eq. 523–531's weighted OLS, used everywhere
above) recovers $\hat\gamma = 1.8926$ from the **origin** run: closer to $91/48$ than the
south run's $1.9161$. That is not evidence that the origin observable is better. The
origin fit has $5.7\times$ the relative residual ($6.33\times10^{-3}$ vs
$1.11\times10^{-3}$), and its two independent estimators disagree by $0.082$ in $\gamma$
and $0.164$ in $\omega_1$ ($\hat\gamma_\infty = 1.8104$, $\hat\omega_1 = 0.426$ from the
bias decay, against $1.8926$ and $0.262$ from the direct fit) — where the south run's two
estimators disagree by $0.010$ and $0.051$. A four-parameter fit with a large $\hat a_1$
($1.44$, against south's $0.42$) has enough freedom to absorb a big correction and land
near the truth; that it did so once, with one replicate and no error bars, is not a
measurement. The head-to-head above is run with the article's own estimators and $R=12$
replicates, which is why it is the one reported.

**Conclusion: yes, the change of experiment makes $\hat d_f$ converge faster** — and it
does so through both channels at once:

- **noise**: $\mathrm{cv}$ about $3.3\times$ smaller at $i=512$ ($0.425$ vs $1.411$), so
  $\approx11\times$ in variance — *and*, unlike the origin arm's, not growing with $i$,
  so Assumption 6 is satisfied rather than violated;
- **bias**: $8.9\times$ smaller at all points, $18\times$ smaller at $m_0=2$.

The bias channel is the larger of the two, and the one that no budget can buy back. At
$4\times10^{8}$ work units the side anchor already puts $\hat d_f$ within $0.33\%$ of
$91/48$; the origin anchor is $6\%$ off and stuck there.


---

## Experiment P4 — cylinder vs box at equal budget

Every cell of P3 was bias-dominated, so at that point *no* amount of extra budget could
improve $\hat d_f$: only reducing the correction-to-scaling term could. The box has four
walls; for a **south**-anchored count the east and west walls are pure finite-size
contamination, and making $x$ periodic (a cylinder of circumference $i$ and height $i$)
removes them. The derivation of $\gamma = 91/48$ above never used the side walls, so the
exponent is unchanged and this is again a pure bias/noise question — exactly the shape
P3 already knows how to settle.

`ndimage.label` cannot wrap, so the cylinder is labelled as a box and the labels joined
across the $x$-boundary are merged afterwards, by a vectorized pointer-jumping union-find
over *labels* (a few percent as many as there are sites). Measured overhead
**1.01×–1.07×**, falling with $i$; the cost probe above confirms it lands in the affine
model's constant $a$ and not in $d$.

```bash
python3 src/estimate/compare_observables.py \
  --arm box=experiments/03_percolation_zd/recipes/samples_geom_box.json \
  --arm cylinder=experiments/03_percolation_zd/recipes/samples_geom_cylinder.json \
  --replicates 12 --truth 1.8958333333333333 --seed 20260905 --tag compare_geometry
```

**Acceptance criteria.** (a) the cylinder's RMSE is smaller; (b) the reduction is in the
*bias*, not only the variance; (c) the cost overhead stays under 1.15× and the measured
$d$ is unchanged.

**Result (2026-09-05, PASS on all three).** $R=12$, $4\times10^{8}$ work units per
replicate, scales $8..512$, seed `20260905`.

| | box | cylinder |
|---|---|---|
| $\mathrm{cv}(Y_i)$, $i=8\to512$ | $0.476\to0.426$ | $0.435\to0.372$ |
| $\hat\gamma$ (all points) | $1.8810$, sd $0.0028$ | $\mathbf{1.8995}$, sd $0.0018$ |
| bias, RMSE | $-0.0149$, $0.0151$ | $+0.0036$, $\mathbf{0.0040}$ |
| $\hat\gamma$ ($m_0=2$) | $1.8898$, sd $0.0043$ | $\mathbf{1.8975}$, sd $0.0030$ |
| bias, RMSE ($m_0=2$) | $-0.0060$, $0.0073$ | $+0.0017$, $\mathbf{0.0033}$ |

**$3.7\times$ better in RMSE at all points, $2.2\times$ at $m_0=2$**, at $1.04\times$ the
cost — and the sign of the bias flips, which is the tell that a boundary term rather
than the estimator is responsible.

The result that matters most is not in the RMSE column. **The cylinder's $m_0=2$ cell is
the first in this project whose bias ($+0.0017$) is smaller than its spread ($0.0030$)** —
the driver stops printing `BIAS-DOMINATED` for it. Every box and origin cell so far has
been bias-limited, meaning extra budget bought nothing. On a cylinder at $m_0=2$, extra
budget starts working again.

### It is not unbiased, and the residual does not decay

Two matched deep runs at the full $4\times10^{9}$ budget (`data/df_south`,
`data/df_cylinder`), $\overline Y_i/i^{91/48}$:

| $i$ | 8 | 16 | 32 | 64 | 128 | 256 | 512 | drift |
|---|---|---|---|---|---|---|---|---|
| box | 0.4989 | 0.4842 | 0.4742 | 0.4695 | 0.4651 | 0.4660 | 0.4661 | $-6.6\%$ |
| cylinder | 0.5578 | 0.5611 | 0.5633 | 0.5640 | 0.5671 | 0.5682 | 0.5698 | $+2.1\%$ |

Relative standard errors are $0.06\%$–$0.49\%$, so the cylinder's $+2.1\%$ rise is real
($\approx5\sigma$), not noise — an earlier, cheaper probe that suggested the amplitude was
flat past $i\approx32$ simply lacked the samples to see it. The correction is about
$3\times$ smaller than the box's and of opposite sign; it is not gone.

Its signature in the estimator is a $\hat\gamma$ that is *flat and slightly high* rather
than converging. The cylinder's drop-leading ladder:

| $m_0$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| box | 1.8804 | 1.8855 | 1.8898 | 1.8930 | 1.8974 | 1.8962 |
| cylinder | 1.9008 | 1.9003 | 1.9002 | 1.9005 | 1.8992 | 1.8999 |

The box climbs toward $91/48$ from below; the cylinder sits at $\approx1.900$ and does
not move. That residual $+0.004$ ($\approx7\sigma$ against the replicate spread) is
**not** explained away by this experiment.

One consequence, worth recording because it looks like a failure and is not:
`estimate_omega1.py`'s bias-decay estimator **does not converge** on the cylinder run
($\hat\omega_1 = 10.6$, $\hat a = 1.6\times10^6$, `converged=False`), and the driver
correctly reports the two $\omega_1$ estimates as disagreeing by $7.8$. That is the
expected behaviour when there is no decay left to fit: the estimator's input is the
$\hat\gamma(m_0)$ sequence above, which is flat. The direct fit of eq. (232) still works
and gives $\hat\gamma = 1.9002$, $\hat\omega_1 = 2.80$, $\hat a_1 = -1.13$,
rel_rmse $6.7\times10^{-4}$ — but a fitted $\omega_1$ of $2.8$ on a window of six
doublings is a shape parameter, not a measurement.

**Verdict: use the cylinder.** It is $1.04\times$ the cost, $2.2$–$3.7\times$ better in
RMSE, and it moves the $m_0=2$ estimate out of the bias-limited regime. Its residual
$+0.004$ offset is now the leading error and is the thing to chase next — with a wider
ladder, which is the same prerequisite $\omega_1$ has needed since P2.

## Open

- **The cylinder's residual $+0.004$.** Now the leading error, $\approx7\sigma$, and it
  does not decay with $m_0$ over $8\le i\le512$. Needs a wider ladder to separate "a very
  slowly decaying correction" from "an amplitude effect the one-correction model cannot
  express".
- **$\omega_1$** — still unsettled on the box ($0.33$ vs $0.38$ against a literature
  $\Omega=72/91\approx0.79$) and now *un-fittable* on the cylinder, for the good reason
  above. It is what blocks a Wilson interval (eq. 720) on $\hat d_f$, since
  $\mathcal B_{\rm fs}$ needs $\omega_1,a_1$ — and $\mathcal B_{\rm fs}$ is already the
  dominant term of the interval in the autopilot run recorded in
  `data/fractaldimautopilottest/`.
- **A wider ladder** ($128\dots4096$ rather than $8\dots512$) is the prerequisite for
  both of the above. At $31.7$ ms/sample at $i=1024$ and single-threaded generation,
  that needs the fan-out below first.
- **Parallel generation.** `generate.py` is single-threaded; this machine has 12 threads
  and ground rule 2's spawned streams already make replicate-level fan-out safe and
  reproducible. This is what makes $i\ge1024$ affordable.
- **$d\ge3$**: a general-$d$ simulator, where $\mathrm{cost}(i)=i^d$ starts to bite and
  the allocation theory should earn its keep.

### Measured and rejected

- **Bit-sampling the lattice instead of uniform coupling.** The draw is only 16% of the
  runtime (labelling is 66%, the gather 17%). `rng.bytes`/`integers` at 8 or 16 bits are
  1.9–3.5× faster on the draw but **break block invariance** — numpy packs several
  sub-64-bit values per draw and discards the remainder per call, so splitting $n$ into
  blocks changes the output (the trap documented in `models/srw.py`, verified again
  here). 8-bit also cannot hold $p_c$: the threshold rounds to $152/256 = 0.59375$, a
  systematically supercritical lattice. The one block-invariant alternative, `uint32`, is
  1.33× on a 16% stage — 4% overall, not worth invalidating every recorded run.
- **BFS from the south row instead of labelling the whole lattice.** The site-count
  argument is right (the south set is only $\approx24\%$ of sites at $i=512$), but the
  only C implementation available, `ndimage.binary_propagation`, is **2.1–2.4× slower**:
  it iterates dilation to a fixed point, and at $p_c$ the chemical distance grows like
  $i^{1.13}$, so it needs hundreds of passes where union-find needs one. A real
  frontier-queue BFS would win, but it means Cython/numba and the ceiling is ~2×.
- **All four sides from one labelling.** The labelling is 66% of the cost and is shared,
  so south/north/east/west look nearly free. Variance ratio $0.68$ — but the cost is
  $1.6\times$ (four gathers), and spending that same $1.6\times$ on more independent
  lattices gives $0.625$. Strictly worse; the sides correlate at $0.55$.
- **`bincount` instead of the fancy-index gather.** $1.01$–$1.04\times$. Noise.
