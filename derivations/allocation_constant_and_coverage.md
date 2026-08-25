# Two constants that were being dropped

*The multiplicative constant in `prop:opt`'s allocation, and the quantile in our confidence intervals.*

`loglog_validation` project notes — 2026-08-24

> Typeset source: [`allocation_constant_and_coverage.tex`](allocation_constant_and_coverage.tex)
> (compile with `pdflatex` twice, for the table of contents). This Markdown copy is the
> same content, for reading without a LaTeX toolchain.

Two independent findings from Experiments B and C, both of the same shape: a constant that
a theorem legitimately discards, and that costs real accuracy when the theorem is applied
literally.

- **Part I** derives, in closed form, the multiplicative constant $\kappa$ that Proposition
  `prop:opt` drops from its budget allocation — worth a factor 8–38 in compute — and shows
  it is recoverable from measured quantities alone, with no appeal to ground truth.
- **Part II** explains why every "95%" confidence interval this project published was in
  fact an 88% interval, and derives the figure $0.8784$ that the calibration experiment
  then measured as $0.877$–$0.878$.

---

# Part I — The allocation constant $\kappa$

## 1. What is being computed, and from what

Proposition `prop:opt` (eq. 945–946) states that, given a compute budget $B$, the choice

$$n = \kappa B^{\theta_1}, \qquad m_0 = \theta_2\log_\rho B, \qquad \theta_1 = \frac{2\omega_1}{d+2\omega_1}, \qquad \theta_2 = \frac{1}{d+2\omega_1}$$

achieves the error decay $|\hat\gamma-\gamma| \lesssim B^{-\omega_1/(d+2\omega_1)}$.

It is a **rate** result: correct up to constants, and the constant multiplying $B$ inside
the logarithm is exactly what it discards. Empirically that omission costs a factor
2.0–3.4 in RMSE at every budget from $10^4$ to $10^9$; since $\mathrm{RMSE}\sim B^{-1/3}$,
a factor $c$ in RMSE is a factor $c^3$ in budget — **8 to 38 times the compute**.

The strategy below: write the three quantities that matter — bias, standard deviation, and
cost — as exact power laws in $(n,m_0)$ with explicit constants, then minimize. Only the
constants require measurement; the exponents are the paper's.

> **No ground truth is used.** The inputs are $a_1$ and $\omega_1$ (Experiment B), $d$ (the
> affine cost fit), and cv (read off the samples). Every one is a measured quantity. The
> true values for simple random walk are known, but they enter only when *scoring* a
> finished answer, never when computing one.

## 2. Bias

The article's estimator is $\hat\beta=\sum_k w_{k,m}\log\overline Y_{\rho^k}$ with
$\hat\gamma=\hat\beta/\log\rho$, on the consecutive grid $k=m_0+1,\dots,m_0+m$, weights
(eq. 526)

$$w_{k,m}=\frac{12\left(k-m_0-\frac{m+1}{2}\right)}{m(m^2-1)}.$$

Substituting the one-correction truncation of eq. (232),

$$\log\mathbb{E} Y_{\rho^k}=\log a_0+\gamma\, k\log\rho+a_1\rho^{-k\omega_1},$$

and using the weight identities of eq. (542), $\sum_k w_{k,m}=0$ and $\sum_k w_{k,m}k=1$,
the first two terms vanish identically and **only the correction survives**:

$$\mathbb{E}\hat\gamma-\gamma=\frac{a_1}{\log\rho}\sum_k w_{k,m}\rho^{-k\omega_1}.$$

Now write $k=m_0+j$ with $j=1,\dots,m$. The weights depend on $j$ alone,
$w_j=\frac{12(j-\frac{m+1}{2})}{m(m^2-1)}$, and $\rho^{-k\omega_1}=\rho^{-m_0\omega_1}\rho^{-j\omega_1}$
factors — so the dependence on $m_0$ separates completely:

$$\boxed{\ |\text{bias}|=C_b\,\rho^{-\omega_1 m_0}, \qquad C_b=\frac{\left|a_1\sum_{j=1}^{m}w_j\rho^{-j\omega_1}\right|}{\log\rho}\ }$$

This is the finite-size bias of Prop. (820), with its constant made explicit.

## 3. Standard deviation

By the delta method, $\mathrm{Var}(\log\overline Y_k)\approx \sigma_k^2/(n\mu_k^2)=\mathrm{cv}^2/n$
whenever the coefficient of variation $\mathrm{cv}=\mathrm{sd}(Y_i)/\mathbb{E}(Y_i)$ is
scale-free. (It is for $|S_k|$: measured $0.7575$–$0.8222$ across the grid, against the
half-normal limit $\sqrt{\pi/2-1}=0.7555$.) Distinct scales use independent samples, so

$$\mathrm{Var}(\hat\gamma)=\frac{\mathrm{cv}^2\|w\|^2}{n\log^2\rho}, \qquad \|w\|^2=\sum_{j=1}^m w_j^2=\frac{12}{m(m^2-1)},$$

the last equality from $\sum_j(j-\frac{m+1}{2})^2=m(m^2-1)/12$. Hence

$$\boxed{\ \mathrm{sd}=C_s\,n^{-1/2}, \qquad C_s=\frac{\mathrm{cv}\,\|w\|}{\log\rho}\ }$$

> **Consistency check.** Since $m(m^2-1)\to m^3$, this reproduces eq. (583)'s
> $\mathrm{Var}(\hat\gamma)=12\sigma_\infty^2/(nm^3\log^2\rho)$ — independent confirmation
> that $C_s$ carries the right normalization.

## 4. Cost

Under Assumption `cost_is_power_law` (eq. 353), $\mathrm{cost}(i)=i^d$, Lemma `lem:budget`
gives the geometric sum in closed form:

$$B=n\sum_{j=1}^{m}\rho^{d(m_0+j)}=n\,\rho^{dm_0}G, \qquad G=\rho^{d}\,\frac{\rho^{dm}-1}{\rho^{d}-1}.$$

## 5. The optimization

Eliminating $n=B\rho^{-dm_0}/G$, the mean squared error becomes a function of $m_0$ alone:

$$\mathrm{MSE}(m_0)=\underbrace{C_b^2\rho^{-2\omega_1 m_0}}_{\text{falls in } m_0}+\underbrace{\frac{C_s^2G}{B}\,\rho^{dm_0}}_{\text{rises in } m_0}$$

The entire tradeoff is visible here: a deeper ladder reduces finite-size bias but costs
more per replicate, hence buys fewer samples. Setting $u=m_0\log\rho$ and differentiating,

$$2\omega_1C_b^2e^{-2\omega_1u}=d\,\frac{C_s^2G}{B}e^{du} \quad\Longrightarrow\quad e^{(d+2\omega_1)u}=\frac{2\omega_1C_b^2}{d\,C_s^2G}\,B$$

and therefore

$$\boxed{\ m_0^{\star}=\theta_2\log_\rho(\kappa B)=\underbrace{\theta_2\log_\rho B}_{\texttt{prop:opt}}+\underbrace{\theta_2\log_\rho\kappa}_{\text{the dropped offset}}, \qquad \kappa=\frac{2\omega_1C_b^2}{d\,C_s^2\,G}\ }$$

**The second term does not depend on $B$.** That is precisely why a rate argument may
discard it — and precisely why the gap between `prop:opt`'s $m_0$ and the empirical optimum
is a *constant* rather than a growing discrepancy.

Substituting back, and using $1-d\theta_2=\theta_1$:

$$n^{\star}=\frac{B^{\theta_1}}{G\,\kappa^{d\theta_2}}$$

which also identifies the unnamed constant $\kappa$ appearing in `prop:opt`'s own statement.

### 5.1 Two structural corollaries

**The optimal bias-to-noise ratio.** The stationarity condition
$2\omega_1\,\text{bias}^2=d\,\mathrm{sd}^2$ rearranges to

$$\frac{|\text{bias}|}{\mathrm{sd}}\bigg|_{m_0^\star}=\sqrt{\frac{d}{2\omega_1}}=0.7071 \quad \text{for } (d,\omega_1)=(1,1),$$

**independently of $B$**. This is the sharpest available diagnostic. Measured at the
empirical argmin it reads $0.44$–$1.50$ (the scatter is grid granularity: one step of $m_0$
moves the ratio by $\rho^{\omega_1+d/2}=2.83$). Measured at `prop:opt`'s own $m_0$ it reads
$0.05$–$0.19$ — an order of magnitude below the balance its own derivation argues for.
`prop:opt` drives the bias far beneath the noise floor and pays for it in samples.

**Why the constant is forgiving.** Since $\kappa\propto C_b^2\propto a_1^2$ and
$\kappa\propto C_s^{-2}\propto \mathrm{cv}^{-2}$, and $\kappa$ enters only through a
logarithm,

$$\Delta(\text{offset})=2\theta_2\log_\rho\left|\frac{a_1'}{a_1}\right|=-2\theta_2\log_\rho\frac{\mathrm{cv}'}{\mathrm{cv}}$$

At $(d,\omega_1,\rho)=(1,1,2)$ this is $\pm 2/3$ of a step per doubling. Being one step off
costs at most $1.19\times$ in RMSE, so a factor-two error in $a_1$ is nearly free. The
exception is $\omega_1$, which enters $\theta_2$, $C_b$ and $\kappa$ simultaneously: halving
it moves the offset by $2.12$ steps. It is also the noisiest quantity we measure — the
reason Experiment B pools its replicates rather than averaging their fits.

## 6. The numbers

With $(d,\omega_1,\rho,m)=(1,1,2,6)$ and the measured $a_1=-0.2748$, $\mathrm{cv}=0.7771$:

$$w_j=(-0.14286,\ -0.08571,\ -0.02857,\ 0.02857,\ 0.08571,\ 0.14286)$$

$$\sum_j w_j=2.8\times10^{-17}, \qquad \sum_j j\,w_j=1.000000, \qquad \|w\|^2=0.057143=\tfrac{12}{6\cdot 35}$$

$$S:=\sum_j w_j\rho^{-j\omega_1}=-0.089732$$

$$C_b=\frac{|-0.2748\times(-0.089732)|}{0.693147}=0.035575, \qquad C_s=\frac{0.7771\times0.239046}{0.693147}=0.267999, \qquad G=2\cdot\frac{2^{6}-1}{2-1}=126$$

$$\kappa=\frac{2(1)(0.035575)^2}{(1)(0.267999)^2(126)}=2.7969\times10^{-4}, \qquad \text{offset}=\tfrac13\log_2\left(2.7969\times10^{-4}\right)=\mathbf{-3.9346}$$

## 7. Verification against Experiment C

Sweeping $m_0$ at each budget ($R=40$ replicates per cell, so every other $m_0$ acts as a
control arm for the one `prop:opt` names):

| $B$ | $m_0$ `prop:opt` | $m_0$ tuned | $m_0$ measured | penalty `prop:opt` | penalty tuned |
|---|---|---|---|---|---|
| $10^4$ | 4 | 0 | 0 | 3.35× | **1.00×** |
| $10^5$ | 5 | 2 | 1 | 2.59× | **1.01×** |
| $10^6$ | 6 | 3 | 2 | 2.29× | **1.00×** |
| $10^7$ | 7 | 4 | 4 | 2.02× | **1.00×** |
| $10^8$ | 8 | 5 | 4 | 2.69× | **1.02×** |
| $10^9$ | 9 | 6 | 6 | 2.26× | **1.00×** |

Penalty is RMSE relative to the best $m_0$ available on the grid. Three observations:

1. The gap $m_0^{\texttt{prop:opt}}-m_0^{\text{measured}}$ regressed on $\log_\rho B$ has
   slope $-0.05$ over five decades: **it is flat**, confirming $\theta_2$ is right and only
   the constant was missing.
2. The closed-form penalty for being $\delta$ steps too deep,
   $\sqrt{(r^2\rho^{-2\omega_1\delta}+\rho^{d\delta})/(r^2+1)}$ with $r^2=d/2\omega_1$,
   predicts $2.31\times$ at $\delta=3$ and $3.27\times$ at $\delta=4$; measured
   $2.02, 2.26$ and $3.35, 2.59, 2.29, 2.69$.
3. After tuning, the residual penalty is $\le 2\%$ even where $m_0$ is off by one, because
   MSE is flat near its minimum: at $B=10^6$ the two best cells differ by 0.03%.

The RATE half of `prop:opt` is unaffected and passes: the measured decay exponent is
$-0.3259\pm0.0116$ at the best $m_0$ against a predicted $-0.3307\pm0.0127$ — agreement to
0.28 standard errors.

---

# Part II — Why our "95%" intervals were 88% intervals

## 8. The interval as it was computed

Experiment B reports a parameter $\theta\in\{\omega_1,a_1,\gamma\}$ as

$$\hat\theta \pm 1.96\,\widehat{\mathrm{se}}, \qquad \widehat{\mathrm{se}}=\frac{s}{\sqrt R}, \qquad s^2=\frac{1}{R-1}\sum_{r=1}^{R}(\hat\theta_r-\bar\theta)^2$$

where $\hat\theta_1,\dots,\hat\theta_R$ are the fits from $R=5$ independent replicates.
(The point estimate $\hat\theta$ is the *pooled* refit rather than $\bar\theta$; see §11.1.)

## 9. The error

The multiplier $1.96$ is the standard normal quantile, appropriate when $\widehat{\mathrm{se}}$
is the **true** standard error. Here it is not: $s$ is estimated from five numbers. If
$\hat\theta_r\sim\mathcal N(\theta,\sigma^2)$ i.i.d., the exact pivot is Student's:

$$T=\frac{\bar\theta-\theta}{s/\sqrt R}\sim t_{R-1}=t_4, \qquad \text{not } \mathcal N(0,1)$$

so the true coverage of the nominally-95% interval is

$$\boxed{\ \Pr\left(|t_4|<1.9600\right)=2F_{t_4}(1.9600)-1=0.8784\ }$$

Two distinct effects push in the same direction:

- $s$ is **downward-biased**: $\mathbb{E}[s]=c_4(R)\sigma$ with
  $c_4(R)=\sqrt{\frac{2}{R-1}}\frac{\Gamma(R/2)}{\Gamma((R-1)/2)}$, and $c_4(5)=0.9400$ — a
  five-point standard deviation reads 6% low on average.
- $s$ is **noisy**, which fattens the tails of $T$ beyond the normal.

The $t_{R-1}$ quantile corrects both simultaneously.

## 10. Correct multipliers

| nominal level | normal | $t_4$ | ratio |
|---|---|---|---|
| 95% | 1.9600 | **2.7764** | 1.417 |
| 68.3% | 1.0000 | **1.1417** | 1.142 |

The widely-used "$\hat\theta\pm 1\,\mathrm{se}$" shorthand is a 68.3% claim, and suffers the
same defect: $\Pr(|t_4|<1)=2F_{t_4}(1)-1=0.6261$.

## 11. Measured coverage

`src/estimate/check_coverage.py` replays Experiment B's exact configuration (scales $8..256$, the
real per-scale $n$, $R=5$) 2000 times against the known ground truth for $|S_k|$ and counts
interval hits. Bounds on the coverage itself are Wilson score intervals.

| quantity | centre | $q$ = normal | $q = t_4$ |
|---|---|---|---|
| $\omega_1$ | pooled | **0.880** [0.866, 0.894] | 0.948 [0.937, 0.956] |
| $\omega_1$ | mean-of-fits | **0.877** | 0.946 |
| $a_1$ | pooled | **0.880** | 0.950 |
| $a_1$ | mean-of-fits | **0.878** | 0.955 |
| $\gamma$ | pooled | **0.891** | 0.952 |
| $\gamma$ | mean-of-fits | **0.878** | 0.950 |

The mean-of-fits rows match the predicted $0.8784$ **to three decimal places**, as they
must: for that centre the pivot really is $t_4$. At the 68.3% level the same agreement
holds — $0.625$ measured for $\omega_1$ against $0.6261$ predicted.

### 11.1 Two deviations, both explicable

**The pooled centre is mildly conservative.** The pooled rows read slightly *above*
$0.8784$. The pooled estimate is not $\bar\theta$, so its sampling distribution is not
exactly $t_4$; pooling the $\overline Y_i$ before fitting reduces the centre's scatter (for
$a_1$: $0.0349$ against mean-of-fits' $0.0390$) while the stated $\widehat{\mathrm{se}}$ is
unchanged. The ratio $\widehat{\mathrm{se}}/\mathrm{sd}$ improves from $0.870$ to $0.974$
and the interval becomes slightly wide rather than slightly narrow.

**Bias costs more in a narrow interval.** At the 68.3% level, $a_1$/mean-of-fits reads
$0.589$ against $0.6261$ predicted. That centre carries a residual bias of $-0.0155$ (the
nonlinearity of the fit, which pooling removes: pooled bias is $-0.0019$). A fixed bias
consumes a larger share of a $\pm1\,\mathrm{se}$ interval than of a $\pm1.96\,\mathrm{se}$ one.

## 12. Diagnosis, and what was changed

**The defect is the quantile, not the width.** The diagnostic
$\widehat{\mathrm{se}}/\mathrm{sd}(\hat\theta)$ measures $0.94$–$1.03$ for the pooled
estimates: the error bar is the right size. Nothing about the model, the estimator, or the
pooling was at fault.

Accordingly the printed $\pm$ values were left alone — a standard error is a perfectly good
thing to report — and the fix was applied where a standard error becomes a **decision**. The
consistency test in `src/report/plot_allocation.py` used $|z|<2$, valid only for an exactly known
se; here $z$ combines an analytic slope error with one derived from $\omega_1$'s replicate
error. Components are now combined by Welch–Satterthwaite,

$$\mathrm{se}^2=\sum_j\mathrm{se}_j^2, \qquad \nu_{\mathrm{eff}}=\frac{\left(\sum_j\mathrm{se}_j^2\right)^2}{\sum_j\mathrm{se}_j^4/\nu_j}$$

and the cut-off is $t_{\nu_{\mathrm{eff}}}$. For Experiment C, $\nu_{\mathrm{eff}}=16.9$ and
the threshold is $2.111$ rather than $1.960$. Both existing verdicts are unchanged — the
desirable outcome: the correction widens the threshold without overturning a conclusion.

## 13. Practical consequence

At $R$ replicates, use $t_{R-1}$. Note that $R=5$ is an expensive operating point:

| $R$ | 5 | 10 | 20 | $\infty$ |
|---|---|---|---|---|
| $t_{R-1}$ at 95% | 2.776 | 2.262 | 2.093 | 1.960 |
| excess width | +42% | +15% | +7% | — |

If replicates are cheap relative to the ladder itself, buying more of them is a better use
of budget than it appears.

Finally, **this calibration is not portable.** Sweeping the sample size shows coverage
holding to $n$-scale $\approx 0.1$ and then collapsing — not because the interval degrades
but because the estimator does: $(a_1,\omega_1)$ ceases to be identified once noise swamps
the correction, and at $n$-scale $0.003$ the fit returns $\hat a_1\approx-2\times10^{10}$.
Coverage must be re-measured before an error bar is trusted at a smaller budget, or on a
different model.

---

## Appendix — Reproducing the numbers

```bash
# Part I: constants, tuned allocation, and the m0 sweep
python3 src/budget/allocation_experiment.py \
    -meta experiments/01_srw/recipes/sweep_wide.json --tag allocation_wide
python3 src/report/plot_allocation.py -data experiments/01_srw/data/allocation_wide
python3 src/budget/allocation_table.py --compare

# Part II: coverage calibration
python3 src/estimate/check_coverage.py --arm planted  --trials 2000 --centre both
python3 src/estimate/check_coverage.py --arm planting --trials 3000 --planting-n 10000
```

The constants themselves are `tools/allocation.py`'s `allocation_constants` /
`tuned_allocation`; the calibration machinery is `tools/coverage.py`. Narrative write-ups
with the full context live in `experiments/01_srw/README.md`.
