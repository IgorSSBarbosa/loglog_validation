# Experiment 45 — the literature's common practice for the elephant random walk, with the log-log estimator

Part of `plans/literature_fss_protocols.md` (arm B, user, 2026-10-26). **Status: arm B
done (2026-10-26, xingu, 500 replicates per point); arm A's design open.** Where the
exponent is a plain power law, the log-log regression on this allocation covers
94.4–95.2%; the single-point estimator is off by its amplitude term, up to 0.052; at
the critical $p=3/4$ everything is confidently wrong.

## Question

The ERW's exponent is proven: $\gamma=1/2$ for $p<3/4$, $\gamma=2p-1$ for $p>3/4$, and at
$p=3/4$ the variance is $t\log t$. So any protocol's coverage can be measured exactly.
No ERW paper found states a complete simulation protocol: Schütz & Trimper 2004 and
Kumar, Harbola & Lindenberg 2010 are analytical, and Cressoni, da Silva & Viswanathan
2007 (cond-mat/0611477, the memory-loss generalisation, which is the ERW at $f=1$)
state only "10³ realizations", with their time range shown in a figure inset.

So this experiment tests the **practice**, labelled as such and not attributed to one
paper: $M=10^3$ walks per time (Cressoni et al.) on Avena & Thomann 2012's time grid
$n=2^{13}..2^{16}$ (experiment 44). Same three estimates as experiment 44: `full`,
`window` (each with the finite-m standard error and $\pm1.96$ se coverage) and
`single_point` (no error bar).

## Points

| recipe | $p$ | truth | regime |
|---|---|---|---|
| `protocol_practice_p0.25.json` | 0.25 | 1/2 | diffusive |
| `protocol_practice_p0.6.json` | 0.6 | 1/2 | diffusive, slow correction ($\omega=\lvert3-4p\rvert/2=0.3$, experiment 17) |
| `protocol_practice_p0.75.json` | 0.75 | 1/2 with a $\sqrt{\log t}$ factor | critical: Assumption 1 fails |
| `protocol_practice_p0.9.json` | 0.9 | 0.8 | superdiffusive |

Model `erw_urn_gpu`. $R=500$ replicates per point, under a second each.

## Known values — acceptance criteria, not inputs (ground rule 4)

The truths in the table (Bercu 2018; see `experiments/11_erw/README.md`).

## Commands

```bash
bash experiments/44_lit_avena_thomann2012_rwre/run_arm_b.sh   # runs 44 then 45
```

Run 2026-10-26 on xingu, seeds 202610264500–3, 14 s per point. Logs:
`data/practice_*.log`; artifacts: `data/practice_*/protocol_arm.json`.

## Result — arm B (2026-10-26)

$R=500$ independent replicates per point. Coverage is of $\hat\gamma\pm1.96$ se.

| $p$ | truth | `full`: bias / RMSE / coverage | `window`: bias / RMSE / coverage | `single_point`: bias / RMSE |
|---|---|---|---|---|
| 0.25 | 1/2 | −0.0010 / 0.0161 / **472/500 = 94.4%** | −0.0009 / 0.0211 / 445/488 = 91.2% | **−0.0518** / 0.0518 |
| 0.6 | 1/2 | −0.0001 / 0.0155 / **473/500 = 94.6%** | +0.0009 / 0.0207 / 457/494 = 92.5% | +0.0025 / 0.0033 |
| 0.9 | 0.8 | +0.0007 / 0.0108 / **475/500 = 95.0%** | +0.0007 / 0.0152 / 444/485 = 91.5% | +0.0175 / 0.0176 |
| 0.75 | 1/2 (+ log) | **+0.0469** / 0.0493 / **58/500 = 11.6%** | +0.0459 / 0.0502 / 93/490 = 19.0% | +0.0910 / 0.0910 |

Readings:

1. **Power-law regimes: the regression is unbiased and covers at the nominal rate**
   ($|{\rm bias}|\le0.001$, coverage 94.4–95.0%, finite-m se within 4% of the observed
   sd). On this budget the corrections are below the noise, even at $p=0.6$ where
   $\omega=0.3$.
2. **The single-point estimator's bias is its amplitude term, in closed form.** For
   $p<3/4$, $\mathbb E|S_n|\to c\sqrt n$ with $c=\sqrt{2/(\pi(3-4p))}$, so the bias is
   $\log c/\log n$ at $n=2^{16}$: $-0.0516$ at $p=0.25$ (measured $-0.0518$) and $+0.0027$
   at $p=0.6$ (measured $+0.0025$). It is small only when $c$ happens to be near 1.
3. **The $\chi^2$ window lowers coverage to 91–92% and raises RMSE by about 35%,** as in
   experiment 44: post-selection with nothing to select.
4. **At $p=3/4$ Assumption 1 fails and every estimator is confidently wrong.**
   $\mathbb E|S_n|\propto\sqrt{n\log n}$ gives a local slope
   $\tfrac12+1/(2\ln n)\approx0.550$ on this grid, so the regression reads $+0.047$ high
   with an se of 0.015, and covers 11.6%. The $\chi^2$ rule does not flag it: the
   log correction is too smooth to bend four points by more than their noise. Same
   lesson as percolation at $d=6$ (experiment 31).
