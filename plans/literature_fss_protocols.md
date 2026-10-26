# Plan — published finite-size-scaling protocols as baselines for the allocation rule

Status: **literature search done (2026-10-26); comparison experiment proposed, not
started. Decisions D1–D4 below are open.**

Question (user, 2026-10-26): how do published percolation studies spread their compute
over system sizes and turn it into an exponent with an error bar, and can we race those
protocols against `autopilot.py` at equal wall clock?

Every number below was read from the paper's own text (arXiv PDF → `pdftotext`), except
where marked *unverified*.

## 1. What the literature does

### 1a. The protocols behind our own acceptance values

| our value | paper | model, observable | sizes and samples | fit and error bar |
|---|---|---|---|---|
| $d_f(3)=2.52293(10)$ | Xu, Wang, Lv, Deng, *Front. Phys.* 9, 113 (2014), arXiv:1310.5399 | 8 lattices (bond and site: sc, bcc, fcc, diamond), periodic; $y_h$ from the largest cluster $C_1$ | $L\le512$; $\approx5\cdot10^8$ samples per $L\le128$, $6\cdot10^7$ at 256, $3\cdot10^7$ at 512 | simultaneous fit across the 8 models with shared universal parameters; $L\ge L_{\min}$ chosen by $\chi^2/\text{DF}\sim O(1)$ |
| (same exponent, earlier) | Wang, Zhou, Zhang, Garoni, Deng, *PRE* 87, 052107 (2013), arXiv:1302.0421 | sc bond and site, periodic, $C_1$ | $L=8,\dots,512$ (10 sizes), $\ge2.5\cdot10^7$ samples each | $y_h=2.52295(15)$; "smallest $L_{\min}$ for which $\chi^2/\text{DF}$ is $O(1)$ and for which further increases in $L_{\min}$ do not cause $\chi^2$ to drop by vastly more than one unit per DF"; $y_i$ free makes fits unstable, so fits use a fixed $b_2L^{-2}$ term |
| $d_f(4)=3.0446(7)$, $d_f(5)=3.5260(14)$ | Zhang, Hou, Fang, Hu, Deng, arXiv:2004.11289 (2020) | **bond** percolation, periodic hypercubes, $C_1$ and leaf-free clusters | 4D: $L=4..192$ (17 sizes), $>10^6$ samples per $L\le64$, $\ge10^5$ above. 5D: $L=4..60$ (13 sizes), $>10^6$ per $L\le20$, $\ge10^5$ above | ansatz $L^{y}(a_0+a_1L^{y_1})$, $y_1$ first free then fixed near its fitted value; same $L_m$/$\chi^2$ rule; error = statistical + spread over $L_m$ and $y_1$ |
| $\tau(4)=2.3142(5)$ | Mertens, Moore, *PRE* 98, 022120 (2018), arXiv:1806.08067 | site and bond on $\mathbb Z^d$, single clusters grown at $p_c$ (Leath / invasion), no box | $T=10^8$ clusters ($10^9$ for site, $d=5,6,7$), capped at $s_{\max}=2^{20}$, binned at $s=\lfloor2^{k/4}\rfloor$, first 30 bins ($s\le180$) discarded | nonlinear LS fit of $\log Q(s)=-(\tau-2)\log s+\log(1+as^{-\Omega})+c$ (four free parameters, the same form as our eq. 232 fit); 90% bootstrap interval, 1000 resamples, **statistical only**. They note $\Omega$ "depends on how many initial bins we discard". Their $\tau(5)=2.419(1)$ is not the 2.4175(2) in experiment 34's README |
| — (4D site, classic) | Ballesteros et al., *PLB* 400, 346 (1997), hep-lat/9612024 | 4D site, periodic, quotient method at fixed $\xi/L$ | $L=8,12,16,24,32,48$, $10^6$ samples each | fit with $L\ge L_{\min}$, refit without the smallest $L$; central value from the first fit, error bar from the second, plus a separate error for $\omega$ moving within its error bar |

### 1b. The pattern

All five follow the same recipe, and none of them derives it:

1. **Allocation.** Roughly equal $n$ per size, stepped down by 10× for the largest sizes
   when they become expensive (Zhang 2020, Xu 2014). Nobody balances bias against
   variance when choosing the sizes or $n$.
2. **Estimator.** A nonlinear fit of a power law plus one or two correction terms. The
   correction exponent is free, or fixed when the free fit is unstable (Wang 2013,
   Zhang 2020).
3. **Systematic error.** Handled by raising $L_{\min}$ until $\chi^2/\text{DF}\lesssim1$
   and stable, then quoting the spread of the fit results over $L_{\min}$ and $\omega$
   as a systematic error. This is a heuristic. It has no coverage statement, and
   Mertens & Moore quote no systematic error at all.

### 1c. Prior work on allocating time over sizes

- **"Optimization of Monte Carlo simulations",** *Physica A* 182, 649–671 (1992). The
  author is probably H.-O. Heuer (*unverified*; ScienceDirect returns 403, abstract only
  via search). The abstract says the optimal distribution of computing time over sizes
  depends on $\delta=d+z-x$ ($x$ the scaling exponent of the relative variance), saving
  60–80% against equal relative weight. It is the closest prior to Prop. (932) and
  eqs. (945)–(946). It optimises variance only. Whether it handles the
  finite-size-bias/variance trade-off, as Prop. (932) does, cannot be checked without the
  full text. **Needs library access.** A follow-up, "On the optimization of Monte-Carlo
  simulations" (*Physica A*, 1997, S0378437197003543), is also unread.
- Not relevant despite the title: Talbot, Tarjus, Viot, "Optimum Monte Carlo
  simulations: some exact results" (cond-mat/0302366), which is about Metropolis
  acceptance rates.

## 2. What reproducing them would cost here

Each budget converted to this repo's unit, $\sum_L n_L L^d$ sites, against the throughput
of our own planned runs (27, 29, 30: $3.9$–$4.2\cdot10^9$ sites/s on one RTX 5090):

| protocol | sites | vs. our run | our GPU time | feasible with the current sampler? |
|---|---|---|---|---|
| Xu 2014, $d=3$, one model of eight | $6.3\cdot10^{15}$ | 42× exp. 27 | ≈ 410 h | no: $L=512$ is fine but the budget is 17 days |
| Wang 2013, $d=3$, one model | $3.8\cdot10^{15}$ | 26× exp. 27 | ≈ 250 h | no, same reason |
| Zhang 2020, $d=4$ | $2.9\cdot10^{14}$ | 20× exp. 29 | ≈ 19 h | **no**: $L$ up to 192, past the int32 cap of 128 |
| Zhang 2020, $d=5$ | $1.2\cdot10^{14}$ | 10× exp. 30 | ≈ 9 h | **yes**: $L\le60<64$ |
| Ballesteros 1997, $d=4$ | $6.8\cdot10^{12}$ | 0.5× exp. 29 | ≈ 0.5 h | yes |

The published wall clocks are not reported in any of these papers, so "measure the
time they took" can only mean timing their protocol on our sampler. The hardware,
algorithms (Leath, Newman–Ziff, union-find on CPUs) and models (bond, other lattices)
differ too much for their own timings to mean anything here.

## 3. Proposed experiment: protocols as arms, equal wall clock

The comparison that isolates the *method* uses the same sampler, observable, hardware and
budget, and changes only the allocation and the estimator.

**Arms.** Each gets the same wall-clock budget $T$ and returns an estimate with *its own*
error bar.

| arm | allocation | estimator | quoted error |
|---|---|---|---|
| A. `autopilot.py` | pilot → tuned $(n,m_0)$, eqs. (945)–(946) with the constant restored | eq. (523)–(531) on the window | eq. (720) |
| B. Deng-group recipe (Wang 2013, Zhang 2020) | the published size list, $n_L$ in the published *ratios*, scaled to $T$ | fit $L^{y}(a_0+a_1L^{y_1})$, $y_1$ free then fixed; $L_m$ by the $\chi^2$ rule | statistical + spread over $L_m$, $y_1$ |
| C. Ballesteros recipe | equal $n$ per $L$, published sizes | fit with $L\ge L_{\min}$ and $L\ge L_{\min}+1$ | error bar of the second fit |
| D. naive | equal $n$ per $L$ on the full ladder | OLS slope on all sizes | OLS standard error |

**Scoring.** For each arm: RMSE over $R$ replicates, and coverage of its own quoted
interval (Wilson CI), exactly as `calibration/score_theorems.py` scores eq. (720).

**Where.**
1. $d=2$, $d_f=91/48$ exact, `percolation2d_gpu`, south anchor. Short $T$ (≈10 min)
   and $R=50$ per arm, so ≈35 h for four arms. The answer is exact, so coverage is
   measured, not estimated.
2. $d=3$ `face_far`, the same with $R=20$, scored against 2.52293(10).
3. Reproduction check: arm B with Zhang 2020's $d=5$ size list at its full budget
   (≈9 h, once). This tests that our implementation of their recipe recovers their
   $3.5260(14)$ within errors before it is trusted as a baseline. Caveat: they used
   bond percolation and $C_1$, while we use site percolation and `face_far`. The
   exponent is universal, the corrections are not.

**Expected outcome, stated before running.** A and B should have similar RMSE: the
theorem is a rate result, and B's allocation is not far from equal-$n$. The difference
should be in coverage. B's $\chi^2$-rule errors are heuristic, and experiment 29
already shows a replicate interval excluding the truth at $d=4$. If B under-covers
where A covers, that is the result for the paper. If B covers with a tighter interval,
eq. (720) is too conservative and that is the result instead.

## 4. Decisions (open)

| decision | proposed |
|---|---|
| D1 — run the arms on our sampler and observable, not reproduce the papers' codes | yes: the only comparison that isolates the method |
| D2 — implement arm B's $\chi^2$/$L_m$ rule as a function in `tools/` with unit tests, citing Wang 2013 §III for the rule's wording | yes (ground rule 3: the rule is taken from the paper, not invented) |
| D3 — $d=5$ reproduction check (≈9 h GPU) | yes, before trusting arm B |
| D4 — get the full text of the 1992 *Physica A* paper (library access) | needed before the paper claims novelty for Prop. (932) |

## Sources

- [Xu, Wang, Lv, Deng 2014, arXiv:1310.5399](https://arxiv.org/abs/1310.5399)
- [Wang, Zhou, Zhang, Garoni, Deng 2013, arXiv:1302.0421](https://arxiv.org/abs/1302.0421)
- [Zhang, Hou, Fang, Hu, Deng 2020, arXiv:2004.11289](https://arxiv.org/abs/2004.11289)
- [Mertens, Moore 2018, arXiv:1806.08067](https://arxiv.org/abs/1806.08067)
- [Ballesteros et al. 1997, hep-lat/9612024](https://arxiv.org/abs/hep-lat/9612024)
- ["Optimization of Monte Carlo simulations", Physica A 182 (1992)](https://www.sciencedirect.com/science/article/abs/pii/037843719290029P) (abstract only)
- ["On the optimization of Monte-Carlo simulations", Physica A (1997)](https://www.sciencedirect.com/science/article/abs/pii/S0378437197003543) (unread)
