# Experiment 50 — long runs of `rwre_gpu_lazy` at $p=0.04$ and $p=0.3$, planned on tail constants

**Status: done (2026-11-03). $p=0.04$ closed on 3 of 5 replicates (missouri went down at 07:37); $p=0.3$ complete, 5 of 5.**

## Question (user, 2026-11-02)

48's sieve gives $\hat\gamma\approx0.515$ for every $p\in[0.02,0.14]$, but its local slopes
are still falling at $2^{21}$, and refitting its long runs with a correction term gives
$0.499$–$0.509$. Is $\gamma_\infty=\tfrac12$? Two long runs, one at $p=0.04$ (no hump) and
one at $p=0.3$ (past 19's hump), to extend the ladder above $2^{21}$ and test
out-of-sample predictions made from data drawn before them.

## Design (decided with the user, 2026-11-02)

- **No pilot is drawn.** Each run reuses data already drawn (`autopilot.py --reuse-pilot`),
  with the pilot's $\omega_1$, $a_1$, cv replaced by a fit on the tail of a long ladder
  (`build_pilots.py`, as 41's `tail_constants.py`). The pilots' own $\omega_1$ are
  short-ladder values (48 at $p=0.04$: 0.401 on 4..1024; 19 at $p=0.3$: 0.009, the hump).

| | $p=0.04$ | $p=0.3$ |
|---|---|---|
| reused data | 48 `sieve_p0.04_3h_cap131072` (pilot 4..1024) | 19 `omega_p0.3_n4e6` (4..$2^{18}$, $n=800{,}000\times5$) |
| $\omega_1$ | eq. (pilot-fit) on 48's local-slope $\ln\overline Y$, $\gamma$ free, $1024..2^{21}$: **$0.2552\pm0.0232$** ($\chi^2$ 11.5/8, p = 0.18) | **borrowed from $p=0.04$** (user): past the hump the tail exponent is taken as $p$-independent, as 19's family fit has it |
| $a_1$ | same fit: $-1.038\pm0.050$ | eq. (pilot-fit) with $\omega_1$ fixed, $2^{14}..2^{18}$: $-1.96\pm0.18$ (incl. $\omega_1$'s se) |
| cv | 48's long run: 0.7417 | 19, $2^{14}..2^{18}$: 0.7338 |
| $d$, throughput | **$1.467\pm0.035$**, $2.50\times10^{11}$/s: `time_window.py` on missouri | **$1.473\pm0.035$**, $2.91\times10^{11}$/s: `time_window.py` on xingu |

- **The kernel (user, 2026-11-02): the pooled, staggered kernel** (`kernel: "pool"`,
  `models/rwre_gpu_lazy.py`, commit 5bcfd2b and later). The warps of a block share a pool of
  pages, and their first samples are delayed so that the widths, which only grow, don't peak
  together. Bit-identical to the warp kernel (every output column, debug invariants on,
  through thousands of evictions). Against the warp kernel at cap $2^{17}$, at $p=0.04$,
  n = 16384 per scale: 2.9× faster at $2^{18}$–$2^{19}$, 2.7× at $2^{20}$, 2.1× at
  $2^{22}$–$2^{23}$.
- **The cap and kernel note (user, 2026-11-02).** Reuse runs the reused pilot's own recipe,
  so each built `pilot.json` carries the run's recipe (`kernel` "pool", `cap` 393216),
  although its draws were made with the warp kernel at cap 131072 (48) or $2^{15}$ (19).
  Neither changes a draw. Recorded in every built `pilot.json` (`provenance_note`) and in
  the constants' sources.
- **$d$ and throughput from the window (2026-11-02).** The reused pilots' $d=0.97$ was timed
  on 4..1024, but on the window the local cost exponent is 1.25–1.8 (work $\sim kW(k)$).
  Every plan built on the old $d$ under-ran: 48's runs by 2.0–2.8×, the warp-kernel smoke
  runs by 3.6–4.8×. The batched probe would take hours per scale at $2^{23}$, and
  autopilot's cost re-fit skips itself here (its short run is a whole replicate). So
  `time_window.py` times one call `simulate(i, 16384)` per window scale with the run's
  kernel and params, and fits eq. (cost) in logs (`data/cost_window_p<p>/cost_probe.json`,
  seeds 202611025010 / 202611025011).
- **Pipeline:** `run_p.sh <p> 24h <seed>`: `autopilot.py --reuse-pilot data/pilot_p<p>_pool
  --time 24h --replicates 5 --max-scale 2^23 --seed <seed>` (gate, plan, cost re-fit, run,
  report, local slopes), then `46/plot_pipeline.py`. $m=6$, $\rho=2$. Seeds 202611025000
  ($p=0.04$), 202611025001 ($p=0.3$).
- **Plan previews** (2026-11-02, before launch): both $m_0=17$, window $2^{18}..2^{23}$,
  capped by `--max-scale` (uncapped the allocation would go to $m_0=20$/21);
  $n=191{,}913$ ($p=0.04$) and $205{,}263$ ($p=0.3$) per scale $\times5$, 288 min per
  replicate. From the window timing directly: 91 ms per sample over the ladder, ×191,913×5
  = 24.3 h.
- **Why $m=6$, not 3** (user asked): on a doubling ladder the top scale dominates the cost,
  so $m=3$ buys only 12% more $n$, costs the $m$-point $\hat\gamma$ about 2.8× in se, and
  pushes ~14% more draws into the local slopes, outside the budget.
- **What each run must do.** At $p=0.3$, with $\omega$ fixed the tail's $\gamma$ moves as
  $d\gamma/d\omega\approx0.23$ (19's data: 0.490, 0.502, 0.515 at $\omega$ = 0.141, 0.18,
  0.255). So $p=0.04$ must pin $\omega$ to about ±0.01 for $p=0.3$ to say anything about
  $\gamma$; $p=0.3$ also gets its own free tail fit as a check.

## Pre-registered predictions (`predict.py`, run before launch)

Local slopes $s_k=\log_2(\overline Y_{2k}/\overline Y_k)$ that eq. (pilot-fit), fitted on the
data above, implies at the long run's pairs. "free": $\gamma$ fitted ($p=0.3$: $\omega$ held
at $p=0.04$'s); "half": $\gamma=\tfrac12$ imposed, $\omega$ fitted.

| p | model | fitted | χ² p | $s$ at $2^{18}$ | $2^{19}$ | $2^{20}$ | $2^{21}$ | $2^{22}$ | $2^{23}$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.04 | free | γ = 0.5065 ± 0.0018, ω = 0.2552 ± 0.0232 | 0.18 | 0.5165 ± 0.0002 | 0.5149 ± 0.0004 | 0.5135 ± 0.0005 | 0.5124 ± 0.0006 | 0.5114 ± 0.0007 | 0.5106 ± 0.0008 |
| 0.04 | half | ω = 0.1927 ± 0.0027 | 0.023 | 0.5160 ± 0.0002 | 0.5140 ± 0.0002 | 0.5123 ± 0.0002 | 0.5107 ± 0.0002 | 0.5094 ± 0.0002 | 0.5082 ± 0.0002 |
| 0.3 | free (ω = 0.04's) | γ = 0.5145 ± 0.0016 | 0.73 | 0.5335 ± 0.0006 | 0.5304 ± 0.0008 | 0.5278 ± 0.0009 | 0.5257 ± 0.0010 | 0.5238 ± 0.0011 | 0.5223 ± 0.0012 |
| 0.3 | half (ω free) | ω = 0.1718 ± 0.0092 | 0.9 | 0.5328 ± 0.0006 | 0.5291 ± 0.0007 | 0.5259 ± 0.0008 | 0.5230 ± 0.0008 | 0.5204 ± 0.0009 | 0.5181 ± 0.0009 |

The $2^{18}$ column was added when the window moved to $2^{18}..2^{23}$ (2026-11-02, still
before anything was drawn; the other columns are unchanged). That window's pairs reach
$k=2^{22}$, so the $2^{23}$ column is not tested.

The two models differ by 0.002–0.004 at $2^{22}$–$2^{23}$. One drawn $s_k$ at the planned $n$
has se ≈ 0.0015, so no single slope decides; the test is all of the window's points
together (and $\ln\overline Y$ fits on the joined ladder).

## Results

### $p=0.04$ (`data/long_p0.04_24h`, seed 202611025000): 3 of 5 replicates

Launched 2026-11-02 18:58 on missouri. Replicates took 14,980 / 14,974 / 14,856 s (4.1 h),
0.86× the plan's 288 min. Missouri went down during replicate 4 (log stops 07:37, host
unreachable since). The user decided (2026-11-03) to report on the three finished
replicates: `run.py --close-partial` turned `final_partial.json` into `final.json`, marked
`closed_partial` with `replicates_planned` = 5. The local slopes then ran with autopilot's
own seed child (`spawn(202611025000, 4)[2]`, `local/slopes_p004.py`, on amazonas, 158 s),
then `46/plot_pipeline.py`. A resume path for replicates 4–5 can be added to `run.py` if
needed.

- **eq. (526):** $\hat\gamma = 0.5125 \pm 0.0065$, eq. (720) 95% bound $[0.5059, 0.5190]$
  (three-term form of 2026-11-04; the four-term bound was $[0.5050, 0.5200]$),
  dominated by $B_{fs}=0.0054$ ($1.96\sigma_{se}=0.0011$). Replicate interval
  $[0.5113, 0.5136]$ (t(2), scatter only). Per replicate 0.51229, 0.51213, 0.51299.
  **The bound is unreliable:** the pilot refitted with $\gamma$ pinned at $\hat\gamma$ has
  $\chi^2 = 1.5\times10^4/6$ (p = 0): 48's pilot (4..1024) is not one correction around
  0.5125. It is also incomplete (no $\omega_2$ term).
- **Pre-registered test** (`compare.py`): the window's five slopes against both models,
  with the full covariance of the drawn and the predicted slopes.

  | $k$ | $2^{18}$ | $2^{19}$ | $2^{20}$ | $2^{21}$ | $2^{22}$ | χ²/5 | p |
  |---|---|---|---|---|---|---|---|
  | drawn $s_k$ (se 0.0020) | 0.5135 | 0.5169 | 0.5116 | 0.5120 | 0.5066 | | |
  | − free ($\gamma=0.5065$) | −0.0031 | +0.0020 | −0.0019 | −0.0004 | −0.0048 | 9.8 | 0.08 |
  | − half ($\gamma=\tfrac12$) | −0.0026 | +0.0029 | −0.0006 | +0.0012 | −0.0028 | 4.5 | 0.48 |

  Neither model is rejected. $\gamma=\tfrac12$ fits better, but by Δχ² = 5.3 at 5 dof each,
  which on this noise is not decisive. Five replicates would bring each slope's se to about
  0.0015 against model differences of 0.0005–0.0020 here, so the two missing replicates
  would not decide it either.
- **Local slopes, 4..$2^{23}$:** monotone within noise down to $s_{2^{22}}=0.5066\pm0.0020$,
  below the free model's γ = 0.5065 ± 0.0018 only at the last point. One power law on the
  whole ladder, $\gamma + c\,i^{-\omega}$: γ = 0.5043, ω = 0.208 ± 0.008, but χ² 68.9/18
  (p = 7e−8), so the ladder is not one correction from 4 up. The $c/\ln i$ form is far worse
  (χ² 700/19).

### $p=0.3$ (`data/long_p0.3_24h`, seed 202611025001)

Launched 2026-11-02 18:58 on xingu, finished 2026-11-03 15:45. Five replicates of
14,790–14,889 s: 74,217 s in all, 0.86× the predicted 24 h. Autopilot's report, local
slopes (611 s) and `46/plot_pipeline.py` ran unattended.

- **eq. (526):** $\hat\gamma = 0.5248$. Replicate interval $[0.5241, 0.5255]$ (t(4)); per
  replicate 0.52503, 0.52515, 0.52406, 0.52444, 0.52533. **The eq. (720) bound is void:**
  $\pm2.72$, from $B_{fs}=2.72$ (unchanged by the three-term form of 2026-11-04). The pilot refitted with $\gamma$ pinned at 0.5248 lands on
  $\omega_1=0.059$ with $\chi^2=1.67\times10^4/14$ (p = 0). This pilot is 19's 4..$2^{17}$,
  hump included, and no single correction describes it.
- **Pre-registered test** (`compare.py`):

  | $k$ | $2^{18}$ | $2^{19}$ | $2^{20}$ | $2^{21}$ | $2^{22}$ | χ²/5 | p |
  |---|---|---|---|---|---|---|---|
  | drawn $s_k$ (se 0.0015) | 0.5298 | 0.5287 | 0.5259 | 0.5212 | 0.5172 | | |
  | − free ($\omega$ = 0.04's, $\gamma=0.5145$) | −0.0037 | −0.0017 | −0.0019 | −0.0044 | −0.0066 | 19.4 | 0.0016 |
  | − half ($\gamma=\tfrac12$, $\omega=0.172$) | −0.0030 | −0.0004 | +0.0000 | −0.0017 | −0.0031 | 7.0 | 0.22 |

  **The free model is rejected.** That is the one with $\omega$ borrowed from $p=0.04$,
  giving $\gamma_\infty=0.5145$. The replicates' own scatter agrees with the CLT se
  (0.0013–0.0014 vs 0.0015). The $\gamma=\tfrac12$ model is not rejected. Both models
  under-predict how fast the slopes fall at the top: every residual at $2^{21}$–$2^{22}$ is
  negative.
- **Local slopes, 4..$2^{23}$:** past the hump (peak 0.580 at 512), $s_k$ falls by about
  0.003–0.006 per doubling all the way to $s_{2^{22}}=0.5172$, with no sign of flattening.
  Neither one-correction form fits the whole ladder: (P) runs onto its bound
  ($\omega\to0$, χ² 919/18) and (L) gets χ² 3082/19.

### Tail fits on the whole ladders (`tail_fits.py`, 2026-11-03)

The user asked for a closer look at the local slopes. These fits use each run's full
`local_slope.json` ladder, 4..$2^{23}$, all at the final's $n$, not just the window's five
slopes. Eq. (pilot-fit) on $\ln\overline Y$ from a cut $i_{\min}$ to $2^{23}$, by weighted
least squares (the cells are independent draws):

| p | $i_{\min}$ | free: $\gamma$ | $\omega$ | χ² p | $\gamma=\tfrac12$: $\omega$ | χ² p | 95% profile $\gamma$ |
|---|---|---|---|---|---|---|---|
| 0.04 | $2^{8}$ | 0.4958 ± 0.0021 | 0.162 | 0.29 | 0.187 | 0.10 | [0.4925, 0.4975] |
| 0.04 | $2^{10}$ | 0.4999 ± 0.0025 | 0.195 | 0.39 | 0.196 | 0.48 | [0.4950, 0.5025] |
| 0.04 | $2^{12}$ | 0.5039 ± 0.0025 | 0.250 | 0.44 | 0.200 | 0.40 | [0.4975, 0.5075] |
| 0.04 | $2^{14}$ | 0.5027 ± 0.0076 | 0.224 | 0.28 | 0.185 | 0.37 | [< 0.40, 0.5075] |
| 0.3 | $2^{12}$ | 0.4808 ± 0.0066 | 0.117 | 0.50 | 0.177 | 0.00 | [0.4650, 0.4900] |
| 0.3 | $2^{13}$ | 0.4937 ± 0.0061 | 0.157 | 0.83 | 0.186 | 0.75 | [0.4775, 0.5025] |
| 0.3 | $2^{14}$ | 0.4912 ± 0.0098 | 0.147 | 0.75 | 0.187 | 0.70 | [0.4550, 0.5025] |
| 0.3 | $2^{15}$ | 0.4884 ± 0.0181 | 0.137 | 0.64 | 0.189 | 0.66 | [< 0.40, 0.5075] |

Both $p$ together, one $\gamma$, with $A$, $a_1$ and $\omega$ fitted per $p$:

| cuts | $\gamma$ | χ² (p) | $\gamma=\tfrac12$: χ² (p) | Δχ² |
|---|---|---|---|---|
| $2^{10}$ / $2^{13}$ | 0.4986 | 15.2/18 (0.65) | 15.6/19 (0.68) | 0.4 |
| $2^{12}$ / $2^{14}$ | 0.5015 | 14.0/15 (0.53) | 14.2/16 (0.59) | 0.2 |
| $2^{14}$ / $2^{15}$ | 0.4943 | 11.4/12 (0.50) | 11.7/13 (0.55) | 0.3 |

### Reading the two together

- **The ladders point to $\gamma=\tfrac12$.** Once a cut clears the pre-asymptotic region,
  each $p$'s free tail fit gives $\gamma$ = 0.494–0.504, and every 95% profile interval
  contains ½. At $p=0.3$ the $2^{12}$ cut is still inside the hump's tail ($\gamma=\tfrac12$
  rejected, p = 0.00); from $2^{13}$ on it is accepted. The joint fit puts $\gamma$ at
  0.494–0.502, and imposing ½ costs Δχ² ≤ 0.4.
- **With $\gamma=\tfrac12$, both $p$ have the same $\omega$:** 0.18–0.20 at every cut. The
  run's slopes are still falling at $2^{22}$ (0.507, 0.517) because that correction is slow.
- **$\gamma\approx0.515$ is excluded.** That is 48's sieve value and the free model of
  `predict.py`. Every profile interval's upper end is ≤ 0.5075, and the pre-registered test
  rejects it at $p=0.3$ (p = 0.0016).
- **The window $\hat\gamma$ values (0.5125, 0.5248) are biased by that slow correction.**
  Eq. (720)'s bound misses this at both $p$: its $B_{fs}$ comes from a pilot that isn't one
  correction around $\hat\gamma$ (pinned-refit χ² p = 0).
- **Caveat.** The profile intervals are wide toward small $\gamma$ at high cuts. One
  correction with $\omega\approx0.19$ is nearly degenerate with $\gamma$ over a factor of
  $2^{10}$ in $i$. The evidence for ½ comes from the cuts that still have lever arm
  ($2^{10}$–$2^{13}$), and those all fit.
## Wilson interval with the AIC pilot refit (2026-11-07)

B_fs now takes $\omega_1, a_1$ from the pilot refitted with $\gamma$ pinned to $\hat\gamma$ on a range of
$\ge 5$ consecutive pilot scales chosen by AIC with data cuts (Jay & Neil 2021;
`local_slope.omega_ranges`, `report.pinned_refit`), not on every pilot scale as in the
2026-11-02 refit. $\hat\gamma$ is unchanged; only the eq. (720) half-width moves. The
$\omega_1$ error is the profile error $\oplus$ the spread over ranges. Old = all-scales pinned
refit (kept in `answer.json` as `pilot_refit_all_scales`), new = AIC refit (`pilot_refit`);
reports rerun 2026-11-07. A warning is $\chi^2$ $p < 0.01$ (old: or $\omega_1$ on its bound).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old $\chi^2$/dof | old radius | new $\hat\omega_1$ | range | new $\chi^2$/dof | new radius |
|---|---|---|---|---|---|---|---|---|
| `long_p0.04_24h` | 0.512469 | 0.354 ± 0.00097 | 1.53e+04/6 ⚠ | 0.00652 | 0.192 ± 0.0012 | 32..512 | 3.03/2 | 0.109 |
| `long_p0.04_smoke_cap131072` | 0.512748 | 0.401 ± 0.0011 | 5.83e+03/6 ⚠ | 0.0106 | 0.189 ± 0.0068 | 32..512 | 2.76/2 | 0.125 |
| `long_p0.04_smoke_pool` | 0.516285 | 0.4 ± 0.0011 | 5.92e+03/6 ⚠ | 0.0117 | 0.201 ± 0.0055 | 32..512 | 5.93/2 | 0.109 |
| `long_p0.3_24h` | 0.524801 | 0.0588 ± 0.00053 | 1.67e+04/14 ⚠ | 2.72 | 0.307 ± 0.075 | 4096..131072 | 2.33/3 | 0.0463 |
| `long_p0.3_smoke_cap131072` | 0.519779 | 0.022 ± 0.00089 | 1.55e+04/14 ⚠ | 65.9 | 0.253 ± 0.058 | 4096..131072 | 1.42/3 | 0.0889 |
| `long_p0.3_smoke_pool` | 0.531202 | 0.0339 ± 0.001 | 1.61e+04/14 ⚠ | 16.9 | 0.369 ± 0.12 | 4096..65536 | 0.314/2 | 0.0348 |

## Wilson interval with the AIC-weighted refit (2026-11-10)

$\omega_1$ is now the AIC-weighted average over the kept ranges (Jay & Neil 2021, eq. 15),
not the single highest-weight range; its error is their eq. (18). $a_1$ is refitted on the
highest-weight range with $\omega_1$ fixed at that average, so B_fs takes one coherent
$(a_1, \omega_1)$ (averaging $a_1$ separately pairs values from different fits). $\hat\gamma$
is unchanged; only B_fs and the eq. (720) half-width move.
Old = the 2026-11-07 highest-weight range, new = the average; reports rerun 2026-11-10.
"best" is the highest-weight range and its share of the kept weight; its $\chi^2$ still
gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `long_p0.04_24h` | 0.512469 | 0.192 | 0.109 | 0.192 ± 0.00116 | 32..512 (1.00) | -0.717 | 0.108 | 0.109 |
| `long_p0.04_smoke_cap131072` | 0.512748 | 0.189 | 0.125 | 0.188 ± 0.00676 | 32..512 (0.98) | -0.733 | 0.118 | 0.126 |
| `long_p0.04_smoke_pool` | 0.516285 | 0.201 | 0.109 | 0.201 ± 0.00551 | 32..512 (1.00) | -0.682 | 0.101 | 0.109 |
| `long_p0.3_24h` | 0.524801 | 0.307 | 0.0463 | 0.292 ± 0.0758 | 4096..131072 (0.29) | -1.77 | 0.0519 | 0.053 |
| `long_p0.3_smoke_cap131072` | 0.519779 | 0.253 | 0.0889 | 0.243 ± 0.0593 | 4096..131072 (0.29) | -1.65 | 0.0885 | 0.1 |
| `long_p0.3_smoke_pool` | 0.531202 | 0.369 | 0.0348 | 0.341 ± 0.117 | 4096..65536 (0.49) | -2.05 | 0.0286 | 0.0422 |

## Wilson interval with the joint refit (2026-11-11)

B_fs now takes $\omega_1, a_1$ from one AIC range fit on the pilot **and** the long run
together ($\gamma$ free, $\ge 6$ consecutive scales, a scale in both merged by inverse
variance; `report.joint_refit`, user 2026-11-11), not from the pilot refitted with $\gamma$
pinned to $\hat\gamma$, whose pin carries $\hat\gamma$'s scatter but not its bias (see
`src/study/README.md`). $\hat\gamma$ is unchanged; only B_fs and the eq. (720) half-width
move. Old = the 2026-11-10 pinned average (still in `answer.json` as `pilot_refit`), new =
`joint_refit`; reports rerun 2026-11-11. "best" is the lowest-AIC range of the union and
its share of the kept weight; its $\chi^2$ gives the warning (⚠).

| study | $\hat\gamma$ | old $\hat\omega_1$ | old radius | new $\hat\omega_1$ | best (weight) | new $\hat a_1$ | new $B_{\rm fs}$ | new radius |
|---|---|---|---|---|---|---|---|---|
| `long_p0.04_24h` | 0.512469 | 0.192 | 0.109 | 0.147 ± 0.072 | 64..524288 (0.19) | -1 | 0.339 | 0.341 |
| `long_p0.04_smoke_cap131072` | 0.512748 | 0.188 | 0.126 | 0.131 ± 0.0178 | 64..8388608 (0.16) | -1.13 | 0.522 | 0.53 |
| `long_p0.04_smoke_pool` | 0.516285 | 0.201 | 0.109 | 0.139 ± 0.0683 | 64..4194304 (0.27) | -1.05 | 0.463 | 0.472 |
| `long_p0.3_24h` | 0.524801 | 0.292 | 0.053 | 0.127 ± 0.0683 | 4096..8388608 (0.22) | -1.93 | 0.961 | 0.962 |
| `long_p0.3_smoke_cap131072` | 0.519779 | 0.243 | 0.1 | 0.0729 ± 0.0377 | 2048..16777216 (0.28) | -3.3 | 5.29 | 5.3 |
| `long_p0.3_smoke_pool` | 0.531202 | 0.341 | 0.0422 | 0.129 ± 0.128 | 2048..4194304 (0.17) | -1.84 | 0.887 | 0.901 |
