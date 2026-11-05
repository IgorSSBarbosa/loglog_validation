# Experiment 52 — the `rwre_gpu_lazy` pipeline across $p$, with $\omega_1$ fixed

**Status: launched 2026-11-04. No results yet.**

## Question (user, 2026-11-04)

Experiment 50 ran the article's pipeline at two values of $p$ only. What does the pipeline
($\hat\gamma$ by eq. (526), its eq. (720) bound, the Step 4 checks) give across the whole
range $p\in[0,\tfrac12]$, on one budget per $p$, when every plan rests on the same tail
correction ($\omega_1$, $a_1$ from 50's $p=0.04$ fit)?

## Acceptance criteria (known values, kept out of the code path)

- $p=\tfrac12$: nothing is ever revealed (`models/rwre_gpu_lazy.py`, skip-read), so the walk
  is srw and $\gamma=\tfrac12$ exactly. The 95% eq. (720) interval must contain ½.
  (srw's own correction is $\omega_1=1$, $a_1=-\tfrac14$, not the borrowed constants below,
  so at $p=\tfrac12$ the plan is built on a wrong correction by design.)
- Every other $p$: nothing proven. The article's conjecture is $\gamma=\tfrac12$ for all $p$.
  50's tail fits put $\gamma$ at 0.494–0.504 at $p=0.04$ and $0.3$.

## Design (decided by the user, 2026-11-04)

- **Model** `rwre_gpu_lazy`, observable $|X_i|$, params as 50's long runs: `alpha` 0.5,
  `kernel` "pool", `cap` 393216, defaults otherwise.
- **Grid:** 11 values $p = 0, 0.05, 0.1, \dots, 0.45, 0.5$.
- **Budget:** `--time 3h` per $p$, 5 replicates, `--max-scale 2^23` (as 50), $m=6$, $\rho=2$.
- **No pilot is drawn.** Each study reuses 19's `omega_p<p>_n4e6` ladder
  (`autopilot.py --reuse-pilot`; $n=800{,}000\times5$, 4..$2^{18}$ for $p\le0.3$,
  4..$2^{20}$ for $p\ge0.35$), as 50 did at $p=0.3$. Its constants are replaced
  (`build_constants.py`, every source in `constants.json`):

  | constant | at every $p$ | source |
  |---|---|---|
  | $\omega_1$ | $0.2552\pm0.0232$ | **borrowed** from 50 `pilot_p0.04_pool`: eq. (pilot-fit) on 48's local-slope $\ln\overline Y$, $\gamma$ free, $1024..2^{21}$ |
  | $a_1$ | $-1.0378\pm0.0497$ | **borrowed**, same fit |
  | cv | per $p$ | 19's ladder at that $p$, mean per-scale cv over its top 5 scales ($2^{14}..2^{18}$ or $2^{16}..2^{20}$). A moment, not a fit |
  | $d$, throughput | per $p$ | `time_window.py`: one call `simulate(i, 16384)` per scale $2^{18}..2^{23}$, pooled kernel, OLS in logs, on the host that draws the run |

- **Why $\omega_1$ and $a_1$ are not refitted per $p$ (user):** with $\omega_1$ fixed, a refit
  of $a_1$ does not work, because the per-$p$ ladders do not have the single-correction shape
  (19's hump). $d$ and the throughput are the only per-$p$ refit.
- **The cap and kernel note** (as 50): the built `pilot.json` carries the run's recipe
  (kernel "pool", cap 393216), although 19's draws were made with the warp kernel at cap
  $2^{15}$. Neither changes a draw. Recorded in `provenance_note`.
- **The cost block:** `pilot.json`'s `cost` is 19's own probe at that $p$ (4..$2^{16}$, warp
  kernel), for `plot_pipeline.py`'s panels only. 19's pilots predate the per-probe list
  `probes`, so it gets one entry (19's aggregated timing).

## Commands

```bash
# one p (time window, constants, autopilot, figures), all on the current host:
bash experiments/52_rwre_lazy_fixed_omega_sweep/run_p.sh <p> 3h <seed>
# the sweep: 11 lines in local/queue_52_sweep.txt, at most 4 hosts at once
nohup bash local/dispatch_pool.sh local/queue_52_sweep.txt > local/queue_52_sweep.nohup 2>&1 &
```

Each study writes `data/cost_window_p<p>_3h/`, `data/pilot_p<p>_3h/`, `data/fixed_p<p>_3h/`
and logs to `data/fixed_p<p>_3h.log`.

**Seeds:** run seed $202611045200 + k$, with $k$ the index of $p$ in the grid
($p=0$: …5200, $p=0.5$: …5210). The window timing's seed is the run seed $+20$
(…5220–…5230). The smoke test (`fixed_p0.3_smoke`, `TW_N=2048`, `--time 5m`) used
202611045206, the same root seed as the $p=0.3$ run (an oversight): its draws overlap that
run's streams, so the smoke study is a plumbing check only and is never combined with or
compared against `fixed_p0.3_3h`.

## Smoke test (2026-11-04, xingu, `fixed_p0.3_smoke`)

`TW_N=2048 run_p.sh 0.3 5m 202611045206 smoke`, the whole path. Window timing at n = 2048
(under one device wave, so slower per sample than 50's): $d=1.412\pm0.066$. Plan: $m_0=16$,
$2^{17}..2^{22}$, $n=1{,}492\times5$, predicted 300 s, drawn in 341 s (1.14×). Report:
$\hat\gamma=0.5255$, t(4) interval $[0.5215, 0.5295]$, eq. (720) bound void ($\pm9.4$; pinned
refit $\chi^2=1.63\times10^4/14$, as at 50's $p=0.3$). Local slopes ran. `plot_pipeline.py`
wrote `pipeline_pilot`, then stopped at `pipeline_allocation`'s assert "eq. (m0-star)
disagrees with the plan": at this tiny budget the article's plug-in crossing and the
planner's minimum-RMSE $m_0$ differ by one. At 3h the plan is capped by `--max-scale`
(as in 50), where the two agree. The other three figures render (checked by hand).

Known figure caveat: `pipeline_pilot` panel (a) draws the reused 19 pilot's own direct fit
($\hat\omega_1\approx0.01$ at $p=0.3$, the hump), not the borrowed constants the plan used.

## Launch (2026-11-04 21:40)

`local/queue_52_sweep.txt` through `local/dispatch_pool.sh` (detached, on purus), hosts
from its default list, each probed idle before use. Pool log
`local/queue_52_sweep.txt.log`; finished jobs in `local/queue_52_sweep.txt.done`.
First wave: $p=0$ on missouri, $0.05$ on xingu, $0.1$ on kolyma (tarim and amazonas failed
the idle probe; tocantins and niagara were in use by others). All three plans: $m_0=17$,
window $2^{18}..2^{23}$ (capped by `--max-scale`), 36 min per replicate, with

| p | $d$ (window) | throughput /s | host | $n$ per scale ×5 |
|---|---|---|---|---|
| 0 | $1.458\pm0.033$ | $2.22\times10^{11}$ | missouri | 24,421 |
| 0.05 | $1.463\pm0.033$ | $2.52\times10^{11}$ | xingu | 25,735 |
| 0.1 | $1.473\pm0.034$ | $2.86\times10^{11}$ | kolyma | 25,084 |

## Results

(none yet)
