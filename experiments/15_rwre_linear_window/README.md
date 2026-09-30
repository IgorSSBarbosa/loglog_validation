# Experiment 15 — `rwre_gpu` on the linear window $W=2k+100$

**Question.** Does 13's sieve of $p$ change when the periodic window grows linearly,
$W=2k+100$, instead of as $12\lceil k^{3/4}\rceil$ (user, 2026-09-30)? Everything else
is 13's: the model (`rwre_gpu`), the grid, the pilots, the wall clock per $p$. So a
difference between 13 and 15 at the same $p$ is the window's.

Why a linear window. 13's rule rests on a *measured* spread, $\lvert X_k\rvert\sim
k^{0.58}$, and a margin over it. On $W=2k+100$ the walker can never reach the seam:
$\lvert X_k\rvert\le k<W/2$. What's left is the environment. Stirring moves a site's
content by ±1 with probability `swap_prob` per sweep, so the value the walker reads
comes from the end of a backward path of variance 2 per walker step (at the defaults).
The torus differs from the line only if two sites the walker reads trace back to sites
$W$ apart. The walker's sites span at most $k+1$, so the two paths must separate by
$k+100$ more. That is impossible for $k\le14$ (paths move at most $4k$). Beyond that, a
Gaussian tail with the paths treated as independent gives
$\exp(-(k+100)^2/8k)\le e^{-50}\approx2\times10^{-22}$, worst at $k=100$. This is a
heuristic, not a bound. The derivation is in `models/rwre.py`, decision (d).

What it costs:

| | 13: $12\lceil k^{3/4}\rceil$ | 15: $2k+100$ |
|---|---|---|
| cost$(k)=k\,W$ | $12\,k^{7/4}$ | $2k^2+100k$ |
| declared $d$, 4..1024 / 128..16384 | 1.74 / 1.75 | 1.54 / 1.94 |
| $W$ at $k=1024$ / 32768 | 2,184 / 29,232 | 2,148 / 65,636 |
| `max_steps` (96 KiB of shared memory) | 165,086 | **49,094** |

The two windows are almost equal at $k=1024$. Above that, the linear one is wider and
more expensive, and the dyadic ladder stops at $2^{15}$ instead of $2^{16}$. The declared
$d$ isn't a constant any more. It's `tools/cost_model.py`'s `declared_exponent`, the
log-log OLS of `cost_hint` over whatever ladder is in use.

## What changed in the code

`models/rwre.py` gained `window_slope` and `window_offset` (both default 0). With
`window_slope > 0`, $W=\lceil\texttt{window\_slope}\cdot k+\texttt{window\_offset}\rceil$,
rounded up to even. `_check` rejects a negative value, an offset without a slope, and a
slope combined with a non-default `window_c` or `window_exponent`. `rwre_gpu` gets the
window through the same `_window(k, q)`, so its kernel is unchanged. At `window_slope =
0` every draw is bit-identical to before (L0). Nothing in `src/`, `tools/` or
`calibration/` changed.

`sweep_p.py` is 13's driver (with its `--pilot neyman|snr`) on this window. Its
templates, `recipes/samples_sweep_neyman.json` and `recipes/samples_calib_p0.5.json`,
already carry the window, and `write_recipe` refuses a template that doesn't.
`--max-scale` is `max_steps` = 49094.

## Acceptance criteria (written 2026-09-30)

- **L0. Nothing else moved.** At the default window and at `window_exponent = 3/4`,
  `window_width` for $k<70000$, `cost_hint` for $i<5000$, and seeded `rwre.walk` draws
  equal HEAD's bit for bit.
- **L1. The window is the one asked for.** $W(k)=2k+100$ exactly for $k<10^5$, and
  `cost_hint` $=k(2k+100)$. `max_steps` runs at the opt-in ceiling, and one step above
  it raises.
- **L2. GPU against CPU on this window.** 13's G1 on $p\in\{1/3,0\}$ ×
  $k\in\{64,1024\}$, $n=4000$ per arm: two-sample KS $p\ge0.01$ and means within 3
  combined se. Plus the zero-check $\lvert\mathbb EX_k\rvert<4$ se, parity and range at
  $k=1024,16384,32768$ ($p=0$).
- **L3. The control arm.** At $p=1/2$ through `autopilot.py`: $\gamma=1/2$, $\omega_1=1$,
  $a_1=-1/4$, each within 2 se.
- **L4. Cost.** `measure_cost.py` on `recipes/cost_probe.json` recovers the declared $d$
  within the driver's 20%.

## L0–L2: PASS (2026-09-30)

The checks are a local script (tests stay out of git). L0 and L1:

```
PASS default & 3/4: window, walk, cost_hint bit-identical to HEAD
PASS linear: W = 2k+100 exactly, cost_hint = k(2k+100)
max_steps default / 3/4 / linear: 67076100 165086 49094
```

L2:

```
PASS  {'p': 1/3, linear} k=64: KS p=0.945  mean cpu=7.250 gpu=7.180  z=-0.58
PASS  {'p': 1/3, linear} k=1024: KS p=0.172  mean cpu=35.593 gpu=36.243  z=+1.12
PASS  {'p': 0.0, linear} k=64: KS p=0.994  mean cpu=10.437 gpu=10.305  z=-0.78
PASS  {'p': 0.0, linear} k=1024: KS p=0.466  mean cpu=48.521 gpu=47.698  z=-1.05
PASS  zero-check E X_k = 0, p=0, k=1024, W=2148: z=-0.92; parity and range ok; mean|X|=48.1
PASS  zero-check E X_k = 0, p=0, k=16384, W=32868: z=+0.44; parity and range ok; mean|X|=212.3
PASS  zero-check E X_k = 0, p=0, k=32768, W=65636: z=-0.20; parity and range ok; mean|X|=313.2
PASS  k = max_steps = 49094 (W = 98288, shared-memory opt-in) runs
PASS  k = 49095 raises: window width 98290 at k = 49095 exceeds the 98288 bytes ...
```

At $k=1024$ the means agree with 13's G1 arms on $12\lceil k^{3/4}\rceil$ (35.73 and
49.22 on CPU). The two windows are nearly equal there, so this is the expected result.

## L3: PASS (2026-09-30, smoke)

`sweep_p.py --tag smoke --time 4m --p 0.5 --pilot snr`, 3.8 min:

| object | measured | exact | |
|---|---|---|---|
| $\omega_1$ (pilot, 4..1024) | $1.158\pm0.209$ | 1 | $0.76\sigma$ |
| $a_1$ (pilot) | $-0.290\pm0.102$ | $-1/4$ | $0.39\sigma$ |
| $\hat\gamma$, final ladder 128..4096 | $0.50019\pm0.00018$ | 1/2 | $1.1\sigma$ |
| eq. (720) 95% interval | $[0.4986, 0.5017]$ | contains 1/2 | |

## L4: PASS (2026-09-30)

```
declared d = 1.9413   (model's cost_hint)
measured d = 1.9325 +/- 0.0307   (affine a + b*i**d)
gap = -0.29 sigma, 0.45% relative
PASS: measured d = 1.9325 against this model's declared 1.9413 (0.5% of it, tolerance 20%)
```

This is closer than 13's E1 (6.9% low). The serial part of a step, $a\approx0.7$ ns, is
now small against $bW$ at every probed $k$, because $W$ is never below ~350 there.

## The sieve (not yet run)

```bash
python3 experiments/15_rwre_linear_window/sweep_p.py --tag 1h --time 1h      # 8 x 1h, resumable
```

Same grid as 13, $p=1/3, 0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5$, with the neyman pilot on
2..1024. The comparison that answers the question is 15 against 13's same-tag sweep, $p$
by $p$: $\hat\gamma$ and the pilot's $\omega_1$. The small-scale local slopes on 4..1024
are part of it too, and there the two windows differ most in relative terms ($W=36$
against 108 at $k=4$). Results go to `data/sweep_p_<tag>.md` and are recorded here once
measured.
