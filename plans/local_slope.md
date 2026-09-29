# Plan — consecutive-point slopes below a finished study

**Status: IMPLEMENTED 2026-09-28 (branch `local-slope`), not yet run: section 6 tests written, section 7 runs pending the GPU.** Designed with Igor on 2026-09-28, after the
`12_erw_gpu` p-sweep: the Wilson interval holds far from $p = 3/4$ and misses the true
$\gamma$ at and near it.

---

## 1. The question

Take any finished study (autopilot, or pilot -> plan -> run). Look at the slope between
each pair of consecutive scales,

$$s_k = \frac{\log \overline Y_{\rho^{k+1}} - \log \overline Y_{\rho^k}}{\log\rho}$$

which is the article's $\hat\gamma$ with $m = 2$ and $m_0 = k$ (eqs. 523–526: with $m = 2$ the
weights are $\mp 1$). Prop. (820) says its bias falls like $\rho^{-\omega_1 k}$, i.e. like
$i^{-\omega_1}$ in the scale $i$. How does $s_k$ approach its limit?

- power law, as Assumption 1 (eq. 232) says: $s_k - \gamma \sim c\, i^{-\omega_1}$
- logarithmic, where Assumption 1 fails: $s_k - \gamma \sim c/\ln i$
- a crossover: log-like at small i, power law beyond some $i^*$

Nothing here is ERW-specific. The tool reads model, params, $\rho$, $d$ and the ladder from
the study's own files and works for any model in `tools/models.py`.

## 2. The x-axis (decided: scale)

$s_k$ is plotted against the scale $i = \rho^k$. The budget rate of eq. (941)/(966),
$B^{-\omega_1/(d+2\omega_1)}$, only holds when $n$ grows with $B$ as eq. (945)-(946) say. That
is Experiment C (`src/budget/allocation_experiment.py`), which needs $\omega_1$ in advance
and so cannot run where Assumption 1 fails. Here $n$ is fixed, so the budget of a
consecutive pair is $B_k \propto n\, i^d$, and the bias in $B$ is $B^{-\omega_1/d}$. Budget is shown
only as a secondary axis label, using $d$ from `constants.json`. The derived
$\omega_1/(d+2\omega_1)$ is printed next to it.

## 3. Samples: reuse first, draw only the gaps, one n everywhere

**Reuse.** `pilot.json` and `final.json` both keep `per_replicate[r]["y_bar"]` per
scale. $s_k$ needs only those means, so the discarded raw samples (`samples_kept: false`)
don't matter.

**$n$ must be the same at every scale (Igor, 2026-09-28).** This isn't a nicety.
$\log \overline Y$ is biased, $\mathbb E \log \overline Y \approx \log \mathbb E Y - \mathrm{cv}^2/(2n)$ (delta method). So if adjacent
scales have different $n$ per replicate, $s_k$ picks up a spurious term

$$\frac{\mathrm{cv}^2}{2\log\rho}\left(\frac1{n_k} - \frac1{n_{k+1}}\right)$$

that has nothing to do with the model. At the ERW $p = 0.75$ study (pilot $n = 122{,}129$,
final $n = 131{,}848$, cv $\approx 0.69$) it is about $2\times10^{-7}$, which is negligible. But it isn't
negligible in general: the pilot's $n$ can come from a Neyman/SNR allocation that varies
by scale, and a small-$n$ pilot makes it large. It also breaks the uniform-$n$ premise that
the weights of eq. (526) are derived under.

Rule: **per-replicate $n$ is equal at every scale**, fixed to the final's $n$ ($n_f$). The
replicate count $R$ may differ between scales. $R$ only changes the se, which is reported
per point.

How each existing scale is brought to $n_f$, using only the stored means:

| stored per-replicate $n$ at a scale | action |
|---|---|
| $= n_f$ | reuse as-is |
| $< n_f$ (typical for the pilot) | **top up**: draw $n_f - n$ fresh samples per replicate at that scale and merge the means, $\overline Y = \bigl(n\,\overline Y_{\rm old} + (n_f - n)\,\overline Y_{\rm new}\bigr)/n_f$. The merge is exact, uses fresh randomness at the same scale, and slices nothing (ground rule 2) |
| $> n_f$ | a mean can't be shrunk: **drop** the stored value and redraw at $n_f$. Printed |
| pilot and final both cover the scale | both are at $n_f$ after the above: pool them as independent replicates |

**Gaps.** Scales between the smallest stored scale and the bottom of the final ladder
with no stored mean are drawn fresh at $n_f$ with the final's $R$. Example: ERW $p = 0.75$ has
pilot $4..4096$ and final $2^{18}..2^{23}$, so the gap is $2^{13}..2^{17}$. **Never above the top of
the final ladder.** `--min-scale` defaults to the pilot's smallest scale.

Every draw goes through `generate()` (the one sampler), with the study's model and
params, under a new `--seed` that is recorded. The streams come from `tools/rng.py`'s
`spawn`, one child per (scale, replicate, top-up/gap), passed as SeedSequence objects.
The new means are written back in the same `per_replicate[r]["y_bar"]` shape, with n
and source per scale, so a later run reuses them too.

**Time.** Before drawing, print the predicted wall clock (pilot's measured throughput x
the steps to draw), then run. There is no confirmation prompt and no cost cap
(decided 2026-09-28).

## 4. What is computed (`tools/local_slope.py`, no I/O)

1. Per scale: mean over replicates of $\log \overline Y_r$, and $\mathrm{se} = \mathrm{sd}_r/\sqrt R$.
2. $s_k$ and $\mathrm{se}(s_k) = \sqrt{\mathrm{se}_k^2 + \mathrm{se}_{k+1}^2}/\log\rho$. Scales are independent draws,
   so the errors add in quadrature and no replicate pairing is needed. Cross-check
   against the CLT (eq. 583) with $m = 2$.
3. Truth-free shape diagnostic:
   - $\Delta s_k = s_{k+1} - s_k$
   - local exponent $w_k = -\log(\Delta s_{k+1}/\Delta s_k)/\log\rho$
   - power law gives flat $w_k$ ($= \omega_1$); a log correction gives $w_k$ drifting to 0
     ($\Delta s_k \propto 1/k^2$); a crossover gives drift then plateau.
   - Each point is marked resolved or not ($|\Delta s_k| > 2\,\mathrm{se}$). Unresolved points are drawn
     hollow and not read.
4. Two fits to $s_k$, weighted by se, compared by $\chi^2$ and $p$-value only, with no
   verdict thresholds:
   - (P) $s = \gamma + c\, i^{-\omega}$
   - (L) $s = \gamma + c/\ln i$
5. The n-mismatch term of §3, evaluated with each scale's cv, printed as a check (should
   be 0 once §3 is applied).

Before writing any of this, check `tools/loglog.py` for an existing two-point estimator
(`artifacts.py` mentions a `"two_point"` key) and reuse it if present.

## 5. Driver (`src/study/local_slope.py`)

```
python3 src/study/local_slope.py --study <s> --data-root <D> --seed <int>
        [--min-scale I] [--truth GAMMA] [--dry-run]
```

- `--dry-run`: print the plan (per scale: reuse / top-up / drop+redraw / gap, n, R, and
  predicted time) and stop.
- `--truth`: reporting only, never passed to a fit. It adds a panel of $|s_k - \gamma_{\rm truth}|$ on
  log-log and is recorded in the output as user-supplied (as
  `compare_observables.py --truth` does). **No true values in code** (ground rule 4).
- Output: `local_slope.json` (new `ARTIFACTS["local_slope"]` entry) and
  `local_slope.png` in the study directory. Records seed, $n$ and $R$ per scale, the source
  of each mean, elapsed time (ground rule 5), fits and $w_k$.
- Panels: (a) $s_k$ vs $\log_2 i$ with error bars, budget as a secondary axis; (b) $w_k$ vs
  $\log_2 i$; (c) only with `--truth`.

## 6. Checks before any real model (ground rule 1)

Local tests in `tools/tests/test_local_slope.py` (gitignored, as all tests are):

- noiseless data with a planted power law: $w_k = \omega$ to machine precision, (P) wins
- noiseless data with planted $s = \gamma + c/\ln i$: (L) wins, $w_k$ drifts
- the `synthetic` model: $\mathrm{se}(s_k)$ agrees with the scatter across replicates
- the top-up merge equals one draw of $n_f$ samples in distribution, and the n-mismatch
  term is 0 after it
- a gap draw is bit-for-bit `generate()` with the same SeedSequence child
- `test_artifacts.py` covers `--help` automatically

## 7. First runs: measurements, not criteria

1. `srw` ($p = 1/2$) as a sanity check: $w_k$ should plateau near its README's $\omega_1$.
2. `12_erw_gpu` sweep, $p \in \{0.6, 0.7, 0.725, 0.75, 0.8\}$. It asks whether $w_k$ levels off
   near $3 - 4p$ below $3/4$ (the conjecture: 0.6, 0.2, 0.1) and drifts to 0 at $3/4$. This is
   a conjecture to test, not a criterion. No pass/fail is written until the results
   have been seen. Results go in `experiments/12_erw_gpu/README.md` with their commands.
3. `13_rwre_gpu` once its sweep finishes.

## 8. Files

| file | change |
|---|---|
| `tools/local_slope.py` | new: §4 |
| `src/study/local_slope.py` | new: §3, §5 |
| `tools/artifacts.py` | `"local_slope": "local_slope.json"` |
| `src/README.md`, `src/study/README.md`, `CATALOG.md` | command reference, one section, entries |

Branch `local-slope` from the current `rwre-gpu` HEAD, where the `*_gpu` models and
sweep data live.
