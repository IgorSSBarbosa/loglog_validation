# Plan — an exact box ladder for the susceptibility: nothing rounded, $L/\xi$ constant

**Status: DESIGN. Nothing implemented.** Designed with Igor on 2026-09-21, after the
d = 4 $\omega_1$ pilot (`pilot_gsusc_d4_low`) returned local slopes that zig-zag at
5–31σ, and a direct fit of eq. (232) had nothing consistent to find. §7 lists the
decisions that need sign-off before T1 starts.

Evidence scripts, both reproducible from data already on disk:
`experiments/09_percolation_susceptibility_gpu/diagnostics/rounding_zigzag.py` and
`.../diagnostics/ladder_stability.py`.

---

## 1. The problem, measured

`percolation_susceptibility(_gpu)` puts rung $x$ at $p = p_c - \varepsilon_0/x$ on a torus of
side $L = \lceil c\,x^{\nu_{\text{box}}}\rceil$ ($c$ = `box_factor`). With a constant ratio
$L/\xi$, the finite-torus deficit is a constant factor that moves $a_0$ and leaves $\gamma$
alone. That was measured in d = 2 (`experiments/06_susceptibility/`, step 1), where
even a −55% deficit gave an unbiased exponent. **The ceiling breaks the constancy.** Each
rung gets its own box factor $c_x = L/x^{\nu_{\text{box}}}$:

| $x$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| $L$ | 7 | 11 | 17 | 28 | 44 | 71 | 114 | 184 |
| $c_x/c - 1$ | +8.47% | +5.66% | +1.22% | +3.34% | +0.66% | +0.68% | +0.20% | +0.25% |

If $S(p, L)$ is still growing with $L$ at $c = 4$, rung $x$ is shifted by
$G\log(c_x/c)$, where $G = \partial\log S/\partial\log L$ at fixed $p$. The shift is not monotone in $x$.

**Test (no new draws).** Pool the three runs that share the design exactly
(`pilot_gsusc_d4_low`, `pilot_gsusc_d4`, `autopilot_gsusc_d4_210m/final`: box_factor 4,
$\nu_{\text{box}} = 0.69$, $\varepsilon_0 = 0.09$, torus, moment 1). They are independent
draws, so rungs pool by inverse variance, 8 rungs over $x = 2..256$. Then fit eq. (232):

| fit | $\chi^2$/dof | $\omega_1$ | $G$ |
|---|---|---|---|
| one correction | **47.0 / 4** (rejected) | 3.4 ± 0.6 | — |
| one correction + $G\log(c_x/c)$ | **3.7 / 3** | 1.49 ± 0.17 | **0.162 ± 0.023** |

$G$ is 7σ from zero and has the physical sign (a bigger torus gives a bigger $S$). The rounding
shift compared with each rung's noise: **70σ at $x = 2$, 32σ at 4, 8σ at 16, 0.6σ at 64,
< 0.1σ at 256.** An $\omega_1$ pilot lives on exactly the small, precise rungs, which is
why it failed, while the production run at $x = 16..256$ barely noticed.

The $\omega_1 = 1.49$ above is a diagnostic, not a constant to pin. It leans on $x = 2$
($x \ge 4$ alone gives $2.7 \pm 2.7$), and it is far from the expected $\Delta_1 \approx 0.77$.

## 2. The design: the same ladder with a different $\rho$

Write the current design with a general ratio: $p_k = p_c - \varepsilon_0\rho^{-k}$,
$L_k = c\,\rho^{k\nu_{\text{box}}}$. At $\rho = 2$, $L_k$ is not an integer and must be rounded.
**Choose $\rho = 2^{1/\nu_{\text{box}}}$ ($= 2.7307$ at 0.69) and $L_k = L_0 2^k$. Every box
is then an integer, nothing is rounded, and $c_k = L_k/x_k^{\nu_{\text{box}}}$ is the same number
on every rung.** Same three constants $(\varepsilon_0, c, \nu_{\text{box}})$, same relation
$L = c\,x^{\nu_{\text{box}}}$, same estimator. Only the choice of which rungs to draw
changes.

Implemented by indexing the ladder by the **box side**:

$$x(L) = (L/c)^{1/\nu_{\text{box}}}, \qquad p(L) = p_c - \varepsilon_0/x(L), \qquad \text{cost}(L) = L^{\text{dim}}.$$

- Recipe `scales` are integer $L$. On $L = 2^k$ the pipeline's integer-$\rho$ grid
  (`tools/allocation.py`'s `ladder`, `gamma_closed_form`) works unchanged.
- In $L$ units eq. (232) holds exactly, with $\gamma_L = \gamma_{\text{susc}}/\nu_{\text{box}}$ and
  $\omega_L = \omega_x/\nu_{\text{box}}$. So **$\gamma_{\text{susc}} = \nu_{\text{box}}\,\gamma_L$ exactly**:
  $\nu_{\text{box}}$ is the design constant that set $p$, not an estimate of $\nu$, so the
  conversion adds no error. $\gamma_L$ is **not** the physical $\gamma/\nu = 2-\eta$.
- Assumption 7 holds exactly: cost $= L^{\text{dim}}$ with no ceiling, declared $d = \text{dim}$.
- Any integer $L$ is an exact rung, not only powers of 2. That is what makes the √2
  pilot grid of §4 possible.

## 3. Stability, and what the constants still do

None of $\varepsilon_0, c, \nu_{\text{box}}$ is measured in either design. $\varepsilon_k$ is set,
and the regression uses it exactly. A "wrong" constant only moves $L/\xi$ away from what
was intended, identically in both designs:

- $\varepsilon_0$ or $c$ wrong → $L/\xi$ is a different **constant** → only $a_0$ moves
  (and $G$ is larger or smaller).
- $\nu_{\text{box}} \ne \nu$ → $L/\xi \propto x^{\nu_{\text{box}}-\nu}$ drifts →
  **bias $= G\,(\nu_{\text{box}}-\nu)$**. That is $+0.0009$ at $G = 0.16$ with $\nu = 0.6845(23)$,
  and 0.0005–0.0013 across that error. The model's documented rule ($\nu_{\text{box}}$ an upper
  bound) fixes its sign and makes it shrink along the ladder.

Noiseless check (`ladder_stability.py`: $F(u) = 1-e^{-u}$ calibrated to $G = 0.16$;
$\hat\gamma-\gamma$ | worst local-slope error, units of $10^{-4}$):

| perturbation | current, $c$ | current, $2c$ | exact, $c$ | exact, $2c$ |
|---|---|---|---|---|
| nominal | −17.7 \| **85.1** | +0.6 \| 1.6 | +8.8 \| 8.9 | +0.9 \| 0.9 |
| $\nu_{\text{box}} = \nu$ exactly | −30.9 \| **85.1** | −0.6 \| 2.6 | **0.0 \| 0.0** | 0.0 \| 0.0 |
| $\varepsilon_0 \times \tfrac12$ | −39.6 \| **188.3** | +3.9 \| 9.8 | +18.9 \| 19.1 | +5.2 \| 5.3 |
| $\nu_{\text{box}} = 0.66$ | −66.2 \| **265.8** | −6.4 \| 11.1 | −47.3 \| 50.8 | −5.3 \| 6.2 |

So a larger $c$ stabilizes **both** designs, because every constant acts only through $G$. The
exact ladder removes the rounding term, the only thing the two do not share. In the
exact design, $c$ is one recipe number, so "a bigger box" stays available as a setting
(T2 decides its value).

## 4. Two ladders, because one estimator cannot take a rounded grid

`gamma_closed_form` (eq. 523–526) accepts any grid whose $\mathrm{rint}(\log L/\log\rho)$
is consecutive, and then weights it as if exact. On a rounded √2 grid of $L$, with zero noise
and an exact power law, that errs by **+0.018, +0.018, −0.009, −0.009, +0.005, +0.005 in
$\gamma_{\text{susc}}$**, up to 3× the literature error. OLS on the actual $\log L$
(`gamma_all_points`) errs by $10^{-15}$. Hence:

| use | grid | why |
|---|---|---|
| **$\omega_1$ pilot** | $L = \mathrm{round}(\sqrt2^{\,k})$ = 8, 11, 16, 23, 32, 45, 64, 91, 128 | 9 rungs where $\rho = 2$ gives 5; the pilot's fits (`fit_correction`, `ladder_check`) use the actual $\log L$, so rounding the *abscissa* is harmless, and $p$ comes from the integer $L$, so $c$ stays exact |
| **$\gamma$ production** | $L = 2^k$, top 128 | `report.py`'s eq. (720) path uses the closed form; exact grid only |

The pilot and the production run share the variable $L$, so $(\omega_L, a_1)$ pass from one to
the other with no conversion.

**Reach in d = 4, as things stand.** int32 labels cap one sample at $(L+1)L^3 < 2^{31}$,
so $L \le 215$, and 24 B/site puts $L = 181$ at 24.1 GiB and $L = 215$ at 48 GiB. On the
$2^k$ grid the top is **$L = 128$**, i.e. $x \le 152$ at $c = 4$, $x \le 84$ at $c = 6$, $x \le 56$ at
$c = 8$. The old production window was $x = 16..256$. Going higher needs T6 (Phase 6).

---

## 5. Tasks, in order

Each task lists what it touches, what it depends on, and the **numeric check that makes it
done** (ground rule 1). Tests go in `tools/tests/` (local, gitignored).

### Phase 0 — housekeeping

**T0.1 — land the diagnostics.** Commit `experiments/09_.../diagnostics/rounding_zigzag.py`
and `ladder_stability.py` (written 2026-09-21, uncommitted).
*Done when:* rerunning prints $\chi^2$ 47.0/4 → 3.7/3 with $G = 0.1622 \pm 0.0228$, and the nominal
stability row −17.7|85.1, +0.6|1.6, +8.8|8.9, +0.9|0.9.

**T0.2 — clear the working tree before `report.py` is touched.** `gpu-models` carries
uncommitted edits to `src/study/report.py`, `src/study/run.py`, `tools/artifacts.py` and
`src/study/README.md`, plus untracked recipes. They are not part of this plan. Igor decides
whether to commit or stash them. Then branch `exact-box-ladder` off the result.
*Done when:* `git status` is clean on the new branch.

### Phase 1 — the model

**T1.1 — extract the lattice core, bit-identically.** Split
`percolation_susceptibility()` into ladder arithmetic ($x \to p, L$) plus a core
`_susceptibility_at(L, p, n, dim, moment, geometry, rng, block_n)`. Do the same for the GPU
module (`_susceptibility_gpu_at`, which keeps the one `rng.integers(0, 2**63)` per call). The
existing models become thin callers, and their random streams must not change.
*Depends on:* T0.2. *Done when:* for fixed seeds, the old CPU and GPU models return
**bit-identical** arrays before and after the refactor. Record hashes first over
dim ∈ {2, 4}, a few $x$, moment ∈ {1, 2} and every geometry.

**T1.2 — CPU model `models/percolation_susceptibility_L.py`** (name: Q1).
`simulate(L, n, params, rng)`. Params: `dim`, `eps0`, `box_factor` ($c$), `nu_box`, `moment`,
`geometry`, `p_c`. $x = (L/c)^{1/\nu_{\text{box}}}$, $p = p_c - \varepsilon_0/x$, then the core
at $(L, p)$.
- It refuses $p \le 0$ (naming the smallest legal $L$), non-integer or $L < 2$, and
  `_check_size`. The same refuse-don't-clip policy as `p_at`.
- `cost_hint(L) = L**dim`; `declared_cost_exponent = dim`; `exponent_factor(params) = nu_box`
  (T1.4).
- The docstring records §1–§4, the date, and the letter-γ warning: $\gamma_L$ is not
  $\gamma/\nu$.

*Done when:* T1.5 (a)–(c), (e) pass.

**T1.3 — GPU model `models/percolation_susceptibility_L_gpu.py`.** Same parameters, calls
`_susceptibility_gpu_at`, `batched_cost=True`, imports its declarations from the CPU sibling
(the 2026-09-16 rule). *Done when:* T1.5 (a), (d), (f) pass.

**T1.4 — registry.** Add both to `tools/models.py`. Add one optional `ModelSpec` field,
`exponent_factor: Callable[[dict], float] | None = None` (physical exponent = factor ×
ladder exponent; `None` means 1). This is the only registry change.
*Done when:* every existing entry is unchanged and `test_artifacts.py` passes.

**T1.5 — tests** (`tools/tests/test_percolation_susceptibility_L.py`):
- (a) **Coincidence with the old model, bit for bit.** Where the old design's box is an exact
  integer, the two designs are the same rung. At $\nu_{\text{box}} = 1, c = 4$: new $L = 8$ ≡ old
  $x = 2$. At $\nu_{\text{box}} = 0.5, c = 2$: new $L = 8$ ≡ old $x = 16$. Same seed, identical
  arrays, on CPU and on GPU.
- (b) **The property the plan is about.** $L/x(L)^{\nu_{\text{box}}} = c$ to $10^{-12}$ on every
  rung of both §4 grids, and $p(L)$ equals hand-computed values.
- (c) Refusals: $p \le 0$, $L = 1$, $L$ past `_check_size`, non-integer $L$.
- (d) GPU equal in distribution to CPU (KS), on the pattern of
  `test_percolation_susceptibility_gpu.py`.
- (e) `cost_hint(L) == L**dim` exactly, and the declared exponent equals dim.
- (f) The GPU model consumes exactly one integer of the driver's rng per call.

**T1.6 — `calibration/exercise_all.py`.** Add both models to the expected-registry list
(≈ line 1431) and exercise `simulate`, `cost_hint` and each refusal branch.
*Done when:* the new rows PASS and no existing row changes.

### Phase 2 — calibration: how large must $c$ be? (decides Q5)

**T2.1 — `experiments/09_.../calibrate_box.py` (d = 4, GPU).** At fixed $p$ (fixed
$x \in \{16, 32, 64\}$), sweep integer $L$ so that $c = L/x^{0.69} \approx$ 3, 4, 5, 6, 8.
That is $L$ = 20, 27, 34, 41, 54 / 33, 44, 55, 66, 87 / 53, 71, 88, 106, 141, with the largest
at 8.9 GiB. Add a parity pair $L$ = 43, 45 beside 44 at $x = 32$. Calls the T1.1 core at
explicit $(L, p)$, records the **exact** $c$, uses one spawned stream per cell (ground rule 2),
has an `argparse` `--help`, and writes `data/calibration_d4/`.
Target se/$S$ ≈ 0.5% per cell. At the low pilot's 8·10⁸ sites/s that is about 5 min per
$x = 64$ cell and less below, **≈ 35–45 min on one 5090**. Seed 2026092201.
*Depends on:* T1.1.

**T2.2 — read it.** Fit $\log S$ vs $\log c$ per $x$, giving the local elasticity $G(c)$.
Acceptance, all numeric:
- **Independent confirmation of §1:** $G$ at $c \approx 4$ agrees with $0.162 \pm 0.023$
  within 2σ combined. **If not, STOP.** The zig-zag has another cause and this plan is
  wrong.
- Finite-size scaling: $G(c)$ agrees across the three $x$ within 2σ.
- Compute neutrality: $\mathrm{cv}^2 L^4$ is flat across $c$ within ±25% (d = 2 measured
  5.7, 6.7, 7.3 ·10⁴).
- Parity: the odd-$L$ residuals (43, 45) sit within 2σ of the even-$L$ fit.
  **If this fails, the √2 pilot grid must use even $L$ only** (8, 12, 16, 22, 32, 46,
  64, 90, 128, which still keeps $c$ exact). This is `tools/correction.py`'s srw staircase
  warning, checked here rather than assumed.

**T2.3 — choose $c$ and record it** in the experiment README. Rule: the smallest $c$ whose
drift bound $G(c)\cdot(\nu_{\text{box}} - \nu_{\text{low}})$, with $\nu_{\text{low}} = 0.6845 - 2(0.0023)$,
is at most ¼ of the target se. That target is the literature's 0.006 for now, or 0.0006
for "one more decimal". Weigh it against reach: $x_{\text{top}} = (128/c)^{1/0.69}$. Keep
$\nu_{\text{box}} = 0.69$: it is above $\nu$'s 2σ upper end (0.6891), as the model's
upper-bound rule requires.

### Phase 3 — the pipeline

**T3.1 — report the physical exponent.** Where `exponent_factor` is set:
- `pilot.py`, `autopilot.py` and `report.py` print $\gamma_{\text{susc}} = \nu_{\text{box}}\gamma_L$
  and $\omega_x = \nu_{\text{box}}\omega_L$ beside the ladder values.
- `answer.json` gains `gamma_physical`, `se_physical` and the Wilson bounds × factor.
- Every internal number stays in ladder units, because those are the units the article's
  theorems speak.

*Depends on:* T0.2, T1.4. *Done when:* (i) re-running `report.py` on an existing study
directory gives identical output apart from its `created` stamp (factor `None`); (ii) on a test study built from the
new model, the physical value equals `nu_box * gamma_L` to float precision.

**T3.2 — make the closed form refuse an inexact grid.** In `gamma_closed_form`, raise
unless $|\log L_k - k\log\rho| < 10^{-9}$ for every $k$, and point to `gamma_all_points`.
This closes the silent 0.009–0.018 error of §4. *Done when:* every existing recipe grid
still passes (they are exact $2^k$), a rounded √2 grid raises, and the full
`tools/tests/` run passes.

**T3.3 — the planner respects the top of the ladder.** Dry-run `plan.py` for
`percolation_susceptibility_L_gpu`, d = 4, and confirm it never proposes a top $L$ that
`_check_size` or memory refuses (only $L \le 128$ on $2^k$). If it does, add a pre-draw ceiling check to
`plan.py`; the cost probe's `climb_to_target` (`tools/cost_model.py:475`) is the precedent.
*Done when:* a plan asked for more precision than $L = 128$ allows says so, instead of
failing at draw time.

### Phase 4 — the d = 4 runs

**T4.1 — the $\omega_1$ pilot.** Recipe `samples_pilot_gsusc_L_d4.json`:
- model `percolation_susceptibility_L_gpu`, $c$ from T2.3, $\nu_{\text{box}} = 0.69$,
  $\varepsilon_0 = 0.09$, torus, moment 1;
- the √2 grid of §4 (even-only if T2.2's parity check failed);
- neyman at budget 1.2·10⁹ sites, 8 replicates, seed 2026092202;
- ≈ 70 min, like `pilot_gsusc_d4_low`.

*Done when:*
- local slopes change sign between successive windows nowhere by more than 2σ, where
  the old pilot did at 5–31σ;
- the one-correction fit converges with $\chi^2/\text{dof} \lesssim 2$ and se($\omega_1$) < $\omega_1/2$;
- report $\omega_x = \nu_{\text{box}}\omega_L$ beside $\Delta_1 \approx 0.77$ (reporting only,
  ground rule 4).

**T4.2 — the $\gamma$ production run.** `autopilot.py` on the $2^k$ grid, reusing T4.1's
constants, the same `--time 210m` budget as `autopilot_gsusc_d4_210m` (its final draw took
4.1 h), seed 2026092203.
*Done when:* both autopilot gates pass without `--force` (B_fs span over $\omega_1 \pm 1$ se
within 10×; the $\chi^2$ curvature test), and the report gives
$\gamma_{\text{susc}} = \nu_{\text{box}}\gamma_L$ with its eq. (720) interval. Compare it with
1.430(6) only at reporting time. If B_fs dominates because the window stops at $L = 128$,
record that and see T6.

**T4.3 (optional) — rescore the old run.** Recompute eq. (720) for
`autopilot_gsusc_d4_210m` with T4.1's $\omega_x$ and with $a_1$ converted by
$a_{1,x} = a_{1,L}\,c^{-\omega_L}$. It is approximate, since that run's $c_x$ varied by
rung, and must be labelled as such.

### Phase 5 — documentation

**T5.1** `models/README.md`: sections for both models. Note that
"no model file imports another" has not held since the susceptibility models imported
`percolation_zd`, and state the rule as it is actually practised.
**T5.2** `CATALOG.md`: module rows and the `ModelSpec.exponent_factor` field.
**T5.3** `experiments/09_.../README.md`: a new step "exact box ladder" with §1's evidence, T2's
table and decision, and T4's results with their acceptance checks.
**T5.4** `experiments/exponents_report.md`: the d = 4 γ and ω rows from T4.
**T5.5** `TODO.md`: the checkpoint.

### Phase 6 — deferred, only if T4.2 says the window is too shallow

**T6 — reach past $L = 128$.** Either port `plans/streaming_percolation.md` §3's
frontier sweep to the susceptibility (one new piece of bookkeeping: bank $s^2$ when a
component dies; the torus pinning already exists), or teach eq. (720) general OLS weights so
the production run can use the √2 grid up to $L = 181$. Each is its own plan.

---

## 6. Cost

About 35–45 min (T2.1) + 70 min (T4.1) + 3.5–4 h (T4.2) ≈ **6 h on one RTX 5090,
≈ 0.25 GPU-days**, plus test time for T1.5 (d). The largest sample is the calibration's
$L = 141$ at 8.9 GiB; the $2^k$ ladders stop at $L = 128$, 6.0 GiB.

## 7. Decisions for sign-off

| # | question | recommendation |
|---|---|---|
| Q1 | A new model, or a mode of the existing one? Name? | **New** (`percolation_susceptibility_L`, `_L_gpu`): the meaning of `scales` changes from $x$ to $L$, and a recipe must not change meaning through a parameter. Recorded runs stay reproducible (T1.1 is bit-identical) |
| Q2 | Scale unit: $L$ (integer) or a float $x$? | **$L$**: the pipeline casts scales to `int` in `pilot.py`, `autopilot.py` and `allocation.py`, and the $2^k$ grid then needs no change |
| Q3 | Where does $\gamma = \nu_{\text{box}}\gamma_L$ live? | **`ModelSpec.exponent_factor`**, applied only when reporting (T3.1) |
| Q4 | One grid or two? | **Two** (§4): √2 for the pilot, $2^k$ for production |
| Q5 | $c$? | **From T2.3's rule**, not chosen now |
| Q6 | $\nu_{\text{box}}$? | **Keep 0.69** (above $\nu$'s 2σ upper end) |
| Q7 | A CPU sibling, or GPU only? | **Both**: the CPU model is the reference for T1.5 (d) and runs without a GPU |
