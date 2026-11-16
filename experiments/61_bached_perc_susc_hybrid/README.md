# 61 — Batched susceptibility, $d=5$ (hybrid CPU/GPU model)

Model: `models/percolation_susceptibility_hybrid.py` with `batch_factor`, `batch_exponent` (added 2026-11-13).
Theory: `derivations/batched_observable.tex`.

## Why (user, 2026-11-13)

The raw $Y_x=|C(0)|/p$ has $\sigma_x^2\sim x^{2\nu}$, so it breaks Assumptions `sigma_converges` and `2_plus_delta_moments`.
The batched observable $Z_x$ is the mean of $K_x=\lceil \texttt{batch\_factor}\cdot x^{\texttt{batch\_exponent}}\rceil$ independent copies of $Y_x$, each on its own lattice.
It has the same mean, so the same $\gamma$, $a_1$ and $\omega_1$, and $\operatorname{Var}(Z/\mathbb E Z)=\sigma_x^2/K_x$.

Not the explore model's axis seeds (`axis_factor`, `nu_axis`). Those sum several starting sites on one lattice, which changes the observable. A batch averages independent lattices.

## Acceptance criteria

- Model: $K=1$ (the defaults) is bit-for-bit the unbatched model; a batch is the mean of $K$ consecutive raw draws, however the call is split. Tests: `tools/tests/test_percolation_susceptibility_hybrid.py` (18 pass).
- Pilot: per-scale cv of $Z$ flat in $x$ (fitted growth exponent of cv consistent with 0), and $\mathbb E\zeta^3$ (`abs_moment`) flat in $x$ (Prop. `mom` of the note).

## Step 1 — pilot, scales 4…512, $n=10^6$ batches (purus, 2026-11-13)

- Recipe `recipes/samples_gsusc_d5_batched_n1m.json`: experiment 60's `gsusc_d5_hybrid_n25m` params ($\varepsilon_0=0.07$, cut $2^{22}$), same scales.
- `batch_exponent` $=0.77=2\hat\nu$, $\hat\nu=0.385$ fitted on the per-scale cv of experiment 60's $d=5$ pilot (independent data, ground rule 4). `batch_factor` $=1$: $K=3,5,9,15,25,42,72,122$.
  - The rounding of $K$ is at most 3% at $x=4$ and under 1% from $x=8$ on.
- Raw draws per replicate: $2.93\cdot10^8$, against $2\cdot10^8$ for experiment 60's pilot. Estimate from its timings: ~35 min a replicate.
- **Launch.** 2026-11-13 11:13, 5 replicates, seed 2026111361, cost re-probed (not reused: a batched sample costs $K$ raw ones). Log: `data/gsusc_d5_batched_n1m_pilot.log`. Code: working tree on top of `400bdf7`.
- **Result.** Finished 15:22, exit 0. Pilot $\hat\gamma=1.1806$; $\omega_1=0.514\pm0.088$, $a_1=-0.392\pm0.032$, cv $3.71$ (3.56–3.87 over the scales).
  - The cost probe hit its 4988 s cap after **one** probe. Its free affine fit gives $d=1.52\pm0.12$, but the constant $a$ is 101% of the cheapest scale's cost.
  - Local slopes of the per-$Z$ cost: 0.8–0.9 up to $x=64$, then **1.86 and 1.92** for 128→256→512. The free fit mixes the two regimes.

## Step 2 — long run, 10 h, $d$ composed (purus, 2026-11-16)

- **Why not the measured $d$ (user, 2026-11-16).** A batched sample costs $K_x\propto x^{0.77}$ raw ones, so $d=\hat d_{\rm raw}+\texttt{batch\_exponent}=1.0937+0.77=1.8637$.
  - $\hat d_{\rm raw}=1.0937\pm0.0072$ comes from experiment 60's $d=5$ probe on purus (4…4096).
  - This matches the probe's top local slopes. The se of the 0.77 is not recorded, so the plan's $d$ has no se.
- `plan.py --d` (added for this): stamps $d$ as a user override and refits the probe's $a+b\,i^d$ at that $d$ for the throughput: $6.98\cdot10^7$, against $1.17\cdot10^7$ at $d=1.52$.
  - At the run's scales (128…512) the refit overpredicts the probe's cost by 12–21%, so the run should finish under 10 h.
- **Plan.** `--time 10h --d 1.8636677795597638`. $m_0=6$, scales 128…4096, $n=112{,}644$ per scale, 3 replicates.
- **Predicted error.** $|$bias$|$ 0.0076, sd 0.0038 per replicate, RMSE ≈ 0.0079 on the answer. The bias rests on $\omega_1=0.514\pm0.088$; experiment 60's long run had 0.418.
- **Launch.** 2026-11-16 11:19, seed 2026111661, default numba threads (30, as the pilot). Log: `data/gsusc_d5_batched_n1m_run.log`. Code: working tree on top of `400bdf7`.
- **Result.** Finished 15:31, exit 0, in 251.9 min against 600 predicted (0.42×). The fixed-$d$ throughput was 2.4× low; the run's own pace is $\approx1.66\cdot10^8$ units/s at $d=1.8637$ on 128…4096.
  - $\hat\gamma=1.1798$. Eq. (720) bound: 95% $[1.0498,\,1.3098]$, dominated by $B_{fs}=0.122$, marked incomplete (no $\omega_2$ term). $t(2)$: $[1.1666,\,1.1930]$.
  - Replicates: 1.1737, 1.1826, 1.1831. Measured sd 0.00531, against the closed form's 0.00418.
  - $\omega_1$: joint AIC fit (pilot + long run) $0.305\pm0.174$; pilot $0.514\pm0.088$, pinned pilot refit 0.498.
  - cv of $Z$ over the scales: 3.74 → 4.07, a fitted growth exponent of about 0.02.
  - **Against the known values** (experiment 58: $\gamma=1.1792(7)$, $\Delta_1\approx0.41$): $\hat\gamma$ is 0.0006 off. Both intervals contain 1.1792, and both $\omega_1$ values are within 1 se of 0.41.
  - Experiment 60's raw run (same model, 4096…131072) gave $1.0385$, with a bound of $[0.953,\,1.124]$ that excludes 1.1792.
