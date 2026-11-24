# The article, figure by figure

This file maps every figure and table of *The Log-Log Plot Technique* that comes from a
simulation to the run that produced it: the experiment directory, the study (one run of
the pipeline), the sampler, the recipe it drew, and the seed of its long run. Every
number below was checked against the study's own output files and the article source of
2026-11-19.

> **Draft.** Three things this file will point to are not in the repository yet; they are
> tracked in `TODO.md` ("Publication housekeeping"):
> the per-study result files (`pilot.json`, `plan.json`, `final.json`, `answer.json`,
> `report.md`), the exact command chain of each study, and the machine each one ran on.

## How to read a row

A **study** is one pass of the pipeline of the article's Section "Simulations":
pilot (Step 1) → allocation (Step 2) → long run (Step 3) → checks and estimate (Step 4)
→ joint fit of $\omega_1$ (Step 5). It lives in `experiments/<exp>/data/<study>/`.

| article step | command | writes |
|---|---|---|
| Step 1, pilot and cost probe | `src/study/pilot.py` | `pilot.json`, `constants.json` |
| Step 2, allocation (eqs. n-of-m0, bSE, bFS, m0-star) | `src/study/plan.py`, then `--accept` | `plan.json`, `recipes/samples_<study>_final.json` |
| Step 3, long run | `src/study/run.py --seed S` | `final.json` |
| Step 4, local slopes (A1) | `src/study/local_slope.py` | `local_slope.json` |
| Step 4, estimate and interval (Thm. cor:wilson) | `src/study/report.py` | `answer.json`, `report.md` |
| Step 5, joint fit of $\omega_1$ (eqs. aic, omega-avg) | `src/study/report.py` (same call) | `answer.json` → `joint_refit` |
| all steps at once | `src/study/autopilot.py` | all of the above, plus `autopilot.json` |

`src/study/README.md` documents each command; `src/README.md` lists every flag.

**Where each number lives.** In `answer.json`: $\hat\gamma$ is `gamma`, the 95% radius
is `wilson.half_width` and its finite-size part $\hat b_{\rm FS}$ is `wilson.B_fs`. The
Step 5 columns ($\hat\omega_1$, best range, $\chi^2$ $p$-value and dof) are in
`joint_refit`. $n$, the window and $R$ are in `final.json`.

**Recipes.** The long run of study `<study>` draws
`experiments/<exp>/recipes/samples_<study>_final.json`. `plan.py --accept` writes it, and
it fixes the model, its parameters, the window and $n$. The recipe does not carry the
seed (`"seed": null`): `run.py --seed S` draws replicate $k$ from child $k$ of `S`
(`tools/rng.py:spawn`), so the seed is listed in each row below.

**Devices.** `*_gpu` samplers need an NVIDIA GPU and `cupy` (the pinned version is part
of what a GPU seed means; see `requirements.txt`). The others run on the CPU through
numba. `percolation_susceptibility_hybrid` uses both: a cluster that would pass `cut`
sites on the CPU is regrown from its seed on the GPU, giving the same sample.

## Section "Simulations": the worked example (ERW, $p = 0.6$)

Every figure of the section comes from one study:
`experiments/46_pipeline_erwurn_gpu/data/pipeline_p0.6_1h_missouri`. The sampler is
`erw_urn_gpu` (GPU), the pilot is $488{,}519 \times 3$ on $4, \dots, 4096$, and the long
run is $102{,}166{,}094 \times 3$ on $2^{15}, \dots, 2^{20}$, with seed `20261102`.
`experiments/46_pipeline_erwurn_gpu/README.md` ("Commands") has the exact command.

| article | file | drawn by |
|---|---|---|
| Fig. `fig:pipeline-pilot` (Step 1) | `pipeline_pilot.pdf` | `experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study <study>` |
| Fig. `fig:bias-tradeoff` (Step 2) | `pipeline_allocation.pdf` | same call |
| Fig. `fig:pipeline-longrun` (Step 3) | `pipeline_longrun.pdf` | same call |
| Step 4 panels (A1), (A1)+(A2), (A4), (A5), (A6) | `pipeline_check_{A1,A1A2,A4,A5,A6}.pdf` | same call; (A1) needs `local_slope.json` |
| Fig. `fig:pipeline-aic` (Step 5) | `pipeline_aic.pdf` | `experiments/46_pipeline_erwurn_gpu/plot_aic.py --study <study>` |

Both scripts read a finished study directory only and write into it by default
(`--out DIR` to change that).

## Appendix "Experiments": percolation

Tables `tab:perc-gamma` ($\hat\gamma$ and its interval) and `tab:perc-omega` (Step 5) have
one row per study, and both tables come from the same ten studies. Every pilot and long
run has $R = 5$.

| row | experiment | study | sampler | device | long-run seed |
|---|---|---|---|---|---|
| $\gamma_{\rm susc}^\star$, $\mathbb Z^2$ | `53_perc_gamma_susc_d2` | `pilot_gsusc_d2_layers_batched` | `percolation_susceptibility_layers` (batched) | CPU | `2026111764` |
| $\gamma_{\rm susc}$, $\mathbb Z^3$ | `51_perc_gamma_susc_d3` | `gsusc_d3_explore` | `percolation_susceptibility_explore` | CPU | `2026110381` |
| $\gamma_{\rm susc}$, $\mathbb Z^4$ | `54_perc_gamma_d4_explore_pipeline` | `gsusc_d4` | `percolation_susceptibility_explore` | CPU | `2026110810` |
| $\gamma_{\rm susc}^\star$, $\mathbb Z^5$ | `61_bached_perc_susc_hybrid` | `gsusc_d5_batched_n1m` | `percolation_susceptibility_hybrid` (batched) | CPU + GPU | `2026111661` |
| $2-\tau$, $\mathbb Z^2$ | `56_perc_tau_leath_pipeline` | `tau_leath_d2_9h` | `percolation_tau_leath` | CPU | `2026111056` |
| $2-\tau$, $\mathbb Z^3$ | `56_perc_tau_leath_pipeline` | `tau_leath_d3_9h` | `percolation_tau_leath` | CPU | `2026111057` |
| $2-\tau$, $\mathbb Z^4$ | `57_perc_tau_leath_d4_d5` | `tau_leath_d4_9h` | `percolation_tau_leath` | CPU | `2026111158` |
| $2-\tau$, $\mathbb Z^5$ | `57_perc_tau_leath_d4_d5` | `tau_leath_d5_9h` | `percolation_tau_leath` | CPU | `2026111159` |
| $d_f$, $\mathbb Z^2$ | `47_perc_c1_largest_cluster` | `pilot_c1_d2_n10m` | `percolation_c1_gpu` | GPU | `2026111749` |
| $d_f$, $\mathbb Z^3$ | `47_perc_c1_largest_cluster` | `pilot_c1_d3_n1m` | `percolation_c1_gpu` | GPU | `2026111848` |

$\star$ marks the batched observable (Definition `def:batched`). A batched recipe has
`batch_factor` and `batch_exponent` in its `params`, and $n$ counts batches.

| article | file | drawn by |
|---|---|---|
| Fig. `fig:perc-exploration` | `perc_exploration.pdf` | `experiments/58_perc_gamma_d5_layers/plot_exploration.py` (an illustration on $\mathbb Z^2$, not a measurement) |

## Appendix "Experiments": Elephant Random Walk

Twenty-one studies, `experiments/14_erw_urn_gpu/data/sweep_p<p>_2h`, one per $p$. All use
the `erw_urn_gpu` sampler (GPU), a pilot of $488{,}519 \times 3$ on $4, \dots, 4096$, and
two hours per $p$. Tables `tab:erw-gamma` and `tab:erw-omega` have one row per study.

| $p$ | long-run seed | $p$ | long-run seed |
|---|---|---|---|
| 0.05 | `202610311400` | 0.55 | `202610311406` |
| 0.1 | `202610311401` | 0.6 | `130390443380176446595532203513909619517` |
| 0.15 | `202610311402` | 0.65 | `202610311407` |
| 0.2 | `18729442417724517007776518807298360528` | 0.7 | `46924049959556765079563375254726492849` |
| 0.25 | `202610311403` | 0.725 | `308474024117050983585185524709565981859` |
| 0.3 | `313294082553599930240445346426578684487` | 0.75 | `220133050324798733011077753885227403609` |
| 0.35 | `202610311404` | 0.775 | `245016123105541329762693281681067704189` |
| 0.4 | `86642809936569147804583196764930489165` | 0.8 | `267243965295085181000397022863118454730` |
| 0.45 | `202610311405` | 0.85 | `202610311408` |
| 0.5 | `231350868116599947770232765833748242043` | 0.9 | `60654767957214335027162632351927290636` |
| | | 0.95 | `202610311409` |

The long seeds are the entropy `numpy` drew when no `--seed` was given. The study records
it, so the run can be repeated like any other.

| article | file | drawn by |
|---|---|---|
| Fig. `fig:erw-paths` | `erw_paths.pdf` | `experiments/14_erw_urn_gpu/plot_paths.py` (CPU sampler `erw`; asserts each path ends where `erw.walk` does) |
| Fig. `fig:erw-gamma` | `erw_gamma_vs_p.pdf` | `experiments/14_erw_urn_gpu/plot_sweep.py --tag 2h --omega-loglog --article DIR` |
| Fig. `fig:erw-omega` | `erw_omega_vs_p.pdf` | same call |

`--omega-loglog` needs `src/estimate/omega_loglog.py` run on every study first.

## Appendix "Experiments": Random Walk in a dynamic Random Environment

The sampler is `rwre_gpu_lazy` (GPU) throughout.

Table `tab:rwre-long` comes from two studies in `experiments/50_rwre_lazy_long_omega/data/`:

| column | study | long-run seed | notes |
|---|---|---|---|
| $p = 0.04$ | `long_p0.04_24h` | `202611025000` | closed after 3 of 5 replicates (`run.py --close-partial`): the host went down |
| $p = 0.3$ | `long_p0.3_24h` | `202611025001` | |

Neither pilot gave a usable $\hat\omega_1$. Both plans use $\omega_1 = 0.2552 \pm 0.0232$,
fitted by eq. (pilot-fit) on experiment 48's local slopes at $p = 0.04$ over
$1024, \dots, 2^{21}$; `experiments/50_rwre_lazy_long_omega/README.md` documents this.
The row "last local slope" is read from each study's `local_slope.json`.

| article | file | drawn by |
|---|---|---|
| Fig. `fig:rwre-dynamics` (a) | `rwre_step.pdf` | a hand-drawn schematic; its script is in the article's `images/` folder, not here |
| Fig. `fig:rwre-dynamics` (b) | `rwre_dynamics.pdf` | `experiments/50_rwre_lazy_long_omega/plot_dynamics.py` (`--check N` compares its walks with the `rwre` sampler) |
| Fig. `fig:rwre-sweep` | `rwre_sweep.pdf` | `experiments/50_rwre_lazy_long_omega/plot_article.py --out DIR`: (a) and (b) from experiment 19's `n4e6` sweep ($8\cdot10^5 \times 5$, $p = 0, 0.05, \dots, 0.5$), and the long-run slopes from the two studies above |

## Repeating a run

The three levels below cost increasingly more.

1. **Check a number.** Read the study's `report.md`, or the fields of `answer.json`
   named above.
2. **Redraw a figure.** Run the script in the row with `--study` pointing at the study
   directory. This draws nothing new.
3. **Draw the long run again at the article's allocation.** With the study's
   `pilot.json`, `constants.json` and `plan.json` in `experiments/<exp>/data/<study>/`:

   ```bash
   python3 src/study/run.py    --study <study> --data-root experiments/<exp>/data --seed <seed>
   python3 src/study/report.py --study <study> --data-root experiments/<exp>/data
   ```

   With the same seed and the pinned versions of `requirements.txt`, the CPU samplers
   should draw the same samples. The GPU samplers draw the same samples only on the
   same `cupy`/CUDA version. Wall-clock times, and so any plan made from scratch, depend
   on the machine. To start from scratch, run the whole pipeline on the experiment's
   pilot recipe with `src/study/autopilot.py`: the allocation `plan.py` chooses will
   then fit your machine, not ours.
