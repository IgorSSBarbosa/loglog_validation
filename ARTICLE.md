# From the article to the repository

This page gives a quick view of where each simulation in *The Log-Log Plot Technique* lives in this
repository. Details are in the files it points to.

## The pipeline and its files

Every estimate in the article comes from one **study**. A study is one pass of the
five-step pipeline from the article's Section "Simulations":

| step | what it does | command | writes |
|---|---|---|---|
| Step 1, pilot | short run on small scales. The pilot fit gives a first estimate of $\omega_1$ and of the cost per sample | `src/study/pilot.py` | `pilot.json`, `constants.json` |
| Step 2, allocation | chooses $n$, the scales and the window for the time budget, balancing statistical error against finite-size bias | `src/study/plan.py`, then `--accept` | `plan.json`, the long-run recipe |
| Step 3, long run | draws the samples at that allocation | `src/study/run.py --seed S` | `final.json` |
| Step 4, checks and estimate | checks the assumptions (A1–A6) on the long run, then gives $\hat\gamma$ and its 95% interval | `src/study/local_slope.py`, `src/study/report.py` | `local_slope.json`, `answer.json`, `report.md` |
| Step 5, joint fit of $\omega_1$ | refits $\omega_1$ on the long run, choosing the range by AIC | `src/study/report.py` (same call) | `answer.json` |

`src/study/autopilot.py` runs all five steps in one call. `src/study/README.md`
explains each step, and `src/README.md` lists every flag.

**Where each number lives:**
- `answer.json` holds the article's results: $\hat\gamma$ (`gamma`), the 95% radius
  (`wilson.half_width`) with its finite-size part (`wilson.B_fs`), and the Step 5
  $\omega_1$ fit (`joint_refit`).
- `report.md` gives the same numbers in readable form.
- `final.json` records $n$, the scales, the number of replicates and the seed of the
  long run.

## Inside an experiment folder

Each experiment folder `experiments/<exp>/` asks one question. It contains:

- **`README.md`**: the question, the acceptance criteria and the measured results. Read it first.
- **`recipes/`**: the inputs, as JSON. `samples_<study>_pilot.json` or
  `samples_pilot.json` starts a study. `samples_<study>_final.json` is the long-run
  recipe that Step 2 accepted: model, parameters, scales and $n$. It does not hold the
  seed. Seed $S$ is given to `run.py`, and replicate $k$ draws from child $k$ of $S$.
- **`data/<study>/`**: everything the pipeline wrote for that study (the files in the
  table above), plus its plots.
- **plot scripts** (`plot_*.py`): they draw the article's figures from finished
  studies and draw nothing new.

## Where each part of the article lives

| article | folder | studies |
|---|---|---|
| Section "Simulations", the worked example (ERW, $p=0.6$): pilot, allocation, long run, checks, $\omega_1$ fit | `experiments/46_pipeline_erwurn_gpu` | `pipeline_p0.6_1h_missouri`; figures from `plot_pipeline.py` and `plot_aic.py` |
| Elephant Random Walk: $\hat\gamma$ and $\hat\omega_1$ against $p$ | `experiments/14_erw_urn_gpu` | `sweep_p<p>_2h`, one per $p$; figures from `plot_sweep.py`, sample paths from `plot_paths.py` |
| Percolation, susceptibility $\gamma$ on $\mathbb Z^2$ | `experiments/53_perc_gamma_susc_d2` | `pilot_gsusc_d2_layers_batched` |
| … on $\mathbb Z^3$ | `experiments/51_perc_gamma_susc_d3` | `gsusc_d3_explore` |
| … on $\mathbb Z^4$ | `experiments/54_perc_gamma_d4_explore_pipeline` | `gsusc_d4` |
| … on $\mathbb Z^5$ | `experiments/61_bached_perc_susc_hybrid` | `gsusc_d5_batched_n1m` |
| Percolation, cluster-size exponent $2-\tau$ on $\mathbb Z^2$, $\mathbb Z^3$ | `experiments/56_perc_tau_leath_pipeline` | `tau_leath_d2_9h`, `tau_leath_d3_9h` |
| … on $\mathbb Z^4$, $\mathbb Z^5$ | `experiments/57_perc_tau_leath_d4_d5` | `tau_leath_d4_9h`, `tau_leath_d5_9h` |
| Percolation, fractal dimension $d_f$ on $\mathbb Z^2$, $\mathbb Z^3$ | `experiments/47_perc_c1_largest_cluster` | `pilot_c1_d2_n10m`, `pilot_c1_d3_n1m` |
| Percolation, the picture of the exploration on $\mathbb Z^2$ | `experiments/58_perc_gamma_d5_layers` | `plot_exploration.py` (an illustration, not a measurement) |
| Random walk in a dynamic random environment: long runs at $p=0.04$ and $p=0.3$ | `experiments/50_rwre_lazy_long_omega` | `long_p0.04_24h`, `long_p0.3_24h`; figures from `plot_article.py` and `plot_dynamics.py` |
| … the sweep over $p$ in the same figure | `experiments/19_rwre_lazy_omega` | the `n4e6` sweep |

Studies marked "batched" in the article (the susceptibility on $\mathbb Z^2$ and
$\mathbb Z^5$) average blocks of samples before taking the log, so $n$ counts batches.
Their recipes carry `batch_factor` and `batch_exponent`.

Samplers named `*_gpu` need an NVIDIA GPU and `cupy`. The others run on the CPU.

## Repeating a run

1. **Check a number:** read the study's `report.md` or `answer.json`.
2. **Redraw a figure:** run the experiment's plot script on the study folder.
3. **Redraw the long run at the article's allocation:**

   ```bash
   python3 src/study/run.py    --study <study> --data-root experiments/<exp>/data --seed <S>
   python3 src/study/report.py --study <study> --data-root experiments/<exp>/data
   ```

   With the same seed and the versions pinned in `requirements.txt`, the CPU samplers
   should draw the same samples. The GPU samplers do so only with the same `cupy`/CUDA
   version.

To apply the method to your own model, run `src/study/autopilot.py` on a pilot recipe.
The allocation then fits your machine and your time budget. `models/README.md` explains
how to add a model.
