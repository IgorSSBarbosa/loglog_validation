#!/bin/bash
# Experiment 51, exploration arm: Ybar at x = 256, 512 in independent batches,
# each a separate generate.py run with its own seed, so a stop loses at most one.
#   NUMBA_NUM_THREADS=16 bash experiments/51_perc_gamma_susc_d3/run_local_slope.sh <first seed> <batches>
set -eu
cd "$(dirname "$0")/../.."
X=experiments/51_perc_gamma_susc_d3
SEED0=$1; B=$2
for b in $(seq 0 $((B - 1))); do
    echo "$(date -Is) batch $b seed $((SEED0 + b)) start"
    .venv/bin/python -u src/generate/generate.py -meta $X/recipes/samples_gsusc_d3_explore_ls.json \
        --tag explore_ls_256_512/b$b --seed $((SEED0 + b))
    echo "$(date -Is) batch $b done"
done
