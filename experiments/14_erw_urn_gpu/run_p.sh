#!/bin/bash
# One p of the 2h sieve (user, 2026-10-31: complete it to every multiple of 0.05),
# then the article's Step 1-4 figures for that study.
#   bash experiments/14_erw_urn_gpu/run_p.sh <p> <time> <seed> [tag]
set -u
cd "$(dirname "$0")/../.."
X=experiments/14_erw_urn_gpu; D=$X/data; PY=.venv/bin/python
P=$1; T=$2; SEED=$3; TAG=${4:-$T}; S=sweep_p${P}_$TAG
$PY -u $X/sweep_p.py --tag $TAG --time $T --p $P -- --seed $SEED > $D/driver_$S.log 2>&1 || exit $?
$PY experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study $D/$S >> $D/driver_$S.log 2>&1
