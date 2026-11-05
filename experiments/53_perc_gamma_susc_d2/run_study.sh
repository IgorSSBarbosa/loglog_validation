#!/bin/bash
# One study of experiment 53: autopilot (pilot -> plan -> run -> report), then the
# article's Step 1-4 figures.
#   bash experiments/53_perc_gamma_susc_d2/run_study.sh <recipe stem> <time> <seed> <tag> [autopilot args...]
set -u
cd "$(dirname "$0")/../.."
X=experiments/53_perc_gamma_susc_d2; D=$X/data
# Python 3.12 hosts (orinoco) cannot use the 3.10 .venv: PY=.venv-py312/bin/python
PY=${PY:-.venv/bin/python}
K=$1; T=$2; SEED=$3; S=${1}_$4; shift 4
$PY -u src/study/autopilot.py -meta $X/recipes/samples_$K.json --study $S --data-root $D \
    --time $T --replicates 5 --seed $SEED --no-progress "$@" > $D/$S.log 2>&1 || exit $?
$PY experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study $D/$S >> $D/$S.log 2>&1
