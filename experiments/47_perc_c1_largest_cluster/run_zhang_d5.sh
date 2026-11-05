#!/bin/bash
# Zhang et al.'s 5D allocation on the largest cluster as the long run of a study
# (user, 2026-11-05): a pilot (the sqrt(2)-spaced d = 5 design of pipe_c1_d5), then the
# allocation drawn as 5 replicates of n/5, then the pipeline's Step 4 and estimates on it
# (fixed_run.py: the pipeline's own drivers need a rho^k ladder).
#   bash experiments/47_perc_c1_largest_cluster/run_zhang_d5.sh
set -u
cd "$(dirname "$0")/../.."
X=experiments/47_perc_c1_largest_cluster; D=$X/data; PY=.venv/bin/python
S=zhang_c1_d5; LOG=$D/$S.log
echo "$(date -Is) host $(hostname)" > $LOG
$PY -u src/study/pilot.py -meta $X/recipes/samples_pipe_c1_d5.json --study $S \
    --data-root $D --replicates 5 --seed 2026110550 >> $LOG 2>&1 || exit $?
$PY -u $X/fixed_run.py draw --study $D/$S --recipe $X/recipes/samples_zhang_c1_d5.json \
    --replicates 5 --seed 2026110551 >> $LOG 2>&1 || exit $?
$PY $X/fixed_run.py report --study $D/$S >> $LOG 2>&1 || exit $?
$PY $X/fixed_run.py plot --study $D/$S >> $LOG 2>&1
