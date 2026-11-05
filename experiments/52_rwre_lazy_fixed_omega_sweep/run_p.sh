#!/bin/bash
# One study of experiment 52 at one p, all on the host it runs on: time the window
# (d, throughput), build the reused pilot (19's n4e6 ladder, omega1 and a1 borrowed from
# 50's p = 0.04), autopilot on it, then the article's Step 1-4 figures (user, 2026-11-04).
#   bash experiments/52_rwre_lazy_fixed_omega_sweep/run_p.sh <p> <time> <seed> [tag]
# The window timing's seed is <seed> + 20. TW_N (default 16384) is the timing's n per
# scale; only the smoke test lowers it.
set -u
cd "$(dirname "$0")/../.."
X=experiments/52_rwre_lazy_fixed_omega_sweep; D=$X/data; PY=.venv/bin/python
P=$1; T=$2; SEED=$3; TAG=${4:-$T}
S=fixed_p${P}_${TAG}; LOG=$D/$S.log
echo "$(date -Is) host $(hostname) p=$P time=$T seed=$SEED tag=$TAG" > $LOG
$PY -u $X/time_window.py --p $P --tag $TAG --seed $((SEED + 20)) --n ${TW_N:-16384} \
    >> $LOG 2>&1 || exit $?
$PY $X/build_constants.py --p $P --tag $TAG >> $LOG 2>&1 || exit $?
$PY -u src/study/autopilot.py --reuse-pilot $D/pilot_p${P}_${TAG} --study $S \
    --data-root $D --time $T --replicates 5 --max-scale 8388608 --seed $SEED \
    --no-progress >> $LOG 2>&1 || exit $?
$PY experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study $D/$S >> $LOG 2>&1
RC=$?
echo "$(date -Is) done, plot_pipeline exit $RC" >> $LOG
exit $RC
