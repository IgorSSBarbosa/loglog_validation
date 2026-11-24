#!/bin/bash
# One p of experiment 48: autopilot (pilot -> plan -> run -> report) on rwre_gpu_lazy,
# then the article's Step 1-4 figures. On a cap overflow, redo the study at the next
# cap with the same seed (41's fallback; cap changes no draw, only throughput).
#   bash experiments/48_rwre_lazy_psmall_sieve/run_p.sh <p> <time> <seed> [tag]
set -u
cd "$(dirname "$0")/../.."
X=experiments/48_rwre_lazy_psmall_sieve; D=$X/data; PY=.venv/bin/python
P=$1; T=$2; SEED=$3; TAG=${4:-$T}
for CAP in 32768 131072 262144 393216; do
  if [ $CAP = 32768 ]; then R=$X/recipes/samples_sieve_p$P.json; S=sieve_p${P}_$TAG
  else R=$X/recipes/samples_sieve_p${P}_cap$CAP.json; S=sieve_p${P}_${TAG}_cap$CAP; fi
  $PY -u src/study/autopilot.py -meta $R --study $S --data-root $D --time $T \
      --replicates 5 --seed $SEED --no-progress > $D/$S.log 2>&1
  code=$?
  if [ $code = 0 ]; then
    $PY experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study $D/$S >> $D/$S.log 2>&1
    exit $?
  fi
  grep -q "needed more than cap" $D/$S.log || exit $code
  echo "cap $CAP overflowed; next cap" >> $D/$S.log
done
exit 3
