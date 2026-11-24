#!/bin/bash
# One long run of experiment 50: autopilot on a reused pilot (build_pilots.py), then the
# article's Step 1-4 figures. The pooled kernel at cap 393216 (pilot recipe): cap costs
# only page-table bytes there, so there is no cap fallback (user, 2026-11-02).
#   bash experiments/50_rwre_lazy_long_omega/run_p.sh <p> <time> <seed> [tag]
set -u
cd "$(dirname "$0")/../.."
X=experiments/50_rwre_lazy_long_omega; D=$X/data; PY=.venv/bin/python
P=$1; T=$2; SEED=$3; TAG=${4:-$T}
S=long_p${P}_${TAG}
$PY -u src/study/autopilot.py --reuse-pilot $D/pilot_p${P}_pool --study $S \
    --data-root $D --time $T --replicates 5 --max-scale 8388608 --seed $SEED \
    --no-progress > $D/$S.log 2>&1 || exit $?
$PY experiments/46_pipeline_erwurn_gpu/plot_pipeline.py --study $D/$S >> $D/$S.log 2>&1
