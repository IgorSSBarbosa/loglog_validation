#!/bin/bash
# Arm B of experiments 44 and 45, one recipe after another on ONE idle host
# (user, 2026-10-26: "use an idle machine"; xingu was idle when this was started).
# Each recipe's log goes next to its data: experiments/<exp>/data/<tag>.log.
cd "$(dirname "$0")/../.." || exit 1
PY=~/ClaudeWorkspace/loglog_validation/.venv/bin/python3
for r in experiments/44_lit_avena_thomann2012_rwre/recipes/protocol_*.json \
         experiments/45_lit_practice_erw/recipes/protocol_*.json; do
  exp=$(echo "$r" | cut -d/ -f2)
  tag=$(basename "$r" .json | sed 's/^protocol_//')
  mkdir -p "experiments/$exp/data"
  echo "$(date -Is) $(hostname) start $tag"
  $PY src/budget/protocol_arm.py -meta "$r" --tag "$tag" > "experiments/$exp/data/$tag.log" 2>&1
  echo "$(date -Is) $(hostname) exit $? $tag"
done
