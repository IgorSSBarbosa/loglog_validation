#!/bin/bash
# User, 2026-11-11: purus only; d = 5 starts when d = 4 exits, whatever its status.
cd "$(dirname "$0")"
bash run_study.sh tau_leath_d4 9h 2026111158 9h --max-scale 16777216
bash run_study.sh tau_leath_d5 9h 2026111159 9h --max-scale 16777216
