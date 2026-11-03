#!/bin/bash
# after the master queue (PID $1): everything rebased on the no-FDUSD feature set
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/run_model.py --model lgbm --months $M --alts "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue f --h 1 --tag nofd --features @reports/feats_nofd.txt
run --venue s --h 1 --tag nofd --features @reports/feats_nofd.txt
run --venue s --h 1 --tag spotnofd --features @reports/feats_spotonly_nofd.txt
run --venue f --h 1 --tag nofd_roll120 --train_days 120 --features @reports/feats_nofd.txt
run --venue s --h 5 --tag fastnofd --features @reports/feats_fast_nofd.txt
run --venue f --h 5 --tag fastnofd --features @reports/feats_fast_nofd.txt
run --venue s --h 5 --tag nofd --features @reports/feats_nofd.txt
run --venue f --h 1 --tag nofd_lat2 --train_delay 2 --features @reports/feats_nofd.txt
run --venue s --h 1 --tag nofd_lat2 --train_delay 2 --features @reports/feats_nofd.txt
run --venue f --h 1 --tag nofd_hpbig --rounds 800 --params '{"num_leaves": 63, "min_data_in_leaf": 3000}' --features @reports/feats_nofd.txt
echo QUEUE7_DONE
