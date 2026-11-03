#!/bin/bash
# runs after the master queue (waits on its PID passed as $1)
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/run_model.py --model lgbm --months $M --alts --features @reports/feats_noaux.txt "$@" 2>&1 | grep -vE "^\s+fold"; }
# hyper-parameter variants on the perp 1-minute target
run --venue f --h 1 --tag hp_big --rounds 800 --params '{"num_leaves": 63, "min_data_in_leaf": 3000, "learning_rate": 0.03}'
run --venue f --h 1 --tag hp_slow --rounds 1200 --params '{"learning_rate": 0.015}'
run --venue f --h 1 --tag hp_l2 --rounds 400 --params '{"objective": "regression", "alpha": null}'
# latency-aware training: target measured from +2s
run --venue s --h 1 --tag lat2 --train_delay 2
run --venue f --h 1 --tag lat2 --train_delay 2
echo QUEUE5_DONE
