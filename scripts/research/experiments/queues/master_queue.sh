#!/bin/bash
# Sequential experiment queue (no inter-script waiting).
cd "$(dirname "$0")/../.."
M=2025-11,2026-01,2026-03
run() { python3 scripts/research/run_model.py --model lgbm --months $M --alts "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue f --h 1 --tag v3alt
run --venue s --h 1 --tag v3alt
run --venue f --h 5 --tag v3alt
run --venue s --h 5 --tag v3alt
run --venue s --h 1 --tag spotonly --features @configs/feature_sets/spotonly.txt
run --venue f --h 1 --tag noaux --features @configs/feature_sets/noaux.txt
run --venue basis --h 1 --tag v3alt
run --venue s --h 5 --tag spotonly --features @configs/feature_sets/spotonly.txt
run --venue basis --h 5 --tag v3alt
run --venue f --h 5 --tag noaux --features @configs/feature_sets/noaux.txt
python3 scripts/research/experiments/run_tcn.py --venue f --months $M --T 180 --epochs 4 --static 40 2>&1
python3 scripts/research/experiments/run_tcn.py --venue s --months $M --T 180 --epochs 4 --static 40 2>&1
echo QUEUE_DONE
