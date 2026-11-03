#!/bin/bash
# after queue7 (PID $1): Coinbase premium in the satellite
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/research/run_model.py --model lgbm --months $M --alts --more Xcb.parquet --add cb_ "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue f --h 1 --tag nofd_cb --features @configs/feature_sets/nofd.txt
run --venue s --h 1 --tag nofd_cb --features @configs/feature_sets/nofd.txt
run --venue f --h 5 --tag nofd_cb --features @configs/feature_sets/nofd.txt
run --venue s --h 5 --tag nofd_cb --features @configs/feature_sets/nofd.txt
run --venue s --h 1 --tag spotnofd_cb --features @configs/feature_sets/spotonly_nofd.txt
echo QUEUE8_DONE
