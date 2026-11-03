#!/bin/bash
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/run_model.py --model lgbm --months $M --alts "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue s --h 5 --tag fast --features @reports/feats_fast.txt
run --venue f --h 5 --tag fast --features @reports/feats_fast.txt
run --venue f --h 1 --tag fast --features @reports/feats_fast.txt
echo QUEUE6_DONE
