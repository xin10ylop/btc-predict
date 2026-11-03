#!/bin/bash
while pgrep -f queue2.sh >/dev/null; do sleep 10; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/run_model.py --model lgbm --months $M --alts "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue s --h 1 --tag spotonly --features @reports/feats_spotonly.txt
run --venue f --h 1 --tag noaux --features @reports/feats_noaux.txt
run --venue basis --h 1 --tag v3alt
run --venue s --h 5 --tag spotonly --features @reports/feats_spotonly.txt
run --venue basis --h 5 --tag v3alt
run --venue f --h 5 --tag noaux --features @reports/feats_noaux.txt
