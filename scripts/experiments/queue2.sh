#!/bin/bash
M=2025-11,2026-01,2026-03
for spec in "f 1" "s 1" "f 5" "s 5"; do
  set -- $spec
  python3 scripts/run_model.py --model lgbm --venue $1 --h $2 --months $M --tag v3alt --alts 2>&1 | grep -vE "^\s+fold"
done
