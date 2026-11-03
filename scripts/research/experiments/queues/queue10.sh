#!/bin/bash
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
python3 scripts/pipeline/train_final.py --variants full --venues f,s --horizons 1,5 --months 2025-12,2026-02,2026-04 --block 1 --skip_final 2>&1
echo CONFIRM_DONE
