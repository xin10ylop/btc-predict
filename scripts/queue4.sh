#!/bin/bash
while pgrep -f "queue[23].sh" >/dev/null; do sleep 10; done
python3 scripts/run_tcn.py --venue f --months 2025-11,2026-01,2026-03 --T 180 --epochs 4 --static 40 2>&1
python3 scripts/run_tcn.py --venue s --months 2025-11,2026-01,2026-03 --T 180 --epochs 4 --static 40 2>&1
