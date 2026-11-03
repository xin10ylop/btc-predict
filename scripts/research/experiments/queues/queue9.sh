#!/bin/bash
# after queue8 (PID $1): satellite = nofd + Coinbase + wide swarm + ETH perp
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
M=2025-11,2026-01,2026-03
run() { python3 scripts/research/run_model.py --model lgbm --months $M --alts --more Xcb.parquet,Xalt2.parquet "$@" 2>&1 | grep -vE "^\s+fold"; }
run --venue f --h 1 --tag sat2 --features @configs/feature_sets/nofd.txt --add cb_,swarm2_,ethp_,eth_basis
run --venue s --h 1 --tag sat2 --features @configs/feature_sets/nofd.txt --add cb_,swarm2_,ethp_,eth_basis
run --venue f --h 5 --tag sat2 --features @configs/feature_sets/nofd.txt --add cb_,swarm2_,ethp_,eth_basis
run --venue s --h 5 --tag sat2 --features @configs/feature_sets/nofd.txt --add cb_,swarm2_,ethp_,eth_basis
run --venue s --h 1 --tag spotsat2 --features @configs/feature_sets/spotonly_nofd.txt --add cb_,swarm2_
run --venue s --h 5 --tag spotsat2 --features @configs/feature_sets/spotonly_nofd.txt --add cb_,swarm2_
echo QUEUE9_DONE
