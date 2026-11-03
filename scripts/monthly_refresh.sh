#!/bin/bash
# Monthly refresh: pull new days, rebuild features, retrain production models on all data.
# Calibration and the online-layer state are kept from the last full holdout evaluation;
# re-run `python scripts/train_final.py` (2-3 h) every few months to re-validate them.
set -e
cd "$(dirname "$0")/.."
END=$(date -u -d "yesterday" +%F)
python -m btcpredict.download --start 2025-05-01 --end "$END" --workers 8 \
  --jobs spot,um,metrics,depth,premium,um1m,eth,sol,xrp,doge,bnb,btcfdusd,btcusdc,ethum,ada,avax,link,ltc,trx,sui,pepe,shib,dot,ton,near,bch
python -c "from btcpredict import coinbase; coinbase.download('2025-05-01', '$(date -u +%F)')"
python -m btcpredict.panel --start 2025-05-01 --end "$END"
python scripts/build_features.py 2025-05-01 "$END"
python scripts/train_final.py --variants full,spot --production_only
for f in models/*.txt; do gzip -9 -k -f "$f"; done
python scripts/check_live_parity.py 5
