# Agent guide: btc-predict, from A to Z

You are working on **btc-predict**, a system that forecasts Bitcoin's price change over the next
**10 s, 15 s, 1, 2, 3, 4 and 5 minutes**, every minute. It forecasts Binance spot BTC/USDT and
the Binance USD-M perpetual. Your goal is to understand it fully before changing anything, and to
keep it honest: no look-ahead, no tuning on the holdout, and every claim backed by an
out-of-sample number.

## 1. Read these first (in order, ~15 minutes)
1. `README.md`: what it does, the headline results, how to run it.
2. `docs/ARCHITECTURE.md`: data flow, time conventions, modules, models, live loop, invariants.
3. `docs/FINDINGS.md`: the research log. It covers what works and why, what was rejected and
   why, regime change, and training/serving skew.
4. `models/README.md`, `configs/feature_sets/README.md`, `reports/README.md`: artifacts.

## 2. Repo map
```
btcpredict/            the package, one subpackage per pipeline stage
  config.py            every path & constant
  data/                1. download: binance.py (bulk files -> per-second bars), coinbase.py
  features/            2. features & targets: seconds.py, panel.py, matrix.py, crossasset.py,
                          coinbase.py, subminute.py   (same code runs in training AND live)
  modeling/            3. gbm.py (LightGBM), online.py (PriorRLS ensemble), calibrate.py, evaluate.py
  live/                4. sources.py (REST/replay/cache), predictor.py, cli.py  -> python -m btcpredict.live
  research/            rejected experiments (TCN neural net, OKX venue), kept for reference
configs/feature_sets/  which columns each model uses
models/                production boosters (*.txt.gz) + meta.json (calibration, features, online state)
scripts/pipeline/      build_features, train_final (holdout + production), add_regime_calibration, monthly_refresh
scripts/live/          score_live_log, check_live_parity, live_skew_now, check_live_skew
scripts/reports/       results_table, make_report (charts), edge_vs_costs
scripts/research/      run_model.py (one walk-forward experiment) + experiments/ (historical)
reports/               holdout/ (headline numbers), confirm/, figures/, dev/, research/, live/
tests/                 unit tests for the core math
data/, logs/           local only (not in git): raw/per-second data, feature matrices, live logs
```

## 3. Ten concepts you must know
1. **Forecast instant.** Row `ts` = bar open; forecasts are made at `T = ts + 60 s` using only
   data before `T`. `P(T)` = last trade before T.
2. **Venues.** `s` = Binance spot BTCUSDT, `f` = USD-M perp. The perp leads; spot catches up
   within seconds.
3. **Targets.** Log return from T to T + horizon. Models learn `z = return / (sigma·sqrt(h))`,
   which is scale-free.
4. **Two variants.**
   * `full`: needs Binance futures.
   * `spot`: no futures data, about 95% as good; use it where futures are geo-blocked.
5. **Core + satellite.**
   * Core: LightGBM on BTC/ETH microstructure.
   * Satellite: core + cross-venue signals (Coinbase premium, 17-coin swarm, ETH perp, USDC book).
6. **Online ensemble.** `PriorRLS` blends core and satellite, learns from live outcomes at
   h ≤ 1 m, and falls back to the core if a venue relationship breaks (that happened with
   BTC/FDUSD in Feb 2026).
7. **Calibration.**
   * P(up) is logistic in z.
   * Bands are residual quantiles conditioned on the volatility regime, New-York hour and
     variance ratio.
8. **Evaluation.**
   * DEV months are for decisions, confirmation months for checking them.
   * The **HOLDOUT (2026-05 → 2026-10) is frozen** and is the source of the headline numbers.
   * Changes are accepted only with a paired day-block bootstrap.
9. **Latency.** The edge decays within seconds. The CLI fetches at T + 2.0 s (exchange data are
   final from ~1.5 s) and delivers at ~3 s. Report "hit @ +2s/+5s", not only zero-delay numbers.
10. **Training/serving skew.**
    * Live features must equal features recomputed later from final data.
    * Coinbase revises recent candles, so recent minutes are rebuilt from trades.
    * Check with `make parity` and `make skew`.

## 4. Commands (`make help`)
```
make install            pip install -e .
make test               unit tests (fast, no data needed)
make live VARIANT=spot  live forecaster (VARIANT=full needs Binance futures access)
make replay T="2026-10-15 13:37"   one forecast from local historical data
make data / features / train / calibrate / reports   full rebuild (~20 GB, several hours)
make parity             live feature & score path == training (run after ANY feature change)
make skew               live-at-forecast-time vs final data on the real market
make refresh            monthly retrain on new data
```

## 5. Golden rules
* **No look-ahead.** Features use seconds ≤ T−1 s only. `y_*` / `px_*` columns are targets, never
  features.
* **One code path.** Change features only in `btcpredict/features/`; live reuses them. Then run
  `make test` and `make parity`.
* **Booster column order = `meta.json` `features` list.** Never reorder feature lists for
  existing models.
* **Never use the holdout to choose anything.** Decide on DEV months, confirm on confirmation
  months, then report the holdout once.
* **Mechanism first.** Every kept signal has a market reason and is stable month by month.
  Unexplained gains are suspects (see the FDUSD story).
* **Be honest about costs.** Edges are ~1 bp versus 4–10 bp taker fees. This is a forecasting
  signal, not a standalone trading bot.
* **Shell hygiene.** Don't `pkill -f <pattern>` when that pattern is in your own command line;
  kill by PID.

## 6. How to do common tasks
* **Add a feature:**
  1. Implement it in `features/` (array-based, using seconds ≤ `end` only).
  2. Rebuild the matrices (`scripts/pipeline/build_features.py`).
  3. Test it with `scripts/research/run_model.py` on DEV months against a baseline, and compare
     with `evaluate.paired_block_bootstrap`.
  4. If it wins and is stable month by month, add it to a `configs/feature_sets` file, run
     `make train calibrate parity`, and update the README tables.
* **Add a horizon:** add its target (see `features/subminute.py` for seconds), then run
  `train_final.py --horizons <h>` and `add_regime_calibration.py`. The predictor and CLI pick it
  up from `meta.json`.
* **Add a live data source:** implement it in `RestSource`, `ReplaySource` *and* `CachedSource`
  in `live/sources.py`, then run `make parity` and `make skew`.
* **Investigate a live problem:** `scripts/live/score_live_log.py`, `logs/predictions.jsonl`,
  `logs/live_features_*.csv`, then `scripts/live/check_live_skew.py` once the official daily
  files exist.

## 7. Key results to know (holdout, full model; details in README)
* Hit rate: spot +10s **62.4%**, spot +1m **56.9%**, perp +1m **54.4%**, spot +5m **53.4%**,
  perp +5m **52.3%**.
* The top-10% most confident calls hit 57–77%.
* All 126 target-months have positive IC.
* The holdout was a calm period. In volatile months, expect ~53–54% at 1 minute.
* Live forward test (75 min, spot model): all 7 horizons had positive IC.

## 8. Known limits
* The full variant was not live-tested from the build environment (futures REST geo-blocked).
* Bands under-cover in strongly trending sessions.
* In multi-day storms, top-confidence 1-minute calls are ~6 pp overconfident.
* Git history contains ~200 MB of old prediction caches from early commits; squash-merge to drop
  them.
