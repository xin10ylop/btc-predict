# What predicts Bitcoin 1–5 minutes ahead and why: research log

These notes record what was tried, what survived, what was rejected, and the market mechanism
behind each surviving signal. All numbers are out-of-sample walk-forward results on Binance
BTCUSDT. Unless stated otherwise, they come from three DEV test months (2025-11, 2026-01,
2026-03) with monthly retraining and purging.

The information coefficient (IC) is the correlation between forecast and realized log return. As
a rule of thumb, at 1 minute an IC of 0.05 corresponds to a hit rate of about 53%. The final
holdout results are in the README.

## 0. Ground truth about the problem

* A 1-minute BTC return has a standard deviation of about 4–10 bp. Taker fees are 1–10 bp per
  side, so even a perfect *direction* call rarely pays for a round trip at 1 minute. This system
  is a forecaster, not a turnkey trading bot.
* Binance spot BTCUSDT and the USD-M perpetual are 99.1% correlated over 1 minute. **They are not
  equally predictable.** The perp carries about 9× spot's volume and is where price discovery
  happens. Spot is partly a *lagged copy* of the perp, so it is more predictable.
* Most of the 1-minute edge lives in the **first seconds** of the next minute (see the latency
  decay chart). That is the signature of lead-lag and arbitrage, not of "predicting the market".

## 1. Signals that survived, with their mechanisms

| Signal | Mechanism | Evidence |
|---|---|---|
| **Perp–spot basis deviation** (perp minus spot price, relative to its own 1–10 min mean) | Informed flow and leverage hit the perp first; basis arbitrageurs drag spot along with a lag of a few seconds | Best single feature for spot +1m (IC 0.056). The *basis change itself* is predictable with IC 0.63 / R² 0.40 every month, i.e. mechanical convergence. Near zero for the perp target, because the perp is the leader. |
| **Order-flow imbalance of the perp in the last 3–10 s** | Splitting of large metaorders makes order signs autocorrelated, so flow continues for a few seconds | IC 0.04 (spot), 0.017 (perp) |
| **ETH perp vs BTC perp, last 3–30 s** | Crypto-wide shocks hit the higher-beta, thinner ETH market first; BTC reprices next | IC 0.021 (perp +1m), 0.010 (perp +5m), **positive in 19/19 months** |
| **17-coin "swarm breadth"** (share of alts rising over the last 3 s) | Each alt is a noisy witness of a market-wide shock, and their consensus averages out idiosyncratic noise | IC 0.029 (perp +1m), 0.0125 (perp +5m), positive in 17/19 months |
| **Coinbase premium** (Coinbase BTC-USD minus Binance, relative to its 15-min mean) | US institutional and ETF-related spot demand prints on Coinbase; cross-venue arbitrage carries it to Binance | IC 0.058 (spot +1m), 0.012 (perp +5m), **positive in 19/19 months** for the perp 5-min target |
| **Short-term reversal at 5 min** (Bollinger z, VWAP deviation, flow-driven moves) | Aggressive flow overshoots (temporary impact), then liquidity refills | IC −0.01 to −0.013 at 5 min, consistent with Kitron & Wengrowicz (2026) |
| **Volatility state** | The lead-lag is a larger share of total variance in quiet markets | Spot +1m IC is 0.16 in the low-vol tercile vs 0.06 in high vol |

## 2. Things that looked great and were rejected

* **BTC/FDUSD book features.** With them, the perp +1m IC went from 0.052 to 0.061, and they
  were the model's top two features (31% of total gain). Then BTC/FDUSD volume collapsed from
  ~120% of BTC/USDT to ~3% in Feb 2026, when the zero-fee era ended. The model kept trusting a
  stale book, and the perp +1m IC fell to 0.015 in March 2026. **Lesson: a lead-lag feature tied
  to one venue is only as stationary as that venue's market share.** All FDUSD features were
  removed. The core/satellite + online-RLS architecture was built so that the next such break is
  absorbed within days (in that March it recovered IC from 0.015 to 0.032).
* **The "alt-swarm +17%" result.** Once FDUSD was removed, the 5-coin swarm added ≈ 0 to the perp
  model (IC 0.053 vs 0.052). The cross-asset signals that *did* survive are the ones with broad,
  diversified sources: the Coinbase premium, the 17-coin swarm and the ETH perp ("satellite v2").
  * On the three tuning DEV months they add +0.008 IC to the perp model and +0.007 to spot.
    The paired day-block bootstrap gives p < 0.001 for both.
  * On the three **untouched** confirmation months the satellite beats the core on every target:

    | target | core IC | satellite IC |
    |---|---|---|
    | perp +1m | 0.039 | 0.050 |
    | perp +5m | 0.022 | 0.028 |
    | spot +1m | 0.049 | 0.060 |
    | spot +5m | 0.028 | 0.032 |

  * Its gains hold up in March 2026, the month the venue structure broke: perp +1m 0.050 vs
    0.038 for the core.
* **1-second temporal conv net (TCN)** on 180 s of multi-venue sequences. It matches LightGBM's
  hit rate but has a lower raw-return IC (0.040 vs 0.052). It is 79% correlated with LightGBM,
  and adding it *reduces* IC (−0.0036, day-block-bootstrap 95% CI excludes 0). This matches the
  literature: at this signal-to-noise ratio, trees win.
* **Structural decomposition** (spot = perp forecast − forecast basis change): IC 0.073 vs 0.075
  for the direct model, so no gain.
* **OKX BTC-USDT perpetual as a third price-discovery venue** (18 months of trades; OKX files
  are cut on UTC+8 days, verified against Binance with a lag-0 1-s return correlation of 0.86).
  * Univariate signals are tiny but stable: the price deviation vs Binance has IC ≈ 0.004 and is
    positive in 19/19 months; taker-flow imbalance has IC ≈ 0.017.
  * Incremental value on top of the satellite is only **+0.001 IC**. That is significant only
    for spot +1m (p = 0.014), not for the perp (p = 0.10) or at 5 minutes.
  * The two perps are tightly arbitraged at the 1-second level: their cross-correlations are
    symmetric, with no leader.
  * Rejected: it doesn't pay for another live dependency (about 2 s of extra fetch latency
    without websockets) and a full re-validation.
* **Literature signals added on top of microstructure:**
  * Sign-based reversal kernels, quarter-hour clock interactions: ΔIC +0.0008 (noise).
  * Lévy-area / path-signature lead-lag features: univariate IC ≈ 0, no gain.
* **Slow "regime" features at 5 min** (24-h return, hour-of-day, long/short ratios, depth
  imbalance). They had high gain importance but ΔIC = 0 when removed, which is the signature of
  memorizing periods. Removed for h ≥ 2.
* **Latency-aware targets** (training on returns measured from +2 s): no improvement at a 2 s
  delay.
* **Rolling 120-day training window** instead of an expanding window: worse (0.050 vs 0.053).
* **Time-series foundation models** (Kronos, Chronos-Bolt, TimesFM). The published benchmarks
  that were found show near-zero return IC for zero-shot use on crypto (Kronos paper:
  crypto-group IC ≈ 0.037 vs 0.045 for DLinear). The local Kronos benchmark was not run, because
  executing code cloned from GitHub was blocked in this environment.
  `scripts/research/experiments/bench_foundation.py` is ready if you want to run it yourself.

## 2b. The adaptive layer: where online learning helps and where it hurts

The ensemble re-weights [core, satellite] every minute from realized outcomes. Three versions were
compared on two pre-holdout datasets:
* the DEV months, which contain the FDUSD break;
* the untouched confirmation months, which contain no break.

| version | perp +1m, break months | perp +1m, normal months | perp +5m, normal months |
|---|---|---|---|
| satellite alone | 0.0611 | **0.0503** | **0.0279** |
| plain RLS, 2-day half-life, with intercept | 0.0644 | 0.0476 | 0.0193 |
| **discounted ridge, persistent prior "trust satellite", no intercept** | **0.0655** | 0.0495 | 0.0247 |

Three things follow from this table:
1. **Use a persistent prior.** Plain RLS forgets its prior and chases noise: at IC ≈ 0.05, two
   days of data carry very little evidence about the right weights. The persistent prior moves
   weight to the core only when the evidence accumulates, and it still rescued the FDUSD month
   (0.015 → 0.033).
2. **Drop the intercept.** An online intercept is a 3-day "recent drift" forecast, and BTC's
   multi-day drift has no 1–5 minute predictive value. Removing it helped everywhere.
3. **Adapt only at 1 minute.** At 5 minutes, consecutive outcomes overlap by 80%. The effective
   sample is about 5× smaller than it looks, so online weights become noise and the 5-minute
   satellite never broke in the first place. **Final rule: adaptive at h = 1, satellite-only
   at h ≥ 2.**

## 3. When is BTC most predictable? (DEV months)

| Slice | Spot +1m IC | Perp +1m IC |
|---|---|---|
| low-vol tercile | 0.163 | 0.131 |
| high-vol tercile | 0.063 | 0.055 |
| weekend | 0.098 | 0.080 |
| weekday | 0.072 | 0.058 |
| US session 13–21 UTC | 0.063 | 0.054 |
| Asia 00–08 UTC | 0.087 | 0.066 |

These slices fit the mechanism. Predictability is highest when there are fewer participants and
arbitrage is slower: weekends, Asia hours, and quiet markets.

## 3b. Why the holdout spot +1m result is so high: regime, not leakage

The spot +1m ensemble scored IC 0.102 and a 56.9% hit rate on the holdout. On the DEV months it
scored 0.079, and on the confirmation months 0.060. A jump like that has to be explained before it
can be believed.

* **The perp target did not jump** (holdout 0.063 vs DEV 0.062). Whatever changed is specific to
  spot.
* **Leak test.** If the Coinbase candles were shifted in time, their bar return would correlate
  about 0.9 with the *next* Binance minute. The measured correlation is 0.00–0.06 in every month,
  with no jump in the holdout.
* **Model-free evidence.** A single raw feature, the perp-spot basis deviation, shows the same
  regime pattern on its own:

  | period | 1-min vol (bp) | spot trades/min | model-free IC |
  |---|---|---|---|
  | May–Sep 2025 (quiet) | 2.4–3.6 | 1.5–2.5k | 0.075–0.096 |
  | Oct 2025–Feb 2026 (volatile; contains all DEV + confirmation months) | 4.2–7.2 | 2.8–4.6k | 0.014–0.043 |
  | May–Oct 2026 (holdout, quiet) | 2.9–5.0 | 1.6–3.0k | 0.047–0.082 |

**Conclusion.** Spot's lag behind the perp is most predictable in calm, thin markets. There, the
seconds-scale catch-up is a large share of the minute's variance and arbitrage is slower. By bad
luck, every design-phase month fell in the volatile stretch, the worst regime for this signal. So
the holdout number is genuine but regime-dependent. Expect a spot +1m hit rate of ~54% in volatile
months and ~57% in quiet ones, minus the latency decay.

## 3c. Regime change: what was built, what was rejected

The guiding question: does the system stay honest when the market changes character?

| Mechanism | Verdict | Evidence |
|---|---|---|
| Scale-free features (returns in units of current vol, flows in units of typical volume) | kept | a calm month and a wild month look alike to the model |
| Volatility-state features (vol vs its slow average, 5/60/240-min RV ratios) | kept | the forecast's own magnitude already shrinks in storms |
| Core + satellite + online prior-RLS (h=1) | kept | absorbs venue/structure breaks; FDUSD month perp IC 0.015 → 0.033 |
| New-York-hour band width factor (US equity open, 8:30 ET releases; DST-aware) | **kept** | fit May–Jul, tested Aug–Oct: open-hour coverage 75–78% → 78–81%; overall unchanged |
| Variance-ratio band factor (1-min variance / summed 1-s variance: trending vs mean-reverting seconds) | **kept** | out of sample: trending-tercile coverage 77–80% → 79–82%, mean-reverting 83% → 80% |
| Regime-conditional 80/95% bands (vol vs its 1-day median, 9 bins) | **kept** | coverage 78–81% in calm, normal and stormy markets vs 76–84% for static bands; also narrower |
| P(up) with vol-interaction terms / 3–14-day adaptive windows / confidence×regime table | rejected | no log-loss gain; adaptive windows chase noise |
| Rolling 120-day retraining instead of expanding | rejected | IC 0.050 vs 0.053 |

**Known residual weakness.** In *multi-day* storms (vol high relative to its 30-day median), the
spot +1m top-confidence decile is overconfident by about 6 percentage points: it claims 66% and
realizes 60%. The perp +1m is overconfident by about 2.5 pp. Neither the adaptive calibrators nor
the vol interactions fixed this. The live output therefore prints the regime alongside every
forecast.

## 3d. Holdout verdict (frozen May → Oct 2026)

* All **90/90** (target × month) ICs are positive. Satellite beats core on every target
  (p < 0.001).
* Full model, next minute:
  * perp +1m: IC 0.063, hit 54.4%
  * spot +1m: IC 0.102, hit 56.9% (calm regime; see 3b)
* Full model, next 5 minutes:
  * perp +5m: IC 0.030, hit 52.3%
  * spot +5m: IC 0.048, hit 53.4%
* The spot-only model (no futures data) matches the full model at 5 minutes and reaches 95% of
  it at 1 minute.

## 3e. A bug the live test caught

The first live run printed *identical* probabilities every minute. The LightGBM boosters had
been trained on unnamed arrays, and the predictor looked features up by name, so every input was
NaN. The holdout numbers were unaffected, because those forecasts used aligned arrays. The fix
stores each booster's exact feature list. The predictor now refuses to run on a count mismatch,
and the parity test compares live-path output with the booster's output on the training matrix
(exact match, e.g. −0.21760 vs −0.21760). **Lesson: a backtest is not a deployment. Always run
the real code path on real data before trusting it.**

## 3f. Training/serving skew: what the live test exposed

A backtest scores features computed from *final* historical data. A live system computes them
seconds after the bar close, from data that may not be final yet. The live run therefore logged
each feature vector at forecast time and compared it with the same minute recomputed minutes
later from complete data (`scripts/live/live_skew_now.py`).

| source of skew | symptom | fix | after the fix |
|---|---|---|---|
| **Coinbase revises recent candles.** One minute's volume read 18.5 BTC 76 s after the close and 26.3 BTC two minutes later. | The Coinbase-premium features (the #1 satellite feature) were wrong in 4 of 6 minutes; the 1-min model score moved by up to 1 score-SD. | Rebuild the last 15 minutes from raw trades (verified identical to final candles); use candles only once ≥ 15 min old. | 0 of 2,600 minutes mismatch |
| **Coinbase trade publication lag** (0.8–2 s) | Occasional last-minute close mismatch | Completeness barrier: the minute counts as closed once a trade stamped ≥ T is visible | — |
| **Binance 1 s klines are not final at T+0.5–1.0 s** | ETH and alt-coin seconds sometimes missing or changed | Fetch at T+2.0 s; data were final from T+1.5 s in every trial | Binance-side features identical; 4 of 5 minutes give bit-identical scores |

The first live run (13:51–14:33 UTC, 1-minute hit rate weak) used the skewed path, and its log is
kept as `reports/live/predictions_2026-11-03_run1_discarded_skew.jsonl`. **Lesson: parity on historical data is
necessary but not sufficient. Also check live-at-forecast-time against final data, because
exchanges revise and publish late.**

## 3g. Live forward test (2026-11-03, 15:26–16:41 UTC, spot-only model)

Seventy-five forecasts were made with the fixed data path, delivered at a median of 3.8 s after
each bar close. Results:
* hit rate: 58.2% at +10s, 58.1% at +1m, 52.8% at +5m;
* IC positive at all 7 horizons (0.13–0.31; +10s is 2.5 s.e. above zero);
* 80% band coverage 65–76%, during a trending afternoon.

This is consistent with the holdout, but far too short to estimate the level precisely.

## 4. Overfitting safeguards actually used

1. A 6-month **holdout** (2026-05 → 2026-10) that no design decision looked at.
2. Every design decision was made on DEV months, with a **paired day-block bootstrap** of the IC
   difference. A change is accepted only when its 95% CI excludes 0.
3. **Mechanism first.** A feature with no economic story is a suspect. Strong but unexplained
   gains (FDUSD) were traced to their cause before acceptance.
4. **Month-by-month stability** is required. Signals adopted late (Coinbase, ETH perp) are
   positive in 17–19 of 19 months.
5. Scale-free features: every return is in units of current volatility and every flow is in
   units of typical volume, so a calm month and a wild month look alike to the model.
6. Heavily regularized trees: ≥ 2000 rows per leaf, 50% feature subsampling, Huber loss, and
   targets clipped at ±6σ.
7. **Live/training parity tests.** The live code path reproduces the training features exactly
   (`scripts/live/check_live_parity.py`).
