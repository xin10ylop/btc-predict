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
  `scripts/bench_foundation.py` is ready if you want to run it yourself.

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
   (`scripts/check_live_parity.py`).
