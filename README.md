# btc-predict — 1-to-5-minute Bitcoin forecaster

Every minute, this forecasts how BTC will move over the **next 1, 2, 3, 4 and 5 minutes**. It
covers both Binance spot (BTC/USDT) and the USD-M perpetual. Each forecast includes:
* a calibrated **P(up)**
* an expected move
* 80% and 95% bands
* a confidence percentile, with the historical hit rate at that confidence level

The models were built from 18 months of one-second data (~790k minute bars) across 25 instruments
and 3 venues. The headline numbers come from a **frozen 6-month holdout (May → Oct 2026)** that
was not used for any design decision.

> **Read this first.** At 1–5 minute horizons Bitcoin is close to a random walk.
> * A real, durable edge here looks like a **53–55% hit rate on all minutes**. It rises to
>   **60–70% on the most confident 1–10%**.
> * The 1-minute edge is mostly **seconds-scale lead-lag**. Its value halves if you act ~2
>   seconds late (see *Latency*).
> * It is smaller than taker fees (~1–10 bp per side). Treat this as a high-quality *signal* —
>   for market making, execution timing, or short-horizon prediction markets — not as a
>   standalone money printer.
>
> Anyone showing much more than ~56% on *all* 1-minute bars almost certainly has a leak.

RESULTS_SECTION

## How it works

```
 Binance spot BTC 1s ─┐                              ┌─ CORE model (LightGBM)
 Binance perp trades ─┤   second-level features      │  BTC spot/perp + ETH microstructure:
 ETH spot 1s ─────────┤   at each bar close:         │  perp-spot basis deviation, flow
                      ├─► returns, order flow, ─────►│  imbalance, realized vol, reversal ...
 17 alt-coins 1s ─────┤   basis, swarm breadth,      │
 ETH perp trades ─────┤   Coinbase premium ...       └─ SATELLITE model (LightGBM)
 Coinbase BTC-USD ────┤   (all scale-free: in units     core + cross-venue/cross-asset:
 BTC/USDC book ───────┘    of current vol / volume)      Coinbase premium, 17-coin swarm,
                                                         ETH-perp lead, USDC book
                                    │
                                    ▼
     h = 1:  online discounted ridge on [core, satellite], persistent prior "trust the
             satellite", learns from every realized outcome -> falls back to the core
             within days if a cross-venue relationship breaks
     h >= 2: satellite (overlapping outcomes make online weights too noisy)
                                    │
                                    ▼
     calibration: P(up) (logistic), E[r] (shrinkage), bands (residual quantiles),
                  confidence percentile + historical hit rate per confidence decile
```

## What actually predicts Bitcoin at 1–5 minutes, and why

Every signal kept has a market mechanism and is stable month by month. Full log, including
everything rejected: [`docs/FINDINGS.md`](docs/FINDINGS.md).

1. **The perp leads spot.** The perpetual trades ~9× spot's volume, so informed flow hits it
   first, and basis arbitrageurs drag spot along within seconds. The perp–spot basis deviation is
   the #1 feature for spot. The basis change itself is predictable with IC 0.63.
2. **The Coinbase premium.** US institutional and ETF-related demand prints on Coinbase first.
   When Coinbase trades rich vs its own 15-minute norm, Binance follows. It is the top feature for
   both targets, and its IC is positive in **19/19 months** for the perp 5-minute target.
3. **The crypto "swarm" and the ETH perp.** Market-wide shocks hit thinner, higher-beta markets
   first. When ETH perp outruns BTC perp, or most of 17 alts tick up in the last 3 seconds, BTC
   follows. Each is positive in 17–19 of 19 months.
4. **Order flow continuation (seconds), then reversal (minutes).** Metaorder splitting makes
   flow persist for a few seconds. Aggressive moves overshoot and partially revert over 5
   minutes.
5. **Volatility state.** The same signals are 2–3× more predictive in quiet markets, on weekends
   and in Asia hours. That fits the mechanism: fewer participants and slower arbitrage.

**Things that looked great and were rejected:**
* BTC/FDUSD book features: their predictive power died when FDUSD's market share collapsed in Feb
  2026, which motivated the core/satellite design.
* A 1-second neural net (TCN).
* Foundation-model forecasts (from published benchmarks).
* Path-signature (Lévy-area) features.
* Clock / quarter-hour effects.
* Slow "regime" features at 5 minutes. They had high importance but zero contribution, which is a
  sign of memorization.

## Quick start

```bash
pip install -r requirements.txt
```

1. Data. This takes about 45 minutes and ~25 GB, all from public endpoints with no API keys.

```bash
python -m btcpredict.download --start 2025-05-01 --end 2026-11-02 --workers 8
```

```bash
python -m btcpredict.download --start 2025-05-01 --end 2026-11-02 --jobs sol,xrp,doge,bnb,btcfdusd,btcusdc,ethum,ada,avax,link,ltc,trx,sui,pepe,shib,dot,ton,near,bch
```

```bash
python -c "from btcpredict import coinbase; coinbase.download('2025-05-01','2026-11-03')"
```

2. Features.

```bash
python -m btcpredict.panel --start 2025-05-01 --end 2026-11-02
```

```bash
python scripts/build_features.py
```

3. Holdout evaluation and production models.

```bash
python scripts/train_final.py --variants full,spot
```

4. Run live. Each forecast is computed about 2 seconds after the bar close.

```bash
python -m btcpredict.run_live --variant full
```

If Binance futures is unavailable where you are, use the spot-only variant:

```bash
python -m btcpredict.run_live --variant spot
```

Replay any historical minute offline:

```bash
python -m btcpredict.run_live --variant full --replay "2026-10-15 13:37"
```

5. Score your live log against realized prices.

```bash
python scripts/score_live_log.py
```

Example output:

```
=== 2026-11-03 14:02:00 UTC   BTC spot 77,158.01   1m vol 2.7bp ===
target    P(up)  E[r] bp      80% band (bp)             price 80% band  conf%  hist hit
spot+1m   0.571    +0.62 [   -3.6,   +4.9] [   77,130.2,   77,195.8]     93    0.629
...
```

## Repository layout

| path | what |
|---|---|
| `btcpredict/download.py` | bulk data from data.binance.vision → compact per-second bars |
| `btcpredict/secfeat.py` | second-level features at each bar close (shared by training and live) |
| `btcpredict/altfeat.py` | alt-coin swarm, cross-book, ETH-perp features |
| `btcpredict/coinbase.py` | Coinbase premium features |
| `btcpredict/panel.py`, `features.py` | minute-level panel, scale-free model matrix, targets |
| `btcpredict/evaluate.py` | purged walk-forward, metrics, paired day-block bootstrap |
| `btcpredict/online.py` | online discounted ridge (adaptive layer) |
| `btcpredict/live.py`, `run_live.py`, `predictor.py` | live data sources, caching, forecaster CLI |
| `scripts/run_model.py` | single walk-forward experiment (cached OOS predictions) |
| `scripts/train_final.py` | holdout evaluation + production models + calibration |
| `scripts/check_live_parity.py` | proves live features == training features |
| `docs/FINDINGS.md` | the research log: mechanisms, evidence, rejected ideas |
