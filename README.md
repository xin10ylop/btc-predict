# btc-predict — 1-to-5-minute Bitcoin forecaster

Forecasts the BTC price change over the **next 1, 2, 3, 4 and 5 minutes**. It forecasts both
Binance spot (BTC/USDT) and the USD-M perpetual (BTCUSDT-PERP). Every minute it outputs a
calibrated **P(up)**, an expected move, 80%/95% bands, and a confidence percentile.

Everything here comes from walk-forward backtests: 18 months of 1-second data and roughly 790k
minute bars. No training row ever sees the future. The reported numbers come from a frozen
**holdout** (the 6 months 2026-05 → 2026-10). That holdout was not used for any design decision.

> **Read this first.** At 1-5 minute horizons, Bitcoin is close to a random walk. A real, durable
> edge here is a 53–56% hit rate on *all* minutes. The edge rises to 60–75% on the small subset of
> minutes where the model is most confident. The 1-minute edge is mostly a **seconds-scale
> lead-lag**, so it decays when you act late (see *Latency decay*). It is far smaller than taker
> fees (≈ 1–10 bp per side). Treat this as a research-grade signal, not a money printer. Any
> claim of more than about 56% on all 1-minute bars should be read as a leak.

RESULTS_PLACEHOLDER
