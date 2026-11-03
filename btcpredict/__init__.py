"""btcpredict: 10-second to 5-minute Bitcoin forecasting (Binance spot & USD-M perpetual).

Pipeline stages = subpackages:
    data/      download raw market data           (Binance bulk files, Coinbase candles)
    features/  per-second bars -> features/targets (shared verbatim by training and live)
    modeling/  LightGBM models, online layer, calibration, walk-forward evaluation
    live/      real-time data sources, predictor, CLI  (python -m btcpredict.live)
    research/  experiments that were tested and NOT shipped (TCN, OKX)
"""

__version__ = "1.0.0"
