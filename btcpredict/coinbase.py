"""Coinbase Exchange BTC-USD 1-minute candles (public API) -> 'Coinbase premium' features.

Mechanism: US institutional / ETF-related spot demand executes on Coinbase; cross-venue arbitrage
transmits it to Binance with a lag. The USD-vs-USDT level drifts slowly, so only deviations of the
premium from its own recent mean are used.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .config import SEC

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


def download(start: str, end: str, out: str = "coinbase_1m.parquet", sleep: float = 0.15) -> pd.DataFrame:
    s = requests.Session()
    t = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    rows = []
    while t < stop:
        t2 = min(t + pd.Timedelta(minutes=300), stop)
        for k in range(6):
            try:
                r = s.get(URL, params={"granularity": 60, "start": t.isoformat(), "end": (t2 - pd.Timedelta(seconds=1)).isoformat()},
                          timeout=20)
                if r.status_code == 429:
                    time.sleep(1 + k)
                    continue
                r.raise_for_status()
                rows += r.json()
                break
            except Exception:  # noqa: BLE001
                time.sleep(1 + 2 * k)
        t = t2
        time.sleep(sleep)
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]).drop_duplicates("time")
    df["ts"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("ts").drop(columns="time").reset_index(drop=True)
    (SEC / "coinbase").mkdir(parents=True, exist_ok=True)
    df.to_parquet(SEC / "coinbase" / out, index=False)
    return df


def features(panel: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """panel: needs ts, s_close (log), sigma. cb: Coinbase 1m candles (ts = open time)."""
    c = cb.set_index("ts")["close"].reindex(panel["ts"].values)
    v = cb.set_index("ts")["volume"].reindex(panel["ts"].values)
    lc = np.log(c.ffill().values)
    stale = c.isna().values                      # no Coinbase trade in that minute
    prem = lc - panel["s_close"].values
    sig = panel["sigma"].values
    P = pd.Series(prem)
    out = {}
    for k in (15, 60, 1440):
        out[f"cb_prem_dev{k}"] = (P - P.rolling(k, min_periods=5).mean()) / sig
    for k in (1, 5):
        out[f"cb_prem_chg{k}"] = P.diff(k) / sig
    L = pd.Series(lc)
    for k in (1, 5):
        out[f"cb_r{k}"] = L.diff(k) / (sig * np.sqrt(k))
    vv = pd.Series(v.fillna(0).values)
    out["cb_relvol15"] = np.log((vv.rolling(15, min_periods=1).mean() + 1e-6) / (vv.rolling(1440, min_periods=60).mean() + 1e-6))
    out["cb_share"] = np.log((vv.rolling(15, min_periods=1).sum() + 1e-6) / (panel["s_vol60"].astype(float).rolling(15, min_periods=1).sum().values + 1e-6))
    out["cb_stale"] = stale.astype(float)
    return pd.DataFrame(out, index=panel.index).astype("float32").replace([np.inf, -np.inf], np.nan)
