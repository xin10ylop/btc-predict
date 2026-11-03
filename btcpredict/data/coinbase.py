"""Download Coinbase Exchange BTC-USD 1-minute candles (public API, no key).

    python -m btcpredict.data.coinbase --start 2025-05-01 --end 2026-11-03
Writes data/sec/coinbase/coinbase_1m.parquet. Features built from it: btcpredict/features/coinbase.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from ..config import SEC

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default=str(pd.Timestamp.now(tz="UTC").date()))
    a = ap.parse_args()
    d = download(a.start, a.end)
    print("coinbase minutes:", len(d))
