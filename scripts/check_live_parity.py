"""Verify that the live feature builder reproduces the training features exactly.

For random historical instants T, rebuild the feature row from the raw inputs a live process would
have had (minute klines for history, per-second bars for the recent window) via
btcpredict.live.build_features(ReplaySource) and compare with the training matrix.
Also checks that futures 1m klines (um1m files) equal our aggTrade-derived minute aggregates.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict import features as F  # noqa: E402
from btcpredict import live  # noqa: E402
from btcpredict.config import FEAT, SEC  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
panel = pd.read_parquet(FEAT / "panel.parquet")
panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
X = pd.concat([pd.read_parquet(FEAT / "X.parquet"), F.extra_features(panel), pd.read_parquet(FEAT / "Xalt.parquet")], axis=1)
rng = np.random.default_rng(1)
idx = rng.choice(np.arange(5000, len(panel) - 10), n, replace=False)
src = live.ReplaySource()
worst = {}
for i in idx:
    T = panel["ts"].iloc[i] + pd.Timedelta(minutes=1)
    t = time.time()
    dfh, A = live.build_features(src, T, use_futures=True, last_n=1)
    Xl = live.model_inputs(dfh, A).iloc[-1]
    Xt = X.iloc[i]
    cols = [c for c in Xt.index if c in Xl.index]
    a, b = Xl[cols].astype(float), Xt[cols].astype(float)
    both_nan = a.isna() & b.isna()
    err = ((a - b).abs() / (b.abs() + 1e-3)).where(~both_nan, 0).fillna(1.0)
    bad = err[err > 2e-3].sort_values(ascending=False)
    print(f"T={T}  {len(cols)} cols  max rel err {err.max():.2e}  n_bad={len(bad)}  {time.time()-t:.1f}s", flush=True)
    for c, e in bad.head(8).items():
        print(f"     {c:24s} live={a[c]:.6g} train={b[c]:.6g}")
        worst[c] = max(worst.get(c, 0), e)
    missing = [c for c in Xt.index if c not in Xl.index]
    if missing:
        print("   missing in live:", missing[:20])
print("columns with any mismatch:", sorted(worst, key=worst.get, reverse=True)[:30])

# klines vs aggTrade aggregation (futures)
day = "2026-03-10"
k = pd.read_parquet(SEC / "um1m" / f"{day}.parquet")
s = pd.read_parquet(SEC / "um" / f"{day}.parquet")
g = np.arange(86400) // 60
agg = pd.DataFrame({"v": s["v"].groupby(g).sum(), "bv": s["bv"].groupby(g).sum(), "c": s["c"].groupby(g).last()})
print("klines vs aggTrades  max|dv|", float(np.max(np.abs(k["volume"].values - agg["v"].values))),
      " max|dbv|", float(np.max(np.abs(k["taker_buy_volume"].values - agg["bv"].values))),
      " max|dclose|", float(np.nanmax(np.abs(k["close"].values - agg["c"].values))))
