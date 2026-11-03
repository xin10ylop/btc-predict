"""Score logged live forecasts against realized prices (Binance spot 1m klines; perp if reachable)."""
import json
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from btcpredict.config import ROOT  # noqa: E402

rows = [json.loads(l) for l in open(ROOT / "logs" / "predictions.jsonl")]
recs = []
for r in rows:
    for f in r["forecasts"]:
        recs.append({"T": pd.Timestamp(r["T"]), "spot": r["spot"], "perp": r["perp"], **f})
d = pd.DataFrame(recs)
t0 = int(d["T"].min().value // 10**6)
t1 = int((d["T"].max() + pd.Timedelta(minutes=6)).value // 10**6)
k = []
s = t0
while s < t1:
    x = requests.get("https://data-api.binance.vision/api/v3/klines",
                     params={"symbol": "BTCUSDT", "interval": "1m", "startTime": s, "endTime": t1, "limit": 1000}, timeout=10).json()
    if not x:
        break
    k += x
    s = x[-1][0] + 60000
close = pd.Series({pd.Timestamp(int(x[0]), unit="ms") + pd.Timedelta(minutes=1): float(x[4]) for x in k})
d = d[d["venue"] == "s"].copy()
d["realized_bp"] = [1e4 * np.log(close.get(T + pd.Timedelta(minutes=h), np.nan) / p) for T, h, p in zip(d["T"], d["h"], d["spot"])]
d = d.dropna(subset=["realized_bp"])
d = d[d["realized_bp"] != 0]
print(d.groupby("h").apply(lambda g: pd.Series({"n": len(g), "hit": np.mean(np.sign(g.exp_ret_bp) == np.sign(g.realized_bp)),
                                                 "ic": np.corrcoef(g.exp_ret_bp, g.realized_bp)[0, 1] if len(g) > 2 else np.nan,
                                                 "in_80band": np.mean((g.realized_bp >= g.lo80_bp) & (g.realized_bp <= g.hi80_bp))}),
                         include_groups=False).round(3))
