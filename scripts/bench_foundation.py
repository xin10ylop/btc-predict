"""Zero-shot benchmark of time-series foundation models on 1-minute BTC candles.

Kronos (Shi et al. 2025, financial K-line foundation model) and Chronos-Bolt (Amazon) are asked to
forecast the next 1..5 one-minute closes from the previous 400 candles at randomly sampled
instants of the DEV test months. Compared on the SAME instants with our LightGBM OOS forecasts.

Requires: git clone https://github.com/shiyu-coder/Kronos  (path given by --kronos)
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from btcpredict import evaluate as E  # noqa: E402
from btcpredict.config import FEAT, REPORTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kronos", default="/tmp/claude-0/kronos")
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--months", default="2025-11,2026-01,2026-03")
ap.add_argument("--lookback", type=int, default=400)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--models", default="kronos,chronos")
a = ap.parse_args()

cols = ["ts", "s_open", "s_hi", "s_lo", "s_close", "s_vol60", "s_qv60", "y_s1", "y_s5"]
df = pd.read_parquet(FEAT / "panel.parquet", columns=cols)
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
mon = df["ts"].dt.strftime("%Y-%m")
cand = np.where(mon.isin(a.months.split(",")) & df["y_s5"].notna())[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(cand, a.n, replace=False))
ohlc = pd.DataFrame({"open": np.exp(df["s_open"]), "high": np.exp(df["s_hi"]), "low": np.exp(df["s_lo"]),
                     "close": np.exp(df["s_close"]), "volume": df["s_vol60"].astype(float), "amount": df["s_qv60"].astype(float)})
ohlc["high"] = ohlc[["high", "open", "close"]].max(1)
ohlc["low"] = ohlc[["low", "open", "close"]].min(1)
res = {}

if "kronos" in a.models:
    sys.path.insert(0, a.kronos)
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.set_num_threads(4)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pr = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    out1, out5 = [], []
    t = time.time()
    B = 16
    for b in range(0, len(idx), B):
        chunk = idx[b:b + B]
        dfs, xts, yts = [], [], []
        for i in chunk:
            w = ohlc.iloc[i - a.lookback + 1:i + 1].reset_index(drop=True)
            ts = pd.Series(df["ts"].iloc[i - a.lookback + 1:i + 1].values)
            dfs.append(w); xts.append(ts)
            yts.append(pd.Series(pd.date_range(ts.iloc[-1] + pd.Timedelta(minutes=1), periods=5, freq="1min")))
        preds = pr.predict_batch(df_list=dfs, x_timestamp_list=xts, y_timestamp_list=yts, pred_len=5, T=1.0,
                                 top_p=0.9, sample_count=a.samples, verbose=False)
        for i, p in zip(chunk, preds):
            c0 = ohlc["close"].iloc[i]
            out1.append(np.log(p["close"].iloc[0] / c0)); out5.append(np.log(p["close"].iloc[4] / c0))
        if b % 160 == 0:
            print(f"kronos {b}/{len(idx)} {time.time()-t:.0f}s", flush=True)
    res["kronos"] = (np.array(out1), np.array(out5))

if "chronos" in a.models:
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    lc = np.log(ohlc["close"].values)
    out1, out5 = [], []
    for b in range(0, len(idx), 256):
        chunk = idx[b:b + 256]
        ctx = torch.tensor(np.stack([lc[i - a.lookback + 1:i + 1] - lc[i] for i in chunk]), dtype=torch.float32)
        q, mean = pipe.predict_quantiles(ctx, prediction_length=5, quantile_levels=[0.5])
        med = q[:, :, 0].numpy()
        out1 += list(med[:, 0]); out5 += list(med[:, 4])
    res["chronos_bolt"] = (np.array(out1), np.array(out5))

# naive / our model on the same instants
lg = {}
for h in (1, 5):
    f = REPORTS / "preds" / f"lgbm_s{h}_v2_{a.months.replace(',', '-')}.parquet"
    if f.exists():
        p = pd.read_parquet(f).set_index("ts")["pred"]
        lg[h] = p.reindex(df["ts"].iloc[idx]).values
if lg:
    res["lgbm_v2(ours)"] = (lg.get(1), lg.get(5))
rows = []
for name, (p1, p5) in res.items():
    for h, p in ((1, p1), (5, p5)):
        if p is None:
            continue
        y = df[f"y_s{h}"].values[idx]
        m = E.metrics(y, np.asarray(p, float))
        rows.append({"model": name, "h": h, **{k: round(v, 4) for k, v in m.items() if k in ("n", "ic", "rank_ic", "hit", "auc", "hit_top10")}})
out = pd.DataFrame(rows)
print(out.to_string())
out.to_csv(REPORTS / "bench_foundation.csv", index=False)
