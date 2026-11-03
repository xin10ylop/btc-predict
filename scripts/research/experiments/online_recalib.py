"""Static vs rolling-window (online) logistic recalibration, judged overall and in prolonged storms."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from btcpredict.config import FEAT, PREDS, RESEARCH  # noqa: E402

keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{k}" for k in keys])
panel = panel[panel["ts"] >= "2025-05-03"].set_index("ts")
sig = panel["sigma"]
v30 = np.log(sig / sig.rolling(43200, min_periods=1440).median())


def ll(p, y):
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    p, lab = np.clip(p[m], 1e-6, 1 - 1e-6), (y[m] > 0)
    b = lab.mean()
    l0 = -np.mean(lab * np.log(b) + (~lab) * np.log(1 - b))
    return 100 * (l0 + np.mean(lab * np.log(p) + (~lab) * np.log(1 - p))) / l0


out = {}
for k in keys:
    h = int(k[1:])
    pr = pd.read_parquet(PREDS / f"final_full_{k}_holdout.parquet").set_index("ts")
    z = pr["pred"].values; y = panel.loc[pr.index, f"y_{k}"].values
    storm = pd.qcut(v30.loc[pr.index].values, 3, labels=False) == 2
    n = len(z)
    P = {}
    for name, window in (("static_139d", 200000), ("roll_14d", 20160), ("roll_7d", 10080), ("roll_3d", 4320)):
        p = np.full(n, np.nan)
        for s in range(0, n, 1440):
            known = s - h
            if known <= 20000:
                continue
            lo = max(0, known - window)
            zz, yy = z[lo:known], y[lo:known]
            ok = np.isfinite(yy) & (yy != 0)
            lr = LogisticRegression(C=1e3).fit(zz[ok].reshape(-1, 1), (yy[ok] > 0).astype(int))
            p[s:s + 1440] = lr.predict_proba(z[s:s + 1440].reshape(-1, 1))[:, 1]
        P[name] = p
    a = np.abs(z); top = a >= np.nanquantile(a, 0.9)
    r = {}
    for name, p in P.items():
        m = top & storm & np.isfinite(y) & (y != 0) & np.isfinite(p)
        side = np.sign(z[m])
        r[name] = {"skill_all": round(ll(p, y), 3), "skill_storm": round(ll(p[storm], y[storm]), 3),
                   "top10_storm_pred": round(float(np.mean(np.where(side > 0, p[m], 1 - p[m]))), 4),
                   "top10_storm_realized": round(float(np.mean(np.sign(y[m]) == side)), 4)}
    out[k] = r
    print(k, json.dumps(r), flush=True)
(RESEARCH / "online_recalib.json").write_text(json.dumps(out, indent=1))
