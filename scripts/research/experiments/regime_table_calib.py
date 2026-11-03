"""P(up) from a (confidence decile x volatility regime) table of realized hit rates, fit causally,
vs the static logistic map.  Regime = vrel1d = log(sigma / 1-day median sigma) in 3 fixed buckets."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from btcpredict.config import FEAT, PREDS, RESEARCH  # noqa: E402

keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
variant = sys.argv[2] if len(sys.argv) > 2 else "full"
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{k}" for k in keys])
panel = panel[panel["ts"] >= "2025-05-03"].set_index("ts")
sig = panel["sigma"]
vrel = np.log(sig / sig.rolling(1440, min_periods=240).median())
EDGES = [float(x) for x in np.nanquantile(vrel[vrel.index < "2026-05-01"], [1 / 3, 2 / 3])]  # pre-holdout terciles
print("regime edges (vrel1d terciles, pre-holdout):", np.round(EDGES, 3))


def ll(p, y):
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    p, lab = np.clip(p[m], 1e-6, 1 - 1e-6), (y[m] > 0)
    base = lab.mean()
    return 100 * ((-np.mean(lab * np.log(base) + (~lab) * np.log(1 - base))) - (-np.mean(lab * np.log(p) + (~lab) * np.log(1 - p)))) / (-np.mean(lab * np.log(base) + (~lab) * np.log(1 - base)))


out = {"edges": EDGES}
for k in keys:
    h = int(k[1:])
    pr = pd.read_parquet(PREDS / f"final_{variant}_{k}_holdout.parquet").set_index("ts")
    z = pr["pred"].values
    y = panel.loc[pr.index, f"y_{k}"].values
    reg = np.digitize(vrel.loc[pr.index].values, EDGES)          # 0 calm, 1 normal, 2 stormy
    n = len(z)
    p_log = np.full(n, np.nan); p_tab = np.full(n, np.nan)
    for s in range(0, n, 1440):
        known = s - h
        if known <= 20000:
            continue
        lo = max(0, known - 200000)
        zz, yy, rr = z[lo:known], y[lo:known], reg[lo:known]
        ok = np.isfinite(yy) & (yy != 0)
        lr = LogisticRegression(C=1e3).fit(zz[ok].reshape(-1, 1), (yy[ok] > 0).astype(int))
        p_log[s:s + 1440] = lr.predict_proba(z[s:s + 1440].reshape(-1, 1))[:, 1]
        qs = np.quantile(np.abs(zz[ok]), np.linspace(0.1, 0.9, 9))
        dec = np.digitize(np.abs(zz[ok]), qs)
        corr = (np.sign(zz[ok]) == np.sign(yy[ok])).astype(float)
        tab = np.full((10, 3), 0.5)
        for d in range(10):
            for g in range(3):
                m = (dec == d) & (rr[ok] == g)
                # shrink toward the decile's all-regime hit rate (Bayesian, 500 pseudo-obs)
                prior = corr[dec == d].mean() if (dec == d).any() else 0.5
                tab[d, g] = (corr[m].sum() + 500 * prior) / (m.sum() + 500)
        dnew = np.digitize(np.abs(z[s:s + 1440]), qs)
        hit = tab[dnew, reg[s:s + 1440]]
        p_tab[s:s + 1440] = np.where(z[s:s + 1440] >= 0, hit, 1 - hit)
    r = {"logloss_skill": {"static_logistic": round(ll(p_log, y), 3), "decile_x_regime": round(ll(p_tab, y), 3)}}
    a = np.abs(z)
    top = a >= np.nanquantile(a, 0.9)
    for g, name in enumerate(("calm", "normal", "stormy")):
        m = top & (reg == g) & np.isfinite(y) & (y != 0) & np.isfinite(p_log)
        side = np.sign(z[m])
        r[f"top10_{name}"] = {"realized": round(float(np.mean(np.sign(y[m]) == side)), 4),
                              "static": round(float(np.mean(np.where(side > 0, p_log[m], 1 - p_log[m]))), 4),
                              "table": round(float(np.mean(np.where(side > 0, p_tab[m], 1 - p_tab[m]))), 4)}
        mm = (reg == g) & np.isfinite(p_log)
        r[f"skill_{name}"] = {"static": round(ll(p_log[mm], y[mm]), 3), "table": round(ll(p_tab[mm], y[mm]), 3)}
    out[k] = r
    print(k, json.dumps(r), flush=True)
(RESEARCH / f"regime_table_calib_{variant}.json").write_text(json.dumps(out, indent=1))
