"""Is the probability layer honest in every regime?  Static vs regime-aware calibration.

All calibrators are fit CAUSALLY: refit once a day on the trailing window of already-realized
outcomes, applied to the next day. Evaluated by log-loss skill overall and *within* each regime.

Regime state at forecast time (no look-ahead):
  vrel  = log(sigma / trailing-30-day median sigma)   (calm vs stormy relative to recent history)
  lsig  = log(sigma)                                   (absolute volatility level)
  wkend = weekend flag
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict.config import FEAT, PREDS, RESEARCH  # noqa: E402

keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
variant = sys.argv[2] if len(sys.argv) > 2 else "full"
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{k}" for k in keys])
panel = panel[panel["ts"] >= "2025-05-03"].set_index("ts")
sig = panel["sigma"]
vrel_all = np.log(sig / sig.rolling(43200, min_periods=1440).median())


def design(kind, z, vrel, lsig, wk):
    if kind == "static":
        return z[:, None]
    if kind == "vrel":
        return np.column_stack([z, z * vrel, vrel])
    if kind == "vrel+wk":
        return np.column_stack([z, z * vrel, vrel, z * wk, wk])
    if kind == "lsig":
        return np.column_stack([z, z * lsig, lsig])
    raise ValueError(kind)


def causal_fit_predict(kind, z, y, vrel, lsig, wk, h, refit=1440, window=200000, min_obs=20000):
    n = len(z)
    out = np.full(n, np.nan)
    lab = (y > 0).astype(int)
    ok = np.isfinite(y) & (y != 0) & np.isfinite(vrel) & np.isfinite(z)
    D = design(kind, z, vrel, lsig, wk)
    model = None
    for s in range(0, n, refit):
        known = s - h
        if known > min_obs:
            lo = max(0, known - window)
            m = ok[lo:known]
            if m.sum() > min_obs:
                model = LogisticRegression(C=1e3).fit(D[lo:known][m], lab[lo:known][m])
        if model is not None:
            out[s:s + refit] = model.predict_proba(np.nan_to_num(D[s:s + refit]))[:, 1]
    return out


def ll(p, y):
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    p, lab = np.clip(p[m], 1e-6, 1 - 1e-6), (y[m] > 0)
    base = lab.mean()
    l1 = -np.mean(lab * np.log(p) + (~lab) * np.log(1 - p))
    l0 = -np.mean(lab * np.log(base) + (~lab) * np.log(1 - base))
    return 100 * (l0 - l1) / l0


res = {}
for k in keys:
    h = int(k[1:])
    pr = pd.read_parquet(PREDS / f"final_{variant}_{k}_holdout.parquet").set_index("ts")
    z = pr["pred"].values
    y = panel.loc[pr.index, f"y_{k}"].values
    vrel = vrel_all.loc[pr.index].values
    lsig = np.log(sig.loc[pr.index].values)
    wk = (pr.index.dayofweek >= 5).astype(float)
    terc = pd.qcut(vrel, 3, labels=["calm", "normal", "stormy"]).astype(str)
    r = {}
    probs = {}
    for kind in ("static", "vrel", "vrel+wk", "lsig"):
        p = causal_fit_predict(kind, z, y, vrel, lsig, wk, h)
        probs[kind] = p
        r[kind] = {"all": round(ll(p, y), 3), **{t: round(ll(p[terc == t], y[terc == t]), 3) for t in ("calm", "normal", "stormy")}}
    # reliability of the top-confidence decile by regime (static vs vrel)
    rel = {}
    a = np.abs(z)
    top = a >= np.nanquantile(a, 0.9)
    for t in ("calm", "normal", "stormy"):
        m = top & (terc == t) & np.isfinite(y) & (y != 0)
        side = np.sign(z[m])
        realized = np.mean(np.sign(y[m]) == side)
        rel[t] = {"realized_hit": round(float(realized), 4)}
        for kind in ("static", "vrel"):
            p = probs[kind][m]
            pc = np.where(side > 0, p, 1 - p)
            rel[t][f"pred_{kind}"] = round(float(np.nanmean(pc)), 4)
    r["top_decile_reliability"] = rel
    res[k] = r
    print(k, json.dumps(r), flush=True)
(RESEARCH / f"regime_calibration_{variant}.json").write_text(json.dumps(res, indent=1))
