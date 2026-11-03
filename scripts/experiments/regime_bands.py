"""Regime-conditional bands: residual quantiles conditional on vrel = log(sigma / 1-day median sigma).
Causal evaluation: refit daily on trailing realized residuals; compare coverage by regime."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from btcpredict.config import FEAT, MODELS, REPORTS  # noqa: E402

EDGES = np.array([-np.inf, -0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6, 0.9, np.inf])   # vrel bins
keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
meta = json.loads((MODELS / "meta.json").read_text())["full"]["targets"]
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{k}" for k in keys])
panel = panel[panel["ts"] >= "2025-05-03"].set_index("ts")
sig = panel["sigma"]
vrel_all = np.log(sig / sig.rolling(1440, min_periods=240).median())
res = {}
for k in keys:
    h = int(k[1:])
    pr = pd.read_parquet(REPORTS / "preds" / f"final_full_{k}_holdout.parquet").set_index("ts")
    y = panel.loc[pr.index, f"y_{k}"].values
    sc = pr["scale"].values
    center = meta[k]["beta"] * pr["pred"].values * sc
    zres = (y - center) / sc
    vrel = vrel_all.loc[pr.index].values
    b = np.digitize(vrel, EDGES[1:-1])
    n = len(y)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    q = None
    for s in range(0, n, 1440):
        known = s - h
        if known > 20000:
            w0 = max(0, known - 200000)
            zz, bb = zres[w0:known], b[w0:known]
            ok = np.isfinite(zz)
            q = {}
            for j in range(len(EDGES) - 1):
                m = ok & (bb == j)
                src = zz[m] if m.sum() > 2000 else zz[ok]
                q[j] = (np.quantile(src, 0.1), np.quantile(src, 0.9))
        if q is not None:
            for t in range(s, min(n, s + 1440)):
                ql, qh = q[b[t]]
                lo[t], hi[t] = center[t] + ql * sc[t], center[t] + qh * sc[t]
    cov = ((y >= lo) & (y <= hi)).astype(float)
    ok = np.isfinite(lo) & np.isfinite(y)
    m30 = np.log(sig / sig.rolling(43200, min_periods=1440).median()).loc[pr.index].values
    terc = pd.qcut(m30, 3, labels=["calm", "normal", "stormy"]).astype(str)
    shock = (sig / sig.rolling(1440, min_periods=60).median()).loc[pr.index].values > 2
    r = {"all": round(float(cov[ok].mean()), 4), **{t: round(float(cov[ok & (terc == t)].mean()), 4) for t in ("calm", "normal", "stormy")},
         "vol_shock": round(float(cov[ok & shock].mean()), 4), "avg_width_bp": round(float(np.nanmean(hi - lo) * 1e4), 2)}
    # final production table: per-bin quantiles on all OOS residuals
    okz = np.isfinite(zres)
    table = {}
    for j in range(len(EDGES) - 1):
        m = okz & (b == j)
        src = zres[m] if m.sum() > 2000 else zres[okz]
        table[j] = [float(np.quantile(src, x)) for x in (0.025, 0.1, 0.5, 0.9, 0.975)]
    res[k] = {"coverage_regime_bands": r, "table": table}
    print(k, json.dumps(r), flush=True)
res["edges"] = EDGES[1:-1].tolist()
(REPORTS / "regime_bands.json").write_text(json.dumps(res, indent=1))
