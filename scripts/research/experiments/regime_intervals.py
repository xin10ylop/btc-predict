"""80% band coverage by regime: static residual quantiles vs adaptive conformal (ACI)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from btcpredict.modeling import calibrate as C  # noqa: E402
from btcpredict.config import FEAT, MODELS, PREDS, RESEARCH  # noqa: E402

keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
meta = json.loads((MODELS / "meta.json").read_text())["full"]["targets"]
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{k}" for k in keys])
panel = panel[panel["ts"] >= "2025-05-03"].set_index("ts")
sig = panel["sigma"]
vrel_all = np.log(sig / sig.rolling(43200, min_periods=1440).median())
# a vol *shock*: realized |move| much larger than the forecast sigma predicted over the last 5 min
res = {}
for k in keys:
    h = int(k[1:])
    pr = pd.read_parquet(PREDS / f"final_full_{k}_holdout.parquet").set_index("ts")
    y = panel.loc[pr.index, f"y_{k}"].values
    sc = pr["scale"].values
    m = meta[k]
    center = m["beta"] * pr["pred"].values * sc
    lo_s = center + m["resid_q"]["0.1"] * sc
    hi_s = center + m["resid_q"]["0.9"] * sc
    cov_s = ((y >= lo_s) & (y <= hi_s)).astype(float)
    lo_a, hi_a, cov_a = C.aci_intervals(center, sc, y, horizon=h, alpha=0.2, gamma=0.005, warm=5000)
    vrel = vrel_all.loc[pr.index].values
    terc = pd.qcut(vrel, 3, labels=["calm", "normal", "stormy"]).astype(str)
    ok = np.isfinite(y) & np.isfinite(cov_a)
    r = {"static": {"all": round(float(cov_s[ok].mean()), 4)}, "aci": {"all": round(float(np.nanmean(cov_a[ok])), 4)}}
    for t in ("calm", "normal", "stormy"):
        mm = ok & (terc == t)
        r["static"][t] = round(float(cov_s[mm].mean()), 4)
        r["aci"][t] = round(float(np.nanmean(cov_a[mm])), 4)
    # coverage in the hour after a volatility shock (sigma jumps > 2x its 1-day median)
    shock = (sig / sig.rolling(1440, min_periods=60).median()).loc[pr.index].values > 2
    mm = ok & shock
    r["static"]["vol_shock"] = round(float(cov_s[mm].mean()), 4) if mm.any() else None
    r["aci"]["vol_shock"] = round(float(np.nanmean(cov_a[mm])), 4) if mm.any() else None
    r["avg_width_bp"] = {"static": round(float(np.mean(hi_s - lo_s)[()] * 1e4), 2), "aci": round(float(np.nanmean(hi_a - lo_a) * 1e4), 2)}
    res[k] = r
    print(k, json.dumps(r), flush=True)
(RESEARCH / "regime_intervals.json").write_text(json.dumps(res, indent=1))
