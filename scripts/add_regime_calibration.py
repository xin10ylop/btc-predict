"""Add regime-aware calibration artifacts to models/meta.json from the holdout OOS forecasts.

Regime state (available live): vrel = log(sigma / 1-day rolling median sigma)
  * band_table: standardized residual quantiles per vrel bin -> bands keep ~nominal coverage in
    calm, normal and stormy markets (static bands: 76% calm vs 84% stormy)
  * hit_table:  realized hit rate per (confidence decile x regime tercile), shrunk to the decile
    mean -> descriptive "how often were calls like this right in this kind of market"
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from btcpredict.config import FEAT, MODELS, REPORTS  # noqa: E402

BAND_EDGES = [-0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6, 0.9]
meta_path = MODELS / "meta.json"
meta = json.loads(meta_path.read_text())
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{v}{h}" for v in "sf" for h in range(1, 6)])
panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
if (FEAT / "Ysub.parquet").exists():
    panel = pd.concat([panel, pd.read_parquet(FEAT / "Ysub.parquet")], axis=1)
panel = panel.set_index("ts")
sig = panel["sigma"]
vrel_all = np.log(sig / sig.rolling(1440, min_periods=240).median())
REG_EDGES = [float(x) for x in np.nanquantile(vrel_all[vrel_all.index < "2026-05-01"], [1 / 3, 2 / 3])]

for variant, vm in meta.items():
    vm["regime"] = {"vrel_window_min": 1440, "band_edges": BAND_EDGES, "regime_edges": REG_EDGES,
                    "regime_names": ["calm", "normal", "stormy"]}
    for key, m in vm["targets"].items():
        f = REPORTS / "preds" / f"final_{variant}_{key}_holdout.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f).set_index("ts")
        h = m["h"]
        y = panel.loc[pr.index, f"y_{key}"].values
        z = pr["pred"].values
        sc = pr["scale"].values
        vrel = vrel_all.loc[pr.index].values
        zres = (y - m["beta"] * z * sc) / sc
        ok = np.isfinite(zres)
        b = np.digitize(vrel, BAND_EDGES)
        table = []
        for j in range(len(BAND_EDGES) + 1):
            mm = ok & (b == j)
            src = zres[mm] if mm.sum() > 2000 else zres[ok]
            table.append([float(np.quantile(src, q)) for q in (0.025, 0.1, 0.5, 0.9, 0.975)])
        m["band_table"] = table
        reg = np.digitize(vrel, REG_EDGES)
        nz = np.isfinite(y) & (y != 0)
        qs = np.quantile(np.abs(z[nz]), np.linspace(0.1, 0.9, 9))
        dec = np.digitize(np.abs(z), qs)
        corr = (np.sign(z) == np.sign(y)).astype(float)
        hit = []
        for d in range(10):
            row = []
            md = nz & (dec == d)
            prior = corr[md].mean()
            for g in range(3):
                mg = md & (reg == g)
                row.append(round(float((corr[mg].sum() + 500 * prior) / (mg.sum() + 500)), 4))
            hit.append(row)
        m["hit_table"] = hit
        m["absz_deciles"] = qs.tolist()
        print(variant, key, "hit table top decile (calm/normal/stormy):", hit[-1])
meta_path.write_text(json.dumps(meta, indent=1, default=float))
print("regime edges", REG_EDGES)
