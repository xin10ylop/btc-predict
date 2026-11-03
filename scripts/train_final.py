"""Final pipeline: honest HOLDOUT walk-forward evaluation + production models + calibration.

For every variant/target:
  1. walk-forward over the 6 HOLDOUT months (never used for any design decision), monthly retrain
  2. calibration (P(up) logistic map, magnitude shrinkage beta, residual quantiles) from those
     out-of-sample predictions
  3. production model trained on ALL data
Outputs: models/*.txt, models/meta.json, reports/holdout_*.json, reports/preds/final_*.parquet

python scripts/train_final.py --variants full,spot --horizons 1,2,3,4,5
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict import calibrate as C  # noqa: E402
from btcpredict import evaluate as E  # noqa: E402
from btcpredict import features as F  # noqa: E402
from btcpredict import models as M  # noqa: E402
from btcpredict.config import FEAT, MODELS, REPORTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="full,spot")
ap.add_argument("--horizons", default="1,2,3,4,5")
ap.add_argument("--venues", default="f,s")
ap.add_argument("--rounds", type=int, default=400)
ap.add_argument("--params", default="{}")
ap.add_argument("--months", default="holdout")
ap.add_argument("--train_delay", type=int, default=None)
ap.add_argument("--skip_final", action="store_true")
a = ap.parse_args()

VARIANTS = {
    "full": {"features": "reports/feats_noaux.txt", "uses_futures": True, "venues": ["f", "s"]},
    "spot": {"features": "reports/feats_spotonly.txt", "uses_futures": False, "venues": ["s"]},
}

df = pd.read_parquet(FEAT / "panel.parquet")
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
X = pd.concat([pd.read_parquet(FEAT / "X.parquet"), F.extra_features(df), pd.read_parquet(FEAT / "Xalt.parquet")], axis=1)
months = E.HOLDOUT_MONTHS if a.months == "holdout" else a.months.split(",")
PRED = REPORTS / "preds"
meta_path = MODELS / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
params = json.loads(a.params)

for variant in a.variants.split(","):
    spec = VARIANTS[variant]
    cols = [c for c in open(spec["features"]).read().split() if c in X.columns]
    Xv = X[cols]
    vm = meta.setdefault(variant, {"uses_futures": spec["uses_futures"], "features": cols, "targets": {}})
    vm["features"] = cols
    for venue in [v for v in spec["venues"] if v in a.venues.split(",")]:
        for h in [int(x) for x in a.horizons.split(",")]:
            key = f"{venue}{h}"
            t0 = time.time()
            scale = F.target_scale(df, h)
            ycol = f"y_{venue}{h}" if a.train_delay is None else f"y_{venue}{h}_d{a.train_delay}"
            yz = df[ycol] / scale
            folds = E.make_folds(df["ts"], months, purge_minutes=h + 10)
            fp = M.lgbm_factory(params=params, rounds=a.rounds)
            p = E.walk_forward(fp, Xv, yz, folds, verbose=False)
            m = p.notna().values
            pd.DataFrame({"ts": df["ts"][m], "pred": p[m], "scale": scale[m]}).to_parquet(PRED / f"final_{variant}_{key}_{a.months}.parquet")
            res = E.summarize(f"{variant}_{key}", df, p, h, venue=venue, scale=scale)
            res["monthly_ic"] = E.monthly_ic(df["ts"], df[f"y_{venue}{h}"].values, p.values).round(4).to_dict()
            y = df[f"y_{venue}{h}"].values[m]
            z = p.values[m]
            pu = C.causal_logistic(z, y, horizon=h)
            res["calibration"] = C.logloss(pu, y)
            # --- calibration artifacts from all OOS predictions
            from sklearn.linear_model import LogisticRegression
            ok = np.isfinite(y) & (y != 0)
            lr = LogisticRegression(C=1e3).fit(z[ok].reshape(-1, 1), (y[ok] > 0).astype(int))
            yzs = y / scale.values[m]
            fin = np.isfinite(yzs)
            beta = float(np.sum(z[fin] * yzs[fin]) / np.sum(z[fin] ** 2))
            resid = yzs[fin] - beta * z[fin]
            absz = np.abs(z)
            dec = pd.qcut(absz, 10, labels=False, duplicates="drop")
            nz = np.isfinite(y) & (y != 0)
            hit_by = pd.Series(np.sign(z[nz]) == np.sign(y[nz])).groupby(dec[nz]).mean().round(4).tolist()
            entry = {
                "venue": venue, "h": h, "file": f"{variant}_{key}.txt",
                "calib": [float(lr.intercept_[0]), float(lr.coef_[0, 0])], "beta": beta,
                "resid_q": {str(q): float(np.quantile(resid, q)) for q in (0.025, 0.1, 0.5, 0.9, 0.975)},
                "absz_quantiles": np.quantile(absz, np.linspace(0.01, 0.99, 99)).round(5).tolist(),
                "hit_by_conf_decile": hit_by,
                "holdout": {k: v for k, v in res.items() if k not in ("calibration",)},
                "holdout_logloss_skill_pct": res["calibration"]["logloss_skill_pct"],
                "train_delay": a.train_delay,
            }
            # --- production model on all data
            if not a.skip_final:
                tr = np.where(np.isfinite(yz.values))[0]
                fp(Xv.iloc[tr], yz.iloc[tr], Xv.iloc[-5:], None)
                fp.last_model.save_model(str(MODELS / entry["file"]))
            vm["targets"][key] = entry
            meta_path.write_text(json.dumps(meta, indent=1, default=float))
            (REPORTS / f"holdout_{variant}_{key}.json").write_text(json.dumps(res, indent=1, default=float))
            short = {k: round(v, 4) for k, v in res.items() if isinstance(v, float)}
            print(variant, key, json.dumps(short), f"{time.time()-t0:.0f}s", flush=True)
            print("   logloss skill %:", round(res["calibration"]["logloss_skill_pct"], 3), " hit by conf decile:", hit_by, flush=True)
