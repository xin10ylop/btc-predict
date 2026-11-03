"""Experiment 1: baselines vs LightGBM on DEV months (walk-forward, monthly retrain)."""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict import evaluate as E  # noqa: E402
from btcpredict import features as F  # noqa: E402
from btcpredict import models as M  # noqa: E402
from btcpredict.config import FEAT, REPORTS  # noqa: E402

t0 = time.time()
df = pd.read_parquet(FEAT / "panel.parquet")
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
X = F.model_matrix(df)
print("matrix", X.shape, f"{time.time()-t0:.0f}s", flush=True)
months = sys.argv[1].split(",") if len(sys.argv) > 1 else E.DEV_MONTHS
horizons = [int(h) for h in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 5]

rows = []
for h in horizons:
    scale = F.target_scale(df, h)
    yz = df[f"y_s{h}"] / scale
    folds = E.make_folds(df["ts"], months, purge_minutes=h + 10)
    cands = {
        "basis_dev60z": M.single_feature("basis_dev60z"),
        "ridge": M.ridge_factory(alpha=1e4),
        "lgbm": M.lgbm_factory(rounds=400),
    }
    for name, fp in cands.items():
        t = time.time()
        p = E.walk_forward(fp, X, yz, folds, verbose=False)
        r = E.summarize(name, df, p, h, scale=scale)
        r["secs"] = round(time.time() - t)
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        print("   monthly IC:", E.monthly_ic(df["ts"], df[f"y_s{h}"].values, p.values).round(3).to_dict(), flush=True)
        if name == "lgbm":
            imp = pd.Series(fp.last_model.feature_importance("gain"), index=fp.last_features).sort_values(ascending=False)
            print("   top features:", (imp / imp.sum()).head(25).round(3).to_dict(), flush=True)
pd.DataFrame(rows).to_csv(REPORTS / "exp_baseline.csv", index=False)
