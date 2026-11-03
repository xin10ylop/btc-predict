"""Build all model-matrix files from the panel (run after `python -m btcpredict.panel`).

Writes data/features/{X, Xalt, Xalt2, Xcb}.parquet, row-aligned with the panel (ts >= START).
"""
import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict import altfeat, coinbase  # noqa: E402
from btcpredict import features as F  # noqa: E402
from btcpredict.config import FEAT, SEC  # noqa: E402

START = "2025-05-03"   # first two days are warm-up for 24h rolling statistics
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2025-05-01", "2026-11-02")
t = time.time()
panel = pd.read_parquet(FEAT / "panel.parquet")
panel = panel[panel["ts"] >= START].reset_index(drop=True)
F.model_matrix(panel).to_parquet(FEAT / "X.parquet")
print("X", round(time.time() - t), "s", flush=True)
for name, fn in (("Xalt", altfeat.build), ("Xalt2", altfeat.build2)):
    d = fn(start, end)
    d = panel[["ts"]].merge(d, on="ts", how="left")
    d.drop(columns="ts").to_parquet(FEAT / f"{name}.parquet")
    print(name, d.shape, round(time.time() - t), "s", flush=True)
cb = pd.read_parquet(SEC / "coinbase" / "coinbase_1m.parquet")
coinbase.features(panel[["ts", "s_close", "sigma", "s_vol60"]], cb).to_parquet(FEAT / "Xcb.parquet")
print("Xcb done", round(time.time() - t), "s")
