"""Walk-forward training of the cross-venue TCN on 1-second sequences (multi-horizon).

python scripts/research/experiments/run_tcn.py --venue s --months 2025-11,2026-01,2026-03 --T 180 --epochs 4
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
from btcpredict.modeling import evaluate as E  # noqa: E402
from btcpredict.features import matrix as F  # noqa: E402
from btcpredict.research import tcn as seqmodel  # noqa: E402
from btcpredict.research import tcn_data as seqdata  # noqa: E402
from btcpredict.config import DEV, FEAT, PREDS  # noqa: E402
from btcpredict.modeling.gbm import Standardizer  # noqa: E402

PRED = PREDS
ap = argparse.ArgumentParser()
ap.add_argument("--venue", default="s")
ap.add_argument("--months", default="2025-11,2026-01,2026-03")
ap.add_argument("--T", type=int, default=180)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--hid", type=int, default=32)
ap.add_argument("--static", type=int, default=40, help="number of top LGBM features fed as static input (0=none)")
ap.add_argument("--train_rows", type=int, default=400000)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

df = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma"] + [f"y_{a.venue}{h}" for h in range(1, 6)] +
                     [f"y_{a.venue}{h}_d{d}" for h in range(1, 6) for d in (1, 2, 5)])
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
ch, t0 = seqdata.load()
ch = np.ascontiguousarray(ch)
ends = seqdata.end_index(df["ts"], t0)
assert ends.max() < len(ch)
Y = np.stack([df[f"y_{a.venue}{h}"] / F.target_scale(df, h) for h in range(1, 6)], 1).astype(np.float32)

S = None
if a.static:
    imp = pd.read_csv(DEV / f"imp_lgbm_{a.venue}1_v2.csv", index_col=0).iloc[:, 0]
    cols = list(imp.index[: a.static])
    X = pd.read_parquet(FEAT / "X.parquet")
    Xe = F.extra_features(pd.read_parquet(FEAT / "panel.parquet").query("ts >= '2025-05-03'").reset_index(drop=True))
    X = pd.concat([X, Xe], axis=1)[cols]
    del Xe

months = a.months.split(",") if a.months not in ("dev", "holdout") else (E.DEV_MONTHS if a.months == "dev" else E.HOLDOUT_MONTHS)
folds = E.make_folds(df["ts"], months, purge_minutes=15)
preds = np.full((len(df), 5), np.nan, dtype=np.float32)
for f in folds:
    t = time.time()
    tr = f.train[ends[f.train] - a.T >= 0]
    tr = tr[np.isfinite(Y[tr, 0])]
    tr = tr[-a.train_rows:]                      # most recent rows
    Str = Ste = None
    if S is not None or a.static:
        st = Standardizer().fit(X.iloc[tr])
        Str = st.transform(X.iloc[tr]).astype(np.float32)
        Ste = st.transform(X.iloc[f.test]).astype(np.float32)
    p, net = seqmodel.fit_predict_seq(ch, ends[tr], Str, Y[tr], ends[f.test], Ste, T=a.T, epochs=a.epochs,
                                      hid=a.hid, seed=a.seed, verbose=True)
    preds[f.test] = p
    print(f"fold {f.name} done {time.time()-t:.0f}s  ic1={np.corrcoef(p[:,0], np.nan_to_num(Y[f.test,0]))[0,1]:.4f}", flush=True)

name = f"tcn_{a.venue}{('_' + a.tag) if a.tag else ''}"
m = np.isfinite(preds[:, 0])
for h in range(1, 6):
    ps = pd.Series(preds[:, h - 1], index=df.index)
    out = pd.DataFrame({"ts": df["ts"], "pred": ps, "scale": F.target_scale(df, h)})[m]
    out.to_parquet(PRED / f"{name.replace('tcn_' + a.venue, 'tcn_' + a.venue + str(h))}_{a.months.replace(',', '-')}.parquet")
    r = E.summarize(f"{name}_h{h}", df, ps, h, venue=a.venue)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    with open(DEV / "results.jsonl", "a") as fh:
        fh.write(json.dumps({**r, "months": a.months, "args": vars(a)}) + "\n")
