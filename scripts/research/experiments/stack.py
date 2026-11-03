"""Combine cached OOS predictions: averages, structural decomposition, online RLS stacking,
probability calibration. Usage:

python scripts/research/experiments/stack.py --venue s --h 1 --months 2025-11-2026-01-2026-03 --preds lgbm_s1_v2,tcn_s1
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from btcpredict.modeling import calibrate as C  # noqa: E402
from btcpredict.modeling import evaluate as E  # noqa: E402
from btcpredict.modeling import online  # noqa: E402
from btcpredict.config import FEAT, PREDS  # noqa: E402

PRED = PREDS
ap = argparse.ArgumentParser()
ap.add_argument("--venue", default="s")
ap.add_argument("--h", type=int, default=1)
ap.add_argument("--months", default="2025-11-2026-01-2026-03")
ap.add_argument("--preds", required=True)
ap.add_argument("--half_life", type=float, default=2880)
a = ap.parse_args()
h = a.h

cols = ["ts", "sigma", "basis", f"y_{a.venue}{h}"] + [f"y_{a.venue}{h}_d{d}" for d in (1, 2, 5)]
df = pd.read_parquet(FEAT / "panel.parquet", columns=cols)
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True).set_index("ts")
P = {}
for name in a.preds.split(","):
    p = pd.read_parquet(PRED / f"{name}_{a.months}.parquet").set_index("ts")
    if "_basis" in name or name.startswith("lgbm_basis"):
        # basis-change forecast -> contributes NEGATIVELY to spot return, in spot z units
        P[name] = -(p["pred"] * p["scale"]) / (df["sigma"].reindex(p.index) * np.sqrt(h))
    else:
        P[name] = p["pred"]
Pm = pd.DataFrame(P).dropna()
d = df.loc[Pm.index]
y = d[f"y_{a.venue}{h}"].values
yz = y / (d["sigma"].values * np.sqrt(h))
rows = []


def report(name, p):
    r = {"model": name, **E.metrics(y, p)}
    for dl in (1, 2, 5):
        r[f"ic_d{dl}"] = E.metrics(d[f"y_{a.venue}{h}_d{dl}"].values, p)["ic"]
    rows.append(r)


for c in Pm.columns:
    report(c, Pm[c].values)
std = Pm / Pm.std()
report("mean_std", std.mean(1).values)
fcols = [c for c in Pm.columns if c.startswith("lgbm_f")]
bcols = [c for c in Pm.columns if "basis" in c]
if a.venue == "s" and fcols and bcols:
    # structural decomposition: spot return = perp return - change in basis
    for fc in fcols:
        report(f"struct[{fc}+{bcols[0]}]", (Pm[fc] + Pm[bcols[0]]).values)
        scols = [c for c in Pm.columns if c.startswith("lgbm_s")]
        for sc in scols:
            report(f"avg[{sc},struct]", (Pm[sc] / Pm[sc].std() + (Pm[fc] + Pm[bcols[0]]) / (Pm[fc] + Pm[bcols[0]]).std()).values)
# online RLS stacking (causal; uses realized targets with delay h)
Z = np.column_stack([np.ones(len(Pm)), Pm.values])
prior = np.r_[0.0, np.ones(Pm.shape[1]) / Pm.shape[1]]
rls = online.rls_stack(Z, yz, horizon=h, half_life=a.half_life, ridge=1e4, prior=prior)
report("rls_stack", rls)
best = rows[-1]
res = pd.DataFrame(rows).set_index("model")
print(res[["n", "ic", "rank_ic", "hit", "auc", "hit_top10", "hit_top1", "edge_bps", "ic_d1", "ic_d2", "ic_d5"]].round(4).to_string())
# probability calibration of the stack
pu = C.causal_logistic(std.mean(1).values, y, horizon=h, refit_every=1440, min_obs=20000)
ll = C.logloss(pu, y)
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ll.items()}))
