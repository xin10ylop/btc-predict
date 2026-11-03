"""Live BTC forecaster: every minute, forecasts the next 1..5 minutes.

  python -m btcpredict.run_live                     # full model (needs Binance USD-M futures API)
  python -m btcpredict.run_live --variant spot      # spot-only model (Binance spot data only)
  python -m btcpredict.run_live --once              # single forecast and exit
  python -m btcpredict.run_live --replay "2026-10-15 13:37" --variant full   # offline, from history

Every forecast is appended to logs/predictions.jsonl (score it later with scripts/score_live_log.py).
Forecasts refer to the price change from the forecast instant (bar close) to the close h minutes
later. The 1-minute spot edge decays within seconds (see README) - act fast or use h>=2.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import live
from .config import ROOT
from .predictor import Predictor

warnings.simplefilter("ignore")
LOG = ROOT / "logs"


def forecast(pred: Predictor, src, T: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    dfh, A = live.build_features(src, T, use_futures=pred.uses_futures, last_n=1)
    X = live.model_inputs(dfh, A)
    sigma = dfh["sigma"].values[-1:]
    out = pred.predict(X, sigma)
    info = {"T": str(T), "spot": float(np.exp(dfh["s_close"].values[-1])),
            "perp": float(np.exp(dfh["f_close"].values[-1])) if pred.uses_futures else None,
            "sigma_bp": float(sigma[0] * 1e4)}
    return out, info


def render(out: pd.DataFrame, info: dict) -> str:
    lines = [f"\n=== {info['T']} UTC   BTC spot {info['spot']:,.2f}" +
             (f"   perp {info['perp']:,.2f}" if info['perp'] else "") + f"   1m vol {info['sigma_bp']:.1f}bp ==="]
    lines.append(f"{'target':8s} {'P(up)':>6s} {'E[r] bp':>8s} {'80% band (bp)':>18s} {'price 80% band':>26s} {'conf%':>6s} {'hist hit':>8s}")
    for _, r in out.sort_values(["venue", "h"]).iterrows():
        base = info["spot"] if r["venue"] == "s" else info["perp"]
        lo, hi = base * np.exp(r["lo80_bp"] / 1e4), base * np.exp(r["hi80_bp"] / 1e4)
        name = ("spot" if r["venue"] == "s" else "perp") + f"+{r['h']}m"
        hh = r["hist_hit_at_conf"]
        lines.append(f"{name:8s} {r['p_up']:6.3f} {r['exp_ret_bp']:+8.2f} [{r['lo80_bp']:+7.1f},{r['hi80_bp']:+7.1f}] "
                     f"[{lo:11,.1f},{hi:11,.1f}] {r['confidence_pct']:6.0f} {hh if hh is not None else float('nan'):8.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "spot"])
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--replay", default=None, help="UTC timestamp; use historical files instead of live APIs")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds to wait after the minute boundary")
    a = ap.parse_args()
    pred = Predictor(a.variant)
    # attach the per-confidence-decile historical hit rate to each target
    for key, m in pred.meta["targets"].items():
        m["hit_by_conf"] = None
    LOG.mkdir(exist_ok=True)
    logf = LOG / "predictions.jsonl"

    if a.replay:
        src = live.ReplaySource()
        T = pd.Timestamp(a.replay).floor("min")
        out, info = forecast(pred, src, T)
        _attach_hist(out, pred)
        print(render(out, info))
        return

    src = live.CachedSource(live.RestSource(use_futures=pred.uses_futures))
    while True:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        T = now.floor("min")
        target_time = T + pd.Timedelta(seconds=a.delay)
        if now < target_time:
            time.sleep((target_time - now).total_seconds())
        t0 = time.time()
        try:
            out, info = forecast(pred, src, T)
        except Exception as e:  # noqa: BLE001
            print(f"{T}: forecast failed: {e}", flush=True)
            if a.once:
                raise
            time.sleep(5)
            continue
        _attach_hist(out, pred)
        info["latency_s"] = round(time.time() - t0 + a.delay, 2)
        print(render(out, info) + f"\n(computed {info['latency_s']}s after bar close)", flush=True)
        with open(logf, "a") as fh:
            fh.write(json.dumps({**info, "forecasts": out.drop(columns=["hist_hit_at_conf"]).to_dict("records")}) + "\n")
        if a.once:
            break
        nxt = T + pd.Timedelta(minutes=1, seconds=a.delay)
        time.sleep(max(0.0, (nxt - pd.Timestamp.now(tz="UTC").tz_localize(None)).total_seconds()))


def _attach_hist(out: pd.DataFrame, pred: Predictor):
    hh = []
    for _, r in out.iterrows():
        m = pred.meta["targets"][r["target"]]
        dec = m.get("hit_by_conf_decile") or []
        if dec:
            i = min(int(r["confidence_pct"] // 10), len(dec) - 1)
            hh.append(dec[i])
        else:
            hh.append(None)
    out["hist_hit_at_conf"] = hh


if __name__ == "__main__":
    main()
