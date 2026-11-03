"""Live BTC forecaster: every minute, forecasts the next 1..5 minutes.

  python -m btcpredict.run_live                     # full model (needs Binance USD-M futures API)
  python -m btcpredict.run_live --variant spot      # spot-only model (Binance spot data only)
  python -m btcpredict.run_live --once              # single forecast and exit
  python -m btcpredict.run_live --replay "2026-10-15 13:37" --variant full   # offline, from history

Every forecast is appended to logs/predictions.jsonl. As forecasts mature, the realized move
updates the online RLS layer (state persisted in logs/rls_state_<variant>.json), so the model
keeps adapting between retrains. Forecasts refer to the change from the bar close (forecast
instant) to the close h minutes later; the 1-minute edge decays within seconds (see README).
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


def forecast(pred: Predictor, src, T: pd.Timestamp):
    dfh, A = live.build_features(src, T, use_futures=pred.uses_futures, last_n=1)
    X = live.model_inputs(dfh, A)
    sigma = dfh["sigma"].values[-1:]
    out = pred.predict(X, sigma)
    info = {"T": str(T), "spot": float(np.exp(dfh["s_close"].values[-1])),
            "perp": float(np.exp(dfh["f_close"].values[-1])) if pred.uses_futures else None,
            "sigma_bp": float(sigma[0] * 1e4)}
    return out, info, dfh


def render(out: pd.DataFrame, info: dict) -> str:
    lines = [f"\n=== {info['T']} UTC   BTC spot {info['spot']:,.2f}" +
             (f"   perp {info['perp']:,.2f}" if info['perp'] else "") + f"   1m vol {info['sigma_bp']:.1f}bp ==="]
    lines.append(f"{'target':8s} {'P(up)':>6s} {'E[r] bp':>8s} {'80% band (bp)':>18s} {'price 80% band':>26s} {'conf%':>6s} {'hist hit':>8s}")
    for _, r in out.sort_values(["venue", "h"]).iterrows():
        base = info["spot"] if r["venue"] == "s" else info["perp"]
        lo, hi = base * np.exp(r["lo80_bp"] / 1e4), base * np.exp(r["hi80_bp"] / 1e4)
        name = ("spot" if r["venue"] == "s" else "perp") + f"+{r['h']}m"
        lines.append(f"{name:8s} {r['p_up']:6.3f} {r['exp_ret_bp']:+8.2f} [{r['lo80_bp']:+7.1f},{r['hi80_bp']:+7.1f}] "
                     f"[{lo:11,.1f},{hi:11,.1f}] {r['confidence_pct']:6.0f} {r['hist_hit_at_conf']:8.3f}")
    return "\n".join(lines)


def learn(pred: Predictor, pending: list, dfh: pd.DataFrame, T: pd.Timestamp) -> list:
    """Update the online layer with every forecast whose horizon has elapsed by instant T."""
    closes = {"s": pd.Series(dfh["s_close"].values, index=pd.DatetimeIndex(dfh["ts"]) + pd.Timedelta(minutes=1))}
    if pred.uses_futures:
        closes["f"] = pd.Series(dfh["f_close"].values, index=pd.DatetimeIndex(dfh["ts"]) + pd.Timedelta(minutes=1))
    keep = []
    for p in pending:
        t_end = p["T"] + pd.Timedelta(minutes=p["h"])
        if t_end > T:
            keep.append(p)
            continue
        c = closes[p["venue"]]
        if t_end in c.index and p["T"] in c.index:
            r = c[t_end] - c[p["T"]]
            pred.update(p["key"], p["z_core"], p["z_sat"], r / (p["sigma"] * np.sqrt(p["h"])))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "spot"])
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--replay", default=None, help="UTC timestamp; use historical files instead of live APIs")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to wait after the minute boundary")
    a = ap.parse_args()
    pred = Predictor(a.variant)
    LOG.mkdir(exist_ok=True)

    if a.replay:
        out, info, _ = forecast(pred, live.ReplaySource(), pd.Timestamp(a.replay).floor("min"))
        print(render(out, info))
        return

    src = live.CachedSource(live.RestSource(use_futures=pred.uses_futures))
    pending: list = []
    while True:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        T = now.floor("min")
        wake = T + pd.Timedelta(seconds=a.delay)
        if now < wake:
            time.sleep((wake - now).total_seconds())
        t0 = time.time()
        try:
            out, info, dfh = forecast(pred, src, T)
        except Exception as e:  # noqa: BLE001
            print(f"{T}: forecast failed: {e}", flush=True)
            if a.once:
                raise
            time.sleep(5)
            continue
        info["latency_s"] = round(time.time() - t0 + a.delay, 2)
        print(render(out, info) + f"\n(computed {info['latency_s']}s after bar close)", flush=True)
        with open(LOG / "predictions.jsonl", "a") as fh:
            fh.write(json.dumps({**info, "variant": a.variant, "forecasts": out.to_dict("records")}) + "\n")
        pending = learn(pred, pending, dfh, T)
        sig = info["sigma_bp"] / 1e4
        pending += [{"T": T, "key": r["target"], "venue": r["venue"], "h": int(r["h"]), "z_core": r["z_core"],
                     "z_sat": r["z_sat"], "sigma": sig} for _, r in out.iterrows()]
        pred.save_state()
        if a.once:
            break
        nxt = T + pd.Timedelta(minutes=1, seconds=a.delay)
        time.sleep(max(0.0, (nxt - pd.Timestamp.now(tz="UTC").tz_localize(None)).total_seconds()))


if __name__ == "__main__":
    main()
