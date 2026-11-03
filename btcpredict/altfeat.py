"""Cross-asset & cross-book second-level features.

  * altcoin "swarm": last-W-second returns of SOL, XRP, DOGE, BNB, ETH and their equal-weight
    average (scaled by each coin's own 1s volatility) -- does the crowd move before BTC?
  * other BTC spot books on Binance (BTC/FDUSD, BTC/USDC): log-price deviation from BTC/USDT and
    its deviation from its own rolling mean (cross-book convergence, like the perp basis)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .secfeat import NMIN, PRE, _extended

ALTS = ("eth", "sol", "xrp", "doge", "bnb")
BOOKS = ("btcfdusd", "btcusdc")


def day_alt(day: dt.date) -> pd.DataFrame | None:
    S = _extended("spot", day, ["c", "v"])
    if S is None:
        return None
    m = np.arange(NMIN)
    end = PRE + 60 * m + 59
    alts = {a: _extended(a, day, ["c", "v", "bv"]) for a in ALTS}
    books = {b: _extended(b, day, ["c", "v", "n"]) for b in BOOKS}
    return alt_from_arrays(S, alts, books, end, (pd.Timestamp(day) + pd.to_timedelta(60 * m, unit="s")).values)


def alt_from_arrays(S: dict, alts: dict, books: dict, end: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Shared by history and live. Needs >= PRE+60 seconds of history before end[0]."""
    f = {"ts": ts}
    lpS = S["lp"]
    rS = np.diff(lpS, prepend=lpS[0])
    sigS = np.sqrt(pd.Series(rS ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
    zs = []
    for a in ALTS:
        A = alts.get(a)
        if A is None:
            continue
        lp = A["lp"]
        r = np.diff(lp, prepend=lp[0])
        sig = np.sqrt(pd.Series(r ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
        for w in (3, 10, 30, 60):
            z = (lp[end] - lp[end - w]) / (sig[end] * np.sqrt(w))
            f[f"alt_{a}_z{w}"] = z
            if w in (10, 30):
                zs.append((w, z))
        cs = np.concatenate([[0.0], np.cumsum(2 * A["bv"] - A["v"])])
        cv = np.concatenate([[0.0], np.cumsum(A["v"])])
        f[f"alt_{a}_imb30"] = (cs[end + 1] - cs[end + 1 - 30]) / (cv[end + 1] - cv[end + 1 - 30] + 1e-12)
    for w in (10, 30):
        arr = np.stack([z for ww, z in zs if ww == w])
        f[f"alt_swarm_z{w}"] = np.nanmean(arr, 0)
        f[f"alt_swarm_breadth{w}"] = np.nanmean(np.sign(arr), 0)
        btc_z = (lpS[end] - lpS[end - w]) / (sigS[end] * np.sqrt(w))
        f[f"alt_swarm_minus_btc{w}"] = f[f"alt_swarm_z{w}"] - btc_z
    for b in BOOKS:
        B = books.get(b)
        if B is None:
            continue
        dev = B["lp"] - lpS
        cs = np.concatenate([[0.0], np.cumsum(dev)])
        f[f"book_{b}_dev"] = dev[end] / (sigS[end] * np.sqrt(60))
        for w in (10, 60, 300):
            f[f"book_{b}_dev{w}"] = (dev[end] - (cs[end + 1] - cs[end + 1 - w]) / w) / (sigS[end] * np.sqrt(60))
        rb = B["lp"][end] - B["lp"][end - 10]
        f[f"book_{b}_r10"] = rb / (sigS[end] * np.sqrt(10))
        cn = np.concatenate([[0.0], np.cumsum(B["n"])])
        f[f"book_{b}_ln60"] = np.log1p(cn[end + 1] - cn[end + 1 - 60])
    return pd.DataFrame(f)


def build(start: str, end: str, workers: int = 4) -> pd.DataFrame:
    from multiprocessing import Pool

    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    with Pool(workers) as pool:
        parts = [p for p in pool.imap(day_alt, days, chunksize=4) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    num = df.columns.drop("ts")
    df[num] = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df
