"""RESEARCH ONLY (tested and rejected): OKX BTC-USDT perpetual as a third price-discovery venue.

Result (docs/FINDINGS.md): +0.001 IC on top of the satellite model, significant only for spot +1m;
the two perps are tightly arbitraged at the 1-second level. Not used in production.

    python -m btcpredict.research.okx --start 2025-04-30 --end 2026-11-03   # download + assemble
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd

from ..config import SEC
from ..data.binance import DAY_SECONDS, _day_start_ms, _get, trades_to_1s
from ..features.seconds import NMIN, PRE, _extended


# --------------------------------------------------------------------------------------------
# OKX BTC-USDT perpetual (third price-discovery venue). Daily files are cut on UTC+8 days and
# sizes are in contracts (0.01 BTC). Each file -> 86400 per-second bars starting at D-1 16:00 UTC
# (data/sec/okxraw/D), then re-cut into UTC days (data/sec/okx/D).
OKX_URL = "https://www.okx.com/cdn/okex/traderecords/trades/daily/{ymd}/BTC-USDT-SWAP-trades-{day}.zip"


def okx_file(day: dt.date) -> pd.DataFrame | None:
    c = _get(OKX_URL.format(ymd=day.strftime("%Y%m%d"), day=day))
    if c is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(c))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["side", "price", "size", "created_time"], engine="pyarrow")
    t0 = _day_start_ms(day) - 8 * 3600 * 1000
    return trades_to_1s(df["created_time"].to_numpy(np.int64) - t0, df["price"].to_numpy(float),
                        df["size"].to_numpy(float) * 0.01, (df["side"] == "sell").to_numpy(), n=DAY_SECONDS)


def okx_assemble(day: dt.date) -> str:
    out = SEC / "okx" / f"{day}.parquet"
    if out.exists():
        return "cached"
    a = SEC / "okxraw" / f"{day}.parquet"
    b = SEC / "okxraw" / f"{day + dt.timedelta(days=1)}.parquet"
    if not a.exists() or not b.exists():
        return "missing"
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    # raw file D covers [D-1 16:00, D 16:00) UTC -> UTC day D = A[8h:] + B[:8h]
    d = pd.concat([A.iloc[8 * 3600:], B.iloc[:8 * 3600]], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    return "ok"


# ---------------------------------------------------------------------------------------------
# v5: OKX BTC-USDT perpetual as a third price-discovery venue
def okx_from_arrays(Fb: dict, O: dict | None, end: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    f = {"ts": ts}
    if O is None or not np.isfinite(O["lp"]).any():
        return pd.DataFrame(f)
    lpb, lpo = Fb["lp"], O["lp"]
    rb = np.diff(lpb, prepend=lpb[0])
    sig = np.sqrt(pd.Series(rb ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
    dev = lpo - lpb
    cs = np.concatenate([[0.0], np.cumsum(dev)])
    f["okx_dev"] = (dev[end] - (cs[end + 1] - cs[end + 1 - 600]) / 600) / (sig[end] * np.sqrt(60))
    for w in (10, 60, 300):
        f[f"okx_dev{w}"] = (dev[end] - (cs[end + 1] - cs[end + 1 - w]) / w) / (sig[end] * np.sqrt(60))
    for w in (1, 3, 10, 30):
        f[f"okx_z{w}"] = (lpo[end] - lpo[end - w]) / (sig[end] * np.sqrt(w))
        f[f"okx_minus_binp{w}"] = ((lpo[end] - lpo[end - w]) - (lpb[end] - lpb[end - w])) / (sig[end] * np.sqrt(w))
    flow = np.concatenate([[0.0], np.cumsum(2 * O["bv"] - O["v"])])
    vol = np.concatenate([[0.0], np.cumsum(O["v"])])
    for w in (3, 10, 30):
        f[f"okx_imb{w}"] = (flow[end + 1] - flow[end + 1 - w]) / (vol[end + 1] - vol[end + 1 - w] + 1e-12)
    fb_flow = np.concatenate([[0.0], np.cumsum(2 * Fb["bv"] - Fb["v"])]) if "bv" in Fb else None
    if fb_flow is not None:
        fbv = np.concatenate([[0.0], np.cumsum(Fb["v"])])
        f["okx_share60"] = (vol[end + 1] - vol[end + 1 - 60]) / (fbv[end + 1] - fbv[end + 1 - 60] + vol[end + 1] - vol[end + 1 - 60] + 1e-12)
    return pd.DataFrame(f)


def day_okx(day: dt.date) -> pd.DataFrame | None:
    Fb = _extended("um", day, ["c", "v", "bv"])
    if Fb is None:
        return None
    O = _extended("okx", day, ["c", "v", "bv"])
    m = np.arange(NMIN)
    end = PRE + 60 * m + 59
    return okx_from_arrays(Fb, O, end, (pd.Timestamp(day) + pd.to_timedelta(60 * m, unit="s")).values)


def build_okx(start: str, end: str, workers: int = 3) -> pd.DataFrame:
    from multiprocessing import Pool

    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    with Pool(workers) as pool:
        parts = [p for p in pool.imap(day_okx, days, chunksize=4) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    num = df.columns.drop("ts")
    df[num] = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df


def download(start: str, end: str, workers: int = 5) -> None:
    from concurrent.futures import ThreadPoolExecutor

    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]

    def job(d):
        out = SEC / "okxraw" / f"{d}.parquet"
        if out.exists():
            return
        x = okx_file(d)
        if x is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            x.to_parquet(out, index=False)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, days))
    for d in days[1:-1]:
        okx_assemble(d)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-30")
    ap.add_argument("--end", default="2026-11-03")
    a = ap.parse_args()
    download(a.start, a.end)
