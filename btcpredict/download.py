"""Bulk historical data from data.binance.vision, converted to compact per-second bars.

Sources (all public, no API key):
  spot  BTCUSDT 1s klines     -> per-second OHLCV + taker-buy volume + trade count
  um    BTCUSDT aggTrades     -> per-second bars built from tick trades (perp futures)
  um    BTCUSDT metrics       -> 5-minute open interest / long-short ratios
  um    BTCUSDT bookDepth     -> ~30s snapshots of cumulative depth at +-0.2..5%
  um    BTCUSDT premiumIndex  -> 1m premium index klines
  spot  BTCUSDT 1m klines     -> long history minute bars

Usage:
  python -m btcpredict.download --start 2025-11-01 --end 2026-11-02 --workers 6
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from .config import BINANCE_VISION, RAW, SEC

SESSION = requests.Session()
DAY_SECONDS = 86400


def _get(url: str, retries: int = 5) -> bytes | None:
    for k in range(retries):
        try:
            r = SESSION.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001 - network flakiness
            time.sleep(2 ** k)
    raise RuntimeError(f"failed to fetch {url}")


def _read_zip_csv(content: bytes, header) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    raw = z.read(z.namelist()[0])
    # some files carry a header row, others don't -> sniff
    first = raw[:200].split(b"\n")[0]
    has_header = any(c.isalpha() for c in first.decode(errors="ignore").replace("true", "").replace("false", ""))
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, engine="pyarrow")
    if not has_header and header is not None:
        df.columns = header[: df.shape[1]]
    return df


KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
            "is_buyer_maker"]


def _to_ms(t: np.ndarray) -> np.ndarray:
    t = t.astype(np.int64)
    # binance spot switched to microsecond timestamps in 2025
    return np.where(t > 10**14, t // 1000, t)


def _day_start_ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


# --------------------------------------------------------------------------------------------
# spot 1-second klines -> per-second bars
# --------------------------------------------------------------------------------------------
def spot_1s(day: dt.date, symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    url = f"{BINANCE_VISION}/spot/daily/klines/{symbol}/1s/{symbol}-1s-{day}.zip"
    c = _get(url)
    if c is None:
        return None
    df = _read_zip_csv(c, KLINE_COLS)
    ts = _to_ms(df["open_time"].to_numpy())
    sec = (ts - _day_start_ms(day)) // 1000
    ok = (sec >= 0) & (sec < DAY_SECONDS)
    sec = sec[ok]
    out = {}
    n = DAY_SECONDS
    for col, name in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]:
        a = np.full(n, np.nan)
        a[sec] = df[col].to_numpy(dtype=float)[ok]
        out[name] = a
    for col, name in [("volume", "v"), ("quote_volume", "qv"), ("count", "n"), ("taker_buy_volume", "bv")]:
        a = np.zeros(n)
        a[sec] = df[col].to_numpy(dtype=float)[ok]
        out[name] = a
    # seconds without trades carry no price information
    nov = out["n"] == 0
    for k in ("o", "h", "l", "c"):
        out[k][nov] = np.nan
    return pd.DataFrame(out).astype("float64")


# --------------------------------------------------------------------------------------------
# futures aggTrades -> per-second bars
# --------------------------------------------------------------------------------------------
LARGE_QTY = 1.0  # BTC; ~$80-120k notional


def um_aggtrades_1s(day: dt.date, symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    url = f"{BINANCE_VISION}/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
    c = _get(url)
    if c is None:
        return None
    df = _read_zip_csv(c, AGG_COLS)
    return trades_to_1s(
        _to_ms(df["transact_time"].to_numpy()) - _day_start_ms(day),
        df["price"].to_numpy(dtype=float),
        df["quantity"].to_numpy(dtype=float),
        df["is_buyer_maker"].to_numpy().astype(bool),
        n=DAY_SECONDS,
    )


def trades_to_1s(t_ms: np.ndarray, px: np.ndarray, qty: np.ndarray, buyer_maker: np.ndarray, n: int) -> pd.DataFrame:
    """Aggregate tick trades (time in ms relative to window start) into n one-second bars."""
    order = np.argsort(t_ms, kind="stable")
    t_ms, px, qty, buyer_maker = t_ms[order], px[order], qty[order], buyer_maker[order]
    sec = t_ms // 1000
    ok = (sec >= 0) & (sec < n)
    sec, px, qty, buyer_maker = sec[ok], px[ok], qty[ok], buyer_maker[ok]
    buy = ~buyer_maker  # aggressor = buyer
    cnt = np.bincount(sec, minlength=n).astype(float)
    v = np.bincount(sec, weights=qty, minlength=n)
    bv = np.bincount(sec, weights=qty * buy, minlength=n)
    qv = np.bincount(sec, weights=qty * px, minlength=n)
    nb = np.bincount(sec, weights=buy.astype(float), minlength=n)
    large = qty >= LARGE_QTY
    lbv = np.bincount(sec, weights=qty * (buy & large), minlength=n)
    lsv = np.bincount(sec, weights=qty * (~buy & large), minlength=n)
    first = np.searchsorted(sec, np.arange(n), side="left")
    last = np.searchsorted(sec, np.arange(n), side="right") - 1
    has = cnt > 0
    o = np.full(n, np.nan); c = np.full(n, np.nan); h = np.full(n, np.nan); lo = np.full(n, np.nan)
    o[has] = px[first[has]]
    c[has] = px[last[has]]
    if len(px):
        starts = first[has]
        h[has] = np.maximum.reduceat(px, starts)
        lo[has] = np.minimum.reduceat(px, starts)
    mx = np.zeros(n)
    if len(qty):
        mx[has] = np.maximum.reduceat(qty, first[has])
    return pd.DataFrame(dict(o=o, h=h, l=lo, c=c, v=v, qv=qv, n=cnt, bv=bv, nb=nb, lbv=lbv, lsv=lsv, mxq=mx))


# --------------------------------------------------------------------------------------------
# small auxiliary datasets (kept raw, merged at feature stage)
# --------------------------------------------------------------------------------------------
def um_metrics(day: dt.date, symbol="BTCUSDT"):
    c = _get(f"{BINANCE_VISION}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip")
    return None if c is None else _read_zip_csv(c, None)


def um_bookdepth(day: dt.date, symbol="BTCUSDT"):
    c = _get(f"{BINANCE_VISION}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{day}.zip")
    if c is None:
        return None
    df = _read_zip_csv(c, None)
    wide = df.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    wide.columns = [f"d{c:+.2f}" for c in wide.columns]
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide.reset_index()


def um_klines_1m(day: dt.date, symbol="BTCUSDT", kind="klines"):
    c = _get(f"{BINANCE_VISION}/futures/um/daily/{kind}/{symbol}/1m/{symbol}-1m-{day}.zip")
    return None if c is None else _read_zip_csv(c, KLINE_COLS)


def spot_klines_1m_month(year: int, month: int, symbol="BTCUSDT"):
    c = _get(f"{BINANCE_VISION}/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip")
    return None if c is None else _read_zip_csv(c, KLINE_COLS)


def spot_klines_1m_day(day: dt.date, symbol="BTCUSDT"):
    c = _get(f"{BINANCE_VISION}/spot/daily/klines/{symbol}/1m/{symbol}-1m-{day}.zip")
    return None if c is None else _read_zip_csv(c, KLINE_COLS)


# --------------------------------------------------------------------------------------------
JOBS = {
    "spot": lambda d: spot_1s(d, "BTCUSDT"),
    "um": lambda d: um_aggtrades_1s(d, "BTCUSDT"),
    "eth": lambda d: spot_1s(d, "ETHUSDT"),
    "metrics": um_metrics,
    "depth": um_bookdepth,
    "premium": lambda d: um_klines_1m(d, kind="premiumIndexKlines"),
    "um1m": um_klines_1m,
}


def run_job(name: str, day: dt.date) -> str:
    out = SEC / name / f"{day}.parquet"
    if out.exists():
        return "cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = JOBS[name](day)
    if df is None:
        return "missing"
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    return "ok"


def daterange(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-11-01")
    ap.add_argument("--end", default="2026-11-02")
    ap.add_argument("--jobs", default="spot,um,metrics,depth,premium,um1m,eth")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    tasks = [(j, d) for j in a.jobs.split(",") for d in daterange(s, e)]
    t0 = time.time()
    done = 0
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_job, j, d): (j, d) for j, d in tasks}
        for f in as_completed(futs):
            j, d = futs[f]
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                r = f"error:{exc}"
                print(j, d, r, flush=True)
            stats[r] = stats.get(r, 0) + 1
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} {stats} {time.time()-t0:.0f}s", flush=True)
    print("finished", stats, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
