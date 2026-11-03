"""Live feature construction -- the SAME code path as training (secfeat / altfeat / panel / features).

Two interchangeable data sources:
  RestSource   -- official public Binance endpoints (spot mirror data-api.binance.vision and
                  USD-M futures fapi.binance.com).  Futures endpoints are geo-restricted in some
                  jurisdictions; use the spot-only model there.
  ReplaySource -- reads the historical per-second files; used to prove that live features are
                  identical to the training features (scripts/check_live_parity.py).

Window layout at forecast instant T (a minute boundary):
  * minute history: HIST_MIN minutes before T (needed for 24h rolling statistics)
  * second-level detail: SEC_WIN seconds before T for every venue
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import altfeat, features, panel, secfeat

HIST_MIN = 2600        # minutes of minute-level history (slow EWMs converge to <0.1%)
SEC_WIN = 1260         # seconds of per-second detail (>= PRE + 60 for alt vol estimates, + slack)
SPOT_API = ["https://data-api.binance.vision/api/v3", "https://api.binance.com/api/v3"]
FUT_API = "https://fapi.binance.com"
ALT_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT", "doge": "DOGEUSDT", "bnb": "BNBUSDT",
               "btcfdusd": "BTCFDUSD", "btcusdc": "BTCUSDC"}
SEC_COLS = ["o", "h", "l", "c", "v", "qv", "n", "bv"]


def _grid(t0_ms: int, n: int) -> pd.DataFrame:
    return pd.DataFrame(index=np.arange(n))


def kl_to_seconds(rows: list, t0_ms: int, n: int) -> pd.DataFrame:
    """Binance 1s klines -> dense per-second frame of length n starting at t0_ms."""
    out = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    out.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv")})
    if rows:
        a = np.array([[float(x) for x in r[:11]] for r in rows])
        sec = ((a[:, 0] - t0_ms) // 1000).astype(int)
        ok = (sec >= 0) & (sec < n)
        sec, a = sec[ok], a[ok]
        for j, k in ((1, "o"), (2, "h"), (3, "l"), (4, "c"), (5, "v"), (7, "qv"), (8, "n"), (9, "bv")):
            out[k][sec] = a[:, j]
        nov = out["n"] == 0
        for k in ("o", "h", "l", "c"):
            out[k][nov] = np.nan
    return pd.DataFrame(out)


def minute_to_seconds(m: pd.DataFrame, t0_ms: int, n: int) -> pd.DataFrame:
    """Embed 1-minute bars into a per-second grid (all activity on the minute's last second).
    Exact for every per-minute aggregate used by the minute-level features (close, high, low,
    volume, taker-buy volume, quote volume); NOT used for intra-minute features."""
    out = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    out.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv", "nb", "lbv", "lsv", "mxq")})
    if len(m):
        last = ((m["open_ms"].values - t0_ms) // 1000).astype(int) + 59
        ok = (last >= 0) & (last < n)
        last = last[ok]
        mm = m[ok]
        out["h"][last] = mm["h"].values
        out["l"][last] = mm["l"].values
        out["o"][last] = mm["c"].values
        out["c"][last] = mm["c"].values
        for k in ("v", "qv", "n", "bv"):
            out[k][last] = mm[k].values
    return pd.DataFrame(out)


# =============================================================================================
class ReplaySource:
    """Historical files -> the inputs a live process would have had at instant T."""

    def __init__(self):
        from .config import SEC
        self.SEC = SEC
        self._cache: dict = {}

    def _day(self, venue, day):
        key = (venue, day)
        if key not in self._cache:
            p = self.SEC / venue / f"{day}.parquet"
            self._cache[key] = pd.read_parquet(p) if p.exists() else None
        return self._cache[key]

    def _range(self, venue, t0_ms, n):
        t0 = pd.Timestamp(t0_ms, unit="ms")
        days = pd.date_range(t0.normalize(), (t0 + pd.Timedelta(seconds=n)).normalize(), freq="D")
        parts = []
        for d in days:
            x = self._day(venue, d.date())
            if x is None:
                x = pd.DataFrame({c: np.zeros(86400) for c in SEC_COLS})
            parts.append(x)
        x = pd.concat(parts, ignore_index=True)
        off = int((t0 - days[0]).total_seconds())
        return x.iloc[off:off + n].reset_index(drop=True)

    def spot_seconds(self, symbol_key, t0_ms, n):
        return self._range({"btc": "spot"}.get(symbol_key, symbol_key), t0_ms, n)

    def fut_seconds(self, t0_ms, n):
        return self._range("um", t0_ms, n)

    def minutes(self, venue, t0_ms, n_min):
        """1-minute bars aggregated from the per-second history (= what the kline endpoint returns)."""
        x = self._range(venue, t0_ms, n_min * 60)
        g = np.arange(len(x)) // 60
        c = pd.Series(x["c"].values).groupby(g).last()
        out = pd.DataFrame({
            "open_ms": t0_ms + 60000 * np.arange(n_min),
            "c": x["c"].groupby(g).last().values, "h": x["h"].groupby(g).max().values, "l": x["l"].groupby(g).min().values,
            "v": x["v"].groupby(g).sum().values, "qv": x["qv"].groupby(g).sum().values, "n": x["n"].groupby(g).sum().values,
            "bv": x["bv"].groupby(g).sum().values})
        out["c"] = pd.Series(out["c"]).ffill().values
        return out

    def premium(self, t0_ms, n_min):
        days = pd.date_range(pd.Timestamp(t0_ms, unit="ms").normalize(), pd.Timestamp(t0_ms + n_min * 60000, unit="ms").normalize())
        parts = [self._day("premium", d.date()) for d in days]
        p = pd.concat([x for x in parts if x is not None])
        p = p[(p["open_time"] >= t0_ms) & (p["open_time"] < t0_ms + n_min * 60000)]
        return pd.DataFrame({"open_ms": p["open_time"].values, "prem": p["close"].astype(float).values})


# =============================================================================================
class RestSource:
    def __init__(self, use_futures: bool = True, timeout: float = 10.0):
        self.s = requests.Session()
        self.timeout = timeout
        self.use_futures = use_futures

    def _get(self, bases, path, params):
        bases = bases if isinstance(bases, list) else [bases]
        err = None
        for b in bases:
            for k in range(3):
                try:
                    r = self.s.get(b + path, params=params, timeout=self.timeout)
                    if r.status_code == 200:
                        return r.json()
                    err = f"{r.status_code} {r.text[:200]}"
                    if r.status_code in (451, 403):
                        break
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                time.sleep(0.3 * (k + 1))
        raise RuntimeError(f"GET {path} failed: {err}")

    def _klines(self, bases, path, symbol, interval, t0_ms, t1_ms, limit):
        rows, start = [], t0_ms
        step = {"1s": 1000, "1m": 60000}[interval]
        while start < t1_ms:
            r = self._get(bases, path, {"symbol": symbol, "interval": interval, "startTime": start,
                                        "endTime": t1_ms - 1, "limit": limit})
            if not r:
                break
            rows += r
            start = int(r[-1][0]) + step
            if len(r) < limit:
                break
        return rows

    def spot_seconds(self, symbol_key, t0_ms, n):
        sym = "BTCUSDT" if symbol_key == "btc" else ALT_SYMBOLS[symbol_key]
        rows = self._klines(SPOT_API, "/klines", sym, "1s", t0_ms, t0_ms + n * 1000, 1000)
        return kl_to_seconds(rows, t0_ms, n)

    def minutes(self, venue, t0_ms, n_min):
        if venue == "um":
            rows = self._klines(FUT_API, "/fapi/v1/klines", "BTCUSDT", "1m", t0_ms, t0_ms + n_min * 60000, 1500)
        else:
            sym = {"spot": "BTCUSDT"}.get(venue, ALT_SYMBOLS.get(venue))
            rows = self._klines(SPOT_API, "/klines", sym, "1m", t0_ms, t0_ms + n_min * 60000, 1000)
        a = np.array([[float(x) for x in r[:11]] for r in rows])
        return pd.DataFrame({"open_ms": a[:, 0].astype(np.int64), "c": a[:, 4], "h": a[:, 2], "l": a[:, 3], "v": a[:, 5],
                             "qv": a[:, 7], "n": a[:, 8], "bv": a[:, 9]})

    def fut_seconds(self, t0_ms, n):
        """Per-second perp bars from aggTrades (same aggregation as the training data)."""
        trades, start, end = [], t0_ms, t0_ms + n * 1000
        r = self._get(FUT_API, "/fapi/v1/aggTrades", {"symbol": "BTCUSDT", "startTime": start,
                                                       "endTime": min(end - 1, start + 3_599_000), "limit": 1000})
        trades += r
        while r and len(r) == 1000 and r[-1]["T"] < end:
            r = self._get(FUT_API, "/fapi/v1/aggTrades", {"symbol": "BTCUSDT", "fromId": r[-1]["a"] + 1, "limit": 1000})
            trades += [x for x in r if x["T"] < end]
        if not trades:
            return secfeat_empty(n)
        t = np.array([x["T"] for x in trades], dtype=np.int64) - t0_ms
        p = np.array([float(x["p"]) for x in trades]); q = np.array([float(x["q"]) for x in trades])
        m = np.array([x["m"] for x in trades], dtype=bool)
        from .download import trades_to_1s
        return trades_to_1s(t, p, q, m, n)

    def premium(self, t0_ms, n_min):
        rows = self._klines(FUT_API, "/fapi/v1/premiumIndexKlines", "BTCUSDT", "1m", t0_ms, t0_ms + n_min * 60000, 1500)
        return pd.DataFrame({"open_ms": [int(r[0]) for r in rows], "prem": [float(r[4]) for r in rows]})


def secfeat_empty(n):
    d = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    d.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv", "nb", "lbv", "lsv", "mxq")})
    return pd.DataFrame(d)


# =============================================================================================
def _arrays(df: pd.DataFrame, cols) -> dict:
    out = {c: df[c].to_numpy(np.float64) for c in cols if c in df}
    c = pd.Series(df["c"].to_numpy(np.float64)).ffill().bfill().to_numpy()
    out["lp"] = np.log(c)
    return out


def build_features(src, T: pd.Timestamp, use_futures: bool = True, last_n: int = 1) -> pd.DataFrame:
    """Full feature frame (same columns as training) for the `last_n` bars ending at instant T."""
    T = pd.Timestamp(T).floor("min")
    T_ms = int(T.value // 10**6)
    h0_ms = T_ms - HIST_MIN * 60000
    nS = HIST_MIN * 60
    s0_ms = T_ms - SEC_WIN * 1000
    # spot BTC: per-second for the whole history (exact realized variance per minute)
    S = src.spot_seconds("btc", h0_ms, nS)
    # perp: minute bars for history + real per-second bars for the recent window
    if use_futures:
        Fm = src.minutes("um", h0_ms, HIST_MIN)
        F = minute_to_seconds(Fm, h0_ms, nS)
        Fs = src.fut_seconds(s0_ms, SEC_WIN)
        F.iloc[nS - SEC_WIN:] = Fs[F.columns].values
    else:
        F = S.copy()
        for k in ("nb", "lbv", "lsv", "mxq"):
            F[k] = 0.0
    Em = src.minutes("eth", h0_ms, HIST_MIN)
    E = minute_to_seconds(Em, h0_ms, nS)
    Es = src.spot_seconds("eth", s0_ms, SEC_WIN)
    E.iloc[nS - SEC_WIN:, [E.columns.get_loc(c) for c in SEC_COLS]] = Es[SEC_COLS].values
    m = np.arange(HIST_MIN)
    end = 60 * m + 59
    keep = end - 59 >= 660            # need 660s of history inside the window for the second-level windows
    m, end = m[keep], end[keep]
    ts = (pd.Timestamp(h0_ms, unit="ms") + pd.to_timedelta(60 * m, unit="s")).values
    df = secfeat.features_from_arrays(_arrays(S, SEC_COLS), _arrays(F, list(F.columns)), _arrays(E, ["c", "v", "bv"]),
                                      end, ts)
    for c in [c for c in df.columns if c.endswith(("_close", "_open", "_hi", "_lo")) and not c.startswith("e_")]:
        df[c] = df[c] + df["ref"]
    df = df.drop(columns=["ref"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = panel.add_minute_features(df)
    if use_futures:
        P = src.premium(h0_ms - 60000 * 600, HIST_MIN + 600)
        pts = pd.to_datetime(P["open_ms"], unit="ms")
        pr = pd.Series(P["prem"].values, index=pts).reindex(df["ts"])
        df["prem"] = pr.values.astype("float32")
        df["prem_ma480"] = df["prem"].rolling(480, min_periods=30).mean()
        df["prem_dev60"] = df["prem"] - df["prem"].rolling(60, min_periods=5).mean()
    # alt / cross-book features on the recent per-second window (only for the last_n bars)
    alts = {}
    for a in altfeat.ALTS + altfeat.BOOKS:
        x = Es if a == "eth" else src.spot_seconds(a, s0_ms, SEC_WIN)
        alts[a] = _arrays(x, ["c", "v", "bv", "n"])
    Srec = _arrays(S.iloc[nS - SEC_WIN:].reset_index(drop=True), ["c", "v"])
    endr = SEC_WIN - 1 - 60 * np.arange(last_n)[::-1]
    A = altfeat.alt_from_arrays(Srec, {k: alts[k] for k in altfeat.ALTS}, {k: alts[k] for k in altfeat.BOOKS}, endr,
                                df["ts"].values[-last_n:])
    return df.reset_index(drop=True), A.drop(columns="ts").reset_index(drop=True)


def model_inputs(df_hist: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Columns identical to training: model_matrix + extra_features + alt features, for the last
    len(A) bars (rolling statistics use the full minute history in df_hist)."""
    n = len(A)
    X = features.model_matrix(df_hist).iloc[-n:]
    Xe = features.extra_features(df_hist).iloc[-n:]
    out = pd.concat([X.reset_index(drop=True), Xe.reset_index(drop=True), A.astype("float32")], axis=1)
    out.index = df_hist.index[-n:]
    return out
