"""Contiguous per-second channel arrays for sequence models.

Channels (all scale-free; computed causally):
  0 spot 1s log-return / sigma_1s          5 futures log volume (rel. to 1-day mean)
  1 futures 1s log-return / sigma_1s       6 futures large-trade signed flow (rel.)
  2 basis deviation from 300s mean / s1s   7 ETH 1s log-return / sigma_1s
  3 spot signed flow (rel. to mean vol)    8 spot log volume (rel.)
  4 futures signed flow (rel.)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import FEAT, SEC

CHANNELS = ["rs", "rf", "bdev", "fs", "ff", "lvf", "lff", "re", "lvs"]


def _cat(venue: str, days: list[dt.date], cols: list[str]) -> dict[str, np.ndarray]:
    parts = {c: [] for c in cols}
    for d in days:
        p = SEC / venue / f"{d}.parquet"
        if p.exists():
            x = pd.read_parquet(p, columns=cols)
        else:
            x = pd.DataFrame({c: np.full(86400, np.nan if c == "c" else 0.0) for c in cols})
        for c in cols:
            parts[c].append(x[c].to_numpy(np.float64))
    return {c: np.concatenate(v) for c, v in parts.items()}


def build(start: str, end: str, out_name: str = "seq", chunk_days: int = 20, warm_days: int = 1) -> None:
    """Memory-light: processes `chunk_days` at a time (+ warm-up days for rolling statistics) and
    writes into a memory-mapped .npy file."""
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    out = np.lib.format.open_memmap(FEAT / f"{out_name}_X.npy", mode="w+", dtype=np.float32,
                                    shape=(len(days) * 86400, len(CHANNELS)))
    for i in range(0, len(days), chunk_days):
        lo = max(0, i - warm_days)
        block = _channels(days[lo:i + chunk_days])
        skip = (i - lo) * 86400
        out[i * 86400: i * 86400 + len(block) - skip] = block[skip:]
        print("chunk", days[i], flush=True)
    out.flush()
    pd.Series([str(pd.Timestamp(s))]).to_csv(FEAT / f"{out_name}_t0.csv", index=False)


def _channels(days: list[dt.date]) -> np.ndarray:
    S = _cat("spot", days, ["c", "v", "bv"])
    Fu = _cat("um", days, ["c", "v", "bv", "lbv", "lsv"])
    Et = _cat("eth", days, ["c"])
    lps = np.log(pd.Series(S["c"]).ffill().bfill().to_numpy())
    lpf = np.log(pd.Series(Fu["c"]).ffill().bfill().to_numpy())
    lpe = np.log(pd.Series(Et["c"]).ffill().bfill().to_numpy())
    rs = np.diff(lps, prepend=lps[0]); rf = np.diff(lpf, prepend=lpf[0]); re = np.diff(lpe, prepend=lpe[0])
    # causal 1s volatility: EWMA of squared spot returns over ~30 minutes
    s2 = pd.Series(rs ** 2).ewm(halflife=1800, min_periods=600).mean().to_numpy()
    sig = np.sqrt(np.maximum(s2, 1e-14))
    basis = lpf - lps
    bmean = pd.Series(basis).rolling(300, min_periods=1).mean().to_numpy()
    vs = pd.Series(S["v"]).rolling(86400, min_periods=3600).mean().to_numpy() + 1e-9
    vf = pd.Series(Fu["v"]).rolling(86400, min_periods=3600).mean().to_numpy() + 1e-9
    ch = np.stack([
        rs / sig, rf / sig, (basis - bmean) / sig,
        (2 * S["bv"] - S["v"]) / vs, (2 * Fu["bv"] - Fu["v"]) / vf,
        np.log1p(Fu["v"] / vf), (Fu["lbv"] - Fu["lsv"]) / vf,
        re / sig, np.log1p(S["v"] / vs),
    ], axis=1).astype(np.float32)
    return np.clip(np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0), -20, 20)


def load(out_name: str = "seq"):
    ch = np.load(FEAT / f"{out_name}_X.npy", mmap_mode="r")
    t0 = pd.Timestamp(pd.read_csv(FEAT / f"{out_name}_t0.csv").iloc[0, 0])
    return ch, t0


def end_index(ts: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    """Index of the last second inside bar `ts` (the information boundary)."""
    return ((pd.DatetimeIndex(ts) - t0).total_seconds().astype(np.int64) + 59).to_numpy()


if __name__ == "__main__":
    import sys
    build(sys.argv[1], sys.argv[2])
