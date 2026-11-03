"""Load trained artifacts and turn a feature row into calibrated forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS


class Predictor:
    def __init__(self, variant: str = "full", model_dir: Path = MODELS):
        import lightgbm as lgb

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        if variant not in meta:
            raise ValueError(f"variant {variant!r} not trained; available: {list(meta)}")
        self.meta = meta[variant]
        self.variant = variant
        self.boosters = {}
        for key, m in self.meta["targets"].items():
            self.boosters[key] = lgb.Booster(model_file=str(self.dir / m["file"]))

    @property
    def uses_futures(self) -> bool:
        return self.meta["uses_futures"]

    def predict(self, X: pd.DataFrame, sigma: np.ndarray) -> pd.DataFrame:
        """X: feature rows; sigma: per-minute volatility (log-return units) for each row."""
        rows = []
        for key, m in self.meta["targets"].items():
            b = self.boosters[key]
            cols = b.feature_name()
            z = b.predict(X.reindex(columns=cols).values.astype(np.float64))
            h = m["h"]
            scale = sigma * np.sqrt(h)
            a, bb = m["calib"]
            p_up = 1 / (1 + np.exp(-(a + bb * z)))
            exp_ret = m["beta"] * z * scale          # shrink z-forecast to an unbiased return forecast
            q = m["resid_q"]                          # standardized residual quantiles
            for i in range(len(X)):
                rows.append({"target": key, "venue": m["venue"], "h": h, "z": z[i], "p_up": p_up[i],
                             "exp_ret_bp": 1e4 * exp_ret[i],
                             "lo80_bp": 1e4 * (exp_ret[i] + q["0.1"] * scale[i]), "hi80_bp": 1e4 * (exp_ret[i] + q["0.9"] * scale[i]),
                             "lo95_bp": 1e4 * (exp_ret[i] + q["0.025"] * scale[i]), "hi95_bp": 1e4 * (exp_ret[i] + q["0.975"] * scale[i]),
                             "confidence_pct": float(np.mean(np.abs(z[i]) >= np.asarray(m["absz_quantiles"])) * 100),
                             "hist_hit_at_conf": m.get("hit_by_conf")})
        return pd.DataFrame(rows)
