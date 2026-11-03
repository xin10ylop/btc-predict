"""Load trained artifacts and turn feature rows into calibrated forecasts.

Per target: z_core, z_sat (LightGBM) -> z = RLS([1, z_core, z_sat]) -> P(up), E[r], bands.
The RLS layer keeps learning live (update()) and its state is persisted between runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS, ROOT
from .online import OnlineRLS, PriorRLS


def _load_booster(path: Path):
    """Load a LightGBM text model, falling back to the gzip-compressed copy shipped in the repo."""
    import gzip

    import lightgbm as lgb

    if path.exists():
        return lgb.Booster(model_file=str(path))
    gz = path.with_name(path.name + ".gz")
    with gzip.open(gz, "rt") as fh:
        return lgb.Booster(model_str=fh.read())


class Predictor:
    def __init__(self, variant: str = "full", model_dir: Path = MODELS, state_file: Path | None = None):
        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        if variant not in meta:
            raise ValueError(f"variant {variant!r} not trained; available: {list(meta)}")
        self.meta = meta[variant]
        self.variant = variant
        self.boosters = {k: {part: _load_booster(self.dir / f) for part, f in m["files"].items()}
                         for k, m in self.meta["targets"].items()}
        self.state_file = state_file or (ROOT / "logs" / f"rls_state_{variant}.json")
        saved = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.rls = {}
        for k, m in self.meta["targets"].items():
            st = saved.get(k, m["rls_state"])
            self.rls[k] = PriorRLS.from_state(st) if st.get("type") == "prior" else OnlineRLS.from_state(st)

    @property
    def uses_futures(self) -> bool:
        return self.meta["uses_futures"]

    def raw(self, X: pd.DataFrame) -> dict:
        out = {}
        for key, parts in self.boosters.items():
            out[key] = {p: b.predict(X.reindex(columns=b.feature_name()).values.astype(np.float64)) for p, b in parts.items()}
        return out

    def predict(self, X: pd.DataFrame, sigma: np.ndarray) -> pd.DataFrame:
        """X: feature rows; sigma: per-minute volatility (log-return units) for each row."""
        raw = self.raw(X)
        rows = []
        for key, m in self.meta["targets"].items():
            zc, zs = raw[key]["core"], raw[key]["sat"]
            z = np.array([self.rls[key].predict(self._x(key, a, b)) for a, b in zip(zc, zs)])
            h = m["h"]
            scale = sigma * np.sqrt(h)
            a, bb = m["calib"]
            p_up = 1 / (1 + np.exp(-(a + bb * z)))
            exp_ret = m["beta"] * z * scale
            q = m["resid_q"]
            conf = [float(np.mean(abs(zi) >= np.asarray(m["absz_quantiles"])) * 100) for zi in z]
            dec = m.get("hit_by_conf_decile") or [np.nan] * 10
            for i in range(len(X)):
                rows.append({"target": key, "venue": m["venue"], "h": h, "z": z[i], "z_core": zc[i], "z_sat": zs[i],
                             "p_up": p_up[i], "exp_ret_bp": 1e4 * exp_ret[i],
                             "lo80_bp": 1e4 * (exp_ret[i] + q["0.1"] * scale[i]), "hi80_bp": 1e4 * (exp_ret[i] + q["0.9"] * scale[i]),
                             "lo95_bp": 1e4 * (exp_ret[i] + q["0.025"] * scale[i]), "hi95_bp": 1e4 * (exp_ret[i] + q["0.975"] * scale[i]),
                             "confidence_pct": conf[i],
                             "hist_hit_at_conf": dec[min(int(conf[i] // 10), len(dec) - 1)]})
        return pd.DataFrame(rows)

    def _x(self, key, z_core, z_sat):
        # layouts: [core, sat] (current) or [1, core, sat] (older artifacts)
        return [z_core, z_sat] if len(self.rls[key].w) == 2 else [1.0, z_core, z_sat]

    def update(self, key: str, z_core: float, z_sat: float, realized_z: float):
        self.rls[key].update(self._x(key, z_core, z_sat), realized_z)

    def save_state(self):
        self.state_file.parent.mkdir(exist_ok=True)
        self.state_file.write_text(json.dumps({k: r.state() for k, r in self.rls.items()}))
