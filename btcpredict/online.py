"""Online adaptive layer: recursive least squares with exponential forgetting.

Sits on top of the batch models and re-weights their outputs minute by minute, using only
targets that have already been realized (delay = horizon). This lets the system follow fast
regime changes (e.g. momentum <-> reversal flips) between monthly retrains.
"""
from __future__ import annotations

import numpy as np


def rls_stack(Z: np.ndarray, y: np.ndarray, horizon: int, half_life: float = 2880.0, ridge: float = 50.0,
              prior: np.ndarray | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    """Z: (N, k) inputs ordered in time (one row per minute); y: (N,) realized z-target.
    Returns causal predictions: row t uses only y[s] for s <= t - horizon.
    prior: initial coefficient vector (shrinkage target)."""
    n, k = Z.shape
    lam = 0.5 ** (1.0 / half_life)
    P = np.eye(k) / ridge
    w = np.zeros(k) if prior is None else prior.astype(float).copy()
    out = np.full(n, np.nan)
    valid = np.ones(n, bool) if valid is None else valid
    Zc = np.nan_to_num(Z)
    yc = np.clip(np.nan_to_num(y), -6, 6)
    for t in range(n):
        out[t] = Zc[t] @ w
        s = t - horizon          # the newest row whose target is known at time t (+1 minute bar close)
        if s >= 0 and valid[s] and np.isfinite(y[s]):
            x = Zc[s]
            Px = P @ x
            g = Px / (lam + x @ Px)
            w = w + g * (yc[s] - x @ w)
            P = (P - np.outer(g, Px)) / lam
            # keep P bounded (numerical safety during quiet periods)
            if t % 1000 == 0:
                P = 0.5 * (P + P.T)
                tr = np.trace(P)
                if tr > 1e3 * k / ridge:
                    P *= (1e3 * k / ridge) / tr
    return out
