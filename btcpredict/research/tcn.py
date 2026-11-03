"""Cross-venue temporal convolutional network on 1-second bars.

Input per sample:
  * the last T seconds of 9 scale-free channels (spot/perp returns, basis deviation, flows, ...)
  * a static vector of minute-level features (optional)
Output: z-scored forward returns for all horizons 1..5 (multi-task).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.act = nn.GELU()
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.norm = nn.BatchNorm1d(c_out)

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.act(self.norm(y)) + self.res(x)


class CrossVenueTCN(nn.Module):
    def __init__(self, n_ch: int, n_static: int, n_out: int = 5, hid: int = 32, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        layers = []
        c = n_ch
        for d in dilations:
            layers.append(CausalBlock(c, hid, 3, d))
            c = hid
        self.tcn = nn.Sequential(*layers)
        self.static = nn.Sequential(nn.Linear(n_static, 64), nn.GELU()) if n_static else None
        head_in = hid * 3 + (64 if n_static else 0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(head_in, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, seq, stat=None):
        h = self.tcn(seq)                       # (B, hid, T)
        feats = [h[:, :, -1], h[:, :, -10:].mean(-1), h.mean(-1)]
        if self.static is not None:
            feats.append(self.static(stat))
        return self.head(torch.cat(feats, 1))


def gather(ch: np.ndarray, ends: np.ndarray, T: int) -> np.ndarray:
    idx = ends[:, None] + np.arange(-T + 1, 1)[None, :]
    return ch[idx]                              # (B, T, C)


def fit_predict_seq(ch, ends_tr, S_tr, Y_tr, ends_te, S_te, T=180, epochs=4, bs=1024, lr=2e-3, seed=0,
                    hid=32, verbose=False, weight_decay=1e-4):
    """Y_tr: (N, H) z-scored targets (NaN allowed). Returns (N_te, H) predictions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H = Y_tr.shape[1]
    n_static = 0 if S_tr is None else S_tr.shape[1]
    net = CrossVenueTCN(ch.shape[1], n_static, H, hid=hid)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(ends_tr)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1))
    Yt = np.clip(np.nan_to_num(Y_tr, nan=0.0), -6, 6).astype(np.float32)
    M = np.isfinite(Y_tr).astype(np.float32)
    loss_fn = nn.HuberLoss(reduction="none", delta=1.5)
    t = time.time()
    step = 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(n)
        tot = 0.0
        for b in range(n // bs):
            i = perm[b * bs:(b + 1) * bs]
            seq = torch.from_numpy(gather(ch, ends_tr[i], T)).transpose(1, 2)
            st = None if S_tr is None else torch.from_numpy(S_tr[i])
            out = net(seq, st)
            l = (loss_fn(out, torch.from_numpy(Yt[i])) * torch.from_numpy(M[i])).mean()
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot += l.item()
        if verbose:
            print(f"    epoch {ep}: loss {tot / max(n // bs, 1):.4f} {time.time()-t:.0f}s", flush=True)
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(ends_te), 4096):
            seq = torch.from_numpy(gather(ch, ends_te[b:b + 4096], T)).transpose(1, 2)
            st = None if S_te is None else torch.from_numpy(S_te[b:b + 4096])
            preds.append(net(seq, st).numpy())
    return np.concatenate(preds), net
