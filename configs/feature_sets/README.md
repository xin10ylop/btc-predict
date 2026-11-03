# Feature sets

Each file is one column name per line, selecting from the model matrix (`X` + `extra_features` +
`Xalt` + `Xcb` + `Xalt2`). `scripts/pipeline/train_final.py` reads them. The exact list each
production booster was trained on is also stored in `models/meta.json` (`targets.<key>.features`),
and that stored copy is what the live predictor uses.

| file | status | used for |
|---|---|---|
| `core.txt`, `core_fast.txt` | **production** | full model, core part. BTC spot + perp + ETH spot microstructure, minute state |
| `sat.txt`, `sat_fast.txt` | **production** | full model, satellite part. core + 5-alt swarm + BTC/USDC book + Coinbase premium + 17-coin swarm + ETH perp |
| `spotcore.txt`, `spotcore_fast.txt` | **production** | spot-only model, core part (no futures data) |
| `spotsat.txt`, `spotsat_fast.txt` | **production** | spot-only model, satellite part |
| `noaux.txt` | research | everything computable live (no depth / open-interest), incl. BTC/FDUSD |
| `nofd.txt` | research | `noaux` minus BTC/FDUSD |
| `spotonly.txt`, `spotonly_nofd.txt` | research | spot-only predecessors |
| `fast.txt`, `fast_nofd.txt` | research | `noaux` / `nofd` without slow "regime" features |

Rules:
* `*_fast` sets serve horizons ≥ 2 minutes, where slow regime features added zero IC.
* The h = 1 m and 10 s/15 s models use the full sets.
* Never add BTC/FDUSD features back to production. That venue's liquidity collapsed in Feb 2026;
  see `docs/FINDINGS.md`.
