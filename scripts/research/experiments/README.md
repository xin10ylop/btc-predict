# Historical research experiments

These scripts produced the evidence in `docs/FINDINGS.md`. They are kept for reproducibility,
not for production use. The arguments and metrics of every `run_model.py` run are logged in
`reports/dev/results.jsonl`.

| script | question it answered |
|---|---|
| `stack.py` | structural decomposition, online stacking of cached predictions |
| `run_tcn.py` | does a 1-second temporal conv net beat LightGBM? (no) |
| `bench_foundation.py` | zero-shot Kronos / Chronos-Bolt benchmark (not run: needs cloned code) |
| `regime_calibration.py`, `regime_table_calib.py`, `online_recalib.py` | regime-aware P(up) calibration (rejected: static is best) |
| `regime_intervals.py`, `regime_bands.py` | band coverage by regime (led to regime-conditional bands) |
| `exp_baseline.py` | first baselines |
| `queues/*.sh` | the exact experiment batches as they were run |

Run from the repo root, e.g. `python scripts/research/experiments/stack.py --help`.
