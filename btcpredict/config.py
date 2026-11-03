"""Every path and constant used by the project lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- local data (not in git; rebuilt by the pipeline) --------------------------------------
DATA = ROOT / "data"
RAW = DATA / "raw"
SEC = DATA / "sec"              # per-second bars: data/sec/<venue>/<YYYY-MM-DD>.parquet
MIN = DATA / "min"
FEAT = DATA / "features"        # panel.parquet (minute panel + targets), X*.parquet, Ysub.parquet

# ---- versioned artifacts ----------------------------------------------------------------------
MODELS = ROOT / "models"                         # meta.json + <variant>_<target>_<core|sat>.txt.gz
FEATURE_SETS = ROOT / "configs" / "feature_sets"  # which columns each model uses
REPORTS = ROOT / "reports"
HOLDOUT = REPORTS / "holdout"                    # frozen 6-month holdout results (the headline numbers)
CONFIRM = REPORTS / "confirm"                    # untouched DEV "confirmation" months
FIGURES = REPORTS / "figures"
DEV = REPORTS / "dev"                            # research runs: results.jsonl, feature importances
RESEARCH = REPORTS / "research"                  # calibration / regime studies
LIVE_REPORTS = REPORTS / "live"                  # archived live forward-test logs
PREDS = REPORTS / "preds"                        # cached out-of-sample predictions (not in git)
LOGS = ROOT / "logs"                             # live runtime: predictions.jsonl, rls state, features

# ---- endpoints ---------------------------------------------------------------------------------
BINANCE_VISION = "https://data.binance.vision/data"
BINANCE_LIVE = "https://data-api.binance.vision/api/v3"   # spot, public market data mirror

HORIZONS = (1, 2, 3, 4, 5)   # minute horizons; second horizons (10s, 15s) live in features/subminute.py

for p in (RAW, SEC, MIN, FEAT, MODELS, HOLDOUT, CONFIRM, FIGURES, DEV, RESEARCH, LIVE_REPORTS, PREDS):
    p.mkdir(parents=True, exist_ok=True)
