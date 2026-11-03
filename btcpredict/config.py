from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SEC = DATA / "sec"          # per-second bars, one parquet per venue per day
MIN = DATA / "min"          # minute-level panels
FEAT = DATA / "features"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

BINANCE_VISION = "https://data.binance.vision/data"
BINANCE_LIVE = "https://data-api.binance.vision/api/v3"   # spot, public market data mirror

HORIZONS = (1, 2, 3, 4, 5)   # minutes ahead

for p in (RAW, SEC, MIN, FEAT, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
