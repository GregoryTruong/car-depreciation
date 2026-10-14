"""Shared settings: where files live and where the data comes from."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Main dataset: ~30,000 AutoTrader listings with Kelley Blue Book fair price ranges.
# Free sample from Rebrowser (CC BY-NC 4.0), one file per day.
AUTOTRADER_BASE_URL = (
    "https://huggingface.co/datasets/rebrowser/autotrader-dataset/resolve/main/car-listings/data"
)
AUTOTRADER_DAYS = 30
AUTOTRADER_LAST_DAY = "2026-09-30"

# Reality-check dataset: real dealer asking prices from Marketcheck.
MARKETCHECK_URL = (
    "https://huggingface.co/datasets/Marketcheck/us_used_car_inventory_data"
    "/resolve/main/us_used_car_inventory_data.csv"
)

# The year the listings were collected. A 2023 car listed in 2026 is 3 years old.
LISTING_YEAR = 2026

LISTINGS_FILE = PROCESSED_DIR / "listings.parquet"
NEW_PRICES_FILE = PROCESSED_DIR / "new_prices.parquet"
DEALER_PRICES_FILE = PROCESSED_DIR / "dealer_prices.parquet"
MODEL_FILE = MODELS_DIR / "value_model.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
