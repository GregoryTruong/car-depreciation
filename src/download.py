"""Step 1: download the raw data.  Run:  python -m src.download"""

import pandas as pd
import requests

from src.config import (
    AUTOTRADER_BASE_URL,
    AUTOTRADER_DAYS,
    AUTOTRADER_LAST_DAY,
    MARKETCHECK_URL,
    RAW_DIR,
)


def download(url: str, destination) -> None:
    if destination.exists():
        print(f"  already have {destination.name}")
        return
    print(f"  downloading {destination.name}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # stop with an error if the download failed
    destination.write_bytes(response.content)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("AutoTrader listings (one file per day):")
    last_day = pd.Timestamp(AUTOTRADER_LAST_DAY)
    for offset in range(AUTOTRADER_DAYS):
        day = (last_day - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        download(f"{AUTOTRADER_BASE_URL}/{day}.parquet", RAW_DIR / f"autotrader_{day}.parquet")

    print("Marketcheck dealer prices:")
    download(MARKETCHECK_URL, RAW_DIR / "marketcheck.csv")
    print("Done.")


if __name__ == "__main__":
    main()
