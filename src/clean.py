"""Step 2: clean the raw data into tidy tables.  Run:  python -m src.clean

Both datasets describe cars differently ("All wheel drive" vs "4WD", "TRUCKS" vs
"Pickup"), so this step puts them into the same columns with the same labels.
"""

import re

import pandas as pd

from src.config import (
    DEALER_PRICES_FILE,
    LISTING_YEAR,
    LISTINGS_FILE,
    NEW_PRICES_FILE,
    PROCESSED_DIR,
    RAW_DIR,
)

# Cars older than this are rare in the data and start behaving like classics.
MAX_AGE = 15
MAX_MILEAGE = 300_000

BODY_STYLES = {
    "SUV": "SUV", "TRUCKS": "Truck", "Pickup": "Truck", "SEDAN": "Sedan", "Sedan": "Sedan",
    "HATCH": "Hatchback", "Hatchback": "Hatchback", "COUPE": "Coupe", "Coupe": "Coupe",
    "VANS": "Van", "Minivan": "Van", "Van": "Van", "CONVERT": "Convertible",
    "Convertible": "Convertible", "WAGON": "Wagon", "Wagon": "Wagon",
}


def car_age(model_year: pd.Series) -> pd.Series:
    """Years since the car came out. Next year's models (e.g. a 2027 sold in 2026) count as 0."""
    return (LISTING_YEAR - model_year).clip(lower=0)


def simplify_drivetrain(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.lower()
    if "all" in text or "4" in text:
        return "AWD/4WD"
    if "front" in text or text == "fwd":
        return "FWD"
    if "rear" in text or text == "rwd":
        return "RWD"
    return None


# Words that describe drivetrain or body style, which already have their own columns.
# "RAV4 XLE AWD" and "RAV4 XLE" are the same trim; "Camry SE Sedan 4D" is just "SE".
NOT_TRIM = re.compile(
    r"\b(awd|4wd|fwd|rwd|2wd|4x4|4x2|4matic|sedan|hatchback|coupe|wagon|convertible"
    r"|cabriolet|suv|[24]d)\b",
    flags=re.IGNORECASE,
)


def simplify_trim(value) -> str | None:
    """Trim name without drivetrain or body words ("XLE AWD" -> "XLE").

    None means base or unlisted: AutoTrader often lists a base car's trim as just "AWD"
    or "4MATIC Sedan", and Marketcheck calls it "Base", so all of those become None.
    """
    if not isinstance(value, str):
        return None
    text = NOT_TRIM.sub(" ", value)
    text = re.sub(r"\s+", " ", text).strip(" -")
    if text.lower() == "base" or not re.search(r"[A-Za-z0-9]", text):
        return None
    return text


def simplify_fuel(value) -> str:
    """Group fuel types into Gas, Hybrid, Plug-in Hybrid and Electric."""
    if not isinstance(value, str):
        return "Gas"
    text = value.lower()
    if "plug-in" in text or text in ("phev", "erev"):
        return "Plug-in Hybrid"
    if "hybrid" in text or text == "hev":
        return "Hybrid"
    if text in ("electric", "bev"):
        return "Electric"
    return "Gas"


def clean_autotrader(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (used car values, new car prices per trim)."""
    raw = raw.drop_duplicates("listingId")

    # --- New cars: their MSRP tells us what each trim costs brand new. Prices are kept
    # per trim because a top trim can cost twice the base one (F150 XL vs. Raptor).
    new = raw[(raw["listingType"] == "New") & (raw["msrp"] > 0)]
    new_prices = (
        new.assign(trim=new["trim"].map(simplify_trim))
        # dropna=False keeps base / unlisted trims (None), which are often the most common.
        .groupby(["makeName", "modelName", "trim"], dropna=False)["msrp"]
        .agg(new_price="median", new_listings="count")
        .reset_index()
        .rename(columns={"makeName": "make", "modelName": "model"})
    )

    # --- Used cars: the middle of Kelley Blue Book's fair price range is the car's value.
    used = raw[raw["listingType"] != "New"].copy()
    used["value"] = (used["kbbFairPriceLow"] + used["kbbFairPriceHigh"]) / 2
    used = pd.DataFrame({
        "make": used["makeName"],
        "model": used["modelName"],
        "trim": used["trim"].map(simplify_trim),
        "year": used["year"].astype(int),
        "age": car_age(used["year"]).astype(int),
        "mileage": used["mileage"],
        "body": used["bodyStyle"].map(BODY_STYLES),
        "drivetrain": used["drivetrain"].map(simplify_drivetrain),
        "fuel": used["fuelTypeGroup"].map(simplify_fuel),
        "state": used["sellerState"],
        "value": used["value"],
        "kbb_low": used["kbbFairPriceLow"],
        "kbb_high": used["kbbFairPriceHigh"],
    })
    used = remove_bad_rows(used, price_column="value")
    return used, new_prices


def clean_marketcheck(raw: pd.DataFrame) -> pd.DataFrame:
    """Real dealer asking prices, plus each car's original base MSRP."""
    powertrain = raw["neo_powertrain_type"].fillna(raw["neo_fuel_type"])
    dealer = pd.DataFrame({
        "make": raw["neo_make"],
        "model": raw["neo_model"],
        "trim": raw["neo_trim"].map(simplify_trim),
        "year": raw["neo_year"].astype(int),
        "age": car_age(raw["neo_year"]).astype(int),
        "mileage": raw["miles"],
        "body": raw["neo_body_type"].map(BODY_STYLES),
        "drivetrain": raw["neo_drivetrain"].map(simplify_drivetrain),
        "fuel": powertrain.map(simplify_fuel),
        "state": raw["state"],
        "price": raw["price"],
        "original_msrp": raw["neo_base_msrp"],
    })
    return remove_bad_rows(dealer, price_column="price")


def remove_bad_rows(cars: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Drop rows that would confuse the model: missing values, $0 prices, typos."""
    keep = (
        (cars[price_column] >= 1_000)
        & (cars[price_column] <= 250_000)  # removes supercars and data errors
        & cars["mileage"].between(1, MAX_MILEAGE)
        & (cars["age"] <= MAX_AGE)
        & cars["body"].notna()
        & cars["drivetrain"].notna()
    )
    return cars[keep].reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("autotrader_*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    used, new_prices = clean_autotrader(raw)
    used.to_parquet(LISTINGS_FILE)
    new_prices.to_parquet(NEW_PRICES_FILE)
    print(f"AutoTrader: {len(raw):,} listings -> kept {len(used):,} used cars with a KBB value")
    print(f"            new-car prices for {len(new_prices):,} trims")

    raw_dealer = pd.read_csv(RAW_DIR / "marketcheck.csv", low_memory=False)
    dealer = clean_marketcheck(raw_dealer)
    dealer.to_parquet(DEALER_PRICES_FILE)
    print(f"Marketcheck: {len(raw_dealer):,} listings -> kept {len(dealer):,} with real prices")


if __name__ == "__main__":
    main()
