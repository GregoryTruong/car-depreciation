"""Depreciation math: value curves, the "sweet spot" age to buy, and cost of ownership."""

import difflib

import joblib
import numpy as np
import pandas as pd

from src.config import LISTINGS_FILE, MODEL_FILE, NEW_PRICES_FILE
from src.features import prepare_features

MAX_AGE = 10
AGES = range(1, MAX_AGE + 1)  # 1 to 10 years old (age 0 is the new price)


def load_data():
    """Load the cleaned listings, the new-car prices and the trained model."""
    bundle = joblib.load(MODEL_FILE)
    if "known_trims" not in bundle:
        raise SystemExit("The saved model is from before trim was added. "
                         "Rebuild it with:  python -m src.clean && python -m src.train")
    return pd.read_parquet(LISTINGS_FILE), pd.read_parquet(NEW_PRICES_FILE), bundle


def predict_values(bundle: dict, cars: pd.DataFrame):
    """The trained model's estimated market value for each car."""
    features = prepare_features(cars, bundle["known_models"], bundle["known_trims"])
    return bundle["model"].predict(features)


def miles_per_year(listings: pd.DataFrame) -> float:
    """Typical yearly driving, from cars at least a year old."""
    older = listings[listings["age"] >= 1]
    return float((older["mileage"] / older["age"]).median())


def find_model(listings: pd.DataFrame, make: str, model: str) -> tuple[str, str]:
    """Match what the user typed ("toyota", "rav4") to the names in the data."""
    names = listings[["make", "model"]].drop_duplicates()
    match = names[(names["make"].str.lower() == make.lower())
                  & (names["model"].str.lower() == model.lower())]
    if not match.empty:
        return match.iloc[0]["make"], match.iloc[0]["model"]

    same_make = listings[listings["make"].str.lower() == make.lower()]
    if not same_make.empty:
        popular = same_make["model"].value_counts().index[:12]
        raise ValueError(f"No listings for {make} {model}. "
                         f"Most listed {same_make['make'].iloc[0]} models: {', '.join(popular)}")

    options = (names["make"] + " " + names["model"]).tolist()
    close = difflib.get_close_matches(f"{make} {model}", options, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"No listings for {make} {model}.{hint}")


BASE = "Base"  # how a missing (base or unlisted) trim is shown and typed


def trim_label(trim: str | None) -> str:
    return BASE if pd.isna(trim) else trim


def same_trim(column: pd.Series, trim: str | None) -> pd.Series:
    """Rows with this trim. None matches base / unlisted trims."""
    return column.isna() if pd.isna(trim) else column == trim


def trims_for(listings: pd.DataFrame, make: str, model: str) -> pd.Series:
    """How many listings each trim of a model has, most common first.

    Base / unlisted trims are counted too, under the name "Base".
    """
    cars = listings[(listings["make"] == make) & (listings["model"] == model)]
    return cars["trim"].fillna(BASE).value_counts()


def find_trim(listings: pd.DataFrame, make: str, model: str, trim: str) -> str | None:
    """Match what the user typed ("xle") to a trim name in the data ("XLE").

    "base" means the base / unlisted trim and returns None.
    """
    trims = trims_for(listings, make, model)
    for name in trims.index:
        if name.lower() == trim.strip().lower():
            return None if name == BASE else name
    options = ", ".join(f"{name} ({count})" for name, count in trims.head(12).items())
    raise ValueError(f"No {make} {model} listings with trim {trim}. "
                     f"Trims with the most listings: {options}")


MOST_COMMON = object()  # default for typical_specs: pick the model's most common trim


def typical_specs(listings: pd.DataFrame, make: str, model: str, trim=MOST_COMMON) -> dict:
    """The most common trim, body, drivetrain and fuel type for a model
    (e.g. RAV4 -> XLE, SUV, AWD, Gas). Pass `trim` to describe that trim instead
    (None = base / unlisted).
    """
    cars = listings[(listings["make"] == make) & (listings["model"] == model)]
    if trim is MOST_COMMON:
        # A base / unlisted trim can be the most common one (most Lexus RX 350s).
        trim = cars["trim"].value_counts(dropna=False).index[0]
        trim = None if pd.isna(trim) else trim
    cars = cars[same_trim(cars["trim"], trim)]
    specs = {"make": make, "model": model, "trim": trim}
    for column in ("body", "drivetrain", "fuel"):
        specs[column] = cars[column].mode().iloc[0]
    return specs


def value_curve(bundle: dict, specs: dict, yearly_miles: float) -> pd.DataFrame:
    """Predicted value of the same car at ages 1-10, driven a typical amount each year.

    The raw predictions are a bit bumpy (one year can randomly look steeper than
    the next), so we also fit a smooth curve through them: a straight line on
    log(value), which means "loses roughly the same percentage every year".
    """
    rows = pd.DataFrame([{**specs, "age": age, "mileage": age * yearly_miles} for age in AGES])
    rows["predicted"] = predict_values(bundle, rows)
    slope, intercept = np.polyfit(rows["age"], np.log(rows["predicted"]), deg=1)
    rows["value"] = np.exp(intercept + slope * rows["age"])
    return rows[["age", "mileage", "predicted", "value"]]


def typical_first_year_value(listings: pd.DataFrame, new_prices: pd.DataFrame) -> float:
    """What a nearly-new used car is worth as a share of the same trim's MSRP, across all models.

    Used to estimate the new price for trims with no new listings in the data.
    """
    # Base / unlisted trims match each other here: pandas joins missing keys together.
    nearly_new = listings[listings["age"] <= 1].merge(new_prices, on=["make", "model", "trim"])
    return float((nearly_new["value"] / nearly_new["new_price"]).median())


def new_price_for(new_prices: pd.DataFrame, curve: pd.DataFrame, specs: dict,
                  first_year_share: float):
    """Today's MSRP for this trim if we have it, otherwise an estimate.

    Only the exact trim counts: an F150 Tremor's MSRP says little about an F150 XLT.
    Returns (price, where_it_came_from).
    """
    match = new_prices[(new_prices["make"] == specs["make"])
                       & (new_prices["model"] == specs["model"])
                       & same_trim(new_prices["trim"], specs["trim"])]
    if not match.empty:
        return float(match.iloc[0]["new_price"]), "average MSRP of new listings of this trim"
    estimate = float(curve["value"].iloc[0]) / first_year_share
    return estimate, "estimated from the typical first-year drop; no new listings of this trim"


def with_new_price(curve: pd.DataFrame, new_price: float) -> pd.Series:
    """Value by age, starting at age 0 = the new price."""
    values = curve.set_index("age")["value"]
    return pd.concat([pd.Series({0: new_price}), values])


def sweet_spot(values_by_age: pd.Series) -> int:
    """First age after which the car loses value slower than its 10-year average.

    Before the sweet spot you're paying for the steep early drop (including
    "driving off the lot"); after it, depreciation has calmed down.
    """
    ages = values_by_age.index
    average_loss = (values_by_age.iloc[0] - values_by_age.iloc[-1]) / (ages[-1] - ages[0])
    for age in ages[1:-1]:
        next_year_loss = values_by_age[age] - values_by_age[age + 1]
        if next_year_loss < average_loss:
            return int(age)
    # Never slows down (same loss every year): nothing to wait for, so buy as young as possible.
    return int(ages[1])


def ownership_cost(values_by_age: pd.Series, buy_age: int, years: int) -> float:
    """Average value lost per year if you buy at `buy_age` (0 = new) and sell `years` later."""
    return (values_by_age[buy_age] - values_by_age[buy_age + years]) / years
