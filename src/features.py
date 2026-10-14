"""What the model looks at, and how the model is built."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = ["make", "model", "body", "drivetrain", "fuel"]
NUMERIC = ["age", "mileage"]
FEATURES = CATEGORICAL + NUMERIC

# The model can handle at most 255 different values per category, so the
# 250 most common models keep their name and rarer ones become "Other".
MAX_MODELS = 250


def most_common_models(cars: pd.DataFrame) -> set[str]:
    counts = (cars["make"] + " " + cars["model"]).value_counts()
    return set(counts.index[:MAX_MODELS])


def prepare_features(cars: pd.DataFrame, known_models: set[str]) -> pd.DataFrame:
    """Select the model's input columns and replace rare car models with "Other"."""
    features = cars[FEATURES].copy()
    full_name = cars["make"] + " " + cars["model"]
    features.loc[~full_name.isin(known_models), "model"] = "Other"
    return features


def build_model() -> TransformedTargetRegressor:
    """Gradient boosting: hundreds of small decision trees, each fixing the last one's mistakes.

    Two choices worth knowing:
    - It predicts log(value) instead of value, so a $2,000 miss on a $10,000 car
      counts more than a $2,000 miss on an $80,000 car (errors become percentages).
    - Monotonic constraints force the value to go DOWN as age and mileage go up.
      Without them, random noise could make a 6-year-old car "worth more" than
      a 5-year-old one, which would break the depreciation curves.
    """
    encoder = ColumnTransformer(
        [("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1), CATEGORICAL)],
        remainder="passthrough",  # age and mileage go through unchanged
    )
    is_category = [True] * len(CATEGORICAL) + [False] * len(NUMERIC)
    # 0 = no constraint, -1 = value must decrease as this feature increases
    monotonic = [0] * len(CATEGORICAL) + [-1, -1]

    booster = HistGradientBoostingRegressor(
        categorical_features=is_category,
        monotonic_cst=monotonic,
        max_iter=600,
        learning_rate=0.05,
        early_stopping=True,
        random_state=42,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("encode", encoder), ("boost", booster)]),
        func=np.log,
        inverse_func=np.exp,
    )
