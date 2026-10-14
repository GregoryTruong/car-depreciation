"""Step 3: train the value model and measure how accurate it is.  Run:  python -m src.train

To judge a model honestly you need something to compare it against, so this
trains two simple "baselines" first and then the real model:
  1. Guess the median value for every car (the dumbest possible guess)
  2. Linear regression (a straight-line formula, the classic starting point)
  3. Gradient boosting (the real model)
All three are tested on 20% of cars they never saw during training.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LISTINGS_FILE, METRICS_FILE, MODEL_FILE, MODELS_DIR, REPORTS_DIR
from src.features import CATEGORICAL, NUMERIC, build_model, most_common_models, prepare_features


def score(name: str, actual: pd.Series, predicted: np.ndarray, test: pd.DataFrame) -> dict:
    """Several ways of measuring error, each answering a different question."""
    percent_errors = np.abs(predicted - actual) / actual
    within_kbb_range = (predicted >= test["kbb_low"]) & (predicted <= test["kbb_high"])
    return {
        "model": name,
        "average_error_dollars": round(mean_absolute_error(actual, predicted)),
        "median_error_percent": round(100 * float(np.median(percent_errors)), 1),
        "r2": round(r2_score(actual, predicted), 3),
        "within_kbb_range_percent": round(100 * float(within_kbb_range.mean()), 1),
    }


def linear_baseline() -> TransformedTargetRegressor:
    encoder = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numbers", StandardScaler(), NUMERIC),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("encode", encoder), ("linear", Ridge(alpha=1.0))]),
        func=np.log, inverse_func=np.exp,
    )


def main():
    cars = pd.read_parquet(LISTINGS_FILE)
    known_models = most_common_models(cars)
    X = prepare_features(cars, known_models)
    y = cars["value"]

    # random_state=42 makes the split the same every run, so results are repeatable.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_cars = cars.loc[X_test.index]
    print(f"Training on {len(X_train):,} cars, testing on {len(X_test):,} unseen cars\n")

    results = [score("Median guess", y_test, np.full(len(y_test), y_train.median()), test_cars)]

    linear = linear_baseline().fit(X_train, y_train)
    results.append(score("Linear regression", y_test, linear.predict(X_test), test_cars))

    model = build_model().fit(X_train, y_train)
    results.append(score("Gradient boosting", y_test, model.predict(X_test), test_cars))

    table = pd.DataFrame(results).set_index("model")
    table.columns = ["Avg error ($)", "Median error (%)", "R²", "Inside KBB range (%)"]
    print(table.to_string(), "\n")

    # Which inputs matter most? Shuffle one column at a time and see how much worse
    # the model gets. A big drop means the model relies on that column.
    sample = X_test.sample(2000, random_state=42)
    importance = permutation_importance(model, sample, y_test.loc[sample.index],
                                        n_repeats=5, random_state=42, scoring="r2")
    ranked = sorted(zip(sample.columns, importance.importances_mean), key=lambda p: -p[1])
    print("What the model relies on most (drop in R² when shuffled):")
    for feature, drop in ranked:
        print(f"  {feature:<11} {drop:.3f}")

    # Retrain on ALL the data for the final model now that we know it works.
    final_model = build_model().fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "known_models": known_models}, MODEL_FILE)

    REPORTS_DIR.mkdir(exist_ok=True)
    METRICS_FILE.write_text(json.dumps({
        "train_size": len(X_train),
        "test_size": len(X_test),
        "results": results,
        "feature_importance": {f: round(float(d), 4) for f, d in ranked},
    }, indent=2))
    print(f"\nSaved model to {MODEL_FILE.relative_to(MODEL_FILE.parents[1])}")


if __name__ == "__main__":
    main()
