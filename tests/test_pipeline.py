"""Tests for the cleaning rules and depreciation math, using small made-up examples."""

import pandas as pd
import pytest

from src.clean import car_age, remove_bad_rows, simplify_drivetrain, simplify_fuel
from src.depreciation import ownership_cost, sweet_spot
from src.features import prepare_features


def test_car_age_treats_next_years_models_as_new():
    assert car_age(pd.Series([2026, 2027, 2020])).tolist() == [0, 0, 6]


@pytest.mark.parametrize("raw, expected", [
    ("All wheel drive", "AWD/4WD"), ("4 wheel drive", "AWD/4WD"), ("4WD", "AWD/4WD"),
    ("2 wheel drive - front", "FWD"), ("FWD", "FWD"),
    ("2 wheel drive - rear", "RWD"), ("RWD", "RWD"), (None, None),
])
def test_drivetrain_labels_from_both_sources_match(raw, expected):
    assert simplify_drivetrain(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Gas", "Gas"), ("Hybrid: Gas/Electric", "Hybrid"), ("HEV", "Hybrid"),
    ("Plug-in Hybrid: Gas/Electric", "Plug-in Hybrid"), ("PHEV", "Plug-in Hybrid"),
    ("Electric", "Electric"), ("BEV", "Electric"), (None, "Gas"),
])
def test_fuel_labels_from_both_sources_match(raw, expected):
    assert simplify_fuel(raw) == expected


def test_remove_bad_rows_drops_impossible_cars():
    cars = pd.DataFrame({
        "value": [25_000, 0, 25_000, 25_000, 900_000],
        "mileage": [40_000, 40_000, 0, 40_000, 40_000],
        "age": [4, 4, 4, 30, 4],
        "body": ["SUV"] * 5,
        "drivetrain": ["FWD"] * 5,
    })
    kept = remove_bad_rows(cars, price_column="value")
    assert len(kept) == 1  # only the first car is realistic


def test_rare_models_become_other():
    cars = pd.DataFrame({
        "make": ["Toyota", "Lotus"], "model": ["Camry", "Emira"], "body": ["Sedan", "Coupe"],
        "drivetrain": ["FWD", "RWD"], "fuel": ["Gas", "Gas"], "age": [3, 2], "mileage": [30e3, 5e3],
    })
    features = prepare_features(cars, known_models={"Toyota Camry"})
    assert features["model"].tolist() == ["Camry", "Other"]


def test_sweet_spot_is_after_the_steep_early_drop():
    # Loses $10k in year 1, $6k in year 2, then $2k a year: the steep part ends at age 2.
    values = pd.Series([50_000, 40_000, 34_000, 32_000, 30_000, 28_000])
    assert sweet_spot(values) == 2


def test_sweet_spot_for_steady_depreciation_is_early():
    # Same loss every year: no steep part to wait out, so buy as young as possible.
    values = pd.Series([30_000, 27_000, 24_000, 21_000, 18_000])
    assert sweet_spot(values) == 1


def test_ownership_cost_is_value_lost_per_year():
    values = pd.Series([40_000, 32_000, 28_000, 25_000, 22_000])
    assert ownership_cost(values, buy_age=0, years=2) == 6_000
    assert ownership_cost(values, buy_age=2, years=2) == 3_000
