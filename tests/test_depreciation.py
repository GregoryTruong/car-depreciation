"""Tests for the helpers that turn the model's predictions into a depreciation curve.

These use a tiny fake model instead of the trained one, so they run without
downloading any data.
"""

import numpy as np
import pandas as pd
import pytest

from src.depreciation import (
    MAX_AGE,
    find_model,
    find_trim,
    miles_per_year,
    new_price_for,
    typical_first_year_value,
    typical_specs,
    value_curve,
    with_new_price,
)

LISTINGS = pd.DataFrame({
    "make": ["Toyota", "Toyota", "Toyota", "Toyota", "Honda", "Honda"],
    "model": ["RAV4", "RAV4", "RAV4", "Camry", "Civic", "Civic"],
    "trim": ["XLE", "XLE", "Limited", "SE", None, "EX"],
    "body": ["SUV", "SUV", "Sedan", "Sedan", "Sedan", "Sedan"],
    "drivetrain": ["AWD/4WD", "AWD/4WD", "FWD", "FWD", "FWD", "FWD"],
    "fuel": ["Gas", "Hybrid", "Gas", "Gas", "Gas", "Gas"],
    "age": [1, 2, 4, 0, 1, 3],
    "mileage": [12_000, 20_000, 40_000, 10, 10_000, 36_000],
    "value": [32_000, 29_000, 24_000, 30_000, 24_000, 19_000],
})


class FakeModel:
    """Loses 10% of $40,000 per year of age, ignoring everything else."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return 40_000 * 0.9 ** features["age"].to_numpy()


BUNDLE = {"model": FakeModel(), "known_models": {"Toyota RAV4"}, "known_trims": {"XLE"}}


def test_find_model_ignores_capitalization():
    assert find_model(LISTINGS, "toyota", "rav4") == ("Toyota", "RAV4")


def test_find_model_suggests_models_from_the_same_make():
    with pytest.raises(ValueError, match="Most listed Toyota models: RAV4, Camry"):
        find_model(LISTINGS, "Toyota", "Supra")


def test_find_model_suggests_close_spellings():
    with pytest.raises(ValueError, match="Did you mean: Honda Civic"):
        find_model(LISTINGS, "Hondaa", "Civic")


def test_typical_specs_uses_the_most_common_values():
    specs = typical_specs(LISTINGS, "Toyota", "RAV4")
    assert specs == {"make": "Toyota", "model": "RAV4", "trim": "XLE", "body": "SUV",
                     "drivetrain": "AWD/4WD", "fuel": "Gas"}


def test_typical_specs_for_a_chosen_trim_describe_that_trim():
    specs = typical_specs(LISTINGS, "Toyota", "RAV4", trim="Limited")
    assert (specs["trim"], specs["body"], specs["drivetrain"]) == ("Limited", "Sedan", "FWD")


def test_typical_specs_work_when_no_trim_is_recorded():
    listings = LISTINGS.assign(trim=None)
    assert typical_specs(listings, "Honda", "Civic")["trim"] is None


def test_typical_specs_can_pick_the_base_trim_when_it_is_most_common():
    # Two Civics: one without a listed trim, one EX. Add another base Civic so base wins.
    listings = pd.concat([LISTINGS, LISTINGS.iloc[[4]]])
    assert typical_specs(listings, "Honda", "Civic")["trim"] is None


def test_find_trim_ignores_capitalization():
    assert find_trim(LISTINGS, "Toyota", "RAV4", "xle") == "XLE"


def test_find_trim_base_means_no_listed_trim():
    assert find_trim(LISTINGS, "Honda", "Civic", "base") is None


def test_find_trim_lists_the_trims_it_does_have():
    with pytest.raises(ValueError, match=r"XLE \(2\), Limited \(1\)"):
        find_trim(LISTINGS, "Toyota", "RAV4", "TRD Pro")


def test_miles_per_year_skips_brand_new_cars():
    # The 0-year-old Camry would divide by zero. The rest drive 12k, 10k, 10k, 10k and 12k a year.
    assert miles_per_year(LISTINGS) == 10_000


def test_value_curve_covers_every_age_and_never_goes_up():
    specs = typical_specs(LISTINGS, "Toyota", "RAV4")
    curve = value_curve(BUNDLE, specs, yearly_miles=12_000)
    assert curve["age"].tolist() == list(range(1, MAX_AGE + 1))
    assert curve["mileage"].tolist() == [12_000 * age for age in range(1, MAX_AGE + 1)]
    assert curve["value"].is_monotonic_decreasing


def test_value_curve_smoothing_keeps_a_steady_percentage_decline():
    # The fake model already loses exactly 10% a year, so smoothing shouldn't change it.
    curve = value_curve(BUNDLE, typical_specs(LISTINGS, "Toyota", "RAV4"), 12_000)
    assert np.allclose(curve["value"], curve["predicted"])


def test_new_price_comes_from_new_listings_when_available():
    new_prices = pd.DataFrame({"make": ["Toyota"], "model": ["RAV4"], "trim": ["XLE"],
                               "new_price": [36_000]})
    curve = pd.DataFrame({"age": [1, 2], "value": [30_000, 27_000]})
    specs = {"make": "Toyota", "model": "RAV4", "trim": "XLE"}
    price, source = new_price_for(new_prices, curve, specs, first_year_share=0.8)
    assert price == 36_000
    assert "MSRP" in source


def test_new_price_is_estimated_when_there_are_no_new_listings():
    # There's an MSRP for the RAV4 XLE, but not for the Limited, so it gets estimated.
    new_prices = pd.DataFrame({"make": ["Toyota"], "model": ["RAV4"], "trim": ["XLE"],
                               "new_price": [36_000]})
    curve = pd.DataFrame({"age": [1, 2], "value": [24_000, 21_000]})
    specs = {"make": "Toyota", "model": "RAV4", "trim": "Limited"}
    price, source = new_price_for(new_prices, curve, specs, first_year_share=0.8)
    assert price == pytest.approx(30_000)  # $24,000 at age 1 is 80% of $30,000
    assert "estimated" in source


def test_base_trim_uses_the_base_trim_msrp():
    new_prices = pd.DataFrame({"make": ["Honda", "Honda"], "model": ["Civic", "Civic"],
                               "trim": [None, "EX"], "new_price": [25_000, 30_000]})
    curve = pd.DataFrame({"age": [1, 2], "value": [20_000, 18_000]})
    specs = {"make": "Honda", "model": "Civic", "trim": None}
    assert new_price_for(new_prices, curve, specs, first_year_share=0.8)[0] == 25_000


def test_value_curve_works_for_the_base_trim():
    specs = typical_specs(LISTINGS, "Honda", "Civic", trim=None)
    assert value_curve(BUNDLE, specs, yearly_miles=12_000)["value"].is_monotonic_decreasing


def test_typical_first_year_value_compares_nearly_new_cars_to_msrp():
    new_prices = pd.DataFrame({"make": ["Toyota", "Honda"], "model": ["Camry", "Civic"],
                               "trim": ["SE", "EX"], "new_price": [30_000, 30_000]})
    # Camry SE at age 0: $30k of $30k = 100%. The 1-year-old Civic has no trim, so
    # there's no MSRP to compare it to, and the 3-year-old Civic EX is too old.
    # Add a 1-year-old Civic EX worth $24k of $30k = 80%.
    listings = pd.concat([LISTINGS, LISTINGS.iloc[[4]].assign(trim="EX")])
    assert typical_first_year_value(listings, new_prices) == pytest.approx(0.9)


def test_with_new_price_puts_the_msrp_at_age_zero():
    curve = pd.DataFrame({"age": [1, 2], "value": [30_000, 27_000]})
    values = with_new_price(curve, 38_000)
    assert values.to_dict() == {0: 38_000, 1: 30_000, 2: 27_000}
