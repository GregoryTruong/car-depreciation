"""Ask about one car.  Run:  python -m src.predict Toyota RAV4

Optional:
  --hold 5            how many years you plan to keep the car (default 3)
  --miles 15000       how many miles you drive per year (default: typical from the data)
"""

import argparse

from src.depreciation import (
    find_model,
    load_data,
    miles_per_year,
    new_price_for,
    ownership_cost,
    sweet_spot,
    typical_first_year_value,
    typical_specs,
    value_curve,
    with_new_price,
)


def main():
    parser = argparse.ArgumentParser(description="Depreciation curve and best age to buy a car.")
    parser.add_argument("make", help='e.g. Toyota (use quotes for two words: "Land Rover")')
    parser.add_argument("model", help='e.g. RAV4 or "Grand Cherokee"')
    parser.add_argument("--hold", type=int, default=3, help="years you plan to keep it")
    parser.add_argument("--miles", type=float, help="miles you drive per year")
    args = parser.parse_args()

    listings, new_prices, bundle = load_data()
    try:
        make, model = find_model(listings, args.make, args.model)
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    yearly_miles = args.miles or miles_per_year(listings)
    specs = typical_specs(listings, make, model)
    curve = value_curve(bundle, specs, yearly_miles)
    new_price, source = new_price_for(new_prices, curve, make, model,
                                      typical_first_year_value(listings, new_prices))
    values = with_new_price(curve, new_price)
    spot = sweet_spot(values)
    hold = max(1, min(args.hold, 10 - spot))  # the curve only goes to 10 years
    n_listings = ((listings["make"] == make) & (listings["model"] == model)).sum()

    print(f"\n{make} {model}  ({specs['body']}, {specs['drivetrain']}, {specs['fuel']})")
    print(f"Based on {n_listings} listings, {yearly_miles:,.0f} miles/year")
    print(f"New price: ${new_price:,.0f}  ({source})\n")

    print(f"{'Age':>4}  {'Miles':>8}  {'Value':>9}  {'% of new':>8}  {'Lost that year':>14}")
    for age, value in values.items():
        miles = age * yearly_miles
        lost = "" if age == 0 else f"${values[age - 1] - value:,.0f}"
        marker = "  <- sweet spot" if age == spot else ""
        print(f"{age:>4}  {miles:>8,.0f}  ${value:>8,.0f}  {value / new_price:>8.0%}  {lost:>14}{marker}")

    cost_new = ownership_cost(values, 0, hold)
    cost_spot = ownership_cost(values, spot, hold)
    print(f"\nKeeping it {hold} years, average value lost per year:")
    print(f"  Buy new:            ${cost_new:,.0f}/yr")
    print(f"  Buy {spot} years old:    ${cost_spot:,.0f}/yr  (saves ${cost_new - cost_spot:,.0f}/yr)")
    if n_listings < 40:
        print("\nNote: fewer than 40 listings for this model, so treat the curve as rough.")


if __name__ == "__main__":
    main()
