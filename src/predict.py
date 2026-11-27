"""Ask about one car.  Run:  python -m src.predict Toyota RAV4

Optional:
  --hold 5            how many years you plan to keep the car (default 3)
  --miles 15000       how many miles you drive per year (default: typical from the data)
  --trim XLE          a specific trim (default: the model's most common trim)
"""

import argparse

from src.depreciation import (
    MAX_AGE,
    find_model,
    find_trim,
    load_data,
    miles_per_year,
    new_price_for,
    ownership_cost,
    sweet_spot,
    trims_for,
    typical_first_year_value,
    typical_specs,
    value_curve,
    with_new_price,
)


def years_old(age: int) -> str:
    return "1 year old" if age == 1 else f"{age} years old"


def main():
    parser = argparse.ArgumentParser(description="Depreciation curve and best age to buy a car.")
    parser.add_argument("make", help='e.g. Toyota (use quotes for two words: "Land Rover")')
    parser.add_argument("model", help='e.g. RAV4 or "Grand Cherokee"')
    parser.add_argument("--hold", type=int, default=3, help="years you plan to keep it")
    parser.add_argument("--miles", type=float, help="miles you drive per year")
    parser.add_argument("--trim", help="e.g. XLE (default: the most common trim)")
    args = parser.parse_args()
    if args.hold < 1:
        parser.error("--hold must be at least 1 year")
    if args.miles is not None and args.miles <= 0:
        parser.error("--miles must be more than 0")

    listings, new_prices, bundle = load_data()
    try:
        make, model = find_model(listings, args.make, args.model)
        trim = find_trim(listings, make, model, args.trim) if args.trim else None
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    yearly_miles = args.miles if args.miles is not None else miles_per_year(listings)
    typical = typical_specs(listings, make, model)
    specs = typical_specs(listings, make, model, trim) if trim else typical
    curve = value_curve(bundle, specs, yearly_miles)
    new_price, source = new_price_for(new_prices, curve, specs,
                                      typical_first_year_value(listings, new_prices))
    values = with_new_price(curve, new_price)
    spot = sweet_spot(values)
    hold = min(args.hold, MAX_AGE - spot)  # the curve only goes to MAX_AGE years
    n_listings = ((listings["make"] == make) & (listings["model"] == model)).sum()
    trims = trims_for(listings, make, model)

    name = f"{make} {model} {specs['trim']}" if specs["trim"] else f"{make} {model}"
    print(f"\n{name}  ({specs['body']}, {specs['drivetrain']}, {specs['fuel']})")
    trim_note = f", {trims.get(specs['trim'], 0)} of them this trim" if specs["trim"] else ""
    print(f"Based on {n_listings} listings{trim_note}, {yearly_miles:,.0f} miles/year")
    if not args.trim and len(trims) > 1:
        others = ", ".join(t for t in trims.index[:6] if t != specs["trim"])
        print(f"Most common trim shown. Others: {others}  (use --trim)")
    print(f"New price: ${new_price:,.0f}  ({source})\n")

    print(f"{'Age':>4}  {'Miles':>8}  {'Value':>9}  {'% of new':>8}  {'Lost that year':>14}")
    for age, value in values.items():
        miles = age * yearly_miles
        lost = "" if age == 0 else f"${values[age - 1] - value:,.0f}"
        marker = "  <- sweet spot" if age == spot else ""
        print(f"{age:>4}  {miles:>8,.0f}  ${value:>8,.0f}  {value / new_price:>8.0%}  {lost:>14}{marker}")

    cost_new = ownership_cost(values, 0, hold)
    cost_spot = ownership_cost(values, spot, hold)
    if hold < args.hold:
        print(f"\nNote: the curve stops at {MAX_AGE} years, so a car bought "
              f"{years_old(spot)} can only be followed for {hold} more.")
    print(f"\nKeeping it {hold} year{'s' if hold != 1 else ''}, average value lost per year:")
    buy_spot = f"Buy {years_old(spot)}:"
    print(f"  {'Buy new:':<20}${cost_new:,.0f}/yr")
    print(f"  {buy_spot:<20}${cost_spot:,.0f}/yr  (saves ${cost_new - cost_spot:,.0f}/yr)")
    if n_listings < 40:
        print("\nNote: fewer than 40 listings for this model, so treat the curve as rough.")
    elif args.trim and trims.get(specs["trim"], 0) < 15:
        print("\nNote: fewer than 15 listings for this trim, so its curve leans on similar "
              "cars and is rougher than the model's.")


if __name__ == "__main__":
    main()
