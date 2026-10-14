"""Step 4: answer the big questions and save charts.  Run:  python -m src.analyze

Questions:
  1. What does depreciation look like for popular models, and when's the sweet spot?
  2. Which models and brands hold their value best and worst?
  3. Do electric cars lose value faster than gas cars?
  4. Reality check: do the model's estimates match real dealer prices?
"""

import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from src.config import DEALER_PRICES_FILE, FIGURES_DIR, REPORTS_DIR
from src.depreciation import (
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
from src.features import prepare_features

# Only rank models with enough listings to trust their curve.
MIN_LISTINGS = 40
# Electric and hybrid models have fewer listings, so the fuel comparison uses a lower bar.
MIN_LISTINGS_FUEL = 15
EXAMPLES = [("Toyota", "Camry"), ("Honda", "CR-V"), ("Ford", "F150"),
            ("Jeep", "Wrangler"), ("Tesla", "Model Y"), ("Ford", "Mustang Mach-E")]

TEAL, GRAY, RED, INK = "#0F766E", "#9CA3AF", "#B91C1C", "#16181D"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": "#C9CDD3", "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#EEF0F2", "xtick.color": INK, "ytick.color": INK, "axes.axisbelow": True,
})


def model_retention(listings, bundle, yearly_miles, min_listings=MIN_LISTINGS) -> pd.DataFrame:
    """For every common model: share of its value kept from year 1 to year 5.

    Comparing year 5 to year 1 (instead of to the new price) means we don't need
    an MSRP for every model, so every model is measured the same way.
    """
    counts = listings.groupby(["make", "model"]).size()
    rows = []
    for (make, model), n in counts[counts >= min_listings].items():
        specs = typical_specs(listings, make, model)
        curve = value_curve(bundle, specs, yearly_miles).set_index("age")["value"]
        rows.append({**specs, "listings": n, "kept_1_to_5": curve[5] / curve[1]})
    return pd.DataFrame(rows).sort_values("kept_1_to_5", ascending=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)


def chart_examples(listings, new_prices, bundle, yearly_miles, first_year_share):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lines = []
    for i, (make, model) in enumerate(EXAMPLES):
        specs = typical_specs(listings, make, model)
        curve = value_curve(bundle, specs, yearly_miles)
        new_price, _ = new_price_for(new_prices, curve, make, model, first_year_share)
        values = with_new_price(curve, new_price)
        spot = sweet_spot(values)
        share = values / new_price * 100
        color = plt.cm.tab10(i)
        ax.plot(share.index, share.values, color=color, label=f"{make} {model}", linewidth=1.8)
        ax.scatter([spot], [share[spot]], color=color, s=36, zorder=3)
        lines.append({"car": f"{make} {model}", "new_price": new_price, "sweet_spot": spot,
                      "kept_at_5": share[5], "cost_new": ownership_cost(values, 0, 3),
                      "cost_spot": ownership_cost(values, spot, 3)})
    ax.set(xlabel="Age (years)", ylabel="Value (% of new price)", ylim=(0, 105),
           title="Depreciation curves (dots = sweet spot to buy)")
    ax.set_xticks(range(0, 11))
    ax.legend(frameon=False, fontsize=9)
    save(fig, "example_curves.png")
    return pd.DataFrame(lines)


def chart_best_worst(retention):
    top, bottom = retention.head(10), retention.tail(10)
    both = pd.concat([top, bottom]).iloc[::-1]
    labels = both["make"] + " " + both["model"]
    colors = [TEAL] * 10 + [RED] * 10
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, both["kept_1_to_5"] * 100, color=colors[::-1])
    ax.set(xlabel="Value kept from year 1 to year 5 (%)",
           title="Models that hold value best (teal) and worst (red)")
    ax.set_xlim(left=both["kept_1_to_5"].min() * 100 - 5)
    ax.grid(axis="y", visible=False)
    save(fig, "best_worst_models.png")


def chart_groups(retention, column, filename, title, min_models=3):
    groups = retention.groupby(column)["kept_1_to_5"].agg(["median", "count"])
    groups = groups[groups["count"] >= min_models].sort_values("median")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(groups) + 1)))
    ax.barh(groups.index, groups["median"] * 100, color=TEAL)
    ax.set(xlabel="Median value kept from year 1 to year 5 (%)", title=title)
    ax.set_xlim(left=groups["median"].min() * 100 - 5)
    ax.grid(axis="y", visible=False)
    save(fig, filename)
    return groups


def reality_check(bundle):
    """Compare model estimates with real dealer asking prices (a separate dataset)."""
    dealer = pd.read_parquet(DEALER_PRICES_FILE)
    dealer["estimate"] = bundle["model"].predict(prepare_features(dealer, bundle["known_models"]))
    dealer["known_model"] = (dealer["make"] + " " + dealer["model"]).isin(bundle["known_models"])
    dealer["ratio"] = dealer["price"] / dealer["estimate"]

    fig, ax = plt.subplots(figsize=(6, 6))
    for known, color, label in ((True, TEAL, "Model in training data"),
                                (False, GRAY, "Model not in training data")):
        part = dealer[dealer["known_model"] == known]
        ax.scatter(part["estimate"], part["price"], s=10, alpha=0.6, color=color, label=label)
    limit = [2_000, 200_000]
    ax.plot(limit, limit, color=INK, linewidth=1, linestyle="--", label="Perfect match")
    ax.set(xscale="log", yscale="log", xlim=limit, ylim=limit,
           xlabel="Model's estimated value ($)", ylabel="Real dealer asking price ($)",
           title="Reality check against real dealer prices")
    dollars = FuncFormatter(lambda x, _: f"${x / 1000:,.0f}k")
    ticks = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(dollars)
        axis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "reality_check.png")
    return dealer


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    listings, new_prices, bundle = load_data()
    yearly_miles = miles_per_year(listings)
    first_year_share = typical_first_year_value(listings, new_prices)

    examples = chart_examples(listings, new_prices, bundle, yearly_miles, first_year_share)
    retention = model_retention(listings, bundle, yearly_miles)
    chart_best_worst(retention)
    brands = chart_groups(retention, "make", "brands.png", "Brands by value kept (years 1 to 5)")
    retention_more = model_retention(listings, bundle, yearly_miles, MIN_LISTINGS_FUEL)
    fuels = chart_groups(retention_more, "fuel", "fuel_types.png",
                         f"Gas vs. hybrid vs. electric (models with {MIN_LISTINGS_FUEL}+ listings)",
                         min_models=2)
    bodies = chart_groups(retention, "body", "body_styles.png", "Value kept by body style")
    dealer = reality_check(bundle)

    known = dealer[dealer["known_model"]]
    within_15 = (known["ratio"].sub(1).abs() <= 0.15).mean()

    report = [
        "# Findings\n",
        "_Generated by `python -m src.analyze`._\n",
        f"- Listings analyzed: **{len(listings):,}** used cars; typical driving "
        f"**{yearly_miles:,.0f} miles/year**.",
        f"- A nearly-new used car is worth **{first_year_share:.0%}** of MSRP on average: "
        f"about **{1 - first_year_share:.0%}** is lost in the first year.\n",
        "## Depreciation and the sweet spot\n",
        "| Car | New price | Sweet spot | Value at 5 yrs | Cost/yr buying new* | "
        "Cost/yr at sweet spot* |",
        "|---|---|---|---|---|---|",
        *[f"| {r.car} | ${r.new_price:,.0f} | {r.sweet_spot} yr{"s" if r.sweet_spot != 1 else ""} | {r.kept_at_5:.0f}% | "
          f"${r.cost_new:,.0f} | ${r.cost_spot:,.0f} |" for r in examples.itertuples()],
        "\n_*Average value lost per year when keeping the car for 3 years._\n",
        "![Depreciation curves](figures/example_curves.png)\n",
        f"## Models that hold value best and worst (models with {MIN_LISTINGS}+ listings)\n",
        "| Best | Kept yr 1-5 | Worst | Kept yr 1-5 |",
        "|---|---|---|---|",
        *[f"| {b.make} {b.model} | {b.kept_1_to_5:.0%} | {w.make} {w.model} | {w.kept_1_to_5:.0%} |"
          for b, w in zip(retention.head(10).itertuples(),
                          retention.tail(10).iloc[::-1].itertuples())],
        "\n![Best and worst models](figures/best_worst_models.png)\n",
        "## Brands, fuel types and body styles\n",
        "| Fuel type | Median value kept yr 1-5 | Models |",
        "|---|---|---|",
        *[f"| {fuel} | {row['median']:.0%} | {row['count']:.0f} |"
          for fuel, row in fuels.iloc[::-1].iterrows()],
        "",
        "![Fuel types](figures/fuel_types.png)\n",
        "![Brands](figures/brands.png)\n",
        "![Body styles](figures/body_styles.png)\n",
        "## Reality check: real dealer prices\n",
        f"Tested on **{len(dealer):,}** cars from a different source (Marketcheck dealer "
        f"listings). For the **{len(known):,}** whose model was in the training data:\n",
        f"- Median dealer price / model estimate: **{known['ratio'].median():.2f}** "
        "(1.00 would be a perfect match; above 1 means dealers ask more than market value)",
        f"- **{within_15:.0%}** of estimates were within 15% of the real asking price\n",
        "![Reality check](figures/reality_check.png)\n",
    ]
    (REPORTS_DIR / "findings.md").write_text("\n".join(report))

    print(examples.round(0).to_string(index=False), "\n")
    print("Best models:", ", ".join(retention.head(5)["model"]))
    print("Worst models:", ", ".join(retention.tail(5)["model"]))
    print("By fuel type:\n", fuels.round(3).to_string())
    print("Top brands:", ", ".join(brands.index[::-1][:5]))
    print("Body styles:\n", bodies.round(3).to_string())
    print(f"\nReality check: median dealer/estimate ratio {known['ratio'].median():.2f}, "
          f"{within_15:.0%} within 15% ({len(known)} cars)")
    print(f"Saved charts to {FIGURES_DIR.relative_to(REPORTS_DIR.parent)} and reports/findings.md")


if __name__ == "__main__":
    main()
