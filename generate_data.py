"""
generate_data.py
Generates a realistic multi-category sales dataset (3 years, monthly).
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

CATEGORIES = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
REGIONS    = ["North", "South", "East", "West"]

# Monthly seasonality multipliers (Jan–Dec)
SEASONALITY = {
    "Electronics": [0.80, 0.75, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00, 1.10, 1.35, 1.55],
    "Clothing":    [0.70, 0.72, 0.95, 1.05, 1.10, 1.00, 0.95, 0.90, 1.00, 1.05, 1.20, 1.38],
    "Groceries":   [1.00, 0.95, 1.00, 1.05, 1.10, 1.15, 1.10, 1.08, 1.02, 1.00, 1.05, 1.20],
    "Furniture":   [0.80, 0.82, 0.95, 1.05, 1.15, 1.10, 1.00, 1.05, 1.10, 1.00, 0.90, 0.88],
    "Sports":      [0.85, 0.85, 1.00, 1.10, 1.25, 1.20, 1.15, 1.10, 1.00, 0.95, 0.90, 0.85],
}

BASE_SALES = {
    "Electronics": 500000,
    "Clothing":    300000,
    "Groceries":   450000,
    "Furniture":   250000,
    "Sports":      200000,
}

REGION_FACTOR = {"North": 1.10, "South": 0.95, "East": 1.05, "West": 1.00}

rows = []
start = pd.Timestamp("2022-01-01")

for year_offset in range(3):        # 2022, 2023, 2024
    for month in range(1, 13):
        date = start + pd.DateOffset(years=year_offset, months=month - 1)
        trend_factor = 1 + (year_offset * 12 + month - 1) * 0.005   # 0.5% monthly growth

        for category in CATEGORIES:
            for region in REGIONS:
                base      = BASE_SALES[category]
                season    = SEASONALITY[category][month - 1]
                reg_f     = REGION_FACTOR[region]
                noise     = np.random.normal(1.0, 0.06)

                ad_spend  = np.random.randint(5000, 50000)
                discount  = round(np.random.uniform(0, 30), 1)

                # Ad spend and discount affect sales
                ad_boost  = 1 + (ad_spend / 500000)
                disc_lift = 1 + (discount / 200)

                units_sold = int(
                    (base / 500) * season * trend_factor * reg_f * noise * ad_boost * disc_lift
                )
                price_per_unit = round(np.random.uniform(150, 800) if category in ["Electronics", "Furniture"]
                                       else np.random.uniform(20, 200), 2)
                revenue = round(units_sold * price_per_unit * (1 - discount / 100), 2)

                rows.append({
                    "date":           date.strftime("%Y-%m-%d"),
                    "year":           date.year,
                    "month":          date.month,
                    "quarter":        (date.month - 1) // 3 + 1,
                    "category":       category,
                    "region":         region,
                    "ad_spend":       ad_spend,
                    "discount_pct":   discount,
                    "units_sold":     units_sold,
                    "price_per_unit": price_per_unit,
                    "revenue":        revenue,
                })

df = pd.DataFrame(rows)
df.to_csv("data/sales.csv", index=False)

print(f"✅ Dataset generated: {len(df)} records")
print(df.head())
print(f"\nDate range : {df['date'].min()}  →  {df['date'].max()}")
print(f"Total revenue : ₹{df['revenue'].sum():,.0f}")
print(f"Categories : {df['category'].unique().tolist()}")
