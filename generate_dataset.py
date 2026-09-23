"""
generate_dataset.py
-------------------
Generates a realistic synthetic house-price dataset and saves it to
data/house_prices.csv.  Run this once before anything else.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N = 2000

np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. Raw feature generation
# ------------------------------------------------------------------
locations = ["Downtown", "Suburb", "Rural", "Uptown", "Midtown"]
location = np.random.choice(locations, N)

area_sqft      = np.random.randint(500, 5001, N)          # liveable area in sq ft
bedrooms       = np.random.randint(1, 7, N)
bathrooms      = np.clip(np.random.randint(1, bedrooms + 1, N), 1, 6)
parking_spaces = np.random.randint(0, 4, N)
house_age      = np.random.randint(0, 51, N)              # years
floors         = np.random.randint(1, 4, N)
has_garden     = np.random.choice([0, 1], N, p=[0.4, 0.6])
has_pool       = np.random.choice([0, 1], N, p=[0.8, 0.2])
renovation_year = np.where(
    np.random.rand(N) > 0.5,
    np.random.randint(2000, 2024, N),
    np.nan
)

# Location multiplier (Downtown most expensive)
loc_multiplier = {
    "Downtown": 1.40,
    "Uptown":   1.25,
    "Midtown":  1.10,
    "Suburb":   0.90,
    "Rural":    0.70,
}
loc_factor = np.array([loc_multiplier[l] for l in location])

# ------------------------------------------------------------------
# 2. Price formula  (deterministic base + noise)
# ------------------------------------------------------------------
base_price = (
      120 * area_sqft
    + 15_000 * bedrooms
    + 20_000 * bathrooms
    + 8_000  * parking_spaces
    - 1_500  * house_age
    + 25_000 * has_garden
    + 40_000 * has_pool
    + 10_000 * floors
) * loc_factor

noise = np.random.normal(0, base_price * 0.08)   # ±8 % noise
price = np.round(base_price + noise, -2)          # round to nearest 100
price = np.clip(price, 50_000, 3_000_000)

# ------------------------------------------------------------------
# 3. Inject real-world messiness
# ------------------------------------------------------------------
df = pd.DataFrame({
    "location":        location,
    "area_sqft":       area_sqft,
    "bedrooms":        bedrooms,
    "bathrooms":       bathrooms,
    "parking_spaces":  parking_spaces,
    "house_age":       house_age,
    "floors":          floors,
    "has_garden":      has_garden,
    "has_pool":        has_pool,
    "renovation_year": renovation_year,
    "price":           price,
})

# Missing values (~3 % per column for some columns)
for col in ["area_sqft", "bedrooms", "bathrooms", "house_age"]:
    mask = np.random.rand(N) < 0.03
    df.loc[mask, col] = np.nan

# A few duplicate rows
dup_idx = np.random.choice(df.index, 15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# A handful of impossible values (negative area / age)
bad_idx = np.random.choice(df.index, 5, replace=False)
df.loc[bad_idx, "area_sqft"] = -1
bad_idx2 = np.random.choice(df.index, 5, replace=False)
df.loc[bad_idx2, "house_age"] = -5

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
df.to_csv("data/house_prices.csv", index=False)
print(f"Dataset saved  data/house_prices.csv  ({len(df)} rows, {df.shape[1]} columns)")
