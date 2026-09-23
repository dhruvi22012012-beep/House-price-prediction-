"""
train.py
========
Standalone Model Training Script
---------------------------------
This script focuses solely on training the best model (Gradient Boosting)
with hyperparameter tuning and saving it to the models/ directory.

It assumes the dataset already exists at data/house_prices.csv.
If it does not, run:  python generate_dataset.py  first.

Usage:
    python train.py
    python train.py --data data/house_prices.csv
    python train.py --data data/house_prices.csv --output models/
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection   import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing     import OrdinalEncoder
from sklearn.ensemble          import GradientBoostingRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score

# ── Constants ─────────────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.20
LOCATIONS  = ["Rural", "Suburb", "Midtown", "Uptown", "Downtown"]
FEATURES   = [
    "location_enc", "area_sqft", "bedrooms", "bathrooms",
    "parking_spaces", "house_age", "floors",
    "has_garden", "has_pool", "ever_renovated",
]
TARGET     = "price"

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the House Price Prediction model.")
parser.add_argument("--data",   default="data/house_prices.csv",
                    help="Path to the CSV dataset (default: data/house_prices.csv)")
parser.add_argument("--output", default="models/",
                    help="Directory to save model artefacts (default: models/)")
args = parser.parse_args()

DATA_PATH  = args.data
OUTPUT_DIR = args.output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading dataset")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'.")
    print("  Run:  python generate_dataset.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns from '{DATA_PATH}'")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Clean
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Cleaning data")
print("=" * 60)

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Dropped {before - len(df)} duplicate row(s)")

# Fix impossible values
df.loc[df["area_sqft"] < 0, "area_sqft"] = np.nan
df.loc[df["house_age"]  < 0, "house_age"]  = np.nan

# Impute numeric columns with median
for col in ["area_sqft", "bedrooms", "bathrooms", "house_age"]:
    median_val = df[col].median()
    filled     = df[col].isna().sum()
    df[col].fillna(median_val, inplace=True)
    if filled:
        print(f"  Imputed {filled} NaN(s) in '{col}' with median={median_val:.1f}")

# renovation_year -> binary flag
df["ever_renovated"] = df["renovation_year"].notna().astype(int)
df.drop(columns=["renovation_year"], inplace=True)

# Enforce int dtypes
for col in ["bedrooms", "bathrooms", "parking_spaces", "house_age",
            "floors", "has_garden", "has_pool", "ever_renovated"]:
    df[col] = df[col].astype(int)

print(f"  Clean shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Encode
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Encoding categorical features")
print("=" * 60)

encoder = OrdinalEncoder(categories=[LOCATIONS])
df["location_enc"] = encoder.fit_transform(df[["location"]]).astype(int)
df.drop(columns=["location"], inplace=True)
print(f"  OrdinalEncoded 'location' -> {dict(zip(LOCATIONS, range(len(LOCATIONS))))}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Train / test split")
print("=" * 60)

X = df[FEATURES]
y = df[TARGET]

# Stratify on price quartile to preserve price distribution in both splits
y_quartile = pd.qcut(y, q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y_quartile
)
print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Hyperparameter tuning with RandomizedSearchCV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Hyperparameter tuning (RandomizedSearchCV, n_iter=20, cv=5)")
print("=" * 60)

param_grid = {
    "n_estimators":      [100, 200, 300],
    "learning_rate":     [0.05, 0.10, 0.20],
    "max_depth":         [3, 5, 7],
    "subsample":         [0.7, 0.85, 1.0],
    "min_samples_split": [2, 5],
}

base_model = GradientBoostingRegressor(random_state=SEED)
cv         = KFold(n_splits=5, shuffle=True, random_state=SEED)

search = RandomizedSearchCV(
    base_model,
    param_grid,
    n_iter=20,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)
search.fit(X_train, y_train)

best_model  = search.best_estimator_
best_params = search.best_params_
best_cv_r2  = search.best_score_

print(f"\n  Best params : {best_params}")
print(f"  Best CV R2  : {best_cv_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Final evaluation on held-out test set")
print("=" * 60)

y_pred = best_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)
r2     = r2_score(y_test, y_pred)

print(f"  MAE  : ${mae:,.0f}")
print(f"  MSE  : ${mse:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  R2   : {r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Save artefacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 7 – Saving artefacts to '{OUTPUT_DIR}'")
print("=" * 60)

model_path   = os.path.join(OUTPUT_DIR, "final_model.pkl")
encoder_path = os.path.join(OUTPUT_DIR, "location_encoder.pkl")
features_path = os.path.join(OUTPUT_DIR, "feature_names.pkl")
summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")

joblib.dump(best_model, model_path)
joblib.dump(encoder,    encoder_path)
joblib.dump(FEATURES,   features_path)

summary = {
    "model":       "GradientBoostingRegressor",
    "best_params": best_params,
    "cv_r2":       round(best_cv_r2, 4),
    "test_metrics": {
        "MAE":  round(mae,  2),
        "MSE":  round(mse,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
    },
    "train_rows": len(X_train),
    "test_rows":  len(X_test),
    "features":   FEATURES,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Saved: {model_path}")
print(f"  Saved: {encoder_path}")
print(f"  Saved: {features_path}")
print(f"  Saved: {summary_path}")
print("\nTraining complete. Run  python predict.py  to use the model.\n")
