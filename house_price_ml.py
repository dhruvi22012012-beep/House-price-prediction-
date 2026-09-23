"""
house_price_ml.py
=================
End-to-end House Price Prediction Pipeline
------------------------------------------
Steps covered
  1.  Load dataset
  2.  Analyze  – columns, dtypes, missing values, basic stats
  3.  Clean    – duplicates, impossible values, missing-value imputation
  4.  EDA      – correlation heatmap + distribution plots (saved as PNG)
  5.  Encode   – ordinal-encode location; binary columns stay as-is
  6.  Feature selection – drop low-importance / leaky columns
  7.  Train / test split  (80 / 20, stratified by price quartile)
  8.  Train four regression models
  9.  Compare  – MAE, MSE, RMSE, R²
  10. Hyperparameter tuning on the best model (RandomizedSearchCV)
  11. Evaluate final model on the held-out test set
  12. Save model artefacts to models/

Run:  python house_price_ml.py
"""

# -- stdlib --------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings("ignore")

# -- third-party ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (works everywhere)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing     import OrdinalEncoder
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.linear_model       import LinearRegression
from sklearn.tree               import DecisionTreeRegressor
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

os.makedirs("models",  exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load dataset
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 – Loading dataset")
print("=" * 65)

DATA_PATH = "data/house_prices.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found.  Run  python generate_dataset.py  first."
    )

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Analyze
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 2 – Dataset analysis")
print("=" * 65)

print("\n-- Columns & data types --")
print(df.dtypes.to_string())

print("\n-- Missing values --")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "  None")

print("\n-- Basic statistics (numeric columns) --")
print(df.describe(include="all").T.to_string())

print("\n-- Duplicate rows --")
print(f"  {df.duplicated().sum()} duplicate(s) found")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Clean
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 – Data cleaning")
print("=" * 65)

# 3a. Drop exact duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Dropped {before - len(df)} duplicate row(s)")

# 3b. Remove impossible values  (negative area or age)
bad_area = df["area_sqft"] < 0
bad_age  = df["house_age"] < 0
df.loc[bad_area, "area_sqft"] = np.nan
df.loc[bad_age,  "house_age"] = np.nan
print(f"  Set {bad_area.sum()} negative area_sqft -> NaN")
print(f"  Set {bad_age.sum()}  negative house_age  -> NaN")

# 3c. Impute numeric NaNs with median
num_cols_with_nan = ["area_sqft", "bedrooms", "bathrooms", "house_age"]
for col in num_cols_with_nan:
    median_val = df[col].median()
    n_filled   = df[col].isna().sum()
    df[col].fillna(median_val, inplace=True)
    if n_filled:
        print(f"  Imputed {n_filled} NaN(s) in '{col}' with median={median_val:.1f}")

# 3d. renovation_year: NaN means 'never renovated'; fill with 0 (flag)
df["ever_renovated"] = df["renovation_year"].notna().astype(int)
df.drop(columns=["renovation_year"], inplace=True)

# 3e. Ensure correct dtypes
int_cols = ["bedrooms", "bathrooms", "parking_spaces", "house_age",
            "floors", "has_garden", "has_pool", "ever_renovated"]
for col in int_cols:
    df[col] = df[col].astype(int)

print(f"\n  Clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – EDA (save plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 – Exploratory Data Analysis (plots saved to plots/)")
print("=" * 65)

sns.set_theme(style="whitegrid", palette="muted")

# 4a. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["price"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Count")
axes[1].hist(np.log1p(df["price"]), bins=50, color="coral", edgecolor="white")
axes[1].set_title("Log-Price Distribution")
axes[1].set_xlabel("log(Price + 1)")
plt.tight_layout()
plt.savefig("plots/01_price_distribution.png", dpi=120)
plt.close()
print("  Saved plots/01_price_distribution.png")

# 4b. Price by location
fig, ax = plt.subplots(figsize=(8, 4))
order = df.groupby("location")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="location", y="price", order=order, ax=ax)
ax.set_title("Price by Location")
ax.set_xlabel("Location")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig("plots/02_price_by_location.png", dpi=120)
plt.close()
print("  Saved plots/02_price_by_location.png")

# 4c. Correlation heatmap (numeric features only)
numeric_df = df.select_dtypes(include=[np.number])
fig, ax    = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=120)
plt.close()
print("  Saved plots/03_correlation_heatmap.png")

# 4d. Price vs. Area scatter
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["area_sqft"], df["price"], alpha=0.3, s=8, color="steelblue")
ax.set_title("Price vs. Area (sqft)")
ax.set_xlabel("Area (sqft)")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig("plots/04_price_vs_area.png", dpi=120)
plt.close()
print("  Saved plots/04_price_vs_area.png")

# 4e. Price vs. House Age
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["house_age"], df["price"], alpha=0.3, s=8, color="coral")
ax.set_title("Price vs. House Age (years)")
ax.set_xlabel("House Age (years)")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig("plots/05_price_vs_age.png", dpi=120)
plt.close()
print("  Saved plots/05_price_vs_age.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Encode categorical features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 – Encoding categorical features")
print("=" * 65)

# Ordinal encode 'location'
loc_order = ["Rural", "Suburb", "Midtown", "Uptown", "Downtown"]
enc = OrdinalEncoder(categories=[loc_order])
df["location_enc"] = enc.fit_transform(df[["location"]]).astype(int)
print(f"  OrdinalEncoder mapping: {dict(zip(loc_order, range(len(loc_order))))}")
df.drop(columns=["location"], inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Feature selection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 – Feature selection")
print("=" * 65)

FEATURES = [
    "location_enc", "area_sqft", "bedrooms", "bathrooms",
    "parking_spaces", "house_age", "floors",
    "has_garden", "has_pool", "ever_renovated",
]
TARGET = "price"

X = df[FEATURES]
y = df[TARGET]
print(f"  Selected {len(FEATURES)} features: {FEATURES}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – Train / test split  (no data leakage)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 – Train / test split (80 / 20)")
print("=" * 65)

# Stratify by price quartile to ensure representative splits
y_quartile = pd.qcut(y, q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y_quartile
)
print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 & 9 – Train four models and compare
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8 & 9 – Model training and comparison")
print("=" * 65)

MODELS = {
    "Linear Regression":        LinearRegression(),
    "Decision Tree":            DecisionTreeRegressor(random_state=SEED),
    "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=SEED),
}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """Fit model and return a dict of validation metrics."""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae   = mean_absolute_error(y_te, preds)
    mse   = mean_squared_error(y_te, preds)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_te, preds)
    print(f"\n  [{name}]")
    print(f"    MAE  : ${mae:>12,.0f}")
    print(f"    MSE  : ${mse:>12,.0f}")
    print(f"    RMSE : ${rmse:>12,.0f}")
    print(f"    R²   : {r2:.4f}")
    return {"model": model, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

results = {}
for name, mdl in MODELS.items():
    results[name] = evaluate(name, mdl, X_train, y_train, X_test, y_test)

# -- Comparison table ----------------------------------------------------------
metrics_df = pd.DataFrame(
    {k: {"MAE": v["MAE"], "MSE": v["MSE"], "RMSE": v["RMSE"], "R²": v["R2"]}
     for k, v in results.items()}
).T
print("\n-- Comparison table --")
print(metrics_df.to_string(float_format="{:,.2f}".format))

# Save bar-chart comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
metrics_df[["MAE", "RMSE"]].plot(kind="bar", ax=axes[0], rot=25)
axes[0].set_title("MAE & RMSE by Model")
axes[0].set_ylabel("$")
axes[1].plot(metrics_df.index, metrics_df["R²"], marker="o", color="green")
axes[1].set_title("R² by Model")
axes[1].set_ylabel("R²")
axes[1].set_xticks(range(len(metrics_df)))
axes[1].set_xticklabels(metrics_df.index, rotation=25, ha="right")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/06_model_comparison.png", dpi=120)
plt.close()
print("\n  Saved plots/06_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 – Select best model & hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 10 – Best model selection & hyperparameter tuning")
print("=" * 65)

best_name = metrics_df["R²"].idxmax()
print(f"  Best model by R²: {best_name}  (R²={metrics_df.loc[best_name,'R²']:.4f})")

# Tune only tree-based ensembles (skip if Linear Regression wins)
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators":   [100, 200, 300],
        "learning_rate":  [0.05, 0.10, 0.20],
        "max_depth":      [3, 5, 7],
        "subsample":      [0.7, 0.85, 1.0],
        "min_samples_split": [2, 5],
    },
    "Decision Tree": {
        "max_depth":         [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
}

if best_name in PARAM_GRIDS:
    print(f"  Tuning {best_name} with RandomizedSearchCV (n_iter=20, cv=5) …")
    base_model = MODELS[best_name].__class__(random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base_model,
        PARAM_GRIDS[best_name],
        n_iter=20,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  CV R² (tuned): {search.best_score_:.4f}")
else:
    print(f"  {best_name} does not benefit from tree-specific tuning; using as-is.")
    final_model = results[best_name]["model"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 – Final evaluation on test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 11 – Final model evaluation on unseen test data")
print("=" * 65)

y_pred_final = final_model.predict(X_test)
final_mae    = mean_absolute_error(y_test, y_pred_final)
final_mse    = mean_squared_error(y_test, y_pred_final)
final_rmse   = np.sqrt(final_mse)
final_r2     = r2_score(y_test, y_pred_final)

print(f"  Final Model : {best_name} (tuned)")
print(f"  MAE         : ${final_mae:,.0f}")
print(f"  RMSE        : ${final_rmse:,.0f}")
print(f"  R²          : {final_r2:.4f}")

# Actual vs. predicted scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_final, alpha=0.4, s=10, color="steelblue")
lims = [min(y_test.min(), y_pred_final.min()),
        max(y_test.max(), y_pred_final.max())]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title("Actual vs. Predicted Price")
ax.legend()
plt.tight_layout()
plt.savefig("plots/07_actual_vs_predicted.png", dpi=120)
plt.close()
print("  Saved plots/07_actual_vs_predicted.png")

# Residuals
residuals = y_test - y_pred_final
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Residuals Distribution")
axes[0].set_xlabel("Residual ($)")
axes[1].scatter(y_pred_final, residuals, alpha=0.3, s=8, color="coral")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residuals vs. Predicted")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")
plt.tight_layout()
plt.savefig("plots/08_residuals.png", dpi=120)
plt.close()
print("  Saved plots/08_residuals.png")

# Feature importance (tree-based models expose it directly)
if hasattr(final_model, "feature_importances_"):
    imp = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances (Final Model)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/09_feature_importance.png", dpi=120)
    plt.close()
    print("  Saved plots/09_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 – Save artefacts
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 12 – Saving model artefacts to models/")
print("=" * 65)

joblib.dump(final_model, "models/final_model.pkl")
joblib.dump(enc,         "models/location_encoder.pkl")
joblib.dump(FEATURES,    "models/feature_names.pkl")

print("  Saved models/final_model.pkl")
print("  Saved models/location_encoder.pkl")
print("  Saved models/feature_names.pkl")

# -- Write a small results summary JSON for the report ------------------------
import json
summary = {
    "best_model":  best_name,
    "final_metrics": {
        "MAE":  round(final_mae, 2),
        "MSE":  round(final_mse, 2),
        "RMSE": round(final_rmse, 2),
        "R2":   round(final_r2,  4),
    },
    "comparison": metrics_df[["MAE","RMSE","R²"]].round(2).to_dict(orient="index"),
}
with open("models/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved models/results_summary.json")

print("\nPipeline complete.  Run  python predict.py  to get a prediction.\n")

