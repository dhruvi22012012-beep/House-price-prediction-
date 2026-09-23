# House Price Prediction — Machine Learning Project

A complete, beginner-friendly end-to-end ML pipeline that predicts residential
house prices from property features such as location, area, number of rooms,
age, and amenities.

---

## Project Structure

```
house_price_prediction/
│
├── generate_dataset.py      # Generates the synthetic dataset (run once)
├── train.py                 # Standalone model training script
├── house_price_ml.py        # Full pipeline: EDA + training + evaluation
├── predict.py               # Interactive CLI prediction interface
├── requirements.txt         # Python dependencies
│
├── data/
│   └── house_prices.csv     # Generated dataset (2 000 rows)
│
├── models/
│   ├── final_model.pkl      # Serialised trained model
│   ├── location_encoder.pkl # Fitted OrdinalEncoder
│   ├── feature_names.pkl    # Ordered feature name list
│   └── results_summary.json # Model metrics summary
│
└── plots/
    ├── 01_price_distribution.png
    ├── 02_price_by_location.png
    ├── 03_correlation_heatmap.png
    ├── 04_price_vs_area.png
    ├── 05_price_vs_age.png
    ├── 06_model_comparison.png
    ├── 07_actual_vs_predicted.png
    ├── 08_residuals.png
    └── 09_feature_importance.png
```

---

## Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd house_price_prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python generate_dataset.py
```

> This creates `data/house_prices.csv` with 2 000 rows of realistic synthetic
> house data including intentional noise, missing values, and duplicates.

### 5a. Run the full pipeline (EDA + all models + tuning + plots)

```bash
python house_price_ml.py
```

### 5b. Run only the training script (trains & saves the best model)

```bash
python train.py

# Optional flags:
python train.py --data data/house_prices.csv --output models/
```

### 6. Predict a house price interactively

```bash
python predict.py
```

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| `location` | Categorical | Downtown / Uptown / Midtown / Suburb / Rural |
| `area_sqft` | Numeric | Liveable area in square feet |
| `bedrooms` | Integer | Number of bedrooms |
| `bathrooms` | Integer | Number of bathrooms |
| `parking_spaces` | Integer | Number of parking spaces |
| `house_age` | Integer | Age of the property in years |
| `floors` | Integer | Number of floors |
| `has_garden` | Binary | Garden present (0 / 1) |
| `has_pool` | Binary | Swimming pool present (0 / 1) |
| `ever_renovated` | Binary | Renovated at least once (0 / 1) |

---

## Models Trained

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 47,170 | 60,480 | 0.9189 |
| Decision Tree | 64,398 | 83,878 | 0.8441 |
| Random Forest | 39,580 | 52,408 | 0.9391 |
| **Gradient Boosting** ★ | **32,929** | **46,119** | **0.9529** |

★ Best model — selected and tuned with `RandomizedSearchCV`.

---

## Pipeline Overview

```
generate_dataset.py
        |
        v
data/house_prices.csv
        |
        v
house_price_ml.py / train.py
  |-- Step 1: Load dataset
  |-- Step 2: Analyze (dtypes, missing values, stats)
  |-- Step 3: Clean (duplicates, invalid values, imputation)
  |-- Step 4: EDA (distribution, correlation, scatter plots)
  |-- Step 5: Encode (OrdinalEncoder for location)
  |-- Step 6: Feature selection (10 features)
  |-- Step 7: Train/test split (80/20, stratified)
  |-- Step 8: Train 4 models
  |-- Step 9: Compare (MAE, MSE, RMSE, R2)
  |-- Step 10: Hyperparameter tuning (RandomizedSearchCV)
  |-- Step 11: Final evaluation on test set
  |-- Step 12: Save model artefacts
        |
        v
predict.py  <-- loads models/ and accepts user input
```

---

## Data Preprocessing

- **Duplicate removal** — 15 exact duplicate rows dropped
- **Invalid value correction** — negative area/age values replaced with `NaN`
- **Median imputation** — `area_sqft`, `bedrooms`, `bathrooms`, `house_age`
- **Renovation year** — converted to binary `ever_renovated` flag (0/1)
- **No data leakage** — all transformations fitted on training data only

---

## Requirements

- Python 3.9 or higher
- See [`requirements.txt`](requirements.txt) for package versions

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---

## Reproducibility

All random operations use `SEED = 42`. The pipeline produces identical results
on every run given the same dataset.

---

## Example Prediction

```
House Price Prediction -- Interactive Interface

  Location of the property:
    1. Rural   2. Suburb   3. Midtown   4. Uptown   5. Downtown
  Enter number (1-5): 5
  Area in sq ft: 2200
  Number of bedrooms: 3
  Number of bathrooms: 2
  Parking spaces: 1
  House age (years): 10
  Number of floors: 2
  Has garden? [y/n]: y
  Has pool? [y/n]: n
  Ever renovated? [y/n]: y

  +-----------------------------------------------+
    Predicted House Price :        $548,400
  +-----------------------------------------------+
```

---

## License

This project is released for educational purposes. Feel free to use, modify, and
extend it for your own datasets.
