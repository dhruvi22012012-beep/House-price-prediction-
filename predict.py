"""
predict.py
==========
Interactive House Price Prediction Interface
--------------------------------------------
Loads the saved trained model and lets a user enter house details
via a simple command-line menu, then prints the predicted price.

Run:  python predict.py
"""

import os
import sys
import joblib
import numpy as np

# ── Load saved artefacts ──────────────────────────────────────────────────────
MODELS_DIR = "models"
REQUIRED = ["final_model.pkl", "location_encoder.pkl", "feature_names.pkl"]

for f in REQUIRED:
    if not os.path.exists(os.path.join(MODELS_DIR, f)):
        print(f"[ERROR] '{f}' not found in {MODELS_DIR}/")
        print("  Please run  python house_price_ml.py  first.")
        sys.exit(1)

model    = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
encoder  = joblib.load(os.path.join(MODELS_DIR, "location_encoder.pkl"))
features = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

LOCATIONS = ["Rural", "Suburb", "Midtown", "Uptown", "Downtown"]

# ── Helper functions ──────────────────────────────────────────────────────────

def get_int(prompt, lo=None, hi=None):
    """Prompt for an integer, optionally clamped to [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None             # allow blank → use default
        try:
            val = int(raw)
            if lo is not None and val < lo:
                print(f"  ✗ Value must be ≥ {lo}")
                continue
            if hi is not None and val > hi:
                print(f"  ✗ Value must be ≤ {hi}")
                continue
            return val
        except ValueError:
            print("  ✗ Please enter a whole number.")


def get_yes_no(prompt):
    """Prompt for a yes/no answer; returns 1 or 0."""
    while True:
        raw = input(prompt + " [y/n]: ").strip().lower()
        if raw in ("y", "yes", "1"):
            return 1
        if raw in ("n", "no", "0"):
            return 0
        print("  ✗ Please enter y or n.")


def choose_location():
    """Display a numbered menu and return the chosen location string."""
    print("\n  Locations:")
    for i, loc in enumerate(LOCATIONS, 1):
        print(f"    {i}. {loc}")
    while True:
        raw = input("  Enter number (1–5): ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(LOCATIONS):
                return LOCATIONS[choice - 1]
        except ValueError:
            pass
        print("  ✗ Enter a number between 1 and 5.")


def encode_location(loc_str):
    """Return the ordinal integer for a location string."""
    return int(encoder.transform([[loc_str]])[0][0])


def predict_price(house: dict) -> float:
    """Given a dict of raw house features, return the predicted price."""
    loc_enc = encode_location(house["location"])
    row = np.array([[
        loc_enc,
        house["area_sqft"],
        house["bedrooms"],
        house["bathrooms"],
        house["parking_spaces"],
        house["house_age"],
        house["floors"],
        house["has_garden"],
        house["has_pool"],
        house["ever_renovated"],
    ]])
    return float(model.predict(row)[0])


# ── Main interactive loop ─────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  House Price Prediction -- Interactive Interface")
    print("=" * 55)
    print("  Trained model loaded successfully.\n")

    while True:
        print("\n──────────────────────────────────────────────────────")
        print("  Enter house details below (press Ctrl+C to quit)\n")

        # Location
        print("  Location of the property:")
        location = choose_location()

        # Numeric inputs
        area   = get_int("  Area in sq ft (e.g. 1500): ", lo=100, hi=50000)
        beds   = get_int("  Number of bedrooms (1–6): ", lo=1, hi=6)
        baths  = get_int("  Number of bathrooms (1–6): ", lo=1, hi=6)
        parks  = get_int("  Number of parking spaces (0–3): ", lo=0, hi=3)
        age    = get_int("  Age of the house in years (0–50): ", lo=0, hi=50)
        floors = get_int("  Number of floors (1–3): ", lo=1, hi=3)

        # Boolean inputs
        garden   = get_yes_no("  Does the house have a garden?")
        pool     = get_yes_no("  Does the house have a swimming pool?")
        renovated = get_yes_no("  Has the house ever been renovated?")

        house = {
            "location":       location,
            "area_sqft":      area,
            "bedrooms":       beds,
            "bathrooms":      baths,
            "parking_spaces": parks,
            "house_age":      age,
            "floors":         floors,
            "has_garden":     garden,
            "has_pool":       pool,
            "ever_renovated": renovated,
        }

        # Display summary
        print("\n  ── House Details Summary ──")
        for k, v in house.items():
            label = k.replace("_", " ").title()
            print(f"    {label:<22}: {v}")

        # Predict
        price = predict_price(house)

        print("\n" + "+" + "-" * 47 + "+")
        print(f"  Predicted House Price : ${price:>14,.0f}")
        print("+" + "-" * 47 + "+")

        # Ask to predict another
        again = input("\n  Predict another house? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Thank you for using the House Price Predictor. Goodbye!\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")
