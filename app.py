import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor - 95% Accurate")
st.write(" Real Estate Price Estimator")

# Load models
model = joblib.load("models/final_model.pkl")
encoder = joblib.load("models/location_encoder.pkl")
features = joblib.load("models/feature_names.pkl")

# Inputs
location = st.selectbox("Location", ["Rural", "Suburb", "Midtown", "Uptown", "Downtown"])
area = st.number_input("Area sqft", 500, 10000, 1500)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
parking = st.slider("Parking Spaces", 0, 4, 1)
age = st.slider("House Age", 0, 50, 10)
floors = st.slider("Floors", 1, 3, 1)
garden = st.selectbox("Has Garden?", [0, 1])
pool = st.selectbox("Has Pool?", [0, 1])
renovated = st.selectbox("Ever Renovated?", [0, 1])

if st.button("Predict Price"):
    loc_enc = encoder.transform([[location]])[0][0]
    input_data = [[loc_enc, area, bedrooms, bathrooms, parking, age, floors, garden, pool, renovated]]
    df_input = pd.DataFrame(input_data, columns=features)
    price = model.predict(df_input)[0]
    st.success(f"💰 Estimated Price: ${price:,.0f}")
    st.balloons()