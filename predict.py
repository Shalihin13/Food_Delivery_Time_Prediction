import requests
import joblib
import os
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numerize import numerize



# Load model
model = joblib.load("food_delivery_prediction.joblib")

# Judul aplikasi
st.title(" ⏰ Smart Prediction for Food Delivery Time")

st.markdown("Masukkan informasi berikut untuk memprediksi estimasi waktu pengiriman :")

# Input fitur dari user
Distance_km = st.number_input("Jarak Pengiriman (KM)", min_value=0.0, format="%.2f")
Preparation_Time_min = st.number_input("Waktu Persiapan", min_value=0, max_value=30)
Courier_Experience_yrs= st.number_input("Pengalaman Kurir", min_value=0, max_value=9)
Traffic_Level = st.select_slider("Tingkat Kemacetan", options=["Low", "Medium", "High"])
Time_of_Day = st.select_slider("Waktu Pemesanan", options=["Morning", "Afternoon", "Evening", "Night"])
Weather = st.selectbox("Kondisi Cuaca", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
Vehicle_Type = st.selectbox("Kendaraan", ["Bike", "Scooter", "Car"])

# Encode input
def encode_input():
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    
    # One-Hot Encoding (OHE)
    weather_options = ["Clear", "Rainy", "Foggy", "Snowy", "Windy"]
    vehicle_options = ["Bike", "Scooter", "Car"]

    weather_ohe = [1 if Weather == w else 0 for w in weather_options]
    vehicle_ohe = [1 if Vehicle_Type == v else 0 for v in vehicle_options]
    
    features = [
        Distance_km,
        Preparation_Time_min,
        Courier_Experience_yrs,
        traffic_map[Traffic_Level],
        time_map[Time_of_Day],] + weather_ohe + vehicle_ohe
    
    return np.array([features])

# Submit Button
if st.button("🔍Predict Time"):
    input_data = encode_input()
    prediction = model.predict(input_data)[0]
    st.success(f"🎯Estimasi Waktu Pengiriman: **{round(prediction)} menit**")
    
    