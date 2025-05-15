# groundwater_risk_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load saved models
kmeans = joblib.load('groundwater_kmeans_model.pkl')
scaler = joblib.load('groundwater_scaler.pkl')
pca = joblib.load('groundwater_pca.pkl')

# Risk mapping manually set (same as colab order)
risk_levels = ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk']
risk_mapping = {0: 'Very High Risk', 1: 'High Risk', 2: 'Medium Risk', 3: 'Low Risk'}

# Solutions per risk zone
def get_risk_zone_solutions(risk_level):
    solutions = {
        'Very High Risk': {
            'Water Conservation': [
                "Mandatory rainwater harvesting",
                "Strict water rationing",
                "High subsidies for efficient irrigation"
            ],
            'Infrastructure': [
                "Build multiple check dams",
                "Aggressive aquifer recharge programs"
            ],
            'Policy': [
                "Water extraction limits",
                "Heavy penalties for overuse"
            ]
        },
        'High Risk': {
            'Water Management': [
                "Rainwater harvesting incentives",
                "Smart irrigation promotion"
            ],
            'Infrastructure': [
                "Groundwater monitoring expansion",
                "Greywater recycling systems"
            ]
        },
        'Medium Risk': {
            'Prevention': [
                "Awareness drives on water conservation",
                "Minor infrastructure upgrades"
            ],
            'Planning': [
                "Future water demand modeling",
                "Monitoring recharge patterns"
            ]
        },
        'Low Risk': {
            'Maintenance': [
                "Regular checks on groundwater quality",
                "Protection of recharge areas"
            ],
            'Planning': [
                "Sustainability audits",
                "Long-term conservation planning"
            ]
        }
    }
    return solutions.get(risk_level, {})

# Streamlit app
st.set_page_config(page_title="Groundwater Risk Zone Prediction", layout="centered")

st.title("Groundwater Risk Zone Prediction App")
st.write("Provide input values to predict groundwater risk and get recommended solutions.")

# Input sliders
precip = st.slider('Total Precipitation (mm)', 0.0, 100.0, 20.0, step=0.1)
temp = st.slider('Average Temperature (°C)', 10.0, 40.0, 25.0, step=0.1)
humidity = st.slider('Humidity (%)', 10.0, 100.0, 60.0, step=0.1)

if st.button('Predict Risk Zone'):
    input_data = np.array([[precip, temp, humidity]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]  
    risk_level = risk_mapping.get(cluster, "Unknown")

    st.success(f"Predicted Groundwater Risk Level: **{risk_level}**")

    # Display solutions
    solutions = get_risk_zone_solutions(risk_level)

    st.subheader("Recommended Action Plan")
    for category, measures in solutions.items():
        st.markdown(f"### {category}")
        for i, measure in enumerate(measures, 1):
            st.markdown(f"- {measure}")

st.sidebar.title("ℹ️ About")
st.sidebar.write("This application predicts groundwater risk zones based on environmental parameters using Machine Learning techniques It leverages PCA (Principal Component Analysis) for dimensionality reduction and K-Means Clustering to classify regions into four risk levels: Very High Risk, High Risk, Medium Risk, and Low Risk.The app also recommends tailored water conservation strategies depending on the predicted risk zone.")
st.sidebar.write("Developed by Harsh, Purab")

