import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
model = pickle.load(open("medicine_model.pkl", "rb"))

# Streamlit UI
st.title("💊 Medicine Price Prediction App")
st.write("Enter the medicine details below to predict its price.")

# Sidebar for additional information
st.sidebar.header("📌 About the App")
st.sidebar.write("This app predicts the price of a medicine based on ingredient cost, demand, and manufacturing cost using a Machine Learning model (Linear Regression).")

# Input fields for prediction
st.subheader("🔢 Enter Medicine Details")

ingredient_cost = st.number_input("Enter Ingredient Cost ($):", min_value=0.0, step=1.0)
demand = st.number_input("Enter Monthly Demand (Units):", min_value=0, step=10)
manufacturing_cost = st.number_input("Enter Manufacturing Cost ($):", min_value=0.0, step=1.0)

# Prediction button
if st.button("🔍 Predict Price"):
    input_data = np.array([[ingredient_cost, demand, manufacturing_cost]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Medicine Price: **${prediction[0]:.2f}**")

# Footer
st.markdown("---")
st.write("Developed by [Your Name] | Powered by Streamlit 🚀")
