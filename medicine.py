import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Get absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "medicine_model.pkl")

# Load trained model safely
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Error: Model file not found at {model_path}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Streamlit UI
st.title("💊 Medicine Price Prediction App")
st.write("Enter the medicine details below to predict its price.")

# Sidebar information
st.sidebar.header("📌 About the App")
st.sidebar.write("This app predicts the price of a medicine using Machine Learning (Linear Regression).")

# Input fields
st.subheader("🔢 Enter Medicine Details")

ingredient_cost = st.number_input("Enter Ingredient Cost ($):", min_value=0.0, step=1.0)
demand = st.number_input("Enter Monthly Demand (Units):", min_value=0, step=10)
manufacturing_cost = st.number_input("Enter Manufacturing Cost ($):", min_value=0.0, step=1.0)

# Prediction button
if st.button("🔍 Predict Price"):
    input_data = np.array([[ingredient_cost, demand, manufacturing_cost]])

    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted Medicine Price: **${prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed by **[Your Name]** | Powered by Streamlit 🚀")
