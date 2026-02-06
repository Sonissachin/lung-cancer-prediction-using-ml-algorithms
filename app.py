import streamlit as st
import pickle
import numpy as np
import os

# Load model safely
model_path = "lung_cancer_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Make sure 'lung_cancer_model.pkl' is in the same folder as this app.")
    st.stop()

with open(model_path, "rb") as file:
    loaded_obj = pickle.load(file)

# If the pickle file contains a tuple, extract the model
if isinstance(loaded_obj, tuple):
    model = loaded_obj[0]
else:
    model = loaded_obj

# Streamlit App UI
st.title("🫁 Lung Cancer Prediction App")

age = st.number_input("Enter Age", min_value=18, max_value=100)
smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
cough = st.selectbox("Do you have persistent coughing?", ["Yes", "No"])

# Helper function to convert Yes/No to 1/0
def yn(value):
    return 1 if value == "Yes" else 0

# Create input feature array
features = np.array([[age, yn(smoke), yn(alcohol), yn(cough)]])

# Prediction logic
if st.button("Predict"):
    try:
        pred = model.predict(features)
        if pred[0] == 1:
            st.error("⚠️ High Risk of Lung Cancer Detected")
        else:
            st.success("✅ Low Risk of Lung Cancer")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
