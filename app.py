import streamlit as st
import pandas as pd
import pickle
import os

model_path = "lung_cancer_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found! Make sure 'lung_cancer_model.pkl' is in the same folder.")
    st.stop()
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🫁 Lung Cancer Prediction", layout="centered")
st.title("🫁 Lung Cancer Prediction App")
st.markdown("Answer the following questions:")

def yes_no_to_numeric(ans):
    return 1 if ans == "Yes" else 2

feature_names = [
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN"
]

YELLOW_FINGERS = st.selectbox("Do you have yellow fingers?", ["Yes", "No"])
ANXIETY = st.selectbox("Do you feel anxious frequently?", ["Yes", "No"])
PEER_PRESSURE = st.selectbox("Do you face peer pressure?", ["Yes", "No"])
CHRONIC_DISEASE = st.selectbox("Do you have a chronic disease?", ["Yes", "No"])
FATIGUE = st.selectbox("Do you feel fatigue often?", ["Yes", "No"])
ALLERGY = st.selectbox("Do you have any allergies?", ["Yes", "No"])
WHEEZING = st.selectbox("Do you experience wheezing?", ["Yes", "No"])
ALCOHOL_CONSUMING = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
COUGHING = st.selectbox("Do you have persistent coughing?", ["Yes", "No"])
SWALLOWING_DIFFICULTY = st.selectbox("Do you have difficulty swallowing?", ["Yes", "No"])
CHEST_PAIN = st.selectbox("Do you have chest pain?", ["Yes", "No"])

features = pd.DataFrame([[
    yes_no_to_numeric(YELLOW_FINGERS),
    yes_no_to_numeric(ANXIETY),
    yes_no_to_numeric(PEER_PRESSURE),
    yes_no_to_numeric(CHRONIC_DISEASE),
    yes_no_to_numeric(FATIGUE),
    yes_no_to_numeric(ALLERGY),
    yes_no_to_numeric(WHEEZING),
    yes_no_to_numeric(ALCOHOL_CONSUMING),
    yes_no_to_numeric(COUGHING),
    yes_no_to_numeric(SWALLOWING_DIFFICULTY),
    yes_no_to_numeric(CHEST_PAIN)
]], columns=feature_names)

if st.button("Predict"):
    try:
        pred = model.predict(features)
        if pred[0] == 1:
            st.error("⚠️ High Risk of Lung Cancer Detected")
        else:
            st.success("✅ Low Risk of Lung Cancer")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

