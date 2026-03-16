import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("lightgbm_model.pkl")

st.title("🧠 Alzheimer Risk Prediction System")
st.write("AI-based early Alzheimer risk screening")

# User input sliders
Age = st.slider("Age", 40, 100, 65)
MMSE = st.slider("MMSE Score", 0.0, 1.0, 0.5)
ADL = st.slider("ADL Score", 0.0, 1.0, 0.5)
FunctionalAssessment = st.slider("Functional Assessment", 0.0, 1.0, 0.5)
MemoryComplaints = st.slider("Memory Complaints", 0.0, 1.0, 0.5)
BehavioralProblems = st.slider("Behavioral Problems", 0.0, 1.0, 0.5)

# Create dataframe
input_data = pd.DataFrame([[Age,MMSE,ADL,FunctionalAssessment,
                            MemoryComplaints,BehavioralProblems]],
                          columns=["Age","MMSE","ADL",
                                   "FunctionalAssessment",
                                   "MemoryComplaints",
                                   "BehavioralProblems"])

# Predict
if st.button("Predict Alzheimer Risk"):

    prob = model.predict_proba(input_data)[0][1]
    risk = prob * 100

    st.subheader(f"Risk Score: {risk:.2f}%")

    if prob >= 0.42:
        st.error("🚨 High Alzheimer Risk - Clinical evaluation recommended")
    else:
        st.success("✅ Low Risk")
