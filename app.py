import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load model
model = joblib.load("lightgbm_model.pkl")

# Load dataset just to fit scaler
data = pd.read_csv("alzheimers_disease_data.csv")

data = data.drop(columns=["PatientID","DoctorInCharge"], errors="ignore")

X = data.drop("Diagnosis", axis=1)

scaler = MinMaxScaler()
scaler.fit(X)

st.title("🧠 Alzheimer Risk Prediction System")

# User input
Age = st.slider("Age", 40, 100, 65)
MMSE = st.slider("MMSE Score", 0.0, 30.0, 20.0)
ADL = st.slider("ADL Score", 0.0, 10.0, 5.0)
FunctionalAssessment = st.slider("Functional Assessment", 0.0, 10.0, 5.0)
MemoryComplaints = st.slider("Memory Complaints", 0.0, 1.0, 0.5)
BehavioralProblems = st.slider("Behavioral Problems", 0.0, 1.0, 0.5)

# Create dataframe
input_data = pd.DataFrame([[Age,MMSE,ADL,FunctionalAssessment,
                            MemoryComplaints,BehavioralProblems]],
                          columns=[
                              "Age","MMSE","ADL",
                              "FunctionalAssessment",
                              "MemoryComplaints",
                              "BehavioralProblems"
                          ])

# SCALE input
input_scaled = scaler.transform(input_data)

if st.button("Predict Risk"):

    prob = model.predict_proba(input_scaled)[0][1]

    risk = prob*100

    st.write(f"### Alzheimer Risk: {risk:.2f}%")

    if prob >= 0.42:
        st.error("🚨 High Alzheimer Risk")
    else:
        st.success("✅ Low Risk")
