import streamlit as st
import numpy as np
import joblib

st.title("🧠 Alzheimer Early Risk Screening System")
st.write("Answer the following questions for cognitive assessment")

# ============================
# MMSE QUESTIONS
# ============================

st.header("MMSE Cognitive Test")

q1 = st.radio("1. What year is it?", ["Correct", "Wrong"])
q2 = st.radio("2. What month is it?", ["Correct", "Wrong"])
q3 = st.radio("3. What city are we in?", ["Correct", "Wrong"])
q4 = st.radio("4. Repeat these words: Apple, Table, Penny", ["Correct", "Wrong"])
q5 = st.radio("5. Spell WORLD backwards correctly", ["Correct", "Wrong"])
q6 = st.radio("6. Recall the words Apple, Table, Penny after few minutes", ["Correct", "Wrong"])

mmse_score = 0

for q in [q1,q2,q3,q4,q5,q6]:
    if q == "Correct":
        mmse_score += 5

mmse_score = min(mmse_score,30)

st.write("MMSE Score:", mmse_score)

# ============================
# ADL QUESTIONS
# ============================

st.header("ADL Daily Activity Test")

adl1 = st.radio("Can the patient bathe independently?", ["Yes","No"])
adl2 = st.radio("Can the patient dress independently?", ["Yes","No"])
adl3 = st.radio("Can the patient eat independently?", ["Yes","No"])
adl4 = st.radio("Can the patient use toilet independently?", ["Yes","No"])
adl5 = st.radio("Can the patient walk independently?", ["Yes","No"])

adl_score = 0

for a in [adl1,adl2,adl3,adl4,adl5]:
    if a == "Yes":
        adl_score += 1

st.write("ADL Score:", adl_score)

# ============================
# FUNCTIONAL ASSESSMENT
# ============================

st.header("Functional Activity Test")

f1 = st.slider("Can manage finances",0,3)
f2 = st.slider("Can prepare meals",0,3)
f3 = st.slider("Can remember appointments",0,3)
f4 = st.slider("Can take medication correctly",0,3)

functional_score = f1 + f2 + f3 + f4

st.write("Functional Assessment Score:", functional_score)

# ============================
# OTHER FEATURES
# ============================

st.header("Patient Information")

age = st.slider("Age",40,100)

memory_complaints = st.selectbox("Memory complaints", [0,1])
behavioral_problems = st.selectbox("Behavioral problems", [0,1])

# ============================
# LOAD MODEL
# ============================
model = joblib.load("alzheimers_questionnaire_model.pkl")

# ============================
# PREDICTION
# ============================

if st.button("Predict Alzheimer Risk"):

    input_data = np.array([[

        age,
        mmse_score,
        adl_score,
        functional_score,
        memory_complaints,
        behavioral_problems

    ]])

    prob = model.predict_proba(input_data)[0][1]

    risk = prob * 100

    st.subheader("Prediction Result")

    st.write("Alzheimer Risk:", round(risk,2), "%")

    if prob > 0.5:
        st.error("⚠ High Risk - Clinical evaluation recommended")
    else:
        st.success("✅ Low Risk")
