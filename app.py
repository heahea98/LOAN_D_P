import streamlit as st
import numpy as np
import pickle

# Load the model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Predictor")

with st.form("input_form"):
    st.header("Client Information")

    contract_type = st.radio("Contract Type", ["Cash loans", "Revolving loans"])
    education = st.selectbox("Education Level", [
        "Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education"
    ])
    gender = st.radio("Gender", ["Male", "Female"])

    days_employed = st.number_input("Employment Duration (Years)", min_value=0, value=5)
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
    credit_amt = st.number_input("Credit Amount", min_value=0, value=200000)
    children = st.number_input("Number of Children", min_value=0, value=0)
    income = st.number_input("Total Income", min_value=0, value=8000)
    ctos_score = st.slider("CTOS Credit Score", 300, 850, 650)
    experian_score = st.slider("Experian Credit Score", 300, 850, 670)

    submit = st.form_submit_button("Predict")

if submit:
    contract_val = 1 if contract_type == "Cash loans" else 0
    gender_val = 1 if gender == "Male" else 0
    edu_map = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3
    }
    education_val = edu_map[education]

    input_data = np.array([[
        contract_val,
        education_val,
        gender_val,
        days_employed,
        age,
        credit_amt,
        children,
        income,
        ctos_score,
        experian_score
    ]])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("❌ Loan Default Prediction: **Defaulter**")
    else:
        st.success("✅ Loan Default Prediction: **Non-Defaulter**")# Your full Streamlit code here
