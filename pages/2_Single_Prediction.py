import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Single Prediction",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single Customer Loan Default Prediction")
st.markdown("---")

@st.cache_resource
def load_model():
    model        = joblib.load(ROOT / "models/best_model.pkl")
    preprocessor = joblib.load(ROOT / "models/preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_model()
    _model_loaded = True
except Exception as _load_err:
    st.error(f"❌ Could not load model files. Please run `retrain_all.py` first.\n\n{_load_err}")
    _model_loaded = False
    st.stop()

st.subheader("📝 Enter Customer Details")

# ── Form (only inputs + submit button) ────────────────────────────────────────
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Info**")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

        education = st.selectbox(
            "Education",
            ["High School", "Bachelor's", "Master's", "PhD"]
        )

        marital = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        employment = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"]
        )

        months_employed = st.number_input(
            "Months Employed", min_value=0, max_value=600, value=60
        )

    with col2:
        st.markdown("**Financial Info**")
        income = st.number_input(
            "Annual Income (₹)", min_value=0, value=60000, step=1000
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)", min_value=0, value=20000, step=500
        )

        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650
        )

        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, max_value=30.0,
            value=10.5, step=0.1
        )

        dti_ratio = st.slider(
            "Debt-to-Income Ratio", 0.0, 1.0, 0.30, step=0.01
        )

    with col3:
        st.markdown("**Loan Details**")
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

        num_credit_lines = st.number_input(
            "Number of Credit Lines", min_value=0, max_value=20, value=3
        )

        mortgage  = st.selectbox("Has Mortgage?",   ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        purpose = st.selectbox(
            "Loan Purpose",
            ["Home", "Auto", "Business", "Education", "Other"]
        )

        cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

    predict_button = st.form_submit_button("🔮 Predict Loan Risk", use_container_width=True)

# ── Results section (OUTSIDE the form) ────────────────────────────────────────
if predict_button:

    input_df = pd.DataFrame([{
        "Age":             age,
        "Income":          income,
        "LoanAmount":      loan_amount,
        "CreditScore":     credit_score,
        "MonthsEmployed":  months_employed,
        "NumCreditLines":  num_credit_lines,
        "InterestRate":    interest_rate,
        "LoanTerm":        loan_term,
        "DTIRatio":        dti_ratio,
        "Education":       education,
        "EmploymentType":  employment,
        "MaritalStatus":   marital,
        "HasMortgage":     mortgage,
        "HasDependents":   dependents,
        "LoanPurpose":     purpose,
        "HasCoSigner":     cosigner,
    }])

    try:
        processed   = preprocessor.transform(input_df)
        prediction  = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{probability * 100:.2f}%")
        c2.metric("Prediction", "⚠️ Default" if prediction == 1 else "✅ Safe")
        confidence = max(probability, 1 - probability)
        c3.metric("Confidence", f"{confidence * 100:.2f}%")

        st.progress(float(probability))
        st.markdown("---")

        if probability >= 0.70:
            st.error("🔴 HIGH RISK — Recommend: ❌ Reject Loan")
            st.write("Customer has a very high probability of default.")
        elif probability >= 0.40:
            st.warning("🟠 MEDIUM RISK — Recommend: ⚠ Manual Review Required")
            st.write("Additional verification is recommended before approval.")
        else:
            st.success("🟢 LOW RISK — Recommend: ✅ Loan Can Be Approved")
            st.write("Customer has a low probability of default.")

        st.markdown("---")
        st.subheader("📋 Customer Summary")

        display_df = input_df.copy()
        display_df["Prediction"]              = "Default" if prediction == 1 else "Safe"
        display_df["Default Probability (%)"] = round(probability * 100, 2)
        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇ Download Result",
            csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Prediction failed. Please check inputs.")
        st.exception(e)
