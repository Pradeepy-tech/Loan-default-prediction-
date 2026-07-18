import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Batch Loan Default Prediction")

st.markdown("""
Upload a CSV file with customer data.
The model will predict loan default risk for **every customer** in the file.

**Required columns:** `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`,
`NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`, `Education`,
`EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`
""")

st.divider()

REQUIRED_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

@st.cache_resource
def load_models():
    model        = joblib.load(ROOT / "models/best_model.pkl")
    preprocessor = joblib.load(ROOT / "models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_models()

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Check missing columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        if st.button("🔮 Predict All Customers", use_container_width=True):

            try:
                feature_df  = df[REQUIRED_COLS].copy()
                processed   = preprocessor.transform(feature_df)
                predictions = model.predict(processed)
                probabilities = model.predict_proba(processed)[:, 1]

                result_df = df.copy()
                result_df["Prediction"]              = predictions
                result_df["Prediction"]              = result_df["Prediction"].map({0: "Safe", 1: "Default"})
                result_df["Default Probability (%)"] = (probabilities * 100).round(2)
                result_df["Risk Level"]              = pd.cut(
                    probabilities * 100,
                    bins=[0, 40, 70, 100],
                    labels=["🟢 Low", "🟠 Medium", "🔴 High"]
                )

                st.success("✅ Prediction Completed Successfully!")

                st.subheader("📊 Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                st.divider()

                # Summary metrics
                st.subheader("📈 Prediction Summary")
                total   = len(result_df)
                default = (result_df["Prediction"] == "Default").sum()
                safe    = (result_df["Prediction"] == "Safe").sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers",  f"{total:,}")
                c2.metric("✅ Safe",           f"{safe:,}")
                c3.metric("⚠️ Default",        f"{default:,}")
                c4.metric("Default Rate",      f"{default/total*100:.1f}%")

                st.divider()

                # Charts
                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("Prediction Distribution")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    counts = result_df["Prediction"].value_counts()
                    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                           startangle=90, colors=["#2ecc71", "#e74c3c"])
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close()

                with col_b:
                    st.subheader("Default Probability Distribution")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.hist(result_df["Default Probability (%)"], bins=20,
                            color="#3498db", edgecolor="white")
                    ax.set_xlabel("Default Probability (%)")
                    ax.set_ylabel("Number of Customers")
                    st.pyplot(fig)
                    plt.close()

                st.divider()

                # High risk table
                st.subheader("🔴 High Risk Customers (Probability ≥ 70%)")
                high_risk = result_df[result_df["Default Probability (%)"] >= 70]
                if len(high_risk) > 0:
                    st.dataframe(high_risk, use_container_width=True)
                    st.warning(f"Found {len(high_risk):,} high-risk customers.")
                else:
                    st.success("No high-risk customers found.")

                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Prediction File",
                    data=csv,
                    file_name="loan_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("❌ Prediction Failed. See error below.")
                st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin batch prediction.")

    # Show sample format
    with st.expander("📋 View Expected CSV Format"):
        sample = pd.DataFrame([{
            "Age": 35, "Income": 60000, "LoanAmount": 20000,
            "CreditScore": 650, "MonthsEmployed": 60, "NumCreditLines": 3,
            "InterestRate": 10.5, "LoanTerm": 36, "DTIRatio": 0.3,
            "Education": "Bachelor's", "EmploymentType": "Full-time",
            "MaritalStatus": "Married", "HasMortgage": "Yes",
            "HasDependents": "No", "LoanPurpose": "Home", "HasCoSigner": "No"
        }])
        st.dataframe(sample, use_container_width=True)
