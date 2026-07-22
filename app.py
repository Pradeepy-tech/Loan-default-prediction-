import streamlit as st

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🏦 Navigation")
st.sidebar.success("Select a page above")
st.sidebar.info("""
**Loan Default Prediction**

Machine Learning Project

Developer: Pradeep
""")

st.title("🏦 Loan Default Prediction System")

st.markdown("---")

st.header("📌 Project Overview")

st.write("""
This application predicts whether a customer is likely to default on a loan.

The prediction is based on customer financial and personal information using a Machine Learning model.

### Features

- Single Customer Prediction
- Batch Prediction using CSV
- Model Performance
- Feature Importance
- Dashboard
- Download Prediction Results
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "255,347 Rows")
col2.metric("Features", "16")
col3.metric("Algorithms", "9")

st.markdown("---")

st.subheader("Workflow")

st.write("""
Customer Data

⬇

Preprocessing

⬇

Machine Learning Model

⬇

Prediction

⬇

Risk Score

⬇

Business Decision
""")

st.success("Use the sidebar to navigate between pages.")
