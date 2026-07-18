import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("---")

st.header("🏦 Loan Default Prediction System")

st.write("""
The Loan Default Prediction System is a Machine Learning application that predicts
whether a customer is likely to default on a loan.

The project helps banks and financial institutions make better lending decisions
by identifying high-risk customers before loan approval.
""")

st.markdown("---")

st.header("🎯 Project Objectives")

st.markdown("""
- Predict loan default risk
- Reduce financial losses
- Support better lending decisions
- Improve customer risk assessment
- Provide explainable AI insights
""")

st.markdown("---")

st.header("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Dataset Size", "255,347 Records")
    st.metric("Features", "16")
    st.metric("Target", "Default")

with col2:
    st.metric("Missing Values", "0")
    st.metric("Duplicate Rows", "0")
    st.metric("Classes", "2")

st.markdown("---")

st.header("⚙ Machine Learning Workflow")

st.markdown("""
1. Data Collection

⬇

2. Data Cleaning

⬇

3. Exploratory Data Analysis

⬇

4. Feature Engineering

⬇

5. Data Preprocessing

⬇

6. Train-Test Split

⬇

7. SMOTE

⬇

8. Model Training

⬇

9. Hyperparameter Tuning

⬇

10. Model Evaluation

⬇

11. Explainable AI (SHAP)

⬇

12. Streamlit Deployment
""")

st.markdown("---")

st.header("🤖 Algorithms Used")

algorithms = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]

for algo in algorithms:
    st.write(f"✅ {algo}")

st.markdown("---")

st.header("🛠 Technologies")

tech = [
    "Python",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Scikit-Learn",
    "Imbalanced-Learn",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "SHAP",
    "Joblib",
    "Streamlit"
]

cols = st.columns(3)

for i, item in enumerate(tech):
    cols[i % 3].success(item)

st.markdown("---")

st.header("✨ Application Features")

st.markdown("""
✅ Dashboard

✅ Single Prediction

✅ Batch Prediction

✅ Download Prediction Results

✅ Model Performance

✅ Feature Importance

✅ SHAP Explainability

✅ Business Insights

✅ Interactive Charts

✅ Professional UI
""")

st.markdown("---")

st.header("💼 Business Benefits")

st.info("""
✔ Detect high-risk customers

✔ Reduce loan defaults

✔ Improve credit risk management

✔ Faster loan approval

✔ Data-driven decision making

✔ Explainable AI
""")

st.markdown("---")

st.header("🚀 Future Improvements")

st.write("""
• Deep Learning Models

• Real-time Prediction API

• Cloud Deployment (AWS/Azure/GCP)

• Database Integration

• Customer Authentication

• Live Dashboard

• Auto Model Retraining

• Fraud Detection Module
""")

st.markdown("---")

st.header("👨‍💻 Developer")

st.success("""
Name: Pradeep

Role: Data Science & Machine Learning Enthusiast

Project: Loan Default Prediction

Tools: Python | Scikit-Learn | XGBoost | Streamlit
""")

st.markdown("---")

st.header("📬 Contact")

github = st.text_input(
    "GitHub Profile",
    "https://github.com/yourusername"
)

linkedin = st.text_input(
    "LinkedIn Profile",
    "https://linkedin.com/in/yourprofile"
)

email = st.text_input(
    "Email",
    "your_email@gmail.com"
)

st.markdown("---")

st.markdown(
    """
---
<center>

### 🏦 Loan Default Prediction System

Developed using

Python • Scikit-Learn • XGBoost • Streamlit

</center>
""",
    unsafe_allow_html=True
)
