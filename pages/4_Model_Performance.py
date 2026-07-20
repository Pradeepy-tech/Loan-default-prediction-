import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Model Performance")
st.markdown("---")

# ── Load model + preprocessor + comparison CSV ─────────────────
@st.cache_resource
def load_model_and_preprocessor():
    model        = joblib.load(ROOT / "models/best_model.pkl")
    preprocessor = joblib.load(ROOT / "models/preprocessor.pkl")
    return model, preprocessor

@st.cache_data
def load_comparison():
    return pd.read_csv(ROOT / "models/model_comparison.csv")

# ── Rebuild X_test / y_test from CSV (avoids 89 MB processed_data.pkl) ────────
@st.cache_data
def get_test_data():
    df = pd.read_csv(ROOT / "Loan_default.csv")
    TARGET = "Default"
    df = df.drop(columns=["LoanID"], errors="ignore")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

try:
    model, preprocessor = load_model_and_preprocessor()
    comparison          = load_comparison()
    X_test_raw, y_test  = get_test_data()
    with st.spinner("Preparing test predictions…"):
        X_test = preprocessor.transform(X_test_raw)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
except Exception as _load_err:
    st.error(f"❌ Could not load model files. Please run `retrain_all.py` first.\n\n{_load_err}")
    st.stop()

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc       = roc_auc_score(y_test, y_prob)

# ── Performance Metrics ────────────────────────────────────────
st.subheader("🏆 Performance Metrics (Best Model)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{accuracy:.3f}")
c2.metric("Precision", f"{precision:.3f}")
c3.metric("Recall",    f"{recall:.3f}")
c4.metric("F1 Score",  f"{f1:.3f}")
c5.metric("ROC AUC",   f"{roc:.3f}")
st.markdown("---")

# ── Model Comparison ───────────────────────────────────────────
st.subheader("📊 Model Comparison Table")
comparison_sorted = comparison.sort_values(by="ROC AUC", ascending=False)
st.dataframe(
    comparison_sorted.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        color="#d4f0d4"
    ),
    use_container_width=True
)

st.subheader("📊 ROC-AUC Comparison Chart")
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(comparison_sorted))]
bars = ax.barh(comparison_sorted["Model"], comparison_sorted["ROC AUC"], color=colors)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("ROC AUC Score")
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.invert_yaxis()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Confusion Matrix ───────────────────────────────────────────
st.subheader("🔲 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Safe", "Default"],
        yticklabels=["Safe", "Default"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close()

with col_b:
    tn, fp, fn, tp = cm.ravel()
    st.markdown("**Confusion Matrix Breakdown**")
    st.write(f"- ✅ **True Negatives  (TN):** {tn:,}  — Correctly predicted Safe")
    st.write(f"- ❌ **False Positives (FP):** {fp:,}  — Safe predicted as Default")
    st.write(f"- ⚠️ **False Negatives (FN):** {fn:,}  — Default predicted as Safe")
    st.write(f"- ✅ **True Positives  (TP):** {tp:,}  — Correctly predicted Default")

st.markdown("---")

# ── ROC Curve ─────────────────────────────────────────────────
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
col_c, col_d = st.columns(2)

with col_c:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", linewidth=2, label=f"AUC = {roc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with col_d:
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_curve, prec_curve, color="#9b59b6", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Prediction Distribution ────────────────────────────────────
st.subheader("🍩 Prediction Distribution")
pred_labels = pd.Series(y_pred).map({0: "Safe", 1: "Default"})
col_e, col_f = st.columns(2)

with col_e:
    fig, ax = plt.subplots(figsize=(5, 4))
    pred_labels.value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax,
        colors=["#2ecc71", "#e74c3c"]
    )
    ax.set_ylabel("")
    ax.set_title("Predicted Classes")
    st.pyplot(fig)
    plt.close()

with col_f:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=40, color="#3498db", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", label="Threshold = 0.5")
    ax.set_xlabel("Default Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability Distribution")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Evaluation Summary Table ───────────────────────────────────
st.subheader("📋 Evaluation Summary")
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Value":  [round(accuracy, 4), round(precision, 4),
               round(recall, 4),   round(f1, 4), round(roc, 4)]
})
st.table(metrics_df)

st.markdown("---")
st.subheader("💡 Business Interpretation")
st.success("""
• **High Accuracy** → model correctly classifies most customers.

• **High Precision** → fewer false approvals for risky customers (lower financial loss).

• **High Recall** → most default customers are correctly identified (fewer missed risks).

• **ROC-AUC** → measures the model's ability to distinguish defaulters from safe customers. Score > 0.75 is good.

• **Confusion Matrix** → helps tune the decision threshold based on business tolerance for risk.
""")
