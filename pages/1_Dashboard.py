import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Loan Default Dashboard")
st.markdown("---")

TARGET_COL = "Default"

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "Loan_default.csv")
    return df

df = load_data()

# ── Dataset Overview ───────────────────────────────────────────
st.subheader("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows",      f"{df.shape[0]:,}")
col2.metric("Total Columns",   df.shape[1])
col3.metric("Missing Values",  int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows",  int(df.duplicated().sum()))

st.markdown("---")

# ── Preview ────────────────────────────────────────────────────
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("---")

# ── Column Info ────────────────────────────────────────────────
st.subheader("📌 Column Information")
info = pd.DataFrame({
    "Column":    df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing":   df.isnull().sum().values,
    "Unique":    df.nunique().values,
})
st.dataframe(info, use_container_width=True)
st.markdown("---")

# ── Target Distribution ────────────────────────────────────────
st.subheader("🎯 Loan Default Distribution")

col_a, col_b = st.columns([1, 2])

with col_a:
    counts = df[TARGET_COL].value_counts()
    default_rate = round(df[TARGET_COL].mean() * 100, 2)
    st.metric("Default Rate",    f"{default_rate}%")
    st.metric("Total Defaults",  f"{counts.get(1, 0):,}")
    st.metric("Total Safe",      f"{counts.get(0, 0):,}")

with col_b:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x=TARGET_COL, palette="Set2", ax=ax)
    ax.set_xticklabels(["Safe (0)", "Default (1)"])
    ax.set_xlabel("Loan Outcome")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Numerical Feature Distribution ────────────────────────────
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
for c in [TARGET_COL, "LoanID"]:
    if c in num_cols:
        num_cols.remove(c)

st.subheader("📈 Distribution of Numerical Features")
selected_num = st.selectbox("Select Numerical Feature", num_cols, key="num_select")

col_c, col_d = st.columns(2)

with col_c:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x=selected_num, kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribution of {selected_num}")
    st.pyplot(fig)
    plt.close()

with col_d:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=TARGET_COL, y=selected_num,
                palette="Set2", ax=ax)
    ax.set_xticklabels(["Safe", "Default"])
    ax.set_title(f"{selected_num} by Default Status")
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Categorical Features ───────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
if "LoanID" in cat_cols:
    cat_cols.remove("LoanID")

if cat_cols:
    st.subheader("🗂 Categorical Feature Analysis")
    selected_cat = st.selectbox("Select Categorical Feature", cat_cols, key="cat_select")

    fig, ax = plt.subplots(figsize=(8, 4))
    order = df[selected_cat].value_counts().index
    sns.countplot(data=df, x=selected_cat, hue=TARGET_COL,
                  order=order, palette="Set1", ax=ax)
    plt.xticks(rotation=30, ha="right")
    ax.set_title(f"{selected_cat} vs Default")
    ax.legend(title="Default", labels=["Safe", "Default"])
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Correlation Heatmap ────────────────────────────────────────
st.subheader("🔥 Correlation Heatmap")
corr_cols = num_cols + [TARGET_COL]
corr = df[corr_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Summary Statistics ─────────────────────────────────────────
st.subheader("📊 Summary Statistics")
st.dataframe(df[num_cols + [TARGET_COL]].describe().T.round(2),
             use_container_width=True)
st.markdown("---")

# ── Business Insights ──────────────────────────────────────────
st.subheader("💡 Business Insights")
st.success(f"Overall Loan Default Rate : {default_rate}%")
st.info("""
Key observations from the dataset:

• Customers with lower credit scores tend to have a higher default rate.

• Higher Debt-to-Income (DTI) ratios correlate with increased default risk.

• Employment type and loan purpose significantly influence repayment behaviour.

• Income and loan amount should be evaluated together rather than independently.

• Default is an imbalanced target — ROC-AUC and F1-score are better metrics than accuracy.
""")
