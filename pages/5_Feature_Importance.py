import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Feature Importance",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ Feature Importance & Explainability")
st.markdown("---")

@st.cache_resource
def load_resources():
    model        = joblib.load(ROOT / "models/best_model.pkl")
    preprocessor = joblib.load(ROOT / "models/preprocessor.pkl")
    processed    = joblib.load(ROOT / "models/processed_data.pkl")
    return model, preprocessor, processed

model, preprocessor, processed = load_resources()

X_test = processed["X_test"]
y_test = processed["y_test"]

# Get feature names from the preprocessor
try:
    feature_names = preprocessor.get_feature_names_out().tolist()
    # Clean up prefix (num__, cat__)
    feature_names_clean = [
        n.replace("num__", "").replace("cat__", "") for n in feature_names
    ]
except Exception:
    feature_names_clean = [f"Feature_{i}" for i in range(X_test.shape[1])]

# ── Build importance DataFrame ─────────────────────────────────
if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature":    feature_names_clean,
        "Importance": model.feature_importances_
    })
elif hasattr(model, "coef_"):
    importance = pd.DataFrame({
        "Feature":    feature_names_clean,
        "Importance": np.abs(model.coef_[0])
    })
else:
    from sklearn.inspection import permutation_importance
    with st.spinner("Computing permutation importance…"):
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=5, random_state=42, n_jobs=-1
        )
    importance = pd.DataFrame({
        "Feature":    feature_names_clean,
        "Importance": perm.importances_mean
    })

importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)

# ── Feature importance table ───────────────────────────────────
st.subheader("📋 All Features Ranked by Importance")
st.dataframe(importance, use_container_width=True)
st.markdown("---")

# ── Top N bar chart ────────────────────────────────────────────
n_top = st.slider("Select number of top features to display", 5, len(importance), 15)
top_n = importance.head(n_top)

st.subheader(f"📊 Top {n_top} Most Important Features")
fig, ax = plt.subplots(figsize=(10, max(4, n_top * 0.4)))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_top))
bars = ax.barh(top_n["Feature"], top_n["Importance"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_title(f"Top {n_top} Feature Importances")
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Search feature ─────────────────────────────────────────────
st.subheader("🔍 Search a Feature")
selected = st.selectbox("Choose a feature to inspect", importance["Feature"])
row = importance[importance["Feature"] == selected].iloc[0]
rank = importance[importance["Feature"] == selected].index[0] + 1

col_a, col_b, col_c = st.columns(3)
col_a.metric("Importance Score", f"{row['Importance']:.4f}")
col_b.metric("Rank",             f"#{rank} of {len(importance)}")
col_c.metric("Percentile",       f"{(1 - rank/len(importance))*100:.0f}th")
st.markdown("---")

# ── Top 10 table ───────────────────────────────────────────────
st.subheader("🥇 Top 10 Features")
st.table(importance.head(10).reset_index(drop=True))
st.markdown("---")

# ── SHAP Section ───────────────────────────────────────────────
st.subheader("🧠 SHAP Explainability")
show_shap = st.checkbox("Generate SHAP Summary Plot (takes a few seconds)")

if show_shap:
    try:
        import shap
        X_sample = X_test[:200]

        with st.spinner("Computing SHAP values…"):
            explainer   = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False, max_display=15)
        st.pyplot(plt.gcf())
        plt.close()

    except ImportError:
        st.warning("⚠️ SHAP is not installed. Run: `pip install shap`")
    except Exception as e:
        st.warning("SHAP could not be generated.")
        st.exception(e)

st.markdown("---")

# ── Business Insights ──────────────────────────────────────────
st.subheader("💡 Business Insights")
top5 = importance.head(5)
for _, row in top5.iterrows():
    st.write(f"✅ **{row['Feature']}** — Importance: `{row['Importance']:.4f}`")

st.success("""
**Interpretation:**

• Features at the top have the **greatest influence** on loan default prediction.

• Financial variables such as Credit Score, Income, Loan Amount, and DTI Ratio typically contribute the most.

• Understanding feature importance helps banks make **transparent and explainable** lending decisions.

• Low-importance features could be candidates for removal to simplify the model.
""")
