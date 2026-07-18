"""
run_pipeline.py
Full end-to-end pipeline:
  1. Load Loan_default.csv from data/raw/
  2. Preprocess & save to data/processed/
  3. Train RandomForest model & evaluate
  4. Save model + preprocessor to models/
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
RAW_CSV       = PROJECT_ROOT / "data" / "raw"  / "Loan_default.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL  = "Defaulted?"
DROP_COLS   = ["Index"]           # ID column — not a feature


# ── 1. Load ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading raw CSV …")
df = pd.read_csv(RAW_CSV)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Target distribution:\n{df[TARGET_COL].value_counts().to_string()}")


# ── 2. Preprocess ──────────────────────────────────────────────────────────────
print("\nSTEP 2 — Preprocessing …")
df = df.drop(columns=DROP_COLS, errors="ignore")
df = df.drop_duplicates()
df = df.dropna(subset=[TARGET_COL])

processed_path = PROCESSED_DIR / "processed_loans.csv"
df.to_csv(processed_path, index=False)
print(f"  Cleaned rows: {len(df):,}")
print(f"  Saved to: {processed_path}")


# ── 3. Feature split ───────────────────────────────────────────────────────────
print("\nSTEP 3 — Splitting features & target …")
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print(f"  Numeric features    : {numeric_features}")
print(f"  Categorical features: {categorical_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")


# ── 4. Build preprocessor ──────────────────────────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])


# ── 5. Train model ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Training RandomForest model …")
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

pipeline.fit(X_train, y_train)
print("  Training complete ✓")


# ── 6. Evaluate ────────────────────────────────────────────────────────────────
print("\nSTEP 5 — Evaluating on test set …")
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
print(f"\n  ROC-AUC Score : {roc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")


# ── 7. Save model ──────────────────────────────────────────────────────────────
print("\nSTEP 6 — Saving model …")
model_path       = MODELS_DIR / "best_model.pkl"
preprocessor_path = MODELS_DIR / "preprocessor.pkl"

joblib.dump(pipeline,              model_path)
joblib.dump(pipeline["preprocessor"], preprocessor_path)

print(f"  Model saved      : {model_path}")
print(f"  Preprocessor saved: {preprocessor_path}")

print("\n" + "=" * 60)
print("Pipeline complete! ✅")
print("=" * 60)
