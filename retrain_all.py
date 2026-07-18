"""
retrain_all.py  ─ Full pipeline for Loan_default.csv
Trains RandomForest on the 255k-row dataset and saves:
  models/best_model.pkl
  models/preprocessor.pkl
  models/processed_data.pkl
  models/model_comparison.csv
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent

# ── 1. Load ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Loan_default.csv …")
df = pd.read_csv(ROOT / "Loan_default.csv")
print(f"  Shape: {df.shape}")

TARGET   = "Default"
DROP     = ["LoanID"]

df = df.drop(columns=DROP, errors="ignore")

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print(f"  Numeric  : {numeric_features}")
print(f"  Categorical: {categorical_features}")

# ── 2. Train / Test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nSTEP 2 — Split: Train={len(X_train):,}  Test={len(X_test):,}")

# ── 3. Preprocessor ────────────────────────────────────────────
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

print("\nSTEP 3 — Fitting preprocessor …")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
print(f"  Processed shape: {X_train_proc.shape}")

# ── 4. Train & compare models ──────────────────────────────────
print("\nSTEP 4 — Training models …")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
}

results = []
best_model      = None
best_roc        = 0.0
best_model_name = ""

for name, clf in models.items():
    clf.fit(X_train_proc, y_train)
    y_pred  = clf.predict(X_test_proc)
    y_proba = clf.predict_proba(X_test_proc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba)

    results.append({
        "Model":     name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4),
        "ROC AUC":   round(roc,  4),
    })

    print(f"  {name:<22}  ROC={roc:.4f}  F1={f1:.4f}")

    if roc > best_roc:
        best_roc        = roc
        best_model      = clf
        best_model_name = name

print(f"\n  Best model: {best_model_name}  (ROC-AUC={best_roc:.4f})")

# ── 5. Save artefacts ──────────────────────────────────────────
print("\nSTEP 5 — Saving artefacts …")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

joblib.dump(best_model,   MODELS_DIR / "best_model.pkl")
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

processed_data = {
    "X_train": X_train_proc,
    "X_test":  X_test_proc,
    "y_train": y_train.values,
    "y_test":  y_test.values,
    "feature_names": preprocessor.get_feature_names_out().tolist(),
    "X_train_raw": X_train,
    "X_test_raw":  X_test,
}
joblib.dump(processed_data, MODELS_DIR / "processed_data.pkl")

comparison_df = pd.DataFrame(results)
comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

print(f"  best_model.pkl      saved ✓")
print(f"  preprocessor.pkl    saved ✓")
print(f"  processed_data.pkl  saved ✓")
print(f"  model_comparison.csv saved ✓")

print("\n" + "=" * 60)
print("Pipeline complete! ✅")
print("=" * 60)
