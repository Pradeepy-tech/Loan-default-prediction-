"""Training script for the loan default prediction model."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import build_preprocessor, split_features_target
from utils import load_processed_data, save_model


def train_model(
    data_filename: str = "processed_loans.csv",
    target_column: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train a Random Forest classifier and save the fitted pipeline."""
    df = load_processed_data(data_filename)
    X, y = split_features_target(df, target_column=target_column)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    model_path = save_model(pipeline)
    print(f"Model saved to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train_model()
