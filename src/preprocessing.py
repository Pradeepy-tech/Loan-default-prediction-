"""Data preprocessing pipeline for loan default prediction."""

from pathlib import Path
import sys
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import save_processed_data


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "loan_status",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess_and_save(
    df: pd.DataFrame,
    output_filename: str = "processed_loans.csv",
    target_column: str = "loan_status",
) -> pd.DataFrame:
    """Apply basic cleaning and save processed data."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=[target_column])

    save_processed_data(cleaned, output_filename)
    return cleaned
