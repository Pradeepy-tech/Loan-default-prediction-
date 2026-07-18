"""Shared utilities for the Loan Default Prediction project."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_raw_data(filename: str) -> pd.DataFrame:
    """Load a CSV file from the raw data directory."""
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load a CSV file from the processed data directory."""
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to the processed data directory."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    return path


def save_model(model, filename: str = "loan_default_model.joblib") -> Path:
    """Persist a trained model to the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str = "loan_default_model.joblib"):
    """Load a trained model from the models directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
