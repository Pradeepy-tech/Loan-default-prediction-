"""Inference script for loan default prediction."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_model


def predict_default(
    input_data: pd.DataFrame,
    model_filename: str = "loan_default_model.joblib",
) -> pd.DataFrame:
    """Run predictions on new loan applications."""
    model = load_model(model_filename)
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    return pd.DataFrame(
        {
            "prediction": predictions,
            "default_probability": probabilities,
        }
    )


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "loan_amount": [15000],
            "annual_income": [55000],
            "credit_score": [680],
            "employment_status": ["Employed"],
        }
    )
    result = predict_default(sample)
    print(result)
