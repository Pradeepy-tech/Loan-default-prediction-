"""
FastAPI Backend Application for Loan Default Prediction
Powered by Uvicorn ASGI Server
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project directories
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"

# Required feature columns expected by model preprocessor
REQUIRED_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

# Global variables for model and preprocessor
model = None
preprocessor = None


def load_model_artifacts():
    """Load machine learning model and preprocessor pipelines."""
    global model, preprocessor
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts missing. Expected {MODEL_PATH} and {PREPROCESSOR_PATH}. "
            "Please run retrain_all.py first."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML models on app startup."""
    try:
        load_model_artifacts()
        print("✅ ML Model and Preprocessor loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML artifacts: {e}")
    yield


app = FastAPI(
    title="🏦 Loan Default Prediction API",
    description="Machine Learning REST API for predicting customer loan default risk.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic schema for single loan application input validation
class LoanApplication(BaseModel):
    Age: int = Field(..., ge=18, le=100, example=35, description="Applicant age in years")
    Income: float = Field(..., ge=0, example=60000.0, description="Annual income in ₹")
    LoanAmount: float = Field(..., ge=0, example=20000.0, description="Requested loan amount in ₹")
    CreditScore: int = Field(..., ge=300, le=850, example=650, description="Credit score (300 - 850)")
    MonthsEmployed: int = Field(..., ge=0, le=600, example=60, description="Total months employed")
    NumCreditLines: int = Field(..., ge=0, le=20, example=3, description="Number of open credit lines")
    InterestRate: float = Field(..., ge=0.0, le=50.0, example=10.5, description="Interest rate percentage")
    LoanTerm: int = Field(..., example=36, description="Loan duration in months (e.g. 12, 24, 36, 48, 60)")
    DTIRatio: float = Field(..., ge=0.0, le=1.0, example=0.30, description="Debt-to-Income ratio (0.0 to 1.0)")
    Education: Literal["High School", "Bachelor's", "Master's", "PhD"] = Field(
        ..., example="Bachelor's", description="Highest education level"
    )
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"] = Field(
        ..., example="Full-time", description="Current employment type"
    )
    MaritalStatus: Literal["Single", "Married", "Divorced"] = Field(
        ..., example="Married", description="Marital status"
    )
    HasMortgage: Literal["Yes", "No"] = Field(..., example="Yes", description="Has active mortgage?")
    HasDependents: Literal["Yes", "No"] = Field(..., example="No", description="Has financial dependents?")
    LoanPurpose: Literal["Home", "Auto", "Business", "Education", "Other"] = Field(
        ..., example="Home", description="Purpose of loan"
    )
    HasCoSigner: Literal["Yes", "No"] = Field(..., example="No", description="Has co-signer?")


class PredictionResult(BaseModel):
    prediction: Literal["Safe", "Default"]
    prediction_code: int
    default_probability: float
    default_probability_percentage: float
    confidence_percentage: float
    risk_level: Literal["Low", "Medium", "High"]
    recommendation: str


def compute_prediction_response(input_df: pd.DataFrame) -> List[PredictionResult]:
    """Transform input dataframe and compute prediction outputs."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure best_model.pkl and preprocessor.pkl exist.",
        )

    processed_data = preprocessor.transform(input_df)
    preds = model.predict(processed_data)
    probs = model.predict_proba(processed_data)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        prob_pct = round(float(prob) * 100, 2)
        conf_pct = round(max(float(prob), 1.0 - float(prob)) * 100, 2)
        pred_label = "Default" if int(pred) == 1 else "Safe"

        if prob >= 0.70:
            risk_level = "High"
            recommendation = "🔴 HIGH RISK — Reject Loan"
        elif prob >= 0.40:
            risk_level = "Medium"
            recommendation = "🟠 MEDIUM RISK — Manual Review Required"
        else:
            risk_level = "Low"
            recommendation = "🟢 LOW RISK — Loan Can Be Approved"

        results.append(
            PredictionResult(
                prediction=pred_label,
                prediction_code=int(pred),
                default_probability=round(float(prob), 4),
                default_probability_percentage=prob_pct,
                confidence_percentage=conf_pct,
                risk_level=risk_level,
                recommendation=recommendation,
            )
        )
    return results


@app.get("/", tags=["General"])
def root():
    """API Welcome & Documentation Overview."""
    return {
        "message": "Welcome to Loan Default Prediction API",
        "status": "online",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/health",
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint to verify backend state and ML model load status."""
    is_model_ready = model is not None and preprocessor is not None
    return {
        "status": "healthy" if is_model_ready else "degraded",
        "model_loaded": is_model_ready,
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Predictions"])
def predict_single(application: LoanApplication):
    """
    Predict loan default risk for a single customer application.
    """
    df = pd.DataFrame([application.model_dump()])
    results = compute_prediction_response(df)
    return results[0]


@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Predictions"])
def predict_batch(applications: List[LoanApplication]):
    """
    Predict loan default risk for multiple customer applications (JSON batch).
    """
    if not applications:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Input application list cannot be empty."
        )
    df = pd.DataFrame([app.model_dump() for app in applications])
    return compute_prediction_response(df)


@app.post("/predict/csv", tags=["Predictions"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get default predictions for every customer in the file.
    Returns JSON with overall summary stats and predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be a CSV file."
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse CSV file: {str(e)}"
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV missing required columns: {missing}",
        )

    feature_df = df[REQUIRED_COLUMNS].copy()
    results = compute_prediction_response(feature_df)

    output_df = df.copy()
    output_df["Prediction"] = [r.prediction for r in results]
    output_df["Default_Probability_%"] = [r.default_probability_percentage for r in results]
    output_df["Risk_Level"] = [r.risk_level for r in results]
    output_df["Recommendation"] = [r.recommendation for r in results]

    total_customers = len(output_df)
    default_count = sum(1 for r in results if r.prediction == "Default")
    safe_count = total_customers - default_count

    return {
        "summary": {
            "total_customers": total_customers,
            "safe_customers": safe_count,
            "default_customers": default_count,
            "default_rate_percentage": round((default_count / total_customers) * 100, 2) if total_customers > 0 else 0,
        },
        "results": output_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    # Allows running directly via `python main.py`
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🚀 Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
