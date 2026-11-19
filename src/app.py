
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_housing_model.pkl"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)


# FastAPI application
app = FastAPI(
    title="Housing Price Prediction API",
    description="MLOps API powered by an XGBoost regression model",
    version="1.0.0"
)


# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "housing_prediction_requests_total",
    "Total number of prediction requests"
)

PREDICTION_ERRORS = Counter(
    "housing_prediction_errors_total",
    "Total number of failed prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "housing_prediction_duration_seconds",
    "Time taken to generate predictions"
)


# Input schema
class HousingInput(BaseModel):

    MedInc: float = Field(..., ge=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


# Health check
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "XGBoost Regressor",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict_house_price(data: HousingInput):

    PREDICTION_REQUESTS.inc()

    try:
        with PREDICTION_LATENCY.time():

            input_data = pd.DataFrame([data.model_dump()])

            prediction = float(model.predict(input_data)[0])

        return {
            "prediction": round(prediction, 4),
            "currency": "USD",
            "prediction_unit": "100000 USD",
            "estimated_price_usd": round(prediction * 100000, 2),
            "message": "Prediction generated successfully"
        }

    except Exception as error:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

