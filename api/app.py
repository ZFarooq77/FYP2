from fastapi import FastAPI, HTTPException

from api.schemas import InputData, PredictionResponse
from api.inference import run_inference


app = FastAPI(title="Vehicle Fault Prediction API")


@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint used by clients and monitoring."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: InputData) -> PredictionResponse:
    """Run anomaly, severity, and TTF predictions for a single input row.

    The request body should provide `sensor_values` as a mapping from
    feature name to numeric value. The backend aligns these to the
    feature sets expected by the trained models and applies the
    anomaly-based gating logic:

    - If anomaly_label == 0 => severity_score = 0.0, severity_stage = 0
    - If anomaly_label == 1 => return the model predictions
    """

    try:
        result = run_inference(payload.sensor_values)
        return PredictionResponse(**result)
    except Exception as exc:
        # Surface a clean error to the client while keeping details in the log
        raise HTTPException(status_code=500, detail=str(exc))

