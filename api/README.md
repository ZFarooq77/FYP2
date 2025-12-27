# Vehicle Fault Prediction API

This directory contains the FastAPI service that serves predictions from the
trained models in `models/main_models/`.

The API exposes:

- `GET /health` – health check
- `POST /predict` – main inference endpoint

All targets are served using **one model + one scaler per target column**:

- `anomaly_label`
- `severity_score`
- `severity_stage`
- `ttf_km`

The API reuses the single-row evaluation logic from
`evaluation/single/evaluate_single.py` to keep behavior consistent with your
offline evaluation.

## 1. Requirements

Install the minimal dependencies for running the API (inside your existing
virtual environment):

```bash
pip install fastapi "uvicorn[standard]"
```

Ensure the trained model artifacts exist in:

- `models/main_models/`

with (at least) the following files:

- `anomaly_model.pkl`
- `severity_score_model.pkl`, `severity_score_scaler.pkl`, `selected_features_v2.csv`
- `severity_stage_model.pkl`, `severity_stage_scaler.pkl`
- `ttf_model.pkl`, `ttf_scaler.pkl`, `ttf_features.csv`

## 2. Running the API locally

From the project root (`ModelTraining/`):

```bash
uvicorn api.app:app --reload
```

Then open Swagger UI at:

- http://127.0.0.1:8000/docs

You can test both endpoints directly from the browser.

## 3. Endpoints

### 3.1 `GET /health`

Simple liveness check.

**Response:**

```json
{"status": "ok"}
```

### 3.2 `POST /predict`

Runs the full ML pipeline for a single window of sensor data.

**Request body (schema `InputData`):**

```json
{
  "sensor_values": {
    "ENGINE_RPM ()_mean": 1500.0,
    "VEHICLE_SPEED ()_mean": 60.0,
    "THROTTLE ()_mean": 20.0
  }
}
```

`sensor_values` is a flexible mapping from **feature name** → **numeric value**.
The backend aligns this to the exact feature sets required by each model and its
scaler.

**Response body (schema `PredictionResponse`):**

```json
{
  "anomaly_label": 1,
  "severity_score": 0.72,
  "severity_stage": 3,
  "ttf_km": 12000.0
}
```

### 3.3 Severity gating logic

The API enforces the same gating semantics as the offline runtime logic
(`main.py`):

- If `anomaly_label == 0` (normal window):
  - `severity_score` is forced to `0.0`.
  - `severity_stage` is forced to `0`.
- If `anomaly_label == 1` (anomalous window):
  - `severity_score` and `severity_stage` are taken directly from the
    respective models.

`ttf_km` is predicted independently using its own model + scaler and is **not
subject to gating**.

## 4. Implementation overview

Key modules:

- `api/app.py` – FastAPI application, defines `/health` and `/predict`.
- `api/schemas.py` – Pydantic models: `InputData`, `PredictionResponse`.
- `api/inference.py` – Model loading and inference logic, including:
  - One model + one scaler per target column.
  - Reuse of `evaluation/single/evaluate_single.py` for anomaly + severity.
  - Dedicated path for `ttf_km` that follows the training-time scaler +
    feature-selection pipeline.

