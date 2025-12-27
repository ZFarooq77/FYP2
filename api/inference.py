from typing import Any, Dict
import pandas as pd


"""Inference utilities for the FastAPI layer.

This module is responsible for:
- Loading the trained models and scalers from models/main_models/
- Preparing a single-row input in the exact feature layout expected
  by those models
- Running predictions and applying the anomaly-based gating logic

It reuses the single-row evaluation utilities from
`evaluation/single/evaluate_single.py` to stay consistent with the
offline evaluation pipeline.
"""

from evaluation.single.evaluate_single import (
    load_models,
    evaluate_single,
    get_all_required_features,
)


# Load all models and collect the union of required features once, at import time.
# This keeps request latency low and guarantees consistency across calls.
MODELS_BUNDLE = load_models()
ALL_REQUIRED_FEATURES = get_all_required_features()

# TTF-specific metadata: scaler is fitted on a larger feature set, while the model
# itself uses a selected subset stored in ttf_features.csv. We capture both.
TTF_SCALER_FEATURES = []
TTF_MODEL_FEATURES = []

if "ttf_km" in MODELS_BUNDLE:
    ttf_components = MODELS_BUNDLE["ttf_km"]
    scaler = ttf_components.get("scaler")
    if hasattr(scaler, "feature_names_in_"):
        TTF_SCALER_FEATURES = list(scaler.feature_names_in_)
    TTF_MODEL_FEATURES = ttf_components.get("features") or []

# Base feature set we will always populate in the input row
BASE_REQUIRED_FEATURES = sorted(set(ALL_REQUIRED_FEATURES) | set(TTF_SCALER_FEATURES))


def _prepare_input_row(sensor_values: Dict[str, float]) -> Dict[str, float]:
    """Align raw sensor values to the feature space expected by the models.

    Missing features are filled with 0.0 (as done in the evaluation utilities),
    and any extra keys are passed through as-is.
    """

    # Start with all required features for severity_score, severity_stage, TTF, etc.
    row: Dict[str, float] = {}
    for feature in BASE_REQUIRED_FEATURES:
        value = sensor_values.get(feature, 0.0)
        try:
            row[feature] = float(value)
        except (TypeError, ValueError):
            row[feature] = 0.0

    # Include any additional features that might be useful for anomaly model, etc.
    for name, value in sensor_values.items():
        if name not in row:
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                # Ignore non-numeric extras
                continue

    return row




def _predict_ttf_km(input_row: Dict[str, float]) -> float:
    """Predict TTF_km using the dedicated scaler and model.

    This does not apply anomaly gating; it mirrors the training-time
    pipeline where all numeric features are first scaled and then a
    selected subset is passed to the final model.
    """

    if "ttf_km" not in MODELS_BUNDLE:
        return 0.0

    components = MODELS_BUNDLE["ttf_km"]
    scaler = components.get("scaler")
    model = components.get("model")

    if scaler is None or model is None:
        return 0.0

    # Determine the feature set expected by the scaler
    if TTF_SCALER_FEATURES:
        features_for_scaler = TTF_SCALER_FEATURES
    elif hasattr(scaler, "feature_names_in_"):
        features_for_scaler = list(scaler.feature_names_in_)
    else:
        return 0.0

    # Build a single-row DataFrame with columns in the exact order
    # expected by the scaler.
    row_values = [input_row.get(name, 0.0) for name in features_for_scaler]
    X_all = pd.DataFrame([row_values], columns=features_for_scaler)

    try:
        X_scaled_all = scaler.transform(X_all)
    except Exception:
        return 0.0

    X_scaled_df = pd.DataFrame(X_scaled_all, columns=features_for_scaler)

    # Features actually used by the TTF model; if not present, fall back
    # to using the full scaled feature set.
    features_for_model = TTF_MODEL_FEATURES or features_for_scaler

    try:
        X_ttf = X_scaled_df[features_for_model]
    except KeyError:
        # If some model features are missing, default to zeros with the
        # right column names.
        X_ttf = pd.DataFrame([[0.0] * len(features_for_model)], columns=features_for_model)

    try:
        pred = model.predict(X_ttf)
        return float(pred[0])
    except Exception:
        return 0.0

def run_inference(sensor_values: Dict[str, float]) -> Dict[str, Any]:
    """Run full inference pipeline for a single row of sensor data.

    Args:
        sensor_values: Mapping from feature name to numeric value.

    Returns:
        Dictionary compatible with `PredictionResponse` in `api.schemas`:
        {"anomaly_label", "severity_score", "severity_stage", "ttf_km"}.
    """

    if not MODELS_BUNDLE:
        raise RuntimeError("Model bundle is not loaded. Check model files under models/main_models/.")

    # Prepare input in the format expected by evaluation utilities
    input_row = _prepare_input_row(sensor_values)

    # Use the existing single-row evaluation logic for anomaly + severity targets.
    # We deliberately exclude TTF here and compute it with a dedicated path to
    # avoid feature-name alignment issues with the scaler.
    models_for_eval = {k: v for k, v in MODELS_BUNDLE.items() if k != "ttf_km"}

    predictions, _details = evaluate_single(
        input_row=input_row,
        models_bundle=models_for_eval,
        return_dataframe=False,
    )

    # --- Extract and validate core predictions ---
    try:
        anomaly_raw = predictions["anomaly_label"]
        anomaly_label = int(anomaly_raw)
    except Exception as exc:  # KeyError / ValueError / TypeError
        raise RuntimeError(f"Anomaly prediction failed: {predictions.get('anomaly_label')}") from exc

    def _get_float(name: str) -> float:
        value = predictions.get(name)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} prediction failed: {value}") from exc

    severity_score_raw = _get_float("severity_score")

    try:
        severity_stage_raw = predictions.get("severity_stage")
        severity_stage_raw_int = int(severity_stage_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"severity_stage prediction failed: {predictions.get('severity_stage')}") from exc

    # --- Apply anomaly-based gating (same semantics as main.py) ---
    if anomaly_label == 0:
        # For normal windows, force severity outputs to their neutral values
        severity_score = 0.0
        severity_stage = 0
    else:
        severity_score = severity_score_raw
        severity_stage = severity_stage_raw_int

    # --- Independent TTF_km prediction (no gating applied here) ---
    ttf_km_val = _predict_ttf_km(input_row)

    return {
        "anomaly_label": anomaly_label,
        "severity_score": severity_score,
        "severity_stage": severity_stage,
        "ttf_km": ttf_km_val,
    }

