from typing import Dict

from pydantic import BaseModel, Field


class InputData(BaseModel):
    """Request body for a single inference call.

    We keep the schema flexible by accepting an arbitrary mapping from
    feature name to numeric value. On the backend, this is aligned with the
    exact feature sets expected by the trained models.
    """

    sensor_values: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping from feature name to numeric value, e.g. "
            "'ENGINE_RPM ()_mean': 1500.0."
        ),
        example={
            "ENGINE_RPM ()_mean": 1500.0,
            "VEHICLE_SPEED ()_mean": 60.0,
            "THROTTLE ()_mean": 20.0,
        },
    )


class PredictionResponse(BaseModel):
    """Top-level prediction results for the four production targets."""

    anomaly_label: int = Field(
        ...,
        description="Predicted anomaly flag (0 = normal, 1 = anomaly).",
        example=1,
    )
    severity_score: float = Field(
        ...,
        description=(
            "Continuous severity score between 0.0 and 1.0. "
            "Forced to 0.0 when anomaly_label=0."
        ),
        example=0.72,
    )
    severity_stage: int = Field(
        ...,
        description=(
            "Discrete severity stage (0-4). "
            "Forced to 0 when anomaly_label=0."
        ),
        example=3,
    )
    ttf_km: float = Field(
        ...,
        description="Predicted remaining distance to failure in kilometres.",
        example=12000.0,
    )

