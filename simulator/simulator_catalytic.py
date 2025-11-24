"""
simulator_catalytic.py — Enhanced Catalytic Converter Simulator 🚀
================================================================
Purpose:
    Interactive Streamlit simulator for catalytic converter degradation using
    physics-based sensor modeling. Generates real-time synthetic data and 
    accumulates samples for batch testing with trained ML models.

Features:
    - Real-time physics-based sensor simulation
    - Interactive severity slider (0-100%)
    - Sample accumulation and CSV export
    - Model inference integration
    - TTF prediction visualization

Usage:
    streamlit run simulator/simulator_catalytic.py

Author: FYP Team - Catalytic Converter Predictive Maintenance
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import math
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Streamlit and visualization
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths (absolute paths from simulator directory)
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_CSV = os.path.join(PARENT_DIR, "Final.csv")
MODEL_PATHS = {
    "anomaly": os.path.join(PARENT_DIR, "models", "best_anomaly_model_xgboost.pkl"),
    "severity_score": os.path.join(PARENT_DIR, "models", "severity_stacking_model_v2.pkl"),
    "severity_stage": os.path.join(PARENT_DIR, "models", "severity_stage_stacking_model_v1.pkl"),
    "ttf_km": os.path.join(PARENT_DIR, "models", "best_ttf_model_gradient_boosting.pkl")
}

# Key catalytic converter sensors (from your physics code)
KEY_SENSORS = [
    "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean",
    "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean",
    "ENGINE_LOAD ()_mean",
    "INTAKE_MANIFOLD_PRESSURE ()_mean",
    "THROTTLE ()_mean",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()_mean",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()_mean",
    "ENGINE_RPM ()_std"
]

# Sensor roles for physics-based degradation
SENSOR_ROLES = {
    "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean": "up",
    "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean": "up",
    "ENGINE_LOAD ()_mean": "down",
    "INTAKE_MANIFOLD_PRESSURE ()_mean": "up",
    "THROTTLE ()_mean": "up",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()_mean": "trim",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()_mean": "trim",
    "ENGINE_RPM ()_std": "rpm_std"
}

# Physics parameters (from your exact formulas)
ZONE1_MAX = 0.2      # Normal zone boundary
ZONE2_MAX = 0.6      # Moderate zone boundary
TEMP_EXP = 1.2       # Temperature exponent
ZONE2_MULT = 1.0     # Zone 2 drift multiplier
ZONE3_MULT = 1.6     # Zone 3 overshoot multiplier
BASE_YEARS = 4.0     # Expected catalyst lifetime
AVG_KM_PER_YEAR = 15000  # Average yearly driving

@st.cache_data
def load_dataset_stats() -> Dict[str, Dict[str, float]]:
    """
    Load dataset statistics for ALL features to match model expectations.

    Returns:
        Dictionary of sensor_name -> {min, max, mean, std}
    """
    try:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)

            # Remove target columns for feature extraction
            feature_cols = [col for col in df.columns if col not in
                          ['anomaly_label', 'severity_score', 'severity_stage', 'TTF_km', 'TTF_years', 'source', 'failure_type', 'is_gray']]

            stats = {}
            for col in feature_cols:
                if df[col].dtype in ['int64', 'float64']:
                    stats[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }

            return stats
        else:
            return {}
    except Exception as e:
        st.sidebar.error(f"❌ Dataset load failed: {str(e)[:50]}...")
        return {}

# Severity zone mappings
SEVERITY_ZONES = {
    (0, 20): {"name": "Normal", "color": "#28a745", "stage": 0},
    (20, 40): {"name": "Mild", "color": "#ffc107", "stage": 1}, 
    (40, 60): {"name": "Moderate", "color": "#fd7e14", "stage": 2},
    (60, 80): {"name": "Severe", "color": "#dc3545", "stage": 3},
    (80, 100): {"name": "Failure", "color": "#6f42c1", "stage": 4}
}

# Default sensor statistics (fallback if Final.csv not found)
DEFAULT_STATS = {
    "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean": {"min": 300, "mean": 450, "max": 900},
    "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean": {"min": 250, "mean": 400, "max": 850},
    "ENGINE_LOAD ()_mean": {"min": 10, "mean": 50, "max": 100},
    "INTAKE_MANIFOLD_PRESSURE ()_mean": {"min": 20, "mean": 45, "max": 100},
    "THROTTLE ()_mean": {"min": 0, "mean": 20, "max": 90},
    "LONG_TERM_FUEL_TRIM_BANK_1 ()_mean": {"min": -5, "mean": 0, "max": 10},
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()_mean": {"min": -5, "mean": 0, "max": 10},
    "ENGINE_RPM ()_std": {"min": 10, "mean": 35, "max": 120}
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def load_sensor_statistics(data_csv: str = DATA_CSV) -> Dict[str, Dict[str, float]]:
    """
    Load sensor min/mean/max statistics from Final.csv.
    Falls back to defaults if file not found.
    """
    try:
        if os.path.exists(data_csv):
            df = pd.read_csv(data_csv)
            stats = {}
            
            for sensor in KEY_SENSORS:
                if sensor in df.columns:
                    col_data = df[sensor].dropna()
                    if len(col_data) > 0:
                        stats[sensor] = {
                            "min": float(col_data.min()),
                            "mean": float(col_data.mean()), 
                            "max": float(col_data.max())
                        }
                        continue
                
                # Fallback to default if column missing/empty
                stats[sensor] = DEFAULT_STATS.get(sensor, {"min": 0.0, "mean": 1.0, "max": 2.0})
            
            return stats
        else:
            st.warning(f"⚠️ Dataset {data_csv} not found. Using default sensor statistics.")
            return {sensor: DEFAULT_STATS.get(sensor, {"min": 0.0, "mean": 1.0, "max": 2.0}) 
                   for sensor in KEY_SENSORS}
            
    except Exception as e:
        st.error(f"❌ Error loading sensor statistics: {e}")
        return {sensor: DEFAULT_STATS.get(sensor, {"min": 0.0, "mean": 1.0, "max": 2.0}) 
               for sensor in KEY_SENSORS}

def get_severity_zone_info(severity_percent: float) -> Dict[str, Any]:
    """Get zone information for given severity percentage."""
    for (min_val, max_val), info in SEVERITY_ZONES.items():
        if min_val <= severity_percent < max_val:
            return info
    # Handle edge case for 100%
    if severity_percent >= 80:
        return SEVERITY_ZONES[(80, 100)]
    return SEVERITY_ZONES[(0, 20)]  # Default to normal

def minmax_scale(value: float, vmin: float, vmax: float) -> float:
    """Min-max normalization (0-1 scaling)."""
    if vmax - vmin == 0:
        return 0.5  # Neutral value
    return (value - vmin) / (vmax - vmin)

# =============================================================================
# PHYSICS ENGINE - THREE-ZONE SENSOR TRANSFORM
# =============================================================================

def three_zone_sensor_transform(
    baseline_value: float,
    sensor_stats: Dict[str, float], 
    severity: float,
    role: str,
    rng: np.random.RandomState = None
) -> float:
    """
    Apply three-zone transform for a single sensor using your exact physics formulas.
    
    Args:
        baseline_value: Starting sensor value
        sensor_stats: Dict with 'min', 'mean', 'max' keys
        severity: Severity score (0.0-1.0)
        role: Sensor role ('up', 'down', 'trim', 'rpm_std')
        rng: Random number generator
        
    Returns:
        Transformed sensor value
    """
    if rng is None:
        rng = np.random.RandomState()
        
    if sensor_stats is None:
        return baseline_value + rng.normal(0, 1e-6)
    
    mn = sensor_stats["min"]
    mean = sensor_stats["mean"] 
    mx = sensor_stats["max"]
    data_range = mx - mn if (mx - mn) != 0 else 1.0
    
    # Zone 1: Normal operation (0-20%)
    if severity <= ZONE1_MAX:
        # Very small jitter - max 0.5% of baseline value
        jitter_scale = 0.005 * max(abs(baseline_value), 1.0)
        jitter = rng.normal(0, jitter_scale)
        val = baseline_value + jitter
        return float(np.clip(val, mn, mx))
    
    # Zone 2: Mild/Moderate anomalies (20-60%)
    elif severity <= ZONE2_MAX:
        s2 = (severity - ZONE1_MAX) / (ZONE2_MAX - ZONE1_MAX)  # Normalize 0-1
        
        if role == "up":  # Catalyst temps, throttle, MAP
            drift = (mx - mean) * ZONE2_MULT * (s2 ** TEMP_EXP)
            val = baseline_value + drift
        elif role == "down":  # Engine load
            drift = (mean - mn) * ZONE2_MULT * (s2 ** TEMP_EXP)
            val = baseline_value - drift
        elif role == "trim":  # Fuel trims
            drift = (mx - mean) * ZONE2_MULT * (s2 ** (TEMP_EXP * 0.9))
            val = baseline_value + drift + rng.normal(0, 0.01 * data_range)
        elif role == "rpm_std":  # RPM variability
            drift = (mx - mean) * ZONE2_MULT * (s2 ** 1.0)
            val = baseline_value + drift
        else:  # Default behavior
            if mean < (mn + mx) / 2:
                drift = (mx - mean) * ZONE2_MULT * (s2 ** TEMP_EXP)
                val = baseline_value + drift
            else:
                drift = (mean - mn) * ZONE2_MULT * (s2 ** TEMP_EXP)
                val = baseline_value - drift
        
        # Clip to bounds for Zone 2
        return float(np.clip(val, mn, mx))
    
    # Zone 3: Severe/Failure (60-100%) - can exceed bounds
    else:
        s3 = (severity - ZONE2_MAX) / (1.0 - ZONE2_MAX)  # Normalize 0-1
        
        if role == "up":
            overshoot = (mx - mean) * ZONE3_MULT * (s3 ** TEMP_EXP)
            val = baseline_value + overshoot + rng.normal(0, 0.02 * (mx - mean))
        elif role == "down":
            undershoot = (mean - mn) * ZONE3_MULT * (s3 ** TEMP_EXP)
            val = baseline_value - undershoot + rng.normal(0, 0.02 * (mean - mn))
        elif role == "trim":
            overshoot = (mx - mean) * ZONE3_MULT * (s3 ** (TEMP_EXP * 0.95))
            val = baseline_value + overshoot + rng.normal(0, 0.05 * (mx - mean))
        elif role == "rpm_std":
            overshoot = (mx - mean) * ZONE3_MULT * (s3 ** 1.0)
            val = baseline_value + overshoot + rng.normal(0, 0.05 * (mx - mean))
        else:  # Default: random direction beyond bounds
            if rng.rand() > 0.5:
                val = mx + (mx - mean) * ZONE3_MULT * s3
            else:
                val = mn - (mean - mn) * ZONE3_MULT * s3
        
        return float(val)

# =============================================================================
# PHYSICS ENGINE - CORRELATION ADJUSTMENTS
# =============================================================================

def apply_sensor_correlations(
    sensor_values: Dict[str, float],
    sensor_stats: Dict[str, Dict[str, float]],
    rng: np.random.RandomState = None
) -> Dict[str, float]:
    """
    Apply physics-based correlations between sensors using your exact formulas.

    Args:
        sensor_values: Dict of sensor name -> current value
        sensor_stats: Dict of sensor name -> {'min', 'mean', 'max'}
        rng: Random number generator

    Returns:
        Updated sensor values with correlations applied
    """
    if rng is None:
        rng = np.random.RandomState()

    values = sensor_values.copy()

    # Catalyst Temperature correlations
    ct1_name = "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean"
    ct2_name = "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean"
    load_name = "ENGINE_LOAD ()_mean"
    throttle_name = "THROTTLE ()_mean"

    if ct1_name in values and ct1_name in sensor_stats:
        mean_ct1 = sensor_stats[ct1_name]["mean"]
        max_ct1 = sensor_stats[ct1_name]["max"]
        delta_ct1 = values[ct1_name] - mean_ct1
        norm_delta = delta_ct1 / (max_ct1 - mean_ct1) if (max_ct1 - mean_ct1) != 0 else 0.0

        # CT1 ↑ → Engine Load ↓ (pumping losses)
        if load_name in values and load_name in sensor_stats:
            load_adjustment = 0.15 * norm_delta * (sensor_stats[load_name]["max"] - sensor_stats[load_name]["mean"])
            values[load_name] = values[load_name] - load_adjustment
            values[load_name] = np.clip(values[load_name],
                                      sensor_stats[load_name]["min"],
                                      sensor_stats[load_name]["max"])

        # CT1 ↑ → Throttle ↑ (compensation)
        if throttle_name in values and throttle_name in sensor_stats:
            throttle_adjustment = 0.10 * max(norm_delta, 0.0) * (sensor_stats[throttle_name]["max"] - sensor_stats[throttle_name]["mean"])
            values[throttle_name] = values[throttle_name] + throttle_adjustment
            values[throttle_name] = np.clip(values[throttle_name],
                                          sensor_stats[throttle_name]["min"],
                                          sensor_stats[throttle_name]["max"])

        # CT2 follows CT1 (inlet/outlet relationship)
        if ct2_name in values:
            values[ct2_name] = values[ct2_name] + 0.9 * delta_ct1

    return values

# =============================================================================
# PHYSICS ENGINE - HYBRID TTF CALCULATION
# =============================================================================

def compute_hybrid_ttf(sensor_values: Dict[str, float], sensor_stats: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """
    Compute hybrid TTF (years + km) using your exact physics formulas.

    Args:
        sensor_values: Dict of sensor name -> value
        sensor_stats: Dict of sensor name -> {'min', 'mean', 'max'}

    Returns:
        Tuple of (TTF_years, TTF_km)
    """
    # Extract key sensors for TTF calculation
    ct1 = sensor_values.get("CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean", 0)
    ct2 = sensor_values.get("CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean", 0)
    engine_load = sensor_values.get("ENGINE_LOAD ()_mean", 0)
    map_pressure = sensor_values.get("INTAKE_MANIFOLD_PRESSURE ()_mean", 0)
    throttle = sensor_values.get("THROTTLE ()_mean", 0)

    # Min-max scale sensors for physics calculations
    ct1_scaled = minmax_scale(ct1,
                             sensor_stats["CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean"]["min"],
                             sensor_stats["CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean"]["max"])
    ct2_scaled = minmax_scale(ct2,
                             sensor_stats["CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean"]["min"],
                             sensor_stats["CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean"]["max"])
    engine_load_scaled = minmax_scale(engine_load,
                                    sensor_stats["ENGINE_LOAD ()_mean"]["min"],
                                    sensor_stats["ENGINE_LOAD ()_mean"]["max"])
    map_scaled = minmax_scale(map_pressure,
                            sensor_stats["INTAKE_MANIFOLD_PRESSURE ()_mean"]["min"],
                            sensor_stats["INTAKE_MANIFOLD_PRESSURE ()_mean"]["max"])
    throttle_scaled = minmax_scale(throttle,
                                 sensor_stats["THROTTLE ()_mean"]["min"],
                                 sensor_stats["THROTTLE ()_mean"]["max"])

    # Enhanced physics degradation model (your exact formulas)
    cat_health = 0.6 * ct1_scaled + 0.4 * ct2_scaled
    thermal_aging = 0.6 * cat_health + 0.2 * engine_load_scaled + 0.2 * throttle_scaled
    usage_aging = 0.5 * engine_load_scaled + 0.3 * throttle_scaled - 0.2 * map_scaled
    degradation = 0.5 * thermal_aging + 0.5 * usage_aging

    # Get severity score and stage from sensor values
    severity_score = sensor_values.get("severity_score", 0.0)
    severity_stage = sensor_values.get("severity_stage", 0)
    severity_stage_norm = severity_stage / 4.0

    # Hybrid TTF Formula (your exact formula)
    TTF_years = (1 - (0.6 * severity_score + 0.25 * severity_stage_norm + 0.15 * degradation)) * BASE_YEARS
    TTF_years = max(TTF_years, 0.0)
    TTF_km = TTF_years * AVG_KM_PER_YEAR

    return float(TTF_years), float(TTF_km)

# =============================================================================
# MAIN SENSOR GENERATION FUNCTION
# =============================================================================

def generate_sensor_row(severity_percent: float, all_sensor_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Generate a complete sensor row with ALL features to match model expectations.

    Args:
        severity_percent: Severity percentage (0-100)
        all_sensor_stats: ALL sensor statistics dictionary from Final.csv

    Returns:
        Single-row DataFrame with ALL sensor values and derived metrics
    """
    # Convert percentage to 0-1 severity score
    severity_score = np.clip(severity_percent / 100.0, 0.0, 1.0)

    # Add small random noise to severity score
    rng = np.random.RandomState()
    severity_score = np.clip(severity_score + rng.normal(0, 0.01), 0.0, 1.0)

    # Initialize sensor values dictionary
    sensor_values = {}

    # Generate ALL sensor values (not just key sensors)
    for sensor_name, stats in all_sensor_stats.items():
        if sensor_name in KEY_SENSORS:
            # Apply physics-based transformation to key sensors
            baseline = stats["mean"]
            role = SENSOR_ROLES.get(sensor_name, "up")

            # Apply three-zone transform
            new_value = three_zone_sensor_transform(
                baseline_value=baseline,
                sensor_stats=stats,
                severity=severity_score,
                role=role,
                rng=rng
            )
            sensor_values[sensor_name] = new_value
        else:
            # For non-key sensors, apply mild correlation or keep near baseline
            baseline = stats["mean"]

            if severity_score <= 0.2:
                # Normal: very small jitter (max 0.5% of baseline)
                jitter_scale = 0.005 * max(abs(baseline), 1.0)
                jitter = rng.normal(0, jitter_scale)
                sensor_values[sensor_name] = float(np.clip(baseline + jitter, stats['min'], stats['max']))
            else:
                # Mild correlation with severity for non-key sensors
                correlation_factor = 0.1 * severity_score  # Small influence
                if rng.rand() > 0.5:
                    drift = (stats['max'] - baseline) * correlation_factor
                else:
                    drift = (baseline - stats['min']) * correlation_factor

                new_val = baseline + drift
                sensor_values[sensor_name] = float(np.clip(new_val, stats['min'], stats['max']))

    # Apply physics-based correlations to key sensors
    key_sensor_values = {k: v for k, v in sensor_values.items() if k in KEY_SENSORS}
    key_sensor_stats = {k: v for k, v in all_sensor_stats.items() if k in KEY_SENSORS}
    correlated_values = apply_sensor_correlations(key_sensor_values, key_sensor_stats, rng)

    # Update with correlated values
    sensor_values.update(correlated_values)

    # Add derived metrics
    sensor_values["severity_score"] = float(severity_score)

    # Map severity score to discrete stage (0-4)
    if severity_score <= 0.05:
        severity_stage = 0  # Normal
    elif severity_score <= 0.20:
        severity_stage = 1  # Mild
    elif severity_score <= 0.45:
        severity_stage = 2  # Moderate
    elif severity_score <= 0.70:
        severity_stage = 3  # Severe
    else:
        severity_stage = 4  # Failed

    sensor_values["severity_stage"] = int(severity_stage)
    sensor_values["anomaly_label"] = 1 if severity_stage >= 1 else 0

    # Compute hybrid TTF
    ttf_years, ttf_km = compute_hybrid_ttf(sensor_values, all_sensor_stats)
    sensor_values["TTF_years"] = ttf_years
    sensor_values["TTF_km"] = ttf_km

    # Convert to DataFrame
    return pd.DataFrame([sensor_values])

# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

@st.cache_resource
def load_trained_models() -> Dict[str, Any]:
    """
    Load all trained models with caching for performance.

    Returns:
        Dictionary of model_name -> loaded_model (or None if failed)
    """
    loaded_models = {}
    model_status = []

    for model_name, model_path in MODEL_PATHS.items():
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                loaded_models[model_name] = model
                model_status.append(f"✅ {model_name}: Loaded successfully")
            else:
                loaded_models[model_name] = None
                model_status.append(f"⚠️ {model_name}: Model file not found")
        except Exception as e:
            loaded_models[model_name] = None
            model_status.append(f"❌ {model_name}: Failed to load - {str(e)[:50]}...")

    # Display status in sidebar instead of main area
    with st.sidebar:
        st.subheader("🤖 Model Status")
        for status in model_status:
            if "✅" in status:
                st.success(status)
            elif "⚠️" in status:
                st.warning(status)
            else:
                st.error(status)

    return loaded_models

def predict_with_models(df_row: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on all available models.

    Args:
        df_row: Single-row DataFrame with sensor data
        models: Dictionary of loaded models

    Returns:
        Dictionary of model predictions
    """
    predictions = {}

    # Remove target columns for prediction
    target_cols = ['anomaly_label', 'severity_score', 'severity_stage', 'TTF_km', 'TTF_years', 'source', 'failure_type', 'is_gray']
    feature_cols = [col for col in df_row.columns if col not in target_cols]
    X = df_row[feature_cols]

    for model_name, model in models.items():
        if model is None:
            predictions[model_name] = None
            continue

        try:
            # Use only feature columns for prediction
            pred = model.predict(X)
            predictions[model_name] = pred[0] if hasattr(pred, '__len__') else pred
        except Exception as e:
            predictions[model_name] = f"Error: {str(e)}"
            st.error(f"❌ Prediction failed for {model_name}: {str(e)[:100]}...")

    return predictions

# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'accumulated_samples' not in st.session_state:
        st.session_state.accumulated_samples = []
    if 'sensor_stats' not in st.session_state:
        st.session_state.sensor_stats = load_dataset_stats()  # Load ALL sensor stats
    if 'models' not in st.session_state:
        st.session_state.models = load_trained_models()

def render_header():
    """Render the main header and description."""
    # Note: set_page_config moved to main() function to avoid multiple calls

    st.title("🔬 Catalytic Converter Live Fault Simulator")
    st.markdown("""
    **Real-time physics-based simulation** of catalytic converter degradation using your trained ML models.
    Adjust the severity slider to see how sensor values change across different fault conditions.
    """)

    # Display zone information
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, ((min_val, max_val), info) in enumerate(SEVERITY_ZONES.items()):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div style="background-color: {info['color']}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                <strong>{info['name']}</strong><br>
                {min_val}-{max_val}%<br>
                Stage {info['stage']}
            </div>
            """, unsafe_allow_html=True)

def render_severity_controls():
    """Render severity control slider and zone indicator."""
    st.sidebar.header("🎛️ Simulation Controls")

    # Main severity slider
    severity = st.sidebar.slider(
        "Fault Severity (%)",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        help="Adjust the catalytic converter degradation severity from 0% (normal) to 100% (complete failure)"
    )

    # Display current zone
    zone_info = get_severity_zone_info(severity)
    st.sidebar.markdown(f"""
    **Current Zone:** <span style="color: {zone_info['color']}; font-weight: bold;">{zone_info['name']}</span>
    **Severity Stage:** {zone_info['stage']}
    """, unsafe_allow_html=True)

    return severity

def render_sensor_values(df_row: pd.DataFrame, sensor_stats: Dict[str, Dict[str, float]]):
    """Render real-time sensor values in a formatted table."""
    st.subheader("📊 Real-time Sensor Values")

    # Create sensor display table
    sensor_data = []
    for sensor in KEY_SENSORS:
        if sensor in df_row.columns:
            current_val = df_row[sensor].iloc[0]
            baseline = sensor_stats[sensor]["mean"]
            min_val = sensor_stats[sensor]["min"]
            max_val = sensor_stats[sensor]["max"]

            # Calculate percentage change from baseline
            pct_change = ((current_val - baseline) / baseline) * 100 if baseline != 0 else 0

            # Determine status color based on percentage change
            if abs(pct_change) < 2:
                status = "🟢 Normal"
            elif abs(pct_change) < 8:
                status = "🟡 Mild"
            elif abs(pct_change) < 20:
                status = "🟠 Moderate"
            else:
                status = "🔴 Severe"

            sensor_data.append({
                "Sensor": sensor.replace(" ()_mean", "").replace(" ()_std", " (std)"),
                "Current Value": f"{current_val:.2f}",
                "Baseline": f"{baseline:.2f}",
                "Change %": f"{pct_change:+.1f}%",
                "Status": status,
                "Range": f"[{min_val:.0f} - {max_val:.0f}]"
            })

    # Display as table
    sensor_df = pd.DataFrame(sensor_data)
    st.dataframe(sensor_df, use_container_width=True)

def render_model_predictions(predictions: Dict[str, Any], ttf_km: float):
    """Render model predictions in a formatted display."""
    st.subheader("🤖 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classification Models:**")

        # Anomaly Detection
        anomaly_pred = predictions.get("anomaly", "N/A")
        if anomaly_pred is not None and anomaly_pred != "N/A":
            anomaly_status = "🔴 ANOMALY" if anomaly_pred == 1 else "🟢 NORMAL"
            st.markdown(f"**Anomaly Detection:** {anomaly_status}")
        else:
            st.markdown("**Anomaly Detection:** ❌ Model not available")

        # Severity Stage
        stage_pred = predictions.get("severity_stage", "N/A")
        if stage_pred is not None and stage_pred != "N/A":
            stage_names = ["Normal", "Mild", "Moderate", "Severe", "Failed"]
            stage_name = stage_names[int(stage_pred)] if 0 <= int(stage_pred) <= 4 else "Unknown"
            st.markdown(f"**Severity Stage:** Stage {stage_pred} ({stage_name})")
        else:
            st.markdown("**Severity Stage:** ❌ Model not available")

    with col2:
        st.markdown("**Regression Models:**")

        # Severity Score
        score_pred = predictions.get("severity_score", "N/A")
        if score_pred is not None and score_pred != "N/A":
            st.markdown(f"**Severity Score:** {score_pred:.3f}")
        else:
            st.markdown("**Severity Score:** ❌ Model not available")

        # TTF Prediction
        ttf_pred = predictions.get("ttf_km", "N/A")
        if ttf_pred is not None and ttf_pred != "N/A":
            st.markdown(f"**TTF (Model):** {ttf_pred:,.0f} km")
        else:
            st.markdown("**TTF (Model):** ❌ Model not available")

        # Physics-based TTF
        st.markdown(f"**TTF (Physics):** {ttf_km:,.0f} km")

def render_sample_accumulation():
    """Render sample accumulation controls and table."""
    st.subheader("📋 Sample Accumulation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Add Current Sample", type="primary"):
            return "add_sample"

    with col2:
        if st.button("🗑️ Clear All Samples"):
            return "clear_samples"

    with col3:
        if st.button("📤 Export to CSV"):
            return "export_csv"

    with col4:
        max_samples = st.number_input("Max Samples", min_value=10, max_value=1000, value=100)
        st.session_state.max_samples = max_samples

    # Display accumulated samples count
    sample_count = len(st.session_state.accumulated_samples)
    st.markdown(f"**Accumulated Samples:** {sample_count} / {getattr(st.session_state, 'max_samples', 100)}")

    # Display sample table if samples exist
    if sample_count > 0:
        samples_df = pd.DataFrame(st.session_state.accumulated_samples)

        # Show summary statistics
        st.markdown("**Sample Distribution:**")
        if 'severity_stage' in samples_df.columns:
            stage_counts = samples_df['severity_stage'].value_counts().sort_index()
            stage_names = ["Normal", "Mild", "Moderate", "Severe", "Failed"]

            summary_data = []
            for stage, count in stage_counts.items():
                stage_name = stage_names[stage] if 0 <= stage <= 4 else f"Stage {stage}"
                summary_data.append({"Stage": f"{stage} ({stage_name})", "Count": count})

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

        # Show recent samples
        st.markdown("**Recent Samples (Last 10):**")
        display_cols = ["severity_score", "severity_stage", "anomaly_label", "TTF_km"] + KEY_SENSORS[:4]
        recent_samples = samples_df[display_cols].tail(10)
        st.dataframe(recent_samples, use_container_width=True)

    return None

def handle_sample_actions(action: str, current_sample: pd.DataFrame):
    """Handle sample accumulation actions."""
    if action == "add_sample":
        max_samples = getattr(st.session_state, 'max_samples', 100)

        # Add current sample
        sample_dict = current_sample.iloc[0].to_dict()
        st.session_state.accumulated_samples.append(sample_dict)

        # Limit number of samples
        if len(st.session_state.accumulated_samples) > max_samples:
            st.session_state.accumulated_samples = st.session_state.accumulated_samples[-max_samples:]

        st.success(f"✅ Sample added! Total: {len(st.session_state.accumulated_samples)}")

    elif action == "clear_samples":
        st.session_state.accumulated_samples = []
        st.success("🗑️ All samples cleared!")

    elif action == "export_csv":
        if len(st.session_state.accumulated_samples) > 0:
            samples_df = pd.DataFrame(st.session_state.accumulated_samples)

            # Save to CSV in parent directory (for main.py to access)
            output_path = os.path.join(PARENT_DIR, "Simulated_Runtime.csv")

            # Check if file exists and ask user preference
            append_mode = False
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                st.info(f"📄 Found existing file with {len(existing_df)} rows")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Append to existing"):
                        append_mode = True
                with col2:
                    if st.button("🆕 Overwrite file"):
                        append_mode = False

                if not (st.session_state.get('export_decision_made', False)):
                    st.warning("⚠️ Please choose: Append or Overwrite")
                    return

            # Export the data
            if append_mode and os.path.exists(output_path):
                # Append mode
                samples_df.to_csv(output_path, mode='a', header=False, index=False)
                existing_count = len(pd.read_csv(output_path))
                st.success(f"📤 Appended {len(samples_df)} samples. Total: {existing_count} rows")
            else:
                # Overwrite mode
                samples_df.to_csv(output_path, index=False)
                st.success(f"📤 Exported {len(samples_df)} samples to {output_path}")

            # Provide download link
            csv_data = samples_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name="Simulated_Runtime.csv",
                mime="text/csv"
            )

            # Show export summary
            st.info(f"""
            **Export Summary:**
            - File: `{output_path}`
            - Samples: {len(samples_df)}
            - Columns: {len(samples_df.columns)}
            - Ready for main.py testing!
            """)

        else:
            st.warning("⚠️ No samples to export!")

def render_ttf_visualization(accumulated_samples: List[Dict]):
    """Render TTF visualization chart."""
    if len(accumulated_samples) < 2:
        return

    st.subheader("📉 TTF Degradation Visualization")

    samples_df = pd.DataFrame(accumulated_samples)

    # Create TTF vs Severity plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("TTF vs Severity Score", "Severity Distribution"),
        vertical_spacing=0.1
    )

    # TTF scatter plot
    fig.add_trace(
        go.Scatter(
            x=samples_df['severity_score'],
            y=samples_df['TTF_km'],
            mode='markers',
            marker=dict(
                size=8,
                color=samples_df['severity_stage'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Severity Stage")
            ),
            name="TTF vs Severity"
        ),
        row=1, col=1
    )

    # Severity histogram
    fig.add_trace(
        go.Histogram(
            x=samples_df['severity_score'],
            nbinsx=20,
            name="Severity Distribution",
            marker_color='lightblue'
        ),
        row=2, col=1
    )

    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Severity Score", row=1, col=1)
    fig.update_yaxes(title_text="TTF (km)", row=1, col=1)
    fig.update_xaxes(title_text="Severity Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application function."""
    # Set page config first (can only be called once)
    st.set_page_config(
        page_title="Catalytic Converter Simulator",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Render header
    render_header()

    # Render severity controls
    severity = render_severity_controls()

    # Generate current sensor row
    current_sample = generate_sensor_row(severity, st.session_state.sensor_stats)

    # Get model predictions
    predictions = predict_with_models(current_sample, st.session_state.models)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Render sensor values
        render_sensor_values(current_sample, st.session_state.sensor_stats)

        # Render model predictions
        ttf_km = current_sample['TTF_km'].iloc[0]
        render_model_predictions(predictions, ttf_km)

    with col2:
        # Render sample accumulation
        action = render_sample_accumulation()

        # Handle actions
        if action:
            handle_sample_actions(action, current_sample)
            st.rerun()

    # Render TTF visualization if samples exist
    if len(st.session_state.accumulated_samples) > 1:
        render_ttf_visualization(st.session_state.accumulated_samples)

    # Footer
    st.markdown("---")
    st.markdown("""
    **🔬 Physics-Based Simulation:** Uses your exact three-zone sensor transform formulas
    **🤖 Model Integration:** Real-time inference with your trained XGBoost, Stacking, and Gradient Boosting models
    **📊 Data Export:** Accumulate samples and export to CSV for batch model testing
    """)

if __name__ == "__main__":
    main()
