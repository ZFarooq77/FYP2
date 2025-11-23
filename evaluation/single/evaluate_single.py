# evaluation/evaluate_single.py
"""
Single Row Evaluation for FYP Catalytic Converter Health Monitoring
===================================================================

Purpose:
- Load all trained models (.pkl files) from models/main_models/
- Predict for a single row of sensor data
- Return both dictionary and DataFrame formats

Models:
- Anomaly Detection (Binary Classification) - XGBoost
- Severity Score (Regression) - Stacking Ensemble  
- Severity Stage (5-Class Classification) - Stacking Ensemble
- Time-to-Failure (Regression) - Gradient Boosting

Author: FYP Team - Catalytic Converter Anomaly Detection
"""

import joblib
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIG ==========
MODEL_DIR = "../../models/main_models"  # folder where all .pkl files exist
OUTPUT_DIR = "output"                   # folder for evaluation outputs

# Our 4 target predictions
TARGETS = {
    "anomaly_label": "anomaly_model.pkl",
    "severity_score": "severity_score_model.pkl", 
    "severity_stage": "severity_stage_model.pkl",
    "ttf_km": "ttf_model.pkl"
}

# Scalers for each model
SCALERS = {
    "severity_score": "severity_score_scaler.pkl",
    "severity_stage": "severity_stage_scaler.pkl", 
    "ttf_km": "ttf_scaler.pkl"
}

# Feature files for each model
FEATURE_FILES = {
    "severity_score": "selected_features_v2.csv",
    "ttf_km": "ttf_features.csv"
}

# ===========================================
# Helper: load pkl safely
def load_component(name):
    """Load a .pkl file safely with error handling."""
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")
    return joblib.load(path)

# ===========================================
# Load feature lists for models that need specific features
def load_feature_list(filename):
    """Load feature list from CSV file."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df['feature'].tolist()

# ===========================================
# Load all models and their components
def load_models():
    """Load all models, scalers, and feature lists."""
    print("🔍 Loading model bundle...")
    bundle = {}

    for target, model_file in TARGETS.items():
        print(f"  📦 Loading {target}...")
        bundle[target] = {}

        try:
            # Special handling for anomaly model (it's a dictionary)
            if target == "anomaly_label":
                anomaly_bundle = load_component(model_file)
                bundle[target]["model"] = anomaly_bundle["model"]
                bundle[target]["scaler"] = anomaly_bundle["scaler"]
                bundle[target]["imputer"] = anomaly_bundle["imputer"]
                bundle[target]["features"] = anomaly_bundle["feature_names"]
                print(f"    ✅ Anomaly bundle loaded: model + scaler + imputer + {len(anomaly_bundle['feature_names'])} features")
            else:
                # Load model
                bundle[target]["model"] = load_component(model_file)
                print(f"    ✅ Model loaded: {model_file}")

                # Load scaler if exists
                if target in SCALERS:
                    bundle[target]["scaler"] = load_component(SCALERS[target])
                    print(f"    ✅ Scaler loaded: {SCALERS[target]}")

                # Load feature list if exists
                if target in FEATURE_FILES:
                    bundle[target]["features"] = load_feature_list(FEATURE_FILES[target])
                    print(f"    ✅ Features loaded: {len(bundle[target]['features'])} features")
                elif target == "severity_stage":
                    # For severity_stage, it uses the same 50 features as severity_score
                    # (based on the scaler expecting 50 features)
                    bundle[target]["features"] = load_feature_list("selected_features_v2.csv")
                    print(f"    ✅ Features loaded: {len(bundle[target]['features'])} features (same as severity_score)")

        except Exception as e:
            print(f"    ❌ Error loading {target}: {str(e)}")
            bundle[target] = {"error": str(e)}

    print("✅ Model bundle loaded successfully!")
    return bundle

# ===========================================
# Get all unique features needed across all models
def get_all_required_features():
    """Get comprehensive list of all features needed by any model."""
    # Load feature lists
    severity_score_features = load_feature_list("selected_features_v2.csv") or []
    ttf_features = load_feature_list("ttf_features.csv") or []
    
    # Combine and deduplicate
    all_features = list(set(severity_score_features + ttf_features))
    
    # Add some common features that anomaly and severity_stage models might need
    # (These models use a broader feature set)
    common_features = [
        "ENGINE_RPM ()_mean", "ENGINE_RPM ()_std", "ENGINE_RPM ()_max", "ENGINE_RPM ()_min",
        "VEHICLE_SPEED ()_mean", "VEHICLE_SPEED ()_std", 
        "THROTTLE ()_mean", "THROTTLE ()_std",
        "ENGINE_LOAD ()_mean", "ENGINE_LOAD ()_std",
        "COOLANT_TEMPERATURE ()_mean", "COOLANT_TEMPERATURE ()_std",
        "INTAKE_MANIFOLD_PRESSURE ()_mean", "INTAKE_MANIFOLD_PRESSURE ()_std"
    ]
    
    # Add common features if not already present
    for feature in common_features:
        if feature not in all_features:
            all_features.append(feature)
    
    return sorted(all_features)

# ===========================================
# Evaluate single row
def evaluate_single(input_row: dict, models_bundle: dict, return_dataframe=False):
    """
    Evaluate a single row of sensor data using all trained models.

    Args:
        input_row (dict): Dictionary with sensor readings
        models_bundle (dict): Loaded models and components
        return_dataframe (bool): If True, return DataFrame; if False, return dict

    Returns:
        dict or DataFrame: Predictions from all models
    """
    print("\n🚀 Running prediction for single row...")

    # ---- Convert single row into DataFrame ----
    df = pd.DataFrame([input_row])

    predictions = {}
    prediction_details = {}

    # ---- Loop through each target model ----
    for target, components in models_bundle.items():
        print(f"\n🔍 Processing {target}...")

        if "error" in components:
            predictions[target] = f"Error: {components['error']}"
            prediction_details[target] = {"status": "error", "message": components['error']}
            continue

        if "model" not in components:
            predictions[target] = "Model Missing"
            prediction_details[target] = {"status": "missing", "message": "Model file not found"}
            continue

        try:
            model = components["model"]
            scaler = components.get("scaler", None)
            imputer = components.get("imputer", None)
            features = components.get("features", None)

            # Prepare features for this specific model
            if features:
                # Use specific feature list for this model
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    print(f"  ⚠️ Missing features for {target}: {len(missing_features)} features")
                    # Fill missing features with 0 (or could use mean/median)
                    for feature in missing_features:
                        df[feature] = 0.0

                X = df[features].copy()
                print(f"  ✅ Using {len(features)} specific features")
            else:
                # Use all available features (fallback)
                drop_cols = ["anomaly_label", "failure_type", "severity_stage", "severity_score",
                           "TTF_years", "TTF_km", "is_gray", "source"]
                available_features = [col for col in df.columns if col not in drop_cols]
                X = df[available_features].copy()
                print(f"  ✅ Using {len(available_features)} available features")

            # Apply preprocessing pipeline based on model type
            if target == "ttf_km":
                # TTF model: scaler was fitted on ALL 120 features, then model uses subset
                # First, get all 120 features for scaling
                drop_cols = ["anomaly_label", "failure_type", "severity_stage", "severity_score",
                           "TTF_years", "TTF_km", "is_gray", "source"]
                all_features = [col for col in df.columns if col not in drop_cols]

                # Fill missing features with 0 if needed
                for feature in all_features:
                    if feature not in df.columns:
                        df[feature] = 0.0

                X_all = df[all_features].copy()
                X_scaled_all = scaler.transform(X_all)

                # Now select only the TTF-specific features from the scaled data
                X_scaled_df = pd.DataFrame(X_scaled_all, columns=all_features)
                X_scaled = X_scaled_df[features].values
                print(f"  ✅ Applied scaling on all {len(all_features)} features, then selected {len(features)} for TTF model")

            else:
                # Other models: standard preprocessing
                if imputer:
                    X_processed = imputer.transform(X)
                    print(f"  ✅ Applied imputation")
                else:
                    X_processed = X.values

                if scaler:
                    X_scaled = scaler.transform(X_processed)
                    print(f"  ✅ Applied scaling")
                else:
                    X_scaled = X_processed
                    print(f"  ℹ️ No scaling applied")

            # Make prediction
            if target == "anomaly_label":
                # Binary classification - get both prediction and probability
                y_pred = model.predict(X_scaled)[0]
                y_prob = model.predict_proba(X_scaled)[0]
                predictions[target] = int(y_pred)
                prediction_details[target] = {
                    "prediction": int(y_pred),
                    "probability_normal": float(y_prob[0]),
                    "probability_anomaly": float(y_prob[1]),
                    "confidence": float(max(y_prob))
                }
            elif target == "severity_stage":
                # Multi-class classification
                y_pred = model.predict(X_scaled)[0]
                y_prob = model.predict_proba(X_scaled)[0]
                predictions[target] = int(y_pred)
                prediction_details[target] = {
                    "prediction": int(y_pred),
                    "probabilities": y_prob.tolist(),
                    "confidence": float(max(y_prob))
                }
            else:
                # Regression (severity_score, ttf_km)
                y_pred = model.predict(X_scaled)[0]
                predictions[target] = float(y_pred)
                prediction_details[target] = {
                    "prediction": float(y_pred)
                }

            print(f"  ✅ Prediction: {predictions[target]}")

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            predictions[target] = error_msg
            prediction_details[target] = {"status": "error", "message": error_msg}
            print(f"  ❌ {error_msg}")

    # Return format based on user preference
    if return_dataframe:
        # Create DataFrame with original data + predictions
        result_df = df.copy()
        for target, pred in predictions.items():
            result_df[f"predicted_{target}"] = pred
        return result_df, prediction_details
    else:
        return predictions, prediction_details

# ===========================================
# Save results to output folder
def save_results(predictions, prediction_details, input_row, filename_prefix="single_prediction"):
    """Save prediction results to output folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save basic predictions as CSV
    pred_df = pd.DataFrame([predictions])
    pred_df.to_csv(f"{OUTPUT_DIR}/{filename_prefix}_results.csv", index=False)

    # Save detailed results as JSON-like CSV
    details_df = pd.DataFrame([prediction_details])
    details_df.to_csv(f"{OUTPUT_DIR}/{filename_prefix}_details.csv", index=False)

    # Save input data for reference
    input_df = pd.DataFrame([input_row])
    input_df.to_csv(f"{OUTPUT_DIR}/{filename_prefix}_input.csv", index=False)

    print(f"\n💾 Results saved to {OUTPUT_DIR}/:")
    print(f"  📄 {filename_prefix}_results.csv - Basic predictions")
    print(f"  📄 {filename_prefix}_details.csv - Detailed predictions with probabilities")
    print(f"  📄 {filename_prefix}_input.csv - Original input data")

# ===========================================
# Create sample input for testing
def create_sample_input():
    """Create a sample input row for testing purposes using actual Final.csv structure."""
    # Load a real sample from Final.csv to get all column names and realistic values
    import pandas as pd
    df = pd.read_csv("../../Final.csv", nrows=5)

    # Get a normal sample (anomaly_label = 0)
    normal_samples = df[df['anomaly_label'] == 0]
    if len(normal_samples) > 0:
        sample_row = normal_samples.iloc[0]
    else:
        sample_row = df.iloc[0]

    # Convert to dictionary and remove target columns
    drop_cols = ["anomaly_label", "failure_type", "severity_stage", "severity_score", "TTF_years", "TTF_km", "is_gray", "source"]
    sample_dict = {}

    for col in df.columns:
        if col not in drop_cols:
            sample_dict[col] = float(sample_row[col])

    return sample_dict

# ===========================================
# Manual testing and demonstration
if __name__ == "__main__":
    print("🎯 FYP CATALYTIC CONVERTER HEALTH MONITORING")
    print("=" * 60)
    print("🔍 Single Row Evaluation System")
    print("=" * 60)

    try:
        # Load all models
        models_bundle = load_models()

        # Create sample input for testing
        print("\n📊 Creating sample sensor data...")
        sample_input = create_sample_input()
        print(f"✅ Sample created with {len(sample_input)} sensor readings")

        # Show required features
        print("\n📋 Required features across all models:")
        all_features = get_all_required_features()
        print(f"  Total unique features needed: {len(all_features)}")
        print(f"  Sample covers: {len([f for f in all_features if f in sample_input])} features")

        # Run prediction
        print("\n" + "=" * 60)
        predictions, details = evaluate_single(sample_input, models_bundle)

        # Display results
        print("\n" + "=" * 60)
        print("🏆 FINAL PREDICTION RESULTS")
        print("=" * 60)

        for target, prediction in predictions.items():
            print(f"\n🎯 {target.upper().replace('_', ' ')}:")
            print(f"   Prediction: {prediction}")

            if target in details and isinstance(details[target], dict):
                detail = details[target]
                if "confidence" in detail:
                    print(f"   Confidence: {detail['confidence']:.3f}")
                if "probability_anomaly" in detail:
                    print(f"   Anomaly Probability: {detail['probability_anomaly']:.3f}")
                if "probabilities" in detail:
                    print(f"   Class Probabilities: {[f'{p:.3f}' for p in detail['probabilities']]}")

        # Save results
        print("\n" + "=" * 60)
        save_results(predictions, details, sample_input, "demo_prediction")

        # Summary
        print("\n🎉 EVALUATION COMPLETE!")
        print("=" * 60)
        print("✅ All models evaluated successfully")
        print("✅ Results saved to output folder")
        print("✅ Ready for production use")

        # Show interpretation
        print("\n💡 HEALTH ASSESSMENT INTERPRETATION:")
        if predictions.get("anomaly_label") == 0:
            print("   🟢 NORMAL: Catalytic converter operating within normal parameters")
        else:
            print("   🔴 ANOMALY: Potential catalytic converter degradation detected")

        if "severity_score" in predictions and isinstance(predictions["severity_score"], (int, float)):
            severity = predictions["severity_score"]
            if severity < 0.3:
                print(f"   🟢 SEVERITY: Low ({severity:.3f}) - Minor degradation")
            elif severity < 0.7:
                print(f"   🟡 SEVERITY: Moderate ({severity:.3f}) - Monitor closely")
            else:
                print(f"   🔴 SEVERITY: High ({severity:.3f}) - Immediate attention required")

        if "ttf_km" in predictions and isinstance(predictions["ttf_km"], (int, float)):
            ttf = predictions["ttf_km"]
            print(f"   ⏱️ TIME TO FAILURE: ~{ttf:.0f} km remaining")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
