"""
main.py — FYP Model Training Master Orchestrator 🚀
--------------------------------------------------
Purpose:
    Orchestrates all 4 target model training scripts with smart caching,
    graceful error handling, and unified performance reporting.

Features:
    ✅ Smart caching (skip retraining if models exist)
    ✅ Structured return format with metrics
    ✅ Graceful error handling (continues if one module fails)
    ✅ Professional summary reporting
    ✅ Comprehensive logging
    output saved in main_outputs/ directory simulated_runtime.csv
"""

import os
import sys
import logging
import datetime
import pandas as pd
import time
import joblib
import numpy as np
from pathlib import Path

# Add modules directory to Python path
sys.path.append('modules')

# -------------------------------
# 🧠 CONFIGURATION
# -------------------------------
MODEL_MODULES = [
    {
        "display_name": "Anomaly Detection (Binary Classification)",
        "module_name": "train_anomaly",
        "function_name": "main_anomaly_label_model",
        "model_path": "models/main_models/anomaly_model.pkl",
        "summary_path": "main_outputs/output_anomaly_label.csv"
    },
    {
        "display_name": "Severity Score Regression",
        "module_name": "train_severity_score",
        "function_name": "main_severity_score_model",
        "model_path": "models/main_models/severity_score_model.pkl",
        "summary_path": "main_outputs/output_severity_score.csv"
    },
    {
        "display_name": "Severity Stage Classification (5-Class)",
        "module_name": "train_severity_stage",
        "function_name": "main_severity_stage_model",
        "model_path": "models/main_models/severity_stage_model.pkl",
        "summary_path": "main_outputs/output_severity_stage.csv"
    },
    {
        "display_name": "Time-to-Failure (TTF) Regression",
        "module_name": "train_ttf_km",
        "function_name": "main_ttf_km_model",
        "model_path": "models/main_models/ttf_model.pkl",
        "summary_path": "main_outputs/output_ttf.csv"
    }
]

# Create main_outputs directory
MAIN_OUTPUTS_DIR = "main_outputs"
os.makedirs(MAIN_OUTPUTS_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(MAIN_OUTPUTS_DIR, "unified_model_performance_summary.csv")
LOG_FILE = os.path.join(MAIN_OUTPUTS_DIR, "main_pipeline.log")

# -------------------------------
# 🪵 LOGGER SETUP
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# 🔧 HELPER FUNCTIONS
# -------------------------------

def is_cached(model_path, summary_path):
    """Check if both model and summary files exist."""
    model_exists = os.path.exists(model_path)
    summary_exists = os.path.exists(summary_path)

    if model_exists and summary_exists:
        logger.info(f"🟡 Cached files found: {os.path.basename(model_path)} + {os.path.basename(summary_path)}")
        return True
    elif model_exists:
        logger.warning(f"⚠️ Model exists but summary missing: {model_path}")
        return False
    elif summary_exists:
        logger.warning(f"⚠️ Summary exists but model missing: {summary_path}")
        return False
    else:
        logger.info(f"🔄 No cached files found for {os.path.basename(model_path)}")
        return False

def load_cached_metrics(summary_path, model_name):
    """Load metrics from cached summary file."""
    try:
        if summary_path.endswith('.csv'):
            df = pd.read_csv(summary_path)
            if len(df) > 0:
                # Extract metrics from the best performing row
                best_row = df.iloc[0] if len(df) == 1 else df.iloc[-1]  # Last row often contains best model
                metrics = best_row.to_dict()
                return {
                    "status": "cached",
                    "model_name": model_name,
                    "model_path": "cached",
                    "summary_path": summary_path,
                    "metrics": metrics,
                    "training_time": 0.0,
                    "timestamp": datetime.datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"❌ Failed to load cached metrics from {summary_path}: {str(e)}")

    return {
        "status": "cached",
        "model_name": model_name,
        "model_path": "cached",
        "summary_path": summary_path,
        "metrics": {"note": "cached_model_available"},
        "training_time": 0.0,
        "timestamp": datetime.datetime.now().isoformat()
    }

def save_unified_summary(all_results):
    """Save a unified summary CSV for all models."""
    summary_data = []

    for result in all_results:
        row = {
            "Model_Name": result.get("model_name", "Unknown"),
            "Status": result.get("status", "unknown"),
            "Model_Path": result.get("model_path", "N/A"),
            "Training_Time_Seconds": result.get("training_time", 0.0),
            "Timestamp": result.get("timestamp", "N/A")
        }

        # Add metrics
        metrics = result.get("metrics", {})
        for key, value in metrics.items():
            row[f"Metric_{key}"] = value

        # Add error if failed
        if result.get("status") == "failed":
            row["Error"] = result.get("error", "Unknown error")

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(SUMMARY_FILE, index=False)
    logger.info(f"📊 Unified summary saved to {SUMMARY_FILE}")

def print_training_summary(all_results):
    """Print a professional summary table to console."""
    print("\n" + "="*100)
    print("🎯 MODEL TRAINING PIPELINE SUMMARY")
    print("="*100)
    print(f"{'Model':<35} {'Status':<12} {'Performance':<25} {'Time':<10}")
    print("-"*100)

    success_count = 0
    total_time = 0.0

    for result in all_results:
        model_name = result.get("model_name", "Unknown")[:34]
        status = result.get("status", "unknown")
        training_time = result.get("training_time", 0.0)
        total_time += training_time

        if status == "success":
            success_count += 1
            status_icon = "✅ Success"
            metrics = result.get("metrics", {})

            # Extract key performance metric
            if "accuracy" in metrics:
                perf = f"Accuracy: {metrics['accuracy']:.2%}"
            elif "Test_R2" in metrics:
                perf = f"R²: {metrics['Test_R2']:.4f}"
            elif "r2_score" in metrics:
                perf = f"R²: {metrics['r2_score']:.4f}"
            else:
                perf = "Metrics available"

        elif status == "cached":
            status_icon = "🟡 Cached"
            perf = "Using existing model"
        else:
            status_icon = "❌ Failed"
            perf = result.get("error", "Unknown error")[:24]

        time_str = f"{training_time:.1f}s"
        print(f"{model_name:<35} {status_icon:<12} {perf:<25} {time_str:<10}")

    print("-"*100)
    print(f"🎉 PIPELINE COMPLETED: {success_count}/{len(all_results)} models successful")
    print(f"⏱️ Total execution time: {total_time:.1f} seconds")
    print(f"📊 Detailed summary: {SUMMARY_FILE}")
    print(f"📋 Full logs: {LOG_FILE}")
    print("="*100)

# -------------------------------
# 🚀 MAIN EXECUTION FUNCTION
# -------------------------------

def run_model_training(config):
    """
    Execute training for a single model with caching and error handling.
    """
    display_name = config["display_name"]
    module_name = config["module_name"]
    function_name = config["function_name"]
    model_path = config["model_path"]
    summary_path = config["summary_path"]

    logger.info(f"🔹 Processing: {display_name}")

    # Check cache first
    if is_cached(model_path, summary_path):
        logger.info(f"🟡 Using cached model for {display_name}")
        return load_cached_metrics(summary_path, display_name)

    # Train from scratch
    logger.info(f"🔄 Training {display_name} from scratch...")
    start_time = time.time()

    try:
        # Import module and get function
        module = __import__(module_name)
        if not hasattr(module, function_name):
            raise AttributeError(f"Module {module_name} missing function `{function_name}()`")

        train_function = getattr(module, function_name)

        # Execute training
        result = train_function()

        # Add timing and timestamp
        training_time = time.time() - start_time
        result["training_time"] = training_time
        result["timestamp"] = datetime.datetime.now().isoformat()

        logger.info(f"✅ {display_name} completed successfully in {training_time:.1f}s")
        return result

    except Exception as e:
        training_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ {display_name} failed after {training_time:.1f}s: {error_msg}")

        return {
            "status": "failed",
            "model_name": display_name,
            "error": error_msg,
            "training_time": training_time,
            "timestamp": datetime.datetime.now().isoformat()
        }

# -------------------------------
# 🏁 TRAINING EXECUTION FUNCTION
# -------------------------------

def run_training_pipeline():
    """Execute the complete training pipeline."""
    print("🚀 FYP MODEL TRAINING PIPELINE STARTED")
    print(f"📁 Outputs will be saved to: {MAIN_OUTPUTS_DIR}/")

    logger.info("🚀 MAIN PIPELINE STARTED")
    logger.info(f"📊 Dataset: Final.csv")
    logger.info(f"🎯 Training {len(MODEL_MODULES)} models")

    all_results = []
    pipeline_start_time = time.time()

    # Execute each model training
    for config in MODEL_MODULES:
        result = run_model_training(config)
        all_results.append(result)

    # Save unified summary
    save_unified_summary(all_results)

    # Print final summary
    total_pipeline_time = time.time() - pipeline_start_time
    print_training_summary(all_results)

    logger.info(f"🎯 MAIN PIPELINE COMPLETED in {total_pipeline_time:.1f} seconds")
    print(f"\n🎉 Pipeline completed! Check {MAIN_OUTPUTS_DIR}/ for detailed results.")

    return all_results

# =============================================================================
# 🧪 RUNTIME MODEL TESTING FUNCTIONS
# =============================================================================

def run_model_inference_on_runtime_data(runtime_file: str = "Simulated_Runtime.csv") -> dict:
    """
    Load simulated runtime dataset and run inference using trained models.

    Args:
        runtime_file: Path to the simulated runtime CSV file

    Returns:
        Dictionary with inference results and metrics
    """
    print("\n" + "="*80)
    print("🧪 RUNTIME MODEL TESTING - Simulated Data Inference")
    print("="*80)

    # Check if runtime file exists
    if not os.path.exists(runtime_file):
        print(f"❌ No {runtime_file} found. Run simulator first!")
        return {"status": "error", "message": f"File {runtime_file} not found"}

    # Load simulated data
    try:
        df = pd.read_csv(runtime_file)
        print(f"📥 Loaded {len(df)} simulated rows for evaluation")
        print(f"📊 Columns: {len(df.columns)} | Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading {runtime_file}: {e}")
        return {"status": "error", "message": f"Failed to load file: {e}"}

    # Model paths and names
    model_configs = {
        "Anomaly_Detection": {
            "path": "models/best_anomaly_model_xgboost.pkl",
            "type": "classification",
            "target": "anomaly_label"
        },
        "Severity_Score": {
            "path": "models/severity_stacking_model_v2.pkl",
            "type": "regression",
            "target": "severity_score"
        },
        "Severity_Stage": {
            "path": "models/severity_stage_stacking_model_v1.pkl",
            "type": "classification",
            "target": "severity_stage"
        },
        "TTF_Prediction": {
            "path": "models/best_ttf_model_gradient_boosting.pkl",
            "type": "regression",
            "target": "TTF_km"
        }
    }

    # Load available models
    loaded_models = {}
    print(f"\n🤖 Loading trained models...")

    for name, config in model_configs.items():
        try:
            if os.path.exists(config["path"]):
                model = joblib.load(config["path"])
                loaded_models[name] = {"model": model, "config": config}
                print(f"✅ {name}: Loaded successfully")
            else:
                print(f"⚠️ {name}: Model file not found - {config['path']}")
        except Exception as e:
            print(f"❌ {name}: Failed to load - {str(e)[:50]}...")

    if not loaded_models:
        print("❌ No models could be loaded!")
        return {"status": "error", "message": "No models available"}

    # Prepare features (exclude target columns and metadata)
    exclude_cols = ["severity_score", "severity_stage", "anomaly_label", "TTF_km", "TTF_years"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n🔧 Feature preparation:")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"🎯 Target columns excluded: {len(exclude_cols)}")
    print(f"📈 Feature columns: {len(feature_cols)}")

    # Run predictions
    results = {}
    print(f"\n🔮 Running model predictions...")

    for name, model_info in loaded_models.items():
        model = model_info["model"]
        config = model_info["config"]

        try:
            # Try prediction with available features
            X = df[feature_cols].select_dtypes(include=[np.number])
            predictions = model.predict(X)

            # Store predictions
            df[f"{name}_Prediction"] = predictions
            results[name] = {
                "predictions": predictions,
                "count": len(predictions),
                "type": config["type"],
                "status": "success"
            }

            # Calculate basic statistics
            if config["type"] == "regression":
                results[name]["stats"] = {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            else:  # classification
                unique, counts = np.unique(predictions, return_counts=True)
                results[name]["stats"] = {
                    "classes": unique.tolist(),
                    "counts": counts.tolist(),
                    "distribution": dict(zip(unique.astype(str), counts))
                }

            print(f"✅ {name}: {len(predictions)} predictions computed")

        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            print(f"❌ {name}: Prediction failed - {str(e)[:50]}...")

    # Apply gating logic based on anomaly predictions
    try:
        if "Anomaly_Detection_Prediction" in df.columns:
            anomaly_pred = np.array(df["Anomaly_Detection_Prediction"]).astype(int)

            # Gate severity score predictions: only meaningful when anomaly = 1
            if "Severity_Score_Prediction" in df.columns:
                sev_score_raw = np.array(df["Severity_Score_Prediction"]).astype(float)
                sev_score_gated = np.where(anomaly_pred == 1, sev_score_raw, 0.0)
                df["Severity_Score_Prediction"] = sev_score_gated

            # Gate severity stage predictions: force Stage 0 when anomaly = 0
            if "Severity_Stage_Prediction" in df.columns:
                sev_stage_raw = np.array(df["Severity_Stage_Prediction"]).astype(int)
                sev_stage_gated = np.where(anomaly_pred == 1, sev_stage_raw, 0)
                df["Severity_Stage_Prediction"] = sev_stage_gated

            print("\n🔗 Applied gating logic: anomaly → severity_score → severity_stage")
        else:
            print("\n⚠️ Gating logic skipped: Anomaly_Detection_Prediction not available")
    except Exception as e:
        print(f"\n⚠️ Failed to apply gating logic: {e}")


    # Save predictions to new file
    output_path = "Simulated_Predictions.csv"
    try:
        df.to_csv(output_path, index=False)
        print(f"\n📊 Predictions saved → {output_path}")
    except Exception as e:
        print(f"⚠️ Failed to save predictions: {e}")

    # Print summary
    print(f"\n📋 INFERENCE SUMMARY:")
    print("-" * 50)

    for name, result in results.items():
        if result["status"] == "success":
            print(f"✅ {name}:")
            if result["type"] == "regression":
                stats = result["stats"]
                print(f"   Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f}")
                print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                print(f"   Classes: {result['stats']['distribution']}")
        else:
            print(f"❌ {name}: {result.get('error', 'Unknown error')}")

    # Show sample predictions
    if len(df) > 0:
        print(f"\n📊 Sample Predictions (First 5 rows):")
        display_cols = ["severity_score", "severity_stage"] + [col for col in df.columns if "_Prediction" in col]
        sample_df = df[display_cols].head()
        print(sample_df.to_string(index=False))

    return {
        "status": "success",
        "total_samples": len(df),
        "models_tested": len(loaded_models),
        "results": results,
        "output_file": output_path
    }

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="FYP Model Training & Testing Orchestrator")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="train",
                       help="Mode: 'train' (default), 'test' runtime data, or 'both'")
    parser.add_argument("--runtime-file", default="Simulated_Runtime.csv",
                       help="Path to runtime simulation CSV file (default: Simulated_Runtime.csv)")

    args = parser.parse_args()

    print("🚀 FYP Model Training & Testing Orchestrator")
    print("=" * 60)

    if args.mode in ["train", "both"]:
        print("🏋️ Starting model training pipeline...")
        run_training_pipeline()

    if args.mode in ["test", "both"]:
        print("\n🧪 Starting runtime model testing...")
        test_results = run_model_inference_on_runtime_data(args.runtime_file)

        if test_results["status"] == "success":
            print(f"\n🎉 Runtime testing completed successfully!")
            print(f"📊 Tested {test_results['models_tested']} models on {test_results['total_samples']} samples")
        else:
            print(f"\n❌ Runtime testing failed: {test_results.get('message', 'Unknown error')}")

    print("\n🎯 All operations completed!")
