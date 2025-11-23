#!/usr/bin/env python3
"""
🎯 FYP CATALYTIC CONVERTER HEALTH MONITORING - BATCH EVALUATION
============================================================
Evaluate all 4 trained models on the test split from Final.csv
- Apply SAME preprocessing as training
- Recreate train-test split (use ONLY test set)
- Load .pkl models and predict
- Generate comprehensive evaluation outputs
============================================================
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 🔧 CONFIGURATION
# ==========================================================
MODEL_DIR = "../../models/main_models"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Exact same DROP_COLS as training
DROP_COLS = [
    "anomaly_label", "failure_type", "severity_stage", "severity_score", 
    "TTF_years", "TTF_km", "is_gray", "source"
]

print("🎯 FYP CATALYTIC CONVERTER HEALTH MONITORING")
print("=" * 60)
print("🔍 Batch Evaluation System - Test Set Analysis")
print("=" * 60)

# ==========================================================
# 📁 UTILITY FUNCTIONS
# ==========================================================
def ensure_output_dirs():
    """Create all required output directories."""
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/confusion_matrices",
        f"{OUTPUT_DIR}/regression_plots", 
        f"{OUTPUT_DIR}/logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Output directories created")

def log_message(message, log_file="batch_evaluation.log"):
    """Log messages to file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = f"{OUTPUT_DIR}/logs/{log_file}"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

# ==========================================================
# 📊 DATA LOADING & PREPROCESSING
# ==========================================================
def load_and_preprocess_data():
    """
    Load Final.csv and apply EXACT same preprocessing as training.
    Returns test set only for evaluation.
    """
    log_message("🔍 Loading Final.csv...")
    
    # Load dataset
    df = pd.read_csv("../../Final.csv")
    log_message(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Apply same preprocessing as training modules
    log_message("🔄 Applying preprocessing...")
    
    # Remove rows with NaN in target columns (same as training)
    original_rows = len(df)
    df = df.dropna(subset=['anomaly_label', 'severity_score', 'severity_stage', 'TTF_km'])
    log_message(f"✅ Removed NaN targets: {original_rows} → {len(df)} rows")
    
    # Get feature columns (same logic as training)
    available_drop_cols = [col for col in DROP_COLS if col in df.columns]
    feature_cols = [col for col in df.columns if col not in available_drop_cols]
    
    # Keep only numeric features (same as training)
    X_full = df[feature_cols].select_dtypes(include=[np.number]).copy()
    log_message(f"✅ Features after preprocessing: {X_full.shape[1]} numeric columns")
    
    # Prepare targets
    targets = {
        'anomaly_label': df['anomaly_label'].copy(),
        'severity_score': df['severity_score'].copy(), 
        'severity_stage': df['severity_stage'].astype(int).copy(),
        'ttf_km': df['TTF_km'].copy()
    }
    
    log_message(f"✅ Target distributions:")
    log_message(f"   Anomaly: {targets['anomaly_label'].value_counts().to_dict()}")
    log_message(f"   Severity Score: {targets['severity_score'].min():.3f} - {targets['severity_score'].max():.3f}")
    log_message(f"   Severity Stage: {sorted(targets['severity_stage'].unique())}")
    log_message(f"   TTF (km): {targets['ttf_km'].min():.0f} - {targets['ttf_km'].max():.0f}")
    
    return X_full, targets, df

# ==========================================================
# 🔄 TRAIN-TEST SPLIT RECREATION
# ==========================================================
def recreate_train_test_split(X_full, targets):
    """
    Recreate the EXACT same train-test split as used in training.
    Returns only the test sets for evaluation.
    """
    log_message("🔄 Recreating train-test split (same as training)...")
    
    test_sets = {}
    
    # For each target, recreate the split used in training
    for target_name, y in targets.items():
        log_message(f"   Splitting {target_name}...")
        
        if target_name == 'severity_stage':
            # Use stratified split (same as training)
            _, X_test, _, y_test = train_test_split(
                X_full, y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE,
                stratify=y
            )
        else:
            # Regular split for other targets
            _, X_test, _, y_test = train_test_split(
                X_full, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )
        
        test_sets[target_name] = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        log_message(f"   ✅ {target_name}: {len(y_test)} test samples")
    
    return test_sets

# ==========================================================
# 🤖 MODEL LOADING
# ==========================================================
def load_models():
    """Load all trained models with exact same structure as single evaluation."""
    log_message("🔍 Loading trained models...")

    models_bundle = {}

    # Model configurations (same as single evaluation)
    model_configs = {
        "anomaly_label": {
            "model_file": "anomaly_model.pkl",
            "scaler_file": None,  # Bundled in model
            "features_file": None  # Bundled in model
        },
        "severity_score": {
            "model_file": "severity_score_model.pkl",
            "scaler_file": "severity_score_scaler.pkl",
            "features_file": "selected_features_v2.csv"
        },
        "severity_stage": {
            "model_file": "severity_stage_model.pkl",
            "scaler_file": "severity_stage_scaler.pkl",
            "features_file": "selected_features_v2.csv"  # Same as severity_score
        },
        "ttf_km": {
            "model_file": "ttf_model.pkl",
            "scaler_file": "ttf_scaler.pkl",
            "features_file": "ttf_features.csv"
        }
    }

    for target, config in model_configs.items():
        log_message(f"  📦 Loading {target}...")
        models_bundle[target] = {}

        try:
            # Load model
            model_path = f"{MODEL_DIR}/{config['model_file']}"

            if target == "anomaly_label":
                # Anomaly model is bundled (model + scaler + imputer + features)
                anomaly_bundle = joblib.load(model_path)
                models_bundle[target]["model"] = anomaly_bundle["model"]
                models_bundle[target]["scaler"] = anomaly_bundle["scaler"]
                models_bundle[target]["imputer"] = anomaly_bundle["imputer"]
                models_bundle[target]["features"] = anomaly_bundle["feature_names"]
                log_message(f"    ✅ Anomaly bundle loaded: model + scaler + imputer + {len(anomaly_bundle['feature_names'])} features")
            else:
                # Other models have separate files
                models_bundle[target]["model"] = joblib.load(model_path)
                log_message(f"    ✅ Model loaded: {config['model_file']}")

                # Load scaler
                if config["scaler_file"]:
                    scaler_path = f"{MODEL_DIR}/{config['scaler_file']}"
                    models_bundle[target]["scaler"] = joblib.load(scaler_path)
                    log_message(f"    ✅ Scaler loaded: {config['scaler_file']}")

                # Load features
                if config["features_file"]:
                    features_path = f"{MODEL_DIR}/{config['features_file']}"
                    features_df = pd.read_csv(features_path)
                    models_bundle[target]["features"] = features_df["feature"].tolist()
                    log_message(f"    ✅ Features loaded: {len(features_df)} features")

        except Exception as e:
            log_message(f"    ❌ Error loading {target}: {str(e)}")
            models_bundle[target] = {"error": str(e)}

    log_message("✅ All models loaded successfully!")
    return models_bundle

# ==========================================================
# 🔮 PREDICTION FUNCTIONS
# ==========================================================
def predict_single_target(X_test, target_name, model_bundle):
    """
    Make predictions for a single target using exact same logic as single evaluation.
    """
    if "error" in model_bundle:
        raise Exception(f"Model loading error: {model_bundle['error']}")

    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    imputer = model_bundle.get("imputer")
    features = model_bundle["features"]

    log_message(f"🔮 Predicting {target_name}...")
    log_message(f"    Using {len(features)} specific features")

    # Prepare features (same logic as single evaluation)
    X_target = X_test[features].copy()

    # Handle missing features by filling with 0.0
    missing_features = [f for f in features if f not in X_test.columns]
    if missing_features:
        log_message(f"    ⚠️ Missing {len(missing_features)} features, filling with 0.0")
        for feature in missing_features:
            X_target[feature] = 0.0

    # Apply preprocessing (same as single evaluation)
    if imputer is not None:
        X_processed = imputer.transform(X_target)
        log_message(f"    ✅ Applied imputation")
    else:
        X_processed = X_target.values

    if scaler is not None:
        if target_name == "ttf_km":
            # Special TTF preprocessing: scale all features, then select TTF features
            all_features = [col for col in X_test.columns if col not in DROP_COLS]
            X_all = X_test[all_features].copy()
            X_scaled_all = scaler.transform(X_all)
            X_scaled_df = pd.DataFrame(X_scaled_all, columns=all_features, index=X_test.index)
            X_processed = X_scaled_df[features].values
            log_message(f"    ✅ Applied scaling on all {len(all_features)} features, then selected {len(features)} for TTF model")
        else:
            X_processed = scaler.transform(X_processed)
            log_message(f"    ✅ Applied scaling")

    # Make predictions
    predictions = model.predict(X_processed)
    log_message(f"    ✅ Predictions generated: {len(predictions)} samples")

    return predictions

# ==========================================================
# 📊 EVALUATION METRICS
# ==========================================================
def calculate_classification_metrics(y_true, y_pred, target_name):
    """Calculate comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    log_message(f"📊 {target_name} Classification Metrics:")
    log_message(f"    Accuracy:  {metrics['accuracy']:.4f}")
    log_message(f"    Precision: {metrics['precision']:.4f}")
    log_message(f"    Recall:    {metrics['recall']:.4f}")
    log_message(f"    F1-Score:  {metrics['f1_score']:.4f}")

    return metrics

def calculate_regression_metrics(y_true, y_pred, target_name):
    """Calculate comprehensive regression metrics."""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    metrics['mape'] = mape

    log_message(f"📊 {target_name} Regression Metrics:")
    log_message(f"    MAE:   {metrics['mae']:.4f}")
    log_message(f"    RMSE:  {metrics['rmse']:.4f}")
    log_message(f"    R²:    {metrics['r2']:.4f}")
    log_message(f"    MAPE:  {metrics['mape']:.2f}%")

    return metrics

# ==========================================================
# 📈 VISUALIZATION FUNCTIONS
# ==========================================================
def plot_confusion_matrix(y_true, y_pred, target_name):
    """Create and save confusion matrix plot."""
    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title(f'Confusion Matrix - {target_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/confusion_matrices/confusion_{target_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_message(f"    ✅ Confusion matrix saved: {save_path}")

def plot_regression_analysis(y_true, y_pred, target_name):
    """Create comprehensive regression analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Regression Analysis - {target_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Error distribution
    abs_errors = np.abs(residuals)
    axes[1, 1].hist(abs_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Absolute Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/regression_plots/regression_{target_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_message(f"    ✅ Regression analysis saved: {save_path}")

# ==========================================================
# 🎯 MAIN BATCH EVALUATION
# ==========================================================
def run_batch_evaluation():
    """
    Main function to run comprehensive batch evaluation.
    """
    log_message("🚀 Starting batch evaluation...")

    # Setup
    ensure_output_dirs()

    # Load and preprocess data
    X_full, targets, df_original = load_and_preprocess_data()

    # Recreate train-test split
    test_sets = recreate_train_test_split(X_full, targets)

    # Load models
    models_bundle = load_models()

    # Storage for all results
    all_results = {}
    all_metrics = {}
    predictions_df = pd.DataFrame()

    log_message("\n" + "="*60)
    log_message("🔮 RUNNING PREDICTIONS ON TEST SET")
    log_message("="*60)

    # Evaluate each target
    for target_name in ['anomaly_label', 'severity_score', 'severity_stage', 'ttf_km']:
        log_message(f"\n🎯 Evaluating {target_name}...")

        try:
            # Get test data for this target
            X_test = test_sets[target_name]['X_test']
            y_test = test_sets[target_name]['y_test']

            # Make predictions
            y_pred = predict_single_target(X_test, target_name, models_bundle[target_name])

            # Store results
            all_results[target_name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }

            # Calculate metrics and create visualizations
            if target_name in ['anomaly_label', 'severity_stage']:
                # Classification metrics
                metrics = calculate_classification_metrics(y_test, y_pred, target_name)
                plot_confusion_matrix(y_test, y_pred, target_name)
            else:
                # Regression metrics
                metrics = calculate_regression_metrics(y_test, y_pred, target_name)
                plot_regression_analysis(y_test, y_pred, target_name)

            all_metrics[target_name] = metrics

            # Add to predictions DataFrame
            temp_df = pd.DataFrame({
                f'actual_{target_name}': y_test,
                f'predicted_{target_name}': y_pred
            })

            if predictions_df.empty:
                predictions_df = temp_df
            else:
                predictions_df = pd.concat([predictions_df, temp_df], axis=1)

            log_message(f"✅ {target_name} evaluation completed")

        except Exception as e:
            log_message(f"❌ Error evaluating {target_name}: {str(e)}")
            all_metrics[target_name] = {"error": str(e)}

    # Save comprehensive results
    save_results(predictions_df, all_metrics, all_results)

    log_message("\n" + "="*60)
    log_message("🎉 BATCH EVALUATION COMPLETED!")
    log_message("="*60)

    return all_results, all_metrics

def save_results(predictions_df, all_metrics, all_results):
    """Save all evaluation results to files."""
    log_message("💾 Saving evaluation results...")

    # 1. Save predictions vs actual CSV
    predictions_path = f"{OUTPUT_DIR}/predictions_vs_actual.csv"
    predictions_df.to_csv(predictions_path, index=False)
    log_message(f"✅ Predictions saved: {predictions_path}")

    # 2. Save metrics summary
    metrics_summary = []
    for target, metrics in all_metrics.items():
        if "error" not in metrics:
            row = {"target": target}
            row.update(metrics)
            metrics_summary.append(row)

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_path = f"{OUTPUT_DIR}/metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log_message(f"✅ Metrics summary saved: {metrics_path}")

    # 3. Save detailed results with sample comparisons
    detailed_results = []
    for target, results in all_results.items():
        if 'y_true' in results:
            y_true = results['y_true']
            y_pred = results['y_pred']

            # Sample some predictions for detailed view
            sample_indices = np.random.choice(len(y_true), min(20, len(y_true)), replace=False)

            for idx in sample_indices:
                detailed_results.append({
                    'target': target,
                    'sample_id': idx,
                    'actual': y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx],
                    'predicted': y_pred[idx],
                    'absolute_error': abs((y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]) - y_pred[idx])
                })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = f"{OUTPUT_DIR}/detailed_sample_results.csv"
    detailed_df.to_csv(detailed_path, index=False)
    log_message(f"✅ Detailed sample results saved: {detailed_path}")

# ==========================================================
# 🚀 MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    try:
        # Run comprehensive batch evaluation
        results, metrics = run_batch_evaluation()

        # Print final summary
        print("\n" + "🏆 FINAL EVALUATION SUMMARY")
        print("="*60)

        for target, target_metrics in metrics.items():
            if "error" not in target_metrics:
                print(f"\n🎯 {target.upper()}:")
                if target in ['anomaly_label', 'severity_stage']:
                    print(f"   Accuracy: {target_metrics['accuracy']:.4f}")
                    print(f"   F1-Score: {target_metrics['f1_score']:.4f}")
                else:
                    print(f"   R²:   {target_metrics['r2']:.4f}")
                    print(f"   MAE:  {target_metrics['mae']:.4f}")
                    print(f"   MAPE: {target_metrics['mape']:.2f}%")

        print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
        print("   📊 Confusion matrices: confusion_matrices/")
        print("   📈 Regression plots: regression_plots/")
        print("   📋 Predictions CSV: predictions_vs_actual.csv")
        print("   📊 Metrics summary: metrics_summary.csv")
        print("   📝 Logs: logs/batch_evaluation.log")

        print("\n✅ Batch evaluation completed successfully!")

    except Exception as e:
        log_message(f"❌ CRITICAL ERROR: {str(e)}")
        print(f"❌ Batch evaluation failed: {str(e)}")
        raise
