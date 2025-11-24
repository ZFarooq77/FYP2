# modules/train_severity_score.py
"""
Severity Score Stacking Ensemble Model - Production Ready
=========================================================

Features:
- Stacking Ensemble (Gradient Boosting + Random Forest + Ridge meta-learner)
- Balanced severity distribution via oversampling
- Feature selection by importance (top 50 sensors)
- Hyperparameter tuning with GridSearchCV
- Integrated with main.py orchestrator

Author: FYP Team - Catalytic Converter Anomaly Detection
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import RandomOverSampler

# ==========================================================
# 🔧 CONFIGURATION
# ==========================================================
DATA_PATH = "Final.csv"
OUTPUT_DIR = "models/main_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "severity_score"
# Enhanced drop columns for FYP - remove all data leakage sources
DROP_COLS = [
    "anomaly_label", "failure_type", "severity_stage",
    "TTF_years", "TTF_km",
    "is_gray", "source"
]

RANDOM_STATE = 42
N_BINS = 5                    # Quantile bins for stratification
TOP_K_FEATURES = 50           # Top sensor features
CV_SPLITS = 5

# Optimized hyperparameter grids (based on working reference results)
GB_PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "subsample": [0.8, 0.9, 1.0]
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5]
}

# ==========================================================
# 📂 1. LOAD DATA (Non-destructive)
# ==========================================================
def load_and_prepare_data():
    """Load data and prepare for enhanced training."""
    print("🎯 SEVERITY SCORE STACKING ENSEMBLE")
    print("="*50)
    print("📂 Loading data (non-destructive)...")

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape}")

    # Filter to anomaly samples only (severity_score > 0)
    df_anom = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"📊 Anomaly samples: {df_anom.shape[0]} rows ({df_anom.shape[0]/df.shape[0]*100:.1f}%)")

    if df_anom.shape[0] < 50:
        raise RuntimeError("❌ Too few anomaly samples for advanced training. Need more data.")

    # Prepare features and target
    available_drop_cols = [col for col in DROP_COLS if col in df_anom.columns]
    X = df_anom.drop(columns=available_drop_cols + [TARGET], errors='ignore')
    y = df_anom[TARGET].copy()

    # Keep only numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f"✅ Features: {X.shape[1]} sensors")
    print(f"✅ Target range: {y.min():.3f} - {y.max():.3f}")
    print(f"✅ Target mean: {y.mean():.3f}")

    return X, y, df_anom

# ==========================================================
# 📊 2. BALANCED SAMPLING
# ==========================================================
# Removed redundant prepare_features_and_target function

def create_balanced_dataset(X, y):
    """Create balanced dataset using oversampling on severity bins."""
    print("\n📊 Creating balanced severity distribution...")
    
    # Create quantile bins for stratification
    y_binned = pd.qcut(y, q=N_BINS, labels=False, duplicates="drop")
    bin_counts = pd.Series(y_binned).value_counts().sort_index()
    
    print("Original bin distribution:")
    for bin_idx, count in bin_counts.items():
        severity_range = pd.qcut(y, q=N_BINS, duplicates="drop").cat.categories[bin_idx]
        print(f"   Bin {bin_idx} ({severity_range}): {count} samples")
    
    # Apply RandomOverSampler to balance bins
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=RANDOM_STATE)
    
    X_arr = X.values
    y_arr = y.values
    bins_arr = np.array(y_binned)
    
    # Oversample based on bins
    X_res, bins_res = ros.fit_resample(X_arr, bins_arr)
    
    # Map back original y values for resampled data
    rng = np.random.default_rng(RANDOM_STATE)
    bin_to_indices = {b: np.where(bins_arr == b)[0] for b in np.unique(bins_arr)}
    
    y_res = []
    for b in bins_res:
        idx_choice = rng.choice(bin_to_indices[b])
        y_res.append(y_arr[idx_choice])
    
    X_balanced = pd.DataFrame(X_res, columns=X.columns)
    y_balanced = pd.Series(y_res, name=TARGET)
    
    print(f"✅ Balanced dataset: {X_balanced.shape[0]} samples")
    print(f"📈 Oversampling ratio: {X_balanced.shape[0]/X.shape[0]:.2f}x")
    
    return X_balanced, y_balanced, y_binned

# ==========================================================
# 🔍 3. FEATURE SELECTION
# ==========================================================
def select_important_features(X_balanced, y_balanced):
    """Select top K most important sensor features."""
    print(f"\n🔍 Selecting top {TOP_K_FEATURES} most important features...")
    
    # Use lightweight GradientBoosting for feature importance
    feature_selector = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    
    # Scale features for stable importance calculation
    scaler_temp = StandardScaler()
    X_scaled_temp = scaler_temp.fit_transform(X_balanced)
    
    feature_selector.fit(X_scaled_temp, y_balanced)
    
    # Get feature importances
    importances = pd.Series(
        feature_selector.feature_importances_, 
        index=X_balanced.columns
    ).sort_values(ascending=False)
    
    selected_features = importances.head(TOP_K_FEATURES).index.tolist()
    
    print(f"✅ Selected {len(selected_features)} features")
    print("🏆 Top 10 most important sensors:")
    for i, (feature, importance) in enumerate(importances.head(10).items(), 1):
        print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    return selected_features, importances

# ==========================================================
# 🤖 4. ADVANCED MODEL TRAINING
# ==========================================================
def train_stacking_ensemble(X_selected, y_balanced, y_binned_balanced):
    """Train ONLY the Stacking Ensemble (best performing model)."""
    print(f"\n🤖 Training Stacking Ensemble with GridSearchCV...")

    # Scale selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Stratified K-Fold using balanced bins
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    stratified_splits = list(skf.split(X_scaled, y_binned_balanced))

    # Train Gradient Boosting with GridSearch
    print("🔍 Tuning Gradient Boosting...")
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gbr_grid = GridSearchCV(
        gbr, GB_PARAM_GRID, cv=stratified_splits, scoring="r2", n_jobs=-1, verbose=1
    )
    gbr_grid.fit(X_scaled, y_balanced)
    best_gbr = gbr_grid.best_estimator_

    print(f"✅ Best GB params: {gbr_grid.best_params_}")
    print(f"📈 Best GB CV R²: {gbr_grid.best_score_:.4f}")

    # Train Random Forest with GridSearch
    print("\n🔍 Tuning Random Forest...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_grid = GridSearchCV(
        rf, RF_PARAM_GRID, cv=stratified_splits, scoring="r2", n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_scaled, y_balanced)
    best_rf = rf_grid.best_estimator_

    print(f"✅ Best RF params: {rf_grid.best_params_}")
    print(f"📈 Best RF CV R²: {rf_grid.best_score_:.4f}")

    # Create Stacking Ensemble (BEST MODEL)
    print("\n🚀 Creating Stacking Ensemble...")
    estimators = [
        ("gradient_boosting", best_gbr),
        ("random_forest", best_rf)
    ]

    # Use Ridge regression as meta-learner
    meta_learner = Ridge(alpha=1.0)
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    # Evaluate stacking with cross-validation
    cv_scores = cross_val_score(
        stacking_regressor, X_scaled, y_balanced, cv=5, scoring="r2", n_jobs=-1
    )

    print(f"✅ Stacking CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit final stacking model
    stacking_regressor.fit(X_scaled, y_balanced)

    return {
        "stacking_model": stacking_regressor,
        "scaler": scaler,
        "gbr_results": gbr_grid,
        "rf_results": rf_grid,
        "cv_scores": cv_scores
    }

# ==========================================================
# 📊 5. EVALUATION
# ==========================================================
def evaluate_final_model(results, X_original, y_original, selected_features):
    """Evaluate final model on original (unbalanced) data."""
    print("\n📊 Evaluating on original anomaly data...")

    # Prepare original data with selected features
    X_orig_selected = X_original[selected_features]
    X_orig_scaled = results["scaler"].transform(X_orig_selected)

    # Predict with stacking model
    y_pred = results["stacking_model"].predict(X_orig_scaled)

    # Calculate metrics
    r2 = r2_score(y_original, y_pred)
    mae = mean_absolute_error(y_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_original, y_pred))
    mape = np.mean(np.abs((y_original - y_pred) / y_original)) * 100

    print(f"🏆 FINAL MODEL PERFORMANCE:")
    print(f"   R²: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    return {
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
        "y_pred": y_pred, "y_true": y_original
    }

# ==========================================================
# 💾 6. SAVE MODELS AND RESULTS
# ==========================================================
def save_models_and_results(results, selected_features, importances, evaluation):
    """Save all models, scalers, and results."""
    print(f"\n💾 Saving models and results to {OUTPUT_DIR}...")

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("main_outputs", exist_ok=True)

    # Save models
    joblib.dump(results["stacking_model"], os.path.join(OUTPUT_DIR, "severity_score_model.pkl"))
    joblib.dump(results["scaler"], os.path.join(OUTPUT_DIR, "severity_score_scaler.pkl"))

    # Save comprehensive results summary
    summary = {
        "gbr_best_params": [str(results["gbr_results"].best_params_)],
        "gbr_cv_r2": [results["gbr_results"].best_score_],
        "rf_best_params": [str(results["rf_results"].best_params_)],
        "rf_cv_r2": [results["rf_results"].best_score_],
        "stacking_cv_r2_mean": [results["cv_scores"].mean()],
        "stacking_cv_r2_std": [results["cv_scores"].std()],
        "final_r2_original": [evaluation["r2"]],
        "final_mae_original": [evaluation["mae"]],
        "final_rmse_original": [evaluation["rmse"]],
        "final_mape_original": [evaluation["mape"]],
        "n_features_selected": [len(selected_features)],
        "model_type": ["Stacking_Ensemble"]
    }

    pd.DataFrame(summary).to_csv("main_outputs/output_severity_score.csv", index=False)

    print("✅ All models and results saved successfully!")

    return summary

# ==========================================================
# 📊 5. EVALUATION AND DIAGNOSTICS
# ==========================================================
def evaluate_final_model(results, X_original, y_original, selected_features):
    """Evaluate final model on original (unbalanced) data."""
    print("\n📊 Evaluating on original anomaly data...")
    
    # Prepare original data with selected features
    X_orig_selected = X_original[selected_features]
    X_orig_scaled = results["scaler"].transform(X_orig_selected)
    
    # Predict with stacking model
    y_pred = results["stacking_model"].predict(X_orig_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_original, y_pred)
    mae = mean_absolute_error(y_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_original, y_pred))
    mape = np.mean(np.abs((y_original - y_pred) / y_original)) * 100
    
    print(f"🏆 FINAL MODEL PERFORMANCE:")
    print(f"   R²: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
        "y_pred": y_pred, "y_true": y_original
    }

# Removed duplicate save_models_and_results function

# ==========================================================
# 📊 7. COMPREHENSIVE VISUALIZATIONS
# ==========================================================
def create_diagnostic_plots(evaluation, importances, selected_features, results):
    """Create comprehensive diagnostic plots."""
    print("\n📊 Creating diagnostic visualizations...")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Severity Score Model v2 - Diagnostic Analysis',
                 fontsize=16, fontweight='bold')

    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, s=60, ax=ax1)

    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
             label=f'Perfect Prediction\nR² = {evaluation["r2"]:.4f}')

    ax1.set_title("Predicted vs Actual Severity Score", fontweight='bold')
    ax1.set_xlabel("Actual Severity Score")
    ax1.set_ylabel("Predicted Severity Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals Analysis
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, s=60, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_title(f"Residuals vs Predicted\nMAE = {evaluation['mae']:.4f}", fontweight='bold')
    ax2.set_xlabel("Predicted Severity Score")
    ax2.set_ylabel("Residuals (Actual - Predicted)")
    ax2.grid(True, alpha=0.3)

    # 3. Feature Importance (Top 20)
    ax3 = axes[1, 0]
    top_features = importances.head(20)
    top_features.plot(kind='barh', ax=ax3)
    ax3.set_title(f"Top 20 Feature Importances\n({len(selected_features)} selected)", fontweight='bold')
    ax3.set_xlabel("Importance Score")

    # 4. Residuals Distribution
    ax4 = axes[1, 1]
    sns.histplot(residuals, kde=True, bins=25, ax=ax4)
    ax4.axvline(x=0, color='r', linestyle='--', lw=2)
    ax4.set_title(f"Residuals Distribution\nRMSE = {evaluation['rmse']:.4f}", fontweight='bold')
    ax4.set_xlabel("Residuals")
    ax4.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "enhanced_model_diagnostics.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Create model comparison plot
    plt.figure(figsize=(12, 6))

    # Compare individual models vs stacking
    models_comparison = {
        'Gradient Boosting': results["gbr_results"].best_score_,
        'Random Forest': results["rf_results"].best_score_,
        'Stacking Ensemble': results["cv_scores"].mean()
    }

    plt.subplot(1, 2, 1)
    models = list(models_comparison.keys())
    scores = list(models_comparison.values())
    colors = ['skyblue', 'lightgreen', 'gold']

    bars = plt.bar(models, scores, color=colors, alpha=0.8)
    plt.title('Model Performance Comparison\n(Cross-Validation R²)', fontweight='bold')
    plt.ylabel('R² Score')
    plt.ylim(0, max(scores) * 1.1)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)

    # Performance metrics summary
    plt.subplot(1, 2, 2)
    metrics = ['R²', 'MAE', 'RMSE', 'MAPE (%)']
    values = [evaluation['r2'], evaluation['mae'], evaluation['rmse'], evaluation['mape']]

    plt.barh(metrics, values, color=['gold', 'lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Final Model Metrics\n(Original Test Data)', fontweight='bold')
    plt.xlabel('Metric Value')

    # Add value labels
    for i, v in enumerate(values):
        plt.text(v + max(values)*0.01, i, f'{v:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_and_metrics.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Diagnostic plots saved successfully!")

# ==========================================================
# 📝 8. USAGE INSTRUCTIONS
# ==========================================================
def create_usage_instructions():
    """Create usage instructions for the trained model."""
    usage_snippet = f"""
# ENHANCED SEVERITY SCORE MODEL V2 - USAGE INSTRUCTIONS
# ====================================================

import joblib
import pandas as pd
import numpy as np

# 1. Load trained components
model = joblib.load("{os.path.join(OUTPUT_DIR, 'severity_stacking_model_v2.pkl')}")
scaler = joblib.load("{os.path.join(OUTPUT_DIR, 'severity_scaler_v2.pkl')}")
selected_features = pd.read_csv("{os.path.join(OUTPUT_DIR, 'selected_features_v2.csv')}")['feature'].tolist()

# 2. Prepare new data for prediction
# X_new should be a DataFrame with the same sensor columns as training data
X_new_selected = X_new[selected_features]  # Select only important features
X_new_scaled = scaler.transform(X_new_selected)  # Scale features

# 3. Make predictions
severity_predictions = model.predict(X_new_scaled)

# 4. Interpret results
# severity_predictions contains values from 0.0 to 1.0:
# 0.00-0.25: Mild degradation
# 0.26-0.50: Moderate degradation
# 0.51-0.75: Severe degradation
# 0.76-1.00: Critical failure

# Example usage:
for i, severity in enumerate(severity_predictions):
    if severity <= 0.25:
        status = "Mild"
    elif severity <= 0.50:
        status = "Moderate"
    elif severity <= 0.75:
        status = "Severe"
    else:
        status = "Critical"

    print(f"Sample {{i+1}}: Severity = {{severity:.3f}} ({{status}})")

# NOTES:
# - Model trained on {TOP_K_FEATURES} most important sensor features
# - Uses advanced stacking ensemble (Gradient Boosting + Random Forest + Ridge meta-learner)
# - Balanced training data for improved rare severity prediction
# - Production-ready for catalytic converter degradation assessment
"""

    with open(os.path.join(OUTPUT_DIR, "usage_instructions.txt"), "w") as f:
        f.write(usage_snippet)

    print("✅ Usage instructions saved!")

# ==========================================================
# 🚀 MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    try:
        # 1. Load and prepare data
        X_original, y_original, df_anomalies = load_and_prepare_data()

        # 2. Create balanced dataset
        X_balanced, y_balanced, y_binned = create_balanced_dataset(X_original, y_original)

        # 3. Select important features
        selected_features, importances = select_important_features(X_balanced, y_balanced)
        X_selected = X_balanced[selected_features]

        # Create balanced bins for selected data
        y_balanced_binned = pd.qcut(y_balanced, q=N_BINS, labels=False, duplicates="drop")

        # 4. Train ONLY Stacking Ensemble (best model)
        results = train_stacking_ensemble(X_selected, y_balanced, y_balanced_binned)

        # 5. Evaluate on original data
        evaluation = evaluate_final_model(results, X_original, y_original, selected_features)

        # 6. Save models and results
        summary = save_models_and_results(results, selected_features, importances, evaluation)

        # 7. Create visualizations
        create_diagnostic_plots(evaluation, importances, selected_features, results)

        # 8. Create usage instructions
        create_usage_instructions()

        # Final summary
        print("\n" + "="*60)
        print("🏁 ENHANCED SEVERITY SCORE MODEL V2 - COMPLETE!")
        print("="*60)
        print(f"🏆 Final Stacking Model Performance:")
        print(f"   📊 R² Score: {evaluation['r2']:.4f} ({evaluation['r2']*100:.2f}%)")
        print(f"   📏 MAE: {evaluation['mae']:.4f}")
        print(f"   📐 RMSE: {evaluation['rmse']:.4f}")
        print(f"   📊 MAPE: {evaluation['mape']:.2f}%")
        print(f"\n🔧 Model Features:")
        print(f"   🎯 Selected Features: {len(selected_features)}/{X_original.shape[1]} sensors")
        print(f"   🤖 Ensemble: Gradient Boosting + Random Forest + Ridge")
        print(f"   ⚖️ Balanced Training: {X_balanced.shape[0]} samples")
        print(f"   📊 Cross-Validation: {CV_SPLITS}-fold stratified")
        print(f"\n💾 All outputs saved to: {OUTPUT_DIR}/")
        print("🚀 Model ready for production deployment!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# ============================
# MAIN ORCHESTRATOR FUNCTION
# ============================
def main_severity_score_model():
    """
    Main function for severity score stacking ensemble training.
    Returns structured result for main.py orchestrator.
    """
    try:
        # Delete existing severity_score models first
        import glob
        existing_models = glob.glob(f"{OUTPUT_DIR}/severity_score*.pkl")
        for model_file in existing_models:
            if os.path.exists(model_file):
                os.remove(model_file)
                print(f"🗑️ Deleted existing model: {model_file}")

        # 1. Load and prepare data
        X_original, y_original, df_anomalies = load_and_prepare_data()

        # 2. Create balanced dataset
        X_balanced, y_balanced, y_binned = create_balanced_dataset(X_original, y_original)

        # 3. Select important features
        selected_features, importances = select_important_features(X_balanced, y_balanced)
        X_selected = X_balanced[selected_features]

        # Create balanced bins for selected data
        y_balanced_binned = pd.qcut(y_balanced, q=N_BINS, labels=False, duplicates="drop")

        # 4. Train ONLY Stacking Ensemble (best model)
        results = train_stacking_ensemble(X_selected, y_balanced, y_balanced_binned)

        # 5. Evaluate on original data
        evaluation = evaluate_final_model(results, X_original, y_original, selected_features)

        # 6. Save models and results
        save_models_and_results(results, selected_features, importances, evaluation)

        # Extract key metrics
        r2_score_val = evaluation.get('r2', 0.0)
        mae = evaluation.get('mae', 0.0)
        rmse = evaluation.get('rmse', 0.0)
        mape = evaluation.get('mape', 0.0)

        model_path = f"{OUTPUT_DIR}/severity_score_model.pkl"
        summary_path = "main_outputs/output_severity_score.csv"

        return {
            "status": "success",
            "model_name": "StackingEnsemble_SeverityScore",
            "model_path": model_path,
            "summary_path": summary_path,
            "metrics": {
                "r2_score": r2_score_val,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "selected_features": len(selected_features),
                "training_samples": X_balanced.shape[0]
            }
        }

    except Exception as e:
        return {
            "status": "failed",
            "model_name": "StackingEnsemble_SeverityScore",
            "error": str(e)
        }

# ============================
# STANDALONE EXECUTION
# ============================
if __name__ == "__main__":
    result = main_severity_score_model()
    if result["status"] == "success":
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Model saved: {result['model_path']}")
        print(f"📈 Performance: R² = {result['metrics']['r2_score']:.4f}")
    else:
        print(f"\n❌ Training failed: {result['error']}")
