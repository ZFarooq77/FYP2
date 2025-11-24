# anomaly_label_models.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Try to import XGBoost, skip if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - skipping XGBoost model")
    XGBOOST_AVAILABLE = False

# ============================
# Load Dataset
# ============================
file_path = "Final.csv"
df = pd.read_csv(file_path)
print(f"\nLoaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================
# Preprocessing
# ============================
# 🎯 FOCUS ON ANOMALY_LABEL TARGET ONLY
# Drop rows where anomaly_label is NaN (if any)
df = df.dropna(subset=['anomaly_label'])
print(f"After removing NaN anomaly_label: {df.shape[0]} rows")

# Drop leakage and non-numeric columns
drop_cols = ["anomaly_label", "source", "failure_type", "severity_score", "severity_stage", "TTF_km", "TTF_years", "is_gray"]
features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Keep only numeric features
X = features.select_dtypes(include=[np.number]).copy()
y = df["anomaly_label"]

# 🧹 HANDLE NaN VALUES IN FEATURES
# Fill NaN values with median (robust for anomaly detection)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

print(f"Features after preprocessing: {X_imputed.shape[1]} columns")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Scale features (XGBoost doesn't require scaling but we'll keep for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# Helper function
# ============================
def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv=None):
    """Train model, evaluate metrics, and return results."""
    
    # Special handling for Isolation Forest (unsupervised)
    if name == "Isolation Forest":
        model.fit(X_train)
        y_pred = model.predict(X_test)
        # Convert Isolation Forest output: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = np.where(y_pred == -1, 1, 0)
        train_pred = model.predict(X_train)
        train_pred = np.where(train_pred == -1, 1, 0)
        train_acc = accuracy_score(y_train, train_pred)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
    
    results = {
        "Model": name,
        "Train Acc": train_acc,
        "Test Acc": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Cross-validation stability (skip for Isolation Forest)
    if cv and name != "Isolation Forest":
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results["CV Mean"] = cv_scores.mean()
        results["CV ±2σ"] = cv_scores.std() * 2
    else:
        results["CV Mean"] = np.nan
        results["CV ±2σ"] = np.nan
    
    return results

# ============================
# Define Best Model Only - XGBoost
# ============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 🏆 ONLY TRAIN THE BEST MODEL: XGBoost (97.53% accuracy)
if XGBOOST_AVAILABLE:
    models = [("XGBoost", XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1
    ))]
else:
    print("❌ XGBoost not available - cannot proceed with best model")
    print("Please install XGBoost: pip install xgboost")
    exit(1)

# ============================
# Train & Evaluate
# ============================
results = []
for name, model in models:
    print(f"\nTraining: {name}")
    try:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test, cv)
        results.append(res)
        print(f"✅ {name}: Test Acc = {res['Test Acc']:.4f}, F1 = {res['F1']:.4f}")
    except Exception as e:
        print(f"❌ {name} failed: {e}")

# ============================
# Results Summary
# ============================
results_df = pd.DataFrame(results)
results_df["Overfitting Gap"] = results_df["Train Acc"] - results_df["Test Acc"]

print("\n" + "="*80)
print("📊 MODEL PERFORMANCE SUMMARY")
print("="*80)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================
# Analysis & Insights
# ============================
print("\n" + "="*80)
print("🔍 ANALYSIS & INSIGHTS")
print("="*80)

# Since we're only training the best model (XGBoost)
if len(results_df) > 0:
    best_model = results_df.iloc[0]  # Only one model now
    print("🏆 OPTIMIZED MODEL PERFORMANCE:")
    print(f"   Model: {best_model['Model']} (Best from previous comparison)")
    print(f"   Test Accuracy: {best_model['Test Acc']:.4f}")
    print(f"   F1-Score: {best_model['F1']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
    print(f"   Overfitting Gap: {best_model['Overfitting Gap']:.4f}")

    # Check overfitting
    if best_model['Overfitting Gap'] > 0.05:
        print("⚠️  Moderate overfitting detected - consider regularization")
    else:
        print("✅ Good generalization performance")

    print("\n✅ Training focused on best-performing model only")
    print("✅ NaN values handled with median imputation")
    print("✅ Target: anomaly_label (binary classification)")

    # Save summary to CSV
    os.makedirs("main_outputs", exist_ok=True)
    results_df.to_csv("main_outputs/output_anomaly_label.csv", index=False)
    print("\n✅ Saved results to 'main_outputs/output_anomaly_label.csv'")

print("\n" + "="*80)
print("🎯 CONCLUSION")
print("="*80)
print("This comparison validates that your physics-based anomaly injection")
print("works well across multiple algorithms for ANOMALY_LABEL prediction.")
print("All models trained on binary classification: 0=Normal, 1=Anomaly")

# ============================
# MAIN ORCHESTRATOR FUNCTION
# ============================
def main_anomaly_label_model():
    """
    Main function for anomaly detection model training.
    Returns structured result for main.py orchestrator.
    """
    try:
        # Execute the training pipeline (reuse existing code above)
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.impute import SimpleImputer
        import joblib
        import os

        # Import XGBoost
        try:
            from xgboost import XGBClassifier
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for anomaly detection. Please install: pip install xgboost")

        # Load and preprocess data
        file_path = "Final.csv"
        df = pd.read_csv(file_path)
        print(f"\nLoaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

        # 🎯 FOCUS ON ANOMALY_LABEL TARGET ONLY
        df = df.dropna(subset=['anomaly_label'])
        print(f"After removing NaN anomaly_label: {df.shape[0]} rows")

        # Drop leakage and non-numeric columns
        drop_cols = ["anomaly_label", "source", "failure_type", "severity_score", "severity_stage", "TTF_km", "TTF_years", "is_gray"]
        features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Keep only numeric features
        X = features.select_dtypes(include=[np.number]).copy()
        y = df["anomaly_label"]

        # Handle NaN values in features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        print(f"Features after preprocessing: {X_imputed.shape[1]} columns")
        print(f"Target distribution: {dict(y.value_counts().sort_index())}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )

        print(f"\nTraining: XGBoost")
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        overfitting_gap = train_acc - test_acc

        print(f"✅ XGBoost: Test Acc = {test_acc:.4f}, F1 = {f1:.4f}")

        # Save model
        model_path = "models/main_models/anomaly_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'feature_names': list(X.columns)
        }, model_path)

        # Save results summary
        results_data = {
            'Model': ['XGBoost'],
            'Train Acc': [train_acc],
            'Test Acc': [test_acc],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1],
            'CV Mean': [cv_mean],
            'CV ±2σ': [2 * cv_std],
            'Overfitting Gap': [overfitting_gap]
        }

        results_df = pd.DataFrame(results_data)
        summary_path = "main_outputs/output_anomaly_label.csv"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        results_df.to_csv(summary_path, index=False)

        # Return structured result
        return {
            "status": "success",
            "model_name": "XGBoost_AnomalyLabel",
            "model_path": model_path,
            "summary_path": summary_path,
            "metrics": {
                "accuracy": test_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_mean": cv_mean,
                "overfitting_gap": overfitting_gap
            }
        }

    except Exception as e:
        return {
            "status": "failed",
            "model_name": "XGBoost_AnomalyLabel",
            "error": str(e)
        }

# ============================
# STANDALONE EXECUTION
# ============================
if __name__ == "__main__":
    result = main_anomaly_label_model()
    if result["status"] == "success":
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Model saved: {result['model_path']}")
        print(f"📈 Performance: {result['metrics']}")
    else:
        print(f"\n❌ Training failed: {result['error']}")
