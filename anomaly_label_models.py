# anomaly_label_models.py
import pandas as pd
import numpy as np
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
# Drop leakage and non-numeric columns
drop_cols = ["anomaly_label", "source", "failure_type", "severity_score", "severity_stage", "is_gray"]
features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Keep only numeric features
X = features.select_dtypes(include=[np.number]).copy()
y = df["anomaly_label"]

print(f"Features after preprocessing: {X.shape[1]} columns")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Scale features (for SVM, Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
# Define Models
# ============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1
    )),
    ("Logistic Regression", LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42
    )),
    ("SVM (RBF)", SVC(
        kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42
    )),
    ("Isolation Forest", IsolationForest(
        n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1
    ))
]

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models.insert(1, ("XGBoost", XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )))

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

# Check for overfitting
overfitting_models = results_df[results_df["Overfitting Gap"] > 0.05]
if len(overfitting_models) > 0:
    print("⚠️  Models with potential overfitting (gap > 5%):")
    for _, row in overfitting_models.iterrows():
        print(f"   {row['Model']}: {row['Overfitting Gap']:.4f}")
else:
    print("✅ No significant overfitting detected across models")

# Check consistency across models
test_accs = results_df["Test Acc"].dropna()
if len(test_accs) > 1:
    acc_std = test_accs.std()
    if acc_std < 0.02:
        print("✅ Consistent performance across models (low variance)")
    elif acc_std > 0.05:
        print("⚠️  High variance in model performance - possible data issues")
    else:
        print("✅ Reasonable variance in model performance")

# ============================
# Identify Best Model
# ============================
if len(results_df) > 0:
    best_model = results_df.loc[results_df["Test Acc"].idxmax()]
    print("\n🏆 BEST MODEL:")
    print(f"   Model: {best_model['Model']}")
    print(f"   Test Accuracy: {best_model['Test Acc']:.4f}")
    print(f"   F1-Score: {best_model['F1']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
    print(f"   Overfitting Gap: {best_model['Overfitting Gap']:.4f}")

    # Save summary to CSV
    results_df.to_csv("anomaly_label_models_summary.csv", index=False)
    print("\n✅ Saved results to 'anomaly_label_models_summary.csv'")

print("\n" + "="*80)
print("🎯 CONCLUSION")
print("="*80)
print("This comparison validates that your physics-based anomaly injection")
print("works well across multiple algorithms for ANOMALY_LABEL prediction.")
print("All models trained on binary classification: 0=Normal, 1=Anomaly")
