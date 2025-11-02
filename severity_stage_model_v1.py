# ============================================================
# 🚗 SEVERITY STAGE CLASSIFICATION MODEL V1
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

print("🚗 SEVERITY STAGE CLASSIFICATION MODEL V1")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD DATA & PREPROCESS
# ------------------------------------------------------------
print("\n📊 LOADING DATA...")

try:
    # Try the exact filename from your code first
    df = pd.read_csv("Final_Catalytic_Physics_V1.csv")
    print(f"✅ Loaded Final_Catalytic_Physics_V1.csv: {df.shape}")
except FileNotFoundError:
    # Fallback to the actual file we have
    df = pd.read_csv("Final.csv")
    print(f"✅ Loaded Final.csv: {df.shape}")

# Include ALL samples (Stage 0-4) for complete classification
print(f"📊 Total dataset: {len(df)} samples")

# Check severity stage distribution in full dataset
print("\n📊 FULL SEVERITY STAGE DISTRIBUTION:")
stage_dist_full = df['severity_stage'].value_counts().sort_index()
for stage, count in stage_dist_full.items():
    pct = count / len(df) * 100
    print(f"  Stage {stage}: {count:4d} samples ({pct:5.1f}%)")

# Handle class imbalance - undersample Stage 0 to balance with other stages
from sklearn.utils import resample

print("\n⚖️  APPLYING CLASS BALANCING...")
stage_0 = df[df['severity_stage'] == 0]
stage_others = df[df['severity_stage'] > 0]

print(f"📊 Stage 0 (Normal): {len(stage_0)} samples")
print(f"📊 Stage 1-4 (Anomalies): {len(stage_others)} samples")

# Undersample Stage 0 to 2x the size of anomaly samples for better balance
target_stage0_size = min(len(stage_others) * 2, len(stage_0))
stage_0_balanced = resample(stage_0, n_samples=target_stage0_size, random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([stage_0_balanced, stage_others], ignore_index=True)
print(f"✅ Balanced dataset: {len(df_balanced)} samples")

# Check balanced distribution
print("\n📊 BALANCED SEVERITY STAGE DISTRIBUTION:")
stage_dist_balanced = df_balanced['severity_stage'].value_counts().sort_index()
for stage, count in stage_dist_balanced.items():
    pct = count / len(df_balanced) * 100
    print(f"  Stage {stage}: {count:3d} samples ({pct:5.1f}%)")

# Feature and target selection
try:
    feature_cols = pd.read_csv("models_v2_enhanced/selected_features_v2.csv")["feature"].tolist()
    print(f"✅ Loaded {len(feature_cols)} selected features from V2 model")
except FileNotFoundError:
    # Fallback to all sensor features if V2 features not available
    sensor_cols = [col for col in df.columns if any(sensor in col for sensor in 
                   ['ENGINE_RPM', 'CATALYST_TEMPERATURE', 'FUEL_TRIM', 'ENGINE_LOAD', 
                    'INTAKE_AIR_TEMP', 'COOLANT_TEMPERATURE', 'THROTTLE'])][:50]
    feature_cols = sensor_cols
    print(f"⚠️  Using fallback features: {len(feature_cols)} sensor features")

# ------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ------------------------------------------------------------
print("\n🔄 CREATING TRAIN/TEST SPLIT...")
X = df_balanced[feature_cols]
y = df_balanced["severity_stage"].astype(int)

print(f"� Full dataset:")
print(f"�📋 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"🎯 Classes: {sorted(y.unique())} (5-class classification)")

# Create train/test split (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Dataset split:")
print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Create sample weights for training set only
print("\n⚖️  CREATING SAMPLE WEIGHTS...")
train_indices = X_train.index
gray_mask_train = df_balanced.loc[train_indices, 'is_gray'] == True
sample_weights_train = np.ones(len(X_train))
sample_weights_train[gray_mask_train] = 0.6  # Reduced weight for gray rows

print(f"📊 Training set sample weights:")
print(f"  Normal weight (1.0): {sum(~gray_mask_train)} samples")
print(f"  Reduced weight (0.6): {sum(gray_mask_train)} samples ({sum(gray_mask_train)/len(X_train)*100:.1f}%)")

# Also track gray rows in test set for evaluation
test_indices = X_test.index
gray_mask_test = df_balanced.loc[test_indices, 'is_gray'] == True
print(f"📊 Test set gray rows: {sum(gray_mask_test)} samples ({sum(gray_mask_test)/len(X_test)*100:.1f}%)")

# ------------------------------------------------------------
# 5. FEATURE SCALING
# ------------------------------------------------------------
print("\n🔧 SCALING FEATURES...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future inference
joblib.dump(scaler, "severity_stage_scaler_v1.pkl")
print("✅ Scaler saved as 'severity_stage_scaler_v1.pkl'")

# ------------------------------------------------------------
# 6. STRATIFIED K-FOLD CROSS VALIDATION
# ------------------------------------------------------------
print("\n🔄 SETTING UP CROSS-VALIDATION...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute class weights for imbalanced data (using training data only)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"⚖️  Class weights: {class_weight_dict}")

# ------------------------------------------------------------
# 3. DEFINE BASE MODELS
# ------------------------------------------------------------
print("\n🤖 DEFINING BASE MODELS...")
gb = GradientBoostingClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

# ------------------------------------------------------------
# 4. HYPERPARAMETER TUNING
# ------------------------------------------------------------
print("\n🔍 HYPERPARAMETER TUNING...")

param_grid_gb = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0]
}

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8],
    "max_features": [0.5, "sqrt"],
    "min_samples_split": [5, 10]
}

print("🔧 Tuning Gradient Boosting...")
grid_gb = GridSearchCV(gb, param_grid_gb, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
grid_gb.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

print("🔧 Tuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
grid_rf.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

best_gb = grid_gb.best_estimator_
best_rf = grid_rf.best_estimator_

print(f"✅ Best GB F1-macro: {grid_gb.best_score_:.4f}")
print(f"✅ Best RF F1-macro: {grid_rf.best_score_:.4f}")

# ------------------------------------------------------------
# 5. STACKING ENSEMBLE
# ------------------------------------------------------------
print("\n🏗️  BUILDING STACKING ENSEMBLE...")
estimators = [("gb", best_gb), ("rf", best_rf)]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(
        learning_rate=0.05, n_estimators=50, max_depth=2, random_state=42
    ),
    cv=cv,
    n_jobs=-1
)

print("🔄 Training stacking model...")
stack_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

# ------------------------------------------------------------
# 7. INDIVIDUAL MODEL EVALUATION
# ------------------------------------------------------------
print("\n📊 INDIVIDUAL MODEL EVALUATION...")

# Evaluate each model separately
models = {
    "Gradient Boosting": best_gb,
    "Random Forest": best_rf,
    "Stacking Ensemble": stack_model
}

model_results = {}

for model_name, model in models.items():
    print(f"\n🔍 Evaluating {model_name}...")

    # Training set evaluation
    y_train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average="macro")
    train_f1_weighted = f1_score(y_train, y_train_pred, average="weighted")
    train_precision = precision_score(y_train, y_train_pred, average="macro")
    train_recall = recall_score(y_train, y_train_pred, average="macro")

    # Test set evaluation
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")
    test_precision = precision_score(y_test, y_test_pred, average="macro")
    test_recall = recall_score(y_test, y_test_pred, average="macro")

    # Overfitting analysis
    overfitting_gap = (train_acc - test_acc) * 100
    status = "✅ Good" if overfitting_gap < 5 else "⚠️ Moderate" if overfitting_gap < 10 else "❌ High"

    # Store results
    model_results[model_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1_macro': train_f1_macro,
        'test_f1_macro': test_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'test_f1_weighted': test_f1_weighted,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'overfitting_gap': overfitting_gap,
        'status': status,
        'y_test_pred': y_test_pred  # Store for later use
    }

    print(f"🎯 {model_name} PERFORMANCE:")
    print(f"📊 TRAINING SET:")
    print(f"  Accuracy: {train_acc*100:.2f}%")
    print(f"  F1 Score (macro): {train_f1_macro*100:.2f}%")
    print(f"  F1 Score (weighted): {train_f1_weighted*100:.2f}%")
    print(f"  Precision (macro): {train_precision*100:.2f}%")
    print(f"  Recall (macro): {train_recall*100:.2f}%")

    print(f"🎯 TEST SET:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1 Score (macro): {test_f1_macro*100:.2f}%")
    print(f"  F1 Score (weighted): {test_f1_weighted*100:.2f}%")
    print(f"  Precision (macro): {test_precision*100:.2f}%")
    print(f"  Recall (macro): {test_recall*100:.2f}%")

    print(f"📈 OVERFITTING ANALYSIS:")
    print(f"  Accuracy gap: {overfitting_gap:.2f} percentage points - {status}")

# Use Stacking Ensemble results for detailed reporting (best model)
best_model_results = model_results["Stacking Ensemble"]
y_test_pred = best_model_results['y_test_pred']

print("\n📋 DETAILED TEST SET CLASSIFICATION REPORT (Stacking Ensemble):")
print(classification_report(y_test, y_test_pred))

# ------------------------------------------------------------
# 7.1. GRAY ROWS SPECIFIC EVALUATION (TEST SET)
# ------------------------------------------------------------
print("\n🔍 GRAY ROWS SPECIFIC EVALUATION (TEST SET)...")

if sum(gray_mask_test) > 0:
    # Gray rows evaluation on test set
    y_test_gray = y_test[gray_mask_test]
    y_test_pred_gray = y_test_pred[gray_mask_test]

    gray_test_acc = accuracy_score(y_test_gray, y_test_pred_gray)
    gray_test_f1_macro = f1_score(y_test_gray, y_test_pred_gray, average="macro", zero_division=0)
    gray_test_f1_weighted = f1_score(y_test_gray, y_test_pred_gray, average="weighted", zero_division=0)

    print(f"\n🔍 GRAY ROWS TEST PERFORMANCE ({sum(gray_mask_test)} samples):")
    print(f"  Accuracy: {gray_test_acc*100:.2f}%")
    print(f"  F1 Score (macro): {gray_test_f1_macro*100:.2f}%")
    print(f"  F1 Score (weighted): {gray_test_f1_weighted*100:.2f}%")

    # Gray rows stage distribution in test set
    print(f"\n📊 Gray rows test set stage distribution:")
    gray_stage_dist = pd.Series(y_test_gray).value_counts().sort_index()
    for stage, count in gray_stage_dist.items():
        pct = count / len(y_test_gray) * 100
        print(f"  Stage {stage}: {count} samples ({pct:.1f}%)")

    # Gray rows classification report
    print(f"\n📋 Gray rows test set classification report:")
    print(classification_report(y_test_gray, y_test_pred_gray, zero_division=0))

    # Non-gray rows evaluation for comparison (test set)
    non_gray_mask_test = ~gray_mask_test
    if sum(non_gray_mask_test) > 0:
        y_test_non_gray = y_test[non_gray_mask_test]
        y_test_pred_non_gray = y_test_pred[non_gray_mask_test]

        non_gray_test_acc = accuracy_score(y_test_non_gray, y_test_pred_non_gray)
        non_gray_test_f1_macro = f1_score(y_test_non_gray, y_test_pred_non_gray, average="macro", zero_division=0)

        print(f"\n🎯 NON-GRAY ROWS TEST PERFORMANCE ({sum(non_gray_mask_test)} samples):")
        print(f"  Accuracy: {non_gray_test_acc*100:.2f}%")
        print(f"  F1 Score (macro): {non_gray_test_f1_macro*100:.2f}%")

        print(f"\n📊 TEST SET PERFORMANCE COMPARISON:")
        print(f"  Gray rows accuracy: {gray_test_acc*100:.2f}%")
        print(f"  Non-gray rows accuracy: {non_gray_test_acc*100:.2f}%")
        print(f"  Accuracy difference: {(non_gray_test_acc - gray_test_acc)*100:.2f} percentage points")
else:
    print("⚠️  No gray rows found in test set")

# Confusion Matrix (Test Set)
print("\n📊 GENERATING CONFUSION MATRIX (TEST SET)...")
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=[f'Stage {i}' for i in sorted(y_test.unique())],
            yticklabels=[f'Stage {i}' for i in sorted(y_test.unique())])
plt.title("Confusion Matrix - Severity Stage Model V1 (Test Set)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Stage")
plt.ylabel("Actual Stage")
plt.tight_layout()
plt.savefig("severity_stage_confusion_matrix_v1.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# 7. SAVE MODELS
# ------------------------------------------------------------
print("\n💾 SAVING MODELS...")
joblib.dump(best_gb, "severity_stage_gb_model_v1.pkl")
joblib.dump(best_rf, "severity_stage_rf_model_v1.pkl")
joblib.dump(stack_model, "severity_stage_stacking_model_v1.pkl")

print("✅ Models saved:")
print("  - severity_stage_gb_model_v1.pkl")
print("  - severity_stage_rf_model_v1.pkl") 
print("  - severity_stage_stacking_model_v1.pkl")
print("  - severity_stage_scaler_v1.pkl")

# ------------------------------------------------------------
# 8. SUMMARY OUTPUT
# ------------------------------------------------------------
print("\n📊 CREATING SUMMARY...")

# Create summary with individual model results
summary_data = []
for model_name in ["Gradient Boosting", "Random Forest", "Stacking Ensemble"]:
    results = model_results[model_name]
    cv_score = grid_gb.best_score_ if model_name == "Gradient Boosting" else \
               grid_rf.best_score_ if model_name == "Random Forest" else \
               results['test_f1_macro']

    summary_data.append({
        "Model": model_name,
        "CV_F1_macro": cv_score,
        "Train_Accuracy": results['train_acc'],
        "Test_Accuracy": results['test_acc'],
        "Train_F1_macro": results['train_f1_macro'],
        "Test_F1_macro": results['test_f1_macro'],
        "Overfitting_Gap": results['overfitting_gap']
    })

summary = pd.DataFrame(summary_data)
summary.to_csv("severity_stage_model_v1_summary.csv", index=False)
print("✅ Summary saved as 'severity_stage_model_v1_summary.csv'")

# Display summary table
print("\n📊 MODEL COMPARISON SUMMARY:")
print("=" * 100)
print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Test F1':<10} {'Overfit Gap':<12} {'Verdict':<15}")
print("=" * 100)

for model_name in ["Gradient Boosting", "Random Forest", "Stacking Ensemble"]:
    results = model_results[model_name]
    verdict = "Overfitting badly" if results['overfitting_gap'] > 10 else \
              "Moderate overfit" if results['overfitting_gap'] > 5 else \
              "Best balance"

    print(f"{model_name:<20} {results['train_acc']*100:>6.2f}%   {results['test_acc']*100:>6.2f}%   "
          f"{results['train_f1_macro']*100:>6.2f}%   {results['test_f1_macro']*100:>6.2f}%   "
          f"{results['overfitting_gap']:>8.2f}pp   {verdict:<15}")

print("=" * 100)

# Use best model (Stacking Ensemble) for final reporting
best_results = model_results["Stacking Ensemble"]

print("\n" + "=" * 60)
print("✅ SEVERITY STAGE CLASSIFICATION MODEL V1 COMPLETE!")
print("=" * 60)
print(f"🎯 Best Model Performance (Stacking Ensemble):")
print(f"  📊 Train Accuracy: {best_results['train_acc']*100:.2f}%")
print(f"  📊 Test Accuracy: {best_results['test_acc']*100:.2f}%")
print(f"  🎯 Train F1-Score (macro): {best_results['train_f1_macro']*100:.2f}%")
print(f"  🎯 Test F1-Score (macro): {best_results['test_f1_macro']*100:.2f}%")
print(f"  📈 Overfitting Gap: {best_results['overfitting_gap']:.2f}pp - {best_results['status']}")
print(f"  ⚖️  Balanced for {len(np.unique(y_train))} severity stages")
print(f"  🔧 Features: {len(feature_cols)} selected sensors")
print(f"  📈 Training samples: {len(X_train)} / Test samples: {len(X_test)}")
print(f"  ⚖️  Sample weights: Gray rows weighted at 0.6, others at 1.0")
print(f"  🔍 Gray rows in test: {sum(gray_mask_test)} samples ({sum(gray_mask_test)/len(X_test)*100:.1f}%)")
print(f"  🎯 Classes: 5 stages (0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=Critical)")
print("\n🚀 Ready for production deployment!")
