import joblib
import pandas as pd

# Check anomaly model
print("=== ANOMALY MODEL ===")
anomaly = joblib.load('../../models/main_models/anomaly_model.pkl')
print(f"Type: {type(anomaly)}")
print(f"Keys: {anomaly.keys()}")
print(f"Features: {len(anomaly['feature_names'])}")
print(f"First 10: {anomaly['feature_names'][:10]}")

# Check severity_stage model
print("\n=== SEVERITY STAGE MODEL ===")
stage_model = joblib.load('../../models/main_models/severity_stage_model.pkl')
print(f"Type: {type(stage_model)}")
print(f"Has feature_names_in_: {hasattr(stage_model, 'feature_names_in_')}")
if hasattr(stage_model, 'feature_names_in_'):
    print(f"Features: {len(stage_model.feature_names_in_)}")
    print(f"First 10: {stage_model.feature_names_in_[:10]}")

# Check severity_score features from CSV
print("\n=== SEVERITY SCORE FEATURES ===")
severity_features = pd.read_csv('../../models/main_models/selected_features_v2.csv')
print(f"Features: {len(severity_features)}")
print(f"First 10: {severity_features['feature'].head(10).tolist()}")

# Check TTF features from CSV
print("\n=== TTF FEATURES ===")
ttf_features = pd.read_csv('../../models/main_models/ttf_features.csv')
print(f"Features: {len(ttf_features)}")
print(f"First 10: {ttf_features['feature'].head(10).tolist()}")

# Check Final.csv columns
print("\n=== FINAL.CSV COLUMNS ===")
df = pd.read_csv('../../Final.csv')
print(f"Total columns: {len(df.columns)}")
print(f"First 10: {df.columns[:10].tolist()}")

# Find DROP_COLS
drop_cols = ["anomaly_label", "failure_type", "severity_stage", "severity_score", "TTF_years", "TTF_km", "is_gray", "source"]
feature_cols = [col for col in df.columns if col not in drop_cols]
print(f"Feature columns (after dropping targets): {len(feature_cols)}")
print(f"First 10 feature cols: {feature_cols[:10]}")
