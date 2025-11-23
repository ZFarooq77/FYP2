#!/usr/bin/env python3
"""
Test different scenarios with the evaluation system
"""

import pandas as pd
from evaluate_single import load_models, evaluate_single, create_sample_input

def test_multiple_scenarios():
    """Test the evaluation system with different scenarios."""
    print("🧪 TESTING MULTIPLE SCENARIOS")
    print("=" * 60)
    
    # Load models once
    models_bundle = load_models()
    
    # Load some real samples from Final.csv
    df = pd.read_csv("../../Final.csv")
    
    # Test scenarios
    scenarios = [
        {"name": "Normal Sample", "filter": lambda x: x['anomaly_label'] == 0},
        {"name": "Anomaly Sample", "filter": lambda x: x['anomaly_label'] == 1},
        {"name": "High Severity", "filter": lambda x: x['severity_score'] > 0.8},
        {"name": "Low TTF", "filter": lambda x: x['TTF_km'] < 20000}
    ]
    
    results_summary = []
    
    for scenario in scenarios:
        print(f"\n🎯 Testing: {scenario['name']}")
        print("-" * 40)
        
        # Find matching samples
        matching_samples = []
        for idx, row in df.iterrows():
            if scenario['filter'](row):
                matching_samples.append(row)
                if len(matching_samples) >= 1:  # Just take first match
                    break
        
        if not matching_samples:
            print(f"  ⚠️ No samples found for {scenario['name']}")
            continue
            
        # Prepare input (remove target columns)
        sample_row = matching_samples[0]
        drop_cols = ["anomaly_label", "failure_type", "severity_stage", "severity_score", "TTF_years", "TTF_km", "is_gray", "source"]
        input_dict = {col: float(sample_row[col]) for col in df.columns if col not in drop_cols}
        
        # Run prediction
        predictions, details = evaluate_single(input_dict, models_bundle)
        
        # Show results
        print(f"  🔍 Actual vs Predicted:")
        print(f"    Anomaly: {int(sample_row['anomaly_label'])} → {predictions.get('anomaly_label', 'Error')}")

        # Handle severity score formatting
        pred_severity = predictions.get('severity_score', 'Error')
        if isinstance(pred_severity, (int, float)):
            print(f"    Severity Score: {sample_row['severity_score']:.3f} → {pred_severity:.3f}")
        else:
            print(f"    Severity Score: {sample_row['severity_score']:.3f} → {pred_severity}")

        print(f"    Severity Stage: {int(sample_row['severity_stage'])} → {predictions.get('severity_stage', 'Error')}")

        # Handle TTF formatting
        pred_ttf = predictions.get('ttf_km', 'Error')
        if isinstance(pred_ttf, (int, float)):
            print(f"    TTF (km): {sample_row['TTF_km']:.0f} → {pred_ttf:.0f}")
        else:
            print(f"    TTF (km): {sample_row['TTF_km']:.0f} → {pred_ttf}")
        
        # Store for summary
        results_summary.append({
            'scenario': scenario['name'],
            'actual_anomaly': int(sample_row['anomaly_label']),
            'predicted_anomaly': predictions.get('anomaly_label', -1),
            'actual_severity': sample_row['severity_score'],
            'predicted_severity': predictions.get('severity_score', -1),
            'actual_stage': int(sample_row['severity_stage']),
            'predicted_stage': predictions.get('severity_stage', -1),
            'actual_ttf': sample_row['TTF_km'],
            'predicted_ttf': predictions.get('ttf_km', -1)
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("output/scenario_test_summary.csv", index=False)
    print(f"\n💾 Summary saved to output/scenario_test_summary.csv")
    
    print(f"\n✅ Scenario testing complete!")

if __name__ == "__main__":
    test_multiple_scenarios()
