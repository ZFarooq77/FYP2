# 🔬 Catalytic Converter Live Fault Simulator

## Overview
Interactive Streamlit simulator for catalytic converter degradation using physics-based sensor modeling. Generates real-time synthetic data and accumulates samples for batch testing with trained ML models.

## Features
- **🎛️ Real-time Physics Simulation**: Uses exact three-zone sensor transform formulas
- **📊 Interactive Severity Control**: Slider from 0% (normal) to 100% (complete failure)
- **🤖 Model Integration**: Real-time inference with trained XGBoost, Stacking, and Gradient Boosting models
- **📋 Sample Accumulation**: Collect samples across different severity levels
- **📤 CSV Export**: Export accumulated samples to `Simulated_Runtime.csv`
- **📉 TTF Visualization**: Real-time Time-to-Failure degradation charts

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files exist:**
   - `../models/best_anomaly_model_xgboost.pkl`
   - `../models/severity_stacking_model_v2.pkl`
   - `../models/severity_stage_stacking_model_v1.pkl`
   - `../models/best_ttf_model_gradient_boosting.pkl`

3. **Ensure dataset exists:**
   - `../Final.csv` (for sensor statistics)

## Usage

**Run the simulator:**
```bash
streamlit run simulator_catalytic.py
```

**Workflow:**
1. 🎛️ Adjust severity slider (0-100%)
2. 📊 View real-time sensor values and model predictions
3. 🔄 Add samples to accumulation table
4. 📤 Export samples to CSV when ready
5. 🤖 Use exported CSV with enhanced `main.py` for batch model testing

## Physics Model

### Severity Zones
- **0-20%**: Normal (small jitter around baseline)
- **20-40%**: Mild anomaly (drift toward sensor limits)
- **40-60%**: Moderate anomaly (significant drift, clipped to bounds)
- **60-80%**: Severe anomaly (exceed normal bounds)
- **80-100%**: Complete failure (extreme values)

### Key Sensors
- Catalyst Temperature Bank1 Sensor1/2 (°C)
- Engine Load (%)
- Intake Manifold Pressure (kPa)
- Throttle Position (%)
- Long/Short Term Fuel Trim (%)
- Engine RPM Standard Deviation

### Physics Correlations
- Catalyst temp ↑ → Engine load ↓ (pumping losses)
- Catalyst temp ↑ → Throttle ↑ (compensation)
- CT1 ↔ CT2 correlation (inlet/outlet relationship)

## Output Files
- **`Simulated_Runtime.csv`**: Accumulated samples for model testing
- Contains all sensor values, severity metrics, and TTF predictions

## Integration with Main Pipeline
The exported CSV can be used with the enhanced `main.py` for comprehensive model validation:

```python
# Enhanced main.py will include:
def test_runtime_models(runtime_csv_path="simulator/Simulated_Runtime.csv"):
    # Load simulated data and run all trained models
    # Display predictions and performance metrics
```

## Troubleshooting
- **Models not loading**: Check file paths in `MODEL_PATHS` configuration
- **Dataset not found**: Ensure `Final.csv` exists in parent directory
- **Streamlit errors**: Install all requirements and check Python version (3.8+)
