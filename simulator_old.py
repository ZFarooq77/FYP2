# simulator_catalytic.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Catalytic Converter Clogging Simulator", layout="centered")
st.title("🚗 Catalytic Converter Clogging Simulator — Hybrid 3-Zone + Physics")
st.markdown(
    "Slider = choking severity (0% healthy → 100% fully clogged). "
    "Zone 1 = Normal (near mean, inside dataset), "
    "Zone 2 = Mild/Moderate (drift toward dataset edges, still inside), "
    "Zone 3 = Severe/Failed (physics drift allowed to go beyond dataset bounds)."
)

# --- Sensor stats derived from your screenshot (min, mean, max)
# I extracted these numbers from the table you provided.
stats = {
    "CATALYST_TEMPERATURE_BANK1_SENSOR1": {"min": 24.0, "mean": 377.913306, "max": 644.9},
    "CATALYST_TEMPERATURE_BANK1_SENSOR2": {"min": 9.179, "mean": 77.659718, "max": 522.4},
    "ENGINE_LOAD": {"min": 0.0, "mean": 73.781546, "max": 98.824},
    "INTAKE_MANIFOLD_PRESSURE": {"min": 8.627451, "mean": 39.516693, "max": 101.0},
    "THROTTLE": {"min": 0.0, "mean": 32.315923, "max": 100.0},
    "LONG_TERM_FUEL_TRIM_BANK_1": {"min": -20.3125, "mean": 0.101518, "max": 14.84375},
    "SHORT_TERM_FUEL_TRIM_BANK_1": {"min": -19.53125, "mean": 29.269006, "max": 102.0},
    "ENGINE_RPM": {"min": 0.0, "mean": 272.760984, "max": 2500.0}
}

# Helper: robust rng
seed = st.sidebar.number_input("Random seed (repeatable)", value=42, step=1)
rng = np.random.RandomState(int(seed))

# UI: choking slider
choke_pct = st.slider("Choking Severity (%)", min_value=0, max_value=100, value=0, step=1)
c = float(choke_pct) / 100.0

# --- Physics + three-zone hybrid simulation function ---
def simulate_catalytic(stats, c, rng):
    """
    c in [0,1]
    Zone definitions:
      - c <= 0.2 : Normal (tiny jitter around mean, keep inside min/max)
      - 0.2 < c <= 0.6 : Mild/Moderate (physics drift toward min/max, clipped to min/max)
      - c > 0.6 : Severe/Failed (physics drift allowed to exceed min/max)
    Returns: dict of sensor_name -> simulated_value
    """
    svals = {}

    # Choose exponents and multipliers tuned to produce plausible magnitudes
    temp_exp = 1.2
    temp_mult_mild = 0.6   # fraction of (max-mean) used in mild/moderate
    temp_mult_severe = 1.4 # multiplier of (max-mean) used beyond bounds for severe
    pressure_mult = 0.9
    load_mult = 30.0
    throttle_mult = 20.0
    trim_mult = 10.0
    rpm_jitter_scale = 50.0

    if c <= 0.2:
        # Zone 1: normal — tiny jitter (~1% rel)
        for k, v in stats.items():
            mean = float(v["mean"])
            jitter = rng.normal(0, 0.01 * max(abs(mean), 1.0))
            val = mean + jitter
            # ensure inside dataset min/max
            val = float(np.clip(val, v["min"], v["max"]))
            svals[k] = val

    elif c <= 0.6:
        # Zone 2: mild/moderate — physics drift but clipped to dataset bounds
        severity = (c - 0.2) / 0.4  # 0..1 within this zone
        for k, v in stats.items():
            mean = float(v["mean"])
            mn = float(v["min"])
            mx = float(v["max"])

            # sensor-specific physics
            if "CATALYST_TEMPERATURE_BANK1_SENSOR1" == k:
                # rise toward max (nonlinear)
                drift = ( (mx - mean) * temp_mult_mild ) * (severity ** temp_exp)
                val = mean + drift
            elif "CATALYST_TEMPERATURE_BANK1_SENSOR2" == k:
                drift = ( (mx - mean) * (temp_mult_mild * 0.9) ) * (severity ** temp_exp)
                val = mean + drift
            elif "INTAKE_MANIFOLD_PRESSURE" == k:
                # pressure decreases toward min (reduced vacuum)
                drift = ( (mean - mn) * pressure_mult ) * (severity ** 1.1)
                val = mean - drift
            elif "ENGINE_LOAD" == k:
                val = mean - (load_mult * severity)  # drop but will be clipped
            elif "THROTTLE" == k:
                val = mean + (throttle_mult * severity)
            elif "LONG_TERM_FUEL_TRIM_BANK_1" == k or "SHORT_TERM_FUEL_TRIM_BANK_1" == k:
                # ECU adjusts trims; small drift toward either direction — choose sign by seed
                sign = rng.choice([-1, 1])
                val = mean + sign * (trim_mult * severity)
            elif "ENGINE_RPM" == k:
                # increase variability, but keep mean near baseline for mild zone
                val = mean + rng.normal(0, rpm_jitter_scale * severity * 0.5)
            else:
                # default linear drift toward edges
                if mean < (mn + mx) / 2:
                    val = mean + (mx - mean) * severity * 0.5
                else:
                    val = mean - (mean - mn) * severity * 0.5

            # Clip to dataset bounds (mild/moderate must remain inside)
            val = float(np.clip(val, mn, mx))
            svals[k] = val

    else:
        # Zone 3: severe/failed — apply same physics but allow exceeding bounds (no clipping)
        severity = (c - 0.6) / 0.4  # 0..1 within severe zone
        for k, v in stats.items():
            mean = float(v["mean"])
            mn = float(v["min"])
            mx = float(v["max"])

            if "CATALYST_TEMPERATURE_BANK1_SENSOR1" == k:
                # nonlinear large increase beyond max
                drift = ( (mx - mean) * temp_mult_severe ) * (severity ** (temp_exp))
                val = mean + drift
            elif "CATALYST_TEMPERATURE_BANK1_SENSOR2" == k:
                drift = ( (mx - mean) * (temp_mult_severe * 0.9) ) * (severity ** (temp_exp))
                val = mean + drift
            elif "INTAKE_MANIFOLD_PRESSURE" == k:
                # larger drop beyond min
                drift = ( (mean - mn) * (1.2) ) * (severity ** 1.1)
                val = mean - drift
            elif "ENGINE_LOAD" == k:
                val = mean - (load_mult * 1.5 * (0.5 + severity))  # can drop well below min
            elif "THROTTLE" == k:
                val = mean + (throttle_mult * 1.5 * (0.5 + severity))  # may exceed 100
            elif k in ("LONG_TERM_FUEL_TRIM_BANK_1", "SHORT_TERM_FUEL_TRIM_BANK_1"):
                sign = rng.choice([-1, 1])
                val = mean + sign * (trim_mult * 2.0 * (0.5 + severity))
            elif "ENGINE_RPM" == k:
                # big variability, mean may be pushed by engine behavior
                val = mean + rng.normal(0, rpm_jitter_scale * (1.0 + severity))
            else:
                # default: push beyond bounds
                # push to beyond max or below min randomly
                if rng.rand() > 0.5:
                    val = mx + (mx - mean) * severity * 0.8
                else:
                    val = mn - (mean - mn) * severity * 0.8

            svals[k] = float(val)

    return svals

# Generate simulated values
sim = simulate_catalytic(stats, c, rng)

# Build display table including min/max and whether it's out-of-range
rows = []
out_of_range_count = 0
for k, v in sim.items():
    mn = stats[k]["min"]
    mx = stats[k]["max"]
    in_range = (v >= mn) and (v <= mx)
    if not in_range:
        out_of_range_count += 1
    rows.append({
        "Sensor": k,
        "Value": f"{v:.3f}",
        "Mean": f"{stats[k]['mean']:.3f}",
        "Min": f"{mn:.3f}",
        "Max": f"{mx:.3f}",
        "Status": "Normal" if in_range else "OUT-OF-RANGE"
    })

df_display = pd.DataFrame(rows)

# Render robust HTML table with highlight for out-of-range sensors
def render_html_table(df):
    cols = ["Sensor", "Value", "Mean", "Min", "Max", "Status"]
    html = "<table style='border-collapse:collapse; width:95%; font-family: Arial; font-size:13px;'>"
    html += "<thead><tr>"
    for c in cols:
        html += f"<th style='text-align:left; padding:8px; border-bottom:1px solid #ddd'>{c}</th>"
    html += "</tr></thead><tbody>"
    for _, r in df.iterrows():
        bg = "#fff"
        txt = "#000"
        if r["Status"] != "Normal":
            bg = "#ff4d4d"  # red
            txt = "#fff"
        html += f"<tr style='background:{bg}; color:{txt};'>"
        for c in cols:
            html += f"<td style='padding:8px; border-bottom:1px solid #eee'>{r[c]}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

st.subheader("📊 Simulated Sensor Readings")
st.markdown(render_html_table(df_display), unsafe_allow_html=True)

# Summary and quick chart
st.markdown(f"**Out-of-range sensors:** {out_of_range_count} / {len(df_display)}")
st.subheader("📈 Quick Sensor Overview")
plot_series = pd.Series({k: float(v) for k, v in sim.items()})
st.bar_chart(plot_series)

# Diagnostic narrative using three-zone buckets
def stage_text(pct):
    if pct <= 20:
        return "Normal"
    elif pct <= 40:
        return "Mild"
    elif pct <= 60:
        return "Moderate"
    elif pct <= 80:
        return "Severe"
    else:
        return "Failed"

stage = stage_text(choke_pct)
st.subheader("📝 Diagnostic Narrative")
if stage == "Normal":
    st.success("Normal: sensors near dataset mean, all inside expected bounds.")
elif stage == "Mild":
    st.info("Mild: sensors drifting toward normal edges (still within dataset bounds). Watch and monitor.")
elif stage == "Moderate":
    st.warning("Moderate: stronger drift toward bounds; performance may degrade.")
elif stage == "Severe":
    st.error("Severe: sensors moving beyond normal bounds; serious degradation.")
else:
    st.error("Failed: significant anomalies — sensors clearly outside normal operating envelope. Immediate service needed.")

st.caption("Hybrid model: Normal = mean ± jitter (in-range); Mild/Moderate = physics drift toward edges (clipped in-range); Severe/Failed = physics drift allowed outside dataset min/max (true anomaly).")
