
# %% [markdown]
# # Metadata Analysis

# %% [markdown]
# Here we converted all columns to float 64 and dropped WARM_UPS_SINCE_CODES_CLEARED (). total columns are 27. after dropping 26.

# %%
import pandas as pd
import numpy as np
import os
import re

# safe display: works in both notebooks and .py runs
try:
    from IPython.display import display as _ipy_display
    def show(x):
        _ipy_display(x)
except Exception:
    def show(x):
        # fallback for non-notebook contexts
        if isinstance(x, pd.DataFrame):
            print(x.to_string(max_rows=10, max_cols=200))
        else:
            print(x)

file_path = r"merged_datasets2.csv"

print(f"Reading: {file_path}")
print("Exists?", os.path.exists(file_path))

# Check if the file exists before attempting to load it
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
else:
    # Load dataset
    df = pd.read_csv(file_path, index_col=False, low_memory=False)

    # Drop any accidental index column like "Unnamed: 0"
    for col in list(df.columns):
        if str(col).startswith("Unnamed:"):
            df = df.drop(columns=[col])

    # Ensure the column name is consistent
    if df.columns[0] != "ENGINE_RUN_TINE ()":
        df = df.rename(columns={df.columns[0]: "ENGINE_RUN_TINE ()"})

    # Drop unnecessary column if it exists (kept exactly as you had it)
    if "WARM_UPS_SINCE_CODES_CLEARED ()" in df.columns:
        df = df.drop(columns=["WARM_UPS_SINCE_CODES_CLEARED ()"])

    # --- Convert ENGINE_RUN_TINE to numeric (your original fix) ---
    col = "ENGINE_RUN_TINE ()"
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"[^\d\.\-]", "", regex=True)  # keep only digits, dot, minus
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Convert ALL remaining columns to float64 safely ---
    def clean_to_numeric(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        s = s.str.replace(r"[^\d\.\-]+", "", regex=True)  # remove %, units, commas, etc.
        return pd.to_numeric(s, errors="coerce")

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64")
        else:
            df[c] = clean_to_numeric(df[c]).astype("float64")

    # Replace inf with NaN just in case
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Remove duplicate rows ---
    print(f"\n--- Removing Duplicates ---")
    print(f"Original shape: {df.shape}")
    print(f"Duplicated rows: {df.duplicated().sum()}")

    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(keep='first')

    print(f"Shape after removing duplicates: {df.shape}")
    print(f"Rows removed: {df.duplicated().sum()}")

    # Save the cleaned dataset without duplicates
    cleaned_filename = "merged_datasets2_no_duplicates.csv"
    df.to_csv(cleaned_filename, index=False)
    print(f"Saved cleaned dataset (no duplicates) as: {cleaned_filename}")

    # --- Show results (always visible) ---

    print("\n--- Top rows of df ---")
    show(df.head())

    print("\n--- Metadata & Statistics ---")
    print(f"Shape: {df.shape}")
    show(df.describe())

    print("\n--- Dtypes (expect all float64) ---")
    print(df.dtypes)

    print("\nDone.")


# %%
# Display metadata information of the live df dataframe
df.info()

# %% [markdown]
# # Explortory Data Analysis (EDA)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration flag to control figure generation
GENERATE_FIGURES = False  # Set to True for analysis, False for production

# --- Data Loading (already done in a previous cell, using the 'df' dataframe) ---
print("--- Using the loaded 'df' Dataset ---")



print("live df dataset shape:", df.shape)
print("\n")

# --- Statistical Summary (already done in a previous cell) ---
print("--- Statistical Summary of live Dataset ---")
print("live1.csv Metadata & Statistics:")
print(df.describe(include='all'))
print("\n")

# --- Data Cleaning for Correlation and Plotting ---
# Convert 'ABSOLUTE_BAROMETRIC_PRESSURE ()' to numeric, coercing errors
df['ABSOLUTE_BAROMETRIC_PRESSURE ()'] = pd.to_numeric(df['ABSOLUTE_BAROMETRIC_PRESSURE ()'], errors='coerce')

# Drop rows with any missing values (NaN) for correlation calculation and plotting
# This is done on a copy to keep the original combined_df intact for other potential uses
# Since we are only using 'df', we can dropna directly or create a copy if 'df' needs to be preserved
df_cleaned = df.dropna().copy()


print(f"Shape of cleaned data for plotting: {df_cleaned.shape}")
print("\n")


# --- Correlation Analysis ---
print("--- Correlation Analysis ---")
correlation_matrix = df_cleaned.corr(numeric_only=True)
# Get the specific correlation values
rpm_throttle_corr = correlation_matrix.loc['ENGINE_RPM ()', 'THROTTLE ()']
rpm_speed_corr = correlation_matrix.loc['ENGINE_RPM ()', 'VEHICLE_SPEED ()']
throttle_speed_corr = correlation_matrix.loc['THROTTLE ()', 'VEHICLE_SPEED ()']
load_temp_corr = correlation_matrix.loc['ENGINE_LOAD ()', 'COOLANT_TEMPERATURE ()']

# Print the values
print(f"Correlation between ENGINE_RPM() and THROTTLE(): {rpm_throttle_corr}")
print(f"Correlation between ENGINE_RPM() and VEHICLE_SPEED(): {rpm_speed_corr}")
print(f"Correlation between THROTTLE() and VEHICLE_SPEED(): {throttle_speed_corr}")
print(f"Correlation between ENGINE_LOAD() and COOLANT_TEMPERATURE(): {load_temp_corr}")
print("\n")

# --- Visualizations ---
print("--- Generating Visualizations ---")

# Generate and plot a correlation heatmap
if GENERATE_FIGURES:
    plt.figure(figsize=(15, 12))
    sns.heatmap(df_cleaned.corr(numeric_only=True), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Vehicle Sensor Data (Cleaned Data)')
    plt.savefig('correlation_heatmap.png')
    plt.show()

# Plot key time-series data
if GENERATE_FIGURES:
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    # Sort data by run time for correct time series plotting
    df_cleaned_sorted = df_cleaned.sort_values(by='ENGINE_RUN_TINE ()')

    axes[0].plot(df_cleaned_sorted['ENGINE_RUN_TINE ()'], df_cleaned_sorted['ENGINE_RPM ()'])
    axes[0].set_title('Engine RPM Over Time')
    axes[0].set_xlabel('Engine Run Time (s)')
    axes[0].set_ylabel('Engine RPM')

    axes[1].plot(df_cleaned_sorted['ENGINE_RUN_TINE ()'], df_cleaned_sorted['ENGINE_LOAD ()'])
    axes[1].set_title('Engine Load Over Time')
    axes[1].set_xlabel('Engine Run Time (s)')
    axes[1].set_ylabel('Engine Load (%)')

    axes[2].plot(df_cleaned_sorted['ENGINE_RUN_TINE ()'], df_cleaned_sorted['COOLANT_TEMPERATURE ()'])
    axes[2].set_title('Coolant Temperature Over Time')
    axes[2].set_xlabel('Engine Run Time (s)')
    axes[2].set_ylabel('Coolant Temperature $(^\circ C)$') # Corrected escape sequence
    plt.tight_layout()
    plt.savefig('time_series_plots.png')
    plt.show()

# Plot scatter plot
if GENERATE_FIGURES:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='THROTTLE ()', y='ENGINE_LOAD ()', hue='VEHICLE_SPEED ()', data=df_cleaned, palette='viridis', alpha=0.6)
    plt.title('Engine Load vs. Throttle Position by Vehicle Speed')
    plt.xlabel('Throttle Position (%)')
    plt.ylabel('Engine Load (%)')
    plt.legend(title='Vehicle Speed', loc='upper left')
    plt.savefig('scatter_plot.png')
    plt.show()

# %%
# Transpose the statistical summary of the df dataframe
print("--- Transposed Dataset Metadata & Statistics ---")
print(df.describe(include='all').T)
print(df.shape)

# %% [markdown]
# # 3.5.1 Sliding Window and Statistical Feature Extraction

# Old windowing approach removed - using optimized version below
    

# %% [markdown]
# ### Windowing 2nd approach

# %%
#SECOND APPROACH merging + windowing
import pandas as pd
import numpy as np
import glob
import os

# -----------------------------
# Helper: rate of change
# -----------------------------
def calculate_rate_of_change(series, time_index=None):
    valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_series) < 2:
        return 0.0
    y = valid_series.values
    if np.all(y == y[0]):
        return 0.0
    if time_index is not None:
        x = time_index.loc[valid_series.index].values
    else:
        x = np.arange(len(valid_series))
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except np.linalg.LinAlgError:
        return 0.0

# -----------------------------
# Feature extraction for one window
# -----------------------------
def extract_window_features(window_data, source_column, time_column):
    window_features = {}
    window_features[source_column] = (
        window_data[source_column].iloc[0] if source_column in window_data.columns else "unknown"
    )
    numeric_cols = window_data.select_dtypes(include=np.number).columns.tolist()
    if time_column in numeric_cols:
        numeric_cols.remove(time_column)
    # Remove WARM_UPS_SINCE_CODES_CLEARED column if it exists
    warmup_col = "WARM_UPS_SINCE_CODES_CLEARED ()"
    if warmup_col in numeric_cols:
        numeric_cols.remove(warmup_col)
    for col in numeric_cols:
        series = window_data[col].dropna()
        if series.empty:
            window_features[f"{col}_mean"] = np.nan
            window_features[f"{col}_std"]  = np.nan
            window_features[f"{col}_max"]  = np.nan
            window_features[f"{col}_min"]  = np.nan
            window_features[f"{col}_roc"]  = np.nan
        else:
            window_features[f"{col}_mean"] = series.mean()
            window_features[f"{col}_std"]  = series.std()
            window_features[f"{col}_max"]  = series.max()
            window_features[f"{col}_min"]  = series.min()
            window_features[f"{col}_roc"]  = calculate_rate_of_change(series, time_index=window_data[time_column])
    return window_features

# -----------------------------
# Sliding window function
# -----------------------------
def create_sliding_windows(df, window_size, time_column, source_column,
                           step_size=None, use_time=True, per_source=True):
    if step_size is None:
        step_size = window_size
    features_list = []

    def windows_for(df_block, label=None):
        nonlocal features_list
        df_sorted = df_block.sort_values(by=time_column).reset_index(drop=True)
        if use_time:
            start_time = df_sorted[time_column].min()
            end_time   = df_sorted[time_column].max()
            current    = start_time
            # same bound you used:
            while current + window_size <= end_time:
                nxt = current + window_size
                window_data = df_sorted[(df_sorted[time_column] >= current) &
                                        (df_sorted[time_column] <  nxt)]
                if not window_data.empty:
                    features_list.append(extract_window_features(window_data, source_column, time_column))
                current += step_size
            return start_time, end_time
        else:
            total_rows = len(df_sorted)
            i = 0
            while i < total_rows:
                j = i + window_size
                window_data = df_sorted.iloc[i:j]
                if not window_data.empty:
                    features_list.append(extract_window_features(window_data, source_column, time_column))
                i += step_size
            return None, None

    if per_source:
        for src, df_src in df.groupby(source_column):
            windows_for(df_src, label=src)
    else:
        windows_for(df, label=None)

    return pd.DataFrame(features_list)

# -----------------------------
# Merge CSV files + apply windowing
# -----------------------------
def merge_and_window(input_path, output_filepath, time_column, source_column,
                     window_size=15, step_size=10, use_time=True,
                     per_source=True, overwrite=False):
    """
    Merge all CSVs in folder, add source, apply sliding windows, save result.
    - Skip recompute if output exists unless overwrite=True.
    """
    # --- Skip if we already have it
    if (not overwrite) and os.path.exists(output_filepath):
        print(f"✅ Found existing windowed dataset: {output_filepath} — skipping recompute.")
        try:
            cached = pd.read_csv(output_filepath, low_memory=False)
            print(f"Loaded existing: shape={cached.shape}")
            return cached
        except Exception as e:
            print(f"Could not load existing file ({e}); recomputing...")

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return None

    all_dfs = []
    for file in sorted(csv_files):
        try:
            temp_df = pd.read_csv(file, low_memory=False)
            temp_df[source_column] = os.path.splitext(os.path.basename(file))[0]  # e.g., 'live1'
            all_dfs.append(temp_df)
            print(f"Loaded {os.path.basename(file):<20} rows={len(temp_df)}")
        except Exception as e:
            print(f"Skipping {file}, error: {e}")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # ensure time numeric and drop bad
    merged_df[time_column] = pd.to_numeric(merged_df[time_column], errors="coerce")
    merged_df = merged_df.dropna(subset=[time_column]).copy()

    # Small sanity print: per-source duration and estimated windows
    if use_time:
        print("\nPer-source durations & estimated window counts:")
        for src, g in merged_df.groupby(source_column):
            tmin, tmax = g[time_column].min(), g[time_column].max()
            n_est = max(0, int(np.floor((tmax - tmin - window_size) / step_size)) + 1)
            print(f"{src:<10}  span={tmax - tmin:6.1f}  est_windows={n_est}")

    windowed_df = create_sliding_windows(
        merged_df,
        window_size=window_size,
        step_size=step_size,
        time_column=time_column,
        source_column=source_column,
        use_time=use_time,
        per_source=per_source
    )

    windowed_df.to_csv(output_filepath, index=False)
    print(f"\n✅ Saved windowed dataset to {output_filepath}, shape={windowed_df.shape}")
    return windowed_df

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    input_folder = r"G:\BSCS\Semester 7\FYP\codes\Main\data\raw"
    output_file  = r"windowed_merged_dataset.csv"

    time_column   = "ENGINE_RUN_TINE ()"   # check exact header
    source_column = "source"
    window_size   = 15
    step_size     = 10
    use_time      = True

    # per_source=True  → many more windows (sum across each liveN)
    # per_source=False → treat all data as one timeline (usually far fewer windows)
    windowed_dataset = merge_and_window(
        input_folder,
        output_file,
        time_column,
        source_column,
        window_size,
        step_size,
        use_time,
        per_source=True,        # set to False if you want ~277-ish total windows
        overwrite=False         # set True to force recompute
    )

    print("\nHead of windowed data:")
    print(windowed_dataset.head().to_string(index=False) if windowed_dataset is not None else "None")


# %% [markdown]
# ### Windowing Date Analysis
# 
# The `windowed_features_df` DataFrame contains features extracted from the combined vehicle sensor data using a 60-second sliding window.
# 
# *   **Shape:** The DataFrame has been transformed into a shape of **(32, 126)**, meaning there are 32 windows (observations) and 126 features per window.
# *   **Columns:** The DataFrame includes features such as the mean, standard deviation, maximum, minimum, and rate of change for various sensor readings within each time window. It also includes a 'source' column to indicate whether the data in the window originated from the 'drive' or 'idle' dataset.
# *   **Data Types:** The majority of the feature columns are of numerical data types (float64), while the 'source' column is an object type (string).

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Display the shape and info of the windowed features dataframe
# print("Shape of the windowed features dataframe:", windowed_dataset.shape)
# print("\nInfo of the windowed features dataframe:")
# print(windowed_dataset.info())

# Visualize the distribution of a few key features, differentiating by source
# if GENERATE_FIGURES:
#     features_to_visualize = [
#         'ENGINE_RPM ()_mean',
#         'VEHICLE_SPEED ()_mean',
#         'THROTTLE ()_mean',
#         'ENGINE_LOAD ()_mean',
#         'COOLANT_TEMPERATURE ()_mean'
#     ]

#     plt.figure(figsize=(15, 10))
#     for i, feature in enumerate(features_to_visualize):
#         plt.subplot(2, 3, i + 1)
#         # The hue will now only show 'unknown' as the source
#         sns.histplot(data=windowed_dataset, x=feature, hue='source', kde=True)
#         plt.title(f'Distribution of {feature}')
#     plt.tight_layout()
#     plt.show()

#     # Plot a scatter plot of two relevant features, colored by source
#     plt.figure(figsize=(10, 6))
#     # The hue will now only show 'unknown' as the source
#     sns.scatterplot(data=windowed_dataset, x='ENGINE_RPM ()_mean', y='VEHICLE_SPEED ()_mean', hue='source')
#     plt.title('Mean Engine RPM vs. Mean Vehicle Speed by Source')
#     plt.xlabel('Mean Engine RPM')
#     plt.ylabel('Mean Vehicle Speed')
#     plt.show()

# Scaling is handled in the windowing function - see merge_and_window_csvs()


# %% [markdown]
# # 3.6 Anomaly Injection Implementation
# 

# %% [markdown]
# # Task
# Generate Python code to implement anomaly injection into the scaled and windowed dataset based on the provided methodology for engine component wear and fuel efficiency use cases. Create a new column 'anomaly_label' to mark injected anomalies with 1 and normal data with 0. Also, generate a markdown text cell describing the anomaly injection process.

# %% [markdown]
# ## Create a copy of the scaled features dataframe
# 
# ### Subtask:
# Create a copy of the scaled features dataframe.
# 

# %%
# Create a copy of the scaled features dataframe for anomaly injection
# scaled_features_anomalies_df = scaled_features_df.copy()

# # Display the head of the new dataframe
# print(scaled_features_anomalies_df.head())
# scaled_features_anomalies_df.to_csv(r"scaled_features_anomalies_df.csv", index=False)

# %% [markdown]
# ## Define and implement anomaly injection rules for engine component wear
# 
# ### Subtask:
# Define and implement anomaly injection rules for engine component wear by identifying windows that represent "normal" driving or idling conditions and simulating gradual mechanical failure by modifying relevant features within a selected subset of these windows in the copied DataFrame.
# 

# %% [markdown]
# **Reasoning**:
# Select normal windows and define anomaly injection rules for engine component wear by modifying relevant features in a subset of these windows in the copied DataFrame.
# 
# 

# %% [markdown]
# # Multi-class Anomaly Injection Code

# %% [markdown]
# ### What this code does
# 
# 1. Setup label columns
# 
# - anomaly_label → 0 (normal), 1 (anomaly)
# 
# - failure_type → 0 = normal, 1–6 = fault types
# 
# - severity_score → continuous 0–1 
#                                     {0.0 → Normal (no anomaly)
# 
#                                     0.2 → Mild
# 
#                                     0.4–0.5 → Moderate
# 
#                                     0.7 → Severe
# 
#                                     1.0 → Complete failure}
# 
# - severity_stage → discrete {0 Normal, 1 Mild, 2 Moderate, 3 Severe, 4 Failed}
# 
# 2. Candidate selection
# 
# - Picks “normal” windows by filtering out extremes in RPM and speed (15–85% quantiles).
# 
# - Shuffles them, then splits into 6 groups (EGR, InjectorRich, InjectorLean, TurboUnder, TurboOver, Catalytic).
# 
# 3. Injection
# 
# - For each group, modifies certain sensor features gradually (controlled by score).
# 
# Example:
# 
# EGR clogged → coolant temp ↑, manifold pressure ↑, timing advance ↓
# 
# Injector rich → fuel trims ↑, catalyst temp ↑
# 
# Injector lean → trims ↓, RPM_std ↑
# 
# Turbo underboost → manifold pressure ↓, load ↑
# 
# Turbo overboost → manifold pressure ↑, trims ↓
# 
# Catalytic clogging → catalyst temps ↑, load ↓

# Old multi-class anomaly injection removed - using enhanced physics-based approach below


# %% Enhanced Catalytic Converter Anomaly Injection
"""
Improved catalytic converter anomaly injection (hybrid 3-zone + physics-inspired).

Features:
 - Appends synthetic anomaly rows (preserves original normals).
 - Three zones: Normal (in-range), Mild/Moderate (drift to edges, clipped), Severe/Failed (drift beyond bounds).
 - Nonlinear drifts, correlated sensor adjustments, progressive worsening, TTF estimation.
 - Gray zone anomalies for robustness testing.
 - Adds metadata for each injection.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# ---------------------------
# Utility helpers
# ---------------------------
def map_severity_to_stage(score: float) -> int:
    """Map continuous severity (0..1) to discrete stage."""
    if score <= 0:
        return 0
    if score <= 0.25:
        return 1
    if score <= 0.5:
        return 2
    if score <= 0.75:
        return 3
    return 4

def map_severity_to_stage_v2(score: float, fuzziness: float = 0.05, rng=None) -> int:
    """
    Enhanced fuzzy mapping from continuous severity (0..1) to discrete stage (0–4).

    Improvements:
    - Reduces label noise near stage boundaries
    - Introduces fuzzy stage transitions (soft probability mapping)
    - Retains ordinal structure (0 < 1 < 2 < 3 < 4)
    - Allows mild drift in normal samples
    - Ensures backward compatibility

    Args:
        score: Continuous severity score [0.0, 1.0]
        fuzziness: Controls transition smoothness (0.01-0.1 recommended)
        rng: Random number generator for reproducibility

    Returns:
        Discrete stage [0, 1, 2, 3, 4]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base thresholds with improved boundaries
    thresholds = [0.0, 0.20, 0.45, 0.70, 1.0]  # Slightly adjusted for better distribution

    # Clip input for safety
    score = np.clip(score, 0.0, 1.0)

    # Handle pure normal case
    if score <= 0.05:  # Allow mild drift in normal samples
        return 0

    # Find appropriate stage with fuzzy transitions
    for i in range(1, len(thresholds)):
        lower, upper = thresholds[i-1], thresholds[i]

        if score <= upper:
            # Calculate distance from boundary
            boundary = upper
            dist_from_boundary = abs(score - boundary)

            # Fuzzy transition probability (Gaussian-like)
            if dist_from_boundary <= fuzziness * 2:  # Within fuzzy zone
                # Probability of staying in current stage vs moving to next
                transition_prob = np.exp(-((dist_from_boundary / fuzziness) ** 2))

                # Add some randomness for boundary cases
                if rng.random() < transition_prob * 0.3:  # 30% chance of fuzzy transition
                    # Decide between current and adjacent stage
                    if score < boundary and i > 1:
                        return i - 1  # Stay in lower stage
                    elif score > boundary and i < len(thresholds) - 1:
                        return min(i, 4)  # Move to higher stage (capped at 4)

            # Fixed: Return correct stage index
            if i == len(thresholds) - 1:  # Last threshold (1.0)
                return 4 if score > thresholds[-2] else i - 1
            else:
                return i - 1  # Default to current stage

    return 4  # Maximum stage

def map_severity_to_stage_enhanced(score: float, use_fuzzy: bool = True, fuzziness: float = 0.05, rng=None) -> int:
    """
    Wrapper function for backward compatibility and easy switching between versions.

    Args:
        score: Continuous severity score [0.0, 1.0]
        use_fuzzy: Whether to use fuzzy transitions (True) or classic mapping (False)
        fuzziness: Fuzzy transition parameter
        rng: Random number generator

    Returns:
        Discrete stage [0, 1, 2, 3, 4]
    """
    if use_fuzzy:
        return map_severity_to_stage_v2(score, fuzziness, rng)
    else:
        return map_severity_to_stage(score)  # Classic version

def safe_get_stats(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """Return min/mean/max for a column if present, else None."""
    if col not in df.columns:
        return None
    return {"min": float(df[col].min()), "mean": float(df[col].mean()), "max": float(df[col].max())}

def compute_hybrid_ttf_physics(df, base_years=4.0, avg_km_per_year=15000):
    """
    Compute hybrid TTF (years + km) using enhanced physics model.

    Args:
        df: DataFrame with sensor data and severity columns
        base_years: Expected catalytic converter lifetime (years)
        avg_km_per_year: Average yearly driving distance (km)

    Returns:
        DataFrame with TTF_years and TTF_km columns added
    """
    df = df.copy()

    # Helper to safely access sensor columns
    def safe_get(col_pattern):
        matches = [c for c in df.columns if col_pattern.upper() in c.upper()]
        return df[matches[0]] if matches else pd.Series(np.nan, index=df.index)

    # Extract key sensors
    cat_temp1 = safe_get("CATALYST_TEMPERATURE_BANK1_SENSOR1")
    cat_temp2 = safe_get("CATALYST_TEMPERATURE_BANK1_SENSOR2")
    engine_load = safe_get("ENGINE_LOAD")
    manifold_pressure = safe_get("INTAKE_MANIFOLD_PRESSURE")
    throttle = safe_get("THROTTLE")

    # Get severity columns (with defaults)
    severity_score = df.get("severity_score", pd.Series(0.0, index=df.index))
    severity_stage = df.get("severity_stage", pd.Series(0, index=df.index))

    # Internal min-max normalization (doesn't modify original df)
    def minmax_safe(series):
        if series.isna().all():
            return series
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)  # neutral value
        return (series - min_val) / (max_val - min_val)

    # Normalize sensors for physics calculations
    cat1_scaled = minmax_safe(cat_temp1)
    cat2_scaled = minmax_safe(cat_temp2)
    engine_load_scaled = minmax_safe(engine_load)
    map_scaled = minmax_safe(manifold_pressure)
    throttle_scaled = minmax_safe(throttle)

    # Enhanced physics degradation model
    cat_health = 0.6 * cat1_scaled + 0.4 * cat2_scaled
    thermal_aging = 0.6 * cat_health + 0.2 * engine_load_scaled + 0.2 * throttle_scaled
    usage_aging = 0.5 * engine_load_scaled + 0.3 * throttle_scaled - 0.2 * map_scaled

    # Combined hybrid degradation (0 = healthy, 1 = high wear)
    degradation = 0.5 * thermal_aging + 0.5 * usage_aging

    # Severity-Physics Hybrid TTF Formula
    severity_stage_norm = severity_stage / 4.0  # normalize 0-4 → 0-1

    df["TTF_years"] = (1 - (0.6 * severity_score + 0.25 * severity_stage_norm + 0.15 * degradation)) * base_years
    df["TTF_years"] = df["TTF_years"].clip(lower=0.0, upper=base_years)

    # Convert years → km (distance-based TTF)
    df["TTF_km"] = df["TTF_years"] * avg_km_per_year
    df["TTF_km"] = df["TTF_km"].clip(lower=0.0, upper=base_years * avg_km_per_year)

    # time_to_failure_years column removed - only keep TTF_years and TTF_km

    print(f"✅ Hybrid TTF computed: TTF_years (0-{base_years}), TTF_km (0-{base_years * avg_km_per_year:,})")
    return df

def severity_to_ttf_years(sev: float, min_months: float = 2.4, max_years: float = 5.0) -> float:
    """
    Convert severity in [0,1] to a rough time-to-failure estimate.
    Higher severity -> smaller remaining life.
    Returns values in years, but for high severity (>0.8) returns months converted to years.

    Args:
        sev: Severity score [0,1]
        min_months: Minimum time in months for severe cases (converted to years)
        max_years: Maximum time in years for mild cases

    Returns:
        Time to failure in years (may be fractional for months)
    """
    if sev >= 0.8:
        # High severity: return months converted to years (2.4 to 12 months)
        months = np.interp(sev, [0.8, 1.0], [12.0, min_months])
        return float(round(months / 12.0, 3))  # Convert months to years
    else:
        # Lower severity: return years (1 to 5 years)
        return float(np.interp(sev, [0.0, 0.8], [max_years, 1.0]))

# ---------------------------
# Core three-zone transform for a single sensor
# ---------------------------
def three_zone_sensor_transform(
    baseline_value: float,
    sensor_stats: Dict[str, float],
    severity: float,
    role: str,
    rng: np.random.RandomState,
    zone1_max: float = 0.2,
    zone2_max: float = 0.6,
    temp_exp: float = 1.2,
    zone2_mult: float = 1.0,
    zone3_mult: float = 1.6
) -> float:
    """
    Apply three-zone transform for a single sensor.
    role indicates expected direction ("up" or "down") or special sensors like 'trim' or 'rpm_std'.
    sensor_stats must have 'min','mean','max' (in the same scaled space).
    Returns new_value (float). In zone2 values are clipped to min/max. In zone3 values can go beyond.
    """
    if sensor_stats is None:
        # no stats => return baseline + tiny jitter
        return baseline_value + rng.normal(0, 1e-6)

    mn = sensor_stats["min"]
    mean = sensor_stats["mean"]
    mx = sensor_stats["max"]
    data_range = mx - mn if (mx - mn) != 0 else 1.0

    # choose zone
    if severity <= zone1_max:
        # Zone 1: normal jitter (small)
        jitter = rng.normal(0, 0.01 * data_range)
        val = baseline_value + jitter
        return float(np.clip(val, mn, mx))

    elif severity <= zone2_max:
        # Zone 2: drift toward an edge; severity normalized inside this zone
        s2 = (severity - zone1_max) / (zone2_max - zone1_max)  # 0..1
        if role == "up":  # e.g., catalyst temp, throttle, trim (increase)
            drift = (mx - mean) * zone2_mult * (s2 ** temp_exp)
            val = baseline_value + drift
        elif role == "down":  # e.g., engine_load or timing_advance might go down
            drift = (mean - mn) * zone2_mult * (s2 ** temp_exp)
            val = baseline_value - drift
        elif role == "trim":
            # trims can go either way, but for catalytic clogging we bias upward
            drift = (mx - mean) * zone2_mult * (s2 ** (temp_exp * 0.9))
            val = baseline_value + drift + rng.normal(0, 0.01 * data_range)
        elif role == "rpm_std":
            drift = (mx - mean) * zone2_mult * (s2 ** 1.0)
            val = baseline_value + drift
        else:
            # default: push toward closer edge
            if mean < (mn + mx) / 2:
                # mean nearer min, push upward
                drift = (mx - mean) * zone2_mult * (s2 ** temp_exp)
                val = baseline_value + drift
            else:
                drift = (mean - mn) * zone2_mult * (s2 ** temp_exp)
                val = baseline_value - drift

        # clip inside bounds for mild/moderate
        return float(np.clip(val, mn, mx))

    else:
        # Zone 3: severe/failure: allow exceed bounds
        s3 = (severity - zone2_max) / (1.0 - zone2_max)  # 0..1 within severe zone
        if role == "up":
            overshoot = (mx - mean) * zone3_mult * (s3 ** temp_exp)
            val = baseline_value + overshoot + rng.normal(0, 0.02 * (mx - mean))
        elif role == "down":
            undershoot = (mean - mn) * zone3_mult * (s3 ** temp_exp)
            val = baseline_value - undershoot + rng.normal(0, 0.02 * (mean - mn))
        elif role == "trim":
            overshoot = (mx - mean) * zone3_mult * (s3 ** (temp_exp * 0.95))
            val = baseline_value + overshoot + rng.normal(0, 0.05 * (mx - mean))
        elif role == "rpm_std":
            overshoot = (mx - mean) * zone3_mult * (s3 ** 1.0)
            val = baseline_value + overshoot + rng.normal(0, 0.05 * (mx - mean))
        else:
            # default push beyond bounds randomly direction
            if rng.rand() > 0.5:
                val = mx + (mx - mean) * zone3_mult * s3
            else:
                val = mn - (mean - mn) * zone3_mult * s3

        return float(val)

# ---------------------------
# Correlation adjustments
# ---------------------------
def apply_correlations(row: pd.Series,
                       stats: Dict[str, Dict[str, float]],
                       rng=None):
    """
    Apply simple physics-inspired correlations between sensors after individual transforms.
    Examples:
      - catalyst temp up -> engine load down (pumping losses / inefficiency)
      - catalyst temp up -> throttle up (driver or ECU compensation)
    This function mutates and returns the row.
    """
    # Guard for presence
    name_ct1 = "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean"
    name_load = "ENGINE_LOAD ()_mean"
    name_throttle = "THROTTLE ()_mean"
    name_ct2 = "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean"

    if name_ct1 in row.index and name_load in row.index:
        mean_ct1 = stats[name_ct1]["mean"]
        mx_ct1 = stats[name_ct1]["max"]
        delta_ct1 = row[name_ct1] - mean_ct1
        norm_delta = delta_ct1 / (mx_ct1 - mean_ct1) if (mx_ct1 - mean_ct1) != 0 else 0.0
        # If catalyst temp increases, engine load tends to fall (scale down)
        row[name_load] = row[name_load] - (0.15 * norm_delta * (stats[name_load]["max"] - stats[name_load]["mean"]))
        # Throttle/injector compensation: small increase
        if name_throttle in row.index:
            row[name_throttle] = row[name_throttle] + (0.10 * max(norm_delta, 0.0) * (stats[name_throttle]["max"] - stats[name_throttle]["mean"]))
        # CT2 usually correlates with CT1 (inlet/outlet)
        if name_ct2 in row.index:
            row[name_ct2] = row[name_ct2] + 0.9 * delta_ct1  # keep roughly aligned

    # Ensure we did not produce NaNs
    return row

# ---------------------------
# Main enhanced injection function
# ---------------------------
def inject_catalytic_anomalies_realistic(
    scaled_df: pd.DataFrame,
    key_sensors: List[str] = None,
    n_frac: float = 0.05,
    base_severity: float = 0.5,
    create_trajectories: bool = False,
    trajectory_len: int = 3,
    gray_frac: float = 0.10,
    random_state: int = 42,
    use_fuzzy_stages: bool = True,
    stage_fuzziness: float = 0.05
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Append synthetic catalytic converter anomalies to `scaled_df` (which should be your windowed+scaled dataset).

    Enhanced Features:
    - Fuzzy severity stage transitions to reduce label noise
    - Improved boundary handling for better model training
    - Backward compatibility with classic stage mapping

    Args:
        scaled_df: Input windowed and scaled dataset
        key_sensors: List of key sensor columns to modify
        n_frac: Fraction of anomalies to inject
        base_severity: Base severity level for anomaly generation
        create_trajectories: Whether to create progressive degradation trajectories
        trajectory_len: Length of degradation trajectories
        gray_frac: Fraction of gray (ambiguous) anomalies
        random_state: Random seed for reproducibility
        use_fuzzy_stages: Whether to use fuzzy stage transitions (True) or classic mapping (False)
        stage_fuzziness: Fuzziness parameter for stage transitions (0.01-0.1 recommended)

    Returns: (df_with_injections, injections_metadata)
    """

    rng = np.random.RandomState(random_state)

    df_out = scaled_df.copy(deep=True).reset_index(drop=True)

    # Clean missing values in original data (fill with column means for numeric columns)
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_out[col].isna().any():
            missing_count = df_out[col].isna().sum()
            mean_val = df_out[col].mean()
            df_out[col] = df_out[col].fillna(mean_val)  # Use assignment instead of inplace
            print(f"  Filled {missing_count} missing values in {col} with mean {mean_val:.3f}")

    # ensure anomaly_label exists (reset to 0)
    df_out['anomaly_label'] = 0

    # Initialize anomaly-related columns for all rows
    df_out['failure_type'] = 0
    df_out['severity_score'] = 0.0
    df_out['severity_stage'] = 0

    df_out['is_gray'] = False
    if 'failure_type' not in df_out.columns:
        df_out['failure_type'] = 0
    # add columns to hold severity / stage / ttf
    df_out['severity_score'] = df_out.get('severity_score', 0.0)
    df_out['severity_stage'] = df_out.get('severity_stage', 0)

    # Initialize TTF columns to ensure templates have them (will be recalculated later)
    df_out['TTF_years'] = 0.0
    df_out['TTF_km'] = 0.0

    # default key sensors if not provided
    if key_sensors is None:
        key_sensors = [
            "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean",
            "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean",
            "ENGINE_LOAD ()_mean",
            "INTAKE_MANIFOLD_PRESSURE ()_mean",
            "THROTTLE ()_mean",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()_mean",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()_mean",
            "ENGINE_RPM ()_std"
        ]

    # compute stats for key sensors from the scaled df
    sensor_stats = {}
    for s in key_sensors:
        st = safe_get_stats(df_out, s)
        if st is not None:
            sensor_stats[s] = st

    # pick candidate templates (avoid extremes by RPM quantiles if available)
    if "ENGINE_RPM ()_mean" in df_out.columns:
        r_lo, r_hi = df_out["ENGINE_RPM ()_mean"].quantile([0.15, 0.85])
        candidates = df_out[df_out["ENGINE_RPM ()_mean"].between(r_lo, r_hi)].index.to_list()
    else:
        candidates = df_out.index.to_list()

    if len(candidates) == 0:
        raise RuntimeError("No template candidates found for anomaly creation.")

    n_anomalies = max(1, int(len(df_out) * n_frac))
    n_anomalies = min(n_anomalies, len(candidates))

    # choose template rows (random without replacement)
    templates = rng.choice(candidates, size=n_anomalies, replace=False)

    injections_meta = []

    synthetic_rows: List[pd.Series] = []

    # Helper to build a synthetic row from a template and severity
    def create_synthetic_from_template(template_idx: int, sev: float, injection_id: str) -> Tuple[pd.Series, Dict[str, Any]]:
        template = df_out.loc[template_idx].copy()
        sensors_touched = []
        out_of_bounds = 0

        # Update each sensor according to three-zone logic
        for s in key_sensors:
            if s not in template.index:
                continue
            role = "up"
            if s in ("ENGINE_LOAD ()_mean", "TIMING_ADVANCE ()_mean"):
                role = "down"
            elif "FUEL_TRIM" in s:
                role = "trim"
            elif "ENGINE_RPM ()_std" in s:
                role = "rpm_std"

            baseline_val = float(template[s])
            stats = sensor_stats.get(s)
            new_val = three_zone_sensor_transform(
                baseline_val, stats, sev, role, rng
            )
            template[s] = new_val
            sensors_touched.append(s)
            # out of range check (based on stats)
            if stats is not None:
                if (new_val < stats["min"]) or (new_val > stats["max"]):
                    out_of_bounds += 1

        # Apply small correlations (CT1 -> load, throttle, CT2)
        template = apply_correlations(template, sensor_stats, rng)

        # Add some small global noise so rows are never identical
        for col in template.index:
            if col not in ['anomaly_label', 'failure_type', 'severity_score', 'severity_stage']:
                # only tiny noise (0.2% of sensor spread)
                if col in sensor_stats:
                    spread = sensor_stats[col]["max"] - sensor_stats[col]["min"]
                    template[col] = float(template[col] + rng.normal(0, 0.002 * (spread if spread != 0 else 1.0)))

        # Mark labels/metadata
        template['anomaly_label'] = 1
        template['failure_type'] = 6  # 6 = catalytic (same encoding you used earlier)
        template['severity_score'] = float(sev)
        # Use enhanced fuzzy severity stage mapping for better boundary handling
        template['severity_stage'] = int(map_severity_to_stage_enhanced(sev, use_fuzzy=use_fuzzy_stages, fuzziness=stage_fuzziness, rng=rng))

        template['is_gray'] = False  # Regular synthetic anomalies are not gray

        meta = {
            "id": injection_id,
            "template_index": int(template_idx),
            "severity": float(sev),
            "stage": template['severity_stage'],
            "sensors_modified": sensors_touched,
            "out_of_bounds_sensors": int(out_of_bounds),
        }
        return template, meta

    # Optionally create progressive trajectories (several windows per anomaly)
    if create_trajectories:
        # choose fewer templates so total synthetic rows ~ n_anomalies
        # for simplicity: pick n_anomalies // trajectory_len templates and make sequences
        n_templates_traj = max(1, n_anomalies // trajectory_len)
        template_idxs = rng.choice(templates, size=n_templates_traj, replace=False)
        seq_id = 0
        for t_idx in template_idxs:
            # start and end severity randomize around base
            start = np.clip(np.random.normal(loc=max(0.05, base_severity - 0.2), scale=0.05), 0.05, 0.7)
            end = np.clip(np.random.normal(loc=min(0.95, base_severity + 0.3), scale=0.05), start, 1.0)
            # create trajectory of length trajectory_len with quadratic progression
            for step in range(trajectory_len):
                frac = (step / max(1, trajectory_len - 1))
                sev = start + (end - start) * (frac ** 2.2)  # accelerate toward end
                row, meta = create_synthetic_from_template(t_idx, sev, injection_id=f"cat_traj_{seq_id}_{step}")
                synthetic_rows.append(row)
                injections_meta.append(meta)
            seq_id += 1
    else:
        # simple one-row per template
        for i, t_idx in enumerate(templates):
            # pick severity sampled around base_severity with extra randomness so we cover zones
            sev = float(np.clip(rng.normal(loc=base_severity, scale=0.22), 0.05, 1.0))
            row, meta = create_synthetic_from_template(t_idx, sev, injection_id=f"cat_{i}")
            synthetic_rows.append(row)
            injections_meta.append(meta)

    # Append synthetic rows
    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows).reset_index(drop=True)
        # carry 'source' information if exists (mark as 'synthetic' for traceability)
        if 'source' in df_out.columns:
            synthetic_df['source'] = "synthetic_cat"
        df_combined = pd.concat([df_out, synthetic_df], ignore_index=True, sort=False)
    else:
        df_combined = df_out

    # Create gray ambiguous anomalies (unlabeled or possibly mislabeled)
    n_gray = int(max(1, gray_frac * len(df_out)))
    gray_idxs = rng.choice(list(df_out.index), size=n_gray, replace=False)
    gray_rows = []
    gray_meta = []
    for gi in gray_idxs:
        base_row = df_out.loc[gi].copy()
        # pick some random small drifts on a subset of columns
        drift_cols = [c for c in key_sensors if c in base_row.index]
        if len(drift_cols) == 0:
            continue
        ncols = max(1, int(0.3 * len(drift_cols)))
        chosen = list(rng.choice(drift_cols, size=ncols, replace=False))
        for c in chosen:
            st = sensor_stats.get(c)
            if st:
                # small mild drift (inside range) or sometimes slightly outside
                drift_pct = float(np.clip(rng.normal(loc=0.12, scale=0.06), 0.02, 0.35))
                # direction random
                if rng.rand() > 0.5:
                    candidate = base_row[c] + (st['max'] - st['mean']) * drift_pct
                else:
                    candidate = base_row[c] - (st['mean'] - st['min']) * drift_pct
                # random chance to slightly exceed bounds (10% chance)
                if rng.rand() < 0.10:
                    candidate = candidate + np.sign(candidate - st['mean']) * 0.05 * (st['max'] - st['min'])
                # apply
                base_row[c] = float(candidate)
        # gray labeling strategy: either leave as 0 (unlabeled) or sometimes mark as anomaly 1 (simulate label noise)
        label_noise = rng.rand() < 0.35
        base_row['anomaly_label'] = int(label_noise and 1 or 0)
        base_row['failure_type'] = 0
        base_row['severity_score'] = 0.0  # Gray anomalies have no explicit severity
        base_row['severity_stage'] = 0    # Gray anomalies have no explicit stage

        base_row['is_gray'] = True
        gray_rows.append(base_row)
        gray_meta.append({"template_index": int(gi), "chosen_cols": chosen, "label_noise": bool(label_noise)})

    if gray_rows:
        gray_df = pd.DataFrame(gray_rows).reset_index(drop=True)
        if 'source' in df_out.columns:
            gray_df['source'] = "synthetic_gray"
        df_combined = pd.concat([df_combined, gray_df], ignore_index=True, sort=False)

    # Apply hybrid TTF calculation (years + km) to ALL rows
    print("🔧 Applying hybrid TTF calculation (years + km)...")
    df_combined = compute_hybrid_ttf_physics(df_combined, base_years=4.0, avg_km_per_year=15000)

    # CRITICAL FIX: Ensure ALL synthetic_gray rows are properly marked as gray
    # This fixes the issue where some synthetic_gray rows had is_gray=False
    if 'source' in df_combined.columns:
        synthetic_gray_mask = df_combined['source'] == 'synthetic_gray'
        df_combined.loc[synthetic_gray_mask, 'is_gray'] = True
        print(f"🔧 Fixed: Marked {synthetic_gray_mask.sum()} synthetic_gray rows as is_gray=True")

    # Set TTF columns to NaN for gray rows (ambiguous cases shouldn't have TTF predictions)
    gray_mask = df_combined['is_gray'] == True
    df_combined.loc[gray_mask, 'TTF_years'] = np.nan
    df_combined.loc[gray_mask, 'TTF_km'] = np.nan
    print(f"🔍 Set TTF to NaN for {sum(gray_mask)} gray rows")

    # Final cleanup: Ensure normal rows (anomaly_label=0) have proper values
    # BUT exclude gray rows from this cleanup (gray rows can have anomaly_label=0)
    normal_mask = (df_combined['anomaly_label'] == 0) & (df_combined['is_gray'] == False)
    df_combined.loc[normal_mask, 'failure_type'] = 0  # Normal rows have no failure type
    df_combined.loc[normal_mask, 'severity_score'] = 0.0  # Normal rows have no severity
    df_combined.loc[normal_mask, 'severity_stage'] = 0  # Normal rows have no stage

    # report summary with detailed verification
    print(f"Appended synthetic anomalies: {len(synthetic_rows)}")
    print(f"Appended gray ambiguous rows: {len(gray_rows)}")
    print(f"Final dataset shape: {df_combined.shape}")
    print(f"Anomaly counts (label=1): {df_combined['anomaly_label'].sum()} / {len(df_combined)}")

    # Verification: Check gray row consistency
    if 'source' in df_combined.columns:
        synthetic_gray_count = (df_combined['source'] == 'synthetic_gray').sum()
        is_gray_count = (df_combined['is_gray'] == True).sum()
        gray_with_nan_ttf = df_combined.loc[df_combined['is_gray'] == True, 'TTF_years'].isna().sum()

        print(f"\n🔍 GRAY ROW VERIFICATION:")
        print(f"  synthetic_gray source rows: {synthetic_gray_count}")
        print(f"  is_gray=True rows: {is_gray_count}")
        print(f"  Gray rows with NaN TTF: {gray_with_nan_ttf}")

        if synthetic_gray_count == is_gray_count == gray_with_nan_ttf:
            print(f"  ✅ All gray rows properly configured!")
        else:
            print(f"  ⚠️ Gray row inconsistency detected!")

    # Verification: Check physics-based severity distribution
    severity_dist = df_combined['severity_stage'].value_counts().sort_index()
    print(f"\n📊 SEVERITY STAGE DISTRIBUTION:")
    for stage, count in severity_dist.items():
        print(f"  Stage {stage}: {count} rows")

    # Verification: Check anomaly label distribution by source
    if 'source' in df_combined.columns:
        print(f"\n🏷️ ANOMALY LABEL BY SOURCE:")
        source_anomaly = df_combined.groupby('source')['anomaly_label'].agg(['count', 'sum']).round(2)
        for source, stats in source_anomaly.iterrows():
            anomaly_rate = (stats['sum'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"  {source}: {stats['sum']:.0f}/{stats['count']:.0f} anomalies ({anomaly_rate:.1f}%)")

    # return DF and meta
    meta_all = {
        "created_synthetic_count": len(synthetic_rows),
        "created_gray_count": len(gray_rows),
        "injections": injections_meta,
        "gray_meta": gray_meta,
        "key_sensors": key_sensors
    }

    return df_combined, meta_all

# ---------------------------
# Helper functions for CLI and file operations
# ---------------------------
def run_catalytic_injection_from_file(
    input_path: str,
    output_path: str = "Final.csv",
    meta_path: str = "catalytic_injection_meta.json",
    **kwargs
):
    """Run catalytic anomaly injection from file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    df_out, meta = inject_catalytic_anomalies_realistic(df, **kwargs)
    df_out.to_csv(output_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved injected dataset -> {output_path}")
    print(f"Saved metadata -> {meta_path}")
    return df_out, meta


# -----------------------------
# Wrapper function
# -----------------------------
def run_injection(df, severity=0.6, output_path=None, meta_path=None, random_state=42):
    """
    Run catalytic converter anomaly injection.

    Works in both Jupyter and CLI modes.

    Returns
    -------
    df_anom : pd.DataFrame
    injections : list of dict
    """
    df_anom, meta_dict = inject_catalytic_anomalies_realistic(
        df, base_severity=severity, random_state=random_state
    )

    # Extract injections from metadata for backward compatibility
    injections = meta_dict.get("injections", [])

    # Save if paths provided
    if output_path:
        df_anom.to_csv(output_path, index=False)
        print(f"💾 Saved injected dataset -> {output_path}")

    if meta_path:
        with open(meta_path, "w") as f:
            json.dump(injections, f, indent=2)
        print(f"💾 Saved metadata -> {meta_path}")

    return df_anom, injections


# -----------------------------
# CLI entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Catalytic Converter Anomaly Injector")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (windowed + scaled features)")
    parser.add_argument("--output", type=str, default="catalytic_out.csv", help="Path to save modified CSV")
    parser.add_argument("--meta", type=str, default="catalytic_meta.json", help="Path to save metadata JSON")
    parser.add_argument("--severity", type=float, default=0.6, help="Severity of anomalies (0.0-1.0)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    run_injection(
        df=df,
        severity=args.severity,
        output_path=args.output,
        meta_path=args.meta,
        random_state=args.random_state
    )


def run_complete_pipeline():
    """
    Run the complete data preprocessing pipeline with enhanced anomaly injection.
    This function can be called from Jupyter notebooks or as a standalone script.
    """
    print("🚀 Starting Complete Data Preprocessing Pipeline")
    print("=" * 60)

    # Check if we have existing scaled features or need to create them
    scaled_file = "scaled_features_anomalies_df.csv"

    if os.path.exists(scaled_file):
        print(f"� Loading existing scaled features: {scaled_file}")
        scaled_features_df = pd.read_csv(scaled_file)
        print(f"✅ Loaded scaled features with shape: {scaled_features_df.shape}")
    else:
        print("❌ No existing scaled features found.")
        print("Please run the windowing and scaling sections first, or provide the scaled features file.")
        return None, None

    # Clean any existing anomaly labels to start fresh
    columns_to_clean = ['anomaly_label', 'failure_type', 'severity_score', 'severity_stage', 'time_to_failure_years']
    for col in columns_to_clean:
        if col in scaled_features_df.columns:
            scaled_features_df = scaled_features_df.drop(columns=[col])

    print(f"🧹 Cleaned dataset shape: {scaled_features_df.shape}")

    # Apply enhanced anomaly injection
    print("\n🔧 Applying Enhanced Catalytic Converter Anomaly Injection...")
    print("Features:")
    print("  ✨ Three-zone physics model (Normal/Mild-Moderate/Severe-Failure)")
    print("  🔗 Physics-based sensor correlations")
    print("  📈 Progressive degradation patterns")
    print("  🎯 Gray zone ambiguous anomalies")
    print("  🚨 Out-of-bounds behavior for severe cases")

    enhanced_df, metadata = inject_catalytic_anomalies_realistic(
        scaled_df=scaled_features_df,
        base_severity=0.45,  # Moderate base severity for realistic anomalies
        n_frac=0.05,         # 5% synthetic anomalies
        gray_frac=0.08,      # 8% gray ambiguous cases
        create_trajectories=False,  # Simple one-shot anomalies
        random_state=42
    )

    # Save metadata
    metadata_file = "catalytic_anomaly_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Display results
    print(f"\n✅ Enhanced Anomaly Injection Summary:")
    print(f"💾 Saved injection metadata to: {metadata_file}")
    print(f"📊 Created {metadata['created_synthetic_count']} synthetic anomalies")
    print(f"🔍 Created {metadata['created_gray_count']} gray ambiguous cases")
    print(f"Original dataset: {len(scaled_features_df)} rows")
    print(f"Enhanced dataset: {len(enhanced_df)} rows")

    # Analyze results
    anomaly_counts = enhanced_df['anomaly_label'].value_counts()
    print(f"\n🏷️ Label Distribution:")
    print(f"Normal (0): {anomaly_counts.get(0, 0)}")
    print(f"Anomaly (1): {anomaly_counts.get(1, 0)}")

    # Show sample anomalies
    anomaly_samples = enhanced_df[enhanced_df['anomaly_label'] == 1]
    if len(anomaly_samples) > 0:
        key_sensors = [
            "CATALYST_TEMPERATURE_BANK1_SENSOR1 ()_mean",
            "CATALYST_TEMPERATURE_BANK1_SENSOR2 ()_mean",
            "ENGINE_LOAD ()_mean",
            "THROTTLE ()_mean"
        ]
        available_sensors = [s for s in key_sensors if s in anomaly_samples.columns]
        if available_sensors:
            print(f"\n🔍 Sample Anomaly Sensor Values:")
            print(anomaly_samples[available_sensors].head(3).round(4))

    # Remove duplicates
    print(f"\n🔧 Checking for duplicates...")
    duplicates_count = enhanced_df.duplicated().sum()
    if duplicates_count > 0:
        enhanced_df = enhanced_df.drop_duplicates(keep='first')
        print(f"✅ Removed {duplicates_count} duplicates")
    else:
        print(f"✅ No duplicates found")

    # Save final dataset
    output_file = "Final.csv"
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved final dataset: {output_file}")

    print(f"\n🎯 Pipeline completed successfully!")
    print(f"📈 Dataset ready for model training with Random Forest")

    return enhanced_df, metadata

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Run the complete pipeline
    try:
        final_df, final_metadata = run_complete_pipeline()
        if final_df is not None:
            print(f"\n� Success! Final dataset shape: {final_df.shape}")
            print("Ready for model training!")
        else:
            print("\n❌ Pipeline failed. Please check the requirements.")
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()





