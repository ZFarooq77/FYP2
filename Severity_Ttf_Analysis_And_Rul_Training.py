"""
Severity ↔ TTF Statistical Analysis + Feature Importance + RUL Model Training
File: Severity_Ttf_Analysis_And_Rul_Training.py

Purpose:
1) Statistically analyze relationship between `severity_stage` (0-4) and `TTF_years`, `TTF_km`.
2) Validate which sensor features drive `severity_stage` (feature importance).
3) Check gray-row robustness in RUL space and model performance.
4) Train baseline RUL regression models for `TTF_years` and `TTF_km` (with evaluation).

Run: python Severity_Ttf_Analysis_And_Rul_Training.py

Notes:
- The script assumes Final.csv is available in the working directory.
- It writes plots to ./outputs/ and models to ./models/.
- Uses curated feature list from models_v2_enhanced/selected_features_v2.csv if available.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.inspection import permutation_importance

print("🔬 SEVERITY ↔ TTF STATISTICAL ANALYSIS + FEATURE IMPORTANCE + RUL TRAINING")
print("=" * 80)

# ---------------------------
# Config & paths
# ---------------------------
DATA_CSV = "Final.csv"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# Utility plotting helpers
# ---------------------------
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved plot: {name}")

# ---------------------------
# 0) Load data
# ---------------------------
print('\n📥 LOADING DATASET...')
try:
    df = pd.read_csv(DATA_CSV)
    print(f'✅ Loaded {len(df):,} rows, {df.shape[1]:,} columns')
except FileNotFoundError:
    print(f"❌ Error: {DATA_CSV} not found. Please ensure Final.csv is in the working directory.")
    exit(1)

# ---------------------------
# Feature selection (use curated list if available)
# ---------------------------
print('\n🔧 FEATURE SELECTION...')
try:
    features_df = pd.read_csv('models_v2_enhanced/selected_features_v2.csv')
    features_list = features_df['feature'].tolist()
    print(f'✅ Using curated feature list ({len(features_list)} features)')
except Exception:
    # Fallback: choose numeric columns except target cols
    exclude = {'severity_stage', 'severity_score', 'anomaly_label', 'TTF_years', 'TTF_km', 
               'is_gray', 'failure_type', 'source', 'time_to_failure_years'}
    features_list = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    print(f'⚠️ Using fallback numeric features ({len(features_list)} features)')

# Clean dataset (drop NA in critical targets)
print('\n🧹 CLEANING DATASET...')
initial_rows = len(df)
df = df.dropna(subset=['severity_stage', 'TTF_years', 'TTF_km'])
final_rows = len(df)
print(f"✅ Cleaned dataset: {initial_rows:,} → {final_rows:,} rows ({initial_rows-final_rows:,} removed)")

# Check data availability
print(f"\n📊 DATA OVERVIEW:")
print(f"  Severity stages: {sorted(df['severity_stage'].unique())}")
print(f"  Gray rows: {(df['is_gray'] == True).sum():,} ({(df['is_gray'] == True).sum()/len(df)*100:.1f}%)")
print(f"  TTF_years range: {df['TTF_years'].min():.2f} - {df['TTF_years'].max():.2f}")
print(f"  TTF_km range: {df['TTF_km'].min():,.0f} - {df['TTF_km'].max():,.0f}")

# ---------------------------
# 1) STATISTICAL + CORRELATION ANALYSIS
# ---------------------------
print('\n' + '='*60)
print('📊 STEP 1: STATISTICAL & CORRELATION ANALYSIS')
print('='*60)

# Basic group stats
print('\n📈 Descriptive statistics by severity_stage:')
grouped = df.groupby('severity_stage')[['TTF_years','TTF_km','severity_score']].agg(['count','mean','median','std','min','max'])
print(grouped.round(3))

# Save grouped stats
grouped.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_stats_by_severity.csv'))
print("✅ Saved: descriptive_stats_by_severity.csv")

# Boxplots: TTF_years by severity_stage
print('\n📊 Creating statistical plots...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(x='severity_stage', y='TTF_years', data=df, ax=ax1)
ax1.set_title('TTF_years distribution by severity_stage')
ax1.set_xlabel('Severity Stage')
ax1.set_ylabel('TTF Years')

sns.boxplot(x='severity_stage', y='TTF_km', data=df, ax=ax2)
ax2.set_title('TTF_km distribution by severity_stage')
ax2.set_xlabel('Severity Stage')
ax2.set_ylabel('TTF Kilometers')

save_fig(fig, 'boxplots_TTF_by_severity.png')

# Violin plots (show distribution shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.violinplot(x='severity_stage', y='TTF_years', data=df, inner='quartile', ax=ax1)
ax1.set_title('TTF_years violin by severity_stage')
ax1.set_xlabel('Severity Stage')
ax1.set_ylabel('TTF Years')

sns.violinplot(x='severity_stage', y='TTF_km', data=df, inner='quartile', ax=ax2)
ax2.set_title('TTF_km violin by severity_stage')
ax2.set_xlabel('Severity Stage')
ax2.set_ylabel('TTF Kilometers')

save_fig(fig, 'violin_TTF_by_severity.png')

# Correlation analysis: Spearman (monotonic) and Pearson
print('\n🔗 CORRELATION ANALYSIS:')
spearman_year = stats.spearmanr(df['severity_stage'], df['TTF_years'], nan_policy='omit')
spearman_km = stats.spearmanr(df['severity_stage'], df['TTF_km'], nan_policy='omit')
pearson_year = stats.pearsonr(df['severity_stage'], df['TTF_years'])
pearson_km = stats.pearsonr(df['severity_stage'], df['TTF_km'])

print(f"  Spearman severity vs TTF_years: ρ={spearman_year.correlation:.4f}, p={spearman_year.pvalue:.4e}")
print(f"  Spearman severity vs TTF_km: ρ={spearman_km.correlation:.4f}, p={spearman_km.pvalue:.4e}")
print(f"  Pearson severity vs TTF_years: r={pearson_year[0]:.4f}, p={pearson_year[1]:.4e}")
print(f"  Pearson severity vs TTF_km: r={pearson_km[0]:.4f}, p={pearson_km[1]:.4e}")

# Save correlation results
corr_results = pd.DataFrame({
    'Metric': ['Spearman_TTF_years', 'Spearman_TTF_km', 'Pearson_TTF_years', 'Pearson_TTF_km'],
    'Correlation': [spearman_year.correlation, spearman_km.correlation, pearson_year[0], pearson_km[0]],
    'P_value': [spearman_year.pvalue, spearman_km.pvalue, pearson_year[1], pearson_km[1]]
})
corr_results.to_csv(os.path.join(OUTPUT_DIR, 'correlation_analysis.csv'), index=False)
print("✅ Saved: correlation_analysis.csv")

# Statistical significance across stages: Kruskal-Wallis test
print('\n🧪 STATISTICAL SIGNIFICANCE TESTING:')

# Check normality of TTF_years per stage using Shapiro-Wilk test
normality_results = []
for stage in sorted(df['severity_stage'].unique()):
    stage_data = df[df['severity_stage'] == stage]['TTF_years']
    if len(stage_data) > 3:  # Shapiro needs at least 3 samples
        sample_size = min(500, len(stage_data))  # Shapiro works best with <5000 samples
        sample_data = stage_data.sample(n=sample_size, random_state=42) if len(stage_data) > sample_size else stage_data
        try:
            stat, p = stats.shapiro(sample_data)
            normality_results.append({'stage': stage, 'shapiro_stat': stat, 'shapiro_p': p, 'n_samples': len(stage_data)})
        except Exception as e:
            normality_results.append({'stage': stage, 'shapiro_stat': np.nan, 'shapiro_p': np.nan, 'n_samples': len(stage_data)})

normality_df = pd.DataFrame(normality_results)
print("  Shapiro-Wilk normality test results (p<0.05 indicates non-normal):")
for _, row in normality_df.iterrows():
    print(f"    Stage {row['stage']}: p={row['shapiro_p']:.4e} (n={row['n_samples']})")

# Use Kruskal-Wallis (non-parametric) across stages
groups_years = [group['TTF_years'].values for _, group in df.groupby('severity_stage')]
groups_km = [group['TTF_km'].values for _, group in df.groupby('severity_stage')]

kw_stat_years, kw_p_years = stats.kruskal(*groups_years)
kw_stat_km, kw_p_km = stats.kruskal(*groups_km)

print(f"\n  Kruskal-Wallis test across severity stages:")
print(f"    TTF_years: H={kw_stat_years:.3f}, p={kw_p_years:.4e}")
print(f"    TTF_km: H={kw_stat_km:.3f}, p={kw_p_km:.4e}")

# Post-hoc pairwise tests (Mann-Whitney with Bonferroni correction)
print('\n🔬 POST-HOC PAIRWISE TESTS (Mann-Whitney U):')
from itertools import combinations

stages = sorted(df['severity_stage'].unique())
pairs = list(combinations(stages, 2))
posthoc_results = []

for stage_a, stage_b in pairs:
    data_a = df[df['severity_stage'] == stage_a]['TTF_years']
    data_b = df[df['severity_stage'] == stage_b]['TTF_years']
    
    if len(data_a) > 0 and len(data_b) > 0:
        stat, p = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        # Bonferroni correction
        p_corrected = min(p * len(pairs), 1.0)
        posthoc_results.append({
            'stage_a': stage_a, 
            'stage_b': stage_b, 
            'statistic': stat, 
            'p_value': p, 
            'p_corrected': p_corrected,
            'significant': p_corrected < 0.05
        })
        print(f"    Stage {stage_a} vs {stage_b}: p={p:.4e}, p_corrected={p_corrected:.4e}, significant={p_corrected < 0.05}")

# Save posthoc results
posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df.to_csv(os.path.join(OUTPUT_DIR, 'posthoc_mannwhitney_TTF_years.csv'), index=False)
print("✅ Saved: posthoc_mannwhitney_TTF_years.csv")

# ---------------------------
# 2) FEATURE IMPORTANCE VALIDATION (for severity_stage)
# ---------------------------
print('\n' + '='*60)
print('🎯 STEP 2: FEATURE IMPORTANCE VALIDATION')
print('='*60)

print('\n🔧 Preparing features for importance analysis...')
X = df[features_list].copy()
y = df['severity_stage'].astype(int)

# Handle any remaining NaN values in features
X = X.fillna(X.median())
print(f"  Feature matrix: {X.shape}")
print(f"  Target classes: {sorted(y.unique())}")

# Train/val split stratified
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train set: {X_tr.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")

# Fit RandomForestClassifier
print('\n🌲 Training Random Forest for feature importance...')
rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    max_depth=10,
    min_samples_split=5
)

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', rf_clf)
])

pipe_rf.fit(X_tr, y_tr)

# Validation accuracy
val_accuracy = pipe_rf.score(X_val, y_val)
print(f"✅ Random Forest validation accuracy: {val_accuracy:.4f}")

# Built-in feature importance
print('\n📊 Computing built-in feature importance...')
builtin_importance = pipe_rf.named_steps['rf'].feature_importances_
builtin_df = pd.DataFrame({
    'feature': features_list,
    'builtin_importance': builtin_importance
}).sort_values('builtin_importance', ascending=False)

# Permutation importance (on validation set)
print('🔄 Computing permutation importance (this may take a while)...')
perm = permutation_importance(
    pipe_rf, X_val, y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

perm_df = pd.DataFrame({
    'feature': features_list,
    'perm_importance_mean': perm.importances_mean,
    'perm_importance_std': perm.importances_std
}).sort_values('perm_importance_mean', ascending=False)

# Combine importance measures
importance_combined = pd.merge(builtin_df, perm_df, on='feature')
importance_combined.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_severity.csv'), index=False)
print("✅ Saved: feature_importance_severity.csv")

# Plot top 20 features
print('\n📈 Creating feature importance plots...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Built-in importance
top20_builtin = builtin_df.head(20)
sns.barplot(data=top20_builtin, x='builtin_importance', y='feature', ax=ax1)
ax1.set_title('Top 20 Built-in Feature Importances\n(Random Forest)')
ax1.set_xlabel('Importance Score')

# Permutation importance
top20_perm = perm_df.head(20)
sns.barplot(data=top20_perm, x='perm_importance_mean', y='feature', ax=ax2)
ax2.set_title('Top 20 Permutation Importances\n(Validation Set)')
ax2.set_xlabel('Permutation Importance')

save_fig(fig, 'feature_importance_top20_severity.png')

# Print top 10 features
print('\n🏆 TOP 10 MOST IMPORTANT FEATURES:')
print('  Built-in Importance:')
for i, (_, row) in enumerate(builtin_df.head(10).iterrows(), 1):
    print(f"    {i:2d}. {row['feature']:<30} {row['builtin_importance']:.6f}")

print('\n  Permutation Importance:')
for i, (_, row) in enumerate(perm_df.head(10).iterrows(), 1):
    print(f"    {i:2d}. {row['feature']:<30} {row['perm_importance_mean']:.6f} ± {row['perm_importance_std']:.6f}")

# ---------------------------
# 3) GRAY ROW ROBUSTNESS CHECK
# ---------------------------
print('\n' + '='*60)
print('🔍 STEP 3: GRAY ROW ROBUSTNESS CHECKS')
print('='*60)

gray_mask = df['is_gray'] == True
n_gray = gray_mask.sum()
n_total = len(df)
print(f'\n📊 Gray row analysis:')
print(f"  Gray rows: {n_gray:,} / {n_total:,} ({n_gray/n_total*100:.1f}%)")
print(f"  Non-gray rows: {n_total-n_gray:,} ({(n_total-n_gray)/n_total*100:.1f}%)")

if n_gray > 0:
    # Compare TTF distributions: gray vs non-gray
    print('\n📊 Creating gray vs non-gray comparison plots...')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # TTF_years comparison
    df_plot = df.copy()
    df_plot['Gray_Status'] = df_plot['is_gray'].map({True: 'Gray', False: 'Non-Gray'})

    sns.boxplot(data=df_plot, x='Gray_Status', y='TTF_years', ax=ax1)
    ax1.set_title('TTF_years: Gray vs Non-Gray Distribution')
    ax1.set_ylabel('TTF Years')

    sns.boxplot(data=df_plot, x='Gray_Status', y='TTF_km', ax=ax2)
    ax2.set_title('TTF_km: Gray vs Non-Gray Distribution')
    ax2.set_ylabel('TTF Kilometers')

    # Severity stage distribution
    gray_severity = df[gray_mask]['severity_stage'].value_counts().sort_index()
    non_gray_severity = df[~gray_mask]['severity_stage'].value_counts().sort_index()

    stages = sorted(df['severity_stage'].unique())
    gray_counts = [gray_severity.get(s, 0) for s in stages]
    non_gray_counts = [non_gray_severity.get(s, 0) for s in stages]

    x = np.arange(len(stages))
    width = 0.35

    ax3.bar(x - width/2, gray_counts, width, label='Gray', alpha=0.8)
    ax3.bar(x + width/2, non_gray_counts, width, label='Non-Gray', alpha=0.8)
    ax3.set_xlabel('Severity Stage')
    ax3.set_ylabel('Count')
    ax3.set_title('Severity Stage Distribution: Gray vs Non-Gray')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stages)
    ax3.legend()

    # Severity score comparison
    sns.boxplot(data=df_plot, x='Gray_Status', y='severity_score', ax=ax4)
    ax4.set_title('Severity Score: Gray vs Non-Gray Distribution')
    ax4.set_ylabel('Severity Score')

    save_fig(fig, 'gray_vs_non_gray_analysis.png')

    # Statistical tests for distribution differences
    print('\n🧪 STATISTICAL TESTS (Gray vs Non-Gray):')

    non_gray_ttf_years = df[~gray_mask]['TTF_years'].dropna()
    gray_ttf_years = df[gray_mask]['TTF_years'].dropna()

    non_gray_ttf_km = df[~gray_mask]['TTF_km'].dropna()
    gray_ttf_km = df[gray_mask]['TTF_km'].dropna()

    # Kolmogorov-Smirnov tests
    if len(gray_ttf_years) > 0 and len(non_gray_ttf_years) > 0:
        ks_stat_years, ks_p_years = stats.ks_2samp(non_gray_ttf_years, gray_ttf_years)
        print(f"  KS test (TTF_years): statistic={ks_stat_years:.4f}, p={ks_p_years:.4e}")

        ks_stat_km, ks_p_km = stats.ks_2samp(non_gray_ttf_km, gray_ttf_km)
        print(f"  KS test (TTF_km): statistic={ks_stat_km:.4f}, p={ks_p_km:.4e}")

        # Mann-Whitney U test
        mw_stat_years, mw_p_years = stats.mannwhitneyu(non_gray_ttf_years, gray_ttf_years, alternative='two-sided')
        mw_stat_km, mw_p_km = stats.mannwhitneyu(non_gray_ttf_km, gray_ttf_km, alternative='two-sided')

        print(f"  Mann-Whitney U (TTF_years): statistic={mw_stat_years:.0f}, p={mw_p_years:.4e}")
        print(f"  Mann-Whitney U (TTF_km): statistic={mw_stat_km:.0f}, p={mw_p_km:.4e}")

        # Save statistical test results
        stat_tests = pd.DataFrame({
            'Test': ['KS_TTF_years', 'KS_TTF_km', 'MannWhitney_TTF_years', 'MannWhitney_TTF_km'],
            'Statistic': [ks_stat_years, ks_stat_km, mw_stat_years, mw_stat_km],
            'P_value': [ks_p_years, ks_p_km, mw_p_years, mw_p_km],
            'Significant': [ks_p_years < 0.05, ks_p_km < 0.05, mw_p_years < 0.05, mw_p_km < 0.05]
        })
        stat_tests.to_csv(os.path.join(OUTPUT_DIR, 'gray_vs_non_gray_statistical_tests.csv'), index=False)
        print("✅ Saved: gray_vs_non_gray_statistical_tests.csv")

    # Model performance on gray rows specifically
    print('\n🎯 MODEL PERFORMANCE ON GRAY ROWS:')
    if len(df[gray_mask]) > 0:
        X_gray = df.loc[gray_mask, features_list].fillna(df[features_list].median())
        y_gray = df.loc[gray_mask, 'severity_stage'].astype(int)

        if len(X_gray) > 0:
            y_gray_pred = pipe_rf.predict(X_gray)
            acc_gray = accuracy_score(y_gray, y_gray_pred)
            print(f"  Random Forest accuracy on gray rows: {acc_gray:.4f} ({len(X_gray)} samples)")

            # Compare with overall validation accuracy
            print(f"  Overall validation accuracy: {val_accuracy:.4f}")
            print(f"  Gray row performance difference: {acc_gray - val_accuracy:+.4f}")

            # Save gray row predictions
            gray_predictions = pd.DataFrame({
                'true_severity': y_gray,
                'predicted_severity': y_gray_pred,
                'correct': y_gray == y_gray_pred
            })
            gray_predictions.to_csv(os.path.join(OUTPUT_DIR, 'gray_row_predictions.csv'), index=False)
            print("✅ Saved: gray_row_predictions.csv")
        else:
            print("  No gray rows available for model evaluation")
    else:
        print("  No gray rows found in dataset")
else:
    print("⚠️ No gray rows found in dataset - skipping gray row analysis")

# ---------------------------
# 4) TRAIN TTF REGRESSION MODELS (RUL BASELINE)
# ---------------------------
print('\n' + '='*60)
print('🚀 STEP 4: BASELINE RUL REGRESSION MODELS')
print('='*60)

print('\n🔧 Preparing data for RUL regression...')

# Filter out rows with NaN TTF values (gray rows)
rul_mask = df['TTF_years'].notna() & df['TTF_km'].notna()
df_rul = df[rul_mask].copy()
print(f"  RUL training data: {len(df_rul):,} samples (excluded {len(df) - len(df_rul):,} NaN TTF rows)")

# Prepare features and targets
X_rul = df_rul[features_list].fillna(df_rul[features_list].median())
y_years = df_rul['TTF_years']
y_km = df_rul['TTF_km']

print(f"  Feature matrix: {X_rul.shape}")
print(f"  TTF_years range: {y_years.min():.2f} - {y_years.max():.2f}")
print(f"  TTF_km range: {y_km.min():,.0f} - {y_km.max():,.0f}")

# Stratified split based on severity_stage to maintain stage distribution
X_tr_rul, X_test_rul, y_years_tr, y_years_test = train_test_split(
    X_rul, y_years, test_size=0.2, random_state=42,
    stratify=df_rul['severity_stage']
)

_, _, y_km_tr, y_km_test = train_test_split(
    X_rul, y_km, test_size=0.2, random_state=42,
    stratify=df_rul['severity_stage']
)

print(f"  Train set: {X_tr_rul.shape[0]} samples")
print(f"  Test set: {X_test_rul.shape[0]} samples")

# Define regression models
print('\n🤖 Defining regression models...')
models_rul = {
    'Ridge_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=1.0))
    ]),
    'Random_Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate models for TTF_years
print('\n📊 TRAINING RUL MODELS FOR TTF_YEARS:')
print('-' * 50)

rul_results = []

for name, model in models_rul.items():
    print(f'\n🔄 Training {name} for TTF_years...')

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr_rul, y_years_tr, cv=kf, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit on full training set
    model.fit(X_tr_rul, y_years_tr)

    # Predictions
    y_pred_train = model.predict(X_tr_rul)
    y_pred_test = model.predict(X_test_rul)

    # Training metrics
    train_mae = mean_absolute_error(y_years_tr, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_years_tr, y_pred_train))
    train_r2 = r2_score(y_years_tr, y_pred_train)

    # Test metrics
    test_mae = mean_absolute_error(y_years_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_years_test, y_pred_test))
    test_r2 = r2_score(y_years_test, y_pred_test)

    print(f'  📈 Cross-validation R²: {cv_mean:.4f} ± {cv_std:.4f}')
    print(f'  📊 Training   → MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}')
    print(f'  🎯 Test      → MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')

    # Store results
    rul_results.append({
        'Model': f'{name}_TTF_years',
        'Target': 'TTF_years',
        'CV_R2_mean': cv_mean,
        'CV_R2_std': cv_std,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Test_MAE': test_mae,
        'Test_RMSE': test_rmse,
        'Test_R2': test_r2,
        'Overfitting_Gap': train_r2 - test_r2
    })

    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': y_years_test,
        'y_pred': y_pred_test,
        'residual': y_years_test - y_pred_test,
        'abs_error': np.abs(y_years_test - y_pred_test)
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions_TTF_years.csv'), index=False)
    print(f'  ✅ Saved: {name}_predictions_TTF_years.csv')

    # Save trained model
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_TTF_years.pkl'))
    print(f'  💾 Saved model: {name}_TTF_years.pkl')

# Train and evaluate models for TTF_km
print('\n📊 TRAINING RUL MODELS FOR TTF_KM:')
print('-' * 50)

for name, model in models_rul.items():
    print(f'\n🔄 Training {name} for TTF_km...')

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr_rul, y_km_tr, cv=kf, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit on full training set
    model.fit(X_tr_rul, y_km_tr)

    # Predictions
    y_pred_train = model.predict(X_tr_rul)
    y_pred_test = model.predict(X_test_rul)

    # Training metrics
    train_mae = mean_absolute_error(y_km_tr, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_km_tr, y_pred_train))
    train_r2 = r2_score(y_km_tr, y_pred_train)

    # Test metrics
    test_mae = mean_absolute_error(y_km_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_km_test, y_pred_test))
    test_r2 = r2_score(y_km_test, y_pred_test)

    print(f'  📈 Cross-validation R²: {cv_mean:.4f} ± {cv_std:.4f}')
    print(f'  📊 Training   → MAE: {train_mae:.0f}, RMSE: {train_rmse:.0f}, R²: {train_r2:.4f}')
    print(f'  🎯 Test      → MAE: {test_mae:.0f}, RMSE: {test_rmse:.0f}, R²: {test_r2:.4f}')

    # Store results
    rul_results.append({
        'Model': f'{name}_TTF_km',
        'Target': 'TTF_km',
        'CV_R2_mean': cv_mean,
        'CV_R2_std': cv_std,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Test_MAE': test_mae,
        'Test_RMSE': test_rmse,
        'Test_R2': test_r2,
        'Overfitting_Gap': train_r2 - test_r2
    })

    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': y_km_test,
        'y_pred': y_pred_test,
        'residual': y_km_test - y_pred_test,
        'abs_error': np.abs(y_km_test - y_pred_test)
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f'{name}_predictions_TTF_km.csv'), index=False)
    print(f'  ✅ Saved: {name}_predictions_TTF_km.csv')

    # Save trained model
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_TTF_km.pkl'))
    print(f'  💾 Saved model: {name}_TTF_km.pkl')

# Create comprehensive results summary
rul_results_df = pd.DataFrame(rul_results)
rul_results_df.to_csv(os.path.join(OUTPUT_DIR, 'rul_model_results_summary.csv'), index=False)
print(f'\n✅ Saved comprehensive results: rul_model_results_summary.csv')

# Display results summary
print('\n📊 RUL MODEL PERFORMANCE SUMMARY:')
print('=' * 100)
print(f"{'Model':<25} {'Target':<12} {'Test R²':<10} {'Test MAE':<12} {'Test RMSE':<12} {'Overfit Gap':<12}")
print('=' * 100)

for _, row in rul_results_df.iterrows():
    mae_format = f"{row['Test_MAE']:.4f}" if 'years' in row['Target'] else f"{row['Test_MAE']:.0f}"
    rmse_format = f"{row['Test_RMSE']:.4f}" if 'years' in row['Target'] else f"{row['Test_RMSE']:.0f}"

    print(f"{row['Model']:<25} {row['Target']:<12} {row['Test_R2']:<10.4f} {mae_format:<12} {rmse_format:<12} {row['Overfitting_Gap']:<12.4f}")

print('=' * 100)

# ---------------------------
# Final Summary
# ---------------------------
print('\n' + '='*80)
print('🏁 ANALYSIS COMPLETE - SUMMARY OF OUTPUTS')
print('='*80)

print(f'\n📁 Output Directory: {OUTPUT_DIR}/')
print('  📊 Statistical Analysis:')
print('    - descriptive_stats_by_severity.csv')
print('    - correlation_analysis.csv')
print('    - posthoc_mannwhitney_TTF_years.csv')
print('    - boxplots_TTF_by_severity.png')
print('    - violin_TTF_by_severity.png')

print('\n  🎯 Feature Importance:')
print('    - feature_importance_severity.csv')
print('    - feature_importance_top20_severity.png')

print('\n  🔍 Gray Row Analysis:')
print('    - gray_vs_non_gray_statistical_tests.csv')
print('    - gray_row_predictions.csv')
print('    - gray_vs_non_gray_analysis.png')

print('\n  🚀 RUL Models:')
print('    - rul_model_results_summary.csv')
print('    - Ridge_Regression_predictions_TTF_years.csv')
print('    - Ridge_Regression_predictions_TTF_km.csv')
print('    - Random_Forest_predictions_TTF_years.csv')
print('    - Random_Forest_predictions_TTF_km.csv')

print(f'\n💾 Models Directory: {MODELS_DIR}/')
print('    - Ridge_Regression_TTF_years.pkl')
print('    - Ridge_Regression_TTF_km.pkl')
print('    - Random_Forest_TTF_years.pkl')
print('    - Random_Forest_TTF_km.pkl')

print('\n🎯 KEY FINDINGS:')
print('  1. Statistical relationships between severity stages and TTF validated')
print('  2. Feature importance rankings established for severity prediction')
print('  3. Gray row robustness and distribution differences analyzed')
print('  4. Baseline RUL regression models trained and evaluated')

print('\n🚀 Ready for advanced RUL model development and comparison!')
print('='*80)
