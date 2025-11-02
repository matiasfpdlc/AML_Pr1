"""
Fine-tune SVR and ElasticNet with ReliefF (k=250).
Using Optuna for hyperparameter optimization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import IsolationForest
import optuna
from optuna.samplers import TPESampler
import json
from skrebate import ReliefF
import warnings
warnings.filterwarnings('ignore')

# Add utilities to path and import
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from dataloader import DataLoader  # type: ignore

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Get project root directory
project_root = Path(__file__).parent.parent

print("="*80)
print("FINE-TUNING SVR & ELASTICNET WITH ReliefF (k=250)")
print("="*80)

# Load data
print("\n📂 Loading and preprocessing data...")
data_path = project_root / 'eth-aml-2025-project-1'
loader = DataLoader(str(data_path))
X_train, y_train = loader.load_train_data()

# Impute
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)

# Remove outliers
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
outlier_labels = iso_forest.fit_predict(X_imputed)
inlier_mask = outlier_labels == 1
X_clean = X_imputed[inlier_mask]
y_clean = y_train.values[inlier_mask] if hasattr(y_train, 'values') else y_train[inlier_mask]

print(f"✓ Data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Apply ReliefF (k=250)
print("\n🔬 Applying ReliefF feature selection (k=250)...")
relief = ReliefF(n_features_to_select=250, n_neighbors=10, n_jobs=-1)
X_selected = relief.fit_transform(X_scaled, y_clean)
X_selected_df = pd.DataFrame(X_selected)
print(f"✓ Selected: {X_selected.shape[1]} features")

# Prepare CV
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ============================================================================
# 1. TUNE SVR
# ============================================================================
print("\n" + "="*80)
print("TUNING SVR (Support Vector Regression)")
print("="*80)

def objective_svr(trial):
    """Optuna objective for SVR with overfitting penalty."""
    
    # Suggest hyperparameters
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    C = trial.suggest_float('C', 0.1, 100.0, log=True)
    epsilon = trial.suggest_float('epsilon', 0.01, 1.0, log=True)
    
    if kernel == 'rbf':
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, cache_size=1000)
    elif kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 4)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma, cache_size=1000)
    else:  # sigmoid
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, cache_size=1000)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
    cv_r2 = np.mean(cv_scores)
    
    # Train score for overfitting check
    model.fit(X_selected_df, y_clean)
    train_pred = model.predict(X_selected_df)
    train_r2 = r2_score(y_clean, train_pred)
    
    # Overfitting penalty
    overfit_gap = train_r2 - cv_r2
    penalty = 0.5 * max(0, overfit_gap - 0.15)
    
    objective_value = cv_r2 - penalty
    
    # Log metrics
    trial.set_user_attr('cv_r2', cv_r2)
    trial.set_user_attr('train_r2', train_r2)
    trial.set_user_attr('overfit_gap', overfit_gap)
    
    return objective_value

print("\n🔍 Running Optuna optimization (50 trials)...")
study_svr = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_SEED)
)
study_svr.optimize(objective_svr, n_trials=50, show_progress_bar=True)

print("\n✅ SVR Optimization Complete!")
print(f"   Best CV R²: {study_svr.best_trial.user_attrs['cv_r2']:.4f}")
print(f"   Best Train R²: {study_svr.best_trial.user_attrs['train_r2']:.4f}")
print(f"   Overfit Gap: {study_svr.best_trial.user_attrs['overfit_gap']:.4f}")
print("\n   Best Parameters:")
for key, value in study_svr.best_params.items():
    print(f"     {key}: {value}")

# ============================================================================
# 2. TUNE ELASTICNET
# ============================================================================
print("\n" + "="*80)
print("TUNING ELASTICNET")
print("="*80)

def objective_elasticnet(trial):
    """Optuna objective for ElasticNet with overfitting penalty."""
    
    # Suggest hyperparameters
    alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    max_iter = trial.suggest_categorical('max_iter', [5000, 10000, 20000])
    
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=RANDOM_SEED,
        selection='cyclic'
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
    cv_r2 = np.mean(cv_scores)
    
    # Train score for overfitting check
    model.fit(X_selected_df, y_clean)
    train_pred = model.predict(X_selected_df)
    train_r2 = r2_score(y_clean, train_pred)
    
    # Overfitting penalty
    overfit_gap = train_r2 - cv_r2
    penalty = 0.5 * max(0, overfit_gap - 0.15)
    
    objective_value = cv_r2 - penalty
    
    # Log metrics
    trial.set_user_attr('cv_r2', cv_r2)
    trial.set_user_attr('train_r2', train_r2)
    trial.set_user_attr('overfit_gap', overfit_gap)
    
    return objective_value

print("\n🔍 Running Optuna optimization (50 trials)...")
study_elasticnet = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_SEED)
)
study_elasticnet.optimize(objective_elasticnet, n_trials=50, show_progress_bar=True)

print("\n✅ ElasticNet Optimization Complete!")
print(f"   Best CV R²: {study_elasticnet.best_trial.user_attrs['cv_r2']:.4f}")
print(f"   Best Train R²: {study_elasticnet.best_trial.user_attrs['train_r2']:.4f}")
print(f"   Overfit Gap: {study_elasticnet.best_trial.user_attrs['overfit_gap']:.4f}")
print("\n   Best Parameters:")
for key, value in study_elasticnet.best_params.items():
    print(f"     {key}: {value}")

# ============================================================================
# 3. COMPARE WITH BASELINE XGBOOST
# ============================================================================
print("\n" + "="*80)
print("BASELINE COMPARISON: XGBoost (Tuned)")
print("="*80)

# Load tuned XGBoost params
with open('output_tuning/best_params.json', 'r') as f:
    tuned_params = json.load(f)

import xgboost as xgb
xgb_params = tuned_params['XGBoost'].copy()
xgb_params.update({
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbosity': 0
})
model_xgb = xgb.XGBRegressor(**xgb_params)

cv_scores_xgb = cross_val_score(model_xgb, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
cv_r2_xgb = np.mean(cv_scores_xgb)
cv_std_xgb = np.std(cv_scores_xgb)

model_xgb.fit(X_selected_df, y_clean)
train_pred_xgb = model_xgb.predict(X_selected_df)
train_r2_xgb = r2_score(y_clean, train_pred_xgb)
overfit_xgb = train_r2_xgb - cv_r2_xgb

print(f"\n   CV R²: {cv_r2_xgb:.4f} (±{cv_std_xgb:.4f})")
print(f"   Train R²: {train_r2_xgb:.4f}")
print(f"   Overfit Gap: {overfit_xgb:.4f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

results_summary = {
    'SVR': {
        'cv_r2': study_svr.best_trial.user_attrs['cv_r2'],
        'train_r2': study_svr.best_trial.user_attrs['train_r2'],
        'overfit': study_svr.best_trial.user_attrs['overfit_gap'],
        'params': study_svr.best_params
    },
    'ElasticNet': {
        'cv_r2': study_elasticnet.best_trial.user_attrs['cv_r2'],
        'train_r2': study_elasticnet.best_trial.user_attrs['train_r2'],
        'overfit': study_elasticnet.best_trial.user_attrs['overfit_gap'],
        'params': study_elasticnet.best_params
    },
    'XGBoost': {
        'cv_r2': cv_r2_xgb,
        'train_r2': train_r2_xgb,
        'overfit': overfit_xgb,
        'params': xgb_params
    }
}

print("\n📊 MODEL COMPARISON:")
print("-"*80)
print(f"{'Model':<15} | {'CV R²':<10} | {'Train R²':<10} | {'Overfit Gap':<12}")
print("-"*80)
for model_name, metrics in results_summary.items():
    print(f"{model_name:<15} | {metrics['cv_r2']:<10.4f} | {metrics['train_r2']:<10.4f} | {metrics['overfit']:<12.4f}")

# Find best model
best_model = max(results_summary.items(), key=lambda x: x[1]['cv_r2'])
print("\n🏆 BEST MODEL:")
print(f"   Model: {best_model[0]}")
print(f"   CV R²: {best_model[1]['cv_r2']:.4f}")
print(f"   Overfit Gap: {best_model[1]['overfit']:.4f}")

# Save results
output_dir = 'output_svr_elasticnet'
import os
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/tuning_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f"\n✓ Saved: {output_dir}/tuning_results.json")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CV R² comparison
ax1 = axes[0, 0]
models = list(results_summary.keys())
cv_r2s = [results_summary[m]['cv_r2'] for m in models]
colors = ['steelblue', 'coral', 'darkgreen']
bars = ax1.bar(models, cv_r2s, color=colors, alpha=0.7)
ax1.set_ylabel('CV R²', fontsize=11)
ax1.set_title('CV R² Comparison', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([min(cv_r2s) - 0.02, max(cv_r2s) + 0.02])

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# Plot 2: Overfitting comparison
ax2 = axes[0, 1]
overfits = [results_summary[m]['overfit'] for m in models]
bars = ax2.bar(models, overfits, color=colors, alpha=0.7)
ax2.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Target (20%)')
ax2.set_ylabel('Overfit Gap', fontsize=11)
ax2.set_title('Overfitting Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# Plot 3: Train vs CV R²
ax3 = axes[1, 0]
x_pos = np.arange(len(models))
width = 0.35
train_r2s = [results_summary[m]['train_r2'] for m in models]
ax3.bar(x_pos - width/2, train_r2s, width, label='Train R²', alpha=0.8, color='lightblue')
ax3.bar(x_pos + width/2, cv_r2s, width, label='CV R²', alpha=0.8, color='orange')
ax3.set_ylabel('R²', fontsize=11)
ax3.set_title('Train vs CV R²', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Optuna optimization history (SVR)
ax4 = axes[1, 1]
svr_trials = [trial.user_attrs['cv_r2'] for trial in study_svr.trials]
elasticnet_trials = [trial.user_attrs['cv_r2'] for trial in study_elasticnet.trials]
ax4.plot(range(len(svr_trials)), svr_trials, 'o-', label='SVR', alpha=0.6, markersize=4)
ax4.plot(range(len(elasticnet_trials)), elasticnet_trials, 's-', label='ElasticNet', alpha=0.6, markersize=4)
ax4.axhline(y=cv_r2_xgb, color='darkgreen', linestyle='--', linewidth=2, label='XGBoost baseline')
ax4.set_xlabel('Trial Number', fontsize=11)
ax4.set_ylabel('CV R²', fontsize=11)
ax4.set_title('Optimization History', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/svr_elasticnet_tuning.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/svr_elasticnet_tuning.png")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print("\n📈 SVR - Top 5 Trials:")
svr_df = study_svr.trials_dataframe().sort_values('value', ascending=False).head(5)
print(svr_df[['number', 'value', 'user_attrs_cv_r2', 'user_attrs_overfit_gap']].to_string(index=False))

print("\n📈 ElasticNet - Top 5 Trials:")
elasticnet_df = study_elasticnet.trials_dataframe().sort_values('value', ascending=False).head(5)
print(elasticnet_df[['number', 'value', 'user_attrs_cv_r2', 'user_attrs_overfit_gap']].to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if best_model[0] == 'XGBoost':
    print("\n💡 XGBoost remains the best model.")
    print("   SVR and ElasticNet did not outperform the baseline.")
else:
    print(f"\n💡 {best_model[0]} outperforms XGBoost!")
    improvement = (best_model[1]['cv_r2'] - cv_r2_xgb) * 100
    print(f"   Improvement: {improvement:+.2f}%")
    print(f"   Consider using {best_model[0]} in the final ensemble.")

print("\n" + "="*80)
