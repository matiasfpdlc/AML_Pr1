"""
Final Ensemble: XGBoost + LightGBM + CatBoost + SVR with ReliefF (k=250).
Optimizes ensemble weights using grid search for best CV R².
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
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import json
import os
from datetime import datetime
from skrebate import ReliefF
import warnings
warnings.filterwarnings('ignore')

# Add utilities to path and import
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from dataloader import DataLoader  # type: ignore

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("FINAL OPTIMIZED ENSEMBLE")
print("XGBoost + LightGBM + CatBoost + SVR with ReliefF (k=250)")
print("="*80)

# Load data
print("\n📂 Loading data...")
# Get project root directory (parent of final_models)
project_root = Path(__file__).parent.parent
data_path = project_root / 'eth-aml-2025-project-1'
loader = DataLoader(str(data_path))
X_train, y_train = loader.load_train_data()
X_test, test_ids = loader.load_test_data()

# Convert to numpy arrays if needed
if hasattr(X_train, 'values'):
    X_train = X_train.values
if hasattr(X_test, 'values'):
    X_test = X_test.values

# Impute
print("🔧 Preprocessing...")
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Remove outliers
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
outlier_labels = iso_forest.fit_predict(X_train_imputed)
inlier_mask = outlier_labels == 1
X_clean = X_train_imputed[inlier_mask]
y_clean = y_train.values[inlier_mask] if hasattr(y_train, 'values') else y_train[inlier_mask]

print(f"✓ Training data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
print(f"✓ Test data: {X_test_imputed.shape[0]} samples")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_clean)
X_test_scaled = scaler.transform(X_test_imputed)

# Apply ReliefF (k=250)
print("\n🔬 Applying ReliefF feature selection (k=250)...")
relief = ReliefF(n_features_to_select=250, n_neighbors=10, n_jobs=-1)
X_train_selected = relief.fit_transform(X_train_scaled, y_clean)
X_test_selected = relief.transform(X_test_scaled)
print(f"✓ Selected: {X_train_selected.shape[1]} features")

# Load tuned parameters
print("\n⚙️  Loading tuned hyperparameters...")
tuning_path = project_root / 'experiments' / 'output_tuning' / 'best_params.json'
with open(tuning_path, 'r') as f:
    tuned_params = json.load(f)

svr_path = project_root / 'experiments' / 'output_svr_elasticnet' / 'tuning_results.json'
with open(svr_path, 'r') as f:
    svr_params = json.load(f)

# Prepare CV
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ============================================================================
# TRAIN INDIVIDUAL MODELS
# ============================================================================
print("\n" + "="*80)
print("TRAINING INDIVIDUAL MODELS")
print("="*80)

models = {}
model_predictions = {}
model_cv_scores = {}

# 1. XGBoost
print("\n[1/4] Training XGBoost...")
xgb_params = tuned_params['XGBoost'].copy()
xgb_params.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': 0})
model_xgb = xgb.XGBRegressor(**xgb_params)

cv_scores = cross_val_score(model_xgb, X_train_selected, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
cv_r2_xgb = np.mean(cv_scores)
print(f"   CV R²: {cv_r2_xgb:.4f} (±{np.std(cv_scores):.4f})")

model_xgb.fit(X_train_selected, y_clean)
models['XGBoost'] = model_xgb
model_cv_scores['XGBoost'] = cv_r2_xgb

# 2. LightGBM
print("\n[2/4] Training LightGBM...")
lgb_params = tuned_params['LightGBM'].copy()
lgb_params.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': -1})
model_lgb = lgb.LGBMRegressor(**lgb_params)

cv_scores = cross_val_score(model_lgb, X_train_selected, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
cv_r2_lgb = np.mean(cv_scores)
print(f"   CV R²: {cv_r2_lgb:.4f} (±{np.std(cv_scores):.4f})")

model_lgb.fit(X_train_selected, y_clean)
models['LightGBM'] = model_lgb
model_cv_scores['LightGBM'] = cv_r2_lgb

# 3. CatBoost
print("\n[3/4] Training CatBoost...")
catboost_params = tuned_params['CatBoost'].copy()
catboost_params.update({'random_state': RANDOM_SEED, 'verbose': False})
model_catboost = CatBoostRegressor(**catboost_params)

cv_scores = cross_val_score(model_catboost, X_train_selected, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
cv_r2_catboost = np.mean(cv_scores)
print(f"   CV R²: {cv_r2_catboost:.4f} (±{np.std(cv_scores):.4f})")

model_catboost.fit(X_train_selected, y_clean, verbose=False)
models['CatBoost'] = model_catboost
model_cv_scores['CatBoost'] = cv_r2_catboost

# 4. SVR
print("\n[4/4] Training SVR...")
svr_best_params = svr_params['SVR']['params']
model_svr = SVR(
    kernel=svr_best_params['kernel'],
    C=svr_best_params['C'],
    epsilon=svr_best_params['epsilon'],
    gamma=svr_best_params['gamma'],
    cache_size=1000
)

cv_scores = cross_val_score(model_svr, X_train_selected, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
cv_r2_svr = np.mean(cv_scores)
print(f"   CV R²: {cv_r2_svr:.4f} (±{np.std(cv_scores):.4f})")

model_svr.fit(X_train_selected, y_clean)
models['SVR'] = model_svr
model_cv_scores['SVR'] = cv_r2_svr

# ============================================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*80)

# Generate predictions for each fold
print("\nGenerating cross-validation predictions...")
fold_predictions = {name: [] for name in models.keys()}
fold_targets = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_selected)):
    X_fold_train = X_train_selected[train_idx]
    y_fold_train = y_clean[train_idx]
    X_fold_val = X_train_selected[val_idx]
    y_fold_val = y_clean[val_idx]
    
    fold_targets.append(y_fold_val)
    
    for name, model_class in [
        ('XGBoost', xgb.XGBRegressor),
        ('LightGBM', lgb.LGBMRegressor),
        ('CatBoost', CatBoostRegressor),
        ('SVR', SVR)
    ]:
        if name == 'XGBoost':
            params = xgb_params
            model = model_class(**params)
        elif name == 'LightGBM':
            params = lgb_params
            model = model_class(**params)
        elif name == 'CatBoost':
            params = catboost_params
            model = model_class(**params)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            fold_predictions[name].append(model.predict(X_fold_val))
            continue
        else:  # SVR
            model = SVR(
                kernel=svr_best_params['kernel'],
                C=svr_best_params['C'],
                epsilon=svr_best_params['epsilon'],
                gamma=svr_best_params['gamma'],
                cache_size=1000
            )
        
        model.fit(X_fold_train, y_fold_train)
        fold_predictions[name].append(model.predict(X_fold_val))

# Concatenate all fold predictions
cv_predictions = {name: np.concatenate(preds) for name, preds in fold_predictions.items()}
cv_targets = np.concatenate(fold_targets)

print("✓ Generated predictions for all folds")

# Grid search for best weights
print("\n🔍 Searching for optimal weights...")
print("   Testing weight combinations...")

best_weights = None
best_cv_r2 = -np.inf
weight_results = []

# Generate weight combinations
from itertools import product
weight_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# To reduce search space, we'll test weights that sum to 1
model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'SVR']
n_combinations_tested = 0

for w_xgb in weight_steps:
    for w_lgb in weight_steps:
        for w_cat in weight_steps:
            for w_svr in weight_steps:
                if abs(w_xgb + w_lgb + w_cat + w_svr - 1.0) < 0.001:  # Sum to 1
                    weights = [w_xgb, w_lgb, w_cat, w_svr]
                    
                    # Create weighted ensemble prediction
                    ensemble_pred = np.zeros_like(cv_targets, dtype=float)
                    for weight, name in zip(weights, model_names):
                        ensemble_pred += weight * cv_predictions[name]
                    
                    # Calculate R²
                    cv_r2 = r2_score(cv_targets, ensemble_pred)
                    
                    weight_results.append({
                        'w_xgb': w_xgb,
                        'w_lgb': w_lgb,
                        'w_cat': w_cat,
                        'w_svr': w_svr,
                        'cv_r2': cv_r2
                    })
                    
                    if cv_r2 > best_cv_r2:
                        best_cv_r2 = cv_r2
                        best_weights = weights
                    
                    n_combinations_tested += 1

print(f"✓ Tested {n_combinations_tested} weight combinations")

# Results
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

print(f"\n🏆 BEST ENSEMBLE WEIGHTS:")
print(f"   XGBoost:  {best_weights[0]:.1f}")
print(f"   LightGBM: {best_weights[1]:.1f}")
print(f"   CatBoost: {best_weights[2]:.1f}")
print(f"   SVR:      {best_weights[3]:.1f}")
print(f"\n   Ensemble CV R²: {best_cv_r2:.4f}")

print(f"\n📊 INDIVIDUAL MODEL CV R²:")
for name, cv_r2 in model_cv_scores.items():
    improvement = (best_cv_r2 - cv_r2) * 100
    print(f"   {name:<10s}: {cv_r2:.4f} ({improvement:+.2f}%)")

# Top 10 weight combinations
print(f"\n📈 TOP 10 WEIGHT COMBINATIONS:")
weight_df = pd.DataFrame(weight_results).sort_values('cv_r2', ascending=False)
print(weight_df.head(10).to_string(index=False))

# ============================================================================
# GENERATE FINAL PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING FINAL PREDICTIONS")
print("="*80)

# Train predictions with best weights
train_pred = np.zeros(len(y_clean))
for weight, name in zip(best_weights, model_names):
    train_pred += weight * models[name].predict(X_train_selected)

train_r2 = r2_score(y_clean, train_pred)
overfit = train_r2 - best_cv_r2

print(f"\n✓ Train R²: {train_r2:.4f}")
print(f"✓ CV R²: {best_cv_r2:.4f}")
print(f"✓ Overfit Gap: {overfit:.4f}")

# Test predictions
print("\n🔮 Generating test predictions...")
test_pred = np.zeros(len(X_test_selected))
for weight, name in zip(best_weights, model_names):
    test_pred += weight * models[name].predict(X_test_selected)

# Round to integers
test_pred_rounded = np.round(test_pred).astype(int)

print(f"✓ Test predictions generated")
print(f"   Range: [{test_pred_rounded.min()}, {test_pred_rounded.max()}]")
print(f"   Mean: {test_pred_rounded.mean():.1f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_dir = Path(__file__).parent / 'results'
os.makedirs(output_dir, exist_ok=True)

# Save submission
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_file = output_dir / f'submission_final_ensemble_{timestamp}.csv'

submission_df = pd.DataFrame({
    'id': range(len(test_pred_rounded)),
    'y': test_pred_rounded
})
submission_df.to_csv(submission_file, index=False)
print(f"\n✅ Saved submission: {submission_file}")

# Save ensemble details
ensemble_info = {
    'weights': {
        'XGBoost': float(best_weights[0]),
        'LightGBM': float(best_weights[1]),
        'CatBoost': float(best_weights[2]),
        'SVR': float(best_weights[3])
    },
    'cv_r2': float(best_cv_r2),
    'train_r2': float(train_r2),
    'overfit_gap': float(overfit),
    'individual_cv_r2': {name: float(score) for name, score in model_cv_scores.items()},
    'feature_selection': 'ReliefF k=250',
    'timestamp': timestamp
}

info_file = output_dir / f'ensemble_info_{timestamp}.json'
with open(info_file, 'w') as f:
    json.dump(ensemble_info, f, indent=2)
print(f"✅ Saved ensemble info: {info_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: True vs Predicted (Training)
ax1 = axes[0, 0]
ax1.scatter(y_clean, train_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
min_val = min(y_clean.min(), train_pred.min())
max_val = max(y_clean.max(), train_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Age (years)', fontsize=11)
ax1.set_ylabel('Predicted Age (years)', fontsize=11)
ax1.set_title(f'Training: True vs Predicted\nR²={train_r2:.4f}', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Individual vs Ensemble CV R²
ax2 = axes[0, 1]
all_names = list(model_cv_scores.keys()) + ['Ensemble']
all_scores = list(model_cv_scores.values()) + [best_cv_r2]
colors = ['steelblue', 'darkgreen', 'coral', 'purple', 'gold']
bars = ax2.bar(all_names, all_scores, color=colors, alpha=0.7)
ax2.set_ylabel('CV R²', fontsize=11)
ax2.set_title('Individual Models vs Ensemble', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([min(all_scores) - 0.02, max(all_scores) + 0.02])

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Ensemble weights
ax3 = axes[0, 2]
ax3.pie(best_weights, labels=model_names, autopct='%1.1f%%', startangle=90, colors=colors[:4])
ax3.set_title('Optimal Ensemble Weights', fontsize=12, fontweight='bold')

# Plot 4: Residuals
ax4 = axes[1, 0]
residuals = y_clean - train_pred
ax4.scatter(train_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Age (years)', fontsize=11)
ax4.set_ylabel('Residuals (years)', fontsize=11)
ax4.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Top 20 weight combinations
ax5 = axes[1, 1]
top_20 = weight_df.head(20)
ax5.plot(range(len(top_20)), top_20['cv_r2'].values, 'o-', linewidth=2, markersize=6)
ax5.set_xlabel('Rank', fontsize=11)
ax5.set_ylabel('CV R²', fontsize=11)
ax5.set_title('Top 20 Weight Combinations', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Prediction distribution
ax6 = axes[1, 2]
ax6.hist(test_pred_rounded, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax6.set_xlabel('Predicted Age', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Test Predictions Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = output_dir / f'final_ensemble_analysis_{timestamp}.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization: {plot_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✨ Final Ensemble Performance:")
print(f"   CV R²: {best_cv_r2:.4f}")
print(f"   Improvement over best single model: {(best_cv_r2 - max(model_cv_scores.values()))*100:+.2f}%")
print(f"\n📁 Submission file: {submission_file}")
print("="*80)
