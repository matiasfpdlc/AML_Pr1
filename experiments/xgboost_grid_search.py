"""
Comprehensive XGBoost Grid Search to improve from CV R² 0.5.
Focus on reducing overfitting and maximizing CV performance.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from skrebate import ReliefF
import json
from datetime import datetime
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
print("XGBOOST COMPREHENSIVE GRID SEARCH")
print("Goal: Improve CV R² from 0.5 to > 0.52")
print("="*80)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================
print("\n📂 Loading data...")
data_path = project_root / 'eth-aml-2025-project-1'
loader = DataLoader(str(data_path))
X_train, y_train = loader.load_train_data()
X_test, test_ids = loader.load_test_data()

print(f"✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Test: {X_test.shape[0]} samples")

# Convert to numpy
if hasattr(X_train, 'values'):
    X_train = X_train.values
if hasattr(X_test, 'values'):
    X_test = X_test.values

# Imputation
print("\n🔧 Preprocessing...")
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Remove outliers
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
outlier_labels = iso_forest.fit_predict(X_train_imputed)
inlier_mask = outlier_labels == 1
X_clean = X_train_imputed[inlier_mask]
y_clean = y_train.values[inlier_mask] if hasattr(y_train, 'values') else y_train[inlier_mask]

print(f"✓ After outlier removal: {X_clean.shape[0]} samples")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_clean)
X_test_scaled = scaler.transform(X_test_imputed)

# Feature selection with ReliefF (k=250 was found optimal)
print("\n🔬 Applying ReliefF feature selection (k=250)...")
relief = ReliefF(n_features_to_select=250, n_neighbors=10, n_jobs=-1)
X_train_selected = relief.fit_transform(X_train_scaled, y_clean)
X_test_selected = relief.transform(X_test_scaled)
print(f"✓ Selected: {X_train_selected.shape[1]} features")

# Prepare CV
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ============================================================================
# COMPREHENSIVE GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("RUNNING COMPREHENSIVE GRID SEARCH")
print("="*80)

# Define parameter grid - comprehensive but focused on anti-overfitting
param_grid = {
    # Tree structure - limit complexity
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [3, 5, 8, 10, 15],
    
    # Learning rate and boosting
    'learning_rate': [0.01, 0.02, 0.03, 0.05],
    'n_estimators': [100, 150, 200, 300],
    
    # Sampling - prevent overfitting
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    
    # Regularization - key to reduce overfitting
    'reg_alpha': [0, 0.5, 1, 3, 5],  # L1
    'reg_lambda': [1, 3, 5, 8, 10],  # L2
    
    # Other
    'gamma': [0, 0.1, 0.3, 0.5]  # Minimum loss reduction
}

print(f"\n🔍 Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")
print(f"   This would take too long, using RandomizedSearchCV instead...")

# Use RandomizedSearchCV for efficiency
from sklearn.model_selection import RandomizedSearchCV

print("\n⚙️  Setting up Randomized Search (500 iterations)...")
xgb_model = xgb.XGBRegressor(
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0
)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=500,  # Try 500 random combinations
    cv=kfold,
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_SEED,
    return_train_score=True
)

print("\n🚀 Starting search... (this may take 20-40 minutes)")
print(f"   - Testing 500 random parameter combinations")
print(f"   - 5-fold cross-validation for each")
print(f"   - Total: ~2,500 model fits")

random_search.fit(X_train_selected, y_clean)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SEARCH RESULTS")
print("="*80)

best_params = random_search.best_params_
best_cv_score = random_search.best_score_
print(f"\n🏆 BEST CV R²: {best_cv_score:.4f}")
print(f"\n📋 BEST PARAMETERS:")
for param, value in sorted(best_params.items()):
    print(f"   {param:20s}: {value}")

# Get train score for best model
best_estimator = random_search.best_estimator_
train_pred = best_estimator.predict(X_train_selected)
train_r2 = r2_score(y_clean, train_pred)
overfit_gap = train_r2 - best_cv_score

print(f"\n📊 OVERFITTING ANALYSIS:")
print(f"   Train R²: {train_r2:.4f}")
print(f"   CV R²:    {best_cv_score:.4f}")
print(f"   Gap:      {overfit_gap:.4f} ({overfit_gap/train_r2*100:.1f}%)")

# Analyze top 20 configurations
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

print(f"\n📈 TOP 20 CONFIGURATIONS:")
top_20 = results_df.head(20)[['mean_test_score', 'std_test_score', 'mean_train_score', 
                                'param_max_depth', 'param_learning_rate', 'param_n_estimators',
                                'param_reg_alpha', 'param_reg_lambda']]
print(top_20.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_dir = Path(__file__).parent / 'output_xgb_grid'
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save best parameters
params_file = output_dir / f'best_params_{timestamp}.json'
with open(params_file, 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"\n✅ Saved best parameters: {params_file}")

# Save full results
results_file = output_dir / f'grid_search_results_{timestamp}.csv'
results_df.to_csv(results_file, index=False)
print(f"✅ Saved full results: {results_file}")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

test_pred = best_estimator.predict(X_test_selected)
test_pred_rounded = np.round(test_pred).astype(int)

print(f"\n🔮 Test predictions:")
print(f"   Range: [{test_pred_rounded.min()}, {test_pred_rounded.max()}]")
print(f"   Mean:  {test_pred_rounded.mean():.1f}")

# Save submission
submission_file = output_dir / f'submission_xgb_tuned_{timestamp}.csv'
submission_df = pd.DataFrame({
    'id': range(len(test_pred_rounded)),
    'y': test_pred_rounded
})
submission_df.to_csv(submission_file, index=False)
print(f"\n✅ Saved submission: {submission_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

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

# Plot 2: Residuals
ax2 = axes[0, 1]
residuals = y_clean - train_pred
ax2.scatter(train_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Age (years)', fontsize=11)
ax2.set_ylabel('Residuals (years)', fontsize=11)
ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: CV Score Distribution across folds
ax3 = axes[0, 2]
top_10 = results_df.head(10)
x = range(len(top_10))
ax3.bar(x, top_10['mean_test_score'], yerr=top_10['std_test_score'], 
        alpha=0.7, color='steelblue', capsize=5)
ax3.set_xlabel('Configuration Rank', fontsize=11)
ax3.set_ylabel('CV R²', fontsize=11)
ax3.set_title('Top 10 Configurations', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Parameter Importance - Learning Rate vs Score
ax4 = axes[1, 0]
lr_scores = results_df.groupby('param_learning_rate')['mean_test_score'].mean().sort_index()
ax4.plot(lr_scores.index, lr_scores.values, 'o-', linewidth=2, markersize=8)
ax4.set_xlabel('Learning Rate', fontsize=11)
ax4.set_ylabel('Mean CV R²', fontsize=11)
ax4.set_title('Learning Rate Impact', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Parameter Importance - Max Depth vs Score
ax5 = axes[1, 1]
depth_scores = results_df.groupby('param_max_depth')['mean_test_score'].mean().sort_index()
ax5.bar(depth_scores.index.astype(str), depth_scores.values, alpha=0.7, color='coral')
ax5.set_xlabel('Max Depth', fontsize=11)
ax5.set_ylabel('Mean CV R²', fontsize=11)
ax5.set_title('Max Depth Impact', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Test Predictions Distribution
ax6 = axes[1, 2]
ax6.hist(test_pred_rounded, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax6.set_xlabel('Predicted Age', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Test Predictions Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = output_dir / f'xgb_grid_search_analysis_{timestamp}.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization: {plot_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✨ FINAL RESULTS:")
print(f"   Best CV R²:    {best_cv_score:.4f}")
print(f"   Train R²:      {train_r2:.4f}")
print(f"   Overfit Gap:   {overfit_gap:.4f}")
print(f"\n📁 Output saved to: {output_dir}/")
print(f"   - Best parameters JSON")
print(f"   - Full results CSV")
print(f"   - Test submission CSV")
print(f"   - Analysis plots PNG")

if best_cv_score > 0.52:
    print(f"\n🎉 SUCCESS! Achieved CV R² > 0.52")
elif best_cv_score > 0.51:
    print(f"\n✅ Good improvement! Close to 0.52 target")
else:
    print(f"\n⚠️  Further tuning may be needed")

print("\n" + "="*80)
