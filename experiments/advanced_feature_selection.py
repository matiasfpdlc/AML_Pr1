"""
Advanced feature selection with mRMR and ReliefF.
Tests different configurations and k values with tuned XGBoost.
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
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

# Import mRMR and ReliefF
from mrmr import mrmr_regression
from skrebate import ReliefF

# Add utilities to path and import
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from dataloader import DataLoader  # type: ignore

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Get project root directory
project_root = Path(__file__).parent.parent

print("="*80)
print("ADVANCED FEATURE SELECTION: mRMR & ReliefF")
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
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
outlier_labels = iso_forest.fit_predict(X_imputed)
inlier_mask = outlier_labels == 1
X_clean = X_imputed[inlier_mask]
y_clean = y_train.values[inlier_mask] if hasattr(y_train, 'values') else y_train[inlier_mask]

print(f"✓ Data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Create DataFrame for mRMR (needs column names)
feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
X_df = pd.DataFrame(X_scaled, columns=feature_names)
y_series = pd.Series(y_clean, name='target')

# Load tuned parameters
with open('output_tuning/best_params.json', 'r') as f:
    tuned_params = json.load(f)

# Define k values to test
k_values = [30, 50, 75, 100, 150, 200, 250, 300]

print("\n🔬 Testing feature selection methods...")
print("-"*80)

results = []

# ============================================================================
# 1. mRMR (minimum Redundancy Maximum Relevance)
# ============================================================================
print("\n[mRMR - Minimum Redundancy Maximum Relevance]")
print("Selects features with high relevance to target and low redundancy")
print("-"*80)

for k in k_values:
    try:
        print(f"  Testing k={k}...", end=" ")
        
        # Apply mRMR
        # mRMR returns feature names, we need to get their indices
        selected_features = mrmr_regression(X=X_df, y=y_series, K=k, show_progress=False)
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X_scaled[:, selected_indices]
        X_selected_df = pd.DataFrame(X_selected)
        
        # Train with tuned XGBoost
        xgb_params = tuned_params['XGBoost'].copy()
        xgb_params.update({
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        })
        model = xgb.XGBRegressor(**xgb_params)
        
        # 5-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
        cv_r2 = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train score
        model.fit(X_selected_df, y_clean)
        train_pred = model.predict(X_selected_df)
        train_r2 = r2_score(y_clean, train_pred)
        overfit = train_r2 - cv_r2
        
        results.append({
            'method': 'mRMR',
            'k': k,
            'cv_r2': cv_r2,
            'cv_std': cv_std,
            'train_r2': train_r2,
            'overfit': overfit
        })
        
        print(f"CV R²={cv_r2:.4f} (±{cv_std:.4f}), Gap={overfit:.4f}")
        
    except Exception as e:
        print(f"Failed - {str(e)[:50]}")

# ============================================================================
# 2. ReliefF
# ============================================================================
print("\n[ReliefF - Feature Weighting Algorithm]")
print("Detects feature interactions and handles multi-class problems well")
print("-"*80)

for k in k_values:
    try:
        print(f"  Testing k={k}...", end=" ")
        
        # Apply ReliefF
        # ReliefF can be slow, use fewer neighbors for speed
        relief = ReliefF(n_features_to_select=k, n_neighbors=10, n_jobs=-1)
        X_selected = relief.fit_transform(X_scaled, y_clean)
        X_selected_df = pd.DataFrame(X_selected)
        
        # Train with tuned XGBoost
        xgb_params = tuned_params['XGBoost'].copy()
        xgb_params.update({
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        })
        model = xgb.XGBRegressor(**xgb_params)
        
        # 5-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
        cv_r2 = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train score
        model.fit(X_selected_df, y_clean)
        train_pred = model.predict(X_selected_df)
        train_r2 = r2_score(y_clean, train_pred)
        overfit = train_r2 - cv_r2
        
        results.append({
            'method': 'ReliefF',
            'k': k,
            'cv_r2': cv_r2,
            'cv_std': cv_std,
            'train_r2': train_r2,
            'overfit': overfit
        })
        
        print(f"CV R²={cv_r2:.4f} (±{cv_std:.4f}), Gap={overfit:.4f}")
        
    except Exception as e:
        print(f"Failed - {str(e)[:50]}")

# ============================================================================
# 3. Baseline: f_regression (from previous optimization)
# ============================================================================
print("\n[Baseline - f_regression (ANOVA F-test)]")
print("Standard univariate feature selection")
print("-"*80)

for k in k_values:
    try:
        print(f"  Testing k={k}...", end=" ")
        
        # Apply SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X_scaled, y_clean)
        X_selected_df = pd.DataFrame(X_selected)
        
        # Train with tuned XGBoost
        xgb_params = tuned_params['XGBoost'].copy()
        xgb_params.update({
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        })
        model = xgb.XGBRegressor(**xgb_params)
        
        # 5-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
        cv_r2 = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train score
        model.fit(X_selected_df, y_clean)
        train_pred = model.predict(X_selected_df)
        train_r2 = r2_score(y_clean, train_pred)
        overfit = train_r2 - cv_r2
        
        results.append({
            'method': 'f_regression',
            'k': k,
            'cv_r2': cv_r2,
            'cv_std': cv_std,
            'train_r2': train_r2,
            'overfit': overfit
        })
        
        print(f"CV R²={cv_r2:.4f} (±{cv_std:.4f}), Gap={overfit:.4f}")
        
    except Exception as e:
        print(f"Failed - {str(e)[:50]}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('advanced_feature_selection_results.csv', index=False)
print(f"\n✓ Saved: advanced_feature_selection_results.csv")

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Best overall
best_overall = results_df.loc[results_df['cv_r2'].idxmax()]
print(f"\n🏆 BEST OVERALL:")
print(f"   Method: {best_overall['method']}")
print(f"   k: {best_overall['k']:.0f}")
print(f"   CV R²: {best_overall['cv_r2']:.4f} (±{best_overall['cv_std']:.4f})")
print(f"   Train R²: {best_overall['train_r2']:.4f}")
print(f"   Overfit Gap: {best_overall['overfit']:.4f}")

# Best per method
print(f"\n📊 BEST PER METHOD:")
print("-"*80)
methods = ['mRMR', 'ReliefF', 'f_regression']
for method in methods:
    method_results = results_df[results_df['method'] == method]
    if len(method_results) > 0:
        best_method = method_results.loc[method_results['cv_r2'].idxmax()]
        print(f"  {method:15s}: k={best_method['k']:3.0f}, CV R²={best_method['cv_r2']:.4f}, Gap={best_method['overfit']:.4f}")

# Best with low overfitting (<0.18)
print(f"\n🎯 BEST WITH LOW OVERFITTING (<18%):")
print("-"*80)
low_overfit = results_df[results_df['overfit'] < 0.18]
if len(low_overfit) > 0:
    best_low = low_overfit.loc[low_overfit['cv_r2'].idxmax()]
    print(f"   Method: {best_low['method']}")
    print(f"   k: {best_low['k']:.0f}")
    print(f"   CV R²: {best_low['cv_r2']:.4f} (±{best_low['cv_std']:.4f})")
    print(f"   Overfit Gap: {best_low['overfit']:.4f}")
else:
    print("   No configurations with overfitting < 18%")

# Comparison table
print(f"\n📈 COMPARISON AT KEY k VALUES:")
print("-"*80)
print(f"{'k':<6} | {'mRMR':<15} | {'ReliefF':<15} | {'f_regression':<15}")
print("-"*80)
for k in [50, 75, 100, 150, 200]:
    row = f"{k:<6} |"
    for method in methods:
        method_k = results_df[(results_df['method'] == method) & (results_df['k'] == k)]
        if len(method_k) > 0:
            cv_r2 = method_k.iloc[0]['cv_r2']
            row += f" {cv_r2:.4f}         |"
        else:
            row += " N/A            |"
    print(row)

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = {'mRMR': 'steelblue', 'ReliefF': 'darkgreen', 'f_regression': 'coral'}

# Plot 1: CV R² vs k
ax1 = axes[0, 0]
for method in methods:
    method_data = results_df[results_df['method'] == method]
    if len(method_data) > 0:
        ax1.plot(method_data['k'], method_data['cv_r2'], 'o-', label=method,
                linewidth=2, markersize=8, color=colors[method])
ax1.set_xlabel('k (Number of Features)', fontsize=11)
ax1.set_ylabel('CV R²', fontsize=11)
ax1.set_title('CV R² vs Number of Features', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Overfitting gap
ax2 = axes[0, 1]
for method in methods:
    method_data = results_df[results_df['method'] == method]
    if len(method_data) > 0:
        ax2.plot(method_data['k'], method_data['overfit'], 'o-', label=method,
                linewidth=2, markersize=8, color=colors[method])
ax2.axhline(y=0.18, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Target (18%)')
ax2.set_xlabel('k (Number of Features)', fontsize=11)
ax2.set_ylabel('Overfit Gap (Train R² - CV R²)', fontsize=11)
ax2.set_title('Overfitting vs Number of Features', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Best k per method
ax3 = axes[0, 2]
best_per_method = []
best_k_per_method = []
for method in methods:
    method_results = results_df[results_df['method'] == method]
    if len(method_results) > 0:
        best = method_results.loc[method_results['cv_r2'].idxmax()]
        best_per_method.append(best['cv_r2'])
        best_k_per_method.append(best['k'])
    else:
        best_per_method.append(0)
        best_k_per_method.append(0)

bars = ax3.bar(range(len(methods)), best_per_method, color=[colors[m] for m in methods], alpha=0.7)
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=10)
ax3.set_ylabel('Best CV R²', fontsize=11)
ax3.set_title('Best CV R² per Method', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add k values on top of bars
for i, (bar, k) in enumerate(zip(bars, best_k_per_method)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'k={int(k)}', ha='center', va='bottom', fontsize=9)

# Plot 4: Train vs CV R² (mRMR)
ax4 = axes[1, 0]
mrmr_data = results_df[results_df['method'] == 'mRMR']
if len(mrmr_data) > 0:
    ax4.plot(mrmr_data['k'], mrmr_data['train_r2'], 'o-', label='Train R²', linewidth=2, markersize=8)
    ax4.plot(mrmr_data['k'], mrmr_data['cv_r2'], 's-', label='CV R²', linewidth=2, markersize=8)
    ax4.fill_between(mrmr_data['k'], mrmr_data['train_r2'], mrmr_data['cv_r2'], alpha=0.2)
ax4.set_xlabel('k (Number of Features)', fontsize=11)
ax4.set_ylabel('R²', fontsize=11)
ax4.set_title('mRMR: Train vs CV R²', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Train vs CV R² (ReliefF)
ax5 = axes[1, 1]
relief_data = results_df[results_df['method'] == 'ReliefF']
if len(relief_data) > 0:
    ax5.plot(relief_data['k'], relief_data['train_r2'], 'o-', label='Train R²', linewidth=2, markersize=8)
    ax5.plot(relief_data['k'], relief_data['cv_r2'], 's-', label='CV R²', linewidth=2, markersize=8)
    ax5.fill_between(relief_data['k'], relief_data['train_r2'], relief_data['cv_r2'], alpha=0.2)
ax5.set_xlabel('k (Number of Features)', fontsize=11)
ax5.set_ylabel('R²', fontsize=11)
ax5.set_title('ReliefF: Train vs CV R²', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: Efficiency (CV R² per feature)
ax6 = axes[1, 2]
for method in methods:
    method_data = results_df[results_df['method'] == method].copy()
    if len(method_data) > 0:
        method_data['efficiency'] = method_data['cv_r2'] / method_data['k']
        ax6.plot(method_data['k'], method_data['efficiency'], 'o-', label=method,
                linewidth=2, markersize=8, color=colors[method])
ax6.set_xlabel('k (Number of Features)', fontsize=11)
ax6.set_ylabel('Efficiency (CV R² / k)', fontsize=11)
ax6.set_title('Efficiency: Performance per Feature', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_feature_selection.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: advanced_feature_selection.png")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Trade-off score: CV R² - penalty * max(0, overfit - 0.15)
results_df['tradeoff_score'] = results_df['cv_r2'] - 0.5 * np.maximum(0, results_df['overfit'] - 0.15)
best_tradeoff = results_df.loc[results_df['tradeoff_score'].idxmax()]

print(f"\n💡 RECOMMENDED CONFIGURATION:")
print(f"   Method: {best_tradeoff['method']}")
print(f"   k: {best_tradeoff['k']:.0f}")
print(f"   CV R²: {best_tradeoff['cv_r2']:.4f} (±{best_tradeoff['cv_std']:.4f})")
print(f"   Train R²: {best_tradeoff['train_r2']:.4f}")
print(f"   Overfit Gap: {best_tradeoff['overfit']:.4f}")
print(f"   Trade-off Score: {best_tradeoff['tradeoff_score']:.4f}")

# Compare with previous best
print(f"\n   vs Previous Best (f_regression, k=200): CV R²=0.5049")
improvement = (best_tradeoff['cv_r2'] - 0.5049) * 100
print(f"   Improvement: {improvement:+.2f}%")

print("\n" + "="*80)
print("\n✨ Key Insights:")
print("   • mRMR: Reduces feature redundancy, may improve generalization")
print("   • ReliefF: Captures feature interactions, good for complex relationships")
print("   • f_regression: Fast and effective for linear relationships")
print("="*80)
