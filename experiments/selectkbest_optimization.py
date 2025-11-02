"""
Optimize SelectKBest with different scoring functions and k values.
Tests with both tuned XGBoost and vanilla XGBoost.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, r_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import json
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
print("SELECTKBEST OPTIMIZATION")
print("Testing different scoring functions and k values")
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

# Standardize (needed for mutual_info)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Load tuned parameters
with open('output_tuning/best_params.json', 'r') as f:
    tuned_params = json.load(f)

# Define scoring functions to test
scoring_functions = {
    'f_regression': f_regression,
    'mutual_info_regression': mutual_info_regression,
    'r_regression': r_regression,  # Pearson correlation
}

# Define k values to test
k_values = [30, 50, 75, 100, 150, 200, 250, 300, 350, 400]

print("\n🔬 Testing configurations...")
print("-"*80)

results = []

for score_name, score_func in scoring_functions.items():
    print(f"\n[{score_name.upper()}]")
    
    for k in k_values:
        try:
            # Apply SelectKBest
            if score_name == 'mutual_info_regression':
                selector = SelectKBest(score_func=lambda X, y: score_func(X, y, random_state=RANDOM_SEED), k=k)
            else:
                selector = SelectKBest(score_func=score_func, k=k)
            
            X_selected = selector.fit_transform(X_scaled, y_clean)
            X_selected_df = pd.DataFrame(X_selected)
            
            # Test with TUNED XGBoost
            xgb_tuned_params = tuned_params['XGBoost'].copy()
            xgb_tuned_params.update({
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbosity': 0
            })
            model_tuned = xgb.XGBRegressor(**xgb_tuned_params)
            
            kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            cv_scores_tuned = cross_val_score(model_tuned, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
            cv_r2_tuned = np.mean(cv_scores_tuned)
            cv_std_tuned = np.std(cv_scores_tuned)
            
            model_tuned.fit(X_selected_df, y_clean)
            train_pred_tuned = model_tuned.predict(X_selected_df)
            train_r2_tuned = r2_score(y_clean, train_pred_tuned)
            overfit_tuned = train_r2_tuned - cv_r2_tuned
            
            # Test with VANILLA XGBoost
            vanilla_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.3,
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbosity': 0
            }
            model_vanilla = xgb.XGBRegressor(**vanilla_params)
            
            cv_scores_vanilla = cross_val_score(model_vanilla, X_selected_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
            cv_r2_vanilla = np.mean(cv_scores_vanilla)
            cv_std_vanilla = np.std(cv_scores_vanilla)
            
            model_vanilla.fit(X_selected_df, y_clean)
            train_pred_vanilla = model_vanilla.predict(X_selected_df)
            train_r2_vanilla = r2_score(y_clean, train_pred_vanilla)
            overfit_vanilla = train_r2_vanilla - cv_r2_vanilla
            
            results.append({
                'scoring_func': score_name,
                'k': k,
                'tuned_cv_r2': cv_r2_tuned,
                'tuned_cv_std': cv_std_tuned,
                'tuned_train_r2': train_r2_tuned,
                'tuned_overfit': overfit_tuned,
                'vanilla_cv_r2': cv_r2_vanilla,
                'vanilla_cv_std': cv_std_vanilla,
                'vanilla_train_r2': train_r2_vanilla,
                'vanilla_overfit': overfit_vanilla
            })
            
            print(f"  k={k:3d} | Tuned: CV={cv_r2_tuned:.4f} Gap={overfit_tuned:.4f} | Vanilla: CV={cv_r2_vanilla:.4f} Gap={overfit_vanilla:.4f}")
            
        except Exception as e:
            print(f"  k={k:3d} | Failed - {str(e)[:60]}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('selectkbest_optimization_results.csv', index=False)
print(f"\n✓ Saved: selectkbest_optimization_results.csv")

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Best configurations for tuned model
print("\n🏆 BEST CONFIGURATIONS WITH TUNED XGBOOST:")
print("-"*80)
best_tuned = results_df.loc[results_df['tuned_cv_r2'].idxmax()]
print(f"  Scoring: {best_tuned['scoring_func']}")
print(f"  k: {best_tuned['k']:.0f}")
print(f"  CV R²: {best_tuned['tuned_cv_r2']:.4f} (±{best_tuned['tuned_cv_std']:.4f})")
print(f"  Train R²: {best_tuned['tuned_train_r2']:.4f}")
print(f"  Overfit Gap: {best_tuned['tuned_overfit']:.4f}")

# Best configurations for vanilla model
print("\n🏆 BEST CONFIGURATIONS WITH VANILLA XGBOOST:")
print("-"*80)
best_vanilla = results_df.loc[results_df['vanilla_cv_r2'].idxmax()]
print(f"  Scoring: {best_vanilla['scoring_func']}")
print(f"  k: {best_vanilla['k']:.0f}")
print(f"  CV R²: {best_vanilla['vanilla_cv_r2']:.4f} (±{best_vanilla['vanilla_cv_std']:.4f})")
print(f"  Train R²: {best_vanilla['vanilla_train_r2']:.4f}")
print(f"  Overfit Gap: {best_vanilla['vanilla_overfit']:.4f}")

# Best per scoring function (tuned)
print("\n📊 BEST PER SCORING FUNCTION (Tuned XGBoost):")
print("-"*80)
for score_name in scoring_functions.keys():
    score_results = results_df[results_df['scoring_func'] == score_name]
    if len(score_results) > 0:
        best_score = score_results.loc[score_results['tuned_cv_r2'].idxmax()]
        print(f"  {score_name:25s}: k={best_score['k']:3.0f}, CV R²={best_score['tuned_cv_r2']:.4f}, Gap={best_score['tuned_overfit']:.4f}")

# Best per scoring function (vanilla)
print("\n📊 BEST PER SCORING FUNCTION (Vanilla XGBoost):")
print("-"*80)
for score_name in scoring_functions.keys():
    score_results = results_df[results_df['scoring_func'] == score_name]
    if len(score_results) > 0:
        best_score = score_results.loc[score_results['vanilla_cv_r2'].idxmax()]
        print(f"  {score_name:25s}: k={best_score['k']:3.0f}, CV R²={best_score['vanilla_cv_r2']:.4f}, Gap={best_score['vanilla_overfit']:.4f}")

# Compare tuned vs vanilla
print("\n⚖️  TUNED vs VANILLA COMPARISON:")
print("-"*80)
for score_name in scoring_functions.keys():
    score_results = results_df[results_df['scoring_func'] == score_name]
    if len(score_results) > 0:
        best_tuned_score = score_results.loc[score_results['tuned_cv_r2'].idxmax()]
        best_vanilla_score = score_results.loc[score_results['vanilla_cv_r2'].idxmax()]
        improvement = (best_tuned_score['tuned_cv_r2'] - best_vanilla_score['vanilla_cv_r2']) * 100
        print(f"  {score_name:25s}: Tuned {best_tuned_score['tuned_cv_r2']:.4f} vs Vanilla {best_vanilla_score['vanilla_cv_r2']:.4f} ({improvement:+.2f}%)")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

colors = {'f_regression': 'steelblue', 'mutual_info_regression': 'darkgreen', 'r_regression': 'coral'}

# Plot 1: CV R² for tuned models
ax1 = fig.add_subplot(gs[0, 0])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name]
    if len(score_data) > 0:
        ax1.plot(score_data['k'], score_data['tuned_cv_r2'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax1.set_xlabel('k (Number of Features)', fontsize=10)
ax1.set_ylabel('CV R²', fontsize=10)
ax1.set_title('Tuned XGBoost: CV R² vs k', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: CV R² for vanilla models
ax2 = fig.add_subplot(gs[0, 1])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name]
    if len(score_data) > 0:
        ax2.plot(score_data['k'], score_data['vanilla_cv_r2'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax2.set_xlabel('k (Number of Features)', fontsize=10)
ax2.set_ylabel('CV R²', fontsize=10)
ax2.set_title('Vanilla XGBoost: CV R² vs k', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Direct comparison (best k per scoring)
ax3 = fig.add_subplot(gs[0, 2])
x_pos = np.arange(len(scoring_functions))
width = 0.35
tuned_best = [results_df[results_df['scoring_func'] == s]['tuned_cv_r2'].max() for s in scoring_functions.keys()]
vanilla_best = [results_df[results_df['scoring_func'] == s]['vanilla_cv_r2'].max() for s in scoring_functions.keys()]
ax3.bar(x_pos - width/2, tuned_best, width, label='Tuned', alpha=0.8, color='steelblue')
ax3.bar(x_pos + width/2, vanilla_best, width, label='Vanilla', alpha=0.8, color='orange')
ax3.set_xlabel('Scoring Function', fontsize=10)
ax3.set_ylabel('Best CV R²', fontsize=10)
ax3.set_title('Best CV R²: Tuned vs Vanilla', fontsize=11, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([s.replace('_', '\n') for s in scoring_functions.keys()], fontsize=8)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Overfitting gap (tuned)
ax4 = fig.add_subplot(gs[1, 0])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name]
    if len(score_data) > 0:
        ax4.plot(score_data['k'], score_data['tuned_overfit'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax4.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.20)')
ax4.set_xlabel('k (Number of Features)', fontsize=10)
ax4.set_ylabel('Overfit Gap', fontsize=10)
ax4.set_title('Tuned XGBoost: Overfitting vs k', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Overfitting gap (vanilla)
ax5 = fig.add_subplot(gs[1, 1])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name]
    if len(score_data) > 0:
        ax5.plot(score_data['k'], score_data['vanilla_overfit'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax5.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.20)')
ax5.set_xlabel('k (Number of Features)', fontsize=10)
ax5.set_ylabel('Overfit Gap', fontsize=10)
ax5.set_title('Vanilla XGBoost: Overfitting vs k', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Overfitting comparison
ax6 = fig.add_subplot(gs[1, 2])
x_pos = np.arange(len(scoring_functions))
tuned_overfit = [results_df[results_df['scoring_func'] == s]['tuned_overfit'].mean() for s in scoring_functions.keys()]
vanilla_overfit = [results_df[results_df['scoring_func'] == s]['vanilla_overfit'].mean() for s in scoring_functions.keys()]
ax6.bar(x_pos - width/2, tuned_overfit, width, label='Tuned', alpha=0.8, color='steelblue')
ax6.bar(x_pos + width/2, vanilla_overfit, width, label='Vanilla', alpha=0.8, color='orange')
ax6.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, linewidth=2)
ax6.set_xlabel('Scoring Function', fontsize=10)
ax6.set_ylabel('Avg Overfit Gap', fontsize=10)
ax6.set_title('Avg Overfitting: Tuned vs Vanilla', fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([s.replace('_', '\n') for s in scoring_functions.keys()], fontsize=8)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Efficiency (CV R² per feature) - Tuned
ax7 = fig.add_subplot(gs[2, 0])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name].copy()
    if len(score_data) > 0:
        score_data['efficiency'] = score_data['tuned_cv_r2'] / score_data['k']
        ax7.plot(score_data['k'], score_data['efficiency'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax7.set_xlabel('k (Number of Features)', fontsize=10)
ax7.set_ylabel('Efficiency (CV R² / k)', fontsize=10)
ax7.set_title('Tuned: Efficiency vs k', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Efficiency (CV R² per feature) - Vanilla
ax8 = fig.add_subplot(gs[2, 1])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name].copy()
    if len(score_data) > 0:
        score_data['efficiency'] = score_data['vanilla_cv_r2'] / score_data['k']
        ax8.plot(score_data['k'], score_data['efficiency'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax8.set_xlabel('k (Number of Features)', fontsize=10)
ax8.set_ylabel('Efficiency (CV R² / k)', fontsize=10)
ax8.set_title('Vanilla: Efficiency vs k', fontsize=11, fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Plot 9: Trade-off score (CV R² - penalty*overfit)
ax9 = fig.add_subplot(gs[2, 2])
for score_name in scoring_functions.keys():
    score_data = results_df[results_df['scoring_func'] == score_name].copy()
    if len(score_data) > 0:
        score_data['tradeoff'] = score_data['tuned_cv_r2'] - 0.5 * np.maximum(0, score_data['tuned_overfit'] - 0.15)
        ax9.plot(score_data['k'], score_data['tradeoff'], 'o-', label=score_name, 
                linewidth=2, markersize=6, color=colors[score_name])
ax9.set_xlabel('k (Number of Features)', fontsize=10)
ax9.set_ylabel('Trade-off Score', fontsize=10)
ax9.set_title('Tuned: Performance-Overfit Trade-off', fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.savefig('selectkbest_optimization.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: selectkbest_optimization.png")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Find best overall trade-off
results_df['tradeoff_score'] = results_df['tuned_cv_r2'] - 0.5 * np.maximum(0, results_df['tuned_overfit'] - 0.15)
best_tradeoff = results_df.loc[results_df['tradeoff_score'].idxmax()]

print(f"\n💡 RECOMMENDED CONFIGURATION:")
print(f"   Model: Tuned XGBoost")
print(f"   Scoring Function: {best_tradeoff['scoring_func']}")
print(f"   k: {best_tradeoff['k']:.0f}")
print(f"   CV R²: {best_tradeoff['tuned_cv_r2']:.4f} (±{best_tradeoff['tuned_cv_std']:.4f})")
print(f"   Train R²: {best_tradeoff['tuned_train_r2']:.4f}")
print(f"   Overfit Gap: {best_tradeoff['tuned_overfit']:.4f}")
print(f"   Trade-off Score: {best_tradeoff['tradeoff_score']:.4f}")

# Compare with baseline
baseline_cv = 0.4881  # From previous tuned ensemble
improvement = (best_tradeoff['tuned_cv_r2'] - baseline_cv) * 100
print(f"\n   vs Previous Best (k=100, f_regression): CV R²={baseline_cv:.4f}")
print(f"   Improvement: {improvement:+.2f}%")

print("\n" + "="*80)
