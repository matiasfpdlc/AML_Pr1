"""
Find balanced ensemble combinations with CV R² > 0.503.
Forces minimum weight contribution from each model.
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

# Get project root directory
project_root = Path(__file__).parent.parent

print("="*80)
print("BALANCED ENSEMBLE OPTIMIZATION")
print("Finding combinations with CV R² > 0.503 and balanced weights")
print("="*80)

# Load data
print("\n📂 Loading data...")
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
# GENERATE CV PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING CV PREDICTIONS")
print("="*80)

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

# ============================================================================
# BALANCED WEIGHT SEARCH
# ============================================================================
print("\n" + "="*80)
print("SEARCHING FOR BALANCED ENSEMBLES")
print("="*80)

model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'SVR']

# Strategy 1: Each model gets at least 10% weight
print("\n🔍 Strategy 1: Minimum 10% weight per model")
balanced_results_10 = []
weight_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for w_xgb in weight_steps:
    for w_lgb in weight_steps:
        for w_cat in weight_steps:
            for w_svr in weight_steps:
                # Check if each weight >= 0.1 and sum to 1.0
                weights = [w_xgb, w_lgb, w_cat, w_svr]
                if all(w >= 0.1 for w in weights) and abs(sum(weights) - 1.0) < 0.001:
                    ensemble_pred = np.zeros_like(cv_targets, dtype=float)
                    for weight, name in zip(weights, model_names):
                        ensemble_pred += weight * cv_predictions[name]
                    
                    cv_r2 = r2_score(cv_targets, ensemble_pred)
                    
                    if cv_r2 >= 0.503:
                        balanced_results_10.append({
                            'w_xgb': w_xgb,
                            'w_lgb': w_lgb,
                            'w_cat': w_cat,
                            'w_svr': w_svr,
                            'cv_r2': cv_r2,
                            'balance_score': np.std(weights)  # Lower = more balanced
                        })

print(f"✓ Found {len(balanced_results_10)} combinations with CV R² > 0.503 and min 10% weight")

# Strategy 2: Each model gets at least 15% weight
print("\n🔍 Strategy 2: Minimum 15% weight per model")
balanced_results_15 = []
weight_steps_fine = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

for w_xgb in weight_steps_fine:
    for w_lgb in weight_steps_fine:
        for w_cat in weight_steps_fine:
            w_svr = 1.0 - w_xgb - w_lgb - w_cat
            if 0.15 <= w_svr <= 0.40:
                weights = [w_xgb, w_lgb, w_cat, w_svr]
                
                ensemble_pred = np.zeros_like(cv_targets, dtype=float)
                for weight, name in zip(weights, model_names):
                    ensemble_pred += weight * cv_predictions[name]
                
                cv_r2 = r2_score(cv_targets, ensemble_pred)
                
                if cv_r2 >= 0.503:
                    balanced_results_15.append({
                        'w_xgb': w_xgb,
                        'w_lgb': w_lgb,
                        'w_cat': w_cat,
                        'w_svr': w_svr,
                        'cv_r2': cv_r2,
                        'balance_score': np.std(weights)
                    })

print(f"✓ Found {len(balanced_results_15)} combinations with CV R² > 0.503 and min 15% weight")

# Strategy 3: Equal weights (25% each)
print("\n🔍 Strategy 3: Equal weights (25% each)")
equal_weights = [0.25, 0.25, 0.25, 0.25]
ensemble_pred = np.zeros_like(cv_targets, dtype=float)
for weight, name in zip(equal_weights, model_names):
    ensemble_pred += weight * cv_predictions[name]
cv_r2_equal = r2_score(cv_targets, ensemble_pred)
print(f"   Equal weights CV R²: {cv_r2_equal:.4f}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("BALANCED ENSEMBLE RESULTS")
print("="*80)

if len(balanced_results_10) > 0:
    df_10 = pd.DataFrame(balanced_results_10).sort_values('cv_r2', ascending=False)
    
    print(f"\n📊 TOP 10 BALANCED ENSEMBLES (Min 10% weight):")
    print(df_10.head(10).to_string(index=False))
    
    best_10 = df_10.iloc[0]
    print(f"\n🏆 BEST BALANCED ENSEMBLE (Min 10%):")
    print(f"   XGBoost:  {best_10['w_xgb']:.1f}")
    print(f"   LightGBM: {best_10['w_lgb']:.1f}")
    print(f"   CatBoost: {best_10['w_cat']:.1f}")
    print(f"   SVR:      {best_10['w_svr']:.1f}")
    print(f"   CV R²: {best_10['cv_r2']:.4f}")
    print(f"   Balance Score: {best_10['balance_score']:.4f} (lower = more balanced)")
    
    # Find most balanced (lowest std)
    most_balanced_10 = df_10.sort_values('balance_score').iloc[0]
    print(f"\n🎯 MOST BALANCED ENSEMBLE (Min 10%):")
    print(f"   XGBoost:  {most_balanced_10['w_xgb']:.1f}")
    print(f"   LightGBM: {most_balanced_10['w_lgb']:.1f}")
    print(f"   CatBoost: {most_balanced_10['w_cat']:.1f}")
    print(f"   SVR:      {most_balanced_10['w_svr']:.1f}")
    print(f"   CV R²: {most_balanced_10['cv_r2']:.4f}")
    print(f"   Balance Score: {most_balanced_10['balance_score']:.4f}")
else:
    print("\n❌ No combinations found with min 10% weight and CV R² > 0.503")
    best_10 = None
    most_balanced_10 = None

if len(balanced_results_15) > 0:
    df_15 = pd.DataFrame(balanced_results_15).sort_values('cv_r2', ascending=False)
    
    print(f"\n📊 TOP 10 BALANCED ENSEMBLES (Min 15% weight):")
    print(df_15.head(10).to_string(index=False))
    
    best_15 = df_15.iloc[0]
    print(f"\n🏆 BEST BALANCED ENSEMBLE (Min 15%):")
    print(f"   XGBoost:  {best_15['w_xgb']:.2f}")
    print(f"   LightGBM: {best_15['w_lgb']:.2f}")
    print(f"   CatBoost: {best_15['w_cat']:.2f}")
    print(f"   SVR:      {best_15['w_svr']:.2f}")
    print(f"   CV R²: {best_15['cv_r2']:.4f}")
    print(f"   Balance Score: {best_15['balance_score']:.4f}")
    
    most_balanced_15 = df_15.sort_values('balance_score').iloc[0]
    print(f"\n🎯 MOST BALANCED ENSEMBLE (Min 15%):")
    print(f"   XGBoost:  {most_balanced_15['w_xgb']:.2f}")
    print(f"   LightGBM: {most_balanced_15['w_lgb']:.2f}")
    print(f"   CatBoost: {most_balanced_15['w_cat']:.2f}")
    print(f"   SVR:      {most_balanced_15['w_svr']:.2f}")
    print(f"   CV R²: {most_balanced_15['cv_r2']:.4f}")
    print(f"   Balance Score: {most_balanced_15['balance_score']:.4f}")
else:
    print("\n❌ No combinations found with min 15% weight and CV R² > 0.503")
    best_15 = None
    most_balanced_15 = None

print(f"\n⚖️  EQUAL WEIGHTS (25% each):")
print(f"   CV R²: {cv_r2_equal:.4f}")

# ============================================================================
# SELECT BEST BALANCED AND GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

output_dir = Path(__file__).parent / 'results_balanced'
os.makedirs(output_dir, exist_ok=True)

submissions_created = []

# Function to create submission
def create_submission(weights, name_suffix, cv_r2_val):
    # Train predictions
    train_pred = np.zeros(len(y_clean))
    for weight, name in zip(weights, model_names):
        train_pred += weight * models[name].predict(X_train_selected)
    
    train_r2 = r2_score(y_clean, train_pred)
    overfit = train_r2 - cv_r2_val
    
    # Test predictions
    test_pred = np.zeros(len(X_test_selected))
    for weight, name in zip(weights, model_names):
        test_pred += weight * models[name].predict(X_test_selected)
    
    test_pred_rounded = np.round(test_pred).astype(int)
    
    # Save submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_file = output_dir / f'submission_{name_suffix}_{timestamp}.csv'
    
    submission_df = pd.DataFrame({
        'id': range(len(test_pred_rounded)),
        'y': test_pred_rounded
    })
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\n✅ {name_suffix}:")
    print(f"   Weights: XGB={weights[0]:.2f}, LGB={weights[1]:.2f}, CAT={weights[2]:.2f}, SVR={weights[3]:.2f}")
    print(f"   CV R²: {cv_r2_val:.4f}")
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Overfit: {overfit:.4f}")
    print(f"   File: {submission_file}")
    
    return submission_file

# Create submissions for best configurations
if best_10 is not None:
    weights_best_10 = [best_10['w_xgb'], best_10['w_lgb'], best_10['w_cat'], best_10['w_svr']]
    file1 = create_submission(weights_best_10, 'best_min10pct', best_10['cv_r2'])
    submissions_created.append(('Best (Min 10%)', file1, best_10['cv_r2']))

if most_balanced_10 is not None:
    weights_bal_10 = [most_balanced_10['w_xgb'], most_balanced_10['w_lgb'], most_balanced_10['w_cat'], most_balanced_10['w_svr']]
    file2 = create_submission(weights_bal_10, 'balanced_min10pct', most_balanced_10['cv_r2'])
    submissions_created.append(('Most Balanced (Min 10%)', file2, most_balanced_10['cv_r2']))

if best_15 is not None:
    weights_best_15 = [best_15['w_xgb'], best_15['w_lgb'], best_15['w_cat'], best_15['w_svr']]
    file3 = create_submission(weights_best_15, 'best_min15pct', best_15['cv_r2'])
    submissions_created.append(('Best (Min 15%)', file3, best_15['cv_r2']))

file4 = create_submission(equal_weights, 'equal_25pct', cv_r2_equal)
submissions_created.append(('Equal Weights (25%)', file4, cv_r2_equal))

# ============================================================================
# VISUALIZATION
# ============================================================================
if best_10 is not None:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Use best min 10% for visualization
    best_weights_viz = [best_10['w_xgb'], best_10['w_lgb'], best_10['w_cat'], best_10['w_svr']]
    
    # Get train predictions for best ensemble
    train_pred_best = np.zeros(len(y_clean))
    for weight, name in zip(best_weights_viz, model_names):
        train_pred_best += weight * models[name].predict(X_train_selected)
    
    train_r2_best = r2_score(y_clean, train_pred_best)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: True vs Predicted (Training)
    ax1 = axes[0, 0]
    ax1.scatter(y_clean, train_pred_best, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
    min_val = min(y_clean.min(), train_pred_best.min())
    max_val = max(y_clean.max(), train_pred_best.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Age (years)', fontsize=11)
    ax1.set_ylabel('Predicted Age (years)', fontsize=11)
    ax1.set_title(f'Training: True vs Predicted\nR²={train_r2_best:.4f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual vs Ensemble CV R²
    ax2 = axes[0, 1]
    all_names = list(model_cv_scores.keys()) + ['Best\nBalanced']
    all_scores = list(model_cv_scores.values()) + [best_10['cv_r2']]
    colors = ['steelblue', 'darkgreen', 'coral', 'purple', 'gold']
    bars = ax2.bar(all_names, all_scores, color=colors, alpha=0.7)
    ax2.set_ylabel('CV R²', fontsize=11)
    ax2.set_title('Individual Models vs Balanced Ensemble', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(all_scores) - 0.02, max(all_scores) + 0.02])
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Ensemble weights (Best min 10%)
    ax3 = axes[0, 2]
    ax3.pie(best_weights_viz, labels=model_names, autopct='%1.1f%%', startangle=90, colors=colors[:4])
    ax3.set_title('Best Balanced Weights (Min 10%)', fontsize=12, fontweight='bold')
    
    # Plot 4: Residuals
    ax4 = axes[1, 0]
    residuals = y_clean - train_pred_best
    ax4.scatter(train_pred_best, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Predicted Age (years)', fontsize=11)
    ax4.set_ylabel('Residuals (years)', fontsize=11)
    ax4.set_title('Residuals Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Balance Score Distribution
    ax5 = axes[1, 1]
    if len(balanced_results_10) > 0:
        balance_scores = [r['balance_score'] for r in balanced_results_10]
        cv_r2_vals = [r['cv_r2'] for r in balanced_results_10]
        scatter = ax5.scatter(balance_scores, cv_r2_vals, c=cv_r2_vals, 
                             cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax5.set_xlabel('Balance Score (lower = more balanced)', fontsize=11)
        ax5.set_ylabel('CV R²', fontsize=11)
        ax5.set_title('Balance vs Performance Trade-off', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='CV R²')
    
    # Plot 6: Weight comparison of top 3
    ax6 = axes[1, 2]
    if len(balanced_results_10) >= 3:
        top_3 = df_10.head(3)
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (idx, row) in enumerate(top_3.iterrows()):
            weights = [row['w_xgb'], row['w_lgb'], row['w_cat'], row['w_svr']]
            ax6.bar(x + i*width, weights, width, label=f"#{i+1}: R²={row['cv_r2']:.4f}", alpha=0.7)
        
        ax6.set_xlabel('Model', fontsize=11)
        ax6.set_ylabel('Weight', fontsize=11)
        ax6.set_title('Top 3 Weight Configurations', fontsize=12, fontweight='bold')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(model_names)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = output_dir / f'balanced_ensemble_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {plot_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n📁 Created {len(submissions_created)} submission files:")
for name, file, cv_r2 in submissions_created:
    print(f"   • {name}: CV R²={cv_r2:.4f}")
    print(f"     {file}")

print("\n" + "="*80)
