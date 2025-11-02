"""
Kernel PCA analysis to find best kernel and minimal components.
Tests different kernels with the tuned models.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
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

print("="*70)
print("KERNEL PCA OPTIMIZATION")
print("Finding best kernel and minimal components for tuned models")
print("="*70)

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

# Load tuned parameters
with open('output_tuning/best_params.json', 'r') as f:
    tuned_params = json.load(f)

print("\n🔬 Testing Kernel PCA configurations...")
print("-"*70)

# Test configurations
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
n_components_list = [5,10,15,20]

results = []

for kernel in kernels:
    print(f"\n[{kernel.upper()}] Testing kernel...")
    
    for n_comp in n_components_list:
        try:
            # Apply Kernel PCA
            if kernel == 'linear':
                kpca = KernelPCA(n_components=n_comp, kernel=kernel, random_state=RANDOM_SEED)
            elif kernel == 'poly':
                kpca = KernelPCA(n_components=n_comp, kernel=kernel, degree=2, random_state=RANDOM_SEED)
            elif kernel == 'rbf':
                kpca = KernelPCA(n_components=n_comp, kernel=kernel, gamma=0.01, random_state=RANDOM_SEED)
            elif kernel == 'sigmoid':
                kpca = KernelPCA(n_components=n_comp, kernel=kernel, gamma=0.01, random_state=RANDOM_SEED)
            else:  # cosine
                kpca = KernelPCA(n_components=n_comp, kernel=kernel, random_state=RANDOM_SEED)
            
            X_kpca = kpca.fit_transform(X_scaled)
            X_kpca_df = pd.DataFrame(X_kpca)
            
            # Train XGBoost with tuned params (fastest model)
            import xgboost as xgb
            xgb_params = tuned_params['XGBoost'].copy()
            xgb_params.update({
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbosity': 0
            })
            
            model = xgb.XGBRegressor(**xgb_params)
            
            # Quick 3-fold CV
            kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(model, X_kpca_df, y_clean, cv=kfold, scoring='r2', n_jobs=-1)
            cv_r2 = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Train score
            model.fit(X_kpca_df, y_clean)
            train_pred = model.predict(X_kpca_df)
            train_r2 = r2_score(y_clean, train_pred)
            overfit = train_r2 - cv_r2
            
            results.append({
                'kernel': kernel,
                'n_components': n_comp,
                'cv_r2': cv_r2,
                'cv_std': cv_std,
                'train_r2': train_r2,
                'overfit': overfit
            })
            
            print(f"  n={n_comp:3d}: CV R²={cv_r2:.4f} (±{cv_std:.4f}), Train R²={train_r2:.4f}, Gap={overfit:.4f}")
            
        except Exception as e:
            print(f"  n={n_comp:3d}: Failed - {str(e)[:50]}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Find best configurations
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Best overall CV R²
best_overall = results_df.loc[results_df['cv_r2'].idxmax()]
print(f"\n🏆 Best Overall CV R²:")
print(f"   Kernel: {best_overall['kernel']}")
print(f"   Components: {best_overall['n_components']:.0f}")
print(f"   CV R²: {best_overall['cv_r2']:.4f} (±{best_overall['cv_std']:.4f})")
print(f"   Overfit Gap: {best_overall['overfit']:.4f}")

# Best with minimal components (<100)
results_small = results_df[results_df['n_components'] <= 100]
if len(results_small) > 0:
    best_small = results_small.loc[results_small['cv_r2'].idxmax()]
    print(f"\n🎯 Best with ≤100 Components:")
    print(f"   Kernel: {best_small['kernel']}")
    print(f"   Components: {best_small['n_components']:.0f}")
    print(f"   CV R²: {best_small['cv_r2']:.4f} (±{best_small['cv_std']:.4f})")
    print(f"   Overfit Gap: {best_small['overfit']:.4f}")

# Best per kernel
print(f"\n📊 Best Configuration per Kernel:")
print("-"*70)
for kernel in kernels:
    kernel_results = results_df[results_df['kernel'] == kernel]
    if len(kernel_results) > 0:
        best_kernel = kernel_results.loc[kernel_results['cv_r2'].idxmax()]
        print(f"  {kernel:8s}: n={best_kernel['n_components']:3.0f}, CV R²={best_kernel['cv_r2']:.4f}, Gap={best_kernel['overfit']:.4f}")

# Compare to standard PCA
print(f"\n📈 Comparison to Linear PCA (baseline):")
linear_results = results_df[results_df['kernel'] == 'linear'].sort_values('n_components')
if len(linear_results) > 0:
    print("  Components | CV R² | Overfit")
    print("  -----------|-------|--------")
    for _, row in linear_results.iterrows():
        print(f"  {row['n_components']:6.0f}     | {row['cv_r2']:.4f} | {row['overfit']:.4f}")

# Save results
results_df.to_csv('kernel_pca_results.csv', index=False)
print(f"\n✓ Saved: kernel_pca_results.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CV R² by kernel and components
ax1 = axes[0, 0]
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    if len(kernel_data) > 0:
        ax1.plot(kernel_data['n_components'], kernel_data['cv_r2'], 'o-', label=kernel, linewidth=2, markersize=6)
ax1.set_xlabel('Number of Components', fontsize=11)
ax1.set_ylabel('CV R²', fontsize=11)
ax1.set_title('CV R² vs Number of Components', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Overfitting gap
ax2 = axes[0, 1]
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    if len(kernel_data) > 0:
        ax2.plot(kernel_data['n_components'], kernel_data['overfit'], 'o-', label=kernel, linewidth=2, markersize=6)
ax2.set_xlabel('Number of Components', fontsize=11)
ax2.set_ylabel('Overfit Gap (Train R² - CV R²)', fontsize=11)
ax2.set_title('Overfitting vs Number of Components', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, label='Target gap')

# Plot 3: Best kernel per n_components
ax3 = axes[1, 0]
best_per_n = results_df.loc[results_df.groupby('n_components')['cv_r2'].idxmax()]
bars = ax3.bar(range(len(best_per_n)), best_per_n['cv_r2'], color='steelblue', alpha=0.7)
ax3.set_xticks(range(len(best_per_n)))
ax3.set_xticklabels([f"{int(n)}\n{k[:3]}" for n, k in zip(best_per_n['n_components'], best_per_n['kernel'])], fontsize=9)
ax3.set_xlabel('Components (Kernel)', fontsize=11)
ax3.set_ylabel('Best CV R²', fontsize=11)
ax3.set_title('Best Kernel per Component Count', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Efficiency (CV R² per component)
ax4 = axes[1, 1]
results_df['efficiency'] = results_df['cv_r2'] / results_df['n_components']
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    if len(kernel_data) > 0:
        ax4.plot(kernel_data['n_components'], kernel_data['efficiency'], 'o-', label=kernel, linewidth=2, markersize=6)
ax4.set_xlabel('Number of Components', fontsize=11)
ax4.set_ylabel('Efficiency (CV R² / n_components)', fontsize=11)
ax4.set_title('Efficiency: Performance per Component', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_pca_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: kernel_pca_analysis.png")

# Final recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Find sweet spot: good CV R² with low components
results_df['score'] = results_df['cv_r2'] - 0.002 * results_df['n_components']  # Penalty for complexity
best_tradeoff = results_df.loc[results_df['score'].idxmax()]

print(f"\n💡 Best Trade-off (performance vs complexity):")
print(f"   Kernel: {best_tradeoff['kernel']}")
print(f"   Components: {best_tradeoff['n_components']:.0f}")
print(f"   CV R²: {best_tradeoff['cv_r2']:.4f}")
print(f"   Overfit Gap: {best_tradeoff['overfit']:.4f}")
print(f"\n   vs SelectKBest k=100: CV R²=0.4995")
print(f"   Improvement: {(best_tradeoff['cv_r2'] - 0.4995)*100:+.2f}%")
print("="*70)
