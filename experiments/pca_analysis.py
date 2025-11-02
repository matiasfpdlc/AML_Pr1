"""
Quick PCA variance analysis to determine optimal number of components.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Add utilities to path and import
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from dataloader import DataLoader  # type: ignore

# Get project root directory
project_root = Path(__file__).parent.parent

# Load data
print("Loading data...")
data_path = project_root / 'eth-aml-2025-project-1'
loader = DataLoader(str(data_path))
X_train, y_train = loader.load_train_data()

print(f"Original shape: {X_train.shape}")
print(f"Missing values: {X_train.isnull().sum().sum()}")

# Impute missing values
print("\nImputing missing values with KNN...")
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)

# Standardize
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Run PCA
print("\nRunning PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find components for different variance thresholds
thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
print("\n" + "="*60)
print("PCA VARIANCE ANALYSIS")
print("="*60)
print(f"Total features: {X_train.shape[1]}")
print(f"\nComponents needed for variance capture:")
print("-"*60)

for threshold in thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    variance_captured = cumulative_variance[n_components-1]
    print(f"  {threshold*100:.0f}% variance: {n_components:4d} components ({variance_captured*100:.2f}%)")

print("-"*60)

# Print first 20 components
print(f"\nFirst 20 components variance:")
for i in range(min(20, len(pca.explained_variance_ratio_))):
    cum_var = cumulative_variance[i]
    ind_var = pca.explained_variance_ratio_[i]
    print(f"  PC{i+1:3d}: {ind_var*100:5.2f}%  (cumulative: {cum_var*100:5.2f}%)")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cumulative variance
ax1.plot(range(1, len(cumulative_variance)+1), cumulative_variance*100, 'b-', linewidth=2)
for threshold in thresholds:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    ax1.axhline(y=threshold*100, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=n_comp, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(n_comp+10, threshold*100-2, f'{n_comp} comp.', fontsize=9)

ax1.set_xlabel('Number of Components', fontsize=12)
ax1.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
ax1.set_title('PCA Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, min(300, len(cumulative_variance)))

# Plot 2: Individual variance (first 50 components)
n_show = min(50, len(pca.explained_variance_ratio_))
ax2.bar(range(1, n_show+1), pca.explained_variance_ratio_[:n_show]*100, alpha=0.7)
ax2.set_xlabel('Component', fontsize=12)
ax2.set_ylabel('Explained Variance (%)', fontsize=12)
ax2.set_title('Individual Component Variance (First 50)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot: pca_variance_analysis.png")

# Summary recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"For 90% variance: Use k={n_90} features")
print(f"For 95% variance: Use k={n_95} features")
print(f"\nCurrent SelectKBest k=100 captures: {cumulative_variance[99]*100:.2f}% variance")
print("="*60)
