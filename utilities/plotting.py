"""
Visualization utilities for Brain-Age Prediction.
Creates plots for model evaluation and feature analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Plotter:
    """Visualization utilities for model analysis."""
    
    def __init__(self, save_dir: str = "plots"):
        """
        Args:
            save_dir: Directory to save plots
        """
        from pathlib import Path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Predictions vs Actual Age",
                                   save_name: Optional[str] = None):
        """
        Scatter plot of predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Add metrics to plot
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Age (years)', fontsize=12)
        ax.set_ylabel('Predicted Age (years)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_name: Optional[str] = None):
        """
        Plot residual distribution and residuals vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_name: Filename to save (optional)
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Age (years)', fontsize=12)
        axes[0].set_ylabel('Residuals (years)', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals (years)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        textstr = f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[1].text(0.65, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                             save_name: Optional[str] = None):
        """
        Compare multiple models using bar plots.
        
        Args:
            results_df: DataFrame with model results (from ModelTrainer)
            save_name: Filename to save (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R² scores
        models = results_df['Model'].values
        cv_r2 = results_df['CV_R2_Mean'].values
        cv_r2_std = results_df['CV_R2_Std'].values
        train_r2 = results_df['Train_R2'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, cv_r2, width, yerr=cv_r2_std, label='CV R²', 
                   capsize=5, alpha=0.8, edgecolor='black')
        axes[0].bar(x + width/2, train_r2, width, label='Train R²', 
                   alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('Model Comparison - R² Scores', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(y=0.5, color='r', linestyle='--', lw=1, label='Min Required (0.5)')
        
        # RMSE scores
        cv_rmse = results_df['CV_RMSE_Mean'].values
        train_rmse = results_df['Train_RMSE'].values
        
        axes[1].bar(x - width/2, cv_rmse, width, label='CV RMSE', 
                   alpha=0.8, edgecolor='black')
        axes[1].bar(x + width/2, train_rmse, width, label='Train RMSE', 
                   alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylabel('RMSE (years)', fontsize=12)
        axes[1].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: np.ndarray,
                               top_k: int = 20,
                               save_name: Optional[str] = None):
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            top_k: Number of top features to show
            save_name: Filename to save (optional)
        """
        # Sort by importance
        indices = np.argsort(importances)[-top_k:]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        ax.barh(range(top_k), importances[indices], align='center', 
                alpha=0.8, edgecolor='black')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_heatmap(self, X: pd.DataFrame, 
                                 top_k: int = 30,
                                 save_name: Optional[str] = None):
        """
        Plot correlation heatmap for top features.
        
        Args:
            X: Feature DataFrame
            top_k: Number of features to include
            save_name: Filename to save (optional)
        """
        # Select subset if needed
        if X.shape[1] > top_k:
            # Use features with highest variance
            variances = X.var()
            top_features = variances.nlargest(top_k).index
            X_subset = X[top_features]
        else:
            X_subset = X
        
        # Calculate correlation matrix
        corr = X_subset.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_learning_curves(self, train_scores: List[float], 
                            val_scores: List[float],
                            train_sizes: List[int],
                            save_name: Optional[str] = None):
        """
        Plot learning curves showing train/validation scores vs training size.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.save_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        plt.close()


def create_report_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                       results_df: pd.DataFrame,
                       feature_names: Optional[List[str]] = None,
                       feature_importances: Optional[np.ndarray] = None):
    """
    Generate all standard report plots.
    
    Args:
        y_true: True values
        y_pred: Predictions
        results_df: Model comparison results
        feature_names: Optional feature names
        feature_importances: Optional feature importance scores
    """
    plotter = Plotter()
    
    print("\n📊 Generating plots...")
    
    # Predictions vs Actual
    plotter.plot_predictions_vs_actual(y_true, y_pred, 
                                      save_name='predictions_vs_actual.png')
    
    # Residuals
    plotter.plot_residuals(y_true, y_pred, 
                          save_name='residuals.png')
    
    # Model comparison
    plotter.plot_model_comparison(results_df, 
                                 save_name='model_comparison.png')
    
    # Feature importance (if available)
    if feature_names is not None and feature_importances is not None:
        plotter.plot_feature_importance(feature_names, feature_importances,
                                       top_k=20,
                                       save_name='feature_importance.png')
    
    print("✓ All plots generated and saved to 'plots/' directory\n")


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting utilities...")
    
    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.uniform(20, 90, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)
    
    plotter = Plotter()
    plotter.plot_predictions_vs_actual(y_true, y_pred)
    plotter.plot_residuals(y_true, y_pred)
    
    print("✓ Test plots generated")
