"""
Test different K values for SelectKBest with tuned hyperparameters.
Ranges from 30 to 200 features to find optimal feature count.

Usage:
    python tune_k_features.py --cv_folds 5 --remove_outliers
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import DataLoader
from ensemble import WeightedEnsemble
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class KFeatureTuner:
    """Test different K values with tuned hyperparameters."""
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader(config['data_dir'])
        self.results = []
        
        # Create output directory
        self.output_dir = Path('output_k_tuning')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load tuned parameters
        params_file = Path('output_tuning/best_params.json')
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.tuned_params = json.load(f)
            print(f"✓ Loaded tuned parameters from {params_file}")
        else:
            raise FileNotFoundError(f"Tuned parameters not found at {params_file}")
        
        self._print_header()
    
    def _print_header(self):
        """Print pipeline header."""
        print("\n" + "="*80)
        print("K-FEATURES TUNING WITH OPTIMIZED HYPERPARAMETERS")
        print("Testing SelectKBest K from 30 to 200 with tuned models")
        print("="*80)
        print("Configuration:")
        for key, value in self.config.items():
            print(f"  {key:25s}: {value}")
        print("="*80 + "\n")
    
    def load_and_preprocess_base(self):
        """Load data and do initial preprocessing (before SelectKBest)."""
        print("📂 Loading and Initial Preprocessing")
        print("-" * 80)
        
        # Load data
        self.X_train, self.y_train = self.loader.load_train_data()
        self.X_test, self.test_ids = self.loader.load_test_data()
        
        print(f"✓ Loaded: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        
        # 1. KNN Imputation
        print("🔧 KNN Imputation (n_neighbors=5)...")
        imputer = KNNImputer(n_neighbors=5)
        self.X_train = pd.DataFrame(
            imputer.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            imputer.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # 2. Remove outliers
        if self.config.get('remove_outliers', True):
            print("🔍 Removing outliers...")
            iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
            outlier_labels = iso_forest.fit_predict(self.X_train)
            inlier_mask = outlier_labels == 1
            n_outliers = np.sum(~inlier_mask)
            print(f"   Removed {n_outliers} outliers ({100*n_outliers/len(self.X_train):.1f}%)")
            
            self.X_train = self.X_train[inlier_mask].copy()
            self.y_train = self.y_train[inlier_mask].copy()
        
        # 3. Variance threshold
        print("🔧 Removing low-variance features...")
        self.var_threshold = VarianceThreshold(threshold=0.01)
        self.X_train = pd.DataFrame(
            self.var_threshold.fit_transform(self.X_train),
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.var_threshold.transform(self.X_test),
            index=self.X_test.index
        )
        
        # 4. Remove highly correlated features
        print("🔧 Removing highly correlated features...")
        corr_matrix = self.X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        self.X_train = self.X_train.drop(columns=to_drop)
        self.X_test = self.X_test.drop(columns=to_drop)
        
        self.y_train_processed = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
        
        print(f"✓ Base preprocessing complete: {self.X_train.shape[1]} features available\n")
    
    def test_k_value(self, k: int):
        """Test a specific K value."""
        print(f"\n{'='*80}")
        print(f"TESTING K = {k}")
        print(f"{'='*80}")
        
        # SelectKBest
        print(f"🎯 Selecting top {k} features...")
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        X_train_processed = pd.DataFrame(X_train_scaled, columns=[f"f{i}" for i in range(k)])
        
        cv_folds = self.config.get('cv_folds', 5)
        models = {}
        cv_scores = {}
        
        # Train XGBoost
        print(f"[1/3] XGBoost...")
        try:
            import xgboost as xgb
            xgb_params = self.tuned_params['XGBoost'].copy()
            xgb_params.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': 0})
            
            model = xgb.XGBRegressor(**xgb_params)
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for train_idx, val_idx in kfold.split(X_train_processed):
                X_train_fold = X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = xgb.XGBRegressor(**xgb_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            model.fit(X_train_processed, self.y_train_processed)
            train_pred = model.predict(X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            models['XGBoost'] = model
            cv_scores['XGBoost'] = cv_r2_mean
            
            print(f"   CV R²: {cv_r2_mean:.4f}, Train R²: {train_r2:.4f}, Gap: {train_r2 - cv_r2_mean:.4f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Train LightGBM
        print(f"[2/3] LightGBM...")
        try:
            import lightgbm as lgb
            lgb_params = self.tuned_params['LightGBM'].copy()
            lgb_params.update({'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbose': -1})
            
            model = lgb.LGBMRegressor(**lgb_params)
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for train_idx, val_idx in kfold.split(X_train_processed):
                X_train_fold = X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = lgb.LGBMRegressor(**lgb_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            model.fit(X_train_processed, self.y_train_processed)
            train_pred = model.predict(X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            models['LightGBM'] = model
            cv_scores['LightGBM'] = cv_r2_mean
            
            print(f"   CV R²: {cv_r2_mean:.4f}, Train R²: {train_r2:.4f}, Gap: {train_r2 - cv_r2_mean:.4f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Train CatBoost
        print(f"[3/3] CatBoost...")
        try:
            import catboost as cb
            cat_params = self.tuned_params['CatBoost'].copy()
            cat_params.update({'random_seed': RANDOM_SEED, 'verbose': False, 'thread_count': -1})
            
            model = cb.CatBoostRegressor(**cat_params)
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for train_idx, val_idx in kfold.split(X_train_processed):
                X_train_fold = X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = cb.CatBoostRegressor(**cat_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            model.fit(X_train_processed, self.y_train_processed)
            train_pred = model.predict(X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            models['CatBoost'] = model
            cv_scores['CatBoost'] = cv_r2_mean
            
            print(f"   CV R²: {cv_r2_mean:.4f}, Train R²: {train_r2:.4f}, Gap: {train_r2 - cv_r2_mean:.4f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Create ensemble
        if len(models) > 0:
            ensemble = WeightedEnsemble(models=models, cv_scores=cv_scores)
            
            total_weight = sum(ensemble.weights.values())
            weighted_cv = sum(cv_scores[name] * ensemble.weights[name] for name in models.keys()) / total_weight
            
            print(f"\n✓ Ensemble CV R²: {weighted_cv:.4f}")
            
            # Store results
            result = {
                'K': k,
                'Ensemble_CV_R2': weighted_cv,
                'XGBoost_CV_R2': cv_scores.get('XGBoost', np.nan),
                'LightGBM_CV_R2': cv_scores.get('LightGBM', np.nan),
                'CatBoost_CV_R2': cv_scores.get('CatBoost', np.nan),
            }
            
            for name in models.keys():
                result[f'{name}_Weight'] = ensemble.weights[name]
            
            self.results.append(result)
    
    def run_all_k_values(self):
        """Test all K values."""
        print("\n🔬 TESTING MULTIPLE K VALUES")
        print("="*80)
        
        k_values = [30, 50, 75, 100, 125, 150, 175, 200]
        
        for i, k in enumerate(k_values, 1):
            print(f"\n[{i}/{len(k_values)}] Testing K={k}...")
            try:
                self.test_k_value(k)
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                continue
    
    def create_plots(self):
        """Create visualization plots."""
        print("\n📊 Creating Plots")
        print("-" * 80)
        
        results_df = pd.DataFrame(self.results)
        
        # Plot 1: CV R² vs K
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Ensemble CV R²
        ax1.plot(results_df['K'], results_df['Ensemble_CV_R2'], 
                marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Ensemble')
        ax1.plot(results_df['K'], results_df['XGBoost_CV_R2'], 
                marker='s', linewidth=1.5, markersize=6, alpha=0.7, label='XGBoost')
        ax1.plot(results_df['K'], results_df['LightGBM_CV_R2'], 
                marker='^', linewidth=1.5, markersize=6, alpha=0.7, label='LightGBM')
        ax1.plot(results_df['K'], results_df['CatBoost_CV_R2'], 
                marker='d', linewidth=1.5, markersize=6, alpha=0.7, label='CatBoost')
        
        # Mark best
        best_idx = results_df['Ensemble_CV_R2'].idxmax()
        best_k = results_df.loc[best_idx, 'K']
        best_r2 = results_df.loc[best_idx, 'Ensemble_CV_R2']
        ax1.axvline(best_k, color='red', linestyle='--', alpha=0.5)
        ax1.plot(best_k, best_r2, 'r*', markersize=20, label=f'Best: K={int(best_k)}')
        
        ax1.set_xlabel('Number of Features (K)', fontsize=12)
        ax1.set_ylabel('CV R² Score', fontsize=12)
        ax1.set_title('CV R² vs Number of Features (Tuned Models)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Model comparison at best K
        best_row = results_df.loc[best_idx]
        models = ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']
        scores = [
            best_row['XGBoost_CV_R2'],
            best_row['LightGBM_CV_R2'],
            best_row['CatBoost_CV_R2'],
            best_row['Ensemble_CV_R2']
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax2.barh(models, scores, color=colors)
        ax2.set_xlabel('CV R² Score', fontsize=12)
        ax2.set_title(f'Model Performance at Best K={int(best_k)}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(score + 0.005, i, f'{score:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'k_tuning_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    def save_results(self):
        """Save results."""
        print("\n💾 Saving Results")
        print("-" * 80)
        
        results_df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_file = self.output_dir / 'k_tuning_results.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file}")
        
        # Print summary
        best_idx = results_df['Ensemble_CV_R2'].idxmax()
        best_k = results_df.loc[best_idx, 'K']
        best_r2 = results_df.loc[best_idx, 'Ensemble_CV_R2']
        
        print("\n" + "="*80)
        print("K-TUNING SUMMARY")
        print("="*80)
        print(results_df[['K', 'Ensemble_CV_R2', 'XGBoost_CV_R2', 'LightGBM_CV_R2', 'CatBoost_CV_R2']].to_string(index=False))
        print("="*80)
        print(f"\n🏆 BEST K: {int(best_k)} with Ensemble CV R² = {best_r2:.4f}")
        print("="*80 + "\n")
    
    def run(self):
        """Run full K-tuning pipeline."""
        try:
            start_time = datetime.now()
            
            self.load_and_preprocess_base()
            self.run_all_k_values()
            self.create_plots()
            self.save_results()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n✓ K-Tuning Complete! (Elapsed: {elapsed:.1f}s)\n")
            
        except Exception as e:
            print(f"\n✗ K-Tuning Failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Tune K features with optimized hyperparameters')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': 'eth-aml-2025-project-1',
        'cv_folds': args.cv_folds,
        'remove_outliers': args.remove_outliers
    }
    
    tuner = KFeatureTuner(config)
    tuner.run()


if __name__ == '__main__':
    main()
