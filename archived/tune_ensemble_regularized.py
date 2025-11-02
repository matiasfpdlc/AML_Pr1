"""
Hyperparameter Tuning with Focus on Reducing Overfitting.
Uses Optuna to find optimal regularization parameters.

Usage:
    python tune_ensemble_regularized.py --n_trials 100 --cv_folds 5
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

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

from dataloader import DataLoader

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class RegularizedEnsembleTuner:
    """
    Tunes hyperparameters with focus on reducing overfitting.
    Optimizes for CV R² with penalty for train-CV gap.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader(config['data_dir'])
        self.best_params = {}
        self.study_results = []
        
        # Output directory
        self.output_dir = Path('output_tuning')
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("REGULARIZED HYPERPARAMETER TUNING")
        print("Objective: Minimize overfitting while maximizing CV R²")
        print("="*80)
        print(f"Trials: {config['n_trials']}")
        print(f"CV Folds: {config['cv_folds']}")
        print(f"Overfitting Penalty: {config['overfit_penalty']}")
        print("="*80 + "\n")
    
    def load_and_preprocess(self):
        """Load and preprocess data."""
        print("📂 Loading and Preprocessing Data")
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
        var_threshold = VarianceThreshold(threshold=0.01)
        self.X_train = pd.DataFrame(
            var_threshold.fit_transform(self.X_train),
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            var_threshold.transform(self.X_test),
            index=self.X_test.index
        )
        
        # 4. Remove highly correlated features
        print("🔧 Removing highly correlated features...")
        corr_matrix = self.X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        self.X_train = self.X_train.drop(columns=to_drop)
        self.X_test = self.X_test.drop(columns=to_drop)
        
        # 5. SelectKBest
        k = self.config.get('k_features', 100)
        print(f"🎯 SelectKBest (k={k})...")
        self.selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = self.selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.selector.transform(self.X_test)
        
        # 6. Scale
        print("📏 RobustScaler...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        self.X_train_processed = pd.DataFrame(X_train_scaled, columns=[f"f{i}" for i in range(k)])
        self.X_test_processed = pd.DataFrame(X_test_scaled, columns=[f"f{i}" for i in range(k)])
        self.y_train_processed = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
        
        print(f"✓ Final: {len(self.X_train_processed)} samples, {self.X_train_processed.shape[1]} features\n")
    
    def objective_xgboost(self, trial):
        """Objective function for XGBoost with overfitting penalty."""
        # Suggest hyperparameters with strong regularization bias
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 5),  # Shallow trees
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),  # High min samples
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
            'gamma': trial.suggest_float('gamma', 0.5, 3.0),  # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 3.0, 15.0),  # L1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 20.0),  # L2 regularization
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(**params)
            
            # Cross-validation
            kfold = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(
                model, self.X_train_processed, self.y_train_processed,
                cv=kfold, scoring='r2', n_jobs=-1
            )
            cv_r2 = np.mean(cv_scores)
            
            # Train score
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            # Calculate overfitting gap
            overfit_gap = train_r2 - cv_r2
            
            # Objective: Maximize CV R² with penalty for overfitting
            # Higher penalty means we prioritize reducing overfit over raw CV score
            penalty = self.config['overfit_penalty'] * max(0, overfit_gap - 0.15)  # Allow 15% gap
            objective_value = cv_r2 - penalty
            
            # Store metrics for analysis
            trial.set_user_attr('cv_r2', cv_r2)
            trial.set_user_attr('train_r2', train_r2)
            trial.set_user_attr('overfit_gap', overfit_gap)
            
            return objective_value
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1.0
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM with overfitting penalty."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),  # Fewer leaves
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),  # More samples per leaf
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 15.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.05, 0.5),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(**params)
            
            # Cross-validation
            kfold = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(
                model, self.X_train_processed, self.y_train_processed,
                cv=kfold, scoring='r2', n_jobs=-1
            )
            cv_r2 = np.mean(cv_scores)
            
            # Train score
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            # Calculate overfitting gap
            overfit_gap = train_r2 - cv_r2
            
            # Objective with penalty
            penalty = self.config['overfit_penalty'] * max(0, overfit_gap - 0.15)
            objective_value = cv_r2 - penalty
            
            trial.set_user_attr('cv_r2', cv_r2)
            trial.set_user_attr('train_r2', train_r2)
            trial.set_user_attr('overfit_gap', overfit_gap)
            
            return objective_value
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1.0
    
    def objective_catboost(self, trial):
        """Objective function for CatBoost with overfitting penalty."""
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 15.0),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'thread_count': -1
        }
        
        try:
            import catboost as cb
            model = cb.CatBoostRegressor(**params)
            
            # Cross-validation
            kfold = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(
                model, self.X_train_processed, self.y_train_processed,
                cv=kfold, scoring='r2', n_jobs=-1
            )
            cv_r2 = np.mean(cv_scores)
            
            # Train score
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            
            # Calculate overfitting gap
            overfit_gap = train_r2 - cv_r2
            
            # Objective with penalty
            penalty = self.config['overfit_penalty'] * max(0, overfit_gap - 0.15)
            objective_value = cv_r2 - penalty
            
            trial.set_user_attr('cv_r2', cv_r2)
            trial.set_user_attr('train_r2', train_r2)
            trial.set_user_attr('overfit_gap', overfit_gap)
            
            return objective_value
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1.0
    
    def tune_all_models(self):
        """Tune all models."""
        print("\n🎯 TUNING MODELS")
        print("="*80)
        
        models_to_tune = ['XGBoost', 'LightGBM', 'CatBoost']
        
        for model_name in models_to_tune:
            print(f"\n[{model_name}] Starting tuning...")
            print("-" * 80)
            
            # Select objective function
            if model_name == 'XGBoost':
                objective_func = self.objective_xgboost
            elif model_name == 'LightGBM':
                objective_func = self.objective_lightgbm
            else:
                objective_func = self.objective_catboost
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=RANDOM_SEED)
            )
            
            # Optimize
            study.optimize(
                objective_func,
                n_trials=self.config['n_trials'],
                show_progress_bar=True
            )
            
            # Best trial
            best_trial = study.best_trial
            
            print(f"\n✓ Best Trial #{best_trial.number}")
            print(f"   CV R²: {best_trial.user_attrs['cv_r2']:.4f}")
            print(f"   Train R²: {best_trial.user_attrs['train_r2']:.4f}")
            print(f"   Overfit Gap: {best_trial.user_attrs['overfit_gap']:.4f}")
            print(f"   Objective (penalized): {best_trial.value:.4f}")
            
            # Store results
            self.best_params[model_name] = best_trial.params
            self.study_results.append({
                'model': model_name,
                'cv_r2': best_trial.user_attrs['cv_r2'],
                'train_r2': best_trial.user_attrs['train_r2'],
                'overfit_gap': best_trial.user_attrs['overfit_gap'],
                'objective': best_trial.value,
                'params': best_trial.params
            })
            
            # Save study
            study_file = self.output_dir / f'study_{model_name.lower()}.pkl'
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            print(f"   Saved study: {study_file}")
    
    def save_results(self):
        """Save tuning results."""
        print("\n💾 Saving Results")
        print("-" * 80)
        
        # Save best parameters
        params_file = self.output_dir / 'best_params.json'
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"✓ Saved: {params_file}")
        
        # Save results DataFrame
        results_df = pd.DataFrame(self.study_results)
        results_csv = self.output_dir / 'tuning_results.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"✓ Saved: {results_csv}")
        
        # Print summary
        print("\n" + "="*80)
        print("TUNING SUMMARY")
        print("="*80)
        print(results_df[['model', 'cv_r2', 'train_r2', 'overfit_gap']].to_string(index=False))
        print("="*80)
    
    def run(self):
        """Run full tuning pipeline."""
        try:
            self.load_and_preprocess()
            self.tune_all_models()
            self.save_results()
            
            print("\n✓ Tuning Complete!")
            
        except Exception as e:
            print(f"\n✗ Tuning Failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Tune ensemble with regularization focus')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials per model')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--k_features', type=int, default=100, help='Number of features')
    parser.add_argument('--overfit_penalty', type=float, default=0.5, 
                       help='Penalty weight for overfitting (higher = more penalty)')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': 'eth-aml-2025-project-1',
        'n_trials': args.n_trials,
        'cv_folds': args.cv_folds,
        'k_features': args.k_features,
        'overfit_penalty': args.overfit_penalty,
        'remove_outliers': args.remove_outliers
    }
    
    tuner = RegularizedEnsembleTuner(config)
    tuner.run()


if __name__ == '__main__':
    main()
