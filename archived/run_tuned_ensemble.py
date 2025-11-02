"""
Run ensemble with tuned hyperparameters from Optuna optimization.
Uses the best regularized parameters to reduce overfitting.

Usage:
    python run_tuned_ensemble.py --cv_folds 5 --remove_outliers
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

from dataloader import DataLoader
from ensemble import WeightedEnsemble
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, StackingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class TunedEnsemblePipeline:
    """Pipeline using tuned hyperparameters."""
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader(config['data_dir'])
        self.models = {}
        self.cv_scores = {}
        self.ensemble = None
        
        # Create output directory
        self.output_dir = Path('output_tuned')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
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
        print("TUNED ENSEMBLE PIPELINE")
        print("Using Optuna-optimized hyperparameters for reduced overfitting")
        print("="*80)
        print("Configuration:")
        for key, value in self.config.items():
            print(f"  {key:25s}: {value}")
        print("="*80 + "\n")
    
    def load_and_preprocess(self):
        """Load and preprocess data."""
        print("\n📂 STEP 1: Loading and Preprocessing")
        print("-" * 80)
        
        # Load data
        self.X_train, self.y_train = self.loader.load_train_data()
        self.X_test, self.test_ids = self.loader.load_test_data()
        
        print(f"✓ Loaded training data: X_train shape {self.X_train.shape}, y_train shape {self.y_train.shape}")
        print(f"✓ Loaded test data: X_test shape {self.X_test.shape}")
        
        # 1. KNN Imputation
        print("\n🔧 Imputing missing values with KNN (n_neighbors=5)...")
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
        print(f"   ✓ KNN imputation complete")
        
        # 2. Remove outliers
        if self.config.get('remove_outliers', True):
            print("🔍 Detecting and removing outliers...")
            iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
            outlier_labels = iso_forest.fit_predict(self.X_train)
            inlier_mask = outlier_labels == 1
            
            n_outliers = np.sum(~inlier_mask)
            print(f"   Detected {n_outliers} outliers ({100*n_outliers/len(self.X_train):.2f}%)")
            
            self.X_train = self.X_train[inlier_mask].copy()
            self.y_train = self.y_train[inlier_mask].copy()
        
        # 3. Remove low variance features
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
        
        # 5. SelectKBest feature selection
        k = self.config.get('k_features', 100)
        print(f"🎯 Selecting top {k} features with SelectKBest...")
        self.selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = self.selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.selector.transform(self.X_test)
        
        # 6. Scale features
        print("📏 Scaling features with RobustScaler...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Convert to DataFrames
        feature_names = [f"f{i}" for i in range(k)]
        self.X_train_processed = pd.DataFrame(X_train_scaled, columns=feature_names)
        self.X_test_processed = pd.DataFrame(X_test_scaled, columns=feature_names)
        self.y_train_processed = self.y_train if isinstance(self.y_train, np.ndarray) else self.y_train.values
        
        print(f"\n✓ Preprocessing complete")
        print(f"   Samples: {len(self.X_train_processed)} train, {len(self.X_test_processed)} test")
        print(f"   Features: {self.X_train_processed.shape[1]}\n")
    
    def train_tuned_models(self):
        """Train models with tuned hyperparameters."""
        print("\n🎓 STEP 2: Training Models with Tuned Parameters")
        print("-" * 80)
        
        cv_folds = self.config.get('cv_folds', 5)
        print(f"\nTraining 3 models with {cv_folds}-fold CV:\n")
        
        results_list = []
        
        # 1. XGBoost
        print("[1/3] Training XGBoost (tuned)...")
        try:
            import xgboost as xgb
            xgb_params = self.tuned_params['XGBoost'].copy()
            xgb_params['random_state'] = RANDOM_SEED
            xgb_params['n_jobs'] = -1
            xgb_params['verbosity'] = 0
            
            model = xgb.XGBRegressor(**xgb_params)
            
            # Cross-validation
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_processed)):
                X_train_fold = self.X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = self.X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = xgb.XGBRegressor(**xgb_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)
            
            # Train on full data
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train_processed, train_pred))
            
            self.models['XGBoost'] = model
            self.cv_scores['XGBoost'] = cv_r2_mean
            
            results_list.append({
                'Model': 'XGBoost',
                'CV_R2_Mean': cv_r2_mean,
                'CV_R2_Std': cv_r2_std,
                'Train_R2': train_r2,
                'Train_RMSE': train_rmse,
                'Overfit': train_r2 - cv_r2_mean
            })
            
            print(f"   ✓ CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
            print(f"   ✓ Train R²: {train_r2:.4f}, Overfit Gap: {train_r2 - cv_r2_mean:.4f}\n")
            
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
        
        # 2. LightGBM
        print("[2/3] Training LightGBM (tuned)...")
        try:
            import lightgbm as lgb
            lgb_params = self.tuned_params['LightGBM'].copy()
            lgb_params['random_state'] = RANDOM_SEED
            lgb_params['n_jobs'] = -1
            lgb_params['verbose'] = -1
            
            model = lgb.LGBMRegressor(**lgb_params)
            
            # Cross-validation
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_processed)):
                X_train_fold = self.X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = self.X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = lgb.LGBMRegressor(**lgb_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)
            
            # Train on full data
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train_processed, train_pred))
            
            self.models['LightGBM'] = model
            self.cv_scores['LightGBM'] = cv_r2_mean
            
            results_list.append({
                'Model': 'LightGBM',
                'CV_R2_Mean': cv_r2_mean,
                'CV_R2_Std': cv_r2_std,
                'Train_R2': train_r2,
                'Train_RMSE': train_rmse,
                'Overfit': train_r2 - cv_r2_mean
            })
            
            print(f"   ✓ CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
            print(f"   ✓ Train R²: {train_r2:.4f}, Overfit Gap: {train_r2 - cv_r2_mean:.4f}\n")
            
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
        
        # 3. CatBoost
        print("[3/3] Training CatBoost (tuned)...")
        try:
            import catboost as cb
            cat_params = self.tuned_params['CatBoost'].copy()
            cat_params['random_seed'] = RANDOM_SEED
            cat_params['verbose'] = False
            cat_params['thread_count'] = -1
            
            model = cb.CatBoostRegressor(**cat_params)
            
            # Cross-validation
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            cv_r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_processed)):
                X_train_fold = self.X_train_processed.iloc[train_idx]
                y_train_fold = self.y_train_processed[train_idx]
                X_val_fold = self.X_train_processed.iloc[val_idx]
                y_val_fold = self.y_train_processed[val_idx]
                
                model_fold = cb.CatBoostRegressor(**cat_params)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred = model_fold.predict(X_val_fold)
                cv_r2_scores.append(r2_score(y_val_fold, y_pred))
            
            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)
            
            # Train on full data
            model.fit(self.X_train_processed, self.y_train_processed)
            train_pred = model.predict(self.X_train_processed)
            train_r2 = r2_score(self.y_train_processed, train_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train_processed, train_pred))
            
            self.models['CatBoost'] = model
            self.cv_scores['CatBoost'] = cv_r2_mean
            
            results_list.append({
                'Model': 'CatBoost',
                'CV_R2_Mean': cv_r2_mean,
                'CV_R2_Std': cv_r2_std,
                'Train_R2': train_r2,
                'Train_RMSE': train_rmse,
                'Overfit': train_r2 - cv_r2_mean
            })
            
            print(f"   ✓ CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
            print(f"   ✓ Train R²: {train_r2:.4f}, Overfit Gap: {train_r2 - cv_r2_mean:.4f}\n")
            
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
        
        # Print summary
        self.results_df = pd.DataFrame(results_list).sort_values('CV_R2_Mean', ascending=False)
        print("\n" + "="*80)
        print("TUNED MODEL PERFORMANCE")
        print("="*80)
        print(self.results_df.to_string(index=False))
        print("="*80 + "\n")
    
    def create_ensemble(self):
        """Create weighted ensemble."""
        print("\n🤝 STEP 3: Creating Weighted Ensemble")
        print("-" * 80)
        
        self.ensemble = WeightedEnsemble(
            models=self.models,
            cv_scores=self.cv_scores
        )
        
        # Compute ensemble CV score
        total_weight = sum(self.ensemble.weights.values())
        weighted_cv = sum(
            self.cv_scores[name] * self.ensemble.weights[name] 
            for name in self.models.keys()
        ) / total_weight
        
        self.ensemble_cv_r2 = weighted_cv
        
        print(f"\n✓ Ensemble created with {len(self.models)} models")
        print(f"✓ Ensemble CV R²: {self.ensemble_cv_r2:.4f}")
        print(f"\nEnsemble Weights:")
        for name in sorted(self.ensemble.weights.keys(), 
                         key=lambda k: self.ensemble.weights[k], reverse=True):
            print(f"  {name:15s}: {self.ensemble.weights[name]:.4f}")
        print()
    
    def generate_predictions(self):
        """Generate predictions for test set."""
        print("\n🔮 STEP 4: Generating Test Predictions")
        print("-" * 80)
        
        raw_predictions = self.ensemble.predict(self.X_test_processed)
        
        # Round predictions to nearest integer (ages are discrete)
        self.predictions = np.round(raw_predictions)
        
        print(f"✓ Generated predictions for {len(self.predictions)} test samples")
        print(f"   Raw prediction range: [{raw_predictions.min():.2f}, {raw_predictions.max():.2f}]")
        print(f"   Rounded prediction range: [{self.predictions.min():.0f}, {self.predictions.max():.0f}]")
        print(f"   Prediction mean: {self.predictions.mean():.2f}")
        print(f"   Prediction std: {self.predictions.std():.2f}\n")
    
    def create_submission(self):
        """Create submission file."""
        print("\n💾 STEP 5: Creating Submission File")
        print("-" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"submission_tuned_{timestamp}.csv"
        
        submission = self.loader.create_submission(
            self.test_ids,
            self.predictions,
            filename=str(filename)
        )
        
        self.submission_file = str(filename)
        print()
    
    def save_artifacts(self):
        """Save models and results."""
        print("\n💾 STEP 6: Saving Artifacts")
        print("-" * 80)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.output_dir / 'models' / f'{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {model_path}")
        
        # Save ensemble
        ensemble_path = self.output_dir / 'models' / 'ensemble.pkl'
        with open(ensemble_path, 'wb') as f:
            pickle.dump(self.ensemble, f)
        print(f"✓ Saved: {ensemble_path}")
        
        # Save preprocessors
        preprocessing_path = self.output_dir / 'models' / 'preprocessing.pkl'
        with open(preprocessing_path, 'wb') as f:
            pickle.dump({
                'selector': self.selector,
                'scaler': self.scaler
            }, f)
        print(f"✓ Saved: {preprocessing_path}")
        
        # Save results
        results_path = self.output_dir / 'results.json'
        results = {
            'config': self.config,
            'models_trained': list(self.models.keys()),
            'best_model': self.results_df.iloc[0]['Model'],
            'best_cv_r2': float(self.results_df.iloc[0]['CV_R2_Mean']),
            'ensemble_cv_r2': float(self.ensemble_cv_r2),
            'weights': self.ensemble.weights,
            'submission_file': self.submission_file,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved: {results_path}")
        
        # Save results CSV
        results_csv = self.output_dir / 'model_comparison.csv'
        self.results_df.to_csv(results_csv, index=False)
        print(f"✓ Saved: {results_csv}\n")
    
    def run(self):
        """Run full pipeline."""
        try:
            start_time = datetime.now()
            
            self.load_and_preprocess()
            self.train_tuned_models()
            self.create_ensemble()
            self.generate_predictions()
            self.create_submission()
            self.save_artifacts()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "="*80)
            print("TUNED PIPELINE COMPLETE ✓")
            print("="*80)
            print(f"Models Trained: {len(self.models)}")
            print(f"\nBest Model: {self.results_df.iloc[0]['Model']}")
            print(f"Best CV R²: {self.results_df.iloc[0]['CV_R2_Mean']:.4f}")
            print(f"Best Overfit Gap: {self.results_df.iloc[0]['Overfit']:.4f}")
            print(f"\nEnsemble CV R²: {self.ensemble_cv_r2:.4f}")
            print(f"\nModel Weights:")
            for name in sorted(self.ensemble.weights.keys(), 
                             key=lambda k: self.ensemble.weights[k], reverse=True):
                print(f"  {name:15s}: {self.ensemble.weights[name]:.4f}")
            print(f"\nSubmission File: {self.submission_file}")
            print(f"Elapsed Time: {elapsed:.1f}s")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n✗ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Run tuned ensemble pipeline')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--k_features', type=int, default=100, help='Number of features')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': 'eth-aml-2025-project-1',
        'cv_folds': args.cv_folds,
        'k_features': args.k_features,
        'remove_outliers': args.remove_outliers
    }
    
    pipeline = TunedEnsemblePipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
