"""
Multi-Model SelectKBest Ensemble for Brain-Age Prediction.
Tests XGBoost, LightGBM, CatBoost, and SVR with polynomial features.

Usage:
    python multi_model_selectkbest.py --cv_folds 5 --remove_outliers
    python multi_model_selectkbest.py --quick  # Fast test
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, StackingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class SimpleStacking:
    """Simple stacking ensemble with Ridge meta-learner."""
    def __init__(self, base_models, meta_learner):
        self.base_models = base_models
        self.meta_learner = meta_learner
    
    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models
        ])
        return self.meta_learner.predict(meta_features)


class MultiModelSelectKBestPipeline:
    """
    Pipeline that trains multiple model types with polynomial features.
    Models: XGBoost, LightGBM, CatBoost, SVR
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader(config['data_dir'])
        self.models = {}
        self.cv_scores = {}
        self.ensemble = None
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'output_multi_model'))
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        self._print_header()
    
    def _print_header(self):
        """Print pipeline header."""
        print("\n" + "="*80)
        print("MULTI-MODEL ENSEMBLE WITH POLYNOMIAL FEATURES")
        print("Testing XGBoost, LightGBM, CatBoost, SVR")
        print(f"K = {self.config.get('k_features', 100)} base features")
        print(f"Polynomial Features: {'Enabled' if self.config.get('use_poly', True) else 'Disabled'}")
        print("="*80)
        print("Configuration:")
        for key, value in self.config.items():
            print(f"  {key:25s}: {value}")
        print("="*80 + "\n")
    
    def load_and_preprocess(self):
        """Load data and do preprocessing with SelectKBest."""
        print("\n📂 STEP 1: Loading and Preprocessing")
        print("-" * 80)
        
        # Load data
        self.X_train, self.y_train = self.loader.load_train_data()
        self.X_test, self.test_ids = self.loader.load_test_data()
        
        print(f"✓ Loaded training data: X_train shape {self.X_train.shape}, y_train shape {self.y_train.shape}")
        print(f"✓ Loaded test data: X_test shape {self.X_test.shape}")
        
        # 1. Impute missing values with KNN
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
        
        # 2. Remove outliers if requested
        if self.config.get('remove_outliers', False):
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
        
        # 7. Add polynomial features (interaction terms) - more conservative
        if self.config.get('use_poly', False):  # Disabled by default
            print("🔬 Creating polynomial interaction features (degree=2)...")
            # Use fewer base features for polynomial to avoid overfitting
            poly_base_k = min(50, k)  # Use only top 50 for polynomial
            poly_base_selector = SelectKBest(score_func=f_regression, k=poly_base_k)
            X_train_poly_base = poly_base_selector.fit_transform(X_train_scaled, self.y_train)
            X_test_poly_base = poly_base_selector.transform(X_test_scaled)
            
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_poly_base)
            X_test_poly = poly.transform(X_test_poly_base)
            
            print(f"   Generated {X_train_poly.shape[1]} interaction features from {poly_base_k} base features")
            
            # Combine original scaled features with best polynomial features
            poly_k = min(50, X_train_poly.shape[1] - poly_base_k)  # Select top 50 new interactions
            if poly_k > 0:
                print(f"   Selecting top {poly_k} new interaction features...")
                # Only select from the NEW polynomial features (exclude originals)
                X_train_poly_new = X_train_poly[:, poly_base_k:]  # Skip original features
                X_test_poly_new = X_test_poly[:, poly_base_k:]
                
                poly_selector = SelectKBest(score_func=f_regression, k=poly_k)
                X_train_poly_selected = poly_selector.fit_transform(X_train_poly_new, self.y_train)
                X_test_poly_selected = poly_selector.transform(X_test_poly_new)
                
                # Combine original scaled features with selected polynomial features
                X_train_final = np.hstack([X_train_scaled, X_train_poly_selected])
                X_test_final = np.hstack([X_test_scaled, X_test_poly_selected])
                
                feature_names = [f"f{i}" for i in range(k)] + [f"poly_{i}" for i in range(poly_k)]
            else:
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
                feature_names = [f"f{i}" for i in range(k)]
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            feature_names = [f"f{i}" for i in range(k)]
        
        # Convert to DataFrames
        self.X_train_processed = pd.DataFrame(X_train_final, columns=feature_names)
        self.X_test_processed = pd.DataFrame(X_test_final, columns=feature_names)
        self.y_train_processed = self.y_train if isinstance(self.y_train, np.ndarray) else self.y_train.values
        
        print(f"\n✓ Preprocessing complete")
        print(f"   Samples: {len(self.X_train_processed)} train, {len(self.X_test_processed)} test")
        print(f"   Features: {self.X_train_processed.shape[1]}\n")
    
    def get_models(self):
        """Get all models to train."""
        models = {}
        
        # 1. XGBoost
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=4,
                min_child_weight=8,
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=0.8,
                reg_alpha=5.0,
                reg_lambda=8.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0
            )
        except ImportError:
            print("⚠️  XGBoost not available")
        
        # 2. LightGBM
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=10,
                num_leaves=80,
                min_child_samples=25,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
        except ImportError:
            print("⚠️  LightGBM not available")
        
        # 3. CatBoost
        try:
            import catboost as cb
            models['CatBoost'] = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=5.0,
                subsample=0.7,
                random_seed=RANDOM_SEED,
                verbose=False,
                thread_count=-1
            )
        except ImportError:
            print("⚠️  CatBoost not available")
        
        # 4. Support Vector Regression
        from sklearn.svm import SVR
        models['SVR'] = SVR(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            epsilon=0.1,
            cache_size=1000
        )
        
        return models
    
    def train_all_models(self):
        """Train all models with cross-validation."""
        print("\n🎓 STEP 2: Training Multiple Model Types")
        print("-" * 80)
        
        models = self.get_models()
        cv_folds = self.config.get('cv_folds', 5)
        
        print(f"\nTraining {len(models)} models with {cv_folds}-fold CV:\n")
        
        results_list = []
        
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"[{i}/{len(models)}] Training {model_name}...")
            
            try:
                # Cross-validation
                kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
                cv_r2_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_processed)):
                    X_train_fold = self.X_train_processed.iloc[train_idx]
                    y_train_fold = self.y_train_processed[train_idx]
                    X_val_fold = self.X_train_processed.iloc[val_idx]
                    y_val_fold = self.y_train_processed[val_idx]
                    
                    # Clone and train
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Validate
                    y_pred_val = fold_model.predict(X_val_fold)
                    r2 = r2_score(y_val_fold, y_pred_val)
                    cv_r2_scores.append(r2)
                
                cv_r2_mean = np.mean(cv_r2_scores)
                cv_r2_std = np.std(cv_r2_scores)
                
                # Train on full data
                model.fit(self.X_train_processed, self.y_train_processed)
                
                # Training performance
                y_pred_train = model.predict(self.X_train_processed)
                train_r2 = r2_score(self.y_train_processed, y_pred_train)
                train_rmse = np.sqrt(mean_squared_error(self.y_train_processed, y_pred_train))
                
                # Store
                self.models[model_name] = model
                self.cv_scores[model_name] = cv_r2_mean
                
                results_list.append({
                    'Model': model_name,
                    'CV_R2_Mean': cv_r2_mean,
                    'CV_R2_Std': cv_r2_std,
                    'Train_R2': train_r2,
                    'Train_RMSE': train_rmse,
                    'Overfit': train_r2 - cv_r2_mean
                })
                
                print(f"   ✓ CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
                print(f"   ✓ Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}\n")
                
            except Exception as e:
                print(f"   ✗ ERROR training {model_name}: {e}\n")
                continue
        
        if not self.models:
            raise RuntimeError("Failed to train any models!")
        
        # Print summary
        self.results_df = pd.DataFrame(results_list).sort_values('CV_R2_Mean', ascending=False)
        print("\n" + "="*80)
        print("MODEL TRAINING SUMMARY")
        print("="*80)
        print(self.results_df.to_string(index=False))
        print("="*80 + "\n")
    
    def create_ensemble(self):
        """Create both weighted ensemble and stacking ensemble."""
        print("\n🤝 STEP 3: Creating Ensembles")
        print("-" * 80)
        
        # Exclude models with negative CV scores from ensemble
        ensemble_models = {name: model for name, model in self.models.items() 
                          if self.cv_scores.get(name, 0) > 0}
        ensemble_cv_scores = {name: score for name, score in self.cv_scores.items() 
                             if score > 0}
        
        if len(ensemble_models) < 2:
            print("⚠️  Only one model available for ensemble, using single model")
            ensemble_models = self.models
            ensemble_cv_scores = self.cv_scores
        
        excluded = set(self.models.keys()) - set(ensemble_models.keys())
        if excluded:
            print(f"ℹ️  Excluding {', '.join(excluded)} from ensemble (negative CV R²)")
        
        # 1. Create weighted ensemble with CV-based weights
        print("\n[1/2] Creating Weighted Ensemble...")
        self.ensemble = WeightedEnsemble(
            models=ensemble_models,
            cv_scores=ensemble_cv_scores
        )
        
        # Compute weighted ensemble CV score
        total_weight = sum(self.ensemble.weights.values())
        weighted_cv = sum(
            ensemble_cv_scores[name] * self.ensemble.weights[name] 
            for name in ensemble_models.keys()
        ) / total_weight
        self.ensemble_cv_r2 = weighted_cv
        
        print(f"✓ Weighted Ensemble created with {len(ensemble_models)} models: {list(ensemble_models.keys())}")
        print(f"✓ Weighted Ensemble CV R²: {self.ensemble_cv_r2:.4f}")
        
        # 2. Create stacking ensemble with Ridge meta-learner
        print("\n[2/2] Creating Stacking Ensemble...")
        estimators = [(name, model) for name, model in ensemble_models.items()]
        
        # Compute stacking CV score with simple holdout validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        stacking_cv_scores = []
        
        print("   Computing stacking CV score (5 folds)...")
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_processed), 1):
            X_train_fold = self.X_train_processed.iloc[train_idx]
            y_train_fold = self.y_train_processed[train_idx]
            X_val_fold = self.X_train_processed.iloc[val_idx]
            y_val_fold = self.y_train_processed[val_idx]
            
            # Use prefit=False, no nested CV for speed
            stacking_fold = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=10.0),
                cv='prefit',  # Use prefit models
                n_jobs=-1
            )
            
            # Train base models on fold
            fold_estimators = []
            for name, model_template in ensemble_models.items():
                # Clone and train model on this fold
                from sklearn.base import clone
                fold_model = clone(model_template)
                fold_model.fit(X_train_fold, y_train_fold)
                fold_estimators.append((name, fold_model))
            
            # Create meta-features for training and validation
            meta_features = np.column_stack([
                model.predict(X_train_fold) for _, model in fold_estimators
            ])
            meta_features_val = np.column_stack([
                model.predict(X_val_fold) for _, model in fold_estimators
            ])
            
            # Train simple Ridge meta-learner
            meta_learner = Ridge(alpha=10.0)
            meta_learner.fit(meta_features, y_train_fold)
            
            # Predict on validation fold
            y_pred = meta_learner.predict(meta_features_val)
            fold_r2 = r2_score(y_val_fold, y_pred)
            stacking_cv_scores.append(fold_r2)
            print(f"      Fold {fold_num}/5: R² = {fold_r2:.4f}")
        
        self.stacking_cv_r2 = np.mean(stacking_cv_scores)
        self.stacking_cv_std = np.std(stacking_cv_scores)
        
        # Train final stacking model on all data
        print("   Training final stacking model on full dataset...")
        
        # Train meta-learner on full dataset
        meta_features_all = np.column_stack([
            model.predict(self.X_train_processed) for _, model in estimators
        ])
        final_meta_learner = Ridge(alpha=10.0)
        final_meta_learner.fit(meta_features_all, self.y_train_processed)
        
        self.stacking = SimpleStacking(estimators, final_meta_learner)
        
        print(f"✓ Stacking Ensemble trained")
        print(f"✓ Stacking Ensemble CV R²: {self.stacking_cv_r2:.4f} (±{self.stacking_cv_std:.4f})")
        
        # Choose best ensemble
        if self.stacking_cv_r2 > self.ensemble_cv_r2:
            self.best_ensemble = 'stacking'
            self.best_cv_r2 = self.stacking_cv_r2
            print(f"\n🏆 Best Ensemble: Stacking (CV R² = {self.stacking_cv_r2:.4f})")
        else:
            self.best_ensemble = 'weighted'
            self.best_cv_r2 = self.ensemble_cv_r2
            print(f"\n🏆 Best Ensemble: Weighted (CV R² = {self.ensemble_cv_r2:.4f})")
        print()
    
    def generate_predictions(self):
        """Generate predictions for test set using best ensemble."""
        print("\n🔮 STEP 4: Generating Test Predictions")
        print("-" * 80)
        
        # Use best ensemble
        if self.best_ensemble == 'stacking':
            print(f"Using Stacking Ensemble (CV R² = {self.stacking_cv_r2:.4f})")
            raw_predictions = self.stacking.predict(self.X_test_processed)
        else:
            print(f"Using Weighted Ensemble (CV R² = {self.ensemble_cv_r2:.4f})")
            raw_predictions = self.ensemble.predict(self.X_test_processed)
        
        # Round predictions to nearest integer (ages are discrete)
        self.predictions = np.round(raw_predictions)
        
        print(f"\n✓ Generated predictions for {len(self.predictions)} test samples")
        print(f"   Raw prediction range: [{raw_predictions.min():.2f}, {raw_predictions.max():.2f}]")
        print(f"   Rounded prediction range: [{self.predictions.min():.0f}, {self.predictions.max():.0f}]")
        print(f"   Prediction mean: {self.predictions.mean():.2f}")
        print(f"   Prediction std: {self.predictions.std():.2f}\n")
    
    def create_plots(self):
        """Create visualization plots."""
        print("\n📊 STEP 5: Creating Plots")
        print("-" * 80)
        
        # Plot 1: Model comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CV R² comparison
        models_sorted = self.results_df.sort_values('CV_R2_Mean', ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models_sorted)))
        
        ax1.barh(models_sorted['Model'], models_sorted['CV_R2_Mean'], color=colors)
        ax1.set_xlabel('CV R² Score', fontsize=12)
        ax1.set_ylabel('Model', fontsize=12)
        ax1.set_title('Cross-Validation Performance Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for i, (idx, row) in enumerate(models_sorted.iterrows()):
            ax1.text(row['CV_R2_Mean'] + 0.01, i, f"{row['CV_R2_Mean']:.4f}", 
                    va='center', fontsize=10)
        
        # Ensemble weights
        weights_data = pd.DataFrame({
            'Model': list(self.ensemble.weights.keys()),
            'Weight': list(self.ensemble.weights.values())
        }).sort_values('Weight', ascending=True)
        
        ax2.barh(weights_data['Model'], weights_data['Weight'], color=colors)
        ax2.set_xlabel('Weight', fontsize=12)
        ax2.set_ylabel('Model', fontsize=12)
        ax2.set_title('Ensemble Weights', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for i, (idx, row) in enumerate(weights_data.iterrows()):
            ax2.text(row['Weight'] + 0.01, i, f"{row['Weight']:.4f}", 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'model_comparison.png', dpi=300)
        plt.close()
        print("   ✓ Saved: model_comparison.png")
        
        # Plot 2: True vs Predicted for each model
        n_models = len(self.models)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, (model_name, model) in enumerate(self.models.items()):
            ax = axes[i]
            
            y_pred = model.predict(self.X_train_processed)
            
            ax.scatter(self.y_train_processed, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(self.y_train_processed.min(), y_pred.min())
            max_val = max(self.y_train_processed.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_xlabel('True Age (years)', fontsize=11)
            ax.set_ylabel('Predicted Age (years)', fontsize=11)
            ax.set_title(f'{model_name}\nCV R² = {self.cv_scores[model_name]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'true_vs_pred_all_models.png', dpi=300)
        plt.close()
        print("   ✓ Saved: true_vs_pred_all_models.png")
        
        print()
    
    def create_submission(self):
        """Create submission file."""
        print("\n💾 STEP 6: Creating Submission File")
        print("-" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"submission_multi_model_{timestamp}.csv"
        
        submission = self.loader.create_submission(
            self.test_ids,
            self.predictions,
            filename=str(filename)
        )
        
        self.submission_file = str(filename)
        print()
    
    def save_artifacts(self):
        """Save models and results."""
        print("\n💾 STEP 7: Saving Artifacts")
        print("-" * 80)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.output_dir / 'models' / f'{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {model_path}")
        
        # Save ensembles
        ensemble_path = self.output_dir / 'models' / 'ensemble_weighted.pkl'
        with open(ensemble_path, 'wb') as f:
            pickle.dump(self.ensemble, f)
        print(f"✓ Saved: {ensemble_path}")
        
        stacking_path = self.output_dir / 'models' / 'ensemble_stacking.pkl'
        with open(stacking_path, 'wb') as f:
            pickle.dump(self.stacking, f)
        print(f"✓ Saved: {stacking_path}")
        
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
            'weighted_ensemble_cv_r2': float(self.ensemble_cv_r2),
            'stacking_ensemble_cv_r2': float(self.stacking_cv_r2),
            'best_ensemble': self.best_ensemble,
            'best_ensemble_cv_r2': float(self.best_cv_r2),
            'weighted_ensemble_weights': self.ensemble.weights,
            'submission_file': self.submission_file,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved: {results_path}")
        
        # Save results CSV
        csv_path = self.output_dir / 'model_comparison.csv'
        self.results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        
        print()
    
    def run(self):
        """Run complete pipeline."""
        start_time = datetime.now()
        
        try:
            self.load_and_preprocess()
            self.train_all_models()
            self.create_ensemble()
            self.generate_predictions()
            self.create_plots()
            self.create_submission()
            self.save_artifacts()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE ✓")
            print("="*80)
            print(f"Models Trained: {len(self.models)}")
            print(f"\nBest Model: {self.results_df.iloc[0]['Model']}")
            print(f"Best CV R²: {self.results_df.iloc[0]['CV_R2_Mean']:.4f}")
            print(f"\nWeighted Ensemble CV R²: {self.ensemble_cv_r2:.4f}")
            print(f"Stacking Ensemble CV R²: {self.stacking_cv_r2:.4f}")
            print(f"\n🏆 Best Ensemble: {self.best_ensemble.capitalize()} (CV R² = {self.best_cv_r2:.4f})")
            print(f"\nWeighted Ensemble Weights:")
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
            raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Model SelectKBest Ensemble Pipeline"
    )
    
    parser.add_argument('--data_dir', type=str, default='eth-aml-2025-project-1',
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='output_multi_model',
                       help='Directory for outputs')
    parser.add_argument('--k_features', type=int, default=100,
                       help='Number of features to select with SelectKBest')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--remove_outliers', action='store_true',
                       help='Remove detected outliers from training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run')
    
    args = parser.parse_args()
    
    if args.quick:
        args.cv_folds = 3
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'k_features': args.k_features,
        'cv_folds': args.cv_folds,
        'remove_outliers': args.remove_outliers,
    }
    
    pipeline = MultiModelSelectKBestPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
