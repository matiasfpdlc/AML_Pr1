"""
Model training and evaluation for Brain-Age Prediction.
Implements multiple regression models with cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# XGBoost and LightGBM (install if available)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")


class ModelTrainer:
    """
    Train and evaluate multiple regression models with cross-validation.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def get_diverse_model_configs(self) -> Dict[str, Any]:
        """
        Get 5 diverse model configurations for ensemble.
        Each config has different hyperparameters for diversity.
        
        Returns:
            Dictionary of 5 model configurations with distinct settings
        """
        configs = {}
        
        # Config 1: Conservative XGBoost (high regularization)
        if HAS_XGBOOST:
            configs['XGB_Conservative'] = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=3,
                min_child_weight=10,
                subsample=0.6,
                colsample_bytree=0.6,
                gamma=1.0,
                reg_alpha=10.0,
                reg_lambda=10.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        else:
            # Fallback: Ridge with high regularization
            configs['Ridge_Conservative'] = Ridge(
                alpha=50.0,
                random_state=self.random_state
            )
        
        # Config 2: Balanced XGBoost (moderate settings)
        if HAS_XGBOOST:
            configs['XGB_Balanced'] = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=4,
                min_child_weight=8,
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=0.8,
                reg_alpha=5.0,
                reg_lambda=8.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        else:
            # Fallback: ElasticNet
            configs['ElasticNet_Balanced'] = ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                random_state=self.random_state,
                max_iter=2000
            )
        
        # Config 3: Aggressive XGBoost (lower regularization)
        if HAS_XGBOOST:
            configs['XGB_Aggressive'] = xgb.XGBRegressor(
                n_estimators=700,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.5,
                reg_alpha=1.0,
                reg_lambda=3.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        else:
            # Fallback: RandomForest
            configs['RF_Aggressive'] = RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        # Config 4: LightGBM (fast, leaf-wise growth)
        if HAS_LIGHTGBM:
            configs['LGBM_Fast'] = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=7,
                num_leaves=50,
                min_child_samples=15,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        else:
            # Fallback: GradientBoosting
            configs['GB_Fast'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            )
        
        # Config 5: LightGBM with different settings
        if HAS_LIGHTGBM:
            configs['LGBM_Deep'] = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=10,
                num_leaves=80,
                min_child_samples=25,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        else:
            # Fallback: Another tree model
            configs['RF_Deep'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        return configs
    
    def get_models(self) -> Dict[str, Any]:
        """
        Define all models to train.
        
        Returns:
            Dictionary of model_name -> model_instance
        """
        models = {
            'Ridge': Ridge(alpha=10.0, random_state=self.random_state),
            
            'Lasso': Lasso(alpha=1.0, random_state=self.random_state, max_iter=2000),
            
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            ),
        }
        
        # Add XGBoost if available
        # Configuration optimized via hyperparameter tuning (tune_xgboost.py)
        # Using 'balanced_config' - CV R²: 0.513, Overfit gap: 0.334 (best CV performance)
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=500,  # Good balance
                learning_rate=0.02,  # Moderate learning rate
                max_depth=4,  # Balanced tree depth
                min_child_weight=8,  # Moderate minimum samples per leaf
                subsample=0.7,  # 70% row sampling
                colsample_bytree=0.7,  # 70% column sampling per tree
                colsample_bylevel=0.7,  # 70% column sampling per level
                colsample_bynode=0.7,  # 70% column sampling per split
                gamma=0.8,  # Moderate minimum split loss
                reg_alpha=5.0,  # Moderate L1 regularization
                reg_lambda=8.0,  # Moderate L2 regularization
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        
        return models
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: np.ndarray, 
                           cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation and return metrics.
        
        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with mean and std of R² scores
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Calculate R² scores
        r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2', n_jobs=self.n_jobs)
        
        # Calculate negative MSE and convert to positive
        neg_mse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=self.n_jobs)
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        return {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'r2_scores': r2_scores.tolist(),
            'rmse_mean': float(np.mean(rmse_scores)),
            'rmse_std': float(np.std(rmse_scores)),
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                        cv: int = 5) -> Dict[str, Dict]:
        """
        Train and cross-validate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of CV folds
            
        Returns:
            Dictionary with results for each model
        """
        print("\n" + "="*70)
        print("MODEL TRAINING & CROSS-VALIDATION")
        print("="*70)
        
        self.models = self.get_models()
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Cross-validation
            cv_results = self.cross_validate_model(model, X_train, y_train, cv=cv)
            
            # Train on full dataset
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            y_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            train_mae = mean_absolute_error(y_train, y_pred)
            
            results[name] = {
                'model': model,
                'cv_r2_mean': cv_results['r2_mean'],
                'cv_r2_std': cv_results['r2_std'],
                'cv_rmse_mean': cv_results['rmse_mean'],
                'cv_rmse_std': cv_results['rmse_std'],
                'train_r2': float(train_r2),
                'train_rmse': float(train_rmse),
                'train_mae': float(train_mae),
            }
            
            print(f"   CV R²: {cv_results['r2_mean']:.4f} (±{cv_results['r2_std']:.4f})")
            print(f"   Train R²: {train_r2:.4f}")
            print(f"   Train RMSE: {train_rmse:.2f} years")
        
        self.results = results
        
        # Find best model by CV R²
        best_name = max(results.keys(), key=lambda k: results[k]['cv_r2_mean'])
        self.best_model_name = best_name
        self.best_model = results[best_name]['model']
        
        print("\n" + "="*70)
        print(f"🏆 Best model: {best_name}")
        print(f"   CV R²: {results[best_name]['cv_r2_mean']:.4f} (±{results[best_name]['cv_r2_std']:.4f})")
        print("="*70 + "\n")
        
        return results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get formatted results as DataFrame."""
        if not self.results:
            raise RuntimeError("No results available. Train models first.")
        
        data = []
        for name, metrics in self.results.items():
            data.append({
                'Model': name,
                'CV_R2_Mean': metrics['cv_r2_mean'],
                'CV_R2_Std': metrics['cv_r2_std'],
                'CV_RMSE_Mean': metrics['cv_rmse_mean'],
                'Train_R2': metrics['train_r2'],
                'Train_RMSE': metrics['train_rmse'],
                'Train_MAE': metrics['train_mae'],
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('CV_R2_Mean', ascending=False)
        return df
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict on
            model_name: Name of model to use (default: best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise RuntimeError("No trained models available")
            model = self.best_model
            name = self.best_model_name
        else:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found")
            model = self.results[model_name]['model']
            name = model_name
        
        print(f"🔮 Predicting with {name}...")
        predictions = model.predict(X)
        return predictions
    
    def get_feature_importance(self, model_name: Optional[str] = None, 
                              top_k: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importances for tree-based models.
        
        Args:
            model_name: Model to get importances from (default: best)
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importances or None
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.results[model_name]['model']
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importances")
            return None
        
        importances = model.feature_importances_
        
        # Assuming feature names are stored somewhere or passed
        # For now, return indices
        indices = np.argsort(importances)[::-1][:top_k]
        
        return pd.DataFrame({
            'feature_idx': indices,
            'importance': importances[indices]
        })


class EnsembleModel:
    """
    Ensemble multiple models by averaging predictions.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: Dictionary of model_name -> trained_model
            weights: Optional dictionary of model_name -> weight (default: equal weights)
        """
        self.models = models
        self.model_names = list(models.keys())
        
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0 / len(models) for name in self.model_names}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble prediction."""
        predictions = np.zeros(len(X))
        
        for name in self.model_names:
            model = self.models[name]
            weight = self.weights[name]
            predictions += weight * model.predict(X)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        predictions = self.predict(X)
        
        return {
            'r2': float(r2_score(y, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions))
        }


if __name__ == "__main__":
    # Test model training
    from dataloader import DataLoader
    from preprocessing import DataPreprocessor
    
    print("Testing model training pipeline...")
    
    # Load data
    loader = DataLoader()
    X_train, y_train = loader.load_train_data()
    
    # Preprocess
    preprocessor = DataPreprocessor(n_features=150)
    X_processed, y_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_processed, y_processed, cv=5)
    
    # Show results
    print("\nResults Summary:")
    print(trainer.get_results_dataframe())
