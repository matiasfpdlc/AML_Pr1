"""
Weighted Ensemble for Brain-Age Prediction.
Combines multiple model configurations with optimized weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class WeightedEnsemble:
    """
    Weighted ensemble of multiple trained models.
    Supports manual weights, CV-based weights, and optimized weights.
    """
    
    def __init__(self, 
                 models: Dict[str, Any], 
                 weights: Optional[Dict[str, float]] = None,
                 cv_scores: Optional[Dict[str, float]] = None):
        """
        Initialize weighted ensemble.
        
        Args:
            models: Dictionary of {model_name: trained_model}
            weights: Optional dictionary of {model_name: weight}
                    If None, uses equal weights or CV-based weights if available
            cv_scores: Optional dictionary of {model_name: cv_r2_score}
                      Used for automatic weight calculation if weights=None
        """
        if not models:
            raise ValueError("At least one model is required for ensemble")
        
        self.models = models
        self.model_names = list(models.keys())
        self.n_models = len(models)
        
        # Initialize weights
        if weights is not None:
            # Use provided weights (normalize)
            self._set_weights(weights)
        elif cv_scores is not None:
            # Use CV scores to compute weights
            self._set_cv_based_weights(cv_scores)
        else:
            # Equal weights
            self._set_equal_weights()
        
        print(f"✓ Weighted Ensemble initialized with {self.n_models} models")
        self._print_weights()
    
    def _set_weights(self, weights: Dict[str, float]):
        """Set and normalize weights."""
        # Ensure all models have weights
        for name in self.model_names:
            if name not in weights:
                raise ValueError(f"Missing weight for model: {name}")
        
        # Normalize to sum to 1
        total = sum(weights[name] for name in self.model_names)
        if total <= 0:
            raise ValueError("Total weight must be positive")
        
        self.weights = {name: weights[name] / total for name in self.model_names}
    
    def _set_equal_weights(self):
        """Set equal weights for all models."""
        weight = 1.0 / self.n_models
        self.weights = {name: weight for name in self.model_names}
    
    def _set_cv_based_weights(self, cv_scores: Dict[str, float]):
        """
        Set weights based on CV scores.
        Uses softmax of CV R² scores for weighting.
        """
        # Ensure all models have scores
        for name in self.model_names:
            if name not in cv_scores:
                raise ValueError(f"Missing CV score for model: {name}")
        
        # Use softmax of scores (emphasizes better models)
        scores = np.array([cv_scores[name] for name in self.model_names])
        
        # Handle negative scores by shifting
        if np.any(scores < 0):
            scores = scores - scores.min() + 0.01
        
        # Softmax with temperature for smoother distribution
        temperature = 2.0
        exp_scores = np.exp(scores / temperature)
        softmax_weights = exp_scores / exp_scores.sum()
        
        self.weights = {name: float(w) for name, w in 
                       zip(self.model_names, softmax_weights)}
    
    def _print_weights(self):
        """Print current weights."""
        print("\nEnsemble Weights:")
        for name in sorted(self.weights.keys(), key=lambda k: self.weights[k], reverse=True):
            print(f"  {name:20s}: {self.weights[name]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted predictions
        """
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        predictions = np.zeros(len(X))
        
        for name in self.model_names:
            try:
                model = self.models[name]
                weight = self.weights[name]
                model_pred = model.predict(X)
                
                # Validate predictions
                if len(model_pred) != len(X):
                    raise ValueError(f"Model {name} returned wrong number of predictions")
                
                predictions += weight * model_pred
                
            except Exception as e:
                print(f"⚠️  Error predicting with {name}: {e}")
                raise
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with R², RMSE, MAE
        """
        predictions = self.predict(X)
        
        metrics = {
            'r2': float(r2_score(y, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions))
        }
        
        return metrics
    
    def evaluate_individual_models(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Evaluate each individual model in the ensemble.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            DataFrame with individual model performances
        """
        results = []
        
        for name in self.model_names:
            try:
                model = self.models[name]
                pred = model.predict(X)
                
                results.append({
                    'Model': name,
                    'Weight': self.weights[name],
                    'R2': r2_score(y, pred),
                    'RMSE': np.sqrt(mean_squared_error(y, pred)),
                    'MAE': mean_absolute_error(y, pred)
                })
            except Exception as e:
                print(f"⚠️  Error evaluating {name}: {e}")
        
        df = pd.DataFrame(results)
        df = df.sort_values('R2', ascending=False)
        return df
    
    def optimize_weights(self, X: pd.DataFrame, y: np.ndarray, 
                        method: str = 'grid_search',
                        cv: int = 5) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            X: Feature matrix for validation
            y: True labels for validation
            method: Optimization method ('grid_search' or 'cv_based')
            cv: Number of CV folds for CV-based method
            
        Returns:
            Optimized weights dictionary
        """
        print(f"\n🔧 Optimizing ensemble weights using {method}...")
        
        if method == 'grid_search':
            best_weights = self._grid_search_weights(X, y)
        elif method == 'cv_based':
            best_weights = self._cv_optimize_weights(X, y, cv)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Update weights
        self._set_weights(best_weights)
        
        print("\n✓ Weights optimized!")
        self._print_weights()
        
        return self.weights
    
    def _grid_search_weights(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Simple grid search for 2-5 models.
        Tests different weight combinations and picks the best.
        """
        if self.n_models > 5:
            print("⚠️  Too many models for grid search, using equal weights")
            return {name: 1.0 / self.n_models for name in self.model_names}
        
        best_r2 = -np.inf
        best_weights = None
        
        # Generate weight combinations
        if self.n_models == 2:
            weight_options = [0.3, 0.4, 0.5, 0.6, 0.7]
            for w1 in weight_options:
                w2 = 1.0 - w1
                weights = {self.model_names[0]: w1, self.model_names[1]: w2}
                r2 = self._eval_weights(weights, X, y)
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = weights
        
        elif self.n_models == 3:
            weight_options = [0.2, 0.3, 0.4, 0.5]
            for w1 in weight_options:
                for w2 in weight_options:
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1 and w3 <= 0.6:
                        weights = {self.model_names[0]: w1, 
                                  self.model_names[1]: w2, 
                                  self.model_names[2]: w3}
                        r2 = self._eval_weights(weights, X, y)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_weights = weights
        
        else:  # 4-5 models: use simpler search
            # Test equal weights
            equal_weights = {name: 1.0 / self.n_models for name in self.model_names}
            best_r2 = self._eval_weights(equal_weights, X, y)
            best_weights = equal_weights
            
            # Test top-2 weighted, others equal
            for i in range(self.n_models):
                for j in range(i + 1, self.n_models):
                    weights = {name: 0.05 for name in self.model_names}
                    remaining = 1.0 - 0.05 * (self.n_models - 2)
                    weights[self.model_names[i]] = remaining * 0.6
                    weights[self.model_names[j]] = remaining * 0.4
                    
                    r2 = self._eval_weights(weights, X, y)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_weights = weights
        
        print(f"   Best validation R²: {best_r2:.4f}")
        return best_weights
    
    def _cv_optimize_weights(self, X: pd.DataFrame, y: np.ndarray, 
                            cv: int = 5) -> Dict[str, float]:
        """Optimize weights using cross-validation."""
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Get CV scores for each model
        cv_scores = {}
        for name in self.model_names:
            model = self.models[name]
            fold_scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_val = X.iloc[val_idx]
                y_val = y[val_idx]
                
                try:
                    pred = model.predict(X_val)
                    r2 = r2_score(y_val, pred)
                    fold_scores.append(r2)
                except Exception as e:
                    print(f"⚠️  Error in CV for {name}: {e}")
                    fold_scores.append(0.0)
            
            cv_scores[name] = np.mean(fold_scores)
        
        print("\nCV Scores:")
        for name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:20s}: {score:.4f}")
        
        # Use softmax of CV scores
        scores = np.array([cv_scores[name] for name in self.model_names])
        scores = scores - scores.min() + 0.01  # Ensure positive
        exp_scores = np.exp(scores / 2.0)
        weights_array = exp_scores / exp_scores.sum()
        
        weights = {name: float(w) for name, w in zip(self.model_names, weights_array)}
        return weights
    
    def _eval_weights(self, weights: Dict[str, float], 
                     X: pd.DataFrame, y: np.ndarray) -> float:
        """Evaluate a specific weight configuration."""
        predictions = np.zeros(len(X))
        total_weight = sum(weights.values())
        
        for name in self.model_names:
            model = self.models[name]
            weight = weights[name] / total_weight
            predictions += weight * model.predict(X)
        
        return r2_score(y, predictions)
    
    def get_summary(self) -> str:
        """Get a summary string of the ensemble."""
        summary = f"Weighted Ensemble ({self.n_models} models)\n"
        summary += "=" * 50 + "\n"
        for name in sorted(self.weights.keys(), key=lambda k: self.weights[k], reverse=True):
            summary += f"{name:20s}: {self.weights[name]:.4f}\n"
        return summary


if __name__ == "__main__":
    # Simple test
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    
    print("Testing WeightedEnsemble...")
    
    # Create dummy data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randn(100) * 10 + 50
    
    # Train simple models
    models = {
        'Ridge': Ridge().fit(X, y),
        'Lasso': Lasso().fit(X, y),
        'RF': RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    }
    
    cv_scores = {'Ridge': 0.5, 'Lasso': 0.45, 'RF': 0.55}
    
    # Create ensemble
    ensemble = WeightedEnsemble(models, cv_scores=cv_scores)
    
    # Predict
    predictions = ensemble.predict(X)
    
    # Evaluate
    metrics = ensemble.evaluate(X, y)
    print(f"\nEnsemble R²: {metrics['r2']:.4f}")
    print(f"Ensemble RMSE: {metrics['rmse']:.2f}")
    
    print("\n✓ Test passed!")
