"""
Preprocessing utilities for Brain-Age Prediction.
Implements:
- Subtask 0: Missing value imputation
- Subtask 1: Outlier detection
- Subtask 2: Feature selection
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import (
    VarianceThreshold, 
    mutual_info_regression,
    SelectKBest,
    f_regression
)
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MissingValueImputer:
    """
    Subtask 0: Handle missing values in the dataset.
    
    Strategies:
    - 'mean': Mean imputation (for normally distributed features)
    - 'median': Median imputation (robust to outliers)
    - 'knn': KNN imputation (uses feature correlations)
    """
    
    def __init__(self, strategy: str = 'median', n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        
        if strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=n_neighbors)
        elif strategy in ['mean', 'median', 'most_frequent']:
            self.imputer = SimpleImputer(strategy=strategy)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer on training data and transform."""
        print(f"🔧 Imputing missing values using '{self.strategy}' strategy...")
        print(f"   Missing values before: {X.isnull().sum().sum()}")
        
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"   Missing values after: {X_imputed.isnull().sum().sum()}")
        assert X_imputed.isnull().sum().sum() == 0, "Imputation failed - NaNs still present!"
        
        return X_imputed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted imputer."""
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        assert X_imputed.isnull().sum().sum() == 0, "Imputation failed - NaNs still present!"
        return X_imputed


class OutlierDetector:
    """
    Subtask 1: Detect outliers in the training set.
    
    Methods:
    - 'isolation_forest': Isolation Forest (good for high-dimensional data)
    - 'lof': Local Outlier Factor (density-based)
    - 'robust_zscore': Modified Z-score using median absolute deviation
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.outlier_mask = None
        
        if method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'lof':
            self.detector = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20
            )
        elif method == 'robust_zscore':
            self.detector = None  # Will use custom implementation
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect outliers in the dataset.
        
        Returns:
            Boolean mask: True for inliers, False for outliers
        """
        print(f"🔍 Detecting outliers using '{self.method}' method...")
        
        if self.method == 'robust_zscore':
            # Use modified Z-score: |0.6745(x - median) / MAD|
            inlier_mask = self._robust_zscore_detection(X)
        elif self.method == 'lof':
            # LOF returns -1 for outliers, 1 for inliers
            predictions = self.detector.fit_predict(X)
            inlier_mask = predictions == 1
        else:
            # Isolation Forest returns -1 for outliers, 1 for inliers
            predictions = self.detector.fit_predict(X)
            inlier_mask = predictions == 1
        
        self.outlier_mask = inlier_mask
        n_outliers = (~inlier_mask).sum()
        outlier_pct = 100 * n_outliers / len(inlier_mask)
        
        print(f"   Detected {n_outliers} outliers ({outlier_pct:.2f}%)")
        print(f"   Inliers: {inlier_mask.sum()}")
        
        return inlier_mask
    
    def _robust_zscore_detection(self, X: pd.DataFrame, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using robust Z-score (MAD-based)."""
        X_array = X.values
        median = np.median(X_array, axis=0)
        mad = np.median(np.abs(X_array - median), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)
        
        modified_z_scores = 0.6745 * np.abs(X_array - median) / mad
        
        # Sample is outlier if any feature has extreme Z-score
        outlier_scores = np.max(modified_z_scores, axis=1)
        inlier_mask = outlier_scores < threshold
        
        return inlier_mask
    
    def remove_outliers(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove detected outliers from dataset."""
        if self.outlier_mask is None:
            raise RuntimeError("Must call detect() before remove_outliers()")
        
        X_clean = X[self.outlier_mask].copy()
        y_clean = y[self.outlier_mask].copy()
        
        print(f"   Removed outliers: {X.shape[0]} -> {X_clean.shape[0]} samples")
        
        return X_clean, y_clean


class FeatureSelector:
    """
    Subtask 2: Select informative features and remove noise.
    
    Multi-stage approach:
    1. Remove low-variance features
    2. Remove highly correlated features (redundant)
    3. Select top K features by mutual information
    4. Optional: Model-based selection using feature importance
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 n_features: int = 200,
                 use_mutual_info: bool = True,
                 use_model_selection: bool = True):
        
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.use_mutual_info = use_mutual_info
        self.use_model_selection = use_model_selection
        
        self.selected_features = None
        self.feature_scores = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelector':
        """Learn which features to select."""
        print(f"🎯 Selecting features from {X.shape[1]} candidates...")
        
        selected_features = X.columns.tolist()
        
        # Stage 1: Remove low-variance features
        print(f"   Stage 1: Removing low-variance features (threshold={self.variance_threshold})")
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)
        high_var_features = X.columns[selector.get_support()].tolist()
        print(f"      {len(selected_features)} -> {len(high_var_features)} features")
        selected_features = high_var_features
        
        # Stage 2: Remove highly correlated features
        print(f"   Stage 2: Removing correlated features (threshold={self.correlation_threshold})")
        X_filtered = X[selected_features]
        uncorrelated_features = self._remove_correlated_features(X_filtered)
        print(f"      {len(selected_features)} -> {len(uncorrelated_features)} features")
        selected_features = uncorrelated_features
        
        # Stage 3: Mutual Information selection
        if self.use_mutual_info:
            print(f"   Stage 3: Mutual information selection")
            X_filtered = X[selected_features]
            mi_scores = mutual_info_regression(X_filtered, y, random_state=42)
            mi_features = self._select_top_k_features(selected_features, mi_scores, self.n_features)
            print(f"      {len(selected_features)} -> {len(mi_features)} features")
            selected_features = mi_features
            self.feature_scores = dict(zip(mi_features, 
                                          sorted(mi_scores[mi_scores > 0], reverse=True)[:len(mi_features)]))
        
        # Stage 4: Model-based selection (optional refinement)
        if self.use_model_selection and len(selected_features) > self.n_features:
            print(f"   Stage 4: Model-based feature importance")
            X_filtered = X[selected_features]
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_filtered, y)
            importances = rf.feature_importances_
            model_features = self._select_top_k_features(selected_features, importances, self.n_features)
            print(f"      {len(selected_features)} -> {len(model_features)} features")
            selected_features = model_features
        
        self.selected_features = selected_features
        print(f"   ✓ Final selection: {len(self.selected_features)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        if self.selected_features is None:
            raise RuntimeError("Must call fit() before transform()")
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """Remove features with correlation > threshold."""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > self.correlation_threshold)]
        
        return [f for f in X.columns if f not in to_drop]
    
    def _select_top_k_features(self, features: List[str], scores: np.ndarray, k: int) -> List[str]:
        """Select top K features by score."""
        k = min(k, len(features))
        top_indices = np.argsort(scores)[-k:]
        return [features[i] for i in top_indices]
    
    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance scores if available."""
        return self.feature_scores


class DataPreprocessor:
    """
    Complete preprocessing pipeline combining all subtasks.
    """
    
    def __init__(self,
                 imputation_strategy: str = 'median',
                 outlier_method: str = 'isolation_forest',
                 outlier_contamination: float = 0.05,
                 remove_outliers: bool = False,
                 n_features: int = 200,
                 scale_features: bool = True):
        
        self.imputer = MissingValueImputer(strategy=imputation_strategy)
        self.outlier_detector = OutlierDetector(method=outlier_method, 
                                               contamination=outlier_contamination)
        self.feature_selector = FeatureSelector(n_features=n_features)
        self.scaler = RobustScaler() if scale_features else None
        self.remove_outliers = remove_outliers
        self.scale_features = scale_features
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Complete preprocessing pipeline for training data."""
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
        
        # Subtask 0: Imputation
        X_clean = self.imputer.fit_transform(X)
        
        # Subtask 1: Outlier detection
        inlier_mask = self.outlier_detector.detect(X_clean, y)
        if self.remove_outliers:
            X_clean, y = self.outlier_detector.remove_outliers(X_clean, y)
        
        # Subtask 2: Feature selection
        X_selected = self.feature_selector.fit_transform(X_clean, y)
        
        # Optional: Feature scaling
        if self.scale_features:
            print("📏 Scaling features using RobustScaler...")
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
            X_selected = X_scaled
        
        print("="*70)
        print(f"✓ Preprocessing complete: {X.shape[1]} -> {X_selected.shape[1]} features")
        print("="*70 + "\n")
        
        return X_selected, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to test data."""
        X_clean = self.imputer.transform(X)
        X_selected = self.feature_selector.transform(X_clean)
        
        if self.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
            X_selected = X_scaled
        
        return X_selected


if __name__ == "__main__":
    # Test preprocessing pipeline
    from dataloader import DataLoader
    
    print("Testing preprocessing pipeline...")
    loader = DataLoader()
    X_train, y_train = loader.load_train_data()
    
    preprocessor = DataPreprocessor(
        imputation_strategy='median',
        outlier_method='isolation_forest',
        outlier_contamination=0.05,
        remove_outliers=True,
        n_features=200
    )
    
    X_processed, y_processed = preprocessor.fit_transform(X_train, y_train)
    print(f"\nFinal shape: {X_processed.shape}")
    print(f"Final samples: {len(y_processed)}")
