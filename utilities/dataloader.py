"""
Data loading utilities for Brain-Age Prediction project.
Handles loading of training, test data, and submission format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """
    Loads and manages datasets for brain-age prediction.
    
    Args:
        data_dir: Path to directory containing CSV files
    """
    
    def __init__(self, data_dir: str = "eth-aml-2025-project-1"):
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
    
    def _validate_data_dir(self):
        """Ensure all required files exist."""
        required_files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'sample.csv']
        for file in required_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"Required file not found: {file}")
    
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training features and labels.
        
        Returns:
            X_train: Training features (without 'id' column)
            y_train: Training labels (ages)
        """
        X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        y_train = pd.read_csv(self.data_dir / 'y_train.csv')
        
        # Store IDs for later reference
        self.train_ids = X_train['id'].values
        
        # Remove ID columns
        X_train = X_train.drop('id', axis=1)
        y_train = y_train['y'].values
        
        print(f"✓ Loaded training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Samples: {X_train.shape[0]}")
        print(f"  - Missing values: {X_train.isnull().sum().sum()}")
        print(f"  - Age range: [{y_train.min():.1f}, {y_train.max():.1f}]")
        
        return X_train, y_train
    
    def load_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load test features.
        
        Returns:
            X_test: Test features (without 'id' column)
            test_ids: Test sample IDs for submission
        """
        X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        
        # Store IDs for submission
        test_ids = X_test['id'].values
        X_test = X_test.drop('id', axis=1)
        
        print(f"✓ Loaded test data: X_test shape {X_test.shape}")
        print(f"  - Features: {X_test.shape[1]}")
        print(f"  - Samples: {X_test.shape[0]}")
        print(f"  - Missing values: {X_test.isnull().sum().sum()}")
        
        return X_test, test_ids
    
    def load_sample_submission(self) -> pd.DataFrame:
        """Load sample submission format."""
        return pd.read_csv(self.data_dir / 'sample.csv')
    
    def create_submission(self, test_ids: np.ndarray, predictions: np.ndarray, 
                         filename: str = 'submission.csv') -> pd.DataFrame:
        """
        Create submission file in correct format.
        
        Args:
            test_ids: Test sample IDs
            predictions: Predicted ages
            filename: Output filename
            
        Returns:
            Submission DataFrame
        """
        submission = pd.DataFrame({
            'id': test_ids,
            'y': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"✓ Created submission file: {filename}")
        print(f"  - Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        return submission
    
    def get_data_summary(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> dict:
        """
        Get comprehensive data summary statistics.
        
        Args:
            X: Feature matrix
            y: Optional labels
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'missing_values': X.isnull().sum().sum(),
            'missing_per_feature': X.isnull().sum().to_dict(),
            'feature_names': X.columns.tolist(),
        }
        
        if y is not None:
            summary.update({
                'age_mean': float(np.mean(y)),
                'age_std': float(np.std(y)),
                'age_min': float(np.min(y)),
                'age_max': float(np.max(y)),
            })
        
        return summary


def quick_load(data_dir: str = "eth-aml-2025-project-1") -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Quick function to load all data at once.
    
    Returns:
        X_train, y_train, X_test, test_ids
    """
    loader = DataLoader(data_dir)
    X_train, y_train = loader.load_train_data()
    X_test, test_ids = loader.load_test_data()
    return X_train, y_train, X_test, test_ids


if __name__ == "__main__":
    # Test data loading
    print("=" * 70)
    print("Testing DataLoader")
    print("=" * 70)
    
    loader = DataLoader()
    X_train, y_train = loader.load_train_data()
    X_test, test_ids = loader.load_test_data()
    
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    summary = loader.get_data_summary(X_train, y_train)
    for key, value in summary.items():
        if key not in ['missing_per_feature', 'feature_names']:
            print(f"{key}: {value}")
