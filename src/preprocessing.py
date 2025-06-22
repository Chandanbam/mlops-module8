"""
Data Preprocessing Module for Dask ML Pipeline

This module handles data preprocessing operations using Dask DataFrames for scalable processing.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import yaml
import time
import psutil
import os
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class DaskPreprocessor:
    """Handles data preprocessing operations using Dask DataFrames."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.performance_metrics = {}
        self.scalers = {}
        self.imputers = {}
        

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _record_performance(self, operation: str, start_time: float, 
                          initial_memory: float, final_memory: float, 
                          additional_info: Dict[str, Any] = None):
        """Record performance metrics for an operation."""
        end_time = time.time()
        
        self.performance_metrics[operation] = {
            'time': end_time - start_time,
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'memory_increase': final_memory - initial_memory,
            **(additional_info or {})
        }
    
    def handle_missing_values(self, df: dd.DataFrame, strategy: str = 'mean') -> dd.DataFrame:
        """
        Handle missing values in Dask DataFrame.
        
        Args:
            df: Input Dask DataFrame
            strategy: Imputation strategy ('mean', 'median', 'constant')
            
        Returns:
            DataFrame with missing values handled
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Handling missing values using strategy: {strategy}")
        
        # Check for missing values
        missing_counts = df.isnull().sum().compute()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            print("No missing values found in the dataset.")
            return df
        
        print(f"Found {total_missing} missing values across {missing_counts[missing_counts > 0].count()} columns")
        
        # Handle missing values using Dask operations
        if strategy == 'mean':
            # Compute means for each column
            means = df.mean().compute()
            df_imputed = df.fillna(means)
        elif strategy == 'median':
            # Compute medians for each column
            medians = df.quantile(0.5).compute()
            df_imputed = df.fillna(medians)
        elif strategy == 'constant':
            # Fill with 0
            df_imputed = df.fillna(0)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'missing_value_handling', start_time, initial_memory, final_memory,
            {'strategy': strategy, 'missing_values_filled': total_missing}
        )
        
        print(f"Missing values handled in {time.time() - start_time:.2f} seconds")
        return df_imputed
    
    def scale_features(self, df: dd.DataFrame, method: str = 'standard') -> dd.DataFrame:
        """
        Scale features using Dask operations.
        
        Args:
            df: Input Dask DataFrame
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            Scaled DataFrame
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Scaling features using method: {method}")
        
        # Compute statistics for scaling
        if method == 'standard':
            # Z-score normalization
            means = df.mean().compute()
            stds = df.std().compute()
            df_scaled = (df - means) / stds
            self.scalers['standard'] = {'means': means, 'stds': stds}
            
        elif method == 'robust':
            # Robust scaling using median and IQR
            medians = df.quantile(0.5).compute()
            q75 = df.quantile(0.75).compute()
            q25 = df.quantile(0.25).compute()
            iqr = q75 - q25
            df_scaled = (df - medians) / iqr
            self.scalers['robust'] = {'medians': medians, 'iqr': iqr}
            
        elif method == 'minmax':
            # Min-max scaling
            mins = df.min().compute()
            maxs = df.max().compute()
            ranges = maxs - mins
            df_scaled = (df - mins) / ranges
            self.scalers['minmax'] = {'mins': mins, 'ranges': ranges}
            
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'feature_scaling', start_time, initial_memory, final_memory,
            {'method': method, 'n_features': len(df.columns)}
        )
        
        print(f"Features scaled in {time.time() - start_time:.2f} seconds")
        return df_scaled
    
    def add_polynomial_features(self, df: dd.DataFrame, degree: int = 2, 
                               include_bias: bool = False) -> dd.DataFrame:
        """
        Add polynomial features using Dask operations.
        
        Args:
            df: Input Dask DataFrame
            degree: Degree of polynomial features
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with polynomial features
        """
        if degree < 2:
            return df
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Adding polynomial features of degree {degree}")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.compute()
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for polynomial features")
            return df
        
        # Create polynomial features for pairs of columns
        poly_features = []
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                # Add interaction term
                interaction_name = f"{col1}_{col2}_interaction"
                poly_features.append((df[col1] * df[col2]).rename(interaction_name))
                
                # Add polynomial terms if degree > 2
                if degree >= 3:
                    poly2_name = f"{col1}_{col2}_poly2"
                    poly_features.append((df[col1] * df[col2]**2).rename(poly2_name))
                    
                    poly3_name = f"{col1}_{col2}_poly3"
                    poly_features.append((df[col1]**2 * df[col2]).rename(poly3_name))
        
        # Combine original features with polynomial features
        if poly_features:
            df_with_poly = dd.concat([df] + poly_features, axis=1)
        else:
            df_with_poly = df
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'polynomial_features', start_time, initial_memory, final_memory,
            {'degree': degree, 'original_features': len(df.columns), 
             'new_features': len(df_with_poly.columns) - len(df.columns)}
        )
        
        print(f"Polynomial features added in {time.time() - start_time:.2f} seconds")
        print(f"  - Original features: {len(df.columns)}")
        print(f"  - New features: {len(df_with_poly.columns) - len(df.columns)}")
        print(f"  - Total features: {len(df_with_poly.columns)}")
        
        return df_with_poly
    
    def feature_selection(self, df: dd.DataFrame, target: dd.Series, 
                         method: str = 'correlation', threshold: float = 0.1) -> dd.DataFrame:
        """
        Perform feature selection using Dask operations.
        
        Args:
            df: Feature DataFrame
            target: Target variable
            method: Selection method ('correlation', 'variance')
            threshold: Threshold for selection
            
        Returns:
            DataFrame with selected features
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Performing feature selection using method: {method}")
        
        if method == 'correlation':
            # Calculate correlations with target
            df_with_target = df.assign(target=target)
            correlations = df_with_target.corr().compute()
            target_correlations = correlations['target'].abs().drop('target')
            
            # Select features above threshold
            selected_features = target_correlations[target_correlations > threshold].index
            df_selected = df[selected_features]
            
        elif method == 'variance':
            # Calculate variance for each feature
            variances = df.var().compute()
            selected_features = variances[variances > threshold].index
            df_selected = df[selected_features]
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'feature_selection', start_time, initial_memory, final_memory,
            {'method': method, 'threshold': threshold, 
             'original_features': len(df.columns), 'selected_features': len(df_selected.columns)}
        )
        
        print(f"Feature selection completed in {time.time() - start_time:.2f} seconds")
        print(f"  - Original features: {len(df.columns)}")
        print(f"  - Selected features: {len(df_selected.columns)}")
        
        return df_selected
    
    def preprocess_pipeline(self, features: dd.DataFrame, target: dd.Series,
                          steps: Optional[Dict[str, Any]] = None) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            steps: Dictionary of preprocessing steps to apply
            
        Returns:
            Tuple of (processed_features, target)
        """
        if steps is None:
            steps = {
                'handle_missing': True,
                'scale_features': True,
                'add_polynomial': False,
                'feature_selection': False
            }
        
        print("Starting preprocessing pipeline...")
        start_time = time.time()
        
        processed_features = features.copy()
        
        # Handle missing values
        if steps.get('handle_missing', True):
            processed_features = self.handle_missing_values(processed_features)
        
        # Scale features
        if steps.get('scale_features', True):
            processed_features = self.scale_features(processed_features, method='standard')
        
        # Add polynomial features
        if steps.get('add_polynomial', False):
            processed_features = self.add_polynomial_features(processed_features, degree=2)
        
        # Feature selection
        if steps.get('feature_selection', False):
            processed_features = self.feature_selection(processed_features, target)
        
        total_time = time.time() - start_time
        print(f"Preprocessing pipeline completed in {total_time:.2f} seconds")
        
        return processed_features, target
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from preprocessing operations."""
        return self.performance_metrics
    
    def get_scalers(self) -> Dict[str, Any]:
        """Get fitted scalers for later use."""
        return self.scalers


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    features, target = loader.load_dataset()
    
    # Preprocess data
    preprocessor = DaskPreprocessor()
    processed_features, processed_target = preprocessor.preprocess_pipeline(features, target)
    
    print("Preprocessing completed successfully!") 