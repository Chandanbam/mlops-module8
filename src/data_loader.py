"""
Data Loader Module for Dask ML Pipeline

This module handles dataset loading, scaling, and preparation for distributed processing.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import yaml
import os
from typing import Tuple, Dict, Any
import time
import psutil


class DataLoader:
    """Handles dataset loading and preparation for distributed processing."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        self.config = self._load_config(config_path)
        self.performance_metrics = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def load_california_housing(self, scale_factor: int = 1) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Load California Housing dataset and scale it for testing.
        
        Args:
            scale_factor: Factor to multiply the dataset size
            
        Returns:
            Tuple of (features, target) as Dask DataFrames
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Loading California Housing dataset with scale factor: {scale_factor}")
        
        # Load original dataset
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target
        
        # Scale the dataset if needed
        if scale_factor > 1:
            print(f"Scaling dataset by factor {scale_factor}...")
            X_scaled = np.tile(X, (scale_factor, 1))
            y_scaled = np.tile(y, scale_factor)
            
            # Add some noise to make it more realistic
            noise_factor = 0.01
            X_scaled += np.random.normal(0, noise_factor, X_scaled.shape)
            y_scaled += np.random.normal(0, noise_factor, y_scaled.shape)
            
            X, y = X_scaled, y_scaled
        
        # Convert to pandas DataFrame first for easier manipulation
        feature_names = housing.feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=4)
        
        # Split features and target
        features = ddf.drop('target', axis=1)
        target = ddf['target']
        
        # Record performance metrics
        end_time = time.time()
        final_memory = self._get_memory_usage()
        
        self.performance_metrics['data_loading'] = {
            'time': end_time - start_time,
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'memory_increase': final_memory - initial_memory,
            'dataset_size': len(df),
            'n_features': len(feature_names)
        }
        
        print(f"Dataset loaded successfully:")
        print(f"  - Size: {len(df):,} samples")
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Loading time: {end_time - start_time:.2f} seconds")
        print(f"  - Memory usage: {final_memory:.2f} MB")
        
        return features, target
    
    def create_synthetic_large_dataset(self, n_samples: int = 100000, n_features: int = 20) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Create a synthetic large dataset for scalability testing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Tuple of (features, target) as Dask DataFrames
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Creating synthetic dataset: {n_samples:,} samples, {n_features} features")
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # Create a synthetic target with some non-linear relationships
        target = (
            0.3 * X[:, 0] + 
            0.5 * X[:, 1]**2 + 
            0.2 * X[:, 2] * X[:, 3] + 
            0.1 * np.random.randn(n_samples)
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = target
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=max(1, n_samples // 10000))
        
        # Split features and target
        features = ddf.drop('target', axis=1)
        target = ddf['target']
        
        # Record performance metrics
        end_time = time.time()
        final_memory = self._get_memory_usage()
        
        self.performance_metrics['synthetic_data_creation'] = {
            'time': end_time - start_time,
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'memory_increase': final_memory - initial_memory,
            'dataset_size': n_samples,
            'n_features': n_features
        }
        
        print(f"Synthetic dataset created successfully:")
        print(f"  - Size: {n_samples:,} samples")
        print(f"  - Features: {n_features}")
        print(f"  - Creation time: {end_time - start_time:.2f} seconds")
        print(f"  - Memory usage: {final_memory:.2f} MB")
        
        return features, target
    
    def split_data(self, features: dd.DataFrame, target: dd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[dd.DataFrame, dd.DataFrame, dd.Series, dd.Series]:
        """
        Split data into train and test sets using Dask.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        start_time = time.time()
        
        print("Splitting data into train and test sets...")
        
        # Add random column for splitting & apply to all partitions
        features_with_random = features.map_partitions(DataLoader.add_random_col, seed=42)
        
        # Split based on random column
        train_mask = features_with_random['random_col'] > test_size
        test_mask = features_with_random['random_col'] <= test_size
        
        X_train = features_with_random[train_mask].drop('random_col', axis=1)
        X_test = features_with_random[test_mask].drop('random_col', axis=1)
        
        # Split target accordingly
        y_train = target[train_mask.index]
        y_test = target[test_mask.index]
        
        end_time = time.time()
        
        print(f"Data split completed in {end_time - start_time:.2f} seconds")
        print(f"  - Train set: {len(X_train):,} samples")
        print(f"  - Test set: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from data loading operations."""
        return self.performance_metrics

    @staticmethod
    def add_random_col(df, seed=None):
        """Add random column to dataframe for splitting."""
        rng = np.random.default_rng(seed)
        df = df.copy()
        df['random_col'] = rng.random(len(df))
        return df
    
    def load_dataset(self, dataset_type: str = "california_housing") -> Tuple[dd.DataFrame, dd.Series]:
        """
        Main method to load dataset based on configuration.
        
        Args:
            dataset_type: Type of dataset to load
            
        Returns:
            Tuple of (features, target)
        """
        if dataset_type == "california_housing":
            scale_factor = self.config['dataset']['scale_factor']
            return self.load_california_housing(scale_factor)
        elif dataset_type == "synthetic":
            n_samples = self.config['dataset'].get('n_samples', 100000)
            n_features = self.config['dataset'].get('n_features', 20)
            return self.create_synthetic_large_dataset(n_samples, n_features)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    features, target = loader.load_dataset()
    print("Dataset loaded successfully!") 