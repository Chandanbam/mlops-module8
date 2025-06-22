"""
Model Training Module for Dask ML Pipeline

This module handles distributed machine learning model training using Dask-ML.
The objective is to train ML models using Dask-ML or integrate Dask with existing ML libraries.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split, GridSearchCV
from dask_ml.metrics import mean_squared_error, accuracy_score, r2_score
from dask_ml.wrappers import Incremental, ParallelPostFit
import yaml
import time
import psutil
import os
import joblib
from typing import Tuple, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn models to be wrapped with Dask-ML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Dask-ML is used for distributed machine learning training
# We wrap scikit-learn models with Dask-ML wrappers for distributed processing

class DaskModelTrainer:
    """Handles distributed machine learning model training using Dask-ML."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model trainer with configuration."""
        self.config = self._load_config(config_path)
        self.performance_metrics = {}
        self.models = {}
        self.training_history = {}
        
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
    
    def prepare_data_for_training(self, features: dd.DataFrame, target: dd.Series) -> Tuple[da.Array, da.Array]:
        """
        Prepare Dask DataFrames for model training using Dask-ML.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Tuple of (X, y) as Dask Arrays for distributed training
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print("Preparing data for Dask-ML model training...")
        
        # Convert to Dask Arrays for distributed training
        X = features.to_dask_array(lengths=True)
        y = target.to_dask_array(lengths=True)
        
        # Ensure proper shapes
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'data_preparation', start_time, initial_memory, final_memory,
            {'X_shape': X.shape, 'y_shape': y.shape, 'framework': 'dask-ml'}
        )
        
        print(f"Data prepared for Dask-ML training:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - Framework: Dask-ML")
        print(f"  - Preparation time: {time.time() - start_time:.2f} seconds")
        
        return X, y
    
    def train_random_forest(self, X: da.Array, y: da.Array, 
                           task_type: str = 'regression',
                           **kwargs) -> Union[Incremental, ParallelPostFit]:
        """
        Train a Random Forest model using Dask-ML wrappers for distributed training.
        
        Args:
            X: Feature array (Dask Array)
            y: Target array (Dask Array)
            task_type: 'regression' or 'classification'
            **kwargs: Additional model parameters
            
        Returns:
            Trained model wrapped with Dask-ML
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Training Random Forest for {task_type} task using Dask-ML...")
        
        # Get model parameters from config or use defaults
        model_params = self.config['model'].copy()
        model_params.update(kwargs)
        
        # Create base scikit-learn model
        if task_type == 'regression':
            base_model = RandomForestRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42),
                n_jobs=model_params.get('n_jobs', -1)
            )
        else:
            base_model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42),
                n_jobs=model_params.get('n_jobs', -1)
            )
        
        # Wrap with Dask-ML for distributed training
        # Use ParallelPostFit for models that can be trained on a subset and applied to all data
        dask_model = ParallelPostFit(base_model)
        
        # Train model using Dask-ML
        dask_model.fit(X, y)
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'random_forest_training', start_time, initial_memory, final_memory,
            {'task_type': task_type, 'n_estimators': model_params.get('n_estimators', 100),
             'max_depth': model_params.get('max_depth', 10), 'framework': 'dask-ml'}
        )
        
        print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")
        print(f"  - Framework: Dask-ML with ParallelPostFit wrapper")
        print(f"  - Model: {type(base_model).__name__}")
        
        return dask_model
    
    def train_linear_model(self, X: da.Array, y: da.Array,
                          task_type: str = 'regression') -> Union[Incremental, ParallelPostFit]:
        """
        Train a linear model using Dask-ML wrappers for distributed training.
        
        Args:
            X: Feature array (Dask Array)
            y: Target array (Dask Array)
            task_type: 'regression' or 'classification'
            
        Returns:
            Trained model wrapped with Dask-ML
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Training Linear model for {task_type} task using Dask-ML...")
        
        # Create base scikit-learn model
        if task_type == 'regression':
            base_model = LinearRegression()
        else:
            base_model = LogisticRegression(random_state=42)
        
        # Wrap with Dask-ML for distributed training
        # Use Incremental for models that can be trained incrementally
        dask_model = Incremental(base_model)
        
        # Train model using Dask-ML
        dask_model.fit(X, y)
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'linear_model_training', start_time, initial_memory, final_memory,
            {'task_type': task_type, 'framework': 'dask-ml'}
        )
        
        print(f"Linear model training completed in {time.time() - start_time:.2f} seconds")
        print(f"  - Framework: Dask-ML with Incremental wrapper")
        print(f"  - Model: {type(base_model).__name__}")
        
        return dask_model
    
    def hyperparameter_tuning(self, X: da.Array, y: da.Array,
                             model_type: str = 'random_forest',
                             task_type: str = 'regression') -> Any:
        """
        Perform hyperparameter tuning using Dask-ML GridSearchCV.
        
        Args:
            X: Feature array (Dask Array)
            y: Target array (Dask Array)
            model_type: Type of model to tune
            task_type: 'regression' or 'classification'
            
        Returns:
            Best model from grid search
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print(f"Performing hyperparameter tuning for {model_type} using Dask-ML...")
        
        # Define parameter grids
        if model_type == 'random_forest':
            if task_type == 'regression':
                base_model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            else:
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
        
        # Wrap with Dask-ML
        dask_model = ParallelPostFit(base_model)
        
        # Perform grid search using Dask-ML
        cv_folds = self.config['training']['cv_folds']
        scoring = self.config['training']['scoring']
        
        grid_search = GridSearchCV(
            dask_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        final_memory = self._get_memory_usage()
        self._record_performance(
            'hyperparameter_tuning', start_time, initial_memory, final_memory,
            {'model_type': model_type, 'task_type': task_type, 'cv_folds': cv_folds,
             'best_score': grid_search.best_score_, 'best_params': grid_search.best_params_,
             'framework': 'dask-ml'}
        )
        
        print(f"Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds")
        print(f"  - Framework: Dask-ML GridSearchCV")
        print(f"  - Best score: {grid_search.best_score_:.4f}")
        print(f"  - Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Any, X_test: da.Array, y_test: da.Array,
                      task_type: str = 'regression') -> Dict[str, float]:
        """
        Evaluate model performance using Dask-ML metrics.
        
        Args:
            model: Trained model (Dask-ML wrapped)
            X_test: Test features (Dask Array)
            y_test: Test targets (Dask Array)
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        print("Evaluating model performance using Dask-ML...")
        
        # Make predictions using Dask-ML
        y_pred = model.predict(X_test)
        
        # Calculate metrics using Dask-ML
        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': accuracy
            }
        
        evaluation_time = time.time() - start_time
        
        print(f"Model evaluation completed in {evaluation_time:.2f} seconds")
        print(f"  - Framework: Dask-ML metrics")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        return metrics
    
    def train_pipeline(self, features: dd.DataFrame, target: dd.Series,
                      model_type: str = 'random_forest',
                      task_type: str = 'regression',
                      tune_hyperparameters: bool = False) -> Tuple[Any, Dict[str, float]]:
        """
        Complete training pipeline using Dask-ML.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            model_type: Type of model to train
            task_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        print("Starting Dask-ML model training pipeline...")
        pipeline_start = time.time()
        
        # Prepare data for Dask-ML
        X, y = self.prepare_data_for_training(features, target)
        
        # Split data using Dask-ML
        test_size = self.config['dataset']['test_size']
        random_state = self.config['dataset']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split: Train {X_train.shape[0]:,} samples, Test {X_test.shape[0]:,} samples")
        print(f"  - Framework: Dask-ML train_test_split")
        
        # Train model
        if tune_hyperparameters:
            model = self.hyperparameter_tuning(X_train, y_train, model_type, task_type)
        else:
            if model_type == 'random_forest':
                model = self.train_random_forest(X_train, y_train, task_type)
            elif model_type == 'linear':
                model = self.train_linear_model(X_train, y_train, task_type)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test, task_type)
        
        # Store model and metrics
        self.models[model_type] = model
        self.training_history[model_type] = {
            'metrics': metrics,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'framework': 'dask-ml'
        }
        
        total_time = time.time() - pipeline_start
        print(f"Dask-ML training pipeline completed in {total_time:.2f} seconds")
        
        return model, metrics
    
    def save_model(self, model: Any, model_name: str, save_path: str = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name for the model
            save_path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if save_path is None:
            save_path = self.config['output']['models_dir']
        
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, f"{model_name}.joblib")
        
        joblib.dump(model, model_file)
        print(f"Model saved to: {model_file}")
        
        return model_file
    
    def load_model(self, model_name: str, load_path: str = None) -> Any:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of the model to load
            load_path: Path to load the model from
            
        Returns:
            Loaded model
        """
        if load_path is None:
            load_path = self.config['output']['models_dir']
        
        model_file = os.path.join(load_path, f"{model_name}.joblib")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = joblib.load(model_file)
        print(f"Model loaded from: {model_file}")
        
        return model
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from training operations."""
        return self.performance_metrics
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and model information."""
        return self.training_history


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from preprocessing import DaskPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    features, target = loader.load_dataset()
    
    preprocessor = DaskPreprocessor()
    processed_features, processed_target = preprocessor.preprocess_pipeline(features, target)
    
    # Train model
    trainer = DaskModelTrainer()
    model, metrics = trainer.train_pipeline(processed_features, processed_target)
    
    print("Model training completed successfully!") 