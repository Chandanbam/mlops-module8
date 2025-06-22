"""
Unit Tests for Dask ML Pipeline

This module contains comprehensive unit tests for all pipeline components.
"""

import unittest
import numpy as np
import pandas as pd
import dask.dataframe as dd
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock

# Import pipeline components
import sys
sys.path.append('..')

from src.data_loader import DataLoader
from src.preprocessing import DaskPreprocessor
from src.model_training import DaskModelTrainer
from src.performance_analysis import PerformanceAnalyzer
from src.visualization import DaskVisualizer
from src.pipeline import MLPipeline


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'dataset': {
                'name': 'california_housing',
                'scale_factor': 1,
                'test_size': 0.2,
                'random_state': 42
            },
            'dask': {
                'n_workers': 2,
                'threads_per_worker': 1,
                'memory_limit': '1GB'
            },
            'model': {
                'type': 'RandomForestRegressor',
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'training': {
                'cv_folds': 3,
                'scoring': 'neg_mean_squared_error',
                'verbose': False
            },
            'output': {
                'models_dir': self.temp_dir,
                'plots_dir': self.temp_dir,
                'reports_dir': self.temp_dir
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.loader = DataLoader(self.config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config(self):
        """Test configuration loading."""
        config = self.loader.config
        self.assertIsInstance(config, dict)
        self.assertIn('dataset', config)
        self.assertIn('model', config)
    
    def test_load_california_housing(self):
        """Test California Housing dataset loading."""
        features, target = self.loader.load_california_housing(scale_factor=1)
        
        self.assertIsInstance(features, dd.DataFrame)
        self.assertIsInstance(target, dd.Series)
        self.assertGreater(len(features), 0)
        self.assertGreater(len(target), 0)
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        features, target = self.loader.create_synthetic_large_dataset(
            n_samples=1000, n_features=5
        )
        
        self.assertIsInstance(features, dd.DataFrame)
        self.assertIsInstance(target, dd.Series)
        self.assertEqual(len(features), 1000)
        self.assertEqual(len(features.columns), 5)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        # Load some data to generate metrics
        features, target = self.loader.load_california_housing(scale_factor=1)
        metrics = self.loader.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('data_loading', metrics)


class TestDaskPreprocessor(unittest.TestCase):
    """Test cases for DaskPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'dataset': {'scale_factor': 1},
            'output': {'plots_dir': self.temp_dir}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.preprocessor = DaskPreprocessor(self.config_path)
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        self.test_features = dd.from_pandas(df.drop('target', axis=1), npartitions=2)
        self.test_target = dd.from_pandas(df['target'], npartitions=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Add some missing values
        features_with_missing = self.test_features.copy()
        features_with_missing = features_with_missing.assign(
            feature_0=features_with_missing['feature_0'].where(
                features_with_missing.index % 10 != 0, np.nan
            )
        )
        
        result = self.preprocessor.handle_missing_values(features_with_missing)
        
        self.assertIsInstance(result, dd.DataFrame)
        self.assertEqual(len(result), len(features_with_missing))
    
    def test_scale_features(self):
        """Test feature scaling."""
        result = self.preprocessor.scale_features(self.test_features, method='standard')
        
        self.assertIsInstance(result, dd.DataFrame)
        self.assertEqual(len(result), len(self.test_features))
        self.assertEqual(len(result.columns), len(self.test_features.columns))
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        steps = {
            'handle_missing': True,
            'scale_features': True,
            'add_polynomial': False,
            'feature_selection': False
        }
        
        processed_features, processed_target = self.preprocessor.preprocess_pipeline(
            self.test_features, self.test_target, steps
        )
        
        self.assertIsInstance(processed_features, dd.DataFrame)
        self.assertIsInstance(processed_target, dd.Series)
        self.assertEqual(len(processed_features), len(self.test_features))


class TestDaskModelTrainer(unittest.TestCase):
    """Test cases for DaskModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'dataset': {'test_size': 0.2, 'random_state': 42},
            'model': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'training': {
                'cv_folds': 3,
                'scoring': 'neg_mean_squared_error'
            },
            'output': {'models_dir': self.temp_dir}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.trainer = DaskModelTrainer(self.config_path)
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        self.test_features = dd.from_pandas(df.drop('target', axis=1), npartitions=2)
        self.test_target = dd.from_pandas(df['target'], npartitions=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_data_for_training(self):
        """Test data preparation for training."""
        X, y = self.trainer.prepare_data_for_training(self.test_features, self.test_target)
        
        self.assertIsInstance(X, dd.array.Array)
        self.assertIsInstance(y, dd.array.Array)
        self.assertEqual(X.shape[0], y.shape[0])
    
    def test_train_random_forest(self):
        """Test Random Forest training."""
        X, y = self.trainer.prepare_data_for_training(self.test_features, self.test_target)
        
        model = self.trainer.train_random_forest(X, y, task_type='regression')
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))
    
    def test_train_pipeline(self):
        """Test complete training pipeline."""
        model, metrics = self.trainer.train_pipeline(
            self.test_features, self.test_target,
            model_type='random_forest',
            task_type='regression'
        )
        
        self.assertIsNotNone(model)
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
        self.assertIn('r2_score', metrics)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        X, y = self.trainer.prepare_data_for_training(self.test_features, self.test_target)
        model = self.trainer.train_random_forest(X, y, task_type='regression')
        
        # Save model
        model_path = self.trainer.save_model(model, 'test_model')
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = self.trainer.load_model('test_model')
        self.assertIsNotNone(loaded_model)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'dataset': {'test_size': 0.2, 'random_state': 42},
            'model': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'output': {'reports_dir': self.temp_dir}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.analyzer = PerformanceAnalyzer(self.config_path)
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        self.test_features = dd.from_pandas(df.drop('target', axis=1), npartitions=2)
        self.test_target = dd.from_pandas(df['target'], npartitions=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_traditional_ml_pipeline(self):
        """Test traditional ML pipeline."""
        X_numpy = self.test_features.compute().values
        y_numpy = self.test_target.compute().values
        
        results = self.analyzer.traditional_ml_pipeline(X_numpy, y_numpy)
        
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertIn('training_time', results)
        self.assertIn('memory_usage', results)
    
    def test_compare_pipelines(self):
        """Test pipeline comparison."""
        comparison_results = self.analyzer.compare_pipelines(
            self.test_features, self.test_target
        )
        
        self.assertIsInstance(comparison_results, dict)
        self.assertIn('traditional_pipeline', comparison_results)
        self.assertIn('dask_pipeline', comparison_results)
        self.assertIn('performance_comparison', comparison_results)
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        # Run comparison first
        self.analyzer.compare_pipelines(self.test_features, self.test_target)
        
        report_path = self.analyzer.generate_performance_report()
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith('.json'))


class TestDaskVisualizer(unittest.TestCase):
    """Test cases for DaskVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'output': {'plots_dir': self.temp_dir}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.visualizer = DaskVisualizer(self.config_path)
        
        # Create mock comparison results
        self.mock_comparison_results = {
            'traditional_pipeline': {
                'training_time': 10.0,
                'memory_usage': 500.0,
                'metrics': {'mse': 0.5, 'r2_score': 0.8}
            },
            'dask_pipeline': {
                'training_time': 8.0,
                'memory_usage': 400.0,
                'metrics': {'mse': 0.45, 'r2_score': 0.85}
            },
            'performance_comparison': {
                'mse': {'traditional': 0.5, 'dask': 0.45, 'improvement_percent': 10.0},
                'r2_score': {'traditional': 0.8, 'dask': 0.85, 'improvement_percent': 6.25}
            },
            'time_improvement_percent': 20.0,
            'memory_improvement_percent': 20.0
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_plot_performance_comparison(self):
        """Test performance comparison plotting."""
        plot_path = self.visualizer.plot_performance_comparison(
            self.mock_comparison_results, save_plot=True
        )
        
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        dashboard_path = self.visualizer.create_interactive_dashboard(
            self.mock_comparison_results
        )
        
        self.assertIsNotNone(dashboard_path)
        self.assertTrue(os.path.exists(dashboard_path))
        self.assertTrue(dashboard_path.endswith('.html'))
    
    def test_generate_all_visualizations(self):
        """Test generation of all visualizations."""
        visualization_paths = self.visualizer.generate_all_visualizations(
            self.mock_comparison_results
        )
        
        self.assertIsInstance(visualization_paths, dict)
        self.assertGreater(len(visualization_paths), 0)


class TestMLPipeline(unittest.TestCase):
    """Test cases for MLPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        config = {
            'dataset': {
                'name': 'california_housing',
                'scale_factor': 1,
                'test_size': 0.2,
                'random_state': 42
            },
            'dask': {
                'n_workers': 2,
                'threads_per_worker': 1,
                'memory_limit': '1GB'
            },
            'model': {
                'type': 'RandomForestRegressor',
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'training': {
                'cv_folds': 3,
                'scoring': 'neg_mean_squared_error',
                'verbose': False
            },
            'output': {
                'models_dir': self.temp_dir,
                'plots_dir': self.temp_dir,
                'reports_dir': self.temp_dir
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.pipeline = MLPipeline(self.config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.data_loader)
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.trainer)
        self.assertIsNotNone(self.pipeline.analyzer)
        self.assertIsNotNone(self.pipeline.visualizer)
    
    def test_load_data(self):
        """Test data loading step."""
        features, target = self.pipeline.load_data("california_housing")
        
        self.assertIsInstance(features, dd.DataFrame)
        self.assertIsInstance(target, dd.Series)
        self.assertGreater(len(features), 0)
    
    def test_preprocess_data(self):
        """Test data preprocessing step."""
        # First load data
        features, target = self.pipeline.load_data("california_housing")
        
        # Then preprocess
        processed_features, processed_target = self.pipeline.preprocess_data(features, target)
        
        self.assertIsInstance(processed_features, dd.DataFrame)
        self.assertIsInstance(processed_target, dd.Series)
    
    def test_train_model(self):
        """Test model training step."""
        # Load and preprocess data
        features, target = self.pipeline.load_data("california_housing")
        processed_features, processed_target = self.pipeline.preprocess_data(features, target)
        
        # Train model
        model, metrics = self.pipeline.train_model(
            processed_features, processed_target,
            model_type='random_forest',
            task_type='regression'
        )
        
        self.assertIsNotNone(model)
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
    
    @patch('src.performance_analysis.PerformanceAnalyzer.compare_pipelines')
    def test_analyze_performance(self, mock_compare):
        """Test performance analysis step."""
        # Mock the comparison results
        mock_compare.return_value = {
            'traditional_pipeline': {'training_time': 10.0, 'memory_usage': 500.0},
            'dask_pipeline': {'training_time': 8.0, 'memory_usage': 400.0}
        }
        
        # Load data
        features, target = self.pipeline.load_data("california_housing")
        
        # Analyze performance
        comparison_results = self.pipeline.analyze_performance(features, target)
        
        self.assertIsInstance(comparison_results, dict)
        mock_compare.assert_called_once()
    
    def test_get_results(self):
        """Test results retrieval."""
        results = self.pipeline.get_results()
        self.assertIsInstance(results, dict)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        metrics = self.pipeline.get_performance_metrics()
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 