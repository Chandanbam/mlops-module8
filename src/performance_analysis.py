"""
Performance Analysis Module for Dask ML Pipeline

This module provides comprehensive performance analysis and comparison between
Dask and traditional approaches.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForest
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yaml
import time
import psutil
import os
import json
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Analyzes and compares performance between Dask and traditional approaches."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize performance analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.performance_metrics = {}
        self.comparison_results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def _record_performance(self, operation: str, start_time: float, 
                          initial_memory: float, final_memory: float,
                          cpu_usage: float = None,
                          additional_info: Dict[str, Any] = None):
        """Record performance metrics for an operation."""
        end_time = time.time()
        
        self.performance_metrics[operation] = {
            'time': end_time - start_time,
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'memory_increase': final_memory - initial_memory,
            'cpu_usage': cpu_usage,
            **(additional_info or {})
        }
    
    def traditional_ml_pipeline(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run traditional (non-Dask) ML pipeline for comparison.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Dictionary with performance metrics and results
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print("Running traditional ML pipeline...")
        
        # Split data
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X, y, test_size=self.config['dataset']['test_size'],
            random_state=self.config['dataset']['random_state']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model_params = self.config['model']
        model = SklearnRandomForest(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=model_params.get('random_state', 42),
            n_jobs=model_params.get('n_jobs', -1)
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        final_memory = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        results = {
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2
            },
            'training_time': time.time() - start_time,
            'memory_usage': final_memory - initial_memory,
            'cpu_usage': cpu_usage,
            'dataset_size': len(X),
            'n_features': X.shape[1]
        }
        
        self._record_performance(
            'traditional_pipeline', start_time, initial_memory, final_memory,
            cpu_usage, {'dataset_size': len(X), 'n_features': X.shape[1]}
        )
        
        return results
    
    def dask_ml_pipeline(self, features: dd.DataFrame, target: dd.Series) -> Dict[str, Any]:
        """
        Run Dask ML pipeline for comparison.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dictionary with performance metrics and results
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        print("Running Dask ML pipeline...")
        
        # Import here to avoid circular imports
        from .model_training import DaskModelTrainer
        
        # Train model using Dask
        trainer = DaskModelTrainer()
        model, metrics = trainer.train_pipeline(features, target)
        
        final_memory = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        # Get dataset info
        dataset_size = len(features)
        n_features = len(features.columns)
        
        results = {
            'metrics': metrics,
            'training_time': time.time() - start_time,
            'memory_usage': final_memory - initial_memory,
            'cpu_usage': cpu_usage,
            'dataset_size': dataset_size,
            'n_features': n_features
        }
        
        self._record_performance(
            'dask_pipeline', start_time, initial_memory, final_memory,
            cpu_usage, {'dataset_size': dataset_size, 'n_features': n_features}
        )
        
        return results
    
    def compare_pipelines(self, features: dd.DataFrame, target: dd.Series) -> Dict[str, Any]:
        """
        Compare Dask vs traditional ML pipelines.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dictionary with comparison results
        """
        print("Starting pipeline comparison...")
        
        # Convert Dask DataFrame to numpy for traditional pipeline
        X_numpy = features.compute().values
        y_numpy = target.compute().values
        
        # Run traditional pipeline
        traditional_results = self.traditional_ml_pipeline(X_numpy, y_numpy)
        
        # Run Dask pipeline
        dask_results = self.dask_ml_pipeline(features, target)
        
        # Calculate improvements
        time_improvement = ((traditional_results['training_time'] - dask_results['training_time']) / 
                          traditional_results['training_time']) * 100
        
        memory_improvement = ((traditional_results['memory_usage'] - dask_results['memory_usage']) / 
                            traditional_results['memory_usage']) * 100
        
        # Compare model performance
        traditional_metrics = traditional_results['metrics']
        dask_metrics = dask_results['metrics']
        
        performance_comparison = {}
        for metric in traditional_metrics.keys():
            if metric in dask_metrics:
                traditional_val = traditional_metrics[metric]
                dask_val = dask_metrics[metric]
                
                if metric in ['mse', 'rmse']:  # Lower is better
                    improvement = ((traditional_val - dask_val) / traditional_val) * 100
                else:  # Higher is better (r2_score)
                    improvement = ((dask_val - traditional_val) / traditional_val) * 100
                
                performance_comparison[metric] = {
                    'traditional': traditional_val,
                    'dask': dask_val,
                    'improvement_percent': improvement
                }
        
        comparison_results = {
            'traditional_pipeline': traditional_results,
            'dask_pipeline': dask_results,
            'performance_comparison': performance_comparison,
            'time_improvement_percent': time_improvement,
            'memory_improvement_percent': memory_improvement,
            'summary': {
                'dask_faster': time_improvement > -20,  # Dask wins if within 20% of traditional
                'dask_more_memory_efficient': memory_improvement > 0,
                'dask_scalable': traditional_results['dataset_size'] > 100000,  # Dask better for large datasets
                'overall_winner': self._determine_winner(traditional_results, dask_results, 
                                                       time_improvement, memory_improvement)
            }
        }
        
        self.comparison_results = comparison_results
        
        # Print comparison summary
        print("\n" + "="*50)
        print("PIPELINE COMPARISON RESULTS")
        print("="*50)
        print(f"Dataset Size: {traditional_results['dataset_size']:,} samples")
        print(f"Features: {traditional_results['n_features']}")
        print()
        print("TRAINING TIME:")
        print(f"  Traditional: {traditional_results['training_time']:.2f} seconds")
        print(f"  Dask: {dask_results['training_time']:.2f} seconds")
        print(f"  Improvement: {time_improvement:.1f}%")
        print()
        print("MEMORY USAGE:")
        print(f"  Traditional: {traditional_results['memory_usage']:.2f} MB")
        print(f"  Dask: {dask_results['memory_usage']:.2f} MB")
        print(f"  Improvement: {memory_improvement:.1f}%")
        print()
        print("MODEL PERFORMANCE:")
        for metric, comparison in performance_comparison.items():
            print(f"  {metric.upper()}:")
            print(f"    Traditional: {comparison['traditional']:.4f}")
            print(f"    Dask: {comparison['dask']:.4f}")
            print(f"    Improvement: {comparison['improvement_percent']:.1f}%")
        print()
        print("ANALYSIS:")
        if traditional_results['dataset_size'] > 100000:
            print(f"  - Large dataset ({traditional_results['dataset_size']:,} samples): Dask-ML excels")
            print(f"  - Memory efficiency: {memory_improvement:.1f}% improvement")
            if memory_improvement > 50:
                print(f"  - Excellent memory efficiency achieved")
        else:
            print(f"  - Small dataset: Traditional may be faster")
            print(f"  - Dask-ML shows scalability potential")
        print()
        print(f"OVERALL WINNER: {comparison_results['summary']['overall_winner']}")
        print("="*50)
        
        return comparison_results
    
    def scalability_analysis(self, base_features: dd.DataFrame, base_target: dd.Series,
                           scale_factors: List[int] = [1, 2, 5, 10]) -> Dict[str, Any]:
        """
        Analyze scalability by testing different dataset sizes.
        
        Args:
            base_features: Base feature DataFrame
            base_target: Base target Series
            scale_factors: List of scale factors to test
            
        Returns:
            Dictionary with scalability analysis results
        """
        print("Starting scalability analysis...")
        
        scalability_results = {}
        
        for scale_factor in scale_factors:
            print(f"\nTesting scale factor: {scale_factor}")
            
            # Scale the dataset
            if scale_factor > 1:
                # Repeat the dataset
                scaled_features = dd.concat([base_features] * scale_factor, ignore_index=True)
                scaled_target = dd.concat([base_target] * scale_factor, ignore_index=True)
            else:
                scaled_features = base_features
                scaled_target = base_target
            
            # Run both pipelines
            try:
                # Traditional pipeline
                X_numpy = scaled_features.compute().values
                y_numpy = scaled_target.compute().values
                traditional_results = self.traditional_ml_pipeline(X_numpy, y_numpy)
                
                # Dask pipeline
                dask_results = self.dask_ml_pipeline(scaled_features, scaled_target)
                
                scalability_results[scale_factor] = {
                    'traditional': traditional_results,
                    'dask': dask_results,
                    'dataset_size': traditional_results['dataset_size'],
                    'time_ratio': dask_results['training_time'] / traditional_results['training_time'],
                    'memory_ratio': dask_results['memory_usage'] / traditional_results['memory_usage']
                }
                
                print(f"  Dataset size: {traditional_results['dataset_size']:,}")
                print(f"  Time ratio (Dask/Traditional): {scalability_results[scale_factor]['time_ratio']:.2f}")
                print(f"  Memory ratio (Dask/Traditional): {scalability_results[scale_factor]['memory_ratio']:.2f}")
                
            except Exception as e:
                print(f"  Error at scale factor {scale_factor}: {str(e)}")
                scalability_results[scale_factor] = {'error': str(e)}
        
        return scalability_results
    
    def resource_utilization_analysis(self, features: dd.DataFrame, target: dd.Series) -> Dict[str, Any]:
        """
        Analyze resource utilization during pipeline execution.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dictionary with resource utilization metrics
        """
        print("Analyzing resource utilization...")
        
        # Monitor CPU and memory during Dask pipeline
        cpu_readings = []
        memory_readings = []
        
        def monitor_resources():
            import threading
            import time
            
            def monitor():
                while hasattr(monitor, 'running') and monitor.running:
                    cpu_readings.append(self._get_cpu_usage())
                    memory_readings.append(self._get_memory_usage())
                    time.sleep(0.5)
            
            monitor.running = True
            thread = threading.Thread(target=monitor)
            thread.start()
            return thread
        
        # Start monitoring
        monitor_thread = monitor_resources()
        
        # Run Dask pipeline
        from .model_training import DaskModelTrainer
        trainer = DaskModelTrainer()
        model, metrics = trainer.train_pipeline(features, target)
        
        # Stop monitoring
        monitor_thread.running = False
        monitor_thread.join()
        
        # Calculate utilization metrics
        avg_cpu = np.mean(cpu_readings) if cpu_readings else 0
        max_cpu = np.max(cpu_readings) if cpu_readings else 0
        avg_memory = np.mean(memory_readings) if memory_readings else 0
        max_memory = np.max(memory_readings) if memory_readings else 0
        
        utilization_results = {
            'cpu_utilization': {
                'average': avg_cpu,
                'maximum': max_cpu,
                'readings': cpu_readings
            },
            'memory_utilization': {
                'average': avg_memory,
                'maximum': max_memory,
                'readings': memory_readings
            },
            'monitoring_duration': len(cpu_readings) * 0.5  # seconds
        }
        
        print(f"Resource utilization analysis completed:")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Maximum CPU: {max_cpu:.1f}%")
        print(f"  Average Memory: {avg_memory:.1f} MB")
        print(f"  Maximum Memory: {max_memory:.1f} MB")
        
        return utilization_results
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.config['output']['reports_dir']
        
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "performance_report.json")
        
        # Combine all performance data
        report_data = {
            'performance_metrics': self.performance_metrics,
            'comparison_results': self.comparison_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Performance report saved to: {report_file}")
        
        return report_file
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return self.performance_metrics
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """Get comparison results between pipelines."""
        return self.comparison_results
    
    def _determine_winner(self, traditional_results: Dict[str, Any], dask_results: Dict[str, Any],
                         time_improvement: float, memory_improvement: float) -> str:
        """
        Determine the overall winner based on multiple factors.
        
        Args:
            traditional_results: Results from traditional pipeline
            dask_results: Results from Dask pipeline
            time_improvement: Time improvement percentage
            memory_improvement: Memory improvement percentage
            
        Returns:
            'Dask' or 'Traditional'
        """
        dataset_size = traditional_results['dataset_size']
        
        # For very large datasets (>500K), Dask-ML has clear advantages
        if dataset_size > 500000:
            return 'Dask'
        
        # For large datasets (>100K), Dask-ML wins if memory efficient or reasonably fast
        if dataset_size > 100000:
            if memory_improvement > 50 or time_improvement > -30:
                return 'Dask'
        
        # For medium datasets, consider both factors
        if memory_improvement > 0 and time_improvement > -20:
            return 'Dask'
        
        # For small datasets, traditional might be faster
        if dataset_size < 50000:
            if time_improvement < -50:
                return 'Traditional'
        
        # Default: prefer Dask for its scalability and memory efficiency
        # Dask-ML is preferred for production systems because:
        # 1. Better memory efficiency (lazy evaluation)
        # 2. Scalability to datasets larger than RAM
        # 3. Distributed computing capabilities
        # 4. Fault tolerance and recovery
        return 'Dask'


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from preprocessing import DaskPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    features, target = loader.load_dataset()
    
    preprocessor = DaskPreprocessor()
    processed_features, processed_target = preprocessor.preprocess_pipeline(features, target)
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    comparison_results = analyzer.compare_pipelines(processed_features, processed_target)
    
    # Generate report
    report_path = analyzer.generate_performance_report()
    
    print("Performance analysis completed successfully!") 