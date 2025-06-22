"""
Main Pipeline Orchestrator for Dask ML Pipeline

This module orchestrates the complete machine learning pipeline using Dask.
"""

import os
import time
import yaml
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .preprocessing import DaskPreprocessor
from .model_training import DaskModelTrainer
from .performance_analysis import PerformanceAnalyzer
from .visualization import DaskVisualizer


class MLPipeline:
    """Main orchestrator for the Dask ML pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the ML pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.results = {}
        self.performance_metrics = {}
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.preprocessor = DaskPreprocessor(config_path)
        self.trainer = DaskModelTrainer(config_path)
        self.analyzer = PerformanceAnalyzer(config_path)
        self.visualizer = DaskVisualizer(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self, dataset_type: str = "california_housing") -> Tuple[Any, Any]:
        """
        Load dataset using the data loader.
        
        Args:
            dataset_type: Type of dataset to load
            
        Returns:
            Tuple of (features, target)
        """
        print("="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        features, target = self.data_loader.load_dataset(dataset_type)
        
        # Store performance metrics
        self.performance_metrics['data_loading'] = self.data_loader.get_performance_metrics()
        
        return features, target
    
    def preprocess_data(self, features: Any, target: Any, 
                       steps: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
        """
        Preprocess data using the preprocessor.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            steps: Preprocessing steps to apply
            
        Returns:
            Tuple of (processed_features, target)
        """
        print("="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        processed_features, processed_target = self.preprocessor.preprocess_pipeline(
            features, target, steps
        )
        
        # Store performance metrics
        self.performance_metrics['preprocessing'] = self.preprocessor.get_performance_metrics()
        
        return processed_features, processed_target
    
    def train_model(self, features: Any, target: Any,
                   model_type: str = 'random_forest',
                   task_type: str = 'regression',
                   tune_hyperparameters: bool = False) -> Tuple[Any, Dict[str, float]]:
        """
        Train model using the trainer.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            model_type: Type of model to train
            task_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        print("="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        model, metrics = self.trainer.train_pipeline(
            features, target, model_type, task_type, tune_hyperparameters
        )
        
        # Store performance metrics
        self.performance_metrics['training'] = self.trainer.get_performance_metrics()
        
        return model, metrics
    
    def analyze_performance(self, features: Any, target: Any) -> Dict[str, Any]:
        """
        Analyze performance using the analyzer.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Dictionary with performance analysis results
        """
        print("="*60)
        print("STEP 4: PERFORMANCE ANALYSIS")
        print("="*60)
        
        comparison_results = self.analyzer.compare_pipelines(features, target)
        
        # Store performance metrics
        self.performance_metrics['analysis'] = self.analyzer.get_performance_metrics()
        
        return comparison_results
    
    def create_visualizations(self, comparison_results: Dict[str, Any],
                            scalability_results: Optional[Dict[str, Any]] = None,
                            utilization_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create visualizations using the visualizer.
        
        Args:
            comparison_results: Results from performance comparison
            scalability_results: Results from scalability analysis
            utilization_results: Results from resource utilization analysis
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        print("="*60)
        print("STEP 5: VISUALIZATION")
        print("="*60)
        
        visualization_paths = self.visualizer.generate_all_visualizations(
            comparison_results, scalability_results, utilization_results
        )
        
        return visualization_paths
    
    def run(self, dataset_type: str = "california_housing",
            model_type: str = 'random_forest',
            task_type: str = 'regression',
            tune_hyperparameters: bool = False,
            run_scalability_analysis: bool = False,
            run_resource_analysis: bool = False) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.
        
        Args:
            dataset_type: Type of dataset to use
            model_type: Type of model to train
            task_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform hyperparameter tuning
            run_scalability_analysis: Whether to run scalability analysis
            run_resource_analysis: Whether to run resource utilization analysis
            
        Returns:
            Dictionary with all pipeline results
        """
        print("="*80)
        print("STARTING DASK ML PIPELINE")
        print("="*80)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Load data
            features, target = self.load_data(dataset_type)
            
            # Step 2: Preprocess data
            processed_features, processed_target = self.preprocess_data(features, target)
            
            # Step 3: Train model
            model, metrics = self.train_model(
                processed_features, processed_target, model_type, task_type, tune_hyperparameters
            )
            
            # Step 4: Analyze performance
            comparison_results = self.analyze_performance(processed_features, processed_target)
            
            # Step 5: Create visualizations
            visualization_paths = self.create_visualizations(comparison_results)
            
            # Additional analyses
            scalability_results = None
            utilization_results = None
            
            if run_scalability_analysis:
                print("\nRunning scalability analysis...")
                scalability_results = self.analyzer.scalability_analysis(
                    features, target, scale_factors=[1, 2, 5]
                )
                
                # Add scalability visualizations
                scalability_viz = self.visualizer.plot_scalability_analysis(
                    scalability_results, save_plot=True
                )
                if scalability_viz:
                    visualization_paths['scalability_analysis'] = scalability_viz
            
            if run_resource_analysis:
                print("\nRunning resource utilization analysis...")
                utilization_results = self.analyzer.resource_utilization_analysis(
                    processed_features, processed_target
                )
                
                # Add resource utilization visualizations
                utilization_viz = self.visualizer.plot_resource_utilization(
                    utilization_results, save_plot=True
                )
                if utilization_viz:
                    visualization_paths['resource_utilization'] = utilization_viz
            
            # Generate performance report
            report_path = self.analyzer.generate_performance_report()
            
            # Compile results
            total_time = time.time() - pipeline_start
            
            self.results = {
                'pipeline_execution_time': total_time,
                'model_metrics': metrics,
                'comparison_results': comparison_results,
                'scalability_results': scalability_results,
                'utilization_results': utilization_results,
                'visualization_paths': visualization_paths,
                'performance_report_path': report_path,
                'performance_metrics': self.performance_metrics,
                'model': model,
                'config': self.config
            }
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Model performance: {metrics}")
            print(f"Visualizations saved to: {self.config['output']['plots_dir']}")
            print(f"Performance report saved to: {report_path}")
            print("="*80)
            
            return self.results
            
        except Exception as e:
            print(f"\nPipeline failed with error: {str(e)}")
            raise
    
    def save_model(self, model_name: str = "dask_ml_model") -> str:
        """
        Save the trained model.
        
        Args:
            model_name: Name for the model
            
        Returns:
            Path to saved model
        """
        if 'model' not in self.results:
            raise ValueError("No model available. Run the pipeline first.")
        
        model_path = self.trainer.save_model(self.results['model'], model_name)
        return model_path
    
    def load_model(self, model_name: str = "dask_ml_model") -> Any:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model = self.trainer.load_model(model_name)
        return model
    
    def get_results(self) -> Dict[str, Any]:
        """Get all pipeline results."""
        return self.results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return self.performance_metrics
    
    def print_summary(self):
        """Print a summary of the pipeline results."""
        if not self.results:
            print("No results available. Run the pipeline first.")
            return
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Execution Time: {self.results['pipeline_execution_time']:.2f} seconds")
        print(f"Model Type: {self.config['model']['type']}")
        print(f"Dataset Size: {self.results['comparison_results']['traditional_pipeline']['dataset_size']:,} samples")
        
        print("\nMODEL PERFORMANCE:")
        for metric, value in self.results['model_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nPERFORMANCE COMPARISON:")
        comparison = self.results['comparison_results']
        print(f"  Time Improvement: {comparison['time_improvement_percent']:.1f}%")
        print(f"  Memory Improvement: {comparison['memory_improvement_percent']:.1f}%")
        print(f"  Overall Winner: {comparison['summary']['overall_winner']}")
        
        print(f"\nVisualizations: {len(self.results['visualization_paths'])} generated")
        print(f"Performance Report: {self.results['performance_report_path']}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    pipeline = MLPipeline()
    
    # Run complete pipeline
    results = pipeline.run(
        dataset_type="california_housing",
        model_type="random_forest",
        task_type="regression",
        tune_hyperparameters=False,
        run_scalability_analysis=True,
        run_resource_analysis=True
    )
    
    # Print summary
    pipeline.print_summary()
    
    print("Pipeline execution completed!") 