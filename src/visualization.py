"""
Visualization Module for Dask ML Pipeline

This module provides interactive visualizations and dashboards for performance analysis
and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from bokeh.plotting import figure, show, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Legend
import yaml
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DaskVisualizer:
    """Handles visualization of pipeline performance and results."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration."""
        self.config = self._load_config(config_path)
        self.plots_dir = self.config['output']['plots_dir']
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def plot_performance_comparison(self, comparison_results: Dict[str, Any], 
                                  save_plot: bool = True) -> str:
        """
        Create performance comparison plots.
        
        Args:
            comparison_results: Results from performance comparison
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        traditional = comparison_results['traditional_pipeline']
        dask = comparison_results['dask_pipeline']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dask vs Traditional ML Pipeline Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Training time comparison
        times = [traditional['training_time'], dask['training_time']]
        labels = ['Traditional', 'Dask']
        colors = ['#ff7f0e', '#1f77b4']
        
        axes[0, 0].bar(labels, times, color=colors, alpha=0.7)
        axes[0, 0].set_title('Training Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(times):
            axes[0, 0].text(i, v + max(times) * 0.01, f'{v:.2f}s', 
                          ha='center', va='bottom', fontweight='bold')
        
        # Memory usage comparison
        memory_usage = [traditional['memory_usage'], dask['memory_usage']]
        
        axes[0, 1].bar(labels, memory_usage, color=colors, alpha=0.7)
        axes[0, 1].set_title('Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(memory_usage):
            axes[0, 1].text(i, v + max(memory_usage) * 0.01, f'{v:.1f}MB', 
                          ha='center', va='bottom', fontweight='bold')
        
        # Model performance metrics
        traditional_metrics = traditional['metrics']
        dask_metrics = dask['metrics']
        
        metrics_names = list(traditional_metrics.keys())
        traditional_values = [traditional_metrics[m] for m in metrics_names]
        dask_values = [dask_metrics[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, traditional_values, width, label='Traditional', 
                      color='#ff7f0e', alpha=0.7)
        axes[1, 0].bar(x + width/2, dask_values, width, label='Dask', 
                      color='#1f77b4', alpha=0.7)
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([m.upper() for m in metrics_names])
        axes[1, 0].legend()
        
        # Improvement percentages
        improvements = []
        for metric in metrics_names:
            traditional_val = traditional_metrics[metric]
            dask_val = dask_metrics[metric]
            
            if metric in ['mse', 'rmse']:  # Lower is better
                improvement = ((traditional_val - dask_val) / traditional_val) * 100
            else:  # Higher is better
                improvement = ((dask_val - traditional_val) / traditional_val) * 100
            
            improvements.append(improvement)
        
        colors_improvement = ['green' if x > 0 else 'red' for x in improvements]
        axes[1, 1].bar(metrics_names, improvements, color=colors_improvement, alpha=0.7)
        axes[1, 1].set_title('Performance Improvement (%)')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(improvements):
            axes[1, 1].text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', 
                          ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.plots_dir, 'performance_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {plot_path}")
        
        plt.show()
        return plot_path if save_plot else None
    
    def plot_scalability_analysis(self, scalability_results: Dict[str, Any], 
                                save_plot: bool = True) -> str:
        """
        Create scalability analysis plots.
        
        Args:
            scalability_results: Results from scalability analysis
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        scale_factors = []
        traditional_times = []
        dask_times = []
        traditional_memory = []
        dask_memory = []
        dataset_sizes = []
        
        for scale_factor, results in scalability_results.items():
            if 'error' not in results:
                scale_factors.append(scale_factor)
                traditional_times.append(results['traditional']['training_time'])
                dask_times.append(results['dask']['training_time'])
                traditional_memory.append(results['traditional']['memory_usage'])
                dask_memory.append(results['dask']['memory_usage'])
                dataset_sizes.append(results['dataset_size'])
        
        if not scale_factors:
            print("No valid scalability results to plot")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Analysis: Dask vs Traditional', 
                    fontsize=16, fontweight='bold')
        
        # Training time scaling
        axes[0, 0].plot(scale_factors, traditional_times, 'o-', label='Traditional', 
                       color='#ff7f0e', linewidth=2, markersize=8)
        axes[0, 0].plot(scale_factors, dask_times, 's-', label='Dask', 
                       color='#1f77b4', linewidth=2, markersize=8)
        axes[0, 0].set_title('Training Time Scaling')
        axes[0, 0].set_xlabel('Scale Factor')
        axes[0, 0].set_ylabel('Training Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage scaling
        axes[0, 1].plot(scale_factors, traditional_memory, 'o-', label='Traditional', 
                       color='#ff7f0e', linewidth=2, markersize=8)
        axes[0, 1].plot(scale_factors, dask_memory, 's-', label='Dask', 
                       color='#1f77b4', linewidth=2, markersize=8)
        axes[0, 1].set_title('Memory Usage Scaling')
        axes[0, 1].set_xlabel('Scale Factor')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Speedup ratio
        speedup_ratios = [t/d for t, d in zip(traditional_times, dask_times)]
        axes[1, 0].plot(scale_factors, speedup_ratios, 'o-', 
                       color='#2ca02c', linewidth=2, markersize=8)
        axes[1, 0].set_title('Speedup Ratio (Traditional/Dask)')
        axes[1, 0].set_xlabel('Scale Factor')
        axes[1, 0].set_ylabel('Speedup Ratio')
        axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory efficiency ratio
        memory_ratios = [t/d for t, d in zip(traditional_memory, dask_memory)]
        axes[1, 1].plot(scale_factors, memory_ratios, 'o-', 
                       color='#d62728', linewidth=2, markersize=8)
        axes[1, 1].set_title('Memory Efficiency Ratio (Traditional/Dask)')
        axes[1, 1].set_xlabel('Scale Factor')
        axes[1, 1].set_ylabel('Memory Ratio')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.plots_dir, 'scalability_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Scalability analysis plot saved to: {plot_path}")
        
        plt.show()
        return plot_path if save_plot else None
    
    def plot_resource_utilization(self, utilization_results: Dict[str, Any], 
                                save_plot: bool = True) -> str:
        """
        Create resource utilization plots.
        
        Args:
            utilization_results: Results from resource utilization analysis
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        cpu_readings = utilization_results['cpu_utilization']['readings']
        memory_readings = utilization_results['memory_utilization']['readings']
        
        if not cpu_readings or not memory_readings:
            print("No resource utilization data to plot")
            return None
        
        # Create time axis
        time_points = np.arange(len(cpu_readings)) * 0.5  # 0.5 second intervals
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Resource Utilization During Dask Pipeline Execution', 
                    fontsize=16, fontweight='bold')
        
        # CPU utilization over time
        axes[0].plot(time_points, cpu_readings, color='#1f77b4', linewidth=2)
        axes[0].set_title('CPU Utilization Over Time')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=utilization_results['cpu_utilization']['average'], 
                       color='red', linestyle='--', alpha=0.7, 
                       label=f"Average: {utilization_results['cpu_utilization']['average']:.1f}%")
        axes[0].legend()
        
        # Memory utilization over time
        axes[1].plot(time_points, memory_readings, color='#ff7f0e', linewidth=2)
        axes[1].set_title('Memory Utilization Over Time')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Memory Usage (MB)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=utilization_results['memory_utilization']['average'], 
                       color='red', linestyle='--', alpha=0.7, 
                       label=f"Average: {utilization_results['memory_utilization']['average']:.1f} MB")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.plots_dir, 'resource_utilization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Resource utilization plot saved to: {plot_path}")
        
        plt.show()
        return plot_path if save_plot else None
    
    def create_interactive_dashboard(self, comparison_results: Dict[str, Any],
                                   scalability_results: Dict[str, Any] = None,
                                   utilization_results: Dict[str, Any] = None) -> str:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            comparison_results: Results from performance comparison
            scalability_results: Results from scalability analysis
            utilization_results: Results from resource utilization analysis
            
        Returns:
            Path to saved dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Time Comparison', 'Memory Usage Comparison',
                          'Model Performance Metrics', 'Performance Improvements',
                          'Scalability Analysis', 'Resource Utilization'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training time comparison
        traditional = comparison_results['traditional_pipeline']
        dask = comparison_results['dask_pipeline']
        
        fig.add_trace(
            go.Bar(x=['Traditional', 'Dask'], 
                  y=[traditional['training_time'], dask['training_time']],
                  name='Training Time', marker_color=['#ff7f0e', '#1f77b4']),
            row=1, col=1
        )
        
        # Memory usage comparison
        fig.add_trace(
            go.Bar(x=['Traditional', 'Dask'], 
                  y=[traditional['memory_usage'], dask['memory_usage']],
                  name='Memory Usage', marker_color=['#ff7f0e', '#1f77b4']),
            row=1, col=2
        )
        
        # Model performance metrics
        traditional_metrics = traditional['metrics']
        dask_metrics = dask['metrics']
        metrics_names = list(traditional_metrics.keys())
        
        fig.add_trace(
            go.Bar(x=metrics_names, y=[traditional_metrics[m] for m in metrics_names],
                  name='Traditional', marker_color='#ff7f0e'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics_names, y=[dask_metrics[m] for m in metrics_names],
                  name='Dask', marker_color='#1f77b4'),
            row=2, col=1
        )
        
        # Performance improvements
        improvements = []
        for metric in metrics_names:
            traditional_val = traditional_metrics[metric]
            dask_val = dask_metrics[metric]
            
            if metric in ['mse', 'rmse']:
                improvement = ((traditional_val - dask_val) / traditional_val) * 100
            else:
                improvement = ((dask_val - traditional_val) / traditional_val) * 100
            
            improvements.append(improvement)
        
        colors_improvement = ['green' if x > 0 else 'red' for x in improvements]
        fig.add_trace(
            go.Bar(x=metrics_names, y=improvements, 
                  marker_color=colors_improvement, name='Improvement %'),
            row=2, col=2
        )
        
        # Scalability analysis (if available)
        if scalability_results:
            scale_factors = []
            speedup_ratios = []
            
            for scale_factor, results in scalability_results.items():
                if 'error' not in results:
                    scale_factors.append(scale_factor)
                    traditional_time = results['traditional']['training_time']
                    dask_time = results['dask']['training_time']
                    speedup_ratios.append(traditional_time / dask_time)
            
            if scale_factors:
                fig.add_trace(
                    go.Scatter(x=scale_factors, y=speedup_ratios, 
                              mode='lines+markers', name='Speedup Ratio',
                              line=dict(color='#2ca02c', width=3)),
                    row=3, col=1
                )
        
        # Resource utilization (if available)
        if utilization_results and utilization_results['cpu_utilization']['readings']:
            cpu_readings = utilization_results['cpu_utilization']['readings']
            time_points = list(range(len(cpu_readings)))
            
            fig.add_trace(
                go.Scatter(x=time_points, y=cpu_readings, 
                          mode='lines', name='CPU Usage %',
                          line=dict(color='#1f77b4', width=2)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Dask ML Pipeline Performance Dashboard",
            showlegend=True,
            height=1200,
            width=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Pipeline", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Pipeline", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_xaxes(title_text="Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Metrics", row=2, col=2)
        fig.update_yaxes(title_text="Improvement (%)", row=2, col=2)
        fig.update_xaxes(title_text="Scale Factor", row=3, col=1)
        fig.update_yaxes(title_text="Speedup Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=2)
        fig.update_yaxes(title_text="CPU Usage (%)", row=3, col=2)
        
        # Save dashboard
        dashboard_path = os.path.join(self.plots_dir, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        print(f"Interactive dashboard saved to: {dashboard_path}")
        
        return dashboard_path
    
    def create_bokeh_dashboard(self, comparison_results: Dict[str, Any]) -> str:
        """
        Create a Bokeh dashboard for real-time monitoring.
        
        Args:
            comparison_results: Results from performance comparison
            
        Returns:
            Path to saved dashboard
        """
        traditional = comparison_results['traditional_pipeline']
        dask = comparison_results['dask_pipeline']
        
        # Training time comparison
        p1 = figure(x_range=['Traditional', 'Dask'], 
                   title='Training Time Comparison',
                   width=400, height=300)
        p1.vbar(x=['Traditional', 'Dask'], 
               top=[traditional['training_time'], dask['training_time']],
               width=0.5, color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        p1.yaxis.axis_label = 'Time (seconds)'
        
        # Memory usage comparison
        p2 = figure(x_range=['Traditional', 'Dask'], 
                   title='Memory Usage Comparison',
                   width=400, height=300)
        p2.vbar(x=['Traditional', 'Dask'], 
               top=[traditional['memory_usage'], dask['memory_usage']],
               width=0.5, color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        p2.yaxis.axis_label = 'Memory (MB)'
        
        # Model performance metrics
        traditional_metrics = traditional['metrics']
        dask_metrics = dask['metrics']
        metrics_names = list(traditional_metrics.keys())
        
        p3 = figure(x_range=metrics_names, 
                   title='Model Performance Metrics',
                   width=400, height=300)
        
        traditional_values = [traditional_metrics[m] for m in metrics_names]
        dask_values = [dask_metrics[m] for m in metrics_names]
        
        p3.vbar(x=[x - 0.2 for x in range(len(metrics_names))], 
               top=traditional_values, width=0.4, 
               color='#ff7f0e', alpha=0.7, legend_label='Traditional')
        p3.vbar(x=[x + 0.2 for x in range(len(metrics_names))], 
               top=dask_values, width=0.4, 
               color='#1f77b4', alpha=0.7, legend_label='Dask')
        
        p3.xaxis.major_label_orientation = 45
        p3.yaxis.axis_label = 'Score'
        
        # Layout
        layout = row(column(p1, p2), p3)
        
        # Save dashboard
        dashboard_path = os.path.join(self.plots_dir, 'bokeh_dashboard.html')
        save(layout, dashboard_path)
        print(f"Bokeh dashboard saved to: {dashboard_path}")
        
        return dashboard_path
    
    def generate_all_visualizations(self, comparison_results: Dict[str, Any],
                                  scalability_results: Dict[str, Any] = None,
                                  utilization_results: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate all visualizations and return paths.
        
        Args:
            comparison_results: Results from performance comparison
            scalability_results: Results from scalability analysis
            utilization_results: Results from resource utilization analysis
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualization_paths = {}
        
        # Generate static plots
        if comparison_results:
            visualization_paths['performance_comparison'] = self.plot_performance_comparison(
                comparison_results, save_plot=True)
        
        if scalability_results:
            visualization_paths['scalability_analysis'] = self.plot_scalability_analysis(
                scalability_results, save_plot=True)
        
        if utilization_results:
            visualization_paths['resource_utilization'] = self.plot_resource_utilization(
                utilization_results, save_plot=True)
        
        # Generate interactive dashboards
        if comparison_results:
            visualization_paths['interactive_dashboard'] = self.create_interactive_dashboard(
                comparison_results, scalability_results, utilization_results)
            visualization_paths['bokeh_dashboard'] = self.create_bokeh_dashboard(
                comparison_results)
        
        print(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths


if __name__ == "__main__":
    # Example usage
    from performance_analysis import PerformanceAnalyzer
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
    
    # Create visualizations
    visualizer = DaskVisualizer()
    visualization_paths = visualizer.generate_all_visualizations(comparison_results)
    
    print("Visualizations generated successfully!") 