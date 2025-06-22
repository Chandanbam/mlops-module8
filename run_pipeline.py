#!/usr/bin/env python3
"""
Simple script to run the complete Dask ML Pipeline.

This script demonstrates how to use the pipeline with different configurations.
"""

import sys
import os
import argparse
from src.pipeline import MLPipeline


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Run Dask ML Pipeline')
    parser.add_argument('--dataset', type=str, default='california_housing',
                       choices=['california_housing', 'synthetic'],
                       help='Dataset to use (default: california_housing)')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'linear'],
                       help='Model type to train (default: random_forest)')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Task type (default: regression)')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability analysis')
    parser.add_argument('--resource', action='store_true',
                       help='Run resource utilization analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DASK ML PIPELINE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Hyperparameter Tuning: {args.tune}")
    print(f"Scalability Analysis: {args.scalability}")
    print(f"Resource Analysis: {args.resource}")
    print("="*80)
    
    try:
        # Initialize pipeline
        pipeline = MLPipeline(args.config)
        
        # Run pipeline
        results = pipeline.run(
            dataset_type=args.dataset,
            model_type=args.model,
            task_type=args.task,
            tune_hyperparameters=args.tune,
            run_scalability_analysis=args.scalability,
            run_resource_analysis=args.resource
        )
        
        # Print summary
        pipeline.print_summary()
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results['performance_report_path']}")
        print(f"Visualizations saved to: {results['visualization_paths']}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 