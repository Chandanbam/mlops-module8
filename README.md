# Scalable Machine Learning Pipeline with Dask

## Project Overview
This project demonstrates building a scalable machine learning pipeline using Dask for distributed computing. The pipeline is designed to handle large datasets efficiently and provides comprehensive performance analysis, visualization capabilities, and automated reporting.

## Features
- **Distributed Data Processing**: Uses Dask DataFrames for scalable data manipulation
- **Scalable ML Training**: Implements distributed machine learning with Dask-ML
- **Performance Monitoring**: Real-time performance tracking and memory usage analysis
- **Interactive Visualizations**: Dynamic dashboards using Plotly and Bokeh
- **Comprehensive Analysis**: Comparison between Dask and traditional approaches
- **Automated Reporting**: Generate Word documents, Markdown, and JSON reports
- **Test Coverage**: Comprehensive unit tests with detailed test reports
- **CLI Interface**: Easy-to-use command-line interface for pipeline execution

## Dataset
The pipeline uses the **California Housing Dataset** for demonstration, which can be easily scaled to larger datasets. The dataset includes:
- 20,640 samples (scalable up to 2M+ samples)
- 8 features (median income, housing age, etc.)
- Target: median house value

## Project Structure
```
mlops-module8/
├── src/
│   ├── data_loader.py          # Dataset loading and generation
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── model_training.py       # Distributed model training
│   ├── performance_analysis.py # Performance comparison and analysis
│   ├── visualization.py        # Interactive visualizations
│   └── pipeline.py             # Main pipeline orchestrator
├── notebooks/
│   ├── pipeline_demo.ipynb     # Main pipeline demonstration
│   └── performance_comparison.ipynb # Performance analysis
├── results/
│   ├── models/                 # Trained models
│   ├── plots/                  # Generated visualizations
│   └── reports/                # Performance reports (JSON, MD, DOCX)
├── config/
│   └── config.yaml            # Configuration parameters
├── tests/
│   └── test_pipeline.py       # Unit tests
├── run_pipeline.py            # CLI script
├── generate_report.py         # Word document report generator
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: At least 2GB free space
- **OS**: Linux, macOS, or Windows

### Dependencies
- Dask and Dask-ML for distributed computing
- Scikit-learn for traditional ML comparison
- Pandas and NumPy for data manipulation
- Matplotlib, Plotly, and Bokeh for visualization
- PyYAML for configuration management
- Pytest for testing

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd mlops-module8
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv myenv

# Activate virtual environment
# On Linux/macOS:
source myenv/bin/activate
# On Windows:
myenv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install additional packages for testing and reporting
pip install pytest python-docx matplotlib
```

### Step 4: Verify Installation
```bash
# Test basic imports
python -c "import dask, dask_ml, sklearn, pandas, numpy; print('All packages installed successfully!')"

# Run basic tests
PYTHONPATH=. pytest tests/test_pipeline.py::TestDataLoader::test_load_config -v
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Activate virtual environment
source myenv/bin/activate

# Run complete pipeline with default settings
python run_pipeline.py

# Check results
ls -la results/
```

### Option 2: Step-by-Step Execution
```bash
# 1. Load and preprocess data
python -c "
from src.pipeline import MLPipeline
pipeline = MLPipeline()
features, target = pipeline.load_data()
print(f'Loaded dataset: {len(features)} samples, {len(features.columns)} features')
"

# 2. Run complete pipeline
python -c "
from src.pipeline import MLPipeline
pipeline = MLPipeline()
results = pipeline.run()
print('Pipeline completed successfully!')
"
```

## Detailed Usage Guide

### Command Line Interface (CLI)

The pipeline provides a comprehensive CLI with multiple options:

#### Basic Pipeline Execution
```bash
# Run with default settings (California Housing dataset, scale factor 25)
python run_pipeline.py

# Run with specific dataset and scale factor
python run_pipeline.py --dataset california_housing --scale-factor 10

# Run with synthetic dataset
python run_pipeline.py --dataset synthetic --n-samples 10000
```

#### Advanced Pipeline Options
```bash
# Run with different model types
python run_pipeline.py --model sgd_regressor
python run_pipeline.py --model random_forest
python run_pipeline.py --model linear

# Run with hyperparameter tuning
python run_pipeline.py --tune

# Run with scalability analysis
python run_pipeline.py --scalability

# Run with resource utilization analysis
python run_pipeline.py --resource

# Run with all analyses enabled
python run_pipeline.py --tune --scalability --resource

# Use custom configuration file
python run_pipeline.py --config my_config.yaml
```

#### Complete Example
```bash
# Run complete pipeline with all features
python run_pipeline.py \
    --dataset california_housing \
    --scale-factor 25 \
    --model sgd_regressor \
    --tune \
    --scalability \
    --resource \
    --verbose
```

### Python API

#### Quick Start
```python
from src.pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# Run complete pipeline
results = pipeline.run()
print(f"Pipeline completed in {results['execution_time']:.2f} seconds")
```

#### Step-by-Step Execution
```python
from src.pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# 1. Load data
features, target = pipeline.load_data(dataset_type="california_housing", scale_factor=10)

# 2. Preprocess data
processed_features, processed_target = pipeline.preprocess_data(features, target)

# 3. Train model
model, metrics = pipeline.train_model(
    processed_features, 
    processed_target,
    model_type='sgd_regressor',
    task_type='regression',
    tune_hyperparameters=True
)

# 4. Analyze performance
comparison_results = pipeline.analyze_performance(processed_features, processed_target)

# 5. Create visualizations
visualization_paths = pipeline.create_visualizations(comparison_results)

# Print summary
pipeline.print_summary()
```

#### Advanced Usage with Custom Configuration
```python
from src.pipeline import MLPipeline

# Initialize with custom config
pipeline = MLPipeline(config_path="my_config.yaml")

# Run with specific options
results = pipeline.run(
    dataset_type="synthetic",
    model_type="sgd_regressor",
    task_type="regression",
    tune_hyperparameters=True,
    run_scalability_analysis=True,
    run_resource_analysis=True
)

# Access results
print(f"Model performance: {results['model_metrics']}")
print(f"Execution time: {results['execution_time']:.2f} seconds")
print(f"Visualizations: {results['visualization_paths']}")
```

### Jupyter Notebooks

Run the provided Jupyter notebooks for interactive exploration:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open notebooks/
# - pipeline_demo.ipynb: Complete pipeline demonstration
# - performance_comparison.ipynb: Detailed performance analysis
```

## Testing

### Running Tests
```bash
# Run all unit tests
PYTHONPATH=. pytest tests/ -v

# Run specific test file
PYTHONPATH=. pytest tests/test_pipeline.py -v

# Run with coverage
PYTHONPATH=. pytest tests/ --cov=src --cov-report=html

# Run tests and save output
PYTHONPATH=. pytest tests/test_pipeline.py -v --tb=short > test_report.txt
```

### Test Report Generation
```bash
# Generate tabular test report
python -c "
import subprocess
result = subprocess.run(['PYTHONPATH=.', 'pytest', 'tests/test_pipeline.py', '-v', '--tb=short'], 
                       capture_output=True, text=True)
print('Test Results:')
print(result.stdout)
"
```

## Reporting

### Generate Performance Reports

#### Word Document Report
```bash
# Generate Word document with tables and graphs
python generate_report.py
```

#### Custom Report Generation
```python
from generate_report import create_performance_report

# Generate report with custom data
create_performance_report()
```

### Report Types Available
- **JSON Report**: `results/reports/performance_report.json` - Machine-readable format
- **Markdown Report**: `results/reports/performance_report.md` - Human-readable format
- **Word Document**: `results/reports/performance_report.docx` - Professional format with tables and graphs

## Configuration

### Default Configuration
The pipeline uses `config/config.yaml` for configuration. Key parameters:

```yaml
dataset:
  name: california_housing
  scale_factor: 25  # Dataset size multiplier
  test_size: 0.2
  random_state: 42

dask:
  n_workers: 4
  threads_per_worker: 2
  memory_limit: 2GB

model:
  type: SGDRegressor
  n_estimators: 100
  max_depth: 10
  random_state: 42

training:
  cv_folds: 5
  scoring: neg_mean_squared_error
  verbose: true
```

### Custom Configuration
Create your own configuration file:

```bash
# Copy default config
cp config/config.yaml my_config.yaml

# Edit configuration
nano my_config.yaml

# Use custom config
python run_pipeline.py --config my_config.yaml
```

## Performance Analysis

The pipeline includes comprehensive performance analysis:

### Metrics Tracked
- **Memory Usage**: Real-time memory tracking and comparison
- **Processing Time**: Distributed vs. traditional approach comparison
- **Scalability**: Performance scaling with dataset size
- **Resource Utilization**: CPU and memory efficiency
- **Model Performance**: MSE, RMSE, R² scores

### Scalability Testing
```bash
# Test with different scale factors
python run_pipeline.py --scale-factor 1   # Original dataset
python run_pipeline.py --scale-factor 10  # 10x larger
python run_pipeline.py --scale-factor 25  # 25x larger
python run_pipeline.py --scale-factor 50  # 50x larger (if memory allows)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Set PYTHONPATH
export PYTHONPATH=.
# or
PYTHONPATH=. python your_script.py
```

#### 2. Memory Issues
```bash
# Error: Process killed due to memory constraints
# Solution: Reduce scale factor or increase memory limit
python run_pipeline.py --scale-factor 5  # Use smaller dataset
```

#### 3. Dask Worker Issues
```bash
# Error: Dask worker connection issues
# Solution: Check Dask cluster status
python -c "import dask.distributed; print(dask.distributed.Client())"
```

#### 4. Test Failures
```bash
# Error: Test failures in preprocessing
# Solution: Check Dask version compatibility
pip install "dask[complete]>=2023.1.0"
pip install "dask-ml>=2023.1.0"
```

### Performance Optimization

#### For Large Datasets
1. **Increase Dask workers**: Modify `n_workers` in config
2. **Adjust memory limits**: Set appropriate `memory_limit`
3. **Use chunked processing**: Configure `chunk_size` in data loading
4. **Enable disk spilling**: Set `local_directory` in Dask config

#### For Better Performance
1. **Use incremental models**: SGDRegressor with Dask-ML Incremental wrapper
2. **Enable parallel processing**: Set `n_jobs=-1` in model config
3. **Optimize chunk sizes**: Balance memory usage and parallelism

## Results and Outputs

### Generated Files
After running the pipeline, you'll find:

```
results/
├── models/
│   ├── dask_model.pkl          # Trained Dask model
│   └── traditional_model.pkl   # Trained traditional model
├── plots/
│   ├── performance_comparison.png    # Performance comparison plot
│   ├── interactive_dashboard.html    # Interactive dashboard
│   └── bokeh_dashboard.html          # Bokeh dashboard
└── reports/
    ├── performance_report.json       # JSON performance report
    ├── performance_report.md         # Markdown report
    └── performance_report.docx       # Word document report
```

### Understanding Results

#### Performance Comparison
- **Time Ratio**: Dask execution time / Traditional execution time
- **Memory Ratio**: Dask memory usage / Traditional memory usage
- **Speedup**: Traditional time / Dask time (values > 1 indicate Dask is faster)
- **Memory Efficiency**: Lower memory usage indicates better efficiency

#### Model Performance
- **MSE (Mean Squared Error)**: Lower is better
- **RMSE (Root Mean Squared Error)**: Lower is better
- **R² Score**: Higher is better (closer to 1.0)

## Key Components

### 1. Data Preprocessing (`src/preprocessing.py`)
- Feature scaling and normalization
- Missing value handling
- Feature engineering
- Distributed data transformations
- Polynomial feature generation
- Feature selection

### 2. Model Training (`src/model_training.py`)
- Distributed model training with Dask-ML
- Incremental learning with SGDRegressor
- Hyperparameter optimization
- Cross-validation with Dask
- Model persistence and loading

### 3. Performance Analysis (`src/performance_analysis.py`)
- Memory profiling and tracking
- Processing time comparison
- Scalability analysis across different dataset sizes
- Resource utilization monitoring
- Performance ratio calculations

### 4. Visualization (`src/visualization.py`)
- Interactive performance dashboards
- Real-time monitoring plots
- Model performance visualizations
- Scalability charts
- Memory usage plots

### 5. Pipeline Orchestration (`src/pipeline.py`)
- End-to-end pipeline execution
- Configuration management
- Result aggregation
- Error handling and logging

## Performance Benefits

- **Scalability**: Handles datasets larger than available RAM
- **Speed**: Parallel processing across multiple cores/nodes
- **Memory Efficiency**: Lazy evaluation and chunked processing
- **Fault Tolerance**: Automatic task retry and recovery
- **Resource Optimization**: Better CPU and memory utilization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests to ensure everything works (`PYTHONPATH=. pytest tests/ -v`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the test outputs for debugging information

## Acknowledgments

- Dask development team for the excellent distributed computing framework
- Scikit-learn team for the machine learning algorithms
- California Housing dataset providers
- Open source community for the supporting libraries