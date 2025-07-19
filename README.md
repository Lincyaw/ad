# Trace Anomaly Detection

A microservice trace anomaly detection tool based on Graph Neural Networks. This project uses Graph Attention Networks (GAT) to perform anomaly detection on microservice trace data.

## Features

- Anomaly detection based on Graph Attention Network (GAT)
- Support for structured and attributed modeling of microservice trace data
- Automated model training and evaluation pipeline
- Rich evaluation metrics and visualization charts
- Support for multiple aggregation methods in prediction

## Dependencies

The project uses Python 3.13+ and depends on the following main libraries:

- PyTorch & PyTorch Geometric: Deep learning framework
- Pandas & NumPy: Data processing
- Scikit-learn: Machine learning evaluation
- Matplotlib: Visualization
- Typer: Command line interface

## Installation

Install dependencies using uv:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

## Data Format

Training and testing data should be in Parquet format and contain the following fields:

**Required fields:**
- `trace_id`: Trace ID
- `span_id`: Span ID  
- `parent_span_id`: Parent Span ID
- `span_name`: Span name
- `primary_service`: Primary service name
- `start_time`: Start time
- `duration`: Duration

**Aggregated feature fields:**
- `span_count`: Number of spans
- `total_duration`: Total duration
- `avg_duration`: Average duration
- `max_duration`: Maximum duration
- `min_duration`: Minimum duration
- `duration_std`: Duration standard deviation
- `unique_services`: Number of unique services
- `unique_spans`: Number of unique spans
- `error_rate`: Error rate
- `root_span`: Root span name

## Usage

### Model Training

Train anomaly detection model using training data:

```bash
# Basic training command
uv run python cli.py train --data data/training_data.parquet

# Custom training parameters
uv run python cli.py train \
    --data data/training_data.parquet \
    --output models \
    --epochs 150 \
    --learning-rate 0.001 \
    --latent-dim 32 \
    --batch-size 2
```

**Training parameter description:**
- `--data`: Training data file path (required)
- `--output`: Model output directory (default: models)
- `--epochs`: Number of training epochs (default: 100)
- `--learning-rate`: Learning rate (default: 0.005)
- `--latent-dim`: Latent space dimension (default: 16)
- `--batch-size`: Batch size (default: 1)

After training is complete, model files will be saved in the specified output directory:
- `model.pth`: Trained neural network model
- `processor.joblib`: Data preprocessor
- `config.joblib`: Model configuration
- `trace_config.joblib`: Data configuration

### Model Evaluation

Perform anomaly detection evaluation on test data:

```bash
# Basic evaluation command
uv run python cli.py evaluate

# Specify model and aggregation method
uv run python cli.py evaluate \
    --model models \
    --aggregation max \
    --output-dir evaluation_results
```

**Evaluation parameter description:**
- `--model`: Model directory path (default: models)
- `--aggregation`: Aggregation method, options: max, mean, percentile_95 (default: max)
- `--output-dir`: Evaluation results output directory (default: evaluation_results)

**Test data structure requirements:**
```
data/test/
├── dataset1/
│   ├── normal_traces.parquet
│   └── abnormal_traces.parquet
├── dataset2/
│   ├── normal_traces.parquet
│   └── abnormal_traces.parquet
└── ...
```

### Evaluation Results

After evaluation is complete, the following files will be generated:

1. **evaluation_results.json**: Detailed evaluation metrics
2. **confusion_matrix.png**: Confusion matrix chart
3. **roc_curve.png**: ROC curve chart
4. **precision_recall_curve.png**: Precision-recall curve chart
5. **score_distribution.png**: Anomaly score distribution chart

**Main evaluation metrics:**
- ROC AUC: Area under the receiver operating characteristic curve
- Precision: Proportion of predicted anomalies that are actually anomalous
- Recall: Proportion of actual anomalies that are correctly predicted
- F1 Score: Harmonic mean of precision and recall
- Specificity: Proportion of normal samples correctly identified

## Project Architecture

```
├── cli.py              # Command line interface
├── config.py           # Configuration class definitions
├── detector.py         # Main anomaly detector class
├── data_processor.py   # Data preprocessing module
├── models.py           # Neural network model definitions
├── evaluation.py       # Model evaluation module
├── utils.py            # Utility functions
├── pyproject.toml      # Project configuration
└── README.md           # Project documentation
```

## Model Principle

This project implements an anomaly detection method based on Graph Attention Networks:

1. **Data Preprocessing**: Convert trace data into graph structure, where nodes represent spans and edges represent call relationships
2. **Feature Engineering**: Extract categorical and numerical features, perform standardization
3. **Graph Encoding**: Use GAT encoder to learn latent representations of graph structure
4. **Reconstruction Decoding**: Reconstruct original features and graph structure through decoder
5. **Anomaly Detection**: Calculate anomaly scores based on reconstruction error

## Example Workflow

```bash
# 1. Train model
uv run python cli.py train --data data/normal_traces.parquet --epochs 200

# 2. Evaluate model
uv run python cli.py evaluate --aggregation max

# 3. View results
ls evaluation_results/
# Output: confusion_matrix.png, roc_curve.png, precision_recall_curve.png, ...
```

