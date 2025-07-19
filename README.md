# Trace Anomaly Detection System

A deep learning-based trace anomaly detection tool for distributed systems using Graph Attention Networks (GAT) with dual reconstruction loss.

## Features

- **Graph-based Structure**: Uses GAT to process graph structural features of traces
- **Dual Reconstruction Loss**: Combines structural reconstruction loss and feature reconstruction loss
- **Engineering Design**: Modular architecture, easy to extend and maintain
- **Multiple Interfaces**: Supports both Python API and command-line interface
- **Performance Evaluation**: Built-in evaluation tools and visualization features

## System Architecture

```
ad/
├── config.py          # Configuration classes
├── data_processor.py  # Data preprocessing
├── models.py          # Deep learning models
├── detector.py        # Main detector
├── evaluation.py      # Evaluation tools
├── utils.py           # Utility functions
├── cli.py             # Command line interface
├── train_with_parquet.py # Training script
└── README.md          # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torch-geometric scikit-learn pandas numpy tqdm joblib matplotlib
```

### 2. Use Python API

```python
from config import ModelConfig, TraceDataConfig
from detector import TraceAnomalyDetector

# Configure model
config = ModelConfig(
    epochs=150,
    learning_rate=0.005,
    latent_dim=16,
    loss_alpha=0.6
)

# Create detector
detector = TraceAnomalyDetector(config)

# Train model
detector.fit(train_df)

# Predict anomaly scores
scores = detector.predict_score(test_df)

# Save model
detector.save("./model")
```

### 3. Use Command Line Interface

```bash
# Train model
python cli.py train --data train_data.parquet --output ./model --epochs 150

# Predict anomaly scores
python cli.py predict --model ./model --data test_data.parquet --output results.csv

# Evaluate model performance
python cli.py evaluate --model ./model --normal-data normal.parquet --anomaly-data anomaly.parquet
```

### 4. Run Training Script

```bash
python train_with_parquet.py
```

## Data Format

Input data should contain the following columns:

### Required Fields
- `trace_id`: Trace ID
- `span_id`: Span ID  
- `parent_span_id`: Parent span ID
- `span_name`: Span name
- `service_name`: Service name
- `duration`: Duration

### Optional Fields
- `attr.http.request.method`: HTTP request method
- `attr.http.response.status_code`: HTTP response status code
- `attr.span_kind`: Span type
- `attr.k8s.pod.name`: Kubernetes Pod name
- `attr.k8s.service.name`: Kubernetes Service name
- `attr.k8s.namespace.name`: Kubernetes namespace
- Other custom attributes

## Model Configuration

```python
# Basic configuration
config = ModelConfig(
    # Feature configuration
    categorical_features=['primary_service', 'root_span'],
    numerical_features=['span_count', 'total_duration', 'avg_duration', 
                       'max_duration', 'min_duration', 'duration_std',
                       'unique_services', 'unique_spans', 'error_rate'],
    
    # Model parameters
    latent_dim=16,        # Latent space dimension
    gat_heads=4,          # Number of GAT attention heads
    
    # Training parameters
    epochs=100,           # Number of epochs
    learning_rate=0.005,  # Learning rate
    batch_size=1,         # Batch size
    loss_alpha=0.6,       # Loss weight
)

# Extended configuration
trace_config = TraceDataConfig(
    missing_value_strategy='fill',  # Missing value handling strategy
    categorical_fill_value='N/A',   # Categorical feature fill value
    numerical_fill_value=0.0,       # Numerical feature fill value
)
```

## Core Components

### 1. Data Processor (DataProcessor)
- Data cleaning and preprocessing
- Feature encoding and standardization
- Graph structure construction

### 2. Models
- **GATEncoder**: Graph Attention Network encoder
- **FeatureDecoder**: Feature reconstruction decoder
- **AdvancedGAE**: Combined graph autoencoder

### 3. Detector (TraceAnomalyDetector)
- Model training and prediction
- Model saving and loading
- Batch processing support

### 4. Evaluator (ModelEvaluator)
- Performance evaluation metrics
- Visualization analysis
- Evaluation report generation

## Evaluation Metrics

- **Separation Ratio**: Ratio of anomaly scores to normal scores
- **ROC AUC**: Area under the ROC curve
- **Score Distribution**: Distribution of normal and anomaly sample scores
- **Statistics**: Mean, standard deviation, percentiles, etc.

## Performance Optimization

1. **GPU Acceleration**: Automatic GPU detection and usage
2. **Early Stopping**: Prevents overfitting
3. **Learning Rate Scheduling**: Dynamic learning rate adjustment
4. **Batch Processing**: Supports large-scale data processing

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or latent_dim
2. **Training Not Converging**: Adjust learning_rate or loss_alpha
3. **Poor Separation**: Increase epochs or adjust model architecture
4. **Data Format Error**: Check if required fields exist

### Debugging Tips

```python
# Check model info
model_info = detector.get_model_info()
print(f"Model parameters: {model_info['model_parameters']}")

# Check feature info
feature_info = detector.processor.get_feature_info()
print(f"Feature info: {feature_info}")

# Verbose logging
detector.fit(train_df, verbose=True)
```

## License

MIT License
