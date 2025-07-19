#!/usr/bin/env python3
"""
Script for training with parquet data.
"""

import pandas as pd
from pathlib import Path

from config import ModelConfig, TraceDataConfig
from detector import TraceAnomalyDetector
from evaluation import ModelEvaluator
from utils import log_message, validate_trace_data


def main() -> bool:
    data_dir = Path("data/ts0-ts-admin-user-service-gc-n78v5d")
    normal_data_path = data_dir / "normal_traces.parquet"
    anomaly_data_path = data_dir / "abnormal_traces.parquet"

    output_dir = Path("models/trace_anomaly_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_message("Starting to load training data...")

    try:
        normal_df = pd.read_parquet(normal_data_path)
        log_message(f"Successfully loaded normal data, {len(normal_df)} records")
    except Exception as e:
        log_message(f"Failed to load normal data: {e}", "ERROR")
        return False

    if not validate_trace_data(normal_df):
        log_message("Normal data format validation failed", "ERROR")
        return False

    config = ModelConfig(
        epochs=50,
        learning_rate=0.005,
        latent_dim=16,
        batch_size=128,
    )

    trace_config = TraceDataConfig()

    log_message(f"Using categorical features: {config.categorical_features}")
    log_message(f"Using numerical features: {config.numerical_features}")

    # Create detector and train
    detector = TraceAnomalyDetector(config, trace_config)

    try:
        log_message("Starting model training...")
        detector.fit(normal_df, verbose=True)

        log_message("Saving model...")
        detector.save(str(output_dir))

        log_message("Model training completed!")

        # If anomaly data also exists, perform evaluation
        if anomaly_data_path.exists():
            log_message("Starting model evaluation...")
            evaluate_model(detector, normal_df, anomaly_data_path, output_dir)

        return True

    except Exception as e:
        log_message(f"Model training failed: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False


def evaluate_model(detector, normal_df, anomaly_data_path, output_dir) -> bool:
    """Evaluate model performance."""
    try:
        # Load anomaly data
        anomaly_df = pd.read_parquet(anomaly_data_path)
        log_message(f"Successfully loaded anomaly data, {len(anomaly_df)} records")

        # Validate anomaly data format
        if not validate_trace_data(anomaly_df):
            log_message("Anomaly data format validation failed", "ERROR")
            return False

        # Predict on normal data (sample to save time)
        normal_sample = normal_df.sample(min(1000, len(normal_df)), random_state=42)
        normal_results = detector.predict_score(normal_sample)
        normal_scores = [r["anomaly_score"] for r in normal_results]

        # Predict on anomaly data (sample to save time)
        anomaly_sample = anomaly_df.sample(min(1000, len(anomaly_df)), random_state=42)
        anomaly_results = detector.predict_score(anomaly_sample)
        anomaly_scores = [r["anomaly_score"] for r in anomaly_results]

        # Evaluate
        evaluator = ModelEvaluator()
        report = evaluator.generate_evaluation_report(normal_scores, anomaly_scores)

        log_message("Evaluation report:")
        print(report)

        # Save evaluation results
        eval_output = output_dir / "evaluation_results.json"
        evaluator.save_evaluation_results(str(eval_output))
        log_message(f"Evaluation results saved to {eval_output}")

        return True

    except Exception as e:
        log_message(f"Model evaluation failed: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
