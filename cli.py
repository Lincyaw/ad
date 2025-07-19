"""
Command line interface module.
"""

import argparse
import sys
import pandas as pd

from config import ModelConfig, TraceDataConfig
from detector import TraceAnomalyDetector
from evaluation import ModelEvaluator
from utils import log_message, validate_trace_data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trace anomaly detection tool")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--data", required=True, help="Training data file path (parquet format)"
    )
    train_parser.add_argument("--output", required=True, help="Model output directory")
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.005, help="Learning rate"
    )
    train_parser.add_argument(
        "--latent-dim", type=int, default=16, help="Latent space dimension"
    )
    train_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # Prediction command
    predict_parser = subparsers.add_parser("predict", help="Predict anomaly scores")
    predict_parser.add_argument("--model", required=True, help="Model directory path")
    predict_parser.add_argument(
        "--data", required=True, help="Prediction data file path (parquet format)"
    )
    predict_parser.add_argument("--output", help="Result output file path")

    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model", required=True, help="Model directory path")
    eval_parser.add_argument(
        "--normal-data", required=True, help="Normal data file path (parquet format)"
    )
    eval_parser.add_argument(
        "--anomaly-data", required=True, help="Anomaly data file path (parquet format)"
    )
    eval_parser.add_argument("--output", help="Evaluation result output file path")

    return parser.parse_args()


def train_command(args: argparse.Namespace) -> bool:
    log_message(f"Starting model training, data file: {args.data}")

    # Load data
    try:
        df = pd.read_parquet(args.data)
        log_message(f"Successfully loaded data, {len(df)} records")
    except Exception as e:
        log_message(f"Failed to load data: {e}", "ERROR")
        return False

    # Validate data format
    if not validate_trace_data(df):
        log_message("Data format validation failed", "ERROR")
        return False

    # Configure model
    config = ModelConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )

    trace_config = TraceDataConfig()

    # Create detector and train
    detector = TraceAnomalyDetector(config, trace_config)

    try:
        detector.fit(df, verbose=True)
        detector.save(args.output)
        log_message("Model training completed")
        return True
    except Exception as e:
        log_message(f"Model training failed: {e}", "ERROR")
        return False


def predict_command(args: argparse.Namespace) -> bool:
    """Handle prediction command."""
    log_message(f"Starting prediction, model: {args.model}, data: {args.data}")

    # Load model
    try:
        detector = TraceAnomalyDetector.load(args.model)
        log_message("Model loaded successfully")
    except Exception as e:
        log_message(f"Failed to load model: {e}", "ERROR")
        return False

    # Load data
    try:
        df = pd.read_parquet(args.data)
        log_message(f"Successfully loaded data, {len(df)} records")
    except Exception as e:
        log_message(f"Failed to load data: {e}", "ERROR")
        return False

    # Validate data format
    if not validate_trace_data(df):
        log_message("Data format validation failed", "ERROR")
        return False

    # Predict
    try:
        results = detector.predict_score(df)
        log_message(f"Prediction completed, {len(results)} results")

        # Output results
        if args.output:
            output_df = pd.DataFrame(results)
            output_df.to_csv(args.output, index=False)
            log_message(f"Results saved to {args.output}")
        else:
            # Print first 5 results
            for i, result in enumerate(results[:5]):
                log_message(f"Result {i + 1}: {result}")

        return True
    except Exception as e:
        log_message(f"Prediction failed: {e}", "ERROR")
        return False


def evaluate_command(args: argparse.Namespace) -> bool:
    """Handle evaluation command."""
    log_message(f"Starting evaluation, model: {args.model}")

    # Load model
    try:
        detector = TraceAnomalyDetector.load(args.model)
        log_message("Model loaded successfully")
    except Exception as e:
        log_message(f"Failed to load model: {e}", "ERROR")
        return False

    # Load normal data
    try:
        normal_df = pd.read_parquet(args.normal_data)
        log_message(f"Successfully loaded normal data, {len(normal_df)} records")
    except Exception as e:
        log_message(f"Failed to load normal data: {e}", "ERROR")
        return False

    # Load anomaly data
    try:
        anomaly_df = pd.read_parquet(args.anomaly_data)
        log_message(f"Successfully loaded anomaly data, {len(anomaly_df)} records")
    except Exception as e:
        log_message(f"Failed to load anomaly data: {e}", "ERROR")
        return False

    # Validate data format
    if not validate_trace_data(normal_df) or not validate_trace_data(anomaly_df):
        log_message("Data format validation failed", "ERROR")
        return False

    # Predict
    try:
        normal_results = detector.predict_score(normal_df)
        anomaly_results = detector.predict_score(anomaly_df)

        normal_scores = [r["anomaly_score"] for r in normal_results]
        anomaly_scores = [r["anomaly_score"] for r in anomaly_results]

        # Evaluate
        evaluator = ModelEvaluator()
        report = evaluator.generate_evaluation_report(normal_scores, anomaly_scores)

        print(report)

        # Save evaluation results
        if args.output:
            evaluator.save_evaluation_results(args.output)

        return True
    except Exception as e:
        log_message(f"Evaluation failed: {e}", "ERROR")
        return False


def main() -> None:
    """Main function."""
    args = parse_args()

    if args.command == "train":
        success = train_command(args)
    elif args.command == "predict":
        success = predict_command(args)
    elif args.command == "evaluate":
        success = evaluate_command(args)
    else:
        log_message("Please specify command: train, predict, or evaluate", "ERROR")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
