"""
Evaluation module for model performance evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import log_message, calculate_statistics


class ModelEvaluator:
    """Model evaluator for anomaly detection performance."""

    def __init__(self) -> None:
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate_with_threshold(
        self, scores: List[float], labels: List[int], threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance with binary classification threshold."""
        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        if threshold is None:
            fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]

        # Binary predictions based on threshold
        predictions = (scores_arr >= threshold).astype(int)

        # Calculate metrics
        precision = precision_score(labels_arr, predictions)
        recall = recall_score(labels_arr, predictions)
        f1 = f1_score(labels_arr, predictions)

        # Confusion matrix
        cm = confusion_matrix(labels_arr, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # ROC AUC
        try:
            auc_score = roc_auc_score(labels_arr, scores_arr)
        except ValueError:
            auc_score = None

        results = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "auc_score": auc_score,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "classification_report": classification_report(
                labels_arr, predictions, output_dict=True
            ),
        }

        self.evaluation_results = results
        return results

    def load_labeled_data(
        self, file_paths: List[str], labels: List[int], detector=None
    ) -> Tuple[List[float], List[int]]:
        """Load data from multiple files with corresponding labels."""
        all_scores = []
        all_labels = []

        for file_path, label in zip(file_paths, labels):
            try:
                import pandas as pd

                df = pd.read_parquet(file_path)

                # Use detector to get anomaly scores if provided
                if detector is not None:
                    # Convert DataFrame to the format expected by detector
                    traces = df.to_dict("records")
                    scores = []
                    for trace in traces:
                        try:
                            score = detector.detect_anomaly(trace)
                            scores.append(score)
                        except Exception as e:
                            log_message(
                                f"Error detecting anomaly for trace: {e}", "WARNING"
                            )
                            scores.append(0.0)  # Default score on error
                else:
                    # If no detector provided, use placeholder scores
                    log_message(
                        "No detector provided, using placeholder scores", "WARNING"
                    )
                    scores = [0.0] * len(df)  # Placeholder

                all_scores.extend(scores)
                all_labels.extend([label] * len(df))

            except Exception as e:
                log_message(f"Error loading {file_path}: {e}", "ERROR")

        return all_scores, all_labels

    def evaluate_anomaly_detection(
        self, normal_scores: List[float], anomaly_scores: List[float]
    ) -> Dict[str, Any]:
        """Evaluate anomaly detection performance."""
        # Calculate statistics
        normal_stats = calculate_statistics(normal_scores)
        anomaly_stats = calculate_statistics(anomaly_scores)

        # Calculate separation ratio
        separation_ratio = (
            np.mean(anomaly_scores) / np.mean(normal_scores)
            if np.mean(normal_scores) > 0
            else 0
        )

        # Calculate ROC AUC (if labels available)
        y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
        y_scores = normal_scores + anomaly_scores

        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_score = None

        results = {
            "normal_stats": normal_stats,
            "anomaly_stats": anomaly_stats,
            "separation_ratio": separation_ratio,
            "auc_score": auc_score,
            "num_normal": len(normal_scores),
            "num_anomaly": len(anomaly_scores),
        }

        self.evaluation_results = results
        return results

    def plot_score_distribution(
        self,
        normal_scores: List[float],
        anomaly_scores: List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot score distribution graphs."""
        plt.figure(figsize=(12, 8))

        # Distribution plot
        plt.subplot(2, 2, 1)
        plt.hist(normal_scores, bins=30, alpha=0.7, label="Normal", color="blue")
        plt.hist(anomaly_scores, bins=30, alpha=0.7, label="Anomaly", color="red")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.title("Score Distribution")
        plt.legend()

        # Box plot
        plt.subplot(2, 2, 2)
        data = [normal_scores, anomaly_scores]
        labels = ["Normal", "Anomaly"]
        plt.boxplot(data)
        plt.xticks([1, 2], labels)
        plt.ylabel("Anomaly Score")
        plt.title("Score Box Plot")

        # ROC curve
        plt.subplot(2, 2, 3)
        y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
        y_scores = normal_scores + anomaly_scores

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
        except ValueError:
            plt.text(0.5, 0.5, "Unable to compute ROC curve", ha="center", va="center")

        # PR curve
        plt.subplot(2, 2, 4)
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
        except ValueError:
            plt.text(0.5, 0.5, "Unable to compute PR curve", ha="center", va="center")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            log_message(f"Score distribution plot saved to {save_path}")

        plt.show()

    def analyze_trace_complexity(
        self, trace_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trace complexity features."""
        complexities = []
        struct_errors = []
        feature_errors = []

        for result in trace_results:
            # Add more complexity calculation logic here
            complexity = result.get("num_spans", 1)  # Simple example
            complexities.append(complexity)
            struct_errors.append(result.get("struct_error", 0))
            feature_errors.append(result.get("feature_error", 0))

        return {
            "complexity_stats": calculate_statistics(complexities),
            "struct_error_stats": calculate_statistics(struct_errors),
            "feature_error_stats": calculate_statistics(feature_errors),
            "correlation_complexity_struct": np.corrcoef(complexities, struct_errors)[
                0, 1
            ]
            if len(complexities) > 1
            else 0,
            "correlation_complexity_feature": np.corrcoef(complexities, feature_errors)[
                0, 1
            ]
            if len(complexities) > 1
            else 0,
        }

    def save_evaluation_results(self, filepath: str) -> None:
        """Save evaluation results."""
        import json

        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        converted_results = convert_numpy(self.evaluation_results)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)

        log_message(f"Evaluation results saved to {filepath}")
