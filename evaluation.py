"""
Evaluation module for model performance evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

from utils import log_message, calculate_statistics


class ModelEvaluator:
    """Model evaluator for anomaly detection performance."""

    def __init__(self) -> None:
        self.evaluation_results: Dict[str, Any] = {}

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

    def generate_evaluation_report(
        self, normal_scores: List[float], anomaly_scores: List[float]
    ) -> str:
        """Generate evaluation report."""
        results = self.evaluate_anomaly_detection(normal_scores, anomaly_scores)

        report = f"""
=== Anomaly Detection Model Evaluation Report ===

Data Statistics:
- Normal samples: {results["num_normal"]}
- Anomaly samples: {results["num_anomaly"]}

Normal Sample Score Statistics:
- Mean: {results["normal_stats"]["mean"]:.6f}
- Std: {results["normal_stats"]["std"]:.6f}
- Min: {results["normal_stats"]["min"]:.6f}
- Max: {results["normal_stats"]["max"]:.6f}

Anomaly Sample Score Statistics:
- Mean: {results["anomaly_stats"]["mean"]:.6f}
- Std: {results["anomaly_stats"]["std"]:.6f}
- Min: {results["anomaly_stats"]["min"]:.6f}
- Max: {results["anomaly_stats"]["max"]:.6f}

Model Performance:
- Separation Ratio: {results["separation_ratio"]:.3f}

Evaluation Conclusion:
"""

        if results["separation_ratio"] > 2.0:
            report += (
                "✓ Model successfully distinguishes between normal and anomaly samples"
            )
        elif results["separation_ratio"] > 1.5:
            report += "⚠ Model partially distinguishes normal and anomaly samples, tuning recommended"
        else:
            report += "✗ Model failed to effectively distinguish normal and anomaly samples, retraining needed"

        return report

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
