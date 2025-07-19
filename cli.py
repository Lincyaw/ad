import typer
from pathlib import Path
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from config import ModelConfig, TraceDataConfig
from detector import TraceAnomalyDetector
from evaluation import ModelEvaluator
from utils import validate_trace_data

app = typer.Typer(help="Trace Anomaly Detection Tool", pretty_exceptions_enable=False)


def run_parallel_tasks(tasks, max_workers=8):
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(task): i for i, task in enumerate(tasks)}
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append((index, result))
                except Exception as e:
                    print(f"Task {index} failed with error: {e}")
                    results.append((index, {"error": str(e)}))
                pbar.update(1)

    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


def process_test_file(test_file, label, model_path, aggregation):
    detector = TraceAnomalyDetector.load(str(model_path))

    test_df = pd.read_parquet(test_file)
    validate_trace_data(test_df)

    result = detector.predict_aggregated(test_df, aggregation_method=aggregation)

    return {
        "anomaly_score": result["anomaly_score"],
        "label": label,
        "file": str(test_file),
    }


@app.command()
def train(
    data: Path = typer.Option(
        ..., "--data", help="Training data file path (parquet format)"
    ),
    output: Path = Path("models"),
    epochs: int = typer.Option(100, "--epochs", help="Number of epochs"),
    learning_rate: float = typer.Option(0.005, "--learning-rate", help="Learning rate"),
    latent_dim: int = typer.Option(16, "--latent-dim", help="Latent space dimension"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size"),
) -> None:
    df = pd.read_parquet(data)

    validate_trace_data(df)  # This will assert if validation fails

    # Configure model
    config = ModelConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
        batch_size=batch_size,
    )
    trace_config = TraceDataConfig()

    # Create detector and train
    detector = TraceAnomalyDetector(config, trace_config)
    detector.fit(df, verbose=True)
    detector.save(str(output))


@app.command()
def evaluate(
    model: Path = Path("models"),
    aggregation: str = "max",  # max, mean, percentile_95
    output_dir: Path = Path("evaluation_results"),
) -> None:
    from rich.console import Console

    console = Console()
    output_dir.mkdir(exist_ok=True)

    base_dir = Path("data/test")

    test_file_list = []
    for datapack in base_dir.iterdir():
        assert datapack.is_dir(), f"Expected directory but found file: {datapack}"
        test_file_list.append((str(datapack / "normal_traces.parquet"), 0))
        test_file_list.append((str(datapack / "abnormal_traces.parquet"), 1))

    all_scores = []
    all_labels = []

    evaluator = ModelEvaluator()

    tasks = [
        partial(
            process_test_file,
            test_file,
            label,
            str(model),
            aggregation,
        )
        for test_file, label in test_file_list
    ]

    results = run_parallel_tasks(tasks, max_workers=4)

    for result in results:
        if "error" in result:
            continue

        all_scores.append(result["anomaly_score"])
        all_labels.append(result["label"])

    assert all_scores, "No valid data processed"

    results = evaluator.evaluate_with_threshold(all_scores, all_labels)

    _create_evaluation_visualizations(
        all_scores, all_labels, results, output_dir, console
    )


def _create_evaluation_visualizations(scores, labels, results, output_dir, console):
    import numpy as np

    scores_array = np.array(scores)
    labels_array = np.array(labels)

    _print_evaluation_metrics(results, console)
    _plot_confusion_matrix(labels_array, results, output_dir)
    _plot_roc_curve(labels_array, scores_array, results, output_dir)
    _plot_precision_recall_curve(labels_array, scores_array, output_dir)
    _plot_score_distribution(scores_array, labels_array, output_dir)
    evaluator = ModelEvaluator()
    evaluator.evaluation_results = results
    evaluator.save_evaluation_results(str(output_dir / "evaluation_results.json"))
    console.print(f"\n[green]All visualization charts saved to: {output_dir}[/green]")


def _print_evaluation_metrics(results, console):
    from rich.table import Table

    console.print("\n[bold blue]═══ Model Evaluation Results ═══[/bold blue]")

    table = Table(
        title="Classification Metrics", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("Value", style="green", width=10)
    table.add_column("Description", style="yellow")

    table.add_row(
        "Accuracy", f"{results['accuracy']:.4f}", "Ratio of correct predictions"
    )
    table.add_row(
        "Precision",
        f"{results['precision']:.4f}",
        "Ratio of true anomalies in predicted anomalies",
    )
    table.add_row(
        "Recall", f"{results['recall']:.4f}", "Ratio of correctly identified anomalies"
    )
    table.add_row(
        "F1 Score",
        f"{results['f1_score']:.4f}",
        "Harmonic mean of precision and recall",
    )
    table.add_row(
        "Specificity",
        f"{results['specificity']:.4f}",
        "Ratio of correctly identified normals",
    )
    table.add_row("AUC Score", f"{results['auc_score']:.4f}", "Area under ROC curve")
    table.add_row(
        "Optimal Threshold",
        f"{results['threshold']:.4f}",
        "Threshold with maximum Youden index",
    )

    console.print(table)

    cm = results["confusion_matrix"]
    console.print("\n[bold blue]Confusion Matrix:[/bold blue]")
    console.print(f"True Negatives (TN): {cm['tn']}")
    console.print(f"False Positives (FP): {cm['fp']}")
    console.print(f"False Negatives (FN): {cm['fn']}")
    console.print(f"True Positives (TP): {cm['tp']}")


def _plot_confusion_matrix(labels, results, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    cm_data = np.array(
        [
            [results["confusion_matrix"]["tn"], results["confusion_matrix"]["fp"]],
            [results["confusion_matrix"]["fn"], results["confusion_matrix"]["tp"]],
        ]
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_data, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.colorbar()

    classes = ["Normal", "Anomaly"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm_data.max() / 2.0
    for i, j in np.ndindex(cm_data.shape):
        plt.text(
            j,
            i,
            format(cm_data[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm_data[i, j] > thresh else "black",
            fontsize=14,
            fontweight="bold",
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_roc_curve(labels, scores, results, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = results["auc_score"]

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )

    optimal_threshold = results["threshold"]
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        markersize=8,
        label=f"Optimal Threshold = {optimal_threshold:.4f}",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_precision_recall_curve(labels, scores, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR Curve (AP = {avg_precision:.4f})",
    )

    pos_ratio = np.sum(labels) / len(labels)
    plt.axhline(
        y=pos_ratio,
        color="red",
        linestyle="--",
        label=f"Random Classifier (AP = {pos_ratio:.4f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_score_distribution(scores, labels, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    normal_scores = scores[labels == 0]
    abnormal_scores = scores[labels == 1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.hist(
        normal_scores, bins=30, alpha=0.7, label="Normal", color="blue", density=True
    )
    ax1.hist(
        abnormal_scores, bins=30, alpha=0.7, label="Anomaly", color="red", density=True
    )
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Anomaly Score Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    data = [normal_scores, abnormal_scores]
    labels_box = ["Normal", "Anomaly"]
    bp = ax2.boxplot(data, labels=labels_box, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("Anomaly Score Box Plot")
    ax2.grid(True, alpha=0.3)

    x_normal = np.sort(normal_scores)
    y_normal = np.arange(1, len(x_normal) + 1) / len(x_normal)
    x_abnormal = np.sort(abnormal_scores)
    y_abnormal = np.arange(1, len(x_abnormal) + 1) / len(x_abnormal)

    ax3.plot(x_normal, y_normal, label="Normal", color="blue", linewidth=2)
    ax3.plot(x_abnormal, y_abnormal, label="Anomaly", color="red", linewidth=2)
    ax3.set_xlabel("Anomaly Score")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Cumulative Distribution Function")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.axis("off")
    stats_text = f"""
    Normal Sample Statistics:
    Count: {len(normal_scores)}
    Mean: {np.mean(normal_scores):.4f}
    Std Dev: {np.std(normal_scores):.4f}
    Median: {np.median(normal_scores):.4f}
    
    Anomaly Sample Statistics:
    Count: {len(abnormal_scores)}
    Mean: {np.mean(abnormal_scores):.4f}
    Std Dev: {np.std(abnormal_scores):.4f}
    Median: {np.median(abnormal_scores):.4f}
    
    Separation:
    Mean Ratio: {np.mean(abnormal_scores) / np.mean(normal_scores):.4f}
    """
    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    app()
