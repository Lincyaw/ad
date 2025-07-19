import typer
from typing import Optional
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from config import ModelConfig, TraceDataConfig
from detector import TraceAnomalyDetector
from evaluation import ModelEvaluator
from utils import log_message, validate_trace_data

app = typer.Typer(help="Trace Anomaly Detection Tool")
console = Console()


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
    console.print("[blue]Starting model training...[/blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading training data...", total=None)
            df = pd.read_parquet(data)
            progress.update(task, description=f"Loaded {len(df)} records")
        console.print(f"[green]✓[/green] Successfully loaded data: {len(df)} records")
    except Exception as e:
        console.print(f"[red]✗ Failed to load data: {e}[/red]")
        raise typer.Exit(1)

    if not validate_trace_data(df):
        console.print("[red]✗ Data format validation failed[/red]")
        raise typer.Exit(1)

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

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            detector.fit(df, verbose=True)
            progress.update(task, description="Saving model...")
            detector.save(str(output))
        console.print("[green]✓ Model training completed successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Model training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model: Path = Path("models"),
    threshold: Optional[float] = typer.Option(None),
    aggregation: str = typer.Option(
        "max", help="Aggregation method: max, mean, percentile_95"
    ),
) -> None:
    """Evaluate model performance with aggregated DataFrame-level predictions."""
    base_dir = Path("data/test")

    test_file_list = []
    for datapack in base_dir.iterdir():
        if not datapack.is_dir():
            continue

        test_file_list.append((datapack / "normal_traces.parquet", 0))
        test_file_list.append((datapack / "abnormal_traces.parquet", 1))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        detector = TraceAnomalyDetector.load(str(model))
    console.print("[green]✓ Model loaded successfully[/green]")

    all_scores = []
    all_labels = []

    evaluator = ModelEvaluator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, (test_file, label) in enumerate(test_file_list):
            test_df = pd.read_parquet(test_file)

            if not validate_trace_data(test_df):
                console.print(
                    f"[red]✗ Data format validation failed for {test_file}[/red]"
                )
                continue

            # Use aggregated prediction instead of window-level
            result = detector.predict_aggregated(
                test_df, aggregation_method=aggregation, threshold=threshold
            )

            all_scores.append(result["anomaly_score"])
            all_labels.append(label)

            console.print(
                f"[blue]处理文件 ({i + 1}/{len(test_file_list)}) - "
                f"{test_file.name}: Score={result['anomaly_score']:.2e}, "
                f"Windows={result['num_windows']}, Confidence={result['confidence']:.4f}, "
                f"Range=[{result['score_statistics']['min']:.2e}, {result['score_statistics']['max']:.2e}][/blue]"
            )

            # Show interim results if we have both classes
            if len(set(all_labels)) > 1 and len(all_scores) > 0:
                try:
                    interim_results = evaluator.evaluate_with_threshold(
                        all_scores, all_labels, threshold
                    )
                    console.print(
                        f"[green]当前进度 ({i + 1}/{len(test_file_list)}) - "
                        f"Precision: {interim_results['precision']:.4f}, "
                        f"Recall: {interim_results['recall']:.4f}, "
                        f"F1: {interim_results['f1_score']:.4f}[/green]"
                    )
                except Exception:
                    pass

    if not all_scores:
        console.print("[red]✗ No valid data processed[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Total files processed: {len(all_scores)}[/green]")

    # Final evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing final evaluation metrics...", total=None)

        results = evaluator.evaluate_with_threshold(all_scores, all_labels, threshold)

        # Display results
        table = Table(
            title=f"Aggregated Evaluation Results ({aggregation.upper()} method)"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Precision", f"{results['precision']:.4f}")
        table.add_row("Recall", f"{results['recall']:.4f}")
        table.add_row("F1-Score", f"{results['f1_score']:.4f}")
        table.add_row("Accuracy", f"{results['accuracy']:.4f}")
        table.add_row(
            "AUC-ROC", f"{results['auc_score']:.4f}" if results["auc_score"] else "N/A"
        )
        table.add_row("Threshold", f"{results['threshold']:.4f}")

        console.print(table)
        progress.update(task, description="Evaluation completed")


if __name__ == "__main__":
    app()
