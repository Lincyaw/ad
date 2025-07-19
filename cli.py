import typer
from pathlib import Path
import pandas as pd
from functools import partial
from rich.table import Table
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
) -> None:
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


if __name__ == "__main__":
    app()
