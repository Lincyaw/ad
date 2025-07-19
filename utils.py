"""
Utility functions module.
"""

import os
import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np


def ensure_directory(directory: str) -> None:
    """Ensure directory exists, create if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def log_message(message: str, level: str = "INFO") -> None:
    """Log a message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def format_trace_data(trace_dict: Dict[str, Any]) -> str:
    """Format trace data for display."""
    formatted = []
    for key, value in trace_dict.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.6f}")
        else:
            formatted.append(f"{key}: {value}")
    return ", ".join(formatted)


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate score statistics."""
    if not scores:
        return {}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
    }


def validate_trace_data(df: pd.DataFrame) -> bool:
    required_columns = ["trace_id", "span_id", "parent_span_id"]
    for col in required_columns:
        if col not in df.columns:
            log_message(f"Missing required column: {col}", "ERROR")
            return False
    if df["trace_id"].isna().any():
        log_message("Found null trace_id values", "WARNING")
    return True


def create_sample_trace_data() -> pd.DataFrame:
    """Create sample trace data for testing."""

    data = {
        "time": pd.to_datetime(
            [
                "2025-07-17 02:23:35.480",
                "2025-07-17 02:23:35.481",
                "2025-07-17 02:23:35.482",
                "2025-07-17 02:24:10.100",
                "2025-07-17 02:24:10.101",
                "2025-07-17 02:24:10.102",
                "2025-07-17 02:25:01.200",
                "2025-07-17 02:25:01.201",
                "2025-07-17 02:25:01.202",
                "2025-07-17 02:26:05.300",
                "2025-07-17 02:26:05.301",
                "2025-07-17 02:26:05.302",
                "2025-07-17 02:26:05.303",
            ]
        ),
        "trace_id": [
            "trace_id_0",
            "trace_id_0",
            "trace_id_0",
            "trace_id_1",
            "trace_id_1",
            "trace_id_1",
            "trace_id_2",
            "trace_id_2",
            "trace_id_2",
            "trace_id_3_anomaly",
            "trace_id_3_anomaly",
            "trace_id_3_anomaly",
            "trace_id_3_anomaly",
        ],
        "span_id": [f"span_{i}" for i in range(13)],
        "parent_span_id": [
            None,
            "span_0",
            "span_1",
            None,
            "span_3",
            "span_4",
            None,
            "span_6",
            "span_7",
            None,
            "span_9",
            "span_10",
            "span_9",
        ],
        "span_name": [
            "HTTP GET",
            "DB Query",
            "Cache Read",
            "HTTP GET",
            "DB Query",
            "Cache Read",
            "HTTP GET",
            "DB Query",
            "Cache Read",
            "HTTP GET",
            "DB Query",
            "Cache Read",
            "HTTP POST",
        ],
        "service_name": [
            "api-gateway",
            "user-service",
            "redis-cache",
            "api-gateway",
            "user-service",
            "redis-cache",
            "api-gateway",
            "user-service",
            "redis-cache",
            "api-gateway",
            "user-service",
            "redis-cache",
            "malicious-service",
        ],
        "duration": [
            10464,
            5030,
            1010,
            11500,
            5500,
            1200,
            9800,
            4900,
            990,
            35000,
            6000,
            1300,
            25000,
        ],
        "attr.http.request.method": [
            "GET",
            "N/A",
            "N/A",
            "GET",
            "N/A",
            "N/A",
            "GET",
            "N/A",
            "N/A",
            "GET",
            "N/A",
            "N/A",
            "POST",
        ],
        "attr.http.response.status_code": [
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            500.0,
        ],
    }

    return pd.DataFrame(data)


def create_extended_sample_trace_data() -> pd.DataFrame:
    """Create extended sample trace data with more fields."""
    # Base data
    base_data = create_sample_trace_data()

    # Add extended fields
    extended_fields = {
        "attr.span_kind": ["Server", "Client", "Internal"] * 4 + ["Server"],
        "attr.k8s.pod.name": [f"pod-{i}" for i in range(13)],
        "attr.k8s.service.name": [f"service-{i % 3}" for i in range(13)],
        "attr.k8s.namespace.name": ["default"] * 13,
        "attr.http.request.content_length": [
            100.0 if i % 3 == 0 else None for i in range(13)
        ],
        "attr.http.response.content_length": [
            200.0 if i % 3 == 0 else None for i in range(13)
        ],
        "attr.status_code": [200.0] * 12 + [500.0],
    }

    # Merge data
    for key, value in extended_fields.items():
        base_data[key] = value

    return base_data
