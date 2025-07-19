import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "primary_service",
            "root_span",
        ]
    )
    numerical_features: List[str] = field(
        default_factory=lambda: [
            "span_count",
            "total_duration",
            "avg_duration",
            "max_duration",
            "min_duration",
            "duration_std",
            "unique_services",
            "unique_spans",
            "error_rate",
        ]
    )

    in_channels: int = -1
    latent_dim: int = 16
    gat_heads: int = 4

    # Training configuration
    learning_rate: float = 0.005
    epochs: int = 100
    batch_size: int = 1
    # Loss weight: alpha * structural loss + (1-alpha) * attribute loss
    loss_alpha: float = 0.6

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TraceDataConfig:
    """Configuration for trace data."""

    extended_categorical_features: List[str] = field(
        default_factory=lambda: [
            "span_name",
            "attr.span_kind",
            "service_name",
            "attr.http.request.method",
            "attr.k8s.pod.name",
            "attr.k8s.service.name",
            "attr.k8s.namespace.name",
            "attr.status_code",
        ]
    )
    extended_numerical_features: List[str] = field(
        default_factory=lambda: [
            "duration",
            "attr.http.request.content_length",
            "attr.http.response.content_length",
            "attr.http.response.status_code",
        ]
    )

    # Data processing configuration
    missing_value_strategy: str = "fill"  # 'fill' or 'drop'
    categorical_fill_value: str = "N/A"
    numerical_fill_value: float = 0.0

    # Graph construction configuration
    edge_type: str = "parent_child"  # 'parent_child' or 'temporal'
    add_self_loops: bool = True
