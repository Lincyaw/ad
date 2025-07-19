"""
Trace anomaly detector module.
"""

import os
import joblib
import torch
from typing import List, Dict, Any, Optional
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import ModelConfig, TraceDataConfig
from data_processor import DataProcessor
from models import GATEncoder, FeatureDecoder, AdvancedGAE
from utils import ensure_directory, log_message


class TraceAnomalyDetector:
    """Trace anomaly detector using Graph Attention Networks."""

    def __init__(
        self, config: ModelConfig, trace_config: Optional[TraceDataConfig] = None
    ) -> None:
        self.config = config
        self.trace_config = trace_config or TraceDataConfig()
        self.processor = DataProcessor(config, trace_config)
        self.model: Optional[AdvancedGAE] = None
        self.is_trained = False

    def fit(self, df_train, verbose: bool = True) -> None:
        if verbose:
            log_message("[1/4] Starting Preprocessing")

        self.processor.fit(df_train)
        df_train_proc, features_train = self.processor.transform(df_train)
        # Use actual graph node feature dimension instead of trace-level feature dimension
        self.config.in_channels = self.processor.get_node_feature_dim()

        if verbose:
            log_message(
                f"[2/4] Building Graph Dataset (Input Dim: {self.config.in_channels})"
            )

        # Build graph dataset
        train_graphs = self.processor.build_graphs()
        train_loader = DataLoader(
            train_graphs, batch_size=self.config.batch_size, shuffle=True
        )

        if verbose:
            log_message("[3/4] Initializing Model and Optimizer")

        # Initialize model
        encoder = GATEncoder(
            self.config.in_channels, self.config.latent_dim, self.config.gat_heads
        )
        feature_decoder = FeatureDecoder(
            self.config.latent_dim, self.config.in_channels
        )
        self.model = AdvancedGAE(encoder, feature_decoder).to(self.config.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        if verbose:
            log_message("[4/4] Starting Model Training")

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in tqdm(
            range(self.config.epochs), desc="Training Progress", disable=not verbose
        ):
            total_loss = 0
            epoch_losses = []

            for data in train_loader:
                data = data.to(self.config.device)
                optimizer.zero_grad()

                z = self.model.encode(data.x, data.edge_index)

                loss_struct = self.model.recon_loss(z, data.edge_index)
                loss_feat = self.model.recon_loss_features(z, data.x)

                loss = (
                    self.config.loss_alpha * loss_struct
                    + (1 - self.config.loss_alpha) * loss_feat
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                epoch_losses.append(loss.item())

            scheduler.step()
            avg_loss = total_loss / len(train_loader)

            # Early stopping mechanism
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        log_message(f"Early stopping at epoch {epoch}")
                    break

        self.is_trained = True
        if verbose:
            log_message("Training completed successfully.")

    def predict_score(self, df) -> List[Dict[str, Any]]:
        assert self.is_trained and self.model is not None, (
            "Model is not trained. Call fit() first."
        )

        self.model.eval()
        df_proc, features = self.processor.transform(df)
        graphs = self.processor.build_graphs()

        results = []
        with torch.no_grad():
            for data in graphs:
                data = data.to(self.config.device)
                # Ensure data.x and data.edge_index are not None
                assert data.x is not None and data.edge_index is not None, (
                    "Graph data missing node features or edge indices"
                )

                z = self.model.encode(data.x, data.edge_index)

                loss_struct = self.model.recon_loss(z, data.edge_index)
                loss_feat = self.model.recon_loss_features(z, data.x)

                score = (
                    self.config.loss_alpha * loss_struct
                    + (1 - self.config.loss_alpha) * loss_feat
                )

                results.append(
                    {
                        "window_start": str(getattr(data, "window_start", "unknown")),
                        "anomaly_score": score.item(),
                        "struct_error": loss_struct.item(),
                        "feature_error": loss_feat.item(),
                    }
                )
        return results

    def predict_aggregated(
        self, df, aggregation_method: str = "max", threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        window_results = self.predict_score(df)

        assert window_results, "No window results found for prediction"

        # Extract scores
        anomaly_scores = [r["anomaly_score"] for r in window_results]
        struct_errors = [r["struct_error"] for r in window_results]
        feature_errors = [r["feature_error"] for r in window_results]

        # Aggregate scores based on method
        if aggregation_method == "max":
            agg_score = max(anomaly_scores)
            agg_struct = max(struct_errors)
            agg_feat = max(feature_errors)
        elif aggregation_method == "mean":
            agg_score = sum(anomaly_scores) / len(anomaly_scores)
            agg_struct = sum(struct_errors) / len(struct_errors)
            agg_feat = sum(feature_errors) / len(feature_errors)
        elif aggregation_method == "percentile_95":
            import numpy as np

            agg_score = float(np.percentile(anomaly_scores, 95))
            agg_struct = float(np.percentile(struct_errors, 95))
            agg_feat = float(np.percentile(feature_errors, 95))
        else:
            assert False, f"Unsupported aggregation method: {aggregation_method}"

        # Calculate confidence (normalized consistency measure)
        if len(anomaly_scores) > 1:
            import numpy as np

            scores_array = np.array(anomaly_scores)
            # Use coefficient of variation (CV) for relative variability
            mean_score = np.mean(scores_array)
            if mean_score > 0:
                cv = np.std(scores_array) / mean_score  # Coefficient of variation
                confidence = 1.0 / (
                    1.0 + cv
                )  # Higher confidence for lower relative variance
            else:
                confidence = 1.0
        else:
            confidence = 1.0

        result = {
            "anomaly_score": agg_score,
            "struct_error": agg_struct,
            "feature_error": agg_feat,
            "confidence": confidence,
            "num_windows": len(window_results),
            "aggregation_method": aggregation_method,
            "window_scores": anomaly_scores,  # Keep individual window scores for analysis
            "score_statistics": {
                "min": float(min(anomaly_scores)),
                "max": float(max(anomaly_scores)),
                "mean": float(sum(anomaly_scores) / len(anomaly_scores)),
                "std": float(np.std(anomaly_scores))
                if len(anomaly_scores) > 1
                else 0.0,
            },
        }

        # Add binary classification if threshold is provided
        if threshold is not None:
            result["has_anomaly"] = agg_score >= threshold
            result["threshold"] = threshold

        return result

    def predict_batch(self, df, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Batch prediction for large datasets."""
        assert self.is_trained and self.model is not None, (
            "Model is not trained. Call fit() first."
        )

        self.model.eval()
        df_proc, features = self.processor.transform(df)
        graphs = self.processor.build_graphs()

        # Create data loader
        test_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

        results = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.config.device)
                z = self.model.encode(batch.x, batch.edge_index)

                loss_struct = self.model.recon_loss(z, batch.edge_index)
                loss_feat = self.model.recon_loss_features(z, batch.x)

                score = (
                    self.config.loss_alpha * loss_struct
                    + (1 - self.config.loss_alpha) * loss_feat
                )

                results.append(
                    {
                        "window_start": str(getattr(batch, "window_start", "unknown")),
                        "anomaly_score": score.item(),
                        "struct_error": loss_struct.item(),
                        "feature_error": loss_feat.item(),
                    }
                )

        return results

    def save(self, directory: str) -> None:
        """Save model and preprocessor."""
        assert self.is_trained and self.model is not None, (
            "Cannot save an untrained model."
        )

        ensure_directory(directory)

        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pth"))
        # Save preprocessor
        joblib.dump(self.processor, os.path.join(directory, "processor.joblib"))
        # Save configuration
        joblib.dump(self.config, os.path.join(directory, "config.joblib"))
        joblib.dump(self.trace_config, os.path.join(directory, "trace_config.joblib"))

        log_message(f"Model saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> "TraceAnomalyDetector":
        config = joblib.load(os.path.join(directory, "config.joblib"))

        trace_config_path = os.path.join(directory, "trace_config.joblib")
        if os.path.exists(trace_config_path):
            trace_config = joblib.load(trace_config_path)
        else:
            trace_config = TraceDataConfig()

        detector = cls(config, trace_config)

        detector.processor = joblib.load(os.path.join(directory, "processor.joblib"))

        encoder = GATEncoder(config.in_channels, config.latent_dim, config.gat_heads)
        feature_decoder = FeatureDecoder(config.latent_dim, config.in_channels)
        model = AdvancedGAE(encoder, feature_decoder)
        model.load_state_dict(torch.load(os.path.join(directory, "model.pth")))
        model.to(config.device)

        detector.model = model
        detector.is_trained = True

        return detector

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        assert self.is_trained and self.model is not None, "Model is not trained."

        return {
            "config": self.config,
            "trace_config": self.trace_config,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "feature_info": self.processor.get_feature_info(),
        }
