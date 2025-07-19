"""
Deep learning model definitions, including GAT encoder and decoder.
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss
from torch_geometric.nn import GATConv, GAE


class GATEncoder(Module):
    """Encoder using Graph Attention Network (GAT)."""

    def __init__(self, in_channels: int, latent_dim: int, heads: int = 4) -> None:
        super().__init__()
        # Note the dimension changes in GAT
        self.conv1 = GATConv(in_channels, latent_dim * 2, heads=heads, dropout=0.2)
        self.conv2 = GATConv(latent_dim * 2 * heads, latent_dim, heads=1, dropout=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class FeatureDecoder(Module):
    """Decoder for reconstructing node features."""

    def __init__(self, latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(latent_dim, latent_dim * 2),
            ReLU(),
            Linear(latent_dim * 2, latent_dim * 4),
            ReLU(),
            Linear(latent_dim * 4, out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class AdvancedGAE(GAE):
    """GAE model combining GAT encoder and feature decoder."""

    def __init__(self, encoder: Module, feature_decoder: Module) -> None:
        super().__init__(encoder)
        self.feature_decoder = feature_decoder
        self.feature_loss_fn = MSELoss()

    def recon_loss_features(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate node feature reconstruction loss."""
        x_recon = self.feature_decoder(z)
        return self.feature_loss_fn(x_recon, x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encode(x, edge_index)

    def reconstruct_features(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct node features."""
        return self.feature_decoder(z)
