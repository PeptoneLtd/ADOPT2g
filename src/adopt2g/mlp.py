"""MLP architecture for disorder prediction."""

import torch
import torch.nn as nn

from . import config


class DisorderMLP(nn.Module):
    """
    Simple MLP for predicting disorder g-scores from concatenated embeddings.

    Input: ESM residue embedding (1280) + ST sequence embedding (320) = 1600
    Output: G-score in [0, 1]
    """

    def __init__(
        self,
        input_dim: int = config.TOTAL_EMBED_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = config.MLP_DROPOUT,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = config.MLP_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ],
            )
            prev_dim = hidden_dim

        # Output layer with sigmoid for g-score [0, 1]
        layers.extend(
            [
                nn.Linear(prev_dim, 1),
                nn.Sigmoid(),
            ],
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Embeddings of shape (batch_size, input_dim)

        Returns:
            G-scores of shape (batch_size,), values in [0, 1]
        """
        return self.net(x).squeeze(-1)
