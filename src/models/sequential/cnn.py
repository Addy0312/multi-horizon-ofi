"""
1D-CNN model for multi-horizon LOB prediction.

Reference: Kolm, Turiel & Westray (2023)
    "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons
     from the Limit Order Book"

Architecture:
    Input (batch, seq_len, n_features)
    → permute to (batch, n_features, seq_len)  [channels-first]
    → Conv1D → BN → ReLU → MaxPool
    → Conv1D → BN → ReLU → MaxPool
    → Conv1D → BN → ReLU → AdaptiveAvgPool
    → Flatten → Dense → (H × 3)
"""

import torch
import torch.nn as nn
from typing import List


class CNNClassifier(nn.Module):
    """
    1D-CNN for multi-horizon LOB price movement prediction.

    Treats the feature dimension as "channels" and convolves
    along the time dimension.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        n_classes: int = 3,
        channels: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]

        self.n_horizons = n_horizons
        self.n_classes = n_classes

        conv_layers = []
        in_ch = n_features
        for out_ch, ks in zip(channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ])
            in_ch = out_ch

        conv_layers.append(nn.AdaptiveAvgPool1d(1))  # → (batch, last_ch, 1)
        self.convs = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels[-1], 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        logits : (batch, n_horizons, n_classes)
        """
        # Conv1D expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.convs(x)       # (batch, last_ch, 1)
        x = x.squeeze(-1)       # (batch, last_ch)
        x = self.dropout(x)

        out = torch.stack([head(x) for head in self.heads], dim=1)
        return out  # (batch, n_horizons, n_classes)
