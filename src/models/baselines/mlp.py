"""
Multi-Layer Perceptron (MLP) for multi-horizon LOB prediction.

Reference: Kolm, Turiel & Westray (2023)
    "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons
     from the Limit Order Book"

This is a feedforward network that takes a *flattened* feature vector
(no temporal dimension) and outputs multi-horizon class probabilities.

Architecture:
    Input (F features) → Dense → BN → ReLU → Dropout
                       → Dense → BN → ReLU → Dropout
                       → Dense → (H horizons × 3 classes)
"""

import torch
import torch.nn as nn
from typing import List

from src.features.labels import DEFAULT_HORIZONS


class MLPClassifier(nn.Module):
    """
    MLP baseline for multi-horizon 3-class LOB prediction.

    Takes flat feature vectors (batch, n_features) and outputs
    logits (batch, n_horizons, 3).
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.3,
        n_classes: int = 3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_horizons = n_horizons
        self.n_classes = n_classes

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        # One output head per horizon
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, n_classes) for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_features)

        Returns
        -------
        logits : (batch, n_horizons, n_classes)
        """
        h = self.backbone(x)
        # Stack per-horizon outputs
        out = torch.stack([head(h) for head in self.heads], dim=1)
        return out


class MLPRegressor(nn.Module):
    """
    MLP for multi-horizon regression (predict ΔP directly).
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_horizons = n_horizons

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_features)

        Returns
        -------
        predictions : (batch, n_horizons)
        """
        h = self.backbone(x)
        return self.output(h)
