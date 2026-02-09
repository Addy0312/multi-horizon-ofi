"""
LSTM model for multi-horizon LOB prediction.

Reference: Kolm, Turiel & Westray (2023)
    "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons
     from the Limit Order Book"

Architecture:
    Input (batch, seq_len, n_features)
    → LSTM (2 layers, bidirectional optional)
    → Take last hidden state
    → Dense → (H horizons × 3 classes)
"""

import torch
import torch.nn as nn
from typing import List


class LSTMClassifier(nn.Module):
    """
    LSTM for multi-horizon 3-class LOB price movement prediction.

    Takes sequential feature input (batch, seq_len, n_features) and
    outputs logits (batch, n_horizons, 3).
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        n_classes: int = 3,
    ):
        super().__init__()

        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

        # One classification head per horizon
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc_input, 32),
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
        # LSTM output: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step's output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size * dirs)
        last_out = self.dropout(last_out)

        # Per-horizon classification
        out = torch.stack([head(last_out) for head in self.heads], dim=1)
        return out  # (batch, n_horizons, n_classes)
