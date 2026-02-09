"""
DeepLOB: CNN-LSTM hybrid for LOB prediction.

Reference: Zhang, Zohren & Roberts (2019)
    "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"

Architecture (adapted for OFI features):
    Input (batch, seq_len, n_features)
    → Reshape to (batch, 1, seq_len, n_features)  [treat as 2D "image"]
    → Conv2D block 1: (1→32, kernel 1×2) → BN → LeakyReLU → Conv2D (32→32, 4×1) → BN → LeakyReLU
    → Conv2D block 2: (32→32, kernel 1×2) → BN → LeakyReLU → Conv2D (32→32, 4×1) → BN → LeakyReLU
    → Conv2D block 3: (32→32, kernel 1×2) → BN → LeakyReLU → Conv2D (32→32, 4×1) → BN → LeakyReLU
    → Inception-like module (optional)
    → Reshape to (batch, seq_len', features')
    → LSTM (64 hidden)
    → Dense → (H × 3)

The key insight of DeepLOB is treating the LOB as a spatial structure:
    - Convolutions across *levels* capture cross-level patterns
    - Convolutions across *time* capture temporal dynamics
    - LSTM on top captures long-range dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DeepLOB(nn.Module):
    """
    DeepLOB: CNN-LSTM model for multi-horizon LOB prediction.

    Original paper used raw LOB (40 features: 10 levels × 4).
    We adapt it to work with any n_features (OFI + microstructure).
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        n_classes: int = 3,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_horizons = n_horizons
        self.n_classes = n_classes

        # --- Convolutional blocks (spatial: across features) ---
        # Block 1
        self.conv1_1 = nn.Conv2d(1, conv_channels, kernel_size=(1, 2), padding=(0, 0))
        self.bn1_1 = nn.BatchNorm2d(conv_channels)
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(4, 1), padding=(1, 0))
        self.bn1_2 = nn.BatchNorm2d(conv_channels)

        # Block 2
        self.conv2_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 2), padding=(0, 0))
        self.bn2_1 = nn.BatchNorm2d(conv_channels)
        self.conv2_2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(4, 1), padding=(1, 0))
        self.bn2_2 = nn.BatchNorm2d(conv_channels)

        # Block 3
        self.conv3_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 2), padding=(0, 0))
        self.bn3_1 = nn.BatchNorm2d(conv_channels)
        self.conv3_2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=(4, 1), padding=(1, 0))
        self.bn3_2 = nn.BatchNorm2d(conv_channels)

        # --- Inception-inspired module ---
        self.inception_conv1 = nn.Conv2d(conv_channels, 64, kernel_size=(1, 1))
        self.inception_bn1 = nn.BatchNorm2d(64)
        self.inception_conv2 = nn.Conv2d(conv_channels, 64, kernel_size=(3, 1), padding=(1, 0))
        self.inception_bn2 = nn.BatchNorm2d(64)

        # --- LSTM for temporal processing ---
        # After convolutions, we need to compute the feature dimension dynamically
        self.n_features_orig = n_features
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=128,  # 64 + 64 from inception
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Per-horizon output
        self.heads = nn.ModuleList([
            nn.Linear(lstm_hidden, n_classes)
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
        batch_size = x.size(0)

        # Treat as 2D "image": (batch, 1, seq_len, n_features)
        x = x.unsqueeze(1)

        # Conv Block 1
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)), 0.01)
        x = F.leaky_relu(self.bn1_2(self.conv1_2(x)), 0.01)

        # Conv Block 2
        x = F.leaky_relu(self.bn2_1(self.conv2_1(x)), 0.01)
        x = F.leaky_relu(self.bn2_2(self.conv2_2(x)), 0.01)

        # Conv Block 3
        x = F.leaky_relu(self.bn3_1(self.conv3_1(x)), 0.01)
        x = F.leaky_relu(self.bn3_2(self.conv3_2(x)), 0.01)

        # Inception module
        inc1 = F.leaky_relu(self.inception_bn1(self.inception_conv1(x)), 0.01)
        inc2 = F.leaky_relu(self.inception_bn2(self.inception_conv2(x)), 0.01)
        x = torch.cat([inc1, inc2], dim=1)  # (batch, 128, T', F')

        # Pool across the (remaining) feature dimension → (batch, 128, T', 1)
        x = F.adaptive_avg_pool2d(x, (x.size(2), 1))
        x = x.squeeze(-1)           # (batch, 128, T')
        x = x.permute(0, 2, 1)      # (batch, T', 128)

        # LSTM
        lstm_out, _ = self.lstm(x)   # (batch, T', lstm_hidden)
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        last_out = self.dropout(last_out)

        # Per-horizon classification
        out = torch.stack([head(last_out) for head in self.heads], dim=1)
        return out  # (batch, n_horizons, n_classes)
