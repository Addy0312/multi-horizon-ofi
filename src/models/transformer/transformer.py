"""
Transformer model for LOB prediction.

Reference: Wallbridge (2020)
    "Transformers for Limit Order Books"

Architecture:
    Input (batch, seq_len, n_features)
    → Linear projection to d_model
    → Positional Encoding
    → N × Transformer Encoder Layers (Multi-Head Self-Attention + FFN)
    → Take [CLS] token or global average pool
    → Dense → (H × 3)

Key differences from LSTM-based models:
    - Parallel processing (no sequential bottleneck)
    - Self-attention captures long-range dependencies directly
    - Positional encoding injects order information
"""

import torch
import torch.nn as nn
import math
from typing import List

from src.models.layers.positional import PositionalEncoding


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for multi-horizon LOB price movement prediction.

    Uses a stack of Transformer encoder layers with multi-head
    self-attention, followed by classification heads per horizon.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        n_classes: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # --- Input projection ---
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # --- CLS token (learnable) ---
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- Positional encoding ---
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len + 1, dropout=dropout
        )

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- Output heads ---
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes),
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
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_proj(x)     # (batch, seq_len, d_model)
        x = self.input_norm(x)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # x: (batch, 1 + seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, 1+seq_len, d_model)

        # Extract representation
        if self.use_cls_token:
            rep = x[:, 0, :]  # CLS token output
        else:
            rep = x.mean(dim=1)  # Global average pooling

        rep = self.dropout(rep)  # (batch, d_model)

        # Per-horizon classification
        out = torch.stack([head(rep) for head in self.heads], dim=1)
        return out  # (batch, n_horizons, n_classes)
