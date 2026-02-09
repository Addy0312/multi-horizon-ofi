"""
Seq2Seq LSTM with Attention for multi-horizon LOB prediction.

Reference: Zhang & Zohren (2021)
    "Multi-Horizon Forecasting for Limit Order Books:
     Novel Deep Learning Approaches and Hardware Acceleration"

Architecture:
    Encoder:
        Input (batch, seq_len, n_features)
        → LSTM → encoder_outputs (batch, seq_len, hidden)

    Decoder (autoregressive over horizons):
        For each horizon step k:
            → Attention over encoder outputs
            → LSTM decoder step
            → Linear → class logits

    This produces one prediction per horizon in a sequence-to-sequence
    manner, where the decoder "unfolds" across prediction horizons
    rather than across time steps.
"""

import torch
import torch.nn as nn
from typing import List

from src.models.layers.attention import BahdanauAttention


class Seq2SeqAttention(nn.Module):
    """
    Seq2Seq LSTM with Bahdanau Attention for multi-horizon
    LOB price movement prediction.

    The encoder reads the lookback window.
    The decoder produces one output per horizon, using attention
    to dynamically focus on different parts of the input sequence.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        n_classes: int = 3,
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        attn_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bidirectional → project to decoder dim
        self.enc_to_dec_h = nn.Linear(encoder_hidden * 2, decoder_hidden)
        self.enc_to_dec_c = nn.Linear(encoder_hidden * 2, decoder_hidden)

        # --- Attention ---
        self.attention = BahdanauAttention(
            encoder_dim=encoder_hidden * 2,
            decoder_dim=decoder_hidden,
            attn_dim=attn_dim,
        )

        # --- Decoder ---
        # Input: context (enc_hidden*2) + previous prediction embedding
        self.decoder = nn.LSTM(
            input_size=encoder_hidden * 2 + n_classes,
            hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(
            decoder_hidden + encoder_hidden * 2, n_classes
        )

        # Initial decoder input (learnable start token embedding)
        self.start_token = nn.Parameter(torch.zeros(n_classes))

    def forward(
        self, x: torch.Tensor, teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)
        teacher_forcing_ratio : float
            Not used in inference; reserved for training with ground truth.

        Returns
        -------
        logits : (batch, n_horizons, n_classes)
        """
        batch_size = x.size(0)

        # --- Encode ---
        enc_outputs, (h_n, c_n) = self.encoder(x)
        # enc_outputs: (batch, seq_len, enc_hidden * 2)

        # Combine bidirectional hidden states for decoder init
        # h_n: (num_layers * 2, batch, enc_hidden) → (batch, enc_hidden * 2)
        h_n_fwd = h_n[-2]  # last forward layer
        h_n_bwd = h_n[-1]  # last backward layer
        c_n_fwd = c_n[-2]
        c_n_bwd = c_n[-1]

        dec_h = torch.tanh(
            self.enc_to_dec_h(torch.cat([h_n_fwd, h_n_bwd], dim=-1))
        ).unsqueeze(0)  # (1, batch, dec_hidden)
        dec_c = torch.tanh(
            self.enc_to_dec_c(torch.cat([c_n_fwd, c_n_bwd], dim=-1))
        ).unsqueeze(0)  # (1, batch, dec_hidden)

        # --- Decode: one step per horizon ---
        outputs = []
        dec_input = self.start_token.unsqueeze(0).expand(batch_size, -1)
        # dec_input: (batch, n_classes)

        for h_idx in range(self.n_horizons):
            # Attention
            context, attn_w = self.attention(
                enc_outputs, dec_h.squeeze(0)
            )
            # context: (batch, enc_hidden * 2)

            # Decoder input: [context; previous prediction]
            dec_in = torch.cat([context, dec_input], dim=-1)
            dec_in = dec_in.unsqueeze(1)  # (batch, 1, dim)

            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            dec_out = dec_out.squeeze(1)  # (batch, dec_hidden)

            # Output: concat decoder output + context
            combined = torch.cat([dec_out, context], dim=-1)
            combined = self.dropout(combined)
            logits = self.output_proj(combined)  # (batch, n_classes)
            outputs.append(logits)

            # Next decoder input is the current prediction (greedy)
            dec_input = F.softmax(logits, dim=-1)

        # Stack: (batch, n_horizons, n_classes)
        return torch.stack(outputs, dim=1)


# Need F for softmax in forward
import torch.nn.functional as F
