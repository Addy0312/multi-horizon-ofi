"""
Seq2Seq LSTM with Temporal Attention — Target Model.

This is the primary model of the project: a Sequence-to-Sequence
LSTM with a custom *temporal attention* mechanism designed to learn
horizon-specific attention patterns over the LOB event history.

Key innovations over the baseline Seq2Seq (Zhang & Zohren 2021):
    1. Horizon-aware temporal attention: each prediction horizon
       learns its own attention distribution — short horizons can
       focus on recent events while long horizons attend to broader
       patterns.
    2. Multi-scale encoder: parallel LSTM branches at different
       temporal resolutions (event-level + sub-sampled).
    3. Residual connections in the decoder.
    4. Attention interpretability: the per-horizon attention maps
       can be visualised to explain model behaviour.

Architecture:
    Encoder:
        Input (batch, seq_len, n_features)
        → Bidirectional LSTM
        → encoder_outputs (batch, seq_len, 2*hidden)

    Decoder (autoregressive over horizons):
        For each horizon k = 1..H:
            → TemporalAttention(encoder_outputs, decoder_state, horizon=k)
              → horizon-specific context vector
            → LSTM decoder step(context + prev_output)
            → Residual + LayerNorm
            → Linear → class logits

    Outputs:
        logits (batch, n_horizons, n_classes)
        attention_weights (batch, n_horizons, seq_len) [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.models.layers.attention import TemporalAttention


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale bidirectional LSTM encoder.

    Processes the input at the original resolution *and* at a
    sub-sampled resolution (every k-th event), then fuses them.
    This captures both fine-grained microstructure and broader trends.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        subsample_factor: int = 5,
    ):
        super().__init__()

        self.subsample_factor = subsample_factor

        # Fine-grained encoder (every event)
        self.encoder_fine = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Coarse encoder (sub-sampled events)
        self.encoder_coarse = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Fusion: project concatenated outputs to common dim
        self.fusion = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        fused_outputs : (batch, seq_len, hidden*2)
        (h_n, c_n) : final hidden states from fine encoder
        """
        # Fine encoder
        fine_out, (h_n, c_n) = self.encoder_fine(x)
        # fine_out: (batch, seq_len, hidden*2)

        # Coarse encoder: sub-sample along time
        x_coarse = x[:, :: self.subsample_factor, :]
        coarse_out, _ = self.encoder_coarse(x_coarse)
        # coarse_out: (batch, seq_len//factor, hidden*2)

        # Upsample coarse to match fine resolution
        coarse_upsampled = F.interpolate(
            coarse_out.permute(0, 2, 1),  # (batch, hidden*2, coarse_len)
            size=fine_out.size(1),
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # (batch, seq_len, hidden*2)

        # Fuse
        combined = torch.cat([fine_out, coarse_upsampled], dim=-1)
        fused = self.layer_norm(self.fusion(combined))

        return fused, (h_n, c_n)


class Seq2SeqTemporalAttention(nn.Module):
    """
    Seq2Seq LSTM with Temporal Attention.

    Target model for the multi-horizon OFI project.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 4,
        n_classes: int = 3,
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        encoder_layers: int = 2,
        attn_dim: int = 64,
        dropout: float = 0.3,
        use_multi_scale: bool = True,
        subsample_factor: int = 5,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.use_multi_scale = use_multi_scale

        enc_output_dim = encoder_hidden * 2  # bidirectional

        # --- Encoder ---
        if use_multi_scale:
            self.encoder = MultiScaleEncoder(
                n_features=n_features,
                hidden_size=encoder_hidden,
                num_layers=encoder_layers,
                dropout=dropout,
                subsample_factor=subsample_factor,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=encoder_hidden,
                num_layers=encoder_layers,
                batch_first=True,
                dropout=dropout if encoder_layers > 1 else 0.0,
                bidirectional=True,
            )

        # Bridge: encoder final states → decoder initial states
        self.bridge_h = nn.Linear(enc_output_dim, decoder_hidden)
        self.bridge_c = nn.Linear(enc_output_dim, decoder_hidden)

        # --- Temporal Attention ---
        self.temporal_attention = TemporalAttention(
            encoder_dim=enc_output_dim,
            decoder_dim=decoder_hidden,
            n_horizons=n_horizons,
            attn_dim=attn_dim,
        )

        # --- Decoder ---
        self.decoder_cell = nn.LSTMCell(
            input_size=enc_output_dim + n_classes,
            hidden_size=decoder_hidden,
        )

        # Residual projection (if dimensions differ)
        self.residual_proj = nn.Linear(
            enc_output_dim + n_classes, decoder_hidden
        )
        self.decoder_norm = nn.LayerNorm(decoder_hidden)

        self.dropout = nn.Dropout(dropout)

        # --- Output projection ---
        self.output_proj = nn.Sequential(
            nn.Linear(decoder_hidden + enc_output_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, n_classes),
        )

        # Start token
        self.start_token = nn.Parameter(torch.zeros(n_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)
        return_attention : bool
            If True, also return attention weights (batch, n_horizons, seq_len)
            for interpretability.

        Returns
        -------
        logits : (batch, n_horizons, n_classes)
        attn_weights : (batch, n_horizons, seq_len) [optional]
        """
        batch_size = x.size(0)

        # --- Encode ---
        if self.use_multi_scale:
            enc_outputs, (h_n, c_n) = self.encoder(x)
        else:
            enc_outputs, (h_n, c_n) = self.encoder(x)
        # enc_outputs: (batch, seq_len, enc_hidden*2)

        # Bridge: init decoder state
        h_n_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, enc_hidden*2)
        c_n_cat = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        dec_h = torch.tanh(self.bridge_h(h_n_cat))  # (batch, dec_hidden)
        dec_c = torch.tanh(self.bridge_c(c_n_cat))

        # --- Decode with temporal attention ---
        outputs = []
        all_attn_weights = []
        prev_output = self.start_token.unsqueeze(0).expand(batch_size, -1)

        for h_idx in range(self.n_horizons):
            # Horizon-specific attention
            context, attn_w = self.temporal_attention(
                enc_outputs, dec_h, horizon_idx=h_idx
            )
            # context: (batch, enc_hidden*2)
            # attn_w: (batch, seq_len)

            all_attn_weights.append(attn_w)

            # Decoder input
            dec_input = torch.cat([context, prev_output], dim=-1)
            # (batch, enc_hidden*2 + n_classes)

            # Decoder step with residual connection
            residual = self.residual_proj(dec_input)
            dec_h_new, dec_c = self.decoder_cell(dec_input, (dec_h, dec_c))
            dec_h = self.decoder_norm(dec_h_new + residual)

            # Output
            combined = torch.cat([dec_h, context], dim=-1)
            combined = self.dropout(combined)
            logits = self.output_proj(combined)  # (batch, n_classes)
            outputs.append(logits)

            # Feed prediction to next step
            prev_output = F.softmax(logits, dim=-1)

        logits = torch.stack(outputs, dim=1)  # (batch, n_horizons, n_classes)

        if return_attention:
            attn_weights = torch.stack(all_attn_weights, dim=1)
            # (batch, n_horizons, seq_len)
            return logits, attn_weights

        return logits

    def get_attention_maps(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to extract attention maps for visualisation.

        Returns
        -------
        logits : (batch, n_horizons, n_classes)
        attn_weights : (batch, n_horizons, seq_len)
        """
        return self.forward(x, return_attention=True)
