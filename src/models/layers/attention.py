"""
Attention mechanisms for sequential LOB models.

Implements:
    1. Bahdanau (additive) attention
    2. Luong (dot-product) attention
    3. Temporal attention (custom, for Seq2Seq temporal attention model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

    score(s_t, h_j) = v^T · tanh(W_s · s_t + W_h · h_j)

    Used in Zhang & Zohren (2021) Seq2Seq model.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W_h = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_outputs : (batch, src_len, encoder_dim)
        decoder_hidden : (batch, decoder_dim)

        Returns
        -------
        context : (batch, encoder_dim)
        attn_weights : (batch, src_len)
        """
        # decoder_hidden → (batch, 1, attn_dim)
        query = self.W_s(decoder_hidden).unsqueeze(1)
        # encoder_outputs → (batch, src_len, attn_dim)
        keys = self.W_h(encoder_outputs)

        # Additive attention score
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)  # (batch, src_len)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_dim)

        return context, attn_weights


class LuongAttention(nn.Module):
    """
    Multiplicative (Luong) attention.

    score(s_t, h_j) = s_t^T · W · h_j

    Simpler than Bahdanau, often faster.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.W = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_outputs : (batch, src_len, encoder_dim)
        decoder_hidden : (batch, decoder_dim)

        Returns
        -------
        context : (batch, encoder_dim)
        attn_weights : (batch, src_len)
        """
        # Project encoder outputs
        keys = self.W(encoder_outputs)  # (batch, src_len, decoder_dim)

        # Dot product
        scores = torch.bmm(
            keys, decoder_hidden.unsqueeze(-1)
        ).squeeze(-1)  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)

        return context, attn_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for the custom Seq2Seq model.

    This is a horizon-aware attention that learns different attention
    patterns for different prediction horizons.

    For each horizon h:
        - Learns which past time steps are most relevant
        - Produces horizon-specific context vectors
        - Enables interpretability: "which past events drive
          short-term vs long-term predictions?"

    score(s_t, h_j, k) = v_k^T · tanh(W_s · s_t + W_h · h_j + W_k · e_k)

    where e_k is a learnable horizon embedding.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        n_horizons: int = 4,
        attn_dim: int = 64,
    ):
        super().__init__()

        self.n_horizons = n_horizons
        self.W_h = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(decoder_dim, attn_dim, bias=False)

        # Horizon-specific components
        self.horizon_embeddings = nn.Embedding(n_horizons, attn_dim)
        self.v = nn.ModuleList([
            nn.Linear(attn_dim, 1, bias=False)
            for _ in range(n_horizons)
        ])

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        horizon_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_outputs : (batch, src_len, encoder_dim)
        decoder_hidden : (batch, decoder_dim)
        horizon_idx : int or None
            If None, compute attention for ALL horizons and return
            (batch, n_horizons, encoder_dim) context.

        Returns
        -------
        context : (batch, encoder_dim) or (batch, n_horizons, encoder_dim)
        attn_weights : (batch, src_len) or (batch, n_horizons, src_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        keys = self.W_h(encoder_outputs)  # (batch, src_len, attn_dim)
        query = self.W_s(decoder_hidden).unsqueeze(1)  # (batch, 1, attn_dim)

        if horizon_idx is not None:
            # Single horizon
            h_emb = self.horizon_embeddings(
                torch.tensor([horizon_idx], device=encoder_outputs.device)
            ).unsqueeze(0).expand(batch_size, src_len, -1)

            energy = torch.tanh(query + keys + h_emb)
            scores = self.v[horizon_idx](energy).squeeze(-1)
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            return context.squeeze(1), attn_weights
        else:
            # All horizons
            all_contexts = []
            all_weights = []
            for h in range(self.n_horizons):
                h_emb = self.horizon_embeddings(
                    torch.tensor([h], device=encoder_outputs.device)
                ).unsqueeze(0).expand(batch_size, src_len, -1)

                energy = torch.tanh(query + keys + h_emb)
                scores = self.v[h](energy).squeeze(-1)
                attn_w = F.softmax(scores, dim=-1)
                ctx = torch.bmm(attn_w.unsqueeze(1), encoder_outputs).squeeze(1)

                all_contexts.append(ctx)
                all_weights.append(attn_w)

            contexts = torch.stack(all_contexts, dim=1)  # (batch, H, enc_dim)
            weights = torch.stack(all_weights, dim=1)    # (batch, H, src_len)
            return contexts, weights
