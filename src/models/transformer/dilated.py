import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalDilatedConv1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=self.pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        y = self.drop(F.gelu(self.bn(y)))
        r = self.res(x)
        if r.shape[-1] != y.shape[-1]:
            r = r[..., -y.shape[-1]:]
        return y + r

class DilatedMaskedTransformer(nn.Module):

    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, d_model: int=96, n_heads: int=4, n_layers: int=2, dropout: float=0.15, max_len: int=1024):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.tcn = nn.ModuleList([CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=1, dropout=dropout), CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout), CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout)])
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon_count * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        z = self.input_proj(x.transpose(1, 2))
        for block in self.tcn:
            z = block(z)
        z = z.transpose(1, 2)
        pos = self.pos_emb[:, :seq_len] if self.pos_emb.size(1) >= seq_len else F.interpolate(self.pos_emb.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        z += pos
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=z.device), diagonal=1)
        z = self.norm(self.encoder(z, mask=attn_mask)[:, -1, :])
        return self.head(z).view(bsz, self.horizon_count, self.num_classes)

