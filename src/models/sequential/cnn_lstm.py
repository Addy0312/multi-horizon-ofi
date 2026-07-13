import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InceptionBlock1D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b1 = out_ch // 4
        b2 = out_ch // 4
        b3 = out_ch // 4
        b4 = out_ch - (b1 + b2 + b3)
        self.branch1 = nn.Sequential(nn.Conv1d(in_ch, b1, 1), nn.BatchNorm1d(b1), nn.GELU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_ch, b2, 1), nn.BatchNorm1d(b2), nn.GELU(), nn.Conv1d(b2, b2, 3, padding=1), nn.BatchNorm1d(b2), nn.GELU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_ch, b3, 1), nn.BatchNorm1d(b3), nn.GELU(), nn.Conv1d(b3, b3, 5, padding=2), nn.BatchNorm1d(b3), nn.GELU())
        self.branch4 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, b4, 1), nn.BatchNorm1d(b4), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class HybridCNNInceptionLSTM(nn.Module):

    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, channels: int=96, lstm_hidden: int=128, lstm_layers: int=2, dropout: float=0.15):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        self.stem = nn.Sequential(nn.Conv1d(input_dim, channels, kernel_size=7, padding=3), nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout))
        self.inception = InceptionBlock1D(channels, channels)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.lstm = nn.LSTM(channels, lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, horizon_count * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        z = self.stem(x.transpose(1, 2))
        z = self.inception(z)
        z = self.pool(z).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        z = self.drop(h[-1])
        return self.head(z).view(bsz, self.horizon_count, self.num_classes)

