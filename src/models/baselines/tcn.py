import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, dropout):
        super().__init__()
        # Causal padding
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        if self.pad > 0: out = out[:, :, :-self.pad]
        out = self.drop1(F.gelu(self.bn1(out)))
        
        out = self.conv2(out)
        if self.pad > 0: out = out[:, :, :-self.pad]
        out = self.drop2(F.gelu(self.bn2(out)))
        
        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)

class TCNBaseline(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling.
    Uses dilated causal convolutions to build a large receptive field.
    """
    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, num_channels: list=[64, 64, 64], kernel_size: int=3, dropout: float=0.2):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(ResidualBlock(in_ch, out_ch, dilation, kernel_size, dropout))
            
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], horizon_count * num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (bsz, seq_len, input_dim)
        x = x.transpose(1, 2) # (bsz, input_dim, seq_len)
        
        # Pass through TCN
        out = self.network(x) # (bsz, out_channels, seq_len)
        
        # Take the last timestep for prediction
        out = out[:, :, -1]
        
        out = self.head(out)
        return out.view(x.shape[0], self.horizon_count, self.num_classes)
