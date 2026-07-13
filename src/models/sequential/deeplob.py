import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepLOBEncoder(nn.Module):

    def __init__(self, conv_channels: int=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)), nn.BatchNorm2d(16), nn.GELU(), nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, conv_channels, kernel_size=(3, 1), padding=(1, 0)), nn.BatchNorm2d(conv_channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)
        z = self.conv(z)
        z = z.mean(dim=3)
        z = z.transpose(1, 2)
        return z

