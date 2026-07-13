import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    A simple but highly effective Multi-Layer Perceptron baseline.
    Flattens the entire sequence into a single vector and processes it through Dense layers.
    Often outperforms complex temporal models on extremely noisy tabular time-series.
    """
    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, seq_len: int=100, hidden_dim: int=512, dropout: float=0.3):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        
        # Flattened dimension
        flat_dim = seq_len * input_dim
        
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, horizon_count * num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        bsz = x.shape[0]
        
        # Flatten
        x_flat = x.reshape(bsz, -1)
        
        # Forward pass
        out = self.net(x_flat)
        
        # Reshape to multi-horizon output
        return out.view(bsz, self.horizon_count, self.num_classes)
