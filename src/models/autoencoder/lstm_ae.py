import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder(hidden_repeated)
        out = self.output_layer(decoder_out)
        return out

